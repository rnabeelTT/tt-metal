# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""DFly2, Hy4-flash's built-in speculative drafter, on the device (reference: models/experimental/hybrid_moe/
dfly_hy4_ref.py, from vLLM PR 68296).

One draft pass after a target pass whose last accepted row is row n (position anchor - 1) and whose next token
`bonus` sits at position anchor:
    x      = embed([bonus, MASK x 5]);  x[0] += gate * eh_proj(cat(enorm(x[0]), hnorm(h_target)))   (mtp_fusion)
    5 x    x += attn(input_norm(x)); x += mlp(post_norm(x))     (attention = the target's sparse-layer MLA with sinks,
             KV SHARE: the latents of target layer L_i (25, 29, .., 41); the block's own latents go to that cache's
             scratch rows; INDEX SHARE: query row j attends to the target row n's recalled positions (all positions
             < anchor below 2048 tokens) + the block's rows 0..j)
    h      = final_norm(x)
    for k = 1..5: c = h[k] + hidden_correction(h[k], embed(prev)); d_k = argmax(lm_head(c)); prev = d_k
Greedy drafts (one-hot q): exact speculative sampling then stays exact for sampled requests too.
"""

from __future__ import annotations

import os

import torch
import ttnn
from loguru import logger

from models.autoports.tencent_hy4_flash.tt.attention import Hy4Attention
from models.autoports.tencent_hy4_flash.tt.common import AllGather, AllReduce, RMSNorm, ckc, num_devices, upload

W = 6  # the block: bonus + 5 drafts


def _dt(name, default):
    return {"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b, "bfp4": ttnn.bfloat4_b}[os.environ.get(name, default)]


class _DraftMLP:
    """Dense SwiGLU 3072 -> 9216 -> 3072, TP (gate / up column parallel, down row parallel + all-reduce)."""

    def __init__(self, mesh, sd, hidden):
        dt = _dt("HY4_DFLY_DTYPE", "bfp8")
        bf = lambda t: t.to(torch.bfloat16)  # noqa: E731
        I = int(sd["gate_proj.weight"].shape[0])
        self.w1 = upload(mesh, bf(sd["gate_proj.weight"]).T.reshape(1, 1, hidden, I), dt, mapper=("shard", -1))
        self.w3 = upload(mesh, bf(sd["up_proj.weight"]).T.reshape(1, 1, hidden, I), dt, mapper=("shard", -1))
        self.w2 = upload(mesh, bf(sd["down_proj.weight"]).T.reshape(1, 1, I, hidden), dt, mapper=("shard", -2))
        self.ar = AllReduce(mesh)
        self.ckc = ckc(mesh)

    def __call__(self, x, mem):
        g = ttnn.matmul(x, self.w1, compute_kernel_config=self.ckc, memory_config=mem)
        u = ttnn.matmul(x, self.w3, compute_kernel_config=self.ckc, memory_config=mem)
        h = ttnn.multiply(u, g, input_tensor_b_activations=[ttnn.UnaryOpType.SILU], memory_config=mem)
        ttnn.deallocate(g)
        ttnn.deallocate(u)
        part = ttnn.matmul(h, self.w2, compute_kernel_config=self.ckc, memory_config=mem)
        ttnn.deallocate(h)
        out = self.ar(part, mem)
        ttnn.deallocate(part)
        return out


class Hy4DFly:
    def __init__(self, model):
        self.model, self.mesh, self.cfg = model, model.mesh, model.cfg
        mesh, cfg = self.mesh, self.cfg
        import json

        full = json.load(open(os.path.join(model.w.dir, "config.json")))
        self.kv_layers = list(full["dfly_target_kv_layers"])
        self.idx_layers = list(full["dfly_target_index_layer"])
        assert self.kv_layers == self.idx_layers
        self.mask_id = int(full["dfly_mask_token_id"])
        n = int(full["dfly_num_hidden_layers"])
        by_li = {l.li: l for l in model.layers}
        w = model.w.prefixed("model.dfly_layers.")
        H = cfg.hidden
        bf = lambda t: t.to(torch.bfloat16)  # noqa: E731
        sdt = _dt("HY4_DFLY_DTYPE", "bfp8")
        self.layers = []
        for i in range(n):
            sd = w.sub(f"{i}.")
            tgt = by_li[self.kv_layers[i]].attn
            attn = Hy4Attention(mesh, cfg, sd.sub("self_attn."), self.kv_layers[i], 0, 0, share=tgt)
            self.layers.append(dict(
                attn=attn,
                tgt=tgt,
                in_norm=RMSNorm(mesh, sd["input_layernorm.weight"], cfg.rms_eps),
                post_norm=RMSNorm(mesh, sd["post_attention_layernorm.weight"], cfg.rms_eps),
                mlp=_DraftMLP(mesh, sd.sub("mlp."), H),
            ))
        self.enorm = RMSNorm(mesh, w["mtp_fusion.enorm.weight"], cfg.rms_eps)
        self.hnorm = RMSNorm(mesh, w["mtp_fusion.hnorm.weight"], cfg.rms_eps)
        self.eh = upload(mesh, bf(w["mtp_fusion.eh_proj.weight"]).T.reshape(1, 1, 2 * H, H), sdt)
        self.fgate = upload(mesh, bf(w["mtp_fusion.gate"]).reshape(1, 1, 1, H))
        self.final_norm = RMSNorm(mesh, w["final_norm.weight"], cfg.rms_eps)
        self.hc_hnorm = RMSNorm(mesh, w["hidden_correction.hidden_norm.weight"], cfg.rms_eps)
        self.hc_enorm = RMSNorm(mesh, w["hidden_correction.embed_norm.weight"], cfg.rms_eps)
        self.hc_g = upload(mesh, bf(w["hidden_correction.gate_proj.weight"]).T.reshape(1, 1, 2 * H, -1), sdt)
        self.hc_u = upload(mesh, bf(w["hidden_correction.up_proj.weight"]).T.reshape(1, 1, 2 * H, -1), sdt)
        self.hc_d = upload(mesh, bf(w["hidden_correction.down_proj.weight"]).T.reshape(1, 1, -1, H), sdt)
        self.ckc = ckc(mesh)
        self.ag = AllGather(mesh)
        self.N = num_devices(mesh)
        # row-0 selector for the fusion, [1, 1, 32, 1]
        m0 = torch.zeros(1, 1, 32, 1, dtype=torch.bfloat16)
        m0[0, 0, 0, 0] = 1
        self.row0 = upload(mesh, m0)
        # trace-safe scratch-row writes (the tensor form of update_padded_kv_cache: no host write at replay)
        u1 = lambda v: upload(mesh, torch.tensor([v], dtype=torch.int32).reshape(1, 1, 1, 1), ttnn.uint32,  # noqa: E731
                              ttnn.ROW_MAJOR_LAYOUT)
        self.slot0 = u1(0)
        self.scratch_off = u1(self.layers[0]["attn"].scratch_row)
        logger.info(f"DFly2 drafter: {n} layers sharing target layers {self.kv_layers}")

    # ------------------------------------------------------------------ pieces
    def _embed(self, tok_rm, rows):
        x = ttnn.embedding(tok_rm, self.model.emb, layout=ttnn.TILE_LAYOUT)
        return ttnn.reshape(x, (1, 1, rows, self.cfg.hidden))

    def _argmax_rm(self, c, mem):
        """c [1, 1, 1, H] corrected hidden -> the greedy token [1, 1] uint32 RM (bf16 argmax over the full vocab: the
        fp32 120K-wide row segfaults in untilize_with_unpadding; drafts only steer acceptance, never the output)."""
        lg = ttnn.matmul(c, self.model.lm, dtype=ttnn.bfloat16, compute_kernel_config=self.model.ckc,
                         memory_config=ttnn.DRAM_MEMORY_CONFIG)
        full = self.ag(lg, dim=3, mem=ttnn.DRAM_MEMORY_CONFIG) if self.N > 1 else lg
        if full is not lg:
            ttnn.deallocate(lg)
        V = self.cfg.vocab
        part = full
        if tuple(full.shape)[-2:] != (1, V):
            part = ttnn.slice(full, (0, 0, 0, 0), (1, 1, 1, V), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        frm = ttnn.to_layout(part, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        idx = ttnn.argmax(frm, dim=-1, keepdim=True, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # [1, 1, 1, 1] uint32
        for t_ in {id(x): x for x in (full, part, frm)}.values():
            ttnn.deallocate(t_)
        return ttnn.reshape(idx, (1, 1))

    def step(self, ds, n: int, mem=None):
        """One draft pass (trace-safe). ds: the draft inputs (tt/decode.DraftInputs); n: the target pass's last
        accepted row (h_target = model.last_hn row n; the index share = each target layer's sel row n). Returns the
        drafts [1, 5] uint32 RM."""
        mem = mem or ttnn.DRAM_MEMORY_CONFIG
        cfg, H = self.cfg, self.cfg.hidden
        x = self._embed(ds.tok, 32)  # [1, 1, 32, H]: [bonus, MASK x 5, pad]
        # mtp_fusion on row 0 (computed on every row, kept on row 0)
        ht = ttnn.slice(self.model.last_hn, (0, 0, n, 0), (1, 1, n + 1, H), memory_config=mem)
        htr = ttnn.repeat(ht, ttnn.Shape((1, 1, 32, 1)), memory_config=mem)
        ttnn.deallocate(ht)
        cat = ttnn.concat([self.enorm(x, mem), self.hnorm(htr, mem)], dim=-1, memory_config=mem)
        ttnn.deallocate(htr)
        f = ttnn.matmul(cat, self.eh, compute_kernel_config=self.ckc, memory_config=mem)
        ttnn.deallocate(cat)
        f = ttnn.multiply(ttnn.multiply(f, self.fgate, memory_config=mem), self.row0, memory_config=mem)
        x2 = ttnn.add(x, f, memory_config=mem)
        ttnn.deallocate(x)
        ttnn.deallocate(f)
        x = x2
        cos, sin = ds.cos, ds.sin
        for i, L in enumerate(self.layers):
            a = L["attn"]
            xn = L["in_norm"](x, mem)
            q_lat, q_rope, q_resid = a._q(xn, cos, sin, mem)
            ttnn.deallocate(q_resid)
            row = a._kv_row(xn, cos, sin, mem)
            row_rm = ttnn.to_layout(row, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(row)
            ttnn.experimental.deepseek_prefill.update_padded_kv_cache(a.cache, row_rm, self.slot0, self.scratch_off,
                                                                      0, 1, 0)
            ttnn.deallocate(row_rm)
            if ds.sparse:
                sel = L["tgt"].indexer.sel if L["tgt"].indexer is not None else None
                s1 = ttnn.slice(sel, (0, 0, n, 0), (1, 1, n + 1, int(sel.shape[-1])))
                srep = ttnn.repeat(s1, ttnn.Shape((1, 1, 32, 1)))
                ttnn.deallocate(s1)
                idx = ttnn.concat([srep, ds.idx_tail], dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                ttnn.deallocate(srep)
            else:
                idx = ds.idx_dense
            o = a._attend(xn, q_lat, q_rope, idx, mem)
            for t_ in (xn, q_lat, q_rope):
                ttnn.deallocate(t_)
            if idx is not ds.idx_dense:
                ttnn.deallocate(idx)
            x2 = ttnn.add(x, o, memory_config=mem)
            ttnn.deallocate(x)
            ttnn.deallocate(o)
            x = x2
            xn = L["post_norm"](x, mem)
            m = L["mlp"](xn, mem)
            ttnn.deallocate(xn)
            x2 = ttnn.add(x, m, memory_config=mem)
            ttnn.deallocate(x)
            ttnn.deallocate(m)
            x = x2
        h = self.final_norm(x, mem)
        ttnn.deallocate(x)
        prev = ds.bonus  # [1, 1] uint32 RM
        drafts = []
        for k in range(1, W):
            hk = ttnn.slice(h, (0, 0, k, 0), (1, 1, k + 1, H), memory_config=mem)  # [1, 1, 1, H]
            e = self._embed(prev, 1)
            z = ttnn.concat([self.hc_hnorm(hk, mem), self.hc_enorm(e, mem)], dim=-1, memory_config=mem)
            ttnn.deallocate(e)
            g = ttnn.matmul(z, self.hc_g, compute_kernel_config=self.ckc, memory_config=mem)
            u = ttnn.matmul(z, self.hc_u, compute_kernel_config=self.ckc, memory_config=mem)
            ttnn.deallocate(z)
            a_ = ttnn.multiply(u, g, input_tensor_b_activations=[ttnn.UnaryOpType.SILU], memory_config=mem)
            ttnn.deallocate(g)
            ttnn.deallocate(u)
            d = ttnn.matmul(a_, self.hc_d, compute_kernel_config=self.ckc, memory_config=mem)
            ttnn.deallocate(a_)
            c = ttnn.add(hk, d, memory_config=mem)
            ttnn.deallocate(hk)
            ttnn.deallocate(d)
            tok = self._argmax_rm(c, mem)
            ttnn.deallocate(c)
            drafts.append(tok)
            prev = tok
        ttnn.deallocate(h)
        out = ttnn.concat(drafts, dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # [1, 5]
        for t_ in drafts:
            ttnn.deallocate(t_)
        if getattr(self, "out_buf", None) is not None:  # persistent (see Hy4Model.step)
            ttnn.copy(out, self.out_buf)
            ttnn.deallocate(out)
            out = self.out_buf
        return out
