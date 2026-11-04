# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""DFlash 2 drafter on the device (one user), the programs of the traced DFlash step (tt/dflash_traced.py).

    Same math as the host reference tt/dflash_ref.py. Weights are replicated on every die (bfp8 matmul weights); the
    target's embedding and vocab-sharded LM head are reused. Fixed shapes throughout (32 block rows, 16 real), so both
    programs are captured once and replayed:

    ctx   : feats [1,1,32,8H] (the verify pass's context-layer rows, row i at position pos_i) -> fc -> hidden_norm ->
            per layer k / v (k_norm, RoPE) scattered into the layer's K / V ring [1, nkv, R, hd] at slot pos_i mod R
            (one-hot [R, 32] built on device from the host's slot row; rows marked -1 are not written).
    draft : the block [anchor, mask x 15 (+16 pad rows)] -> 6 layers (non-causal attention over [ring; block] with a
            visibility row written by the host) -> final norm -> LM head logits (vocab-sharded, every row) and the
            selector's hidden projection [1,1,32,256] (read back with the logits' per-shard top-k candidates).

    Sliding window: one visibility row for the whole block (positions in (start - window, start)); rows 1..15 of a
    block therefore see up to 15 positions their own window would drop - this only touches draft quality (the verify
    pass decides every emitted token), never the output.
"""
from __future__ import annotations

import json
import os

import torch

import ttnn

_DRAM = ttnn.DRAM_MEMORY_CONFIG
TILE = 32
ROWS = 32  # block rows on device (bs <= 32 real)


def _rot_matrix(hd):
    """x @ M == rotate_half(x) for [.., hd] rows."""
    m = torch.zeros(hd, hd)
    h = hd // 2
    for i in range(h):
        m[i + h, i] = -1.0  # out[i] = -x[i + h]
        m[i, i + h] = 1.0  # out[i + h] = x[i]
    return m


class DFlashDevice:
    def __init__(self, model, ckpt_dir, *, ring=2112, wdtype=ttnn.bfloat8_b, cache_dir=None):
        from safetensors.torch import load_file

        self.model = model
        self.mesh = model.mesh_device
        cfg = json.load(open(os.path.join(ckpt_dir, "config.json")))
        dc = cfg["dflash_config"]
        self.H = cfg["hidden_size"]
        self.nh, self.nkv, self.hd = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"]
        self.g = self.nh // self.nkv
        self.eps = cfg["rms_norm_eps"]
        self.window = cfg["sliding_window"]
        self.theta = cfg["rope_parameters"]["rope_theta"]
        self.bs = dc["block_size"]
        self.mask_id = dc["mask_token_id"]
        self.layer_ids = list(dc["target_layer_ids"])
        self.L = cfg["num_hidden_layers"]
        self.K = dc["conv_kernel_size"]
        self.gs = dc["conv_group_size"]
        self.I = cfg["intermediate_size"]
        self.R = int(ring)
        assert self.R % TILE == 0 and self.R >= self.window + 2 * ROWS
        assert self.K == 2, "the device conv is written for two taps"
        self.ckpt_dir = ckpt_dir
        H, nh, nkv, hd = self.H, self.nh, self.nkv, self.hd
        G = H // self.gs
        cache_dir = cache_dir or os.path.join(os.environ.get("TT_CACHE_PATH", "/tmp"), "dflash2_o122")
        os.makedirs(cache_dir, exist_ok=True)
        w = load_file(os.path.join(ckpt_dir, "model.safetensors"))
        rep = ttnn.ReplicateTensorToMesh(self.mesh)

        def put(name, t, dtype=wdtype, layout=ttnn.TILE_LAYOUT):
            return ttnn.as_tensor(
                t.contiguous(), dtype=dtype, layout=layout, device=self.mesh, memory_config=_DRAM, mesh_mapper=rep,
                cache_file_name=os.path.join(cache_dir, f"{name}_{str(dtype).split('.')[-1]}"),
            )

        def vec(name, t):
            return put(name, t.float().reshape(1, 1, 1, -1), dtype=ttnn.bfloat16)

        def norm_w(name, t):
            # ttnn.rms_norm takes the weight as a ROW_MAJOR [hd / 32, 32] stick view
            return put(name, t.float().reshape(1, 1, -1, TILE), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

        self.fc = put("fc", w["fc.weight"].T)  # [8H, H]
        self.hidden_norm = norm_w("hidden_norm", w["hidden_norm.weight"])
        self.final_norm = norm_w("norm", w["norm.weight"])
        self.sel_proj = put("sel_proj", w["candidate_selector.hidden_projection.weight"].T, dtype=ttnn.bfloat16)
        # one-hot group expansion [G, H] (dynamic kernel of group j -> channels j*gs .. (j+1)*gs - 1)
        E = torch.zeros(G, H)
        for j in range(G):
            E[j, j * self.gs : (j + 1) * self.gs] = 1.0
        self.expand = put("expand", E, dtype=ttnn.bfloat16)
        sh = torch.zeros(ROWS, ROWS)
        for i in range(1, ROWS):
            sh[i, i - 1] = 1.0  # row i <- row i - 1 (row 0 <- 0)
        self.shift = put("shift", sh.reshape(1, 1, ROWS, ROWS), dtype=ttnn.bfloat16)
        self.rot = put("rot", _rot_matrix(hd), dtype=ttnn.bfloat16)
        self.layers = []
        for i in range(self.L):
            p = f"layers.{i}"
            a = f"{p}.self_attn"
            lw = {
                "in_ln": norm_w(f"{p}.in_ln", w[f"{p}.input_layernorm.weight"]),
                "post_ln": norm_w(f"{p}.post_ln", w[f"{p}.post_attention_layernorm.weight"]),
                "wqkv": put(f"{p}.wqkv", torch.cat([w[f"{a}.q_proj.weight"], w[f"{a}.k_proj.weight"],
                                                    w[f"{a}.v_proj.weight"]], 0).T),
                "wkv": put(f"{p}.wkv", torch.cat([w[f"{a}.k_proj.weight"], w[f"{a}.v_proj.weight"]], 0).T),
                "q_norm": norm_w(f"{p}.q_norm", w[f"{a}.q_norm.weight"]),
                "k_norm": norm_w(f"{p}.k_norm", w[f"{a}.k_norm.weight"]),
                "wo": put(f"{p}.wo", w[f"{a}.o_proj.weight"].T),
                "wgu": put(f"{p}.wgu", torch.cat([w[f"{p}.mlp.gate_proj.weight"], w[f"{p}.mlp.up_proj.weight"]], 0).T),
                "wdown": put(f"{p}.wdown", w[f"{p}.mlp.down_proj.weight"].T),
            }
            for conv in ("attention_conv", "mlp_conv"):
                proj = w[f"{p}.{conv}.kernel_projection.weight"]  # [2*K*G, H], rows (half, tap, group)
                base = w[f"{p}.{conv}.base_kernel"]  # [2, K, H]
                for half in range(2):
                    for tap in range(self.K):
                        r0 = (half * self.K + tap) * G
                        lw[f"{conv}.proj{half}{tap}"] = put(f"{p}.{conv}.proj{half}{tap}", proj[r0 : r0 + G].T,
                                                            dtype=ttnn.bfloat16)
                        lw[f"{conv}.base{half}{tap}"] = vec(f"{p}.{conv}.base{half}{tap}", base[half, tap])
            # K / V ring of this layer (replicated), zero-initialised; visibility comes from the host row
            lw["ring_k"] = ttnn.from_torch(torch.zeros(1, nkv, self.R, hd), dtype=ttnn.bfloat16,
                                           layout=ttnn.TILE_LAYOUT, device=self.mesh, memory_config=_DRAM,
                                           mesh_mapper=rep)
            lw["ring_v"] = ttnn.clone(lw["ring_k"])
            self.layers.append(lw)
        del w
        # ---- persistent per-round inputs (host-written)
        def zeros(shape, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
            return ttnn.from_torch(torch.zeros(*shape), dtype=dtype, layout=layout, device=self.mesh,
                                   memory_config=_DRAM, mesh_mapper=rep)

        self._zeros = zeros
        self._cstage = {}  # ctx rows CR -> (slot [1,1,1,CR] fp32: target slot per row, -1 = skip; cos; sin)
        self.d_cos = zeros((1, 1, ROWS, hd))
        self.d_sin = zeros((1, 1, ROWS, hd))
        self.d_vis = zeros((1, 1, 1, self.R + ROWS))  # 0 visible / -inf hidden, ring slots then block rows
        self.d_tok = None  # built lazily from the model's decode-token layout (first draft_inputs call)
        self.feats = None  # [1,1,ROWS,8H] persistent ctx input (the verify pass copies its captures in)
        self.scale = hd ** -0.5
        self.ckc = ttnn.init_device_compute_kernel_config(
            self.mesh.arch(), math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        self.ckc_hi = ttnn.init_device_compute_kernel_config(
            self.mesh.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        # host-side state
        self.slot_pos = torch.full((self.R,), -(1 << 30), dtype=torch.long)  # position held by each ring slot

    # ------------------------------------------------------------------------------------------------ host helpers
    def _rope_rows(self, positions, n=ROWS):
        inv = 1.0 / (self.theta ** (torch.arange(0, self.hd, 2, dtype=torch.float32) / self.hd))
        p = torch.zeros(n)
        p[: len(positions)] = torch.tensor(positions, dtype=torch.float32)
        f = p[:, None] * inv[None]
        emb = torch.cat([f, f], dim=-1)
        return emb.cos().reshape(1, 1, n, self.hd), emb.sin().reshape(1, 1, n, self.hd)

    def _host(self, t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
        return ttnn.from_torch(t, dtype=dtype, layout=layout)

    def _write(self, dev_t, host_t):
        ttnn.copy_host_to_device_tensor(host_t, dev_t)

    def reset(self):
        self.slot_pos[:] = -(1 << 30)

    def ctx_stage(self, CR):
        if CR not in self._cstage:
            z = self._zeros
            iota = ttnn.from_torch(
                torch.arange(self.R, dtype=torch.float32).reshape(1, 1, self.R, 1).expand(1, 1, self.R, CR).contiguous(),
                dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=self.mesh, memory_config=_DRAM,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
            )  # [1,1,R,CR]: slot index down the rows (compared with the per-row target slot, broadcast over rows)
            self._cstage[CR] = (z((1, 1, 1, CR), dtype=ttnn.float32), z((1, 1, CR, self.hd)), z((1, 1, CR, self.hd)),
                                iota)
        return self._cstage[CR]

    def stage_ctx(self, positions, CR):
        """Rows 0..n-1 of the next CR-row ctx run land at `positions` (rows past n are skipped). Only the last
        window + bs positions can ever be visible: older rows are skipped (a long prefill segment)."""
        slot_t, cos_t, sin_t, _iota = self.ctx_stage(CR)
        slots = torch.full((1, 1, 1, CR), -1.0)
        last = positions[-1] if len(positions) else 0
        for i, p in enumerate(positions):
            if p > last - self.R + 2 * ROWS:
                slots[0, 0, 0, i] = float(p % self.R)
                self.slot_pos[p % self.R] = int(p)
        self._write(slot_t, self._host(slots, dtype=ttnn.float32))
        c, s = self._rope_rows(positions, CR)
        self._write(cos_t, self._host(c))
        self._write(sin_t, self._host(s))

    def stage_draft(self, anchor, start, bs=None):
        bs = bs or self.bs
        c, s = self._rope_rows(list(range(start, start + ROWS)))
        self._write(self.d_cos, self._host(c))
        self._write(self.d_sin, self._host(s))
        vis = torch.full((1, 1, 1, self.R + ROWS), float("-inf"))
        ok = (self.slot_pos < start) & (self.slot_pos > start - self.window)
        vis[0, 0, 0, : self.R][ok] = 0.0
        vis[0, 0, 0, self.R : self.R + bs] = 0.0
        self._write(self.d_vis, self._host(vis))
        toks = torch.full((ROWS,), self.mask_id, dtype=torch.int32)
        toks[0] = int(anchor)
        self._write_tokens(toks)

    # ------------------------------------------------------------------------------------------------ device pieces
    def _tok_host(self, toks):
        tok, _p, _r, _pt = self.model.prepare_decode_inputs_host(
            toks, torch.zeros(ROWS, dtype=torch.int32), torch.zeros(ROWS, 1, dtype=torch.int32)
        )
        return tok

    def _write_tokens(self, toks):
        if self.d_tok is None:
            self.d_tok = ttnn.to_device(self._tok_host(toks), self.mesh, memory_config=_DRAM)
        else:
            self._write(self.d_tok, self._tok_host(toks))

    def _mm(self, a, b, hi=False):
        return ttnn.matmul(a, b, memory_config=_DRAM, compute_kernel_config=self.ckc_hi if hi else self.ckc)

    def _rms(self, x, w):
        return ttnn.rms_norm(x, epsilon=self.eps, weight=w, memory_config=_DRAM, compute_kernel_config=self.ckc_hi)

    def _rope(self, x, cos, sin):
        """x [1, n, ROWS, hd] (cos / sin [1,1,ROWS,hd] broadcast over n)."""
        xr = ttnn.matmul(x, self.rot, memory_config=_DRAM, compute_kernel_config=self.ckc_hi)
        a = ttnn.multiply(x, cos, memory_config=_DRAM)
        b = ttnn.multiply(xr, sin, memory_config=_DRAM)
        ttnn.deallocate(xr)
        out = ttnn.add(a, b, memory_config=_DRAM)
        ttnn.deallocate(a)
        ttnn.deallocate(b)
        return out

    def _heads(self, x, n, rows=ROWS):
        """[1,1,rows,n*hd] -> [1,n,rows,hd]."""
        x = ttnn.reshape(x, (1, rows, n, self.hd))
        return ttnn.permute(x, (0, 2, 1, 3))

    def _conv(self, lw, conv, half, h, dyn):
        """One grouped dynamic causal conv over the block rows: sum over taps of (base[tap] + dyn[tap]) * shift^tap(h),
        dyn[tap] the [1,1,ROWS,G] dynamic kernel (expanded to the channels here)."""
        out = None
        x = h
        for tap in range(self.K):
            if tap > 0:
                x = ttnn.matmul(self.shift, x, memory_config=_DRAM, compute_kernel_config=self.ckc_hi)
            de = self._mm(dyn[tap], self.expand, hi=True)  # [1,1,ROWS,H]
            k = ttnn.add(de, lw[f"{conv}.base{half}{tap}"], memory_config=_DRAM)
            ttnn.deallocate(de)
            term = ttnn.multiply(k, x, memory_config=_DRAM)
            ttnn.deallocate(k)
            if tap > 0:
                ttnn.deallocate(x)
            out = term if out is None else ttnn.add(out, term, memory_config=_DRAM)
        return out

    def ctx_body(self, feats):
        """feats [1,1,CR,8H] -> the K / V rows of every layer written into the rings at the staged slots."""
        CR = int(feats.shape[-2])
        slot_t, cos_t, sin_t, iota = self.ctx_stage(CR)
        x = self._mm(feats, self.fc)  # [1,1,CR,H]
        x = self._rms(x, self.hidden_norm)
        # one-hot [1,1,R,CR]: slot j <- row i
        sel = ttnn.eq(iota, slot_t, memory_config=_DRAM)  # [1,1,R,CR] fp32
        sel = ttnn.typecast(sel, ttnn.bfloat16)
        keep = ttnn.sum(sel, dim=-1, keepdim=True)  # [1,1,R,1]
        keep = ttnn.rsub(keep, 1.0)
        sel_h = ttnn.repeat(sel, (1, self.nkv, 1, 1))
        for lw in self.layers:
            kv = self._mm(x, lw["wkv"])  # [1,1,ROWS,2*nkv*hd]
            k = ttnn.slice(kv, (0, 0, 0, 0), (1, 1, CR, self.nkv * self.hd))
            v = ttnn.slice(kv, (0, 0, 0, self.nkv * self.hd), (1, 1, CR, 2 * self.nkv * self.hd))
            ttnn.deallocate(kv)
            k = self._heads(k, self.nkv, CR)
            k = self._rms(k, lw["k_norm"])
            k = self._rope(k, cos_t, sin_t)
            v = self._heads(v, self.nkv, CR)
            for ring, rows in ((lw["ring_k"], k), (lw["ring_v"], v)):
                new = ttnn.matmul(sel_h, rows, memory_config=_DRAM, compute_kernel_config=self.ckc_hi)  # [1,nkv,R,hd]
                kept = ttnn.multiply(ring, keep, memory_config=_DRAM)
                upd = ttnn.add(kept, new, memory_config=_DRAM)
                ttnn.deallocate(kept)
                ttnn.deallocate(new)
                ttnn.copy(upd, ring)
                ttnn.deallocate(upd)
                ttnn.deallocate(rows)
        for t in (x, sel, keep, sel_h):
            ttnn.deallocate(t)
        return ()

    def draft_body(self):
        """The block through the drafter -> (logits [1,1,ROWS,V/TP] vocab-sharded TILE, hp [1,1,ROWS,256])."""
        m = self.model
        x = m.embd(m._decode_tokens_view(self.d_tok, ROWS))
        x = ttnn.to_memory_config(ttnn.reshape(x, (1, 1, ROWS, x.shape[-1])), _DRAM)
        x = self._full(x)
        nh, nkv, hd, g = self.nh, self.nkv, self.hd, self.g
        for lw in self.layers:
            res = x
            h = self._rms(x, lw["in_ln"])
            dyn0 = [self._mm(h, lw[f"attention_conv.proj0{t}"], hi=True) for t in range(self.K)]
            dyn1 = [self._mm(h, lw[f"attention_conv.proj1{t}"], hi=True) for t in range(self.K)]
            hc = self._conv(lw, "attention_conv", 0, h, dyn0)
            ttnn.deallocate(h)
            qkv = self._mm(hc, lw["wqkv"])
            ttnn.deallocate(hc)
            q = ttnn.slice(qkv, (0, 0, 0, 0), (1, 1, ROWS, nh * hd))
            k = ttnn.slice(qkv, (0, 0, 0, nh * hd), (1, 1, ROWS, (nh + nkv) * hd))
            v = ttnn.slice(qkv, (0, 0, 0, (nh + nkv) * hd), (1, 1, ROWS, (nh + 2 * nkv) * hd))
            ttnn.deallocate(qkv)
            q = self._rope(self._rms(self._heads(q, nh), lw["q_norm"]), self.d_cos, self.d_sin)  # [1,nh,ROWS,hd]
            k = self._rope(self._rms(self._heads(k, nkv), lw["k_norm"]), self.d_cos, self.d_sin)  # [1,nkv,ROWS,hd]
            v = self._heads(v, nkv)
            # GQA: query heads of kv head j are j*g .. j*g+g-1 -> [1, nkv, g*ROWS, hd]
            q = ttnn.reshape(q, (1, nkv, g * ROWS, hd))
            ka = ttnn.concat([lw["ring_k"], k], dim=2, memory_config=_DRAM)  # [1,nkv,R+ROWS,hd]
            va = ttnn.concat([lw["ring_v"], v], dim=2, memory_config=_DRAM)
            ttnn.deallocate(k)
            ttnn.deallocate(v)
            kt = ttnn.transpose(ka, -2, -1)
            ttnn.deallocate(ka)
            s = ttnn.matmul(q, kt, memory_config=_DRAM, compute_kernel_config=self.ckc_hi)  # [1,nkv,g*ROWS,R+ROWS]
            ttnn.deallocate(kt)
            ttnn.deallocate(q)
            s = ttnn.multiply(s, self.scale, memory_config=_DRAM)
            s2 = ttnn.add(s, self.d_vis, memory_config=_DRAM)
            ttnn.deallocate(s)
            p = ttnn.softmax(s2, dim=-1, memory_config=_DRAM, compute_kernel_config=self.ckc_hi)
            ttnn.deallocate(s2)
            o = ttnn.matmul(p, va, memory_config=_DRAM, compute_kernel_config=self.ckc_hi)  # [1,nkv,g*ROWS,hd]
            ttnn.deallocate(p)
            ttnn.deallocate(va)
            o = ttnn.reshape(o, (1, nh, ROWS, hd))
            o = ttnn.permute(o, (0, 2, 1, 3))
            o = ttnn.reshape(o, (1, 1, ROWS, nh * hd))
            o2 = self._mm(o, lw["wo"])
            ttnn.deallocate(o)
            o3 = self._conv(lw, "attention_conv", 1, o2, dyn1)
            ttnn.deallocate(o2)
            for d in dyn0 + dyn1:
                ttnn.deallocate(d)
            x = ttnn.add(res, o3, memory_config=_DRAM)
            ttnn.deallocate(o3)
            ttnn.deallocate(res)
            res = x
            h = self._rms(x, lw["post_ln"])
            dyn0 = [self._mm(h, lw[f"mlp_conv.proj0{t}"], hi=True) for t in range(self.K)]
            dyn1 = [self._mm(h, lw[f"mlp_conv.proj1{t}"], hi=True) for t in range(self.K)]
            hc = self._conv(lw, "mlp_conv", 0, h, dyn0)
            ttnn.deallocate(h)
            gu = self._mm(hc, lw["wgu"])
            ttnn.deallocate(hc)
            gate = ttnn.slice(gu, (0, 0, 0, 0), (1, 1, ROWS, self.I))
            up = ttnn.slice(gu, (0, 0, 0, self.I), (1, 1, ROWS, 2 * self.I))
            ttnn.deallocate(gu)
            a = ttnn.multiply(ttnn.silu(gate), up, memory_config=_DRAM)
            ttnn.deallocate(gate)
            ttnn.deallocate(up)
            mo = self._mm(a, lw["wdown"])
            ttnn.deallocate(a)
            mo2 = self._conv(lw, "mlp_conv", 1, mo, dyn1)
            ttnn.deallocate(mo)
            for d in dyn0 + dyn1:
                ttnn.deallocate(d)
            x = ttnn.add(res, mo2, memory_config=_DRAM)
            ttnn.deallocate(mo2)
            ttnn.deallocate(res)
        h = self._rms(x, self.final_norm)
        ttnn.deallocate(x)
        logits = ttnn.linear(h, m.lm_head_weight, compute_kernel_config=m._lm_head_ckc())  # [1,1,ROWS,V/TP]
        hp = self._mm(h, self.sel_proj, hi=True)  # [1,1,ROWS,256]
        ttnn.deallocate(h)
        return logits, hp

    def _full(self, x):
        """A [1,1,ROWS,H/TP] fractured row block -> replicated [1,1,ROWS,H] (no-op when already full)."""
        if x.shape[-1] == self.H:
            return x
        from models.tt_transformers.tt.ccl import tt_all_gather

        return tt_all_gather(x, self.mesh, self.model.tt_ccl, cluster_axis=None, dim=3,
                             topology=self.model.args.ccl_topology())
