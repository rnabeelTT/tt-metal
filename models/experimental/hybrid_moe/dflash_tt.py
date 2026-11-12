# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""DFlash block drafter on a TT mesh (first version: eager, weights replicated on every die).

The drafter (z-lab ``DFlashDraftModel``, e.g. ``nvidia/MiniMax-M2.7-DFlash``): context features = the target's outputs
of ``target_layer_ids`` (concatenated) -> ``fc`` -> ``hidden_norm``; each of its layers attends, NON-causally, from the
block rows ``[last token, mask x (bs - 1)]`` (target embeddings) to ``[context ; block]`` keys / values, where the
context keys / values are that layer's k_proj / v_proj of the context features (k per-head RMSNorm, RoPE at the
context positions). Drafts = the target LM head's argmax over the final-normed block rows 1 .. bs - 1.

Context K / V per layer are cached on the device ([1, heads, ctx, head_dim], grown by concatenation as tokens are
accepted: this first version is eager, not trace-safe).
"""

from __future__ import annotations

import json
import os

import torch
from safetensors import safe_open
from transformers import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

import ttnn
from models.demos.blackhole.qwen36.tt.attn_decode.kvrows import kv_rows_update

DRAM = ttnn.DRAM_MEMORY_CONFIG
# DFLASH_KVROWS (default 1): the drafter's context / block rows written in one multi-row program each
_KVROWS = os.environ.get("DFLASH_KVROWS", "1") == "1"


class DFlashDrafterTT:
    def __init__(self, mesh, snapshot, embed_host, lm_head, vocab_size, dtype=ttnn.bfloat8_b, tp=False):
        """tp: the attention heads and the MLP split over the mesh (q / k / v / gate / up by output column, o / down by
        input row; the caller all-reduces after o and down). embed_host: the target's [V, H] bf16 embedding table (host; the block rows are looked up on the host).
        lm_head: the target's device LM head weight (vocab-sharded [H, V / tp] per die, as model.lm_head_weight)."""
        self.mesh = mesh
        self.cfg = Qwen3Config.from_pretrained(snapshot)
        raw = json.load(open(os.path.join(snapshot, "config.json")))
        self.bs = int(raw["block_size"])
        self.mask_id = int(raw["dflash_config"]["mask_token_id"])
        self.taps = [int(t) for t in raw["dflash_config"]["target_layer_ids"]]
        self.NH, self.NKV, self.HD = self.cfg.num_attention_heads, self.cfg.num_key_value_heads, self.cfg.head_dim
        self.H, self.eps = self.cfg.hidden_size, self.cfg.rms_norm_eps
        if self.NH != self.NKV:
            raise ValueError(f"GQA drafter ({self.NH} q / {self.NKV} kv heads) not implemented")
        self.embed_host, self.lm_head, self.vocab = embed_host, lm_head, vocab_size
        self.tp = tp
        self.nh = self.NH // len(mesh.get_device_ids()) if tp else self.NH  # heads per device
        self.rope = Qwen3RotaryEmbedding(self.cfg)  # YaRN from the drafter's rope_scaling
        rep = ttnn.ReplicateTensorToMesh(mesh)
        f = safe_open(os.path.join(snapshot, "model.safetensors"), "pt")

        def w(key, dt=dtype, split=None):  # [out, in] -> device [in, out]; split: the device dim sharded over the mesh
            mapper = rep if not (tp and split is not None) else ttnn.ShardTensorToMesh(mesh, dim=split)
            return ttnn.from_torch(
                f.get_tensor(key).T.contiguous(),
                dtype=dt,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                mesh_mapper=mapper,
                memory_config=DRAM,
            )

        def g(key):  # norm gamma -> [1, 1, 1, n] bf16 tile
            return ttnn.from_torch(
                f.get_tensor(key).reshape(1, 1, 1, -1).to(torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                mesh_mapper=rep,
                memory_config=DRAM,
            )

        self.fc, self.hidden_norm, self.norm = w("fc.weight"), g("hidden_norm.weight"), g("norm.weight")
        self.layers = []
        for i in range(self.cfg.num_hidden_layers):
            p = f"layers.{i}."
            self.layers.append(
                {
                    "in_norm": g(p + "input_layernorm.weight"),
                    "post_norm": g(p + "post_attention_layernorm.weight"),
                    "q": w(p + "self_attn.q_proj.weight", split=-1),
                    "k": w(p + "self_attn.k_proj.weight", split=-1),
                    "v": w(p + "self_attn.v_proj.weight", split=-1),
                    "o": w(p + "self_attn.o_proj.weight", split=-2),
                    "q_norm": g(p + "self_attn.q_norm.weight"),
                    "k_norm": g(p + "self_attn.k_norm.weight"),
                    "gate": w(p + "mlp.gate_proj.weight", split=-1),
                    "up": w(p + "mlp.up_proj.weight", split=-1),
                    "down": w(p + "mlp.down_proj.weight", split=-2),
                }
            )
        self.debug = None  # a list: draft() appends every layer's output (host, die 0) for comparisons
        self.ckc = ttnn.init_device_compute_kernel_config(
            mesh.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        self.reset()

    def reset(self):
        for L in getattr(self, "layers", []):
            for k in ("K", "V"):
                if L.get(k) is not None:
                    ttnn.deallocate(L[k])
                L[k] = None
        self.ctx_len = 0

    # -- helpers ---------------------------------------------------------------------------------------------------
    def _rms(self, x, gamma):
        return ttnn.multiply(ttnn.rms_norm(x, epsilon=self.eps), gamma)

    def _lin(self, x, wt):
        return ttnn.linear(x, wt, compute_kernel_config=self.ckc, memory_config=DRAM)

    def _cos_sin(self, positions):
        pos = torch.tensor(positions).reshape(1, -1)
        cos, sin = self.rope(torch.zeros(1, 1, dtype=torch.float32), pos)  # [1, n, HD] (YaRN-scaled)
        up = lambda t: ttnn.from_torch(
            t.reshape(1, 1, len(positions), self.HD).to(torch.bfloat16),  # noqa: E731
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
            memory_config=DRAM,
        )
        return up(cos), up(sin)

    def _heads(self, t, n):
        """[1, 1, n, NH * HD] -> [1, NH, n, HD]"""
        t = ttnn.reshape(t, (1, n, self.NH, self.HD))
        return ttnn.permute(t, (0, 2, 1, 3))

    def _rope(self, x, cos, sin):
        h = self.HD // 2
        x1 = ttnn.slice(x, (0, 0, 0, 0), (x.shape[0], x.shape[1], x.shape[2], h))
        x2 = ttnn.slice(x, (0, 0, 0, h), (x.shape[0], x.shape[1], x.shape[2], self.HD))
        rot = ttnn.concat([ttnn.neg(x2), x1], dim=-1)
        return ttnn.add(ttnn.multiply(x, cos), ttnn.multiply(rot, sin))

    # -- context ---------------------------------------------------------------------------------------------------
    def add_context(self, taps, positions):
        """taps: list of len(self.taps) replicated [1, 1, n, H] tensors (the target's outputs of those layers for n
        consecutive tokens at `positions`): appends their context K / V to every layer's cache."""
        n = len(positions)
        if positions[0] != self.ctx_len:
            raise ValueError(f"context positions {positions[0]}.. do not continue the cache (length {self.ctx_len})")
        feats = ttnn.concat(taps, dim=-1)
        c = self._rms(self._lin(feats, self.fc), self.hidden_norm)
        cos, sin = self._cos_sin(positions)
        for L in self.layers:
            k = self._rope(self._rms(self._heads(self._lin(c, L["k"]), n), L["k_norm"]), cos, sin)
            v = self._heads(self._lin(c, L["v"]), n)
            for name, t in (("K", k), ("V", v)):
                if L[name] is None:
                    L[name] = t
                else:
                    cat = ttnn.concat([L[name], t], dim=2)
                    ttnn.deallocate(L[name])
                    ttnn.deallocate(t)
                    L[name] = cat
        self.ctx_len += n

    # -- draft -----------------------------------------------------------------------------------------------------
    def draft(self, token, k):
        """The block [token, mask x (bs - 1)] at positions ctx_len .. ctx_len + bs - 1 (token = the target's next token at
        position ctx_len): returns the first k drafts (tokens for positions ctx_len + 1 ..)."""
        bs, p = self.bs, self.ctx_len
        ids = torch.tensor([token] + [self.mask_id] * (bs - 1))
        x = ttnn.from_torch(
            self.embed_host[ids].reshape(1, 1, bs, self.H).to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
            memory_config=DRAM,
        )
        cos, sin = self._cos_sin(list(range(p, p + bs)))
        scale = self.HD**-0.5
        for L in self.layers:
            h = self._rms(x, L["in_norm"])
            q = self._rope(self._rms(self._heads(self._lin(h, L["q"]), bs), L["q_norm"]), cos, sin)
            kn = self._rope(self._rms(self._heads(self._lin(h, L["k"]), bs), L["k_norm"]), cos, sin)
            vn = self._heads(self._lin(h, L["v"]), bs)
            K = ttnn.concat([L["K"], kn], dim=2) if L["K"] is not None else kn
            V = ttnn.concat([L["V"], vn], dim=2) if L["V"] is not None else vn
            s = ttnn.matmul(q, ttnn.transpose(K, -2, -1), compute_kernel_config=self.ckc)  # [1, NH, bs, ctx + bs]
            s = ttnn.softmax(ttnn.multiply(s, scale), dim=-1)  # non-causal: every block row sees everything
            o = ttnn.matmul(s, V, compute_kernel_config=self.ckc)  # [1, NH, bs, HD]
            o = ttnn.reshape(ttnn.permute(o, (0, 2, 1, 3)), (1, 1, bs, self.NH * self.HD))
            x = ttnn.add(x, self._lin(o, L["o"]))
            h2 = self._rms(x, L["post_norm"])
            mlp = ttnn.multiply(ttnn.silu(self._lin(h2, L["gate"])), self._lin(h2, L["up"]))
            x = ttnn.add(x, self._lin(mlp, L["down"]))
            if self.debug is not None:
                self.debug.append(ttnn.to_torch(ttnn.get_device_tensors(x)[0]).float().reshape(bs, -1))
        h = self._rms(x, self.norm)
        if self.debug is not None:
            self.debug.append(ttnn.to_torch(ttnn.get_device_tensors(h)[0]).float().reshape(bs, -1))
        logits = ttnn.linear(h, self.lm_head, memory_config=DRAM)  # vocab-sharded [1, 1, bs, V / tp]
        lg = ttnn.to_torch(logits, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh, dim=-1)).float()
        lg = lg.reshape(-1, lg.shape[-1])[1 : 1 + k, : self.vocab]
        return [int(t) for t in lg.argmax(-1)]


class DFlashDrafterTraced(DFlashDrafterTT):
    """Trace-safe drafter: a fixed [1, NH, cmax, HD] context K / V cache per layer, written through one-hot matmuls
    (cache = cache * keep + onehot @ rows), so every program has fixed shapes and all inputs live in persistent
    buffers the host overwrites before a replay.

    ctx_body(W): the context features of W rows (self.c_taps[W], [1, 1, W, n_taps * H]) -> each layer's K / V rows
        written at the cache positions given by self.c_onehot[W] ([1, 1, cmax, W]; an all-zero column writes nothing)
        and self.c_keep ([1, 1, cmax, HD]: 0 on written positions).
    draft_body: the block [token, mask x (bs - 1)] (self.d_x, embeddings) attends to the cache (self.d_mask: 0 on the
        valid context, -1e30 elsewhere) and to its own K / V, non-causally -> per-die argmax / max of the LM head.
    All buffers are allocated in __init__: build this BEFORE any trace of the target is captured."""

    def __init__(self, mesh, snapshot, embed_host, lm_head, vocab_size, cmax, widths=(4, 128), dtype=ttnn.bfloat8_b):
        super().__init__(mesh, snapshot, embed_host, lm_head, vocab_size, dtype=dtype)
        if cmax % 32:
            raise ValueError(f"cmax {cmax} must be a multiple of 32")
        self.cmax, self.widths = cmax, tuple(widths)
        z = self._zeros
        for L in self.layers:
            L["K"], L["V"] = z((1, self.NH, cmax, self.HD)), z((1, self.NH, cmax, self.HD))
        nt = len(self.taps)
        self.c_taps = {w: z((1, 1, w, nt * self.H)) for w in widths}
        self.c_cos = {w: z((1, 1, w, self.HD)) for w in widths}
        self.c_sin = {w: z((1, 1, w, self.HD)) for w in widths}
        self.c_onehot = {w: z((1, 1, cmax, w)) for w in widths}
        self.c_keep = z((1, 1, cmax, self.HD))
        bs = self.bs
        self.d_x = z((1, 1, bs, self.H))
        self.d_cos, self.d_sin = z((1, 1, bs, self.HD)), z((1, 1, bs, self.HD))
        self.d_mask = z((1, 1, bs, cmax))
        self.traces, self.outs = {}, {}

    def _zeros(self, shape):
        return ttnn.from_torch(
            torch.zeros(*shape, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
            memory_config=DRAM,
        )

    @staticmethod
    def _put(dev_t, host):
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(host.to(torch.bfloat16).contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT), dev_t
        )

    def reset(self):
        self.ctx_len = 0  # the cache needs no clearing: positions >= ctx_len are masked off and rewritten before use

    # -- bodies ----------------------------------------------------------------------------------------------------
    def ctx_body(self, w):
        c = self._rms(self._lin(self.c_taps[w], self.fc), self.hidden_norm)
        oh = ttnn.repeat(self.c_onehot[w], (1, self.NH, 1, 1))  # [1, NH, cmax, w]
        for L in self.layers:
            k = self._rope(self._rms(self._heads(self._lin(c, L["k"]), w), L["k_norm"]), self.c_cos[w], self.c_sin[w])
            v = self._heads(self._lin(c, L["v"]), w)
            for name, rows in (("K", k), ("V", v)):
                new = ttnn.add(
                    ttnn.multiply(L[name], self.c_keep), ttnn.matmul(oh, rows, compute_kernel_config=self.ckc)
                )
                ttnn.copy(new, L[name])
                ttnn.deallocate(new)
        return ()

    def draft_body(self):
        bs, x, scale = self.bs, self.d_x, self.HD**-0.5
        for L in self.layers:
            h = self._rms(x, L["in_norm"])
            q = self._rope(self._rms(self._heads(self._lin(h, L["q"]), bs), L["q_norm"]), self.d_cos, self.d_sin)
            kn = self._rope(self._rms(self._heads(self._lin(h, L["k"]), bs), L["k_norm"]), self.d_cos, self.d_sin)
            vn = self._heads(self._lin(h, L["v"]), bs)
            s_ctx = ttnn.add(
                ttnn.multiply(ttnn.matmul(q, L["K"], transpose_b=True, compute_kernel_config=self.ckc), scale),
                self.d_mask,
            )
            s_blk = ttnn.multiply(ttnn.matmul(q, kn, transpose_b=True, compute_kernel_config=self.ckc), scale)
            p = ttnn.softmax(ttnn.concat([s_ctx, s_blk], dim=-1), dim=-1)
            p_ctx = ttnn.slice(p, (0, 0, 0, 0), (1, self.NH, bs, self.cmax))
            p_blk = ttnn.slice(p, (0, 0, 0, self.cmax), (1, self.NH, bs, self.cmax + bs))
            o = ttnn.add(
                ttnn.matmul(p_ctx, L["V"], compute_kernel_config=self.ckc),
                ttnn.matmul(p_blk, vn, compute_kernel_config=self.ckc),
            )
            o = ttnn.reshape(ttnn.permute(o, (0, 2, 1, 3)), (1, 1, bs, self.NH * self.HD))
            x = ttnn.add(x, self._lin(o, L["o"]))
            h2 = self._rms(x, L["post_norm"])
            mlp = ttnn.multiply(ttnn.silu(self._lin(h2, L["gate"])), self._lin(h2, L["up"]))
            x = ttnn.add(x, self._lin(mlp, L["down"]))
        h = self._rms(x, self.norm)
        logits = ttnn.linear(h, self.lm_head, memory_config=DRAM)  # vocab-sharded [1, 1, bs, V / tp]
        return argmax_max(logits)

    def pass_body(self):
        """One decode pass of the drafter: the last verify's rows into the cache, then the next block's drafts."""
        self.ctx_body(4)
        return self.draft_body()

    # -- host side -------------------------------------------------------------------------------------------------
    def stage_ctx(self, w, positions, taps_host=None):
        """Context rows for positions (len <= w; the remaining rows of the w-row input write nothing). taps_host:
        [n, n_taps * H] (None: c_taps[w] was filled on the device, e.g. by the verify trace)."""
        n = len(positions)
        if positions and positions[0] != self.ctx_len:
            raise ValueError(f"context positions {positions[0]}.. do not continue the cache (length {self.ctx_len})")
        if self.ctx_len + n + self.bs > self.cmax:
            raise ValueError(f"drafter context {self.ctx_len + n} + block {self.bs} exceeds cmax {self.cmax}")
        if taps_host is not None:
            t = torch.zeros(1, 1, w, taps_host.shape[-1])
            t[0, 0, :n] = taps_host
            self._put(self.c_taps[w], t)
        pos = list(positions) + [0] * (w - n)
        cos, sin = self.rope(torch.zeros(1, 1, dtype=torch.float32), torch.tensor(pos).reshape(1, -1))
        self._put(self.c_cos[w], cos.reshape(1, 1, w, self.HD))
        self._put(self.c_sin[w], sin.reshape(1, 1, w, self.HD))
        oh = torch.zeros(1, 1, self.cmax, w)
        keep = torch.ones(1, 1, self.cmax, self.HD)
        for j, p in enumerate(positions):
            oh[0, 0, p, j] = 1
            keep[0, 0, p] = 0
        self._put(self.c_onehot[w], oh)
        self._put(self.c_keep, keep)
        self.ctx_len += n

    def stage_draft(self, token):
        bs, p = self.bs, self.ctx_len
        ids = torch.tensor([token] + [self.mask_id] * (bs - 1))
        self._put(self.d_x, self.embed_host[ids].reshape(1, 1, bs, self.H))
        cos, sin = self.rope(torch.zeros(1, 1, dtype=torch.float32), torch.arange(p, p + bs).reshape(1, -1))
        self._put(self.d_cos, cos.reshape(1, 1, bs, self.HD))
        self._put(self.d_sin, sin.reshape(1, 1, bs, self.HD))
        m = torch.full((1, 1, bs, self.cmax), -1e30)
        m[..., :p] = 0
        self._put(self.d_mask, m)

    def run(self, name, body):
        """Replay the trace `name` (or run `body` eagerly when not captured); returns the per-row argmax over the
        vocab-sharded logits when the body returns (argmax, max)."""
        if name in self.traces:
            ttnn.execute_trace(self.mesh, self.traces[name], cq_id=0, blocking=False)
            outs = self.outs[name]
        else:
            outs = body()
        return read_argmax(self.mesh, outs, int(ttnn.get_device_tensors(self.lm_head)[0].shape[-1])) if outs else None

    def capture(self, name, body):
        tid = ttnn.begin_trace_capture(self.mesh, cq_id=0)
        self.outs[name] = body()
        ttnn.end_trace_capture(self.mesh, tid, cq_id=0)
        self.traces[name] = tid


def argmax_max(logits):
    """Per-die argmax and max of vocab-sharded logits [1, 1, rows, V / tp] (read back by read_argmax)."""
    lg = ttnn.to_layout(logits, ttnn.ROW_MAJOR_LAYOUT)
    am = ttnn.argmax(lg, dim=-1, keepdim=True)
    mx = ttnn.gather(lg, -1, am)
    ttnn.deallocate(lg)
    return am, mx


def read_argmax(mesh, outs, per_die):
    """Global token ids per row from argmax_max's outputs; per_die = the LM head's columns per die."""
    am, mx = outs
    comp = ttnn.ConcatMeshToTensor(mesh, dim=0)
    nd = len(mesh.get_device_ids())
    am_h = ttnn.to_torch(am, mesh_composer=comp).reshape(nd, -1).long()
    mx_h = ttnn.to_torch(mx, mesh_composer=comp).float().reshape(nd, -1)
    shard = mx_h.argmax(0)
    return [int(am_h[shard[r], r]) + int(shard[r]) * per_die for r in range(am_h.shape[1])]


class DFlashDrafterPaged(DFlashDrafterTraced):
    """The drafter on the target's own attention kernels: a paged context K / V cache ([blocks, NH, block, HD] per
    layer; every "user" maps to the same pages) written row by row with paged_update_cache, and the block's attention
    as paged SDPA-decode over 8 "users" (the block rows) that each see positions 0 .. ctx_len + bs - 1: the context
    plus the whole block, non-causally. Costs follow ctx_len, not the cache size.

    pass_body: the last verify's rows (c_taps[R], R = 4) -> K / V rows written at c_idx (row j: -1 except column j =
    that row's position, or -1 for a rejected row); the block (d_x) -> its K / V rows written at b_idx (row j: -1
    except column j = ctx_len + j), SDPA with b_pos (= ctx_len + bs - 1 for every row) -> per-die argmax / max.
    The prompt's context goes in eagerly (fill_prompt). Build BEFORE any trace of the target is captured."""

    def __init__(
        self, mesh, snapshot, embed_host, lm_head, vocab_size, cmax, rows=4, block=64, dtype=ttnn.bfloat8_b, tp=False
    ):
        DFlashDrafterTT.__init__(self, mesh, snapshot, embed_host, lm_head, vocab_size, dtype=dtype, tp=tp)
        if cmax % block:
            raise ValueError(f"cmax {cmax} must be a multiple of the page size {block}")
        self.cmax, self.rows, self.block = cmax, rows, block
        nb = cmax // block
        z = self._zeros
        for L in self.layers:
            L["K"], L["V"] = z((nb, self.nh, block, self.HD)), z((nb, self.nh, block, self.HD))
        bs, nt = self.bs, len(self.taps)

        def i32(t):
            return ttnn.from_torch(
                t.to(torch.int32),
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=mesh,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
                memory_config=DRAM,
            )

        self.pt = {n: i32(torch.arange(nb).repeat(n, 1)) for n in (1, rows, bs)}
        self.c_taps = {rows: z((1, 1, rows, nt * self.H))}
        self.c_cos, self.c_sin = z((1, rows, 1, self.HD)), z((1, rows, 1, self.HD))
        self.c_idx = i32(torch.full((rows, rows), -1))
        self.d_x = z((1, 1, bs, self.H))
        self.d_cos, self.d_sin = z((1, bs, 1, self.HD)), z((1, bs, 1, self.HD))
        self.b_idx = i32(torch.full((bs, bs), -1))
        self.b_pos = i32(torch.zeros(bs))
        self.shard = {
            n: ttnn.create_sharded_memory_config(
                shape=(ttnn.TILE_SIZE, self.HD),
                core_grid=ttnn.CoreGrid(x=n, y=1),
                strategy=ttnn.ShardStrategy.HEIGHT,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
            for n in (rows, bs)
        }
        self.shard_v = {
            n: ttnn.create_sharded_memory_config(
                shape=(ttnn.TILE_SIZE, self.HD),
                core_grid=ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 1), ttnn.CoreCoord(n - 1, 1))}),
                strategy=ttnn.ShardStrategy.HEIGHT,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
            for n in (rows, bs)
        }
        grid = mesh.compute_with_storage_grid_size()
        self.sdpa_cfg = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=(grid.x, grid.y), exp_approx_mode=False, q_chunk_size=0, k_chunk_size=0
        )
        self.traces, self.outs = {}, {}
        self.ctx_len = self.base = 0
        # DFLASH_FUSED (default 1, tp): q|k|v and gate|up fused per die, fc split by column (+ all-gather), and the
        # decode-row projections on explicit 1D program configs (the default ttnn.linear ran these skinny matmuls at
        # 70-240 GB/s of DRAM bandwidth, the 1D configs at ~320-350: bench_drmm.py)
        self.fused = tp and os.environ.get("DFLASH_FUSED", "1") == "1"
        self._mm_cfgs = {}
        if self.fused:
            for L in self.layers:
                L["qkv"] = ttnn.concat([L["q"], L["k"], L["v"]], dim=-1, memory_config=DRAM)
                L["gu"] = ttnn.concat([L["gate"], L["up"]], dim=-1, memory_config=DRAM)
                for name in ("q", "k", "v", "gate", "up"):
                    ttnn.deallocate(L.pop(name))
            f = safe_open(os.path.join(snapshot, "model.safetensors"), "pt")
            ttnn.deallocate(self.fc)
            self.fc = ttnn.from_torch(
                f.get_tensor("fc.weight").T.contiguous(),
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=-1),
                memory_config=DRAM,
            )

    def _mm(self, x, wt):
        """A projection of <= 32 rows on an explicit 1D (mcast_in0) program config (at most 64 cores); more rows (the
        eager prompt fill) on the default linear."""
        M, K, N = int(x.shape[-2]), int(wt.shape[-2]), int(wt.shape[-1])
        if not self.fused or M > ttnn.TILE_SIZE:
            return self._lin(x, wt)
        key = (M, K, N)
        if key not in self._mm_cfgs:
            from models.demos.blackhole.qwen36.tt import tp_common as tpc

            gx = self.mesh.compute_with_storage_grid_size().x
            self._mm_cfgs[key] = tpc.create_matmul_1d_decode_progcfg(M, K, N, min(64, N // ttnn.TILE_SIZE), grid_w=gx)
        return ttnn.linear(x, wt, compute_kernel_config=self.ckc, program_config=self._mm_cfgs[key], memory_config=DRAM)

    def _ctx(self, taps):
        """context features [1, 1, n, n_taps * H] -> fc -> hidden_norm (fc column-split + all-gather when fused)"""
        c = self._mm(taps, self.fc)
        if self.fused:
            g = ttnn.all_gather(c, dim=-1, num_links=1, topology=ttnn.Topology.Linear, memory_config=DRAM)
            ttnn.deallocate(c)
            c = g
        return self._rms(c, self.hidden_norm)

    def _proj(self, x, L, n, names):
        """the named projections ("q" / "k" / "v") of n rows, each [1, 1, n, nh * HD] (one fused matmul + slices)"""
        if not self.fused:
            return [self._mm(x, L[nm]) for nm in names]
        y = self._mm(x, L["qkv"])
        w = self.nh * self.HD
        outs = [ttnn.slice(y, (0, 0, 0, i * w), (1, 1, n, (i + 1) * w)) for i in ("qkv".index(nm) for nm in names)]
        ttnn.deallocate(y)
        return outs

    def _mlp(self, h2):
        if not self.fused:
            return ttnn.multiply(ttnn.silu(self._mm(h2, self.L_cur["gate"])), self._mm(h2, self.L_cur["up"]))
        gu = self._mm(h2, self.L_cur["gu"])
        n, w = int(h2.shape[-2]), int(gu.shape[-1]) // 2
        g = ttnn.slice(gu, (0, 0, 0, 0), (1, 1, n, w))
        u = ttnn.slice(gu, (0, 0, 0, w), (1, 1, n, 2 * w))
        ttnn.deallocate(gu)
        return ttnn.multiply(ttnn.silu(g), u)

    @staticmethod
    def _put_i32(dev_t, host):
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(host.to(torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT), dev_t
        )

    def _users(self, t, n):
        """[1, 1, n, nh * HD] -> [1, n, nh, HD] (n users of this device's nh heads)"""
        return ttnn.reshape(t, (1, n, self.nh, self.HD))

    def _reduce(self, t):
        """the partial sums of a row-split projection (tp) -> the full sum on every device"""
        if not self.tp:
            return t
        r = ttnn.all_reduce(t, topology=ttnn.Topology.Linear, memory_config=DRAM)
        ttnn.deallocate(t)
        return r

    def _heads_n(self, t, n):
        """[1, 1, n, nh * HD] -> [1, nh, n, HD]"""
        return ttnn.permute(ttnn.reshape(t, (1, n, self.nh, self.HD)), (0, 2, 1, 3))

    def _rope_u(self, x, cos, sin):  # x [1, n, NH, HD], cos / sin [1, n, 1, HD]
        return self._rope(x, cos, sin)

    def _write_rows(self, L, k, v, n, idx_rows, idx=None):
        if self.nh < ttnn.TILE_SIZE:  # the update kernel takes one tile of heads per user
            k = ttnn.pad(k, [1, n, ttnn.TILE_SIZE, self.HD], [0, 0, 0, 0], 0.0)
            v = ttnn.pad(v, [1, n, ttnn.TILE_SIZE, self.HD], [0, 0, 0, 0], 0.0)
        ks = ttnn.to_memory_config(k, self.shard[n])
        # K and V of a row in one paged_fused_update_cache (its inputs on disjoint cores: V on the next core row)
        vs = ttnn.to_memory_config(v, self.shard_v[n])
        if idx is not None and _KVROWS:  # all n rows in one program (attn_decode/kvrows.py), the same cache bytes
            kv_rows_update(L["K"], ks, L["V"], vs, idx, self.pt[n])
            n_calls = 0
        else:
            n_calls = n
        for j in range(n_calls):
            ttnn.experimental.paged_fused_update_cache(
                L["K"], ks, L["V"], vs, update_idxs_tensor=idx_rows[j], page_table=self.pt[n]
            )
        ttnn.deallocate(ks)
        ttnn.deallocate(vs)

    def _idx_rows(self, idx, n):
        return [ttnn.reshape(ttnn.slice(idx, (j, 0), (j + 1, n)), (n,)) for j in range(n)]

    # -- bodies ----------------------------------------------------------------------------------------------------
    def pass_body(self):
        n, bs, scale = self.rows, self.bs, self.HD**-0.5
        c = self._ctx(self.c_taps[n])
        c_rows, b_rows = (None, None) if _KVROWS else (self._idx_rows(self.c_idx, n), self._idx_rows(self.b_idx, bs))
        x = self.d_x
        for L in self.layers:
            self.L_cur = L
            kc, vc = self._proj(c, L, n, "kv")
            kc = self._rope_u(self._rms(self._users(kc, n), L["k_norm"]), self.c_cos, self.c_sin)
            vc = self._users(vc, n)
            self._write_rows(L, kc, vc, n, c_rows, self.c_idx)
            h = self._rms(x, L["in_norm"])
            q, kb, vb = self._proj(h, L, bs, "qkv")
            q = self._rope_u(self._rms(self._users(q, bs), L["q_norm"]), self.d_cos, self.d_sin)
            kb = self._rope_u(self._rms(self._users(kb, bs), L["k_norm"]), self.d_cos, self.d_sin)
            vb = self._users(vb, bs)
            self._write_rows(L, kb, vb, bs, b_rows, self.b_idx)
            o = ttnn.transformer.paged_scaled_dot_product_attention_decode(
                q,
                L["K"],
                L["V"],
                page_table_tensor=self.pt[bs],
                cur_pos_tensor=self.b_pos,
                scale=scale,
                program_config=self.sdpa_cfg,
                memory_config=DRAM,
            )  # [1, bs, NH, HD]
            o = ttnn.reshape(o, (1, 1, bs, self.nh * self.HD))
            x = ttnn.add(x, self._reduce(self._mm(o, L["o"])))
            h2 = self._rms(x, L["post_norm"])
            x = ttnn.add(x, self._reduce(self._mm(self._mlp(h2), L["down"])))
        h = self._rms(x, self.norm)
        logits = ttnn.linear(h, self.lm_head, memory_config=DRAM)
        return argmax_max(logits)

    def fill_prompt(self, feats, start=0):
        """Eager: context features [n, n_taps * H] (host) of the tokens at positions start .. start + n - 1 -> every
        layer's cache from physical slot 0 (RoPE at the true positions; base = start). Also the sliding-window refill:
        the caller passes the last window of features when the context outgrows the cache (need_refill)."""
        P = feats.shape[0]
        if P + self.bs > self.cmax:
            raise ValueError(f"context window {P} + block {self.bs} exceeds the drafter cache {self.cmax}")
        S = -(-P // self.block) * self.block
        t = torch.zeros(1, 1, S, feats.shape[-1])
        t[0, 0, :P] = feats
        taps = ttnn.from_torch(
            t.to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
            memory_config=DRAM,
        )
        c = self._ctx(taps)
        cos, sin = self._cos_sin(list(range(start, start + S)))  # [1, 1, S, HD]
        for L in self.layers:
            k, v = self._proj(c, L, S, "kv")
            k = self._rope(self._rms(self._heads_n(k, S), L["k_norm"]), cos, sin)  # [1, nh, S, HD]
            v = self._heads_n(v, S)
            ttnn.experimental.paged_fill_cache(L["K"], k, self.pt[1], batch_idx=0)
            ttnn.experimental.paged_fill_cache(L["V"], v, self.pt[1], batch_idx=0)
            for x in (k, v):
                ttnn.deallocate(x)
        for x in (taps, c, cos, sin):
            ttnn.deallocate(x)
        self.base, self.ctx_len = start, start + P

    def need_refill(self, margin=8):
        """True when the next pass (up to rows accepted rows + the block) would run past the cache."""
        return self.ctx_len - self.base + self.rows + self.bs + margin > self.cmax

    # -- host side -------------------------------------------------------------------------------------------------
    def stage_pass(self, accepted, token):
        """accepted: the positions of the last verify's accepted rows (rows 0 .. a of c_taps; [] for none);
        token: the target's next token, at position ctx_len after those rows."""
        n, bs = self.rows, self.bs
        if accepted and accepted[0] != self.ctx_len:
            raise ValueError(f"context positions {accepted[0]}.. do not continue the cache (length {self.ctx_len})")
        self.ctx_len += len(accepted)
        p = self.ctx_len
        b = self.base  # cache slot = position - base (RoPE uses the true positions)
        if p - b + bs > self.cmax:
            raise ValueError(f"drafter window {p - b} + block {bs} exceeds the cache {self.cmax}: refill first")
        ci = torch.full((n, n), -1)
        cpos = list(accepted) + [0] * (n - len(accepted))
        for j, q in enumerate(accepted):
            ci[j, j] = q - b
        self._put_i32(self.c_idx, ci)
        cos, sin = self.rope(torch.zeros(1, 1, dtype=torch.float32), torch.tensor(cpos).reshape(1, -1))
        self._put(self.c_cos, cos.reshape(1, n, 1, self.HD))
        self._put(self.c_sin, sin.reshape(1, n, 1, self.HD))
        ids = torch.tensor([token] + [self.mask_id] * (bs - 1))
        self._put(self.d_x, self.embed_host[ids].reshape(1, 1, bs, self.H))
        cos, sin = self.rope(torch.zeros(1, 1, dtype=torch.float32), torch.arange(p, p + bs).reshape(1, -1))
        self._put(self.d_cos, cos.reshape(1, bs, 1, self.HD))
        self._put(self.d_sin, sin.reshape(1, bs, 1, self.HD))
        bi = torch.full((bs, bs), -1)
        for j in range(bs):
            bi[j, j] = p - b + j
        self._put_i32(self.b_idx, bi)
        self._put_i32(self.b_pos, torch.full((bs,), p - b + bs - 1))

    def reset(self):
        self.ctx_len = self.base = 0
