# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Hy4 compressed indexer (HF HYV4FlashCompressedIndexer) for the sparse attention layers.

Per token: key k_t = RMSNorm(W_tk x) (128). Every 4 tokens pool into one group key (fp32 gated softmax over the 4 slots
+ intra-group position embedding, then RMSNorm), roped (last 64 dims) at the group's first position, bf16. Per query:
q = RoPE(W_q q_resid) (16 heads x 128), w = W_hw x / sqrt(16 * 128); score(g) = sum_h w_h relu(q_h . key_g). A query at
position p keeps the 511 best complete groups g < p // 4 plus its own group (tt/attention.index_lists expands them).
Rows with p < 2048 attend densely (all of their 511-or-fewer visible groups are kept), so they skip the scoring.

The group keys live in a TILE device cache [1, 1, gcap, 128] (indexer_score_dsa's k).
- Prefill (eager, `forward`): token keys are read back and the groups pooled on the host (exactly the reference's
  fp32 torch ops); the host keeps a mirror of every group key and writes whole 32-group slabs of it.
- Decode (trace-safe, `step`): a 64-row token-key staging window and a 64-group group staging window, updated by
  exact one-hot matmuls the host supplies; the window's 16 groups are pooled on the device (fp32) every step and two
  32-group slabs are written. Scoring runs against a context bucket (kv_len fixed per trace) with the visibility of each
  row masked on the device; the top-512 groups expand to token indices on the device.
"""

from __future__ import annotations

import os

import numpy as np
import torch
import ttnn

from models.autoports.tencent_hy4_flash.tt.common import RMSNorm, ckc, mm, to_torch0, upload
from models.autoports.tencent_hy4_flash.tt.rope import DeviceRope, cos_sin, rkey, rope_torch, rpos

TOPK = 512
SCORE_ROWS = int(os.environ.get("HY4_IDX_SCORE_ROWS", "256"))


class Hy4Indexer:
    _IOTA_ALL = {}  # (mesh, gcap) -> the group-index row every layer's score masks read
    _SLOT_ATTRS = ("gkeys", "_mirror", "_mirror_stale", "_gk_pending", "n_tok", "n_groups", "_open_keys",
                   "_open_pending", "_tk_hist", "_tk_pending", "tk_hist_base", "tk_stage", "g_stage")

    def __init__(self, mesh, cfg, sd, max_ctx: int, slots: int = 1):
        self.mesh, self.cfg = mesh, cfg
        self.nh, self.d, self.r = cfg.idx_heads, cfg.idx_dim, cfg.idx_ratio
        self.rope_d = cfg.qk_rope
        self.theta = cfg.rope_theta["sparse"]
        p = "compressed_indexer."
        bf = lambda t: t.to(torch.bfloat16)  # noqa: E731
        self.ckc = ckc(mesh, fp32=True)
        self.rope = DeviceRope(mesh, self.rope_d)
        self.wq = upload(mesh, bf(sd[p + "query_projection.weight"]).T.reshape(1, 1, cfg.q_lora, self.nh * self.d))
        self.whw = upload(mesh, bf(sd[p + "head_weight_projection.weight"]).T.reshape(1, 1, cfg.hidden, self.nh))
        self.wtk = upload(mesh, bf(sd[p + "token_key_projection.weight"]).T.reshape(1, 1, cfg.hidden, self.d))
        self.tk_norm = RMSNorm(mesh, sd[p + "token_key_norm.weight"], cfg.rms_eps)
        # pooling weights (fp32, as the reference keeps them): host copies (prefill) and device copies (decode)
        self.Wv = sd[p + "key_value_projection.weight"].float()
        self.Wg = sd[p + "key_gate_projection.weight"].float()
        self.ape = sd[p + "intra_group_ape"].float()  # [4, 128]
        self.kn_w = sd[p + "key_norm.weight"].float()
        self.eps = cfg.rms_eps
        self.hw_scale = (self.nh**-0.5) * (self.d**-0.5)
        self.dWvT = upload(mesh, self.Wv.T.contiguous().reshape(1, 1, self.d, self.d), ttnn.float32)
        self.dWgT = upload(mesh, self.Wg.T.contiguous().reshape(1, 1, self.d, self.d), ttnn.float32)
        self.dape = [upload(mesh, self.ape[j].reshape(1, 1, 1, self.d), ttnn.float32) for j in range(self.r)]
        self.dkn = upload(mesh, self.kn_w.reshape(1, 1, 1, self.d), ttnn.float32)
        # slot j of window group g' = token row r * g' + j: E_j [32, 64] (rows 16..31 zero)
        E = torch.zeros(self.r, 32, 64)
        for j in range(self.r):
            for g in range(64 // self.r):
                E[j, g, self.r * g + j] = 1.0
        self.E = [upload(mesh, E[j].reshape(1, 1, 32, 64), ttnn.float32) for j in range(self.r)]
        # the fused decode pool: all r slots in one [r*32, 64] selection, [Wv^T | Wg^T] in one matrix, the gate biases
        # (ape) as a full [r*32, 2d] addend (zero over the value columns)
        self.Ecat = upload(mesh, E.reshape(1, 1, self.r * 32, 64), ttnn.float32)
        self.dWvgT = upload(mesh, torch.cat([self.Wv.T, self.Wg.T], 1).contiguous().reshape(1, 1, self.d, 2 * self.d),
                            ttnn.float32)
        bias = torch.zeros(self.r, 32, 2 * self.d)
        bias[:, :, self.d:] = self.ape[:, None, :]
        self.dvg_bias = upload(mesh, bias.reshape(1, self.r, 32, 2 * self.d), ttnn.float32)
        # group cache (TILE) + host mirror
        # + 64: the decode group window's two 32-group slabs past the context; + 32: the spare tile scoring reads
        self.gcap = -(-(max_ctx // self.r + 96) // 2048) * 2048  # a multiple of 2048: any power-of-two slab divides it
        self.gkeys = upload(mesh, torch.zeros(1, 1, self.gcap, self.d, dtype=torch.bfloat16), ttnn.bfloat16)
        self._mirror = torch.zeros(self.gcap, self.d, dtype=torch.bfloat16)
        self._mirror_stale = None  # (g0, g1): device groups newer than the host mirror (device-pooled prefill)
        self._tk_pending = self._open_pending = None
        self._off_cache = {}
        # 32 columns past any score width (<= gcap): a slice of the whole tensor would alias it, and _iota_rows
        # deallocates its slice (the full-capacity decode bucket freed self.iota)
        # (one tensor shared by the sparse layers: read-only; ~4.5 MB per layer at a 128K slot)
        ik = (id(mesh), self.gcap)
        if ik not in Hy4Indexer._IOTA_ALL:
            Hy4Indexer._IOTA_ALL[ik] = upload(mesh, torch.arange(self.gcap + 32, dtype=torch.float32).reshape(1, 1, 1, -1),
                                              ttnn.float32)
        self.iota = Hy4Indexer._IOTA_ALL[ik]
        self.offs4 = upload(mesh, torch.arange(self.r, dtype=torch.float32).reshape(1, 1, 1, self.r), ttnn.float32)
        self.tk_stage = None
        self.g_stage = None
        self.reset()
        # one sequence state per slot (group cache, host mirror, token history, staging windows); set_slot swaps them
        self.slot = 0
        self._slots = {}
        for u in range(1, max(1, int(slots))):
            self._slots[u] = {"gkeys": upload(mesh, torch.zeros(1, 1, self.gcap, self.d, dtype=torch.bfloat16),
                                              ttnn.bfloat16),
                              "_mirror": torch.zeros(self.gcap, self.d, dtype=torch.bfloat16), "fresh": True}

    def set_slot(self, u: int):
        if u == self.slot:
            return
        self._slots[self.slot] = {a: self.__dict__.get(a) for a in self._SLOT_ATTRS}
        st = self._slots.pop(u)
        fresh = st.pop("fresh", False)
        for a, v in st.items():
            self.__dict__[a] = v
        self.slot = u
        if fresh:
            self.tk_stage = self.g_stage = None
            self.reset()

    # host views of the device-pooled prefill's state, materialized on first use (decode's init_stage, a host-pooled
    # chunk after a device-pooled one): the device path itself never reads back
    @property
    def mirror(self):
        pend = self.__dict__.get("_gk_pending")
        if pend:  # prefill chunks' group keys, kept as bucket-shaped clones (no slice at prompt-dependent offsets)
            for g0, t, n in pend:
                if n > 0:
                    self._mirror[g0 : g0 + n] = to_torch0(t).reshape(-1, self.d)[:n].to(torch.bfloat16)
            self._gk_pending = []
            self._mirror_stale = None
        if self._mirror_stale is not None:
            g0, g1 = self._mirror_stale
            self._mirror_stale = None
            if g1 > g0:
                t = ttnn.slice(self.gkeys, (0, 0, g0, 0), (1, 1, g1, self.d))
                self._mirror[g0:g1] = to_torch0(t).reshape(g1 - g0, self.d).to(torch.bfloat16)
                ttnn.deallocate(t)
        return self._mirror

    @mirror.setter
    def mirror(self, v):
        self._mirror = v

    def _materialize(self, pending):
        """pending: (device tensor, n) -> its first n rows; or (device tensor, a, b) -> rows [a, b) (sliced on the host:
        a device slice at prompt-dependent offsets is a new program per prompt length)."""
        if len(pending) == 3:
            t, a, b = pending
            return to_torch0(t).reshape(-1, self.d)[a:b].to(torch.bfloat16)
        t, n = pending
        return to_torch0(t).reshape(-1, self.d)[:n].to(torch.bfloat16)

    @property
    def tk_hist(self):
        if self._tk_pending is not None:
            self._tk_hist = self._materialize(self._tk_pending)
            self._tk_pending = None
        return self._tk_hist

    @tk_hist.setter
    def tk_hist(self, v):
        self._tk_hist, self._tk_pending = v, None

    @property
    def open_keys(self):
        if self._open_pending is not None:
            self._open_keys = self._materialize(self._open_pending)
            self._open_pending = None
        return self._open_keys

    @open_keys.setter
    def open_keys(self, v):
        self._open_keys, self._open_pending = v, None

    def reset(self):
        # every request starts from a zeroed group cache: decode scores a bucket of groups past the context, and keys
        # left there by an earlier request leaked into the visible scores (HY4_IDX_ZERO_RESET)
        if os.environ.get("HY4_IDX_ZERO_RESET", "1") == "1":
            z = Hy4Indexer._ZERO.get((id(self.mesh), tuple(self.gkeys.shape)))
            if z is None:
                z = Hy4Indexer._ZERO[(id(self.mesh), tuple(self.gkeys.shape))] = upload(
                    self.mesh, torch.zeros(tuple(self.gkeys.shape), dtype=torch.bfloat16), ttnn.bfloat16)
            ttnn.copy(z, self.gkeys)
            # the host mirror too: _write_mirror uploads whole 32-group slabs, whose rows past the complete groups
            # carried the previous request's keys back into the device cache
            self._mirror.zero_()
        self._mirror_stale = None
        self._gk_pending = []
        self.n_tok, self.n_groups = 0, 0
        self.open_keys = torch.zeros(0, self.d, dtype=torch.bfloat16)
        self.tk_hist = torch.zeros(0, self.d, dtype=torch.bfloat16)  # the last token keys (staging init)
        self.tk_hist_base = 0

    def drop_device_views(self):
        """Materialize every slot's lazy host views (group keys, last token keys, the open group's keys: device clones of
        prefill chunks) and drop the clones. Called after a prefill, before any decode pass: a buffer allocated after a
        trace capture and alive at a replay is overwritten by it (tt/model.py prefill_caches_release)."""
        for st in [self.__dict__] + list(self._slots.values()):
            pend = st.get("_gk_pending")
            if pend:
                for g0, t, n in pend:
                    if n > 0:
                        st["_mirror"][g0 : g0 + n] = to_torch0(t).reshape(-1, self.d)[:n].to(torch.bfloat16)
                st["_gk_pending"] = []
                st["_mirror_stale"] = None
            for a, b in (("_tk_pending", "_tk_hist"), ("_open_pending", "_open_keys")):
                if st.get(a) is not None:
                    st[b] = self._materialize(st[a])
                    st[a] = None

    def rewind(self, n: int):
        """Prefix reuse (tt/engine_batch.py): keep this slot's first n tokens' group keys (n a multiple of 32 groups, every
        group complete; decode lanes unloaded) and drop the rest, as a continued prefill from n expects: groups >= n / r
        zeroed on the device (as reset(); the score op reads past the visible groups) and the host mirror re-read."""
        G = n // self.r
        assert n % (32 * self.r) == 0, n
        full = to_torch0(self.gkeys).reshape(-1, self.d)[: self.gcap].to(torch.bfloat16).clone()
        full[G:] = 0
        t = upload(self.mesh, full.reshape(1, 1, self.gcap, self.d), ttnn.bfloat16)  # (reset()'s copy program)
        ttnn.copy(t, self.gkeys)
        ttnn.deallocate(t)
        self._mirror = full
        self._mirror_stale = None
        self._gk_pending = []
        self.n_tok, self.n_groups = n, G
        self.open_keys = torch.zeros(0, self.d, dtype=torch.bfloat16)
        self.tk_hist = torch.zeros(0, self.d, dtype=torch.bfloat16)
        self.tk_hist_base = n

    # ------------------------------------------------------------------ host pooling (reference-exact)
    def _pool(self, tk: torch.Tensor, g0: int) -> torch.Tensor:
        """tk [G * 4, 128] bf16 token keys of groups g0.. -> roped bf16 group keys [G, 128]."""
        G = tk.shape[0] // self.r
        grouped = tk.reshape(G, self.r, self.d).float()
        values = grouped @ self.Wv.T
        gates = grouped @ self.Wg.T + self.ape
        pooled = (values * gates.softmax(dim=1)).sum(dim=1)
        var = pooled.pow(2).mean(-1, keepdim=True)
        keys = (self.kn_w * (pooled * torch.rsqrt(var + self.eps))).to(torch.bfloat16)
        pos = rpos(torch.arange(g0, g0 + G) * self.r)
        cos, sin = cos_sin(pos, self.theta, self.rope_d)
        rot = rope_torch(keys[:, self.d - self.rope_d :], cos, sin)
        return torch.cat([keys[:, : self.d - self.rope_d], rot], dim=-1)

    def _write_mirror(self, g0: int, g1: int):
        """Device group cache <- host mirror for groups [g0, g1) (whole 32-group slabs)."""
        _ = self.mirror  # sync any device-newer groups first (the slabs rewrite whole 32-group blocks)
        a, b = (g0 // 32) * 32, -(-g1 // 32) * 32
        for s0 in range(a, b, 32):  # 32-group slabs: the op wants the cache length a multiple of the written chunk
            slab = upload(self.mesh, self.mirror[s0 : s0 + 32].reshape(1, 1, 32, self.d), ttnn.bfloat16)
            ttnn.experimental.deepseek_prefill.update_padded_kv_cache(self.gkeys, slab, 0, 0, 1, s0, 0)
            ttnn.deallocate(slab)

    def _append_tokens(self, tk: torch.Tensor, p0: int):
        """Token keys of positions p0.. (in order, continuing the sequence) -> pool and mirror complete groups."""
        assert p0 == self.n_tok, (p0, self.n_tok)
        allk = torch.cat([self.open_keys, tk])
        n_complete = allk.shape[0] // self.r
        if n_complete:
            self.mirror[self.n_groups : self.n_groups + n_complete] = self._pool(allk[: n_complete * self.r],
                                                                                 self.n_groups)
            self._write_mirror(self.n_groups, self.n_groups + n_complete)
            self.n_groups += n_complete
        self.open_keys = allk[n_complete * self.r :]
        self.n_tok = p0 + tk.shape[0]
        hist = torch.cat([self.tk_hist, tk])
        keep = min(hist.shape[0], 128)
        self.tk_hist = hist[-keep:]
        self.tk_hist_base = self.n_tok - keep

    def _off(self, v):
        t = self._off_cache.get(v)
        if t is None:
            t = self._off_cache[v] = upload(self.mesh, torch.tensor([v], dtype=torch.int32).reshape(1, 1, 1, 1),
                                            ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        return t

    def _pool_prefill_device(self, tk, p0, T, Tp, mem):
        """The chunk's groups pooled on the device (decode's fp32 math) and written to the group cache: no readback.
        tk [1, 1, Tp, 128] bf16 token keys; p0 a multiple of 128 tokens, Tp of 128. Groups past the real tokens are
        junk, never visible (the rows' visibility) and rewritten once complete."""
        r, d = self.r, self.d
        G = Tp // r
        rm = ttnn.to_layout(tk, ttnn.ROW_MAJOR_LAYOUT, memory_config=mem)
        rows = ttnn.reshape(rm, (1, 1, G, r * d))  # row g = the 4 token keys of group g side by side
        K4 = ttnn.typecast(ttnn.to_layout(rows, ttnn.TILE_LAYOUT, memory_config=mem), ttnn.float32, memory_config=mem)
        ttnn.deallocate(rm)
        vals, gates = [], []
        for j in range(r):
            S = ttnn.slice(K4, (0, 0, 0, j * d), (1, 1, G, (j + 1) * d), memory_config=mem)
            vals.append(ttnn.matmul(S, self.dWvT, dtype=ttnn.float32, compute_kernel_config=self.ckc, memory_config=mem))
            g = ttnn.matmul(S, self.dWgT, dtype=ttnn.float32, compute_kernel_config=self.ckc, memory_config=mem)
            gates.append(ttnn.add(g, self.dape[j], memory_config=mem))
            ttnn.deallocate(S)
            ttnn.deallocate(g)
        ttnn.deallocate(K4)
        m = ttnn.maximum(ttnn.maximum(gates[0], gates[1], memory_config=mem),
                         ttnn.maximum(gates[2], gates[3], memory_config=mem), memory_config=mem)
        num = den = None
        for j in range(r):
            e = ttnn.exp(ttnn.subtract(gates[j], m, memory_config=mem), memory_config=mem)
            ev = ttnn.multiply(e, vals[j], memory_config=mem)
            num = ev if num is None else ttnn.add(num, ev, memory_config=mem)
            den = e if den is None else ttnn.add(den, e, memory_config=mem)
        for t_ in gates + vals + [m]:
            ttnn.deallocate(t_)
        pooled = ttnn.div(num, den, memory_config=mem)
        ttnn.deallocate(num)
        ttnn.deallocate(den)
        n = ttnn.rms_norm(pooled, epsilon=self.eps, compute_kernel_config=self.ckc, memory_config=mem)
        keys = ttnn.typecast(ttnn.multiply(n, self.dkn, memory_config=mem), ttnn.bfloat16, memory_config=mem)
        ttnn.deallocate(pooled)
        ttnn.deallocate(n)
        kp = ttnn.slice(keys, (0, 0, 0, 0), (1, 1, G, d - self.rope_d), memory_config=mem)
        kr = ttnn.slice(keys, (0, 0, 0, d - self.rope_d), (1, 1, G, d), memory_config=mem)
        ttnn.deallocate(keys)
        gkey = (p0, Tp, rkey())
        cache = self.__dict__.setdefault("_gtab", {})
        if gkey not in cache:
            for k in [k for k in cache if k[0] != 0 or k[2] is not None]:  # p0 = 0 tables stay (a traced first chunk reads them)
                for t_ in cache.pop(k):
                    ttnn.deallocate(t_)
            cache[gkey] = self.rope.tables(rpos(torch.arange(p0, p0 + Tp, r)), self.theta)
        gcos, gsin = cache[gkey]
        kr2 = self.rope(kr, gcos, gsin, mem)
        ttnn.deallocate(kr)
        gk = ttnn.concat([kp, kr2], dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(kp)
        ttnn.deallocate(kr2)
        if getattr(self, "_traced_pf", False):  # keep the token keys in a persistent buffer for after the replay
            buf = self.__dict__.setdefault("_tk_keep", {}).get(Tp)
            if buf is None:  # (allocated in the eager pass; the copy below then compiles there too)
                buf = self._tk_keep[Tp] = ttnn.clone(tk)
            ttnn.copy(tk, buf)
        g0 = p0 // r
        blk = 32  # the largest power-of-two slab dividing the offset and the group count (fewer, larger writes)
        while blk < 2048 and g0 % (2 * blk) == 0 and G % (2 * blk) == 0:
            blk *= 2
        for s0 in range(0, G, blk):
            ttnn.experimental.deepseek_prefill.update_padded_kv_cache(
                self.gkeys, gk, self._off(0), self._off(g0 + s0), 0, 1, 0, input_row_offset=s0, input_rows=blk)
        gk_keep = ttnn.clone(gk) if (os.environ.get("HY4_MIRROR_CLONES", "1") == "1"
                                     and not getattr(self, "_traced_pf", False)) else None
        ttnn.deallocate(gk)
        if getattr(self, "_traced_pf", False):
            return  # after_traced_prefill() sets the host views at the real length
        # host bookkeeping (lazy views of the device state)
        n_end = p0 + T
        n_groups = n_end // r
        if os.environ.get("HY4_MIRROR_CLONES", "1") == "1":
            self.__dict__.setdefault("_gk_pending", []).append((g0, gk_keep, n_groups - g0))
        else:
            prev = self._mirror_stale
            self._mirror_stale = (prev[0] if prev else g0, n_groups)
        self.n_tok, self.n_groups = n_end, n_groups
        n_open = n_end - n_groups * r
        keep = min(T, 128)
        full = ttnn.clone(tk)  # (its program depends on the bucket only; the views are sliced on the host when read)
        self._tk_pending = (full, T - keep, T)
        self.tk_hist_base = n_end - keep
        self._open_keys = torch.zeros(0, d, dtype=torch.bfloat16)
        self._open_pending = (full, T - n_open, T) if n_open else None

    def after_traced_prefill(self, T: int, Tb: int):
        """Host views after a traced first-chunk prefill of T real tokens (bucket Tb): token keys from the persistent
        buffer the trace wrote, the groups on the device."""
        r, d = self.r, self.d
        buf = self._tk_keep[Tb]
        self.n_tok, self.n_groups = T, T // r
        self._mirror_stale = (0, T // r)
        keep = min(T, 128)
        self._tk_pending = (ttnn.slice(buf, (0, 0, T - keep, 0), (1, 1, T, d)), keep)
        self.tk_hist_base = T - keep
        self._open_keys = torch.zeros(0, d, dtype=torch.bfloat16)
        n_open = T - (T // r) * r
        self._open_pending = (ttnn.slice(buf, (0, 0, T - n_open, 0), (1, 1, T, d)), n_open) if n_open else None

    # ------------------------------------------------------------------ scoring (shared)
    def _query(self, x, q_resid, cos, sin, mem, hw=None, iq=None):
        """-> roped index query [1, 16, Tp, 128] and head weights [1, 1, Tp, 16]. hw / iq: the decode path's fused
        head-weight columns and streamed query projection."""
        Tp = int(x.shape[-2])
        q = iq(q_resid, mem) if iq is not None else mm(q_resid, self.wq, self.ckc, mem)
        if Tp > 32 and os.environ.get("HY4_PF_NLP_HEADS", "1") == "1":
            q2 = ttnn.experimental.nlp_create_qkv_heads(q, num_heads=self.nh, num_kv_heads=0, transpose_k_heads=False,
                                                        memory_config=mem)[0]
            ttnn.deallocate(q)
            q = q2
        elif Tp == 32 and self.d % 32 == 0 and os.environ.get("HY4_IDX_VIEW", "1") == "1":
            # one tile row: [1, 1, 32, nh * d] and [1, nh, 32, d] share their tile pages (head h, tile c -> page
            # (d / 32) h + c): a zero-copy view instead of reshape + permute
            q = ttnn.experimental.view(q, (1, self.nh, Tp, self.d))
        else:
            q = ttnn.permute(ttnn.reshape(q, (1, Tp, self.nh, self.d)), (0, 2, 1, 3), memory_config=mem)
        qp = ttnn.slice(q, (0, 0, 0, 0), (1, self.nh, Tp, self.d - self.rope_d), memory_config=mem)
        qr = ttnn.slice(q, (0, 0, 0, self.d - self.rope_d), (1, self.nh, Tp, self.d), memory_config=mem)
        ttnn.deallocate(q)
        qr2 = self.rope(qr, cos, sin, mem)
        ttnn.deallocate(qr)
        q = ttnn.concat([qp, qr2], dim=-1, memory_config=mem)
        ttnn.deallocate(qp)
        ttnn.deallocate(qr2)
        if hw is None:
            hw = ttnn.matmul(x, self.whw, compute_kernel_config=self.ckc, memory_config=mem)
        w = ttnn.multiply(hw, self.hw_scale, memory_config=mem)
        ttnn.deallocate(hw)
        return q, w

    _IOTA = {}
    _ZERO = {}

    def _iota_rows(self, rows, n):
        """Group index per column, repeated over `rows` rows: [1, 1, rows, n] fp32, cached per shape (shared by the
        layers; created outside traces: the eager pass of each shape comes first)."""
        key = (id(self.mesh), rows, n)
        t = Hy4Indexer._IOTA.get(key)
        if t is None:
            if rows > 32:  # prefill shapes (kv_len grows every chunk): keep only the latest; the 32-row decode ones
                for k in [k for k in Hy4Indexer._IOTA if k[1] > 32]:  # stay (a captured trace reads them)
                    ttnn.deallocate(Hy4Indexer._IOTA.pop(k))
            one = ttnn.slice(self.iota, (0, 0, 0, 0), (1, 1, 1, n))
            t = Hy4Indexer._IOTA[key] = ttnn.repeat(one, ttnn.Shape((1, 1, rows, 1)))
            ttnn.deallocate(one)
        return t

    def _kv_bucket(self, kv_len):
        """A prefill chunk's score width: the power-of-two bucket of groups, else the full-capacity bucket (a context past
        the largest power-of-two bucket that fits, e.g. 8192+ tokens in a 12K slot, scored at its exact width and
        compiled ~9 programs per 128 tokens of context at serve time, releasing the decode traces; served run 5)."""
        b = next((b for b in (512 * 2**i for i in range(10)) if b >= kv_len and b + 32 <= self.gcap), None)
        if b is None:
            cap = ((self.gcap - 32) // 32) * 32
            b = cap if cap >= kv_len else kv_len
        return b

    def _score_rows(self, kv_len, Tp):
        """Rows per scoring block: the fp32 score block within HY4_IDX_SCORE_MB (default 64 MB)."""
        budget = int(os.environ.get("HY4_IDX_SCORE_MB", "64")) << 20
        rows = max(32, (budget // (4 * (kv_len + 32))) // 32 * 32)
        return min(rows, Tp)

    def _topk(self, q, w, kv_len, vis, mem):
        """Masked scores against the first kv_len groups -> top-512 group ids [1, 1, rows, 512] uint32 RM. (The score
        op's visible scores depend on the group keys it reads past the rows' visible range: reset() zeroes the cache.)"""
        kw = {}
        hg = os.environ.get("HY4_IDX_HG")
        if hg is not None:  # the score op's program config: heads resident per pass (0 = all), q / k chunk rows
            kw["program_config"] = ttnn.IndexerScoreProgramConfig(
                q_chunk_size=int(os.environ.get("HY4_IDX_QC", "32")), k_chunk_size=int(os.environ.get("HY4_IDX_KC", "32")),
                head_group_size=int(hg))
        sc = ttnn.experimental.indexer_score_dsa(q, self.gkeys, w, chunk_start_idx=kv_len, kv_len=kv_len + 32, **kw)
        rows = int(sc.shape[-2])
        if int(sc.shape[-1]) != kv_len + 32:  # the op's output spans the whole cache; only [0, kv_len) is written
            sc2 = ttnn.slice(sc, (0, 0, 0, 0), (1, 1, rows, kv_len + 32), memory_config=mem)
            ttnn.deallocate(sc)
            sc = sc2
        if sc.layout != ttnn.TILE_LAYOUT and os.environ.get("HY4_IDX_TILIZE", "1") == "1":
            # the score op writes row-major; ttnn.where over a row-major fp32 operand and tiled condition / iota wrote
            # wrong values (1.0 for the visible columns in isolation; in the model, rows left unwritten that kept a
            # previous request's data: request-to-request nondeterminism of the selecting chunks). Tilize first.
            st = ttnn.to_layout(sc, ttnn.TILE_LAYOUT, memory_config=mem)
            ttnn.deallocate(sc)
            sc = st
        sc32 = ttnn.typecast(sc, ttnn.float32, memory_config=mem)
        ttnn.deallocate(sc)
        iota = self._iota_rows(rows, kv_len + 32)  # cached [1, 1, rows, n] (one broadcast per op: vis is a column)
        hide = ttnn.ge(iota, vis, memory_config=mem)  # [1, 1, rows, kv_len + 32]: not visible to the row
        m = ttnn.where(hide, float("-inf"), sc32, memory_config=mem)
        for t_ in (sc32, hide):
            ttnn.deallocate(t_)
        mb = ttnn.typecast(m, ttnn.bfloat16, memory_config=mem)
        ttnn.deallocate(m)
        rm = ttnn.to_layout(mb, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(mb)
        ids = ttnn.experimental.topk_large_indices(rm, k=TOPK)
        ttnn.deallocate(rm)
        return ids

    _OFFS = {}

    def _expand_rows(self, f511, rows, mem):
        """Group ids f511 [1, 1, rows, 511] fp32 -> token rows [1, 1, rows, 2044] fp32: g r + j, j < r, group-major.
        HY4_IDX_EXPAND_RI (default 1): repeat_interleave + one offsets row (2K rows: 5.9 -> 0.8 ms; the broadcast form
        padded the [rows, 511, 4] intermediate to 32-wide tiles)."""
        n_sel = int(f511.shape[-1])
        if os.environ.get("HY4_IDX_EXPAND_RI", "1") != "1":
            e = ttnn.add(ttnn.reshape(ttnn.multiply(f511, float(self.r), memory_config=mem), (1, rows, n_sel, 1)),
                         self.offs4, memory_config=mem)
            return ttnn.reshape(e, (1, 1, rows, n_sel * self.r))
        key = (id(self.mesh), n_sel, self.r)
        offs = Hy4Indexer._OFFS.get(key)
        if offs is None:  # (created at the first, eager, use: before any capture)
            offs = Hy4Indexer._OFFS[key] = upload(
                self.mesh, torch.arange(self.r, dtype=torch.float32).repeat(n_sel).reshape(1, 1, 1, n_sel * self.r),
                ttnn.float32)
        s4 = ttnn.multiply(f511, float(self.r), memory_config=mem)
        rep = ttnn.repeat_interleave(s4, self.r, dim=3)
        ttnn.deallocate(s4)
        e = ttnn.add(rep, offs, memory_config=mem)
        ttnn.deallocate(rep)
        return e

    cache_base = 0  # the active slot's first cache row (set by Hy4Attention.set_slot)
    _BASE_T = {}

    def _base_t(self):
        key = (id(self.mesh), int(self.cache_base))
        t = Hy4Indexer._BASE_T.get(key)
        if t is None:
            t = Hy4Indexer._BASE_T[key] = upload(self.mesh, torch.full((1, 1, 1, 1), float(self.cache_base)),
                                                 ttnn.float32)
        return t

    def _expand(self, ids, rows, mem):
        """top-512 group ids [1, 1, rows, 512] uint32 RM -> token rows of the first 511 groups [1, 1, rows, 2044] uint32
        RM (device; the same math as step())."""
        it = ttnn.to_layout(ids, ttnn.TILE_LAYOUT, memory_config=mem)
        f = ttnn.typecast(it, ttnn.float32, memory_config=mem)
        ttnn.deallocate(it)
        n_sel = self.cfg.idx_group_topk - 1
        f511 = ttnn.slice(f, (0, 0, 0, 0), (1, 1, rows, n_sel), memory_config=mem)
        ttnn.deallocate(f)
        e0 = self._expand_rows(f511, rows, mem)
        ttnn.deallocate(f511)
        # + the active slot's cache base (always added, also 0: one program; slots > 0 selected slot 0's rows before)
        e = ttnn.add(e0, self._base_t(), memory_config=mem)
        ttnn.deallocate(e0)
        u = ttnn.typecast(e, ttnn.uint32, memory_config=mem)
        ttnn.deallocate(e)
        urm = ttnn.to_layout(u, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(u)
        return urm

    # ------------------------------------------------------------------ row-parallel selecting chunk (prefill)
    def forward_rows(self, x, x_own, q_resid_own, p0: int, T: int, mem, tail_own):
        """A selecting chunk (p0 >= 2048) with the selection split by rows over the dies: the group keys take every row
        of x (all rows, replicated), the query / scores / top-k / index lists only this die's rows (x_own, q_resid_own,
        the row-sharded list tails tail_own). -> index lists [1, 1, Tp / N, K] uint32 RM of this die's rows."""
        Tp = int(x.shape[-2])
        cos_own, sin_own = self.rope.tables(rpos(torch.arange(p0, p0 + Tp)), self.theta, shard_rows=True)
        vis = torch.tensor([(p0 + min(s, T - 1)) // self.r for s in range(Tp)], dtype=torch.float32)
        vis_own = upload(self.mesh, vis.reshape(1, 1, Tp, 1), ttnn.float32, mapper=("shard", 2))
        tk = self.tk_norm(ttnn.matmul(x, self.wtk, compute_kernel_config=self.ckc, memory_config=mem), mem)
        if (os.environ.get("HY4_PF_DEVICE_POOL", "1") == "1" and p0 == self.n_tok and p0 % self.r == 0
                and Tp % (32 * self.r) == 0 and (p0 // self.r) % 32 == 0
                and self._open_pending is None and self._open_keys.shape[0] == 0):
            self._pool_prefill_device(tk, p0, T, Tp, mem)
        else:
            tk_host = to_torch0(tk).reshape(Tp, self.d)[:T].to(torch.bfloat16)
            self._append_tokens(tk_host, p0)
        ttnn.deallocate(tk)
        q, w = self._query(x_own, q_resid_own, cos_own, sin_own, mem)
        Tq = int(x_own.shape[-2])
        n_vis_max = (p0 + T - 1) // self.r
        kv_len = -(-n_vis_max // 32) * 32
        if os.environ.get("HY4_PF_KV_BUCKETS", "1") == "1":
            kv_len = self._kv_bucket(kv_len)
        parts = []
        SR = self._score_rows(kv_len, Tq)
        for r0 in range(0, Tq, SR):
            r1 = min(Tq, r0 + SR)
            if r0 == 0 and r1 == Tq:
                qs, ws, vs = q, w, vis_own
            else:
                qs = ttnn.slice(q, (0, 0, r0, 0), (1, self.nh, r1, self.d), memory_config=mem)
                ws = ttnn.slice(w, (0, 0, r0, 0), (1, 1, r1, self.nh), memory_config=mem)
                vs = ttnn.slice(vis_own, (0, 0, r0, 0), (1, 1, r1, 1), memory_config=mem)
            ids = self._topk(qs, ws, kv_len, vs, mem)
            parts.append(self._expand(ids, r1 - r0, mem))
            ttnn.deallocate(ids)
            if qs is not q:
                for t_ in (qs, ws, vs):
                    ttnn.deallocate(t_)
        sel = parts[0] if len(parts) == 1 else ttnn.concat(parts, dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if len(parts) > 1:
            for t_ in parts:
                ttnn.deallocate(t_)
        idx = ttnn.concat([sel, tail_own], dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        for t_ in (sel, q, w, cos_own, sin_own, vis_own):
            ttnn.deallocate(t_)
        return idx

    # ------------------------------------------------------------------ eager (prefill)
    def forward(self, x, q_resid, p0: int, T: int, mem, tail=None):
        """Updates the group keys with these T tokens; returns selected group ids [T, 512] (numpy, sorted by score) for
        the rows that need them (rows with p < 2048 get none), or None when no row does."""
        cfg = self.cfg
        Tp = int(x.shape[-2])
        tk = self.tk_norm(ttnn.matmul(x, self.wtk, compute_kernel_config=self.ckc, memory_config=mem), mem)
        if (os.environ.get("HY4_PF_DEVICE_POOL", "1") == "1" and p0 == self.n_tok and p0 % self.r == 0
                and Tp % (32 * self.r) == 0 and (p0 // self.r) % 32 == 0
                and self._open_pending is None and self._open_keys.shape[0] == 0):
            self._pool_prefill_device(tk, p0, T, Tp, mem)
        else:
            tk_host = to_torch0(tk).reshape(Tp, self.d)[:T].to(torch.bfloat16)
            self._append_tokens(tk_host, p0)
        ttnn.deallocate(tk)
        if p0 + T - 1 < cfg.idx_candidate_topk:
            return None
        cos, sin = self.rope.tables(rpos(torch.arange(p0, p0 + Tp)), self.theta)
        q, w = self._query(x, q_resid, cos, sin, mem)
        n_vis_max = (p0 + T - 1) // self.r  # visible complete groups of the last row
        kv_len = -(-n_vis_max // 32) * 32
        if os.environ.get("HY4_PF_KV_BUCKETS", "1") == "1":
            # the score width rounded up to a power-of-two bucket of groups (invisible groups are masked by vis): a
            # chunk's programs then depend on the bucket, not on its position (one compile per bucket per server)
            kv_len = self._kv_bucket(kv_len)
        vis = torch.tensor([(p0 + min(s, T - 1)) // self.r for s in range(Tp)], dtype=torch.float32)
        vis_t = upload(self.mesh, vis.reshape(1, 1, Tp, 1), ttnn.float32)
        if tail is not None and p0 >= cfg.idx_candidate_topk:
            # every row selects: the index lists are built on the device (no top-k readback), with `tail`
            # [1, 1, Tp, K - 2044] (sink, own group, sentinels) shared by the sparse layers of the chunk
            parts = []
            SR = self._score_rows(kv_len, Tp)
            for r0 in range(0, Tp, SR):
                r1 = min(Tp, r0 + SR)
                qs = ttnn.slice(q, (0, 0, r0, 0), (1, self.nh, r1, self.d), memory_config=mem)
                ws = ttnn.slice(w, (0, 0, r0, 0), (1, 1, r1, self.nh), memory_config=mem)
                vs = ttnn.slice(vis_t, (0, 0, r0, 0), (1, 1, r1, 1), memory_config=mem)
                ids = self._topk(qs, ws, kv_len, vs, mem)
                parts.append(self._expand(ids, r1 - r0, mem))
                for t_ in (qs, ws, vs, ids):
                    ttnn.deallocate(t_)
            sel = parts[0] if len(parts) == 1 else ttnn.concat(parts, dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            if len(parts) > 1:
                for t_ in parts:
                    ttnn.deallocate(t_)
            idx = ttnn.concat([sel, tail], dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(sel)
            for t_ in (q, w, vis_t, cos, sin):
                ttnn.deallocate(t_)
            return ("device", idx)
        out = np.full((T, TOPK), 0xFFFFFFFF, dtype=np.uint32)
        for r0 in range(0, min(Tp, -(-T // 32) * 32), SCORE_ROWS):
            r1 = min(Tp, r0 + SCORE_ROWS)
            qs = ttnn.slice(q, (0, 0, r0, 0), (1, self.nh, r1, self.d), memory_config=mem)
            ws = ttnn.slice(w, (0, 0, r0, 0), (1, 1, r1, self.nh), memory_config=mem)
            vs = ttnn.slice(vis_t, (0, 0, r0, 0), (1, 1, r1, 1), memory_config=mem)
            ids = self._topk(qs, ws, kv_len, vs, mem)
            ids_h = ttnn.to_torch(ttnn.get_device_tensors(ids)[0]).reshape(r1 - r0, -1).numpy().astype(np.int64)
            n = min(T, r1) - r0
            out[r0 : r0 + n] = ids_h[:n].astype(np.uint32)
            for t_ in (qs, ws, vs, ids):
                ttnn.deallocate(t_)
        for t_ in (q, w, vis_t, cos, sin):
            ttnn.deallocate(t_)
        return out

    # ------------------------------------------------------------------ trace-safe step (decode / verify rows)
    def init_stage(self, base: int, gbase: int):
        """After a prefill: token staging = token keys of positions [base, base + 64) (those written so far), group
        staging = group keys [gbase, gbase + 64)."""
        tk = torch.zeros(64, self.d, dtype=torch.bfloat16)
        for i in range(64):
            pos = base + i
            j = pos - self.tk_hist_base
            if pos < self.n_tok and 0 <= j < self.tk_hist.shape[0]:
                tk[i] = self.tk_hist[j]
        t = upload(self.mesh, tk.reshape(1, 1, 64, self.d), ttnn.bfloat16)
        g = upload(self.mesh, self.mirror[gbase : gbase + 64].reshape(1, 1, 64, self.d).contiguous(), ttnn.bfloat16)
        if self.tk_stage is None:
            self.tk_stage, self.g_stage = t, g
        else:
            ttnn.copy(t, self.tk_stage)
            ttnn.copy(g, self.g_stage)
            ttnn.deallocate(t)
            ttnn.deallocate(g)

    def _pool_device(self, mem, gcos, gsin):
        """The 16 groups of the token staging window -> roped bf16 group keys [1, 1, 32, 128] (rows 16..31 junk)."""
        K = ttnn.typecast(self.tk_stage, ttnn.float32, memory_config=mem)
        if os.environ.get("HY4_IDX_POOL_FUSED", "1") == "1":
            return self._pool_fused(K, mem, gcos, gsin)
        vals, gates = [], []
        for j in range(self.r):
            S = ttnn.matmul(self.E[j], K, dtype=ttnn.float32, compute_kernel_config=self.ckc, memory_config=mem)
            vals.append(ttnn.matmul(S, self.dWvT, dtype=ttnn.float32, compute_kernel_config=self.ckc, memory_config=mem))
            g = ttnn.matmul(S, self.dWgT, dtype=ttnn.float32, compute_kernel_config=self.ckc, memory_config=mem)
            gates.append(ttnn.add(g, self.dape[j], memory_config=mem))
            ttnn.deallocate(S)
            ttnn.deallocate(g)
        ttnn.deallocate(K)
        m = ttnn.maximum(ttnn.maximum(gates[0], gates[1], memory_config=mem),
                         ttnn.maximum(gates[2], gates[3], memory_config=mem), memory_config=mem)
        num = den = None
        for j in range(self.r):
            e = ttnn.exp(ttnn.subtract(gates[j], m, memory_config=mem), memory_config=mem)
            ev = ttnn.multiply(e, vals[j], memory_config=mem)
            num = ev if num is None else ttnn.add(num, ev, memory_config=mem)
            den = e if den is None else ttnn.add(den, e, memory_config=mem)
        for t_ in gates + vals + [m]:
            ttnn.deallocate(t_)
        pooled = ttnn.div(num, den, memory_config=mem)
        ttnn.deallocate(num)
        ttnn.deallocate(den)
        n = ttnn.rms_norm(pooled, epsilon=self.eps, compute_kernel_config=self.ckc, memory_config=mem)
        keys = ttnn.typecast(ttnn.multiply(n, self.dkn, memory_config=mem), ttnn.bfloat16, memory_config=mem)
        ttnn.deallocate(pooled)
        ttnn.deallocate(n)
        kp = ttnn.slice(keys, (0, 0, 0, 0), (1, 1, 32, self.d - self.rope_d), memory_config=mem)
        kr = ttnn.slice(keys, (0, 0, 0, self.d - self.rope_d), (1, 1, 32, self.d), memory_config=mem)
        ttnn.deallocate(keys)
        kr2 = self.rope(kr, gcos, gsin, mem)
        ttnn.deallocate(kr)
        out = ttnn.concat([kp, kr2], dim=-1, memory_config=mem)
        ttnn.deallocate(kp)
        ttnn.deallocate(kr2)
        return out

    def _pool_fused(self, K, mem, gcos, gsin, Ecat=None, bias=None, n=32, keep_k=False):
        """_pool_device in ~14 ops instead of ~35: the r slots stacked on dim 1 ([1, r, n, .]), one value + gate
        matmul, the softmax over the r slots as dim-1 reductions. (Ecat / bias / n: the batched lanes' block forms.)"""
        d, r = self.d, self.r
        Ecat = self.Ecat if Ecat is None else Ecat
        bias = self.dvg_bias if bias is None else bias
        S = ttnn.matmul(Ecat, K, dtype=ttnn.float32, compute_kernel_config=self.ckc, memory_config=mem)
        if not keep_k:
            ttnn.deallocate(K)
        vg = ttnn.matmul(S, self.dWvgT, dtype=ttnn.float32, compute_kernel_config=self.ckc, memory_config=mem)
        ttnn.deallocate(S)
        vg = ttnn.add(ttnn.reshape(vg, (1, r, n, 2 * d)), bias, memory_config=mem)
        v = ttnn.slice(vg, (0, 0, 0, 0), (1, r, n, d), memory_config=mem)
        g = ttnn.slice(vg, (0, 0, 0, d), (1, r, n, 2 * d), memory_config=mem)
        ttnn.deallocate(vg)
        m = ttnn.max(g, dim=1, keepdim=True, memory_config=mem)
        e = ttnn.exp(ttnn.subtract(g, m, memory_config=mem), memory_config=mem)
        ttnn.deallocate(g)
        ttnn.deallocate(m)
        num = ttnn.sum(ttnn.multiply(e, v, memory_config=mem), dim=1, keepdim=True, memory_config=mem)
        den = ttnn.sum(e, dim=1, keepdim=True, memory_config=mem)
        ttnn.deallocate(e)
        ttnn.deallocate(v)
        pooled = ttnn.div(num, den, memory_config=mem)
        ttnn.deallocate(num)
        ttnn.deallocate(den)
        nrm = ttnn.rms_norm(pooled, epsilon=self.eps, compute_kernel_config=self.ckc, memory_config=mem)
        keys = ttnn.typecast(ttnn.multiply(nrm, self.dkn, memory_config=mem), ttnn.bfloat16, memory_config=mem)
        ttnn.deallocate(pooled)
        ttnn.deallocate(nrm)
        kp = ttnn.slice(keys, (0, 0, 0, 0), (1, 1, n, d - self.rope_d), memory_config=mem)
        kr = ttnn.slice(keys, (0, 0, 0, d - self.rope_d), (1, 1, n, d), memory_config=mem)
        ttnn.deallocate(keys)
        kr2 = self.rope(kr, gcos, gsin, mem)
        ttnn.deallocate(kr)
        out = ttnn.concat([kp, kr2], dim=-1, memory_config=mem)
        ttnn.deallocate(kp)
        ttnn.deallocate(kr2)
        return out

    # ------------------------------------------------------------------ batched decode lanes (tt/decode_batch)
    def lane_tensors(self, L):
        """(token window [L 64, d], group window [L 64, d], group keys [L gcap + 64, d]) of the L-lane batched decode
        (lane l: window rows 64 l .., groups l gcap ..; the last 64 groups: idle lanes' slab target), plus the pooling
        constants; created (eager) on first use."""
        lt = self.__dict__.setdefault("_lanes", {})
        if L not in lt:
            z = lambda n: ttnn.zeros((1, 1, n, self.d), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,  # noqa: E731
                                     device=self.mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            r, d = self.r, self.d
            E = torch.zeros(r * 32 * L, 64 * L)
            for j in range(r):
                for l in range(L):
                    for g in range(64 // r):
                        E[j * 32 * L + 32 * l + g, 64 * l + r * g + j] = 1.0
            bias = torch.zeros(r, 32 * L, 2 * d)
            bias[:, :, d:] = self.ape[:, None, :]
            lt[L] = {"tk": z(L * 64), "g": z(L * 64), "gk": z(L * self.gcap + 64),
                     "Ecat": upload(self.mesh, E.reshape(1, 1, r * 32 * L, 64 * L), ttnn.float32),
                     "bias": upload(self.mesh, bias.reshape(1, r, 32 * L, 2 * d), ttnn.float32)}
        return lt[L]

    def _lane_bias(self, n):
        """The pooling's gate biases (ape) for n pooled rows [1, r, n, 2d] fp32 (created eagerly on first use)."""
        c = self.__dict__.setdefault("_bias_n", {})
        if n not in c:
            bias = torch.zeros(self.r, n, 2 * self.d)
            bias[:, :, self.d:] = self.ape[:, None, :]
            c[n] = upload(self.mesh, bias.reshape(1, self.r, n, 2 * self.d), ttnn.float32)
        return c[n]

    def _iota_lane(self, L, n):
        """Group index per column over the lanes' key tensor, repeated over 32 rows: [1, 1, 32, n] fp32 (cached per
        width; created at the first, eager, pass of the width)."""
        T = self.lane_tensors(L)
        c = T.setdefault("iota", {})
        if n not in c:
            one = upload(self.mesh, torch.arange(n, dtype=torch.float32).reshape(1, 1, 1, n), ttnn.float32)
            c[n] = ttnn.repeat(one, ttnn.Shape((1, 1, 32, 1)))
            ttnn.deallocate(one)
        return c[n]

    def _region_offs(self, base, n):
        key = ("region", base, n)
        t = self._off_cache.get(key)
        if t is None:
            o = torch.zeros(1, 1, 1, -(-n // 8) * 8, dtype=torch.int32)
            o[0, 0, 0, :n] = base + 32 * torch.arange(n, dtype=torch.int32)
            t = self._off_cache[key] = upload(self.mesh, o, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        return t

    def lane_load(self, L, lane, Mk, Mp):
        """Lane `lane` <- this slot's token / group windows (tk_stage / g_stage) and group keys (gkeys)."""
        from models.autoports.tencent_hy4_flash.tt.slab_write import slab_write

        T = self.lane_tensors(L)
        for name, src in (("tk", self.tk_stage), ("g", self.g_stage)):
            a = ttnn.matmul(Mk, T[name], compute_kernel_config=self.ckc)
            b = ttnn.matmul(Mp, src, compute_kernel_config=self.ckc)
            new = ttnn.add(a, b)
            ttnn.copy(new, T[name])
            for t_ in (a, b, new):
                ttnn.deallocate(t_)
        n = self.gcap // 32
        slab_write(self.gkeys, T["gk"], self._region_offs(lane * self.gcap, n), n)

    def lane_unload(self, L, lane, Ms):
        """This slot's windows / group keys <- lane `lane`."""
        from models.autoports.tencent_hy4_flash.tt.slab_write import slab_write

        T = self.lane_tensors(L)
        for name, dst in (("tk", self.tk_stage), ("g", self.g_stage)):
            w = ttnn.matmul(Ms, T[name], compute_kernel_config=self.ckc)
            ttnn.copy(w, dst)
            ttnn.deallocate(w)
        reg = ttnn.slice(T["gk"], (0, 0, lane * self.gcap, 0), (1, 1, (lane + 1) * self.gcap, self.d))
        n = self.gcap // 32
        slab_write(reg, self.gkeys, self._region_offs(0, n), n)
        ttnn.deallocate(reg)

    def step_lanes(self, x, q_resid, si, mem, tk=None, hw=None, iq=None):
        """step() of a batched pass: every lane's token / group windows at once (block-diagonal maps), the group slabs
        to the lanes' group-key regions (one slab_write), then the rows' index lists: scores over the lanes' keys
        [0, si.width) masked per row to its lane's visible groups [lo, hi), top-512, token rows (+ the row's cache
        offset), [selection | tail] or (rows < 2048 tokens) the dense list, mixed per row in fp32 (sentinel 2^32)."""
        from models.autoports.tencent_hy4_flash.tt.slab_write import slab_write

        L = si.L
        T = self.lane_tensors(L)
        if tk is None:
            tk = ttnn.matmul(x, self.wtk, compute_kernel_config=self.ckc, memory_config=mem)
        tk_raw = tk
        tk = self.tk_norm(tk_raw, mem)
        ttnn.deallocate(tk_raw)
        a = ttnn.matmul(si.A, T["tk"], compute_kernel_config=self.ckc, memory_config=mem)
        b = ttnn.matmul(si.P, tk, compute_kernel_config=self.ckc, memory_config=mem)
        ttnn.deallocate(tk)
        new = ttnn.add(a, b, memory_config=mem)
        ttnn.deallocate(a)
        ttnn.deallocate(b)
        ttnn.copy(new, T["tk"])
        ttnn.deallocate(new)
        if getattr(si, "Esel", None) is not None:  # only the groups this pass can change (decode_batch.LANE_POOL)
            # bf16 one-hot x bf16 window, fp32 out: the exact token keys (as the fp32 Ecat x typecast window)
            n = int(si.gcos.shape[-2])
            gk = self._pool_fused(T["tk"], mem, si.gcos, si.gsin, Ecat=si.Esel, bias=self._lane_bias(n), n=n, keep_k=True)
        else:
            K = ttnn.typecast(T["tk"], ttnn.float32, memory_config=mem)
            gk = self._pool_fused(K, mem, si.gcos, si.gsin, Ecat=T["Ecat"], bias=T["bias"], n=32 * L)
        a = ttnn.matmul(si.Ag, T["g"], compute_kernel_config=self.ckc, memory_config=mem)
        b = ttnn.matmul(si.Pg, gk, compute_kernel_config=self.ckc, memory_config=mem)
        ttnn.deallocate(gk)
        new = ttnn.add(a, b, memory_config=mem)
        ttnn.deallocate(a)
        ttnn.deallocate(b)
        ttnn.copy(new, T["g"])
        ttnn.deallocate(new)
        slab_write(T["g"], T["gk"], si.lane_offs["grp"], 2 * L)
        if not getattr(si, "sel", True):  # every row in the dense regime: the host-built dense lists, no scoring
            u = ttnn.typecast(si.dense_f, ttnn.uint32, memory_config=mem)
            idx = ttnn.to_layout(u, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(u)
            return idx
        cos, sin = si.cos_sin["sparse"]
        q, w = self._query(x, q_resid, cos, sin, mem, hw=hw, iq=iq)
        W = si.width
        sc = ttnn.experimental.indexer_score_dsa(q, T["gk"], w, chunk_start_idx=W, kv_len=W + 32)
        ttnn.deallocate(q)
        ttnn.deallocate(w)
        if int(sc.shape[-1]) != W + 32:
            sc2 = ttnn.slice(sc, (0, 0, 0, 0), (1, 1, 32, W + 32), memory_config=mem)
            ttnn.deallocate(sc)
            sc = sc2
        if sc.layout != ttnn.TILE_LAYOUT:
            st = ttnn.to_layout(sc, ttnn.TILE_LAYOUT, memory_config=mem)
            ttnn.deallocate(sc)
            sc = st
        sc32 = ttnn.typecast(sc, ttnn.float32, memory_config=mem)
        ttnn.deallocate(sc)
        iota = self._iota_lane(L, W + 32)
        lo = ttnn.lt(iota, si.lo, memory_config=mem)
        hi = ttnn.ge(iota, si.hi, memory_config=mem)
        hide = ttnn.logical_or(lo, hi, memory_config=mem)
        ttnn.deallocate(lo)
        ttnn.deallocate(hi)
        m = ttnn.where(hide, float("-inf"), sc32, memory_config=mem)
        for t_ in (sc32, hide):
            ttnn.deallocate(t_)
        mb = ttnn.typecast(m, ttnn.bfloat16, memory_config=mem)
        ttnn.deallocate(m)
        rm = ttnn.to_layout(mb, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(mb)
        ids = ttnn.experimental.topk_large_indices(rm, k=TOPK)
        ttnn.deallocate(rm)
        it = ttnn.to_layout(ids, ttnn.TILE_LAYOUT, memory_config=mem)
        ttnn.deallocate(ids)
        f = ttnn.typecast(it, ttnn.float32, memory_config=mem)
        ttnn.deallocate(it)
        n_sel = self.cfg.idx_group_topk - 1
        f511 = ttnn.slice(f, (0, 0, 0, 0), (1, 1, 32, n_sel), memory_config=mem)
        ttnn.deallocate(f)
        e0 = self._expand_rows(f511, 32, mem)
        ttnn.deallocate(f511)
        e = ttnn.add(e0, si.delta, memory_config=mem)  # + the row's cache base - 4 x its lane's group base
        ttnn.deallocate(e0)
        full = ttnn.concat([e, si.tail_f], dim=-1, memory_config=mem)  # [32, K] fp32
        ttnn.deallocate(e)
        sel = ttnn.multiply(full, si.nmask, memory_config=mem)
        ttnn.deallocate(full)
        mix = ttnn.add(sel, si.dense_f, memory_config=mem)  # dense_f: the dense rows' lists (zero on the other rows)
        ttnn.deallocate(sel)
        u = ttnn.typecast(mix, ttnn.uint32, memory_config=mem)
        ttnn.deallocate(mix)
        idx = ttnn.to_layout(u, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(u)
        return idx

    def step(self, x, q_resid, si, mem, tk=None, hw=None, iq=None):
        """Token / group staging updates and slab writes; in the sparse regime the index lists of the rows
        [1, 1, 32, K] uint32 RM (device-built), else si.idx_dense. tk / hw / iq: the decode path's fused projections."""
        if tk is None:
            tk = ttnn.matmul(x, self.wtk, compute_kernel_config=self.ckc, memory_config=mem)
        tk_raw = tk
        tk = self.tk_norm(tk_raw, mem)
        ttnn.deallocate(tk_raw)
        a = ttnn.matmul(si.A, self.tk_stage, compute_kernel_config=self.ckc, memory_config=mem)
        b = ttnn.matmul(si.P, tk, compute_kernel_config=self.ckc, memory_config=mem)
        ttnn.deallocate(tk)
        new = ttnn.add(a, b, memory_config=mem)
        ttnn.deallocate(a)
        ttnn.deallocate(b)
        ttnn.copy(new, self.tk_stage)
        ttnn.deallocate(new)
        gk = self._pool_device(mem, si.gcos, si.gsin)
        a = ttnn.matmul(si.Ag, self.g_stage, compute_kernel_config=self.ckc, memory_config=mem)
        b = ttnn.matmul(si.Pg, gk, compute_kernel_config=self.ckc, memory_config=mem)
        ttnn.deallocate(gk)
        new = ttnn.add(a, b, memory_config=mem)
        ttnn.deallocate(a)
        ttnn.deallocate(b)
        ttnn.copy(new, self.g_stage)
        ttnn.deallocate(new)
        for j in range(2):
            ttnn.experimental.deepseek_prefill.update_padded_kv_cache(
                self.gkeys, self.g_stage, si.slot, si.off_grp[j], 0, 1, 0, input_row_offset=32 * j, input_rows=32)
        if not si.sparse:
            if hw is not None:
                ttnn.deallocate(hw)
            return si.idx_dense
        cos, sin = si.cos_sin["sparse"]
        q, w = self._query(x, q_resid, cos, sin, mem, hw=hw, iq=iq)
        ids = self._topk(q, w, si.bucket, si.vis, mem)
        ttnn.deallocate(q)
        ttnn.deallocate(w)
        # group ids -> token rows: the first 511 groups x 4 (+ the tail: sink, own group up to p, sentinels)
        it = ttnn.to_layout(ids, ttnn.TILE_LAYOUT, memory_config=mem)
        ttnn.deallocate(ids)
        f = ttnn.typecast(it, ttnn.float32, memory_config=mem)
        ttnn.deallocate(it)
        n_sel = self.cfg.idx_group_topk - 1
        f511 = ttnn.slice(f, (0, 0, 0, 0), (1, 1, 32, n_sel), memory_config=mem)
        ttnn.deallocate(f)
        e = self._expand_rows(f511, 32, mem)
        ttnn.deallocate(f511)
        if getattr(si, "base_f", None) is not None:  # + the active slot's cache base
            e2 = ttnn.add(e, si.base_f, memory_config=mem)
            ttnn.deallocate(e)
            e = e2
        u = ttnn.typecast(e, ttnn.uint32, memory_config=mem)
        ttnn.deallocate(e)
        urm = ttnn.to_layout(u, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(u)
        idx = ttnn.concat([urm, si.idx_tail], dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        sel_buf = getattr(self, "sel_buf", None)
        if sel_buf is not None:  # persistent (see Hy4Model.step)
            ttnn.copy(urm, sel_buf)
            ttnn.deallocate(urm)
            urm = sel_buf
        self.sel = urm  # the rows' selected token rows [1, 1, 32, 2044] (kept: the DFly2 drafter's index share)
        return idx
