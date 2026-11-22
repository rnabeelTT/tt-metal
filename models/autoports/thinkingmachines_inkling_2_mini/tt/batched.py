# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Batched (multi-user) decode for Inkling-2-Mini: B slots, one token per slot per step, every per-user state on device.

  * KV: tt/attention.AttnCache with `slots` = B (paged pool / per-slot rings, per-slot page-table rows); one
    paged_update_cache writes all B users' K / V, the SDPA kernel runs B items of 1 row with per-item page tables.
  * Short convs: per-slot states as rows of fp32 row-major [1, 1, B, C] tensors s0, s1, s2 (x_{t-3}, x_{t-2}, x_{t-1}),
    advanced in place (ttnn.copy into the persistent tensors: trace-safe). After a slot's prefill (which runs on the
    application's single-sequence lane convs) `load_slot` copies the lane state into row b.
  * Positions: an int32 [B] device tensor (-1 = idle slot: the SDPA skips it, the KV write goes to a scratch slot
    position); tau (log scaling of the global layers) computed on device from it.

Everything a step does is device ops on persistent tensors, so `step` can be captured as a trace.
"""

from __future__ import annotations

import math

import torch

import ttnn

from .dmm import dmm
from models.autoports.thinkingmachines_inkling_2_mini.tt.attention import _l2clip, _rms_unweighted, proj
from models.autoports.thinkingmachines_inkling_2_mini.tt.attn.inkling_sdpa import inkling_sdpa_decode
from models.autoports.thinkingmachines_inkling_2_mini.tt.ccl import all_reduce
from models.autoports.thinkingmachines_inkling_2_mini.tt.sconv import K

DRAM = ttnn.DRAM_MEMORY_CONFIG
SDPA_K_CHUNK = 256
# I2M_GLUE_L1 (default 1): the decode step's activations live in L1 (interleaved): the glue ops (norms, mixes, gates,
# residual adds) inherit their input's memory config, so the chain stays out of DRAM
GLUE_L1 = __import__("os").environ.get("I2M_GLUE_L1", "1") == "1"
# I2M_MIX_CACHE (default 1): a block's mix logits against every later query that reads it, in one matmul when the block
# is made (MixPlan); each mix then computes only its current source's logits and the mix kernel reads every source and
# logit tile in place (no stacked copy): per mix ~1 small matmul + 1 program instead of concat + an n-row matmul + 1
MIX_CACHE = __import__("os").environ.get("I2M_MIX_CACHE", "1") == "1"


class MixPlan:
    """Which blocks each decode mix reads (the step's block bookkeeping, walked once), and per block the concatenated
    query columns of all its readers (Qcat) with each reader's offset. Shared by every decode width of a model."""

    def __init__(self, model):
        m, cfg, ops = model, model.cfg, model.ops
        if ops.mixk is None:
            from .mixw import MixKernel
            ops.mixk = MixKernel(m.mesh, m.H)
        srcs, blocks, pre = {}, [], None
        for a, (li, p) in enumerate(m.sched):
            L = m.layers[li]
            if li == cfg.loop_start:
                if p == 0:
                    pre = list(blocks)
                else:
                    blocks = list(pre)
            if L.nq:
                srcs[("attn", a)] = list(blocks)
            srcs[("mlp", a)] = blocks + [a] if L.boundary else list(blocks)
            if L.boundary:
                blocks = blocks + [a]
        srcs[("head", -1)] = list(blocks)
        self.srcs = srcs
        readers = {}
        for key, bl in srcs.items():
            for b in bl:
                readers.setdefault(b, []).append(key)
        self.qcat, self.off = {}, {}
        for b, keys in readers.items():
            qs, off = [], 0
            for key in keys:
                Q, nq = self.query(m, key)
                self.off[(b, key)] = off
                qs.append(ops.qrep[id(Q)])
                off += nq
            self.qcat[b] = ttnn.concat(qs, dim=3) if len(qs) > 1 else qs[0]

    @staticmethod
    def query(m, key):
        kind, a = key
        if kind == "head":
            return m.final_Q, 1
        L = m.layers[m.sched[a][0]]
        return (L.attn_Q, L.nq) if kind == "attn" else (L.mlp_Q, 1)
# I2M_BCONV_KERNEL (default 1): the static short convs' decode step as one program (tt/bconv.py)
BCONV_KERNEL = __import__("os").environ.get("I2M_BCONV_KERNEL", "1") == "1"
# I2M_APREP (default 1): the decode attention prep (clips / rms / tau / sink / KV layout) as one program (tt/aprep.py)
APREP = __import__("os").environ.get("I2M_APREP", "1") == "1"
APREP_OLD = __import__("os").environ.get("I2M_APREP_OLD", "")  # (debug) letters of q r k v s from the op path
APREP_DRAM = __import__("os").environ.get("I2M_APREP_DRAM", "0") == "1"  # (debug) SDPA inputs moved to DRAM
L1 = ttnn.L1_MEMORY_CONFIG


def _f32(t):
    return ttnn.typecast(t, ttnn.float32) if t.dtype != ttnn.float32 else t


def _bf(t):
    return ttnn.typecast(t, ttnn.bfloat16) if t.dtype != ttnn.bfloat16 else t


class BatchedConv:
    """Static short conv (y = x + sum_j w_j x_{t-3+j}) over B slots, one row each. taps: [C_full, K] (the full
    checkpoint weight); mapper shards the channel dim like the activations (None: replicated). States and taps are fp32
    TILE tensors (no row-major round trips per step)."""

    def __init__(self, mesh, taps: torch.Tensor, mapper, c_local: int, B: int):
        self.mesh, self.B, self.C = mesh, B, c_local
        up = lambda t: ttnn.from_torch(t.contiguous(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,  # noqa: E731
                                       device=mesh, mesh_mapper=mapper)
        self.w = [up(taps[:, j].float().reshape(1, 1, 1, -1)) for j in range(K - 1)]
        self.w_cur = up((taps[:, K - 1].float() + 1.0).reshape(1, 1, 1, -1))  # the current tap + the residual
        z = torch.zeros(1, 1, B, taps.shape[0])
        self.s = [up(z) for _ in range(K - 1)]  # s[0] = x_{t-3}, s[1] = x_{t-2}, s[2] = x_{t-1}
        self.kernel = None
        if BCONV_KERNEL:
            from models.autoports.thinkingmachines_inkling_2_mini.tt.bconv import BConvKernel

            self.kernel = BConvKernel(mesh, taps, mapper, c_local)

    def step(self, x):
        """x [1, 1, B, C] tile (any float) -> y [1, 1, B, C] bf16 tile; states advance by one token (in place)."""
        if self.kernel is not None:
            return self.kernel.step(x, self.s)
        xf = _f32(x)
        y = ttnn.multiply(xf, self.w_cur)
        for j in range(K - 1):
            y = ttnn.add(y, ttnn.multiply(self.s[j], self.w[j]))
        for j in range(K - 2):
            ttnn.copy(self.s[j + 1], self.s[j])
        ttnn.copy(xf, self.s[K - 2])
        return _bf(y)

    def load_slot(self, b: int, lane_state):
        """Row b of the states := the lane's [1, 1, K-1, C] row-major fp32 state (after a slot's prefill)."""
        lt = ttnn.to_layout(lane_state, ttnn.TILE_LAYOUT)
        for j in range(K - 1):
            _set_row(self.s[j], b, ttnn.slice(lt, [0, 0, j, 0], [1, 1, j + 1, self.C]), self.B, self.C)


def _set_row(t, b, row, B, C):
    """In place: row b of the [1, 1, B, C] tensor t := row [1, 1, 1, C] (same layout)."""
    parts = []
    if b > 0:
        parts.append(ttnn.slice(t, [0, 0, 0, 0], [1, 1, b, C]))
    parts.append(row)
    if b < B - 1:
        parts.append(ttnn.slice(t, [0, 0, b + 1, 0], [1, 1, B, C]))
    ttnn.copy(ttnn.concat(parts, dim=2) if len(parts) > 1 else parts[0], t)


class BatchedDynConv:
    """Dynamic short conv of the global layers' K / V over B slots (see sconv.DynShortConv; the 1/3 is folded into the
    projection weights): y = x + sum_L (static[L] + dyn(h)[head, L]) x_{t-L}. Reuses the lane conv's device weights;
    fp32 TILE states."""

    def __init__(self, lane, B: int):
        self.lane, self.B, self.C, self.mesh = lane, B, lane.C, lane.mesh
        N = self.mesh.get_num_devices()
        z = torch.zeros(1, 1, B, lane.C * N)
        mk = lambda: ttnn.from_torch(z, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=self.mesh,  # noqa: E731
                                     mesh_mapper=ttnn.ShardTensorToMesh(self.mesh, dim=3) if N > 1 else None)
        self.s = [mk() for _ in range(K - 1)]
        self.static = [ttnn.to_layout(t, ttnn.TILE_LAYOUT) for t in lane.static]
        self.kernel = None
        if BCONV_KERNEL:  # the one-program conv step with per-row taps (any decode width: one tile row of states)
            from models.autoports.thinkingmachines_inkling_2_mini.tt.bconv import BConvKernel

            self.kernel = BConvKernel(self.mesh, torch.zeros(lane.C * N, K), ttnn.ShardTensorToMesh(self.mesh, dim=-1)
                                      if N > 1 else None, lane.C)

    def step(self, x, h):
        lane = self.lane
        dyn = dmm(_bf(h), lane.wproj, lane.ckc, dtype=ttnn.bfloat16)  # [1,1,B,nh*K]
        if self.kernel is not None:
            # y = x + sum_L tap_L x_{t-L} = x (tap_0 + 1) + s2 tap_1 + s1 tap_2 + s0 tap_3   (s0 = x_{t-3})
            tap = [ttnn.add(dmm(dyn, lane.expand[L], lane.ckc, dtype=ttnn.float32), self.static[L]) for L in range(K)]
            return self.kernel.step(x, self.s, taps=(tap[3], tap[2], tap[1], ttnn.add(tap[0], 1.0)))
        xf = _f32(x)
        hist = [xf, self.s[2], self.s[1], self.s[0]]  # x_{t-L} for L = 0..3
        y = xf
        for L in range(K):
            wl = ttnn.add(dmm(dyn, lane.expand[L], lane.ckc, dtype=ttnn.float32),
                          self.static[L])
            y = ttnn.add(y, ttnn.multiply(hist[L], wl))
        ttnn.copy(self.s[1], self.s[0])
        ttnn.copy(self.s[2], self.s[1])
        ttnn.copy(xf, self.s[2])
        return _bf(y)

    def load_slot(self, b: int, lane_state):
        lt = ttnn.to_layout(lane_state, ttnn.TILE_LAYOUT)
        for j in range(K - 1):
            _set_row(self.s[j], b, ttnn.slice(lt, [0, 0, j, 0], [1, 1, j + 1, self.C]), self.B, self.C)


class BatchedApp:
    """Batched decode state of one application: the k / v / attention-output / MLP-output convs over B slots."""

    def __init__(self, app, layer, B: int, shared: "BatchedApp" = None):
        """shared: the full-width BatchedApp whose conv states this narrower decode width uses (rows 0 .. B-1 of the
        same one-tile-row state tensors)."""
        c, mesh = app.cache, layer.ops.mesh
        N = mesh.get_num_devices()
        w = layer.attn
        kv_map = ttnn.ShardTensorToMesh(mesh, dim=-1) if N > 1 else None
        rep = ttnn.ReplicateTensorToMesh(mesh) if N > 1 else None
        # this width's page tables: rows 0 .. B-1 of the cache's slot tables (slot = row)
        if c.pages is not None:  # global layers: the shared tables (vLLM's block ids when paged)
            self.read_table = self.write_table = c.pages.table(B)
        elif shared is None:
            self.read_table, self.write_table = c.read_table, c.write_table
        else:
            self.read_table = c._up_i32(c.read_host[:B])
            self.write_table = c._up_i32(c.write_host[:B])
        if shared is not None:
            self.k, self.v, self.attn, self.mlp = shared.k, shared.v, shared.attn, shared.mlp
        elif w.sliding:
            self.k = BatchedConv(mesh, w.k_taps, kv_map, w.NKV * w.HD, B)
            self.v = BatchedConv(mesh, w.v_taps, kv_map, w.NKV * w.HD, B)
        else:
            self.k = BatchedDynConv(c.k_conv, B)
            self.v = BatchedDynConv(c.v_conv, B)
        if shared is None:
            H = layer.attn_conv_w.shape[0]
            self.attn = BatchedConv(mesh, layer.attn_conv_w, rep, H, B)
            self.mlp = BatchedConv(mesh, layer.mlp_conv_w, rep, H, B)
        self.aprep = None
        # KV write: [1, B, NKV, HD] padded to 32 head rows, height-sharded one user per core
        grid = mesh.compute_with_storage_grid_size()
        self.kv_shard = ttnn.create_sharded_memory_config(
            shape=(32, w.HD), core_grid=ttnn.num_cores_to_corerangeset(B, grid, row_wise=True),
            strategy=ttnn.ShardStrategy.HEIGHT, orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True)

    def load_slot(self, b, app, lane=0):
        """After slot b's prefill on prefill lane `lane`: copy the lane's conv states into row b."""
        k, v, at, ml = app.lanes[lane]
        self.k.load_slot(b, k.st)
        self.v.load_slot(b, v.st)
        self.attn.load_slot(b, at.st)
        self.mlp.load_slot(b, ml.st)


def attention_batched(w, c, ba: BatchedApp, xq, xk, xv, pos, pos_wr, pos_sdpa, tau, tt_ccl=None):
    """B users, one row each. pos [B] int32 (logical positions; global-layer writes), pos_wr [B] int32 (sliding-layer
    write slots: pos mod ring period), pos_sdpa [B, 1] int32, tau [1, B, 1, 1] fp32."""
    NH, NKV, HD, DR, N = w.NH, w.NKV, w.HD, w.DR, w.N
    B = int(xq.shape[2])
    ckc = w.ckc
    pq = proj(w, "q", xq, ttnn.float32)
    pk = proj(w, "k", xk)
    pv = proj(w, "v", xv)
    if w.dram_sharded:
        pk = ttnn.slice(pk, (0, 0, 0, 0), (1, 1, B, w.kd))
        pv = ttnn.slice(pv, (0, 0, 0, 0), (1, 1, B, w.kd))
    if APREP and B <= 32 and not w.dram_sharded:
        if ba.aprep is None:
            from models.autoports.thinkingmachines_inkling_2_mini.tt.aprep import AttnPrep

            ba.aprep = AttnPrep(w, B, ba.kv_shard)
        if w.sliding:
            k, v = ba.k.step(pk), ba.v.step(pv)
        else:
            k, v = ba.k.step(pk, xk), ba.v.step(pv, xv)
        q, rq, kp, vp, sk = ba.aprep(pq, k, v, tau)
        if APREP_DRAM:
            q, rq, sk = (ttnn.to_memory_config(t, DRAM) for t in (q, rq, sk))
        if APREP_OLD:  # (debug) take these from the op path
            oq, orq, ok, ov, osk = _old_prep(w, pq, k, v, tau, B)
            q = oq if "q" in APREP_OLD else q
            rq = orq if "r" in APREP_OLD else rq
            sk = osk if "s" in APREP_OLD else sk
            if "k" in APREP_OLD:
                kp = ttnn.to_memory_config(ttnn.pad(ok, [1, B, 32, w.HD], [0, 0, 0, 0], 0.0), ba.kv_shard)
            if "v" in APREP_OLD:
                vp = ttnn.to_memory_config(ttnn.pad(ov, [1, B, 32, w.HD], [0, 0, 0, 0], 0.0), ba.kv_shard)
        wpos = pos_wr if w.sliding else pos
        ttnn.experimental.paged_update_cache(c.k_cache, kp, update_idxs_tensor=wpos, page_table=ba.write_table)
        ttnn.experimental.paged_update_cache(c.v_cache, vp, update_idxs_tensor=wpos, page_table=ba.write_table)
        g0 = w.qd + w.rd + w.sd
        g = ttnn.slice(pq, (0, 0, 0, g0), (1, 1, B, g0 + w.gd))
        return _attn_out(w, c, ba, q, rq, sk, g, pos_sdpa, B, tt_ccl)
    o0 = 0
    q = ttnn.slice(pq, (0, 0, 0, o0), (1, 1, B, o0 + w.qd)); o0 += w.qd
    r = ttnn.slice(pq, (0, 0, 0, o0), (1, 1, B, o0 + w.rd)); o0 += w.rd
    s = ttnn.slice(pq, (0, 0, 0, o0), (1, 1, B, o0 + NH)); o0 += w.sd
    g = ttnn.slice(pq, (0, 0, 0, o0), (1, 1, B, o0 + w.gd))
    if w.sliding:
        k, v = ba.k.step(pk), ba.v.step(pv)
    else:
        k, v = ba.k.step(pk, xk), ba.v.step(pv, xv)
    q = _l2clip(_f32(ttnn.reshape(q, (1, B, NH, HD))), w.qk_clip, w.q_scale)
    r = _l2clip(_f32(ttnn.reshape(r, (1, B, NH, DR))), w.rel_clip, w.r_scale)
    k = _l2clip(_f32(ttnn.reshape(k, (1, B, NKV, HD))), w.qk_clip)
    v = _rms_unweighted(_f32(ttnn.reshape(v, (1, B, NKV, HD))), w.eps)
    if not w.sliding:
        q = ttnn.multiply(q, tau)
        r = ttnn.multiply(r, tau)
    q, k, v = _bf(q), _bf(k), _bf(v)
    wpos = pos_wr if w.sliding else pos
    for cache, t in ((c.k_cache, k), (c.v_cache, v)):
        tp = ttnn.pad(t, [1, B, 32, HD], [0, 0, 0, 0], 0.0)
        tsh = ttnn.to_memory_config(tp, ba.kv_shard)
        ttnn.experimental.paged_update_cache(cache, tsh, update_idxs_tensor=wpos, page_table=ba.write_table)
        ttnn.deallocate(tsh)
        ttnn.deallocate(tp)
    rq = ttnn.pad(_bf(r), [1, B, NH, 32], [0, 0, 0, 0], 0.0)
    sk = _bf(ttnn.multiply(_f32(ttnn.reshape(s, (1, B, NH, 1))), w.sink_bcast))
    return _attn_out(w, c, ba, q, rq, sk, g, pos_sdpa, B, tt_ccl)


def _old_prep(w, pq, k, v, tau, B):
    NH, NKV, HD, DR = w.NH, w.NKV, w.HD, w.DR
    q = ttnn.slice(pq, (0, 0, 0, 0), (1, 1, B, w.qd))
    r = ttnn.slice(pq, (0, 0, 0, w.qd), (1, 1, B, w.qd + w.rd))
    s = ttnn.slice(pq, (0, 0, 0, w.qd + w.rd), (1, 1, B, w.qd + w.rd + NH))
    q = _l2clip(_f32(ttnn.reshape(q, (1, B, NH, HD))), w.qk_clip, w.q_scale)
    r = _l2clip(_f32(ttnn.reshape(r, (1, B, NH, DR))), w.rel_clip, w.r_scale)
    k = _l2clip(_f32(ttnn.reshape(k, (1, B, NKV, HD))), w.qk_clip)
    v = _rms_unweighted(_f32(ttnn.reshape(v, (1, B, NKV, HD))), w.eps)
    if not w.sliding:
        q, r = ttnn.multiply(q, tau), ttnn.multiply(r, tau)
    rq = ttnn.pad(_bf(r), [1, B, NH, 32], [0, 0, 0, 0], 0.0)
    sk = _bf(ttnn.multiply(_f32(ttnn.reshape(s, (1, B, NH, 1))), w.sink_bcast))
    return _bf(q), rq, _bf(k), _bf(v), sk


def _attn_out(w, c, ba, q, rq, sk, g, pos_sdpa, B, tt_ccl):
    NH, HD, N = w.NH, w.HD, w.N
    o = inkling_sdpa_decode(q, c.k_cache, c.v_cache, pos_sdpa, ba.read_table, rq, w.table, w.scale, w.W, w.E,
                            sink=sk, pt_per_item=True, k_chunk_size=SDPA_K_CHUNK, out_memory_config=DRAM)
    o = ttnn.reshape(o, (1, 1, B, NH * HD))
    o = _bf(ttnn.multiply(_f32(o), ttnn.sigmoid(_f32(g))))
    out = proj(w, "o", o)
    if w.dram_sharded and int(out.shape[-1]) != w.H:
        out = ttnn.slice(out, (0, 0, 0, 0), (1, 1, B, w.H))
    return all_reduce(out, w.mesh, tt_ccl, site="mixer") if N > 1 else out


class BatchedDecoder:
    """B-slot decode on an Inkling2Model built with slots=B."""

    def __init__(self, model, B: int, shared: "BatchedDecoder" = None):
        """shared: a wider decoder on the same model whose conv states this one uses (slots 0 .. B-1 = its rows
        0 .. B-1; its rows past B are left stale and re-initialised by the next prefill into them)."""
        self.m, self.B, self.shared = model, B, shared
        assert shared is None or shared.B >= B
        self.apps = [BatchedApp(model.apps[a], model.layers[li], B, shared=shared.apps[a] if shared else None)
                     for a, (li, _) in enumerate(model.sched)]
        for L in model.layers.values():  # the fused MoE rows ops of this width, built before any trace capture
            if getattr(L.moe, "fused", False):
                L.moe.fe.build_rows([B] if B <= L.moe.fe.rows_max else [L.moe.fe.rows_max, B - L.moe.fe.rows_max])
        mesh = model.mesh
        self.rep = ttnn.ReplicateTensorToMesh(mesh) if mesh.get_num_devices() > 1 else None
        any_sl = next(model.apps[a].cache for a, (li, _) in enumerate(model.sched) if model.layers[li].sliding)
        self.ring_period = any_sl.ring_period
        cfg = model.cfg
        self.n_floor, self.alpha = cfg.log_scaling_n_floor, cfg.log_scaling_alpha
        # persistent per-step inputs (written before each step / trace replay)
        self.tokens = ttnn.from_torch(torch.zeros(1, B, dtype=torch.int32), dtype=ttnn.uint32,
                                      layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, mesh_mapper=self.rep)
        i32 = lambda t: ttnn.from_torch(t, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh,  # noqa
                                        mesh_mapper=self.rep)
        self.pos_sdpa = i32(torch.full((B, 1), -1, dtype=torch.int32))  # -1: idle slot (skipped by the SDPA)
        self.pos_w = i32(torch.zeros(B, dtype=torch.int32))  # global-layer KV write position (idle: 0)
        self.pos_wr = i32(torch.zeros(B, dtype=torch.int32))  # sliding-layer write slot (pos mod ring period)
        self.tau = ttnn.from_torch(torch.ones(1, B, 1, 1), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh,
                                   mesh_mapper=self.rep)
        self.plan = None
        if MIX_CACHE:
            if getattr(model, "_mixplan", None) is None:
                model._mixplan = MixPlan(model)
            self.plan = model._mixplan

    def load_slot(self, b, lane=0):
        if self.shared is not None:
            return self.shared.load_slot(b, lane)
        for a in range(len(self.apps)):
            self.apps[a].load_slot(b, self.m.apps[a], lane)

    def set_inputs(self, tokens: torch.Tensor, positions: torch.Tensor):
        """tokens [B] int, positions [B] int (-1 = idle slot)."""
        from models.autoports.thinkingmachines_inkling_2_mini.tt.attention import tau_of

        pos = positions.to(torch.int64)
        pw = pos.clamp(min=0)  # idle slots write at position 0 of their own slot (a slot is prefilled before use)
        host = [
            (tokens.reshape(1, -1).to(torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, self.tokens),
            (pos.reshape(-1, 1).to(torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT, self.pos_sdpa),
            (pw.to(torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT, self.pos_w),
            ((pw % self.ring_period).to(torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT, self.pos_wr),
            (tau_of(pw, self.n_floor, self.alpha).reshape(1, -1, 1, 1), ttnn.float32, ttnn.TILE_LAYOUT, self.tau),
        ]
        for t, dt, lay, dst in host:
            ttnn.copy_host_to_device_tensor(ttnn.from_torch(t.contiguous(), dtype=dt, layout=lay, mesh_mapper=self.rep),
                                            dst)

    def step(self):
        """One decode step for all B slots from the persistent inputs -> logits [1, 1, B, Vp] fp32 (softcapped)."""
        m, B, ops, cfg = self.m, self.B, self.m.ops, self.m.cfg
        e = ttnn.embedding(self.tokens, m.embed, layout=ttnn.TILE_LAYOUT)
        x = _bf(ops.rms(ttnn.reshape(e, (1, 1, B, m.H)), m.embed_norm))
        if GLUE_L1:
            x = ttnn.to_memory_config(x, L1)
        pos_w, pos_wr, pos_sdpa, tau = self.pos_w, self.pos_wr, self.pos_sdpa, self.tau
        blocks_n, ids, pre_x, pre_blocks, pre_ids, prev_src = [], [], None, None, None, None
        D = {}  # block id (its application) -> its logits against every later query (MixPlan)
        for a, (li, p) in enumerate(m.sched):
            L = m.layers[li]
            if li == cfg.loop_start:
                if p == 0:
                    pre_x, pre_blocks, pre_ids = x, list(blocks_n), list(ids)
                else:
                    x, blocks_n, ids = pre_x, list(pre_blocks), list(pre_ids)
            msgs = m.messages(li - cfg.loop_start, prev_src) if (L.loop and p > 0) else None
            x, blk = self._layer(a, x, blocks_n, msgs, pos_w, pos_wr, pos_sdpa, tau, ids, D)
            if L.boundary:  # (the block: the layer's normed input, made once in _layer)
                blocks_n, ids = blocks_n + [blk], ids + [a]
            if L.loop and li == cfg.loop_end - 1:
                prev_src = [blocks_n[len(pre_blocks) + 1], ops.rms(x)]
        if self.plan is None:
            return m.head(x, blocks_n)
        h = self._mix(("head", -1), blocks_n, ids, D, ops.rms(x), m.final_Q, 1)[0]
        return m.head_tail(h)

    def _mix(self, key, blocks_n, ids, D, cur, Q, nq):
        """mix(blocks_n + [cur], Q, nq): the blocks' logits from their MixPlan matmuls, the current source's here."""
        ops = self.m.ops
        if self.plan is None:
            return ops.mix(blocks_n + [cur], Q, nq)
        cur = ttnn.sharded_to_interleaved(cur, L1) if cur.is_sharded() else cur
        lg = dmm(cur, ops.qrep[id(Q)], ops.ckc, dtype=ttnn.float32, memory_config=L1)
        lgs = [(D[b], self.plan.off[(b, key)]) for b in ids] + [(lg, 0)]
        return ops.mixk.multi(blocks_n + [cur], lgs, nq)

    def _layer(self, a, x, blocks_n, messages, pos, pos_wr, pos_sdpa, tau, ids=(), D=None):
        m, ops = self.m, self.m.ops
        li, _ = m.sched[a]
        L, ba, c = m.layers[li], self.apps[a], m.apps[a].cache
        xn = None
        if L.nq:
            xn = ops.rms(x)
            ins = self._mix(("attn", a), blocks_n, ids, D, xn, L.attn_Q, L.nq)
        else:
            ins = [_f32(x)]
        if L.sliding:
            h = ins[0]
            if messages is not None:
                qi, ki, vi = (L.gates[n](h, mm) for n, mm in zip(("attn", "k", "v"), messages[:3]))
            else:
                qi = ki = vi = h
            bf = ttnn.bfloat16
            qn = ops.rms(qi, L.in_norm, bf)
            kn = qn if ki is qi else ops.rms(ki, L.in_norm, bf)
            vn = qn if vi is qi else ops.rms(vi, L.in_norm, bf)
        else:
            qi, ki, vi = ins if len(ins) == 3 else ins * 3
            if messages is not None:
                qi, ki, vi = (L.gates[n](hh, mm) for n, hh, mm in zip(("attn", "k", "v"), (qi, ki, vi), messages[:3]))
            bf = ttnn.bfloat16
            qn, kn, vn = ops.rms(qi, L.q_norm, bf), ops.rms(ki, L.k_norm, bf), ops.rms(vi, L.v_norm, bf)
        att = attention_batched(L.attn, c, ba, qn, kn, vn, pos, pos_wr, pos_sdpa, tau, tt_ccl=m.tt_ccl)
        att = ba.attn.step(att)
        blk = None
        if L.boundary:
            resid = _f32(att)
            blk = xn if xn is not None else ops.rms(x)
            if self.plan is not None:
                D[a] = dmm(blk, self.plan.qcat[a], ops.ckc, dtype=ttnn.float32, memory_config=L1)
            blocks_n, ids = blocks_n + [blk], list(ids) + [a]
        else:
            resid = ttnn.add(_f32(x), _f32(att))
        mi = self._mix(("mlp", a), blocks_n, ids, D, ops.rms(resid), L.mlp_Q, 1)[0]
        if messages is not None:
            mi = L.gates["mlp"](mi, messages[3])
        mo = ba.mlp.step(L.moe.forward(ops.rms(mi, L.mlp_norm, ttnn.bfloat16)))
        out = _bf(ttnn.add(resid, _f32(mo)))
        return (ttnn.to_memory_config(out, L1) if GLUE_L1 and out.memory_config() != L1 else out), blk


class TracedDecoder:
    """A captured BatchedDecoder.step: capture() compiles (one eager step), then records the step as a trace; step()
    replays it (the per-step inputs are written by set_inputs before each replay). Every program the replays share
    the device with (prefill buckets, sampling) must be compiled BEFORE capture: a program first compiled after it can
    have its kernel binaries overwritten by the replays (Inkling-Small)."""

    def __init__(self, dec: BatchedDecoder):
        self.dec, self.mesh, self.tid, self.out = dec, dec.m.mesh, None, None

    def set_inputs(self, tokens, positions):
        self.dec.set_inputs(tokens, positions)

    def capture(self, compile=True, post=None):
        """compile=False: the step's programs are already compiled (several widths: compile every width, THEN capture
        every trace; a program first compiled after a capture can be overwritten by that trace's replays).
        post(logits): more device work captured into the same trace (its result: self.extra)."""
        if compile:
            out = self.dec.step()
            if post is not None:
                post(out)
        ttnn.synchronize_device(self.mesh)
        self.tid = ttnn.begin_trace_capture(self.mesh, cq_id=0)
        self.out = self.dec.step()
        self.extra = post(self.out) if post is not None else None
        ttnn.end_trace_capture(self.mesh, self.tid, cq_id=0)
        ttnn.synchronize_device(self.mesh)

    def step(self):
        ttnn.execute_trace(self.mesh, self.tid, cq_id=0, blocking=False)
        return self.out

    def release(self):
        if self.tid is not None:
            ttnn.release_trace(self.mesh, self.tid)
            self.tid = None
