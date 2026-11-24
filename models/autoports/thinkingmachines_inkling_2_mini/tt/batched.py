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
from models.autoports.thinkingmachines_inkling_2_mini.tt.attn.inkling_sdpa import MU_MIN, inkling_sdpa_decode
from models.autoports.thinkingmachines_inkling_2_mini.tt.ccl import PackedAR, all_reduce, unpacked
from models.autoports.thinkingmachines_inkling_2_mini.tt.sconv import K

DRAM = ttnn.DRAM_MEMORY_CONFIG
# I2M_SDPA_K_CHUNK: keys per decode SDPA chunk (one core each; smaller chunks spread a short context over more cores)
SDPA_K_CHUNK = int(__import__("os").environ.get("I2M_SDPA_K_CHUNK", "256"))
# I2M_SDPA_K_CHUNK_SW: the sliding layers' chunk (their window keeps the context short: smaller chunks, more cores)
SDPA_K_CHUNK_SW = int(__import__("os").environ.get("I2M_SDPA_K_CHUNK_SW", str(SDPA_K_CHUNK)))
# I2M_GLUE_L1 (default 1): the decode step's activations live in L1 (interleaved): the glue ops (norms, mixes, gates,
# residual adds) inherit their input's memory config, so the chain stays out of DRAM
GLUE_L1 = __import__("os").environ.get("I2M_GLUE_L1", "1") == "1"
# I2M_MIX_CACHE (default 1): a block's mix logits against every later query that reads it, in one matmul when the block
# is made (MixPlan); each mix then computes only its current source's logits and the mix kernel reads every source and
# logit tile in place (no stacked copy): per mix ~1 small matmul + 1 program instead of concat + an n-row matmul + 1
# (default on 2 / 4 dies; off on one die, whose DRAM has no room for the ~184 MB of concatenated queries)
MIX_CACHE = __import__("os").environ.get("I2M_MIX_CACHE")
POST = __import__("os").environ.get("I2M_POST", "1") == "1"
PRE = __import__("os").environ.get("I2M_PRE", "1") == "1"  # the fused pre-attention section (tt/pre.py)
# I2M_GATE_NORM: the MoE-side message gate with the following norm (1) and the router logits (2, default) in one
# program (tt/gate.py GateKernel2); 0: separate ops
GATE_NORM = int(__import__("os").environ.get("I2M_GATE_NORM", "2"))
GATE_NORM_ATTN = __import__("os").environ.get("I2M_GATE_NORM_ATTN", "1") == "1"  # likewise the attention-side gates
POST_ROUTER = __import__("os").environ.get("I2M_POST_ROUTER", "1") == "1"
AR_DIRECT = __import__("os").environ.get("I2M_AR_DIRECT", "1") == "1"
GATE_MULTI = __import__("os").environ.get("I2M_GATE_MULTI", "1") == "1"
KV_FUSED = __import__("os").environ.get("I2M_KV_FUSED", "1") == "1"  # k / v cache rows written by tt/aprep.py  # the attention-side gates in one program  # post / pre read the packed all-reduce rows  # the router logits in tt/post.py
# I2M_MSG_PASS (default 1): a decode pass's message logits for all its loop layers in one matmul (tt/batched.py
# _messages; per layer only the mix program)
MSG_PASS = __import__("os").environ.get("I2M_MSG_PASS", "1") == "1"
# I2M_MSG_ALL (default 1): with MSG_PASS, all of the pass's loop layers' messages mixed in one program (one core group
# per layer, DRAM outputs) instead of one mix program per layer
MSG_ALL = __import__("os").environ.get("I2M_MSG_ALL", "1") == "1"
AGATE = __import__("os").environ.get("I2M_AGATE", "1") == "1"  # the attention output gate in one program (tt/agate.py)
# a one-slot verify pass's R rows as one SDPA item: K / V read once per kv head, not once per row (I2M_SDPA_VROWS=1:
# split over cores as one-row decode; 2: a kv head with <= I2M_SDPA_MU_MIN chunks on one core, as the per-row items)
VROWS = int(__import__("os").environ.get("I2M_SDPA_VROWS", "2"))


class Tail:
    """An application's deferred tail: its output is bf16(resid + conv(mo)) (computed by the next pre program)."""

    def __init__(self, mo, resid, conv):
        self.mo, self.resid, self.conv = mo, resid, conv


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
# I2M_KV_PAIR (default 1): a sliding layer's k and v convs in one program (tt/bconv.py bconv_pair)
KV_PAIR = __import__("os").environ.get("I2M_KV_PAIR", "1") == "1"
# I2M_DYN_FUSED (default 1): a global layer's dynamic k / v conv taps (the head lags expanded over the channels + the
# static taps) made inside the conv program, k and v in one program (was: two matmuls, an add and a conv each)
DYN_FUSED = __import__("os").environ.get("I2M_DYN_FUSED", "1") == "1"
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

    def move_row(self, src: int, dst: int):
        """Row dst of the states := row src (a verify pass's committed row back to its slot's row)."""
        for t in self.s:
            _set_row(t, dst, ttnn.slice(t, [0, 0, src, 0], [1, 1, src + 1, self.C]), self.B, self.C)

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
            # the four taps in one matmul + one add, in the kernel's order [w0 | w1 | w2 | wc] = taps 3, 2, 1, 0 (+ 1)
            self.ecat = ttnn.concat([lane.expand[L] for L in (3, 2, 1, 0)], dim=-1)
            self.scat = ttnn.concat([self.static[3], self.static[2], self.static[1], ttnn.add(self.static[0], 1.0)],
                                    dim=-1)

    def dyn_args(self, h):
        """The kernel's in-program taps (bconv.parts dyn=): (per-head lags [1, 1, B, nh K], expansion, static row)."""
        return dmm(_bf(h), self.lane.wproj, self.lane.ckc, dtype=ttnn.bfloat16), self.ecat, self.scat

    def step(self, x, h):
        lane = self.lane
        if self.kernel is not None and DYN_FUSED:
            return self.kernel.step(x, self.s, dyn=self.dyn_args(h))
        dyn = dmm(_bf(h), lane.wproj, lane.ckc, dtype=ttnn.bfloat16)  # [1,1,B,nh*K]
        if self.kernel is not None:
            # y = x + sum_L tap_L x_{t-L} = x (tap_0 + 1) + s2 tap_1 + s1 tap_2 + s0 tap_3   (s0 = x_{t-3})
            tap = ttnn.add(dmm(dyn, self.ecat, lane.ckc, dtype=ttnn.float32), self.scat)
            return self.kernel.step(x, self.s, taps=(tap,))
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

    move_row = BatchedConv.move_row

    def load_slot(self, b: int, lane_state):
        lt = ttnn.to_layout(lane_state, ttnn.TILE_LAYOUT)
        for j in range(K - 1):
            _set_row(self.s[j], b, ttnn.slice(lt, [0, 0, j, 0], [1, 1, j + 1, self.C]), self.B, self.C)


class BatchedApp:
    """Batched decode state of one application: the k / v / attention-output / MLP-output convs over B slots."""

    def __init__(self, app, layer, B: int, shared: "BatchedApp" = None, row_slots=None):
        """shared: the full-width BatchedApp whose conv states this narrower decode width uses (rows 0 .. B-1 of the
        same one-tile-row state tensors). row_slots: the slot of each row (speculative verify: R rows per slot)."""
        c, mesh = app.cache, layer.ops.mesh
        N = mesh.get_num_devices()
        w = layer.attn
        kv_map = ttnn.ShardTensorToMesh(mesh, dim=-1) if N > 1 else None
        rep = ttnn.ReplicateTensorToMesh(mesh) if N > 1 else None
        # this width's page tables: rows 0 .. B-1 of the cache's slot tables (slot = row)
        if c.pages is not None:  # global layers: the shared tables (vLLM's block ids when paged)
            self.read_table = self.write_table = c.pages.table(tuple(row_slots) if row_slots else B)
        elif row_slots:
            self.read_table = c._up_i32(c.read_host[list(row_slots)])
            self.write_table = c._up_i32(c.write_host[list(row_slots)])
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


def attention_batched(w, c, ba: BatchedApp, xq, xk, xv, pos, pos_wr, pos_sdpa, tau, tt_ccl=None, packed_out=False):
    """B users, one row each. pos [B] int32 (logical positions; global-layer writes), pos_wr [B] int32 (sliding-layer
    write slots: pos mod ring period), pos_sdpa [B, 1] int32, tau [1, B, 1, 1] fp32."""
    NH, NKV, HD, DR, N = w.NH, w.NKV, w.HD, w.DR, w.N
    B = int(xq.shape[2])
    ckc = w.ckc
    if getattr(w, "ms", None) is not None and B <= 16:  # (q | k | v streamed in one program)
        pq, pk, pv = w.ms([xq, xk, xv], w.ms_w, B)
    else:
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
        if w.sliding and KV_PAIR and ba.k.kernel is not None and ba.v.kernel is not None:  # (both convs, one program)
            from models.autoports.thinkingmachines_inkling_2_mini.tt.bconv import bconv_pair

            k, v = bconv_pair(ba.k.kernel, pk, ba.k.s, ba.v.kernel, pv, ba.v.s)
        elif KV_PAIR and DYN_FUSED and ba.k.kernel is not None and ba.v.kernel is not None:  # (global: dynamic taps)
            from models.autoports.thinkingmachines_inkling_2_mini.tt.bconv import bconv_pair

            k, v = bconv_pair(ba.k.kernel, pk, ba.k.s, ba.v.kernel, pv, ba.v.s,
                              dyn=(ba.k.dyn_args(xk), ba.v.dyn_args(xv)))
        elif w.sliding:
            k, v = ba.k.step(pk), ba.v.step(pv)
        else:
            k, v = ba.k.step(pk, xk), ba.v.step(pv, xv)
        wpos = pos_wr if w.sliding else pos
        fkv = KV_FUSED and not APREP_OLD and c.k_cache.dtype == ttnn.bfloat16
        q, rq, kp, vp, sk = ba.aprep(pq, k, v, tau, kv=(c.k_cache, c.v_cache, wpos, ba.write_table) if fkv else None)
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
        if not fkv:
            ttnn.experimental.paged_update_cache(c.k_cache, kp, update_idxs_tensor=wpos, page_table=ba.write_table)
            ttnn.experimental.paged_update_cache(c.v_cache, vp, update_idxs_tensor=wpos, page_table=ba.write_table)
        g0 = w.qd + w.rd + w.sd
        if AGATE and B <= 16 and g0 % 32 == 0:  # (the gate read from pq in the output-gate program)
            return _attn_out(w, c, ba, q, rq, sk, None, pos_sdpa, B, tt_ccl, pq=pq, g0=g0, packed_out=packed_out)
        g = ttnn.slice(pq, (0, 0, 0, g0), (1, 1, B, g0 + w.gd))
        return _attn_out(w, c, ba, q, rq, sk, g, pos_sdpa, B, tt_ccl, packed_out=packed_out)
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


_AGATE = {}


def _attn_out(w, c, ba, q, rq, sk, g, pos_sdpa, B, tt_ccl, pq=None, g0=None, packed_out=False):
    NH, HD, N = w.NH, w.HD, w.N
    one = MU_MIN if VROWS == 2 and int(pos_sdpa.shape[0]) == 1 and int(pos_sdpa.shape[-1]) > 1 else 0  # (VROWS)
    if pq is not None:  # the output gate in one program (tt/agate.py): SDPA output in L1
        o = inkling_sdpa_decode(q, c.k_cache, c.v_cache, pos_sdpa, ba.read_table, rq, w.table, w.scale, w.W, w.E,
                                sink=sk, pt_per_item=True, k_chunk_size=SDPA_K_CHUNK_SW if w.sliding else SDPA_K_CHUNK, out_memory_config=L1,
                                one_min=one)
        key = (id(w.mesh), NH, HD)
        if key not in _AGATE:
            from .agate import AttnGate

            _AGATE[key] = AttnGate(w.mesh, NH, HD)
        og = _AGATE[key](o, pq, g0)
        ttnn.deallocate(o)
        o = og
    else:
        o = inkling_sdpa_decode(q, c.k_cache, c.v_cache, pos_sdpa, ba.read_table, rq, w.table, w.scale, w.W, w.E,
                                sink=sk, pt_per_item=True, k_chunk_size=SDPA_K_CHUNK_SW if w.sliding else SDPA_K_CHUNK, out_memory_config=DRAM,
                                one_min=one)
        o = ttnn.reshape(o, (1, 1, B, NH * HD))
        o = _bf(ttnn.multiply(_f32(o), ttnn.sigmoid(_f32(g))))
    out = proj(w, "o", o)
    if w.dram_sharded and int(out.shape[-1]) != w.H:
        out = ttnn.slice(out, (0, 0, 0, 0), (1, 1, B, w.H))
    return all_reduce(out, w.mesh, tt_ccl, site="mixer", packed_out=packed_out) if N > 1 else out


class BatchedDecoder:
    """B-slot decode on an Inkling2Model built with slots=B."""

    def __init__(self, model, B: int, shared: "BatchedDecoder" = None, verify: int = 0):
        """shared: a wider decoder on the same model whose conv states this one uses (slots 0 .. B-1 = its rows
        0 .. B-1; its rows past B are left stale and re-initialised by the next prefill into them).
        verify R: speculative verify rows, B / R slots (0 .. B / R - 1) of R consecutive rows each: their convs run
        sequentially within a slot from the row of the previous pass set_accept names (tt/kernels/seqshift.hpp); a
        slot's state then lives at row R slot + accepted (load_slot: row R slot, accepted 0)."""
        self.m, self.B, self.shared, self.R = model, B, shared, verify
        assert shared is None or shared.B >= B
        # (verify rows are bitwise the sequential decode at power-of-two widths, the served ones; R = 3 is not)
        assert not verify or (shared is not None and B % verify == 0 and B <= 16 and 3 * (B // verify) <= 16
                              and B & (B - 1) == 0)
        row_slots = [r // verify for r in range(B)] if verify else None
        self.apps = [BatchedApp(model.apps[a], model.layers[li], B, shared=shared.apps[a] if shared else None,
                                row_slots=row_slots) for a, (li, _) in enumerate(model.sched)]
        self.table_key = tuple(row_slots) if verify else B
        # verify: the drafter's taps (the inputs of these applications + the final normed state) -> self.taps
        self.tap_at, self._tap, self.taps = ((5, 23, 41) if verify else ()), {}, None
        if verify:
            self.acc = ttnn.from_torch(torch.zeros(1, 32, dtype=torch.int32), dtype=ttnn.uint32,
                                       layout=ttnn.ROW_MAJOR_LAYOUT, device=model.mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                                       mesh_mapper=ttnn.ReplicateTensorToMesh(model.mesh)
                                       if model.mesh.get_num_devices() > 1 else None)
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
        self.ps_shape = (1, B) if verify and VROWS and B == verify else (B, 1)
        self.pos_sdpa = i32(torch.full(self.ps_shape, -1, dtype=torch.int32))  # -1: idle slot (skipped by the SDPA)
        self.pos_w = i32(torch.zeros(B, dtype=torch.int32))  # global-layer KV write position (idle: 0)
        self.pos_wr = i32(torch.zeros(B, dtype=torch.int32))  # sliding-layer write slot (pos mod ring period)
        self.tau = ttnn.from_torch(torch.ones(1, B, 1, 1), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh,
                                   mesh_mapper=self.rep)
        self.plan = None
        self.post = None
        self.pre = None
        if MIX_CACHE != "0":  # (one die too: its KV pool cap leaves room for the plan's query copies)
            if getattr(model, "_mixplan", None) is None:
                model._mixplan = MixPlan(model)
            self.plan = model._mixplan
            # I2M_POST (default 1): the fused post-attention section (tt/post.py) at <= 16 rows
            if POST and B <= 16:
                if getattr(model, "_post", None) is None:
                    from .post import PostKernel

                    model._post = PostKernel(mesh, model.H, model.cfg.rms_norm_eps)
                    from .gate import GateKernel2

                    rk = next((int(g.wd.shape[-1]) for L in model.layers.values() if L.loop for g in L.gates.values()), 64)
                    model.ops.gate2k = GateKernel2(model._post, rk)
                    model._nw_rows = {li: model._post.rows_weight(L.mlp_norm) for li, L in model.layers.items()}
                    model._q_rows = {li: model._post.rows_weight(L.mlp_Q, col=0) for li, L in model.layers.items()}
                self.post, self.nw_rows, self.q_rows = model._post, model._nw_rows, model._q_rows
                if PRE:
                    if getattr(model, "_pre", None) is None:
                        from .pre import PreKernel

                        model._pre = PreKernel(model._post)
                        rw = model._post.rows_weight
                        model._aq_rows = {li: [rw(L.attn_Q, col=j) for j in range(L.nq)]
                                          for li, L in model.layers.items() if L.nq}
                        nr = {}
                        for li, L in model.layers.items():
                            if L.sliding:
                                nr[(li, "in")] = rw(L.in_norm)
                            else:
                                for k in "qkv":
                                    nr[(li, k)] = rw(getattr(L, f"{k}_norm"))
                        model._norm_rows = nr
                    self.pre, self.aq_rows, self.norm_rows = model._pre, model._aq_rows, model._norm_rows

    def move_row(self, src: int, dst: int = 0):
        """Every conv state's row src -> row dst (leaving speculative decode: the committed row back to the slot's)."""
        seen = set()
        for ba in self.apps:
            for cv in (ba.k, ba.v, ba.attn, ba.mlp):
                if id(cv) not in seen:
                    seen.add(id(cv))
                    cv.move_row(src, dst)

    def set_accept(self, acc):
        """verify: slot g's state is at row R g + acc[g] of the previous pass (its last accepted row)."""
        a = torch.zeros(1, 32, dtype=torch.int32)
        a[0, : len(acc)] = torch.as_tensor(acc, dtype=torch.int32)
        ttnn.copy_host_to_device_tensor(ttnn.from_torch(a, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT,
                                                        mesh_mapper=self.rep), self.acc)

    def load_slot(self, b, lane=0):
        if self.R:  # (verify slot g = the shared decoder's row R g)
            return self.shared.load_slot(self.R * b, lane)
        if self.shared is not None:
            return self.shared.load_slot(b, lane)
        for a in range(len(self.apps)):
            self.apps[a].load_slot(b, self.m.apps[a], lane)

    def set_inputs(self, tokens: torch.Tensor, positions: torch.Tensor):
        """tokens [B] int (None: already on device), positions [B] int (-1 = idle slot)."""
        from models.autoports.thinkingmachines_inkling_2_mini.tt.attention import tau_of

        host = [] if tokens is None else [
            (tokens.reshape(1, -1).to(torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, self.tokens)]
        pos = positions.to(torch.int64)
        pw = pos.clamp(min=0)  # idle slots write at position 0 of their own slot (a slot is prefilled before use)
        host += [
            (pos.reshape(self.ps_shape).to(torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT, self.pos_sdpa),
            (pw.to(torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT, self.pos_w),
            ((pw % self.ring_period).to(torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT, self.pos_wr),
            (tau_of(pw, self.n_floor, self.alpha).reshape(1, -1, 1, 1), ttnn.float32, ttnn.TILE_LAYOUT, self.tau),
        ]
        for t, dt, lay, dst in host:
            ttnn.copy_host_to_device_tensor(ttnn.from_torch(t.contiguous(), dtype=dt, layout=lay, mesh_mapper=self.rep),
                                            dst)

    def step(self):
        """One decode step for all B slots from the persistent inputs -> logits [1, 1, B, Vp] fp32 (softcapped)."""
        if not self.R:
            return self._step()
        from . import bconv

        bconv.SEQ = (self.R, self.B // self.R, self.acc)
        try:
            return self._step()
        finally:
            bconv.SEQ = None

    def _step(self):
        m, B, ops, cfg = self.m, self.B, self.m.ops, self.m.cfg
        self._msg = None  # (the current pass's message logits: _messages)
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
                x = self.materialize(x)
                if p == 0:
                    pre_x, pre_blocks, pre_ids = x, list(blocks_n), list(ids)
                else:
                    x, blocks_n, ids = pre_x, list(pre_blocks), list(pre_ids)
            msgs = self._messages(li - cfg.loop_start, prev_src) if (L.loop and p > 0) else None
            x, blk = self._layer(a, x, blocks_n, msgs, pos_w, pos_wr, pos_sdpa, tau, ids, D)
            if L.boundary:  # (the block: the layer's normed input, made once in _layer)
                blocks_n, ids = blocks_n + [blk], ids + [a]
            if L.loop and li == cfg.loop_end - 1:
                x = self.materialize(x)
                prev_src = [blocks_n[len(pre_blocks) + 1], ops.rms(x)]
        x = self.materialize(x)
        self._msg = None
        if self.plan is None:
            return m.head(x, blocks_n)
        h = self._mix(("head", -1), blocks_n, ids, D, ops.rms(x), m.final_Q, 1)[0]
        if not self.tap_at:
            return m.head_tail(h)
        hn = []
        out = m.head_tail(h, normed=hn)
        tp = [ttnn.reshape(t, (1, 1, B, m.H)) for t in [self._tap[a] for a in self.tap_at] + hn]
        self.taps = ttnn.concat([ttnn.to_memory_config(t, ttnn.DRAM_MEMORY_CONFIG) for t in tp], dim=1)
        self._tap = {}
        return out

    @staticmethod
    def materialize(x):
        """A deferred tail -> the layer output x = bf16(resid + conv(mo)) (the ops path)."""
        if not isinstance(x, Tail):
            return x
        mo = unpacked(x.mo)
        if int(mo.shape[-2]) != int(x.resid.shape[-2]):  # (one die, one row: the MoE's shared 2-row buffer, in place)
            mo = ttnn.slice(mo, [0, 0, 0, 0], [1, 1, int(x.resid.shape[-2]), int(mo.shape[-1])])
        out = _bf(ttnn.add(x.resid, _f32(x.conv.step(mo))))
        return ttnn.to_memory_config(out, L1) if GLUE_L1 and out.memory_config() != L1 else out

    def _mix(self, key, blocks_n, ids, D, cur, Q, nq):
        """mix(blocks_n + [cur], Q, nq): the blocks' logits from their MixPlan matmuls, the current source's here."""
        ops = self.m.ops
        if self.plan is None:
            return ops.mix(blocks_n + [cur], Q, nq)
        cur = ttnn.sharded_to_interleaved(cur, L1) if cur.is_sharded() else cur
        lg = dmm(cur, ops.qrep[id(Q)], ops.ckc, dtype=ttnn.float32, memory_config=L1)
        lgs = [(D[b], self.plan.off[(b, key)]) for b in ids] + [(lg, 0)]
        return ops.mixk.multi(blocks_n + [cur], lgs, nq)

    def _messages(self, i, prev_src):
        """m.messages(i, prev_src): the pass's sources' logits against every loop layer's queries in one matmul (made
        at the pass's first loop layer), then this layer's mix."""
        m = self.m
        T = int(prev_src[0].shape[-2])
        if not MSG_PASS or T > 32 or getattr(m, "msg_qcat", None) is None or m.ops.mixk is None:
            return m.messages(i, prev_src)
        if self._msg is None or self._msg[0] is not prev_src:
            src = [ttnn.sharded_to_interleaved(s, L1) if s.is_sharded() else s for s in prev_src]
            S = ttnn.concat(src, dim=1, memory_config=L1)
            lg = dmm(S, m.msg_qcat, m.ops.ckc, dtype=ttnn.float32, memory_config=L1)
            nl = len(m.msg_Q)
            allm = m.ops.mixk.groups(S, lg, 2, 4, nl, 4 * nl) if MSG_ALL and self.post is not None else None
            self._msg = (prev_src, S, lg, allm)
        _, S, lg, allm = self._msg
        if allm is not None:
            return allm[i]
        return m.ops.mixk(S, lg, 2, 4, out_mc=L1, lg_off=4 * i, lg_st=4 * len(m.msg_Q))

    def _gate_norms(self, L, li, ins, messages):
        """The attention side's three message gates, each with its following norm, one program each (tt/gate.py
        GateKernel2) -> (qn, kn, vn) bf16; None: not available (the separate ops then). (Not kept on self: a decoder
        holding its last layer's three L1 rows kept them alive past every trace capture.)"""
        hs = ins * 3 if len(ins) == 1 else ins
        nws = [self.norm_rows[(li, "in")]] * 3 if L.sliding else [self.norm_rows[(li, k)] for k in "qkv"]
        if GATE_MULTI:  # the three gates in one program, on three core rectangles (tt/gate.py GateKernel2.multi)
            args = [L.gates[n].fused_args(hh, mm, nw) for n, hh, mm, nw in zip(("attn", "k", "v"), hs, messages[:3], nws)]
            g2 = self.m.ops.gate2k
            if all(a is not None for a in args) and len(g2.rects) >= 3:
                return tuple(x for x, _ in g2.multi(args))
        out = []
        for n, hh, mm, nw in zip(("attn", "k", "v"), hs, messages[:3], nws):
            f = L.gates[n].fused_norm(hh, mm, nw)
            if f is None:
                return None
            out.append(f[0])
        return tuple(out)

    def _layer(self, a, x, blocks_n, messages, pos, pos_wr, pos_sdpa, tau, ids=(), D=None):
        m, ops = self.m, self.m.ops
        li, _ = m.sched[a]
        L, ba, c = m.layers[li], self.apps[a], m.apps[a].cache
        xn = None
        fused = None
        if self.pre is not None and L.nq and ids:
            # tail of the previous application + input norm + block mix (+ q / k / v norms) in one program (tt/pre.py)
            if messages is None:
                nws = [self.norm_rows[(li, "in")]] if L.sliding else [self.norm_rows[(li, k)] for k in "qkv"]
            else:
                nws = None
            lgs = [(D[b], self.plan.off[(b, ("attn", a))]) for b in ids]
            if isinstance(x, Tail):
                # (the MoE output read where it is: the packed all-reduce rows, or the MoE's shared buffer, whose
                # first rows are live)
                xin = x.mo
                x, xn, outs = self.pre(xin, blocks_n, lgs, self.aq_rows[li], nws, tail=(xin, x.resid, x.conv),
                                       bound=L.boundary, rows=int(x.resid.shape[-2]))
            else:
                x, xn, outs = self.pre(x, blocks_n, lgs, self.aq_rows[li], nws, bound=L.boundary)
            fused = outs
        x = self.materialize(x)
        if a in self.tap_at:
            self._tap[a] = x
        if fused is not None:
            ins = fused
        elif L.nq:
            xn = ops.rms(x)
            ins = self._mix(("attn", a), blocks_n, ids, D, xn, L.attn_Q, L.nq)
        else:
            ins = [_f32(x)]
        if fused is not None and messages is None:  # (normed in the program)
            qn, kn, vn = (fused * 3) if L.sliding else fused
        elif messages is not None and GATE_NORM_ATTN and self.pre is not None and \
                (gn := self._gate_norms(L, li, ins, messages)) is not None:
            qn, kn, vn = gn
        elif L.sliding:
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
        post_path = self.post is not None and ba.attn.kernel is not None
        att = attention_batched(L.attn, c, ba, qn, kn, vn, pos, pos_wr, pos_sdpa, tau, tt_ccl=m.tt_ccl,
                                packed_out=post_path and AR_DIRECT)
        if not post_path:
            att = unpacked(att)
        blk = None
        if L.boundary:
            blk = xn if xn is not None else ops.rms(x)
            if self.plan is not None:
                D[a] = dmm(blk, self.plan.qcat[a], ops.ckc, dtype=ttnn.float32, memory_config=L1)
            blocks_n, ids = blocks_n + [blk], list(ids) + [a]
        if self.post is not None and ba.attn.kernel is not None:
            # conv + residual + norm + mix (+ the MoE-input norm) in one program (tt/post.py)
            if not isinstance(att, PackedAR):
                att = ttnn.sharded_to_interleaved(att, L1) if att.is_sharded() else att  # (DRAM: read in place)
            lgs = [(D[b], self.plan.off[(b, ("mlp", a))]) for b in ids]
            lg, dup = None, False
            T = int(att.shape[-2])
            if messages is None and GATE_NORM == 2 and POST_ROUTER and getattr(L.moe, "fused", False):
                # + the router logits (and the row duplicated for the one-row MoE) in the same program
                resid, o, lg = self.post(att, None if L.boundary else x, ba.attn, self.q_rows[li], blocks_n, lgs,
                                         nw=self.nw_rows[li], router=L.moe.router_hl, dup=T == 1)
                dup = lg is not None and T == 1
            else:
                resid, o = self.post(att, None if L.boundary else x, ba.attn, self.q_rows[li], blocks_n, lgs,
                                     nw=None if messages is not None else self.nw_rows[li])
            if messages is not None:
                # the gate + the MoE-input norm (+ the router logits) in one program (tt/gate.py GateKernel2)
                T = int(o.shape[-2])
                rt = L.moe.router_hl if GATE_NORM == 2 and getattr(L.moe, "fused", False) else None
                f = L.gates["mlp"].fused_norm(o, messages[3], self.nw_rows[li], rt, dup=T == 1) if GATE_NORM else None
                if f is not None:
                    (mn, lg), dup = f, T == 1
                else:
                    mn = ops.rms(L.gates["mlp"](o, messages[3]), L.mlp_norm, ttnn.bfloat16)
            else:
                mn = o
            pk = self.pre is not None and AR_DIRECT  # (the next pre program reads the packed rows)
            mo = L.moe.forward(mn, logits=lg, dup=dup, packed_out=pk) if (lg is not None or dup or pk) \
                else L.moe.forward(mn)
            if self.pre is not None:  # the tail (MoE-output conv + residual) runs in the next application's program
                return Tail(mo, resid, ba.mlp), blk
            out = _bf(ttnn.add(resid, _f32(ba.mlp.step(unpacked(mo)))))
            return (ttnn.to_memory_config(out, L1) if GLUE_L1 and out.memory_config() != L1 else out), blk
        att = ba.attn.step(att)
        resid = _f32(att) if L.boundary else ttnn.add(_f32(x), _f32(att))
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
