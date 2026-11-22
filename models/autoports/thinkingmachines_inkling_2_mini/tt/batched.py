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
from models.autoports.thinkingmachines_inkling_2_mini.tt.attention import _l2clip, _rms_unweighted
from models.autoports.thinkingmachines_inkling_2_mini.tt.attn.inkling_sdpa import inkling_sdpa_decode
from models.autoports.thinkingmachines_inkling_2_mini.tt.ccl import all_reduce
from models.autoports.thinkingmachines_inkling_2_mini.tt.sconv import K

DRAM = ttnn.DRAM_MEMORY_CONFIG
SDPA_K_CHUNK = 256


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

    def step(self, x):
        """x [1, 1, B, C] tile (any float) -> y [1, 1, B, C] bf16 tile; states advance by one token (in place)."""
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

    def step(self, x, h):
        lane = self.lane
        dyn = ttnn.matmul(_bf(h), lane.wproj, compute_kernel_config=lane.ckc, dtype=ttnn.bfloat16)  # [1,1,B,nh*K]
        xf = _f32(x)
        hist = [xf, self.s[2], self.s[1], self.s[0]]  # x_{t-L} for L = 0..3
        y = xf
        for L in range(K):
            wl = ttnn.add(ttnn.matmul(dyn, lane.expand[L], compute_kernel_config=lane.ckc, dtype=ttnn.float32),
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

    def __init__(self, app, layer, B: int):
        c, mesh = app.cache, layer.ops.mesh
        N = mesh.get_num_devices()
        w = layer.attn
        kv_map = ttnn.ShardTensorToMesh(mesh, dim=-1) if N > 1 else None
        rep = ttnn.ReplicateTensorToMesh(mesh) if N > 1 else None
        if w.sliding:
            self.k = BatchedConv(mesh, w.k_taps, kv_map, w.NKV * w.HD, B)
            self.v = BatchedConv(mesh, w.v_taps, kv_map, w.NKV * w.HD, B)
        else:
            self.k = BatchedDynConv(c.k_conv, B)
            self.v = BatchedDynConv(c.v_conv, B)
        H = layer.attn_conv_w.shape[0]
        self.attn = BatchedConv(mesh, layer.attn_conv_w, rep, H, B)
        self.mlp = BatchedConv(mesh, layer.mlp_conv_w, rep, H, B)
        # KV write: [1, B, NKV, HD] padded to 32 head rows, height-sharded one user per core
        grid = mesh.compute_with_storage_grid_size()
        self.kv_shard = ttnn.create_sharded_memory_config(
            shape=(32, w.HD), core_grid=ttnn.num_cores_to_corerangeset(B, grid, row_wise=True),
            strategy=ttnn.ShardStrategy.HEIGHT, orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True)

    def load_slot(self, b, app):
        """After slot b's prefill on the lane: copy the lane conv states into row b."""
        c = app.cache
        self.k.load_slot(b, c.k_conv.st)
        self.v.load_slot(b, c.v_conv.st)
        self.attn.load_slot(b, app.attn_conv.st)
        self.mlp.load_slot(b, app.mlp_conv.st)


def attention_batched(w, c, ba: BatchedApp, xq, xk, xv, pos, pos_wr, pos_sdpa, tau, tt_ccl=None):
    """B users, one row each. pos [B] int32 (logical positions; global-layer writes), pos_wr [B] int32 (sliding-layer
    write slots: pos mod ring period), pos_sdpa [B, 1] int32, tau [1, B, 1, 1] fp32."""
    NH, NKV, HD, DR, N = w.NH, w.NKV, w.HD, w.DR, w.N
    B = int(xq.shape[2])
    ckc = w.ckc
    pq = ttnn.linear(xq, w.w_q, compute_kernel_config=ckc, memory_config=DRAM, dtype=ttnn.float32)
    pk = ttnn.linear(xk, w.w_k, compute_kernel_config=ckc, memory_config=DRAM)
    pv = ttnn.linear(xv, w.w_v, compute_kernel_config=ckc, memory_config=DRAM)
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
        ttnn.experimental.paged_update_cache(cache, tsh, update_idxs_tensor=wpos, page_table=c.write_table)
        ttnn.deallocate(tsh)
        ttnn.deallocate(tp)
    rq = ttnn.pad(_bf(r), [1, B, NH, 32], [0, 0, 0, 0], 0.0)
    sk = _bf(ttnn.multiply(_f32(ttnn.reshape(s, (1, B, NH, 1))), w.sink_bcast))
    o = inkling_sdpa_decode(q, c.k_cache, c.v_cache, pos_sdpa, c.read_table, rq, w.table, w.scale, w.W, w.E,
                            sink=sk, pt_per_item=True, k_chunk_size=SDPA_K_CHUNK, out_memory_config=DRAM)
    o = ttnn.reshape(o, (1, 1, B, NH * HD))
    o = _bf(ttnn.multiply(_f32(o), ttnn.sigmoid(_f32(g))))
    out = ttnn.linear(o, w.w_o, compute_kernel_config=ckc, memory_config=DRAM)
    return all_reduce(out, w.mesh, tt_ccl) if N > 1 else out


class BatchedDecoder:
    """B-slot decode on an Inkling2Model built with slots=B."""

    def __init__(self, model, B: int):
        self.m, self.B = model, B
        self.apps = [BatchedApp(model.apps[a], model.layers[li], B) for a, (li, _) in enumerate(model.sched)]
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

    def load_slot(self, b):
        for a in range(len(self.apps)):
            self.apps[a].load_slot(b, self.m.apps[a])

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
        pos_w, pos_wr, pos_sdpa, tau = self.pos_w, self.pos_wr, self.pos_sdpa, self.tau
        blocks_n, pre_x, pre_blocks, prev_src = [], None, None, None
        for a, (li, p) in enumerate(m.sched):
            L = m.layers[li]
            if li == cfg.loop_start:
                if p == 0:
                    pre_x, pre_blocks = x, list(blocks_n)
                else:
                    x, blocks_n = pre_x, list(pre_blocks)
            msgs = m.messages(li - cfg.loop_start, prev_src) if (L.loop and p > 0) else None
            xin = x
            x = self._layer(a, x, blocks_n, msgs, pos_w, pos_wr, pos_sdpa, tau)
            if L.boundary:
                blocks_n = blocks_n + [ops.rms(xin)]
            if L.loop and li == cfg.loop_end - 1:
                prev_src = [blocks_n[len(pre_blocks) + 1], ops.rms(x)]
        return m.head(x, blocks_n)

    def _layer(self, a, x, blocks_n, messages, pos, pos_wr, pos_sdpa, tau):
        m, ops = self.m, self.m.ops
        li, _ = m.sched[a]
        L, ba, c = m.layers[li], self.apps[a], m.apps[a].cache
        xn = None
        if L.nq:
            xn = ops.rms(x)
            ins = ops.mix(blocks_n + [xn], L.attn_Q, L.nq)
        else:
            ins = [_f32(x)]
        if L.sliding:
            h = ins[0]
            if messages is not None:
                qi, ki, vi = (L.gates[n](h, mm) for n, mm in zip(("attn", "k", "v"), messages[:3]))
            else:
                qi = ki = vi = h
            qn = _bf(ops.rms(qi, L.in_norm))
            kn = qn if ki is qi else _bf(ops.rms(ki, L.in_norm))
            vn = qn if vi is qi else _bf(ops.rms(vi, L.in_norm))
        else:
            qi, ki, vi = ins if len(ins) == 3 else ins * 3
            if messages is not None:
                qi, ki, vi = (L.gates[n](hh, mm) for n, hh, mm in zip(("attn", "k", "v"), (qi, ki, vi), messages[:3]))
            qn, kn, vn = _bf(ops.rms(qi, L.q_norm)), _bf(ops.rms(ki, L.k_norm)), _bf(ops.rms(vi, L.v_norm))
        att = attention_batched(L.attn, c, ba, qn, kn, vn, pos, pos_wr, pos_sdpa, tau, tt_ccl=m.tt_ccl)
        att = ba.attn.step(att)
        if L.boundary:
            resid = _f32(att)
            blocks_n = blocks_n + [xn if xn is not None else ops.rms(x)]
        else:
            resid = ttnn.add(_f32(x), _f32(att))
        mi = ops.mix(blocks_n + [ops.rms(resid)], L.mlp_Q, 1)[0]
        if messages is not None:
            mi = L.gates["mlp"](mi, messages[3])
        mo = ba.mlp.step(L.moe.forward(_bf(ops.rms(mi, L.mlp_norm))))
        return _bf(ttnn.add(resid, _f32(mo)))


class TracedDecoder:
    """A captured BatchedDecoder.step: capture() compiles (one eager step), then records the step as a trace; step()
    replays it (the per-step inputs are written by set_inputs before each replay). Every program the replays share
    the device with (prefill buckets, sampling) must be compiled BEFORE capture: a program first compiled after it can
    have its kernel binaries overwritten by the replays (Inkling-Small)."""

    def __init__(self, dec: BatchedDecoder):
        self.dec, self.mesh, self.tid, self.out = dec, dec.m.mesh, None, None

    def set_inputs(self, tokens, positions):
        self.dec.set_inputs(tokens, positions)

    def capture(self):
        self.dec.step()  # compile
        ttnn.synchronize_device(self.mesh)
        self.tid = ttnn.begin_trace_capture(self.mesh, cq_id=0)
        self.out = self.dec.step()
        ttnn.end_trace_capture(self.mesh, self.tid, cq_id=0)
        ttnn.synchronize_device(self.mesh)

    def step(self):
        ttnn.execute_trace(self.mesh, self.tid, cq_id=0, blocking=False)
        return self.out

    def release(self):
        if self.tid is not None:
            ttnn.release_trace(self.mesh, self.tid)
            self.tid = None
