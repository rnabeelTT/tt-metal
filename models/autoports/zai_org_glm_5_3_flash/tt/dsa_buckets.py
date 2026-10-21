# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage 10: DSA indexer cost by the batch's filled positions, not by max_seq_len.

Without this, every DSA layer scores and ranks the whole pool-slot range of a slot (P_alloc = T/4 + 32 pools, T =
max_seq_len): decode writes [S, P_alloc] scores and runs the mask and the top-512 over that width; every sparse prefill
chunk scores P_alloc keys per row. The cost grows with max_seq_len (1 slot: 51.1 ms at 16,384 vs 56.2 ms at 280,576).

Scored width buckets: W_b = T_b/4 + 32 pools for bucket contexts T_b (default 4096 * 2^k below T, then T itself). A
query at position p needs the pools [0, (p+1)//4), so the smallest W_b >= max over the batch of (p+1)//4 gives the
same top-512 set. A bucket width is the P_alloc of a model built with max_seq_len T_b, so a bucketed step computes
what that build computes (dsa_decode_scores and the indexer read the same keys through the page table).

Decode (traced, static shapes): the decode step is captured as segments. Each DSA layer ends the running "main"
trace before its selection (dsa_decode_scores -> mask -> top-512), captures that selection once per bucket width
into a small trace that writes the pool ids into a persistent per-layer buffer, and begins the next main trace. A
replay runs main_0, sel_0[b], main_1, ..., sel_10[b], main_11, with b chosen on the host from the positions mirror
(DeviceDecodeInputs.positions). Only the selection varies with b, so the trace region holds one copy of the step
plus the small selection traces.

Prefill (eager): a sparse chunk scores W_b keys (indexer_score_dsa kv_len = W_b) for the smallest W_b that covers the
chunk's last position, and ranks only those columns.

GLM53_DSA_BUCKETS: "auto" (default; on for max_seq_len >= GLM53_DSA_BUCKET_MIN_T) | "0" / "off" (the full-width code,
unchanged) | comma list of bucket contexts (forces it on).
Every bucket program is compiled by install() (eager, before any capture).
"""

from __future__ import annotations

import os

import torch
from loguru import logger

import ttnn

from .dsa import DENSE_MAX_POSITION, KPOOL, TOPK_POOLS

DRAM = ttnn.DRAM_MEMORY_CONFIG
_IOTA = {}  # (id(mesh), W) -> (mesh, [1,1,1,W] fp32 iota), shared by every DSA layer of the mesh


def bucket_contexts(T: int):
    """Bucket contexts (tokens) of a T-token cache, ascending, the last one T; None = bucketing off."""
    env = os.environ.get("GLM53_DSA_BUCKETS", "auto").strip().lower()
    if env in ("0", "off", "none", "false"):
        return None
    if env in ("", "auto", "1", "on"):
        # short caches keep the single decode trace (the segment replays cost more than the narrower selection
        # saves there): GLM53_DSA_BUCKET_MIN_T, default 65536. Measured at 16,384 (stage-10 ticket t4, full model, 1
        # slot): segmented 50.08 / 53.45 / 53.69 ms vs single trace 49.95 / 53.23 / 53.24 ms at positions 300 / 5000 /
        # 15000 (+0.1 to +0.45 ms); the selection at P_alloc 4,128 costs only 0.08 ms per layer
        if T < int(os.environ.get("GLM53_DSA_BUCKET_MIN_T", "65536")):
            return None
        ctx, t = [], 4096
        while t < T:
            ctx.append(t)
            t *= 2
    else:
        ctx = sorted({int(s) for s in env.split(",") if s.strip()})
        ctx = [t for t in ctx if t < T]
    for t in ctx:
        assert (
            t % 128 == 0 and t // KPOOL >= TOPK_POOLS
        ), f"bucket context {t}: multiple of 128, >= {TOPK_POOLS * KPOOL}"
    return ctx + [T]


def width_of(T_b: int) -> int:
    return T_b // KPOOL + 32  # the P_alloc of a T_b-token build


def pick(widths, need: int) -> int:
    """Index of the smallest width >= need pools (the last one otherwise)."""
    for i, W in enumerate(widths):
        if W >= need:
            return i
    return len(widths) - 1


def _inner(a):
    """The GlmDSA object behind a multichip _ReducedOutput wrapper."""
    while "_inner" in a.__dict__:
        a = a.__dict__["_inner"]
    return a


def select_at(dsa, qi, wi, inputs, W: int):
    """GlmDSA.decode_forward's selection at scored width W: top-512 pool ids [1,1,S,512] (uint16 / uint32 TILE)."""
    sc = ttnn.experimental.dsa_decode_scores(qi, wi, dsa.pooled, dsa.page_table, inputs["limit_u32"], W // 32)
    ids = dsa._select_pools(sc, inputs["limit"])
    ttnn.deallocate(sc)
    return ids


def _into_buf(ids, buf):
    if ids.dtype != buf.dtype:  # uint16 ids below 65,536 padded columns
        t = ttnn.typecast(ids, buf.dtype)
        ttnn.deallocate(ids)
        ids = t
    ttnn.copy(ids, buf)
    ttnn.deallocate(ids)


class DsaBuckets:
    """Bucket state of one generator: widths, per-layer persistent id buffers, the segmented decode trace."""

    def __init__(self, mesh, layers, contexts):
        self.mesh = mesh
        self.layers = layers  # inner GlmDSA objects, model order
        self.contexts = contexts
        self.widths = [width_of(t) for t in contexts]
        assert self.widths[-1] == layers[0].P_alloc, (self.widths, layers[0].P_alloc)
        self.mode = None  # "warm" | "capture" while the hook is installed
        self.main, self.sel = [], []
        self.hits = [0] * len(self.widths)
        self._open = None

    # ------------------------------------------------------------------------------------------- setup
    def prepare(self):
        S = self.layers[0].S
        for W in self.widths:
            key = (id(self.mesh), W)
            ent = _IOTA.get(key)
            if ent is None or ent[0] is not self.mesh:
                ent = (self.mesh, self.layers[0]._dev(torch.arange(W).float().reshape(1, 1, 1, W)))
                _IOTA[key] = ent
            for d in self.layers:
                d._const_cache.setdefault(("iota", W), ent[1])
        for d in self.layers:
            d.score_widths = list(self.widths)
            if d.__dict__.get("_sel_buf") is None:
                d._sel_buf = ttnn.from_torch(
                    torch.zeros(1, 1, S, TOPK_POOLS, dtype=torch.int32),
                    dtype=ttnn.uint32,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.mesh,
                    memory_config=DRAM,
                    mesh_mapper=d._rep,
                )

    def _hooks(self, on: bool):
        for d in self.layers:
            if on:
                d.decode_select = self._select
            else:
                d.__dict__.pop("decode_select", None)

    def warm(self, gen):
        """Compile every bucket program, eager, before any capture: one decode step (all slots inactive) that runs
        the selection at every width, then the sparse prefill warm-up plans at every forced width."""
        m = gen.model
        S = gen.max_batch_size
        # one sparse chunk per prefill length (the indexer row block depends on the chunk length only), after one top
        # chunk; the generator's own warm-up already ran every plan at the widths its positions select
        lens = sorted(m.prefill_lengths)
        top = lens[-1]
        plans = [
            (top + n, [top, n])
            for n in lens
            if top + n <= gen.max_seq_len
            and top + n > DENSE_MAX_POSITION + 1
            and (gen.num_blocks is None or -(-(top + n) // 128) < gen.num_blocks)
        ]
        for W in self.widths:
            for d in self.layers:
                d._force_score_width = W
            try:
                for L, sz in plans:
                    m.prefill(torch.zeros(L, dtype=torch.int64), 0, sizes=sz)
            finally:
                for d in self.layers:
                    d.__dict__.pop("_force_score_width", None)
        m.reset_state()
        m.inputs.set_positions([-1] * S)
        self.mode = "warm"
        self._hooks(True)
        try:
            ttnn.deallocate(m.decode_step(gen._tok_view()))
        finally:
            self._hooks(False)
            self.mode = None
        m.inputs.set_positions([-1] * S)
        ttnn.synchronize_device(self.mesh)
        logger.info(f"DSA buckets: widths {self.widths} (contexts {self.contexts}); warm plans {[L for L, _ in plans]}")

    # ------------------------------------------------------------------------------------------- capture
    def _select(self, dsa, qi, wi, inputs):
        """GlmDSA.decode_forward hook: the step's pool ids in the layer's persistent buffer."""
        buf = dsa._sel_buf
        if self.mode == "capture":
            self._end_main()
            row = []
            for W in self.widths:
                tid = ttnn.begin_trace_capture(self.mesh, cq_id=0)
                try:
                    _into_buf(select_at(dsa, qi, wi, inputs, W), buf)
                finally:
                    ttnn.end_trace_capture(self.mesh, tid, cq_id=0)
                row.append(tid)
            self.sel.append(row)
            self._begin_main()
        else:  # warm: every width, eager (the last one leaves the full-width ids)
            for W in self.widths:
                _into_buf(select_at(dsa, qi, wi, inputs, W), buf)
        return buf

    def _begin_main(self):
        self._open = ttnn.begin_trace_capture(self.mesh, cq_id=0)

    def _end_main(self):
        ttnn.end_trace_capture(self.mesh, self._open, cq_id=0)
        self.main.append(self._open)
        self._open = None

    def capture(self, fn):
        """Capture fn() (the decode step) as segments; returns its output."""
        assert not self.main, "decode trace already captured"
        tb0 = _trace_bytes(self.mesh)
        self.mode = "capture"
        self._hooks(True)
        try:
            self._begin_main()
            out = fn()
        finally:
            if self._open is not None:
                self._end_main()
            self._hooks(False)
            self.mode = None
        assert len(self.main) == len(self.sel) + 1 == len(self.layers) + 1, (len(self.main), len(self.sel))
        # the step output is allocated while the earlier segments are live traces: it may sit in their intermediates,
        # which the NEXT step's replay overwrites. It is read only within its own step (sampler trace, host logits).
        mark = getattr(ttnn, "mark_corruptible", None)
        if mark is not None:
            try:
                mark(out)
            except BaseException:  # noqa: BLE001  tracking disabled
                pass
        self.trace_bytes = _trace_bytes(self.mesh) - tb0
        logger.info(
            f"DSA buckets: decode captured as {len(self.main)} segments + {len(self.sel)} x {len(self.widths)} "
            f"selection traces, {self.trace_bytes / 2**20:.1f} MiB of trace region"
        )
        return out

    # ------------------------------------------------------------------------------------------- replay
    def bucket(self, positions) -> int:
        need = max([(p + 1) // KPOOL for p in positions if p >= 0], default=0)
        return pick(self.widths, need)

    def execute(self, positions):
        b = self.bucket(positions)
        self.hits[b] += 1
        for i, tid in enumerate(self.main):
            ttnn.execute_trace(self.mesh, tid, cq_id=0, blocking=False)
            if i < len(self.sel):
                ttnn.execute_trace(self.mesh, self.sel[i][b], cq_id=0, blocking=False)
        return b

    def release(self):
        for tid in self.main + [t for row in self.sel for t in row]:
            ttnn.release_trace(self.mesh, tid)
        self.main, self.sel = [], []


def _trace_bytes(mesh) -> int:
    try:
        v = ttnn.get_memory_view(mesh, ttnn.BufferType.TRACE)
        return int(v.total_bytes_allocated_per_bank) * int(v.num_banks)
    except Exception:  # noqa: BLE001
        return 0


# ------------------------------------------------------------------------------------- generator entry points
def install(gen):
    """Generator warm-up, before any trace capture (idempotent): bucket widths, shared iotas, persistent id buffers,
    every bucket program compiled. Returns the DsaBuckets (gen._dsa_buckets) or None (off / no sparse DSA layer)."""
    if "_dsa_buckets" in gen.__dict__:
        return gen._dsa_buckets
    gen._dsa_buckets = None
    layers = [_inner(d.attn) for d in gen.model.layers if d.attn_kind == "dsa"]
    if not layers or layers[0].T <= TOPK_POOLS * KPOOL:
        return None
    ctx = bucket_contexts(layers[0].T)
    if ctx is None or len(ctx) < 2:
        return None
    ctl = DsaBuckets(gen.mesh, layers, ctx)
    ctl.prepare()
    ctl.warm(gen)
    gen._dsa_buckets = ctl
    return ctl


def capture_decode(gen, fn):
    """(trace handle, output) of the decode step fn(): one trace, or a DsaBuckets segmented capture."""
    ctl = gen.__dict__.get("_dsa_buckets")
    if ctl is None:
        tid = ttnn.begin_trace_capture(gen.mesh, cq_id=0)
        try:
            out = fn()
        finally:
            ttnn.end_trace_capture(gen.mesh, tid, cq_id=0)
        return tid, out
    return ctl, ctl.capture(fn)


def replay_decode(gen):
    """Non-blocking replay of the decode step at the positions the host mirror holds (gen.model.inputs.positions)."""
    t = gen.trace_id
    if isinstance(t, DsaBuckets):
        return t.execute(gen.model.inputs.positions)
    ttnn.execute_trace(gen.mesh, t, cq_id=0, blocking=False)
    return None


def release_decode(gen):
    t = gen.trace_id
    if isinstance(t, DsaBuckets):
        t.release()
    else:
        ttnn.release_trace(gen.mesh, t)
