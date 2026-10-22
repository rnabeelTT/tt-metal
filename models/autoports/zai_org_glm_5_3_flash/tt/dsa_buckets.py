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

GLM53_DSA_BUCKETS: "auto" (default; on for max_seq_len >= GLM53_DSA_BUCKET_MIN_T; 1-slot builds only, S >= 2 takes
none) | "0" / "off" (the full-width code, unchanged) | comma list of bucket contexts (forces it on).
Every bucket program is compiled by install() (eager, before any capture).
"""

from __future__ import annotations

import os

import torch
from loguru import logger

import ttnn

from .dsa import DENSE_MAX_POSITION, KPOOL, TOPK_POOLS

DRAM = ttnn.DRAM_MEMORY_CONFIG


# Multi-slot builds (S >= 2) take no auto DSA decode buckets. With mixed top-k modes (fused <= 16,384 pools, classic
# above; they order tied scores differently) a request's selection depended on the other slots' positions (stage-10
# review #1 P2-2, ticket t20). Classic-only buckets (contexts from 65,536) passed the reduced proof (t22) but not the
# served one: a 4,100-token greedy request differed solo vs next to a ~70K decode (t22, t23 runs ii / iii, async on and
# off), while the same server with DSA buckets off and MoE row buckets on was identical (t23 run iv). 1 slot keeps them.


def bucket_contexts(T: int, sharded: bool = False, slots: int = 1):
    """Bucket contexts (tokens) of a T-token cache, ascending, the last one T; None = bucketing off. The block-sharded
    DSA layout (tt/dsa_shard.py) takes no buckets (unmeasured there; its pool iota is shared already). slots >= 2: no
    auto buckets (served batch invariance, t23); an explicit GLM53_DSA_BUCKETS list still applies. 1 slot starts at
    4,096."""
    env = os.environ.get("GLM53_DSA_BUCKETS", "auto").strip().lower()
    if env in ("0", "off", "none", "false") or sharded:
        return None
    if env in ("", "auto", "1", "on"):
        # short caches keep the single decode trace (the segment replays cost more than the narrower selection
        # saves there): GLM53_DSA_BUCKET_MIN_T, default 65536. Measured at 16,384 (stage-10 ticket t4, full model, 1
        # slot): segmented 50.08 / 53.45 / 53.69 ms vs single trace 49.95 / 53.23 / 53.24 ms at positions 300 / 5000 /
        # 15000 (+0.1 to +0.45 ms); the selection at P_alloc 4,128 costs only 0.08 ms per layer
        if slots >= 2 or T < int(os.environ.get("GLM53_DSA_BUCKET_MIN_T", "65536")):
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


def moe_row_buckets(S: int):
    """Routed-MoE row buckets of an S-slot decode build, ascending, the last one S; None = off.
    GLM53_MOE_BUCKETS: "auto" (default: 1, 2, 4, ... below S, then S, cut to what the trace region holds:
    fit_row_buckets) | "0" / "off" | comma list of row counts (taken as given)."""
    env = os.environ.get("GLM53_MOE_BUCKETS", "auto").strip().lower()
    if env in ("0", "off", "none", "false") or S < 2:
        return None
    if env in ("", "auto", "1", "on"):
        tb, t = [], 1
        while t < S:
            tb.append(t)
            t *= 2
    else:
        tb = sorted({int(x) for x in env.split(",") if x.strip() and int(x) < S})
    return tb + [S]


# Trace-region model of the segmented decode (bytes per die), calibrated on measured captures:
#   decode(S, k) = DECODE_BASE + DECODE_PER_SLOT * (S - 16) [+ DSA_POINTS] + k * n_moe_layers * MOE_POINT_TRACE_BYTES
#   + SAMPLER_MARGIN, k = row buckets below S.
#   16 slots, no DSA buckets (t15b, 16,384): k = 1 -> 57.9 MiB, k = 4 -> 89.4 MiB: 10.5 MiB per bucket (254 KB per
#     point trace x 42) and DECODE_BASE = 47.4 MiB.
#   16 slots, DSA buckets on (stage-9 G9 probe, 260,096, 96 MiB): k = 4 -> 92.4 MiB decode, 93.2 MiB with the sampler
#     traces: DSA_POINTS = 3.0 MiB, sampler about 0.8 MiB.
#   32 slots: the stage-9 single-trace capture grows 41.6 -> 46.4 MiB from 16 to 32 slots: DECODE_PER_SLOT = 0.31 MiB;
#     a point trace is 245,760 B (G9 32-slot OOM, 96 MiB, k = 4: about 95.8 MiB allocated at the failure).
MOE_POINT_TRACE_BYTES = 260_000
DECODE_BASE_BYTES = int(47.4 * (1 << 20))
DECODE_PER_SLOT_BYTES = int(0.31 * (1 << 20))
DSA_POINTS_BYTES = 3 << 20
SAMPLER_MARGIN_BYTES = int(1.5 * (1 << 20))
HEADROOM_BYTES = int(
    1.5 * (1 << 20)
)  # kept free: the 32-slot point traces and the serving sampler variants are not all measured


def decode_trace_estimate(S: int, n_layers: int, k: int, with_dsa: bool = False) -> int:
    """Estimated trace-region bytes of the segmented decode + sampler traces with k MoE row buckets below S."""
    base = DECODE_BASE_BYTES + DECODE_PER_SLOT_BYTES * (S - 16) + (DSA_POINTS_BYTES if with_dsa else 0)
    return base + SAMPLER_MARGIN_BYTES + k * n_layers * MOE_POINT_TRACE_BYTES


def _trace_region(mesh) -> int:
    try:
        v = ttnn.get_memory_view(mesh, ttnn.BufferType.TRACE)
        return int(v.total_bytes_per_bank) * int(v.num_banks)
    except Exception:  # noqa: BLE001
        return 64 << 20


def fit_row_buckets(tbs, n_layers: int, region: int, with_dsa: bool = False):
    """The smallest row buckets (1, 2, 4, ...) whose point traces fit the trace region by decode_trace_estimate (slot
    aware); None when not even the 1-row bucket fits. 16 slots: 64 MiB -> 1, 96 MiB -> 1 / 2 / 4 / 8; 32 slots: 64 MiB ->
    none, 96 MiB -> 1 / 2 / 4."""
    if not tbs:
        return None
    S = tbs[-1]
    k = 0
    while k < len(tbs) - 1 and decode_trace_estimate(S, n_layers, k + 1, with_dsa) + HEADROOM_BYTES <= region:
        k += 1
    return tbs[:k] + [tbs[-1]] if k else None


def moe_family(moe, S: int) -> str:
    """The routed-path family of an S-row decode step (Qwen35MoEOptimized: expert-sorted from
    QWEN35_DECODE_SORTED_MIN_ROWS rows, 32 by default; the pairs path below)."""
    return "sorted" if moe is not None and hasattr(moe, "_decode_sorted") and moe._decode_sorted(S) else "pairs"


def moe_bucket_paths(tbs, family: str = "pairs"):
    """Routed path of every row bucket, the S bucket (tbs[-1]) included. GLM53_MOE_BUCKET_PATH:
      "family" (default): every bucket takes the family of the S bucket's own path (moe_family): a request's routed
                          math is then the same whatever the number of active slots (batch invariance; the pairs and
                          sorted paths are each row-exact but differ from each other in the last bits);
      one name ("pairs" / "sorted" / "auto"): every bucket, the S bucket included;
      "1:pairs,8:sorted,32:pairs": per bucket; an unnamed bucket takes the family.
    The S bucket's value "build" runs the build's own path (the code before stage 10)."""
    env = os.environ.get("GLM53_MOE_BUCKET_PATH", "family").strip().lower()
    if ":" not in env:
        v = family if env == "family" else env
        return {t: v for t in tbs}
    named = {}
    for kv in env.split(","):
        k, v = kv.split(":")
        named[int(k)] = v.strip()
    return {t: named.get(t, family) for t in tbs}


class DsaBuckets:
    """Segmented decode trace of one generator. Two kinds of segment points, each captured once per variant:
    "dsa": the DSA selection of a layer, one variant per scored width (key: the batch's largest position);
    "moe": the routed experts of a MoE layer on the compacted active rows, one variant per row bucket (key: the
           number of active slots). A replay runs main_0, point_0[key], main_1, ..."""

    def __init__(self, mesh, layers, contexts, moe_layers=(), tbs=None, S=1, max_context=None):
        self.mesh = mesh
        self.layers = layers  # inner GlmDSA objects, model order (prefill views / decode widths)
        self.contexts = contexts
        self.widths = [width_of(t) for t in contexts] if contexts else None  # None: DSA decode buckets off
        if self.widths:
            assert self.widths[-1] == layers[0].P_alloc, (self.widths, layers[0].P_alloc)
        self.view_ds = list(getattr(layers[0], "view_ds", None) or []) if layers else []
        self.moe_layers = list(moe_layers) if tbs else []
        self.tbs = tbs
        self.paths = (
            moe_bucket_paths(tbs, moe_family(self.moe_layers[0] if self.moe_layers else None, S)) if tbs else {}
        )
        self.S = S
        self.max_context = max_context
        self.mode = None  # "warm" | "capture" while the hooks are installed
        self.main, self.points = [], []  # points: [(kind, [trace id per variant])]
        self.hits = {"dsa": [0] * len(self.widths or []), "moe": [0] * len(tbs or [])}
        self._open = None
        self._active = None  # host copy of the active-slot list behind the compaction inputs

    @property
    def sel(self):  # the DSA points (kept for callers of the DSA-only version)
        return [t for k, t in self.points if k == "dsa"]

    # ------------------------------------------------------------------------------------------- setup
    def prepare(self):
        S = self.S
        for W in (self.widths or []) + [32 * d for d in self.view_ds]:
            shared = self.layers[0].shared_iota(W)  # one tensor per mesh (GlmDSA.shared_iota)
            for d in self.layers:
                d._const_cache.setdefault(("iota", W), shared)
        for d in self.layers if self.widths else []:
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
        if self.moe_layers:
            rep = ttnn.ReplicateTensorToMesh(self.mesh)
            mk = lambda t, dt, lay: ttnn.from_torch(  # noqa: E731
                t, dtype=dt, layout=lay, device=self.mesh, memory_config=DRAM, mesh_mapper=rep
            )
            # compaction inputs, refreshed on the host when the active set changes (_set_active)
            self.gidx = mk(torch.zeros(1, S, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
            self.sidx = mk(torch.full((1, S), S - 1, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
            self.cpad = mk(torch.ones(1, 1, S, 1), ttnn.float32, ttnn.TILE_LAYOUT)
            self._active = []
            for moe in self.moe_layers:
                if any(p == "sorted" for p in self.paths.values()) and hasattr(moe, "prepare_dispatch"):
                    moe.prepare_dispatch([t for t, p in self.paths.items() if p == "sorted"])
                if not moe.__dict__.get("_bucket_wrapped"):
                    for name in ("_routed_gather", "_routed_masked"):
                        orig = getattr(moe, name)

                        def w(x4, scores, ids, T, *a, _orig=orig, _moe=moe, **k):
                            if self.mode is not None and T == self.S and int(x4.shape[-2]) == self.S:
                                return self._routed(_moe, x4, scores, ids, lambda: _orig(x4, scores, ids, T, *a, **k))
                            return _orig(x4, scores, ids, T, *a, **k)

                        setattr(moe, name, w)
                    moe._bucket_wrapped = True
                    moe._bucket_bufs = {}

    def _hooks(self, on: bool):
        for d in self.layers if self.widths else []:
            if on:
                d.decode_select = self._select
            else:
                d.__dict__.pop("decode_select", None)

    def warm(self, gen):
        """Compile every bucket program, eager, before any capture: the sparse prefill warm-up plans at every forced
        width (DSA), then one decode step (all slots inactive) that runs every point at every variant."""
        m = gen.model
        S = gen.max_batch_size
        plans = []
        forced = ("_force_view_d", self.view_ds) if self.view_ds else ("_force_score_width", self.widths or [])
        if forced[1]:
            # one sparse chunk per prefill length (the indexer row block depends on the chunk length only), after one
            # top chunk; the generator's own warm-up already ran every plan at the widths its positions select
            lens = sorted(m.prefill_lengths)
            top = lens[-1]
            plans = [
                (top + n, [top, n])
                for n in lens
                if top + n <= gen.max_seq_len
                and top + n > DENSE_MAX_POSITION + 1
                and (gen.num_blocks is None or -(-(top + n) // 128) < gen.num_blocks)
            ]
            for v in forced[1]:
                for d in self.layers:
                    setattr(d, forced[0], v)
                try:
                    for L, sz in plans:
                        m.prefill(torch.zeros(L, dtype=torch.int64), 0, sizes=sz)
                finally:
                    for d in self.layers:
                        d.__dict__.pop(forced[0], None)
            m.reset_state()
        m.inputs.set_positions([-1] * S)
        self._set_active([-1] * S)
        self.mode = "warm"
        self._hooks(True)
        try:
            ttnn.deallocate(m.decode_step(gen._tok_view()))
        finally:
            self._hooks(False)
            self.mode = None
        m.inputs.set_positions([-1] * S)
        ttnn.synchronize_device(self.mesh)
        logger.info(
            f"decode buckets: DSA widths {self.widths} (contexts {self.contexts}), prefill views {self.view_ds} blocks; "
            f"warm plans {[L for L, _ in plans]}; "
            f"MoE row buckets {self.tbs} paths {self.paths} on {len(self.moe_layers)} layers"
        )

    # ------------------------------------------------------------------------------------------- points
    def _point(self, kind, bodies):
        """Capture: end the main trace, one trace per variant body, begin the next main trace. Warm: run every body."""
        if self.mode == "capture":
            self._end_main()
            row = []
            for body in bodies:
                tid = ttnn.begin_trace_capture(self.mesh, cq_id=0)
                try:
                    body()
                finally:
                    ttnn.end_trace_capture(self.mesh, tid, cq_id=0)
                row.append(tid)
            self.points.append((kind, row))
            self._begin_main()
        else:
            for body in bodies:
                body()

    def _select(self, dsa, qi, wi, inputs):
        """GlmDSA.decode_forward hook: the step's pool ids in the layer's persistent buffer."""
        buf = dsa._sel_buf
        self._point("dsa", [lambda W=W: _into_buf(select_at(dsa, qi, wi, inputs, W), buf) for W in self.widths])
        return buf

    def _routed(self, moe, x4, scores, ids, full):
        """Routed-experts hook of a MoE layer (decode rows = S): every row bucket writes the step's routed output
        [1,1,S,H] into the layer's persistent buffer; the caller gets a copy (it frees its routed tensor)."""
        bufs = moe._bucket_bufs

        def into(out):
            buf = bufs.get("out")
            if buf is None:  # first use (warm-up, before any capture): the persistent buffer takes the output's spec
                bufs["out"] = ttnn.clone(out, memory_config=DRAM)
                ttnn.deallocate(out)
                return
            if out.dtype != buf.dtype or out.layout != buf.layout:
                raise RuntimeError(
                    f"MoE bucket output spec changed: {out.dtype}/{out.layout} vs {buf.dtype}/{buf.layout}"
                )
            ttnn.copy(out, buf)
            ttnn.deallocate(out)

        bodies = [lambda T_b=T_b: into(self._compact_routed(moe, x4, scores, ids, T_b)) for T_b in self.tbs[:-1]]
        bodies.append(lambda: into(self._full_routed(moe, x4, scores, ids, full)))
        self._point("moe", bodies)
        return ttnn.clone(bufs["out"], memory_config=DRAM)

    def _full_routed(self, moe, x4, scores, ids, full):
        """The S bucket (every slot row, no compaction) on its configured path."""
        path, S = self.paths.get(self.S, "build"), self.S
        if path == "pairs":
            return moe._routed_pairs(x4, scores, ids, S)  # the inactive-row skip applies (T == S)
        if path == "sorted":
            return moe._routed_sorted(x4, scores, ids, S)
        return full()  # "build" / "auto": the build's own path

    def _compact_routed(self, moe, x4, scores, ids, T_b):
        """The routed experts of the first T_b compacted rows (active slots first; padding rows repeat the first active
        slot), scattered back to the S slot rows (inactive rows: zeros). Every move is an exact row copy (embedding)."""
        S, H = self.S, int(x4.shape[-1])
        RM = ttnn.ROW_MAJOR_LAYOUT
        k = int(ids.shape[-1])
        path = self.paths[T_b]
        g = ttnn.slice(self.gidx, (0, 0), (1, T_b))  # [1, T_b] uint32
        # activations
        x_rm = ttnn.to_layout(x4, RM, memory_config=DRAM)
        xc = ttnn.embedding(g, ttnn.reshape(x_rm, (S, H)), layout=ttnn.TILE_LAYOUT, memory_config=DRAM)
        ttnn.deallocate(x_rm, force=False)
        xc = ttnn.reshape(xc, (1, 1, T_b, H))
        # routing weights and expert ids (ids < 256: exact in bf16), one table of 32 columns
        sc_rm = ttnn.to_layout(scores, RM, memory_config=DRAM)
        idb = ttnn.typecast(ids, ttnn.bfloat16)
        id_rm = ttnn.to_layout(idb, RM, memory_config=DRAM)
        ttnn.deallocate(idb)
        tab = ttnn.concat([ttnn.reshape(sc_rm, (S, k)), ttnn.reshape(id_rm, (S, k))], dim=-1, memory_config=DRAM)
        ttnn.deallocate(sc_rm, force=False)
        ttnn.deallocate(id_rm, force=False)
        if 2 * k < 32:
            tp = ttnn.reshape(
                ttnn.pad(ttnn.reshape(tab, (1, 1, S, 2 * k)), [(0, 0), (0, 0), (0, 0), (0, 32 - 2 * k)], 0.0), (S, 32)
            )
            ttnn.deallocate(tab)
            tab = tp
        tc = ttnn.embedding(g, tab, layout=RM, memory_config=DRAM)  # [1, T_b, 32]
        ttnn.deallocate(tab)
        ttnn.deallocate(g)
        tc = ttnn.reshape(tc, (1, 1, T_b, int(tc.shape[-1])))
        sc_c = ttnn.to_layout(ttnn.slice(tc, (0, 0, 0, 0), (1, 1, T_b, k)), ttnn.TILE_LAYOUT, memory_config=DRAM)
        idf = ttnn.typecast(
            ttnn.to_layout(ttnn.slice(tc, (0, 0, 0, k), (1, 1, T_b, 2 * k)), ttnn.TILE_LAYOUT, memory_config=DRAM),
            ttnn.float32,
        )
        ttnn.deallocate(tc)
        if path == "pairs":  # padding rows: expert ids 65535, skipped by the grouped sparse matmul (no weight reads)
            pad = ttnn.slice(self.cpad, (0, 0, 0, 0), (1, 1, T_b, 1))
            keep = ttnn.multiply(idf, ttnn.rsub(pad, 1.0), memory_config=DRAM)
            sent = ttnn.multiply(pad, 65535.0, memory_config=DRAM)
            ttnn.deallocate(idf)
            idf = ttnn.add(keep, sent, memory_config=DRAM)
            for t in (keep, sent, pad):
                ttnn.deallocate(t)
            id_c = ttnn.typecast(idf, ttnn.uint32)
        else:  # padding rows repeat the first active row: no extra expert
            id_c = ttnn.typecast(ttnn.typecast(idf, ttnn.uint32), ids.dtype)
        ttnn.deallocate(idf)
        if path == "pairs":
            out = moe._routed_pairs(xc, sc_c, id_c, T_b)
        elif path == "sorted":
            out = moe._routed_sorted(xc, sc_c, id_c, T_b)
        else:  # the row count's natural decode path (Qwen35MoE.forward)
            out = (
                moe._routed_gather(xc, sc_c, id_c, T_b)
                if T_b <= moe.GATHER_MAX_TOKENS
                else moe._routed_masked(xc, sc_c, id_c, T_b, 32)
            )
        for t in (xc, sc_c, id_c):
            ttnn.deallocate(t)
        # scatter: [T_b, H] + zero rows -> [S, H] -> rows by slot (sidx: compact row of each slot, S - 1 = a zero row)
        dt = out.dtype
        o = out if dt == ttnn.bfloat16 else ttnn.typecast(out, ttnn.bfloat16)
        o_rm = ttnn.to_layout(o, RM, memory_config=DRAM)
        if o is not out:
            ttnn.deallocate(o)
        ttnn.deallocate(out)
        o_p = ttnn.pad(ttnn.reshape(o_rm, (1, 1, T_b, H)), [(0, 0), (0, 0), (0, S - T_b), (0, 0)], 0.0)
        ttnn.deallocate(o_rm, force=False)
        full = ttnn.embedding(self.sidx, ttnn.reshape(o_p, (S, H)), layout=ttnn.TILE_LAYOUT, memory_config=DRAM)
        ttnn.deallocate(o_p, force=False)
        full = ttnn.reshape(full, (1, 1, S, H))
        if dt != ttnn.bfloat16:
            f2 = ttnn.typecast(full, dt)
            ttnn.deallocate(full)
            full = f2
        return full

    def _set_active(self, positions):
        """Host -> device compaction inputs when the active set changes (outside any trace; no device sync)."""
        if not self.moe_layers:
            return
        T = self.max_context
        act = [s for s, p in enumerate(positions) if p >= 0 and (T is None or p < T)]
        if act == self._active:
            return
        S = self.S
        n = len(act)
        g = act + [act[0] if act else 0] * (S - n)
        sidx = [act.index(s) if s in act else S - 1 for s in range(S)]
        pad = [0.0] * n + [1.0] * (S - n)
        rep = ttnn.ReplicateTensorToMesh(self.mesh)
        for t, buf, dt, lay in (
            (torch.tensor([g], dtype=torch.int32), self.gidx, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT),
            (torch.tensor([sidx], dtype=torch.int32), self.sidx, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT),
            (torch.tensor(pad).reshape(1, 1, S, 1), self.cpad, ttnn.float32, ttnn.TILE_LAYOUT),
        ):
            ttnn.copy_host_to_device_tensor(ttnn.from_torch(t, dtype=dt, layout=lay, mesh_mapper=rep), buf)
        self._active = act

    # ------------------------------------------------------------------------------------------- capture
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
        n_pts = (len(self.layers) if self.widths else 0) + len(self.moe_layers)
        assert len(self.main) == len(self.points) + 1 == n_pts + 1, (len(self.main), len(self.points), n_pts)
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
            f"decode buckets: decode captured as {len(self.main)} segments + {len(self.layers) if self.widths else 0} DSA x "
            f"{len(self.widths or [])} + {len(self.moe_layers)} MoE x {len(self.tbs or [])} point traces, "
            f"{self.trace_bytes / 2**20:.1f} MiB of trace region"
        )
        return out

    # ------------------------------------------------------------------------------------------- replay
    def bucket(self, positions) -> int:
        """DSA width bucket of the batch."""
        if not self.widths:
            return None
        need = max([(p + 1) // KPOOL for p in positions if p >= 0], default=0)
        return pick(self.widths, need)

    def moe_bucket(self, positions) -> int:
        if not self.tbs:
            return None
        T = self.max_context
        n = sum(1 for p in positions if p >= 0 and (T is None or p < T))
        return next(i for i, t in enumerate(self.tbs) if t >= n or t == self.tbs[-1])

    def execute(self, positions):
        key = {"dsa": self.bucket(positions), "moe": self.moe_bucket(positions)}
        self._set_active(positions)
        for kind in ("dsa", "moe"):
            if key[kind] is not None:
                self.hits[kind][key[kind]] += 1
        for i, tid in enumerate(self.main):
            ttnn.execute_trace(self.mesh, tid, cq_id=0, blocking=False)
            if i < len(self.points):
                kind, row = self.points[i]
                ttnn.execute_trace(self.mesh, row[key[kind]], cq_id=0, blocking=False)
        return key["dsa"]

    def device_tensors(self) -> dict:
        """Every device tensor this object owns or installed (generator._trace_tensors: address-stability guard)."""
        out = {}
        for i, d in enumerate(self.layers):
            if d.__dict__.get("_sel_buf") is not None:
                out[f"dsa{i}.sel_buf"] = d._sel_buf
        if self.layers:
            for W in (self.widths or []) + [32 * d for d in self.view_ds]:
                t = self.layers[0]._const_cache.get(("iota", W))
                if t is not None:
                    out[f"iota{W}"] = t
        for name in ("gidx", "sidx", "cpad"):
            if getattr(self, name, None) is not None:
                out[f"moe.{name}"] = getattr(self, name)
        for i, moe in enumerate(self.moe_layers):
            buf = moe.__dict__.get("_bucket_bufs", {}).get("out")
            if buf is not None:
                out[f"moe{i}.out"] = buf
        return out

    def release(self):
        """Release the traces and drop every device tensor this object owns (selection / output buffers, compaction
        inputs) and the shared iotas of this mesh, so a torn-down generator keeps no pool or buffer alive."""
        for tid in self.main + [t for _, row in self.points for t in row]:
            ttnn.release_trace(self.mesh, tid)
        self.main, self.points = [], []
        for d in self.layers:
            buf = d.__dict__.pop("_sel_buf", None)
            if buf is not None:
                ttnn.deallocate(buf)
            d.__dict__.pop("score_widths", None)
            for W in (self.widths or []) + [32 * v for v in self.view_ds]:
                if W != d.P_alloc:
                    d._const_cache.pop(("iota", W), None)
        for moe in self.moe_layers:
            buf = moe.__dict__.get("_bucket_bufs", {}).pop("out", None)
            if buf is not None:
                ttnn.deallocate(buf)
        for name in ("gidx", "sidx", "cpad"):
            t = getattr(self, name, None)
            if t is not None:
                ttnn.deallocate(t)
                setattr(self, name, None)
        from .dsa import release_shared_iotas

        release_shared_iotas(self.mesh)
        self.layers, self.moe_layers = [], []


def _trace_bytes(mesh) -> int:
    try:
        v = ttnn.get_memory_view(mesh, ttnn.BufferType.TRACE)
        return int(v.total_bytes_allocated_per_bank) * int(v.num_banks)
    except Exception:  # noqa: BLE001
        return 0


# ------------------------------------------------------------------------------------- generator entry points
def install(gen):
    """Generator warm-up, before any trace capture (idempotent): DSA bucket widths, shared iotas, persistent id
    buffers; MoE row buckets, compaction inputs; every bucket program compiled. Returns the DsaBuckets
    (gen._dsa_buckets) or None (both off)."""
    if "_dsa_buckets" in gen.__dict__:
        return gen._dsa_buckets
    gen._dsa_buckets = None
    m = gen.model
    layers = [_inner(d.attn) for d in m.layers if d.attn_kind == "dsa"]
    ctx = None
    if layers and layers[0].T > TOPK_POOLS * KPOOL:
        ctx = bucket_contexts(
            layers[0].T, sharded=bool(getattr(layers[0], "dsa_shard", False)), slots=gen.max_batch_size
        )
        if ctx is not None and len(ctx) < 2:
            ctx = None
    S = gen.max_batch_size
    tbs = moe_row_buckets(S)
    moe_layers = [_inner(d.mlp) for d in m.layers if d.mlp_kind == "moe"] if tbs else []
    if moe_layers and os.environ.get("GLM53_MOE_BUCKETS", "auto").strip().lower() in ("", "auto", "1", "on"):
        tbs = fit_row_buckets(tbs, len(moe_layers), _trace_region(gen.mesh), with_dsa=ctx is not None)
    if not moe_layers or tbs is None:
        tbs = None
    views = bool(layers) and bool(getattr(layers[0], "view_ds", None))
    if ctx is None and tbs is None and not views:
        return None
    ctl = DsaBuckets(gen.mesh, layers, ctx, moe_layers, tbs, S=S, max_context=getattr(m, "max_context", None))
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
