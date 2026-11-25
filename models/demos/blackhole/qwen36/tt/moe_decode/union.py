# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Multi-row fused MoE decode (2..8 decode rows: concurrent users at small widths, the MTP verify's rows): every
union expert's weights are read once for all rows.

* ``RouteRows``: one core per row: the row's top-k + softmax and shared-gate sigmoid into the route table
  ([M, 32] UINT16: k ids, the shared weight at 16, the routed weights at 17..).
* ``ExpertGateUpRows``: the union of the rows' experts (ascending, slot 0 = the shared expert); each core streams its
  column pair of its union slots in place (as ExpertGateUp), multiplies all rows ([Mp, 32] activation tiles,
  Mp = 2 / 4 / 8 >= M), SwiGLU x the per-row weight (0 for rows that did not route to the expert).
* ``ExpertDownRows``: the down columns of every union slot, accumulated over chunks of slots, M output rows.

Design and the per-row weight load (identity matmul) follow the gpt-oss DFlash union ops
(models/demos/gpt_oss/tt/fused_decode/union.py). Numerics as experts.py (weights stored at 5 significant bits,
activations / weights rounded to 7 for the srcB operand).
"""

from __future__ import annotations

import os

import torch

import ttnn

from .experts import (
    KERNEL_DIR,
    RING_BYTES,
    ROUTERS,
    TILE,
    TILE_BYTES,
    _cb,
    _ep_defines,
    _kernels,
    _page_bytes,
    _range_set,
    bank_columns,
    bank_workers,
    gu_layout,
    gu_ring,
    logits_row_bytes,
    offrow_candidates,
    place_along_rows,
    place_banks_offrow,
)

TILE_ROWS = (1, 2, 4, 8)  # custom_mm activation tile heights


def tile_rows(m):
    """[rows, 32] activation tile height for m decode rows (> 8 rows: groups of 8)."""
    return next(r for r in TILE_ROWS if r >= min(m, 8))


def row_groups(m):
    return -(-m // tile_rows(m))


ACT_PAGE = 64 * 8  # one [8, 32] BF16 tile: every width's [Mp, 32] tile fits a page


def act_buffer(mesh_device, inter, num_experts, k, rows, subs=1):
    """The activations of a `rows`-row step (one [Mp, 32] tile per ACT_PAGE page: union slot, K tile, row group), L1
    interleaved: every down core reads all of it, which from DRAM cost more than the weights at 32 rows (2.7 MB x 64
    cores per device on 2 chips). Allocated per step on the device (ttnn.empty: no host upload, trace-safe) and freed
    after the down op: a persistent buffer of the 32-row size (2.7 MB per device on 2 chips) left too little L1 for the
    model's other programs' circular buffers. Every page the down op reads is written by the gate|up op first."""
    slots = min(num_experts * subs, rows * k * subs) + 1
    return ttnn.empty(
        [slots * inter // TILE * row_groups(rows), ACT_PAGE // 2],
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
        mesh_device,
        ttnn.L1_MEMORY_CONFIG,
    )


class RowsBuffers:
    """Per row count M (model-wide, consumed before the next layer's MoE): route table [M, 32] UINT16, activations
    ([(U_max + 1) * I / 32] pages of one [Mp, 32] tile), output [1, 1, M, H] BF16 TILE in DRAM (zeros; rows 0..Mp-1 of
    each tile are written), the identity tile in the weight dtype."""

    def __init__(self, mesh_device, hidden, inter, num_experts, k, rows, weight_dtype, act=None):
        """act: a persistent activation buffer (op tests); the model allocates one per step (act_buffer)."""
        multi = getattr(mesh_device, "get_num_devices", lambda: 1)() > 1
        rep = {"mesh_mapper": ttnn.ReplicateTensorToMesh(mesh_device)} if multi else {}
        l1 = dict(device=mesh_device, memory_config=ttnn.L1_MEMORY_CONFIG, **rep)
        self.max_slots = min(num_experts, rows * k) + 1
        self.table = ttnn.from_torch(
            torch.zeros(rows, 32, dtype=torch.int32), dtype=ttnn.uint16, layout=ttnn.ROW_MAJOR_LAYOUT, **l1
        )
        self.act = act
        self.out = ttnn.from_torch(
            torch.zeros(1, 1, rows, hidden, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            **rep,
        )
        self.identity = ttnn.from_torch(
            torch.eye(TILE)[None, None],
            dtype=weight_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            **rep,
        )


class RouteRows:
    """(logits [M, E + 32] BF16 (softmax) / [M, E] FP32 (sigb), table) -> table, one core per row (NCRISC and BRISC
    each take half the logits)."""

    def __init__(self, mesh_device, num_experts, k, rows, router="softmax", split=None):
        """split: cores per row (QWEN35_ROUTE_SPLIT, default 1): each takes E / split experts and the part-0 core merges
        (kernels/route_rows_reader.cpp ROUTE_SPLIT); the same selection and weights as one core per row. The sigmoid +
        bias router costs ~100 cycles per expert on these FPU-less cores: M2.7 (E 256, 4 rows) 31 us at 1 core per row.
        """
        assert router in ROUTERS, router
        self.router = router
        self.E, self.k, self.rows = num_experts, k, rows
        if split is None:
            split = int(os.environ.get("QWEN35_ROUTE_SPLIT", "1"))
        while split > 1 and (num_experts % (2 * split) or rows * split > 32):
            split //= 2  # whole even parts (the softmax router reads 2 BF16 logits per word); at most 32 cores
        self.split = split
        gx = mesh_device.compute_with_storage_grid_size().x
        n = rows * split  # core i = row i // split, part i % split
        self.assign = [ttnn.CoreCoord(i % gx, i // gx) for i in range(n)]
        self.cores = _range_set(self.assign)
        self.noc = [mesh_device.worker_core_from_logical_core(c) for c in self.assign] if split > 1 else None

    def __call__(self, logits, table, bias=None):
        """bias: the sigb router's selection bias (Q23 int32 [1, 1, 1, E] row-major, one DRAM page)."""
        assert (bias is not None) == (self.router == "sigb")
        cb_scr, cb_part, cb_lg = 0, 1, 2
        lg_bytes = logits_row_bytes(self.E, self.router)
        cb_gather = 3
        P = self.split
        cbs = [
            _cb(self.cores, cb_scr, ttnn.uint16, lg_bytes + 64, 1),
            _cb(self.cores, cb_part, ttnn.uint32, 8 * self.k, 1),
            _cb(self.cores, cb_lg, ttnn.bfloat16, lg_bytes, 1),
        ]
        if P > 1:
            cbs.append(_cb(self.cores, cb_gather, ttnn.uint32, 8 * self.k * P, 1))
        rd_rt, wr_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        bias_arg = [] if bias is None else [bias.buffer_address()]
        part_e = self.E // P
        for i, c in enumerate(self.assign):
            r, p = divmod(i, P)
            rd = [logits.buffer_address(), table.buffer_address(), r] + bias_arg
            wr = [logits.buffer_address(), r] + bias_arg
            if P > 1:
                e0, e1 = p * part_e, (p + 1) * part_e
                lead = self.noc[r * P]
                rd += [e0, e1, p, lead.x, lead.y]
                wr += [(e0 + e1) // 2, e1]
            rd_rt[c.x][c.y] = rd
            wr_rt[c.x][c.y] = wr
        lg_ct = ttnn.TensorAccessorArgs(logits).get_compile_time_args()
        rd_ct = [cb_scr, cb_part, self.E, self.k] + lg_ct + ttnn.TensorAccessorArgs(table).get_compile_time_args()
        rd_ct += [cb_gather] if P > 1 else []
        wr_ct = [cb_lg, cb_part, self.E, self.k] + lg_ct
        dm = ttnn.DataMovementConfigDescriptor
        defines = [("ROUTER_SIGB", "1")] if self.router == "sigb" else []
        defines += [("ROUTE_SPLIT", str(P))] if P > 1 else []
        # QWEN35_ROUTE_KEYS=1 (sigb): the routing weights from the selection keys, no second sigmoid pass (bitwise)
        defines += [("ROUTE_KEYS", "1")] if self.router == "sigb" and os.environ.get("QWEN35_ROUTE_KEYS") == "1" else []
        # QWEN35_ROUTE_FAST=1: the same selection and weights with cheaper leader / sigmoid code (router_topk.hpp; bitwise)
        defines += [("ROUTE_FAST", "1")] if os.environ.get("QWEN35_ROUTE_FAST") == "1" else []
        kernels = [
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "route_rows_reader.cpp"),
                core_ranges=self.cores,
                compile_time_args=rd_ct,
                runtime_args=rd_rt,
                config=dm(processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.NOC_0),
                defines=defines,
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "route_rows_writer.cpp"),
                core_ranges=self.cores,
                compile_time_args=wr_ct,
                runtime_args=wr_rt,
                config=dm(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_1),
                defines=defines,
            ),
        ]
        io = [logits, table] + ([bias] if bias is not None else [])
        sems = [ttnn.SemaphoreDescriptor(0, ttnn.CoreType.WORKER, self.cores, 0)] if P > 1 else []
        ttnn.generic_op(io, ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems, cbs=cbs))
        return table


class RouteScores(RouteRows):
    """Router tail for a port's own expert paths (GLM-4.5-Air: the sigb router feeding its sparse-matmul decode):
    (logits [1, 1, M, E] (sigb: FP32 TILE), bias) -> (scores [1, 1, M, k] BF16 TILE, ids [1, 1, M, k] UINT16 TILE),
    one core per row (route_rows_* kernels with ROUTE_TILES). The outputs are persistent per M (zero padding, L1;
    build every width before trace capture): the caller must not deallocate them."""

    def __init__(self, mesh_device, num_experts, k, router="sigb"):
        super().__init__(mesh_device, num_experts, k, 32, router=router, split=1)
        self.mesh_device = mesh_device
        self._out = {}

    def outputs(self, rows):
        if rows not in self._out:
            multi = getattr(self.mesh_device, "get_num_devices", lambda: 1)() > 1
            kw = dict(layout=ttnn.TILE_LAYOUT, device=self.mesh_device, memory_config=ttnn.L1_MEMORY_CONFIG)
            if multi:
                kw["mesh_mapper"] = ttnn.ReplicateTensorToMesh(self.mesh_device)
            scores = ttnn.from_torch(torch.zeros(1, 1, rows, self.k), dtype=ttnn.bfloat16, **kw)
            ids = ttnn.from_torch(torch.zeros(1, 1, rows, self.k, dtype=torch.int32), dtype=ttnn.uint16, **kw)
            self._out[rows] = (scores, ids)
        return self._out[rows]

    def __call__(self, logits, rows, bias=None):
        assert (bias is not None) == (self.router == "sigb")
        scores, ids = self.outputs(rows)
        cb_scr, cb_part, cb_lg = 0, 1, 2
        lg_bytes = logits_row_bytes(self.E, self.router)
        cores = _range_set(self.assign[:rows])
        cbs = [
            _cb(cores, cb_scr, ttnn.uint16, lg_bytes + 128, 1),
            _cb(cores, cb_part, ttnn.uint32, 8 * self.k, 1),
            _cb(cores, cb_lg, ttnn.bfloat16, lg_bytes, 1),
        ]
        rd_rt, wr_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        bias_arg = [] if bias is None else [bias.buffer_address()]
        for r, c in enumerate(self.assign[:rows]):
            rd_rt[c.x][c.y] = [logits.buffer_address(), ids.buffer_address(), r] + bias_arg + [scores.buffer_address()]
            wr_rt[c.x][c.y] = [logits.buffer_address(), r] + bias_arg
        lg_ct = ttnn.TensorAccessorArgs(logits).get_compile_time_args()
        rd_ct = [cb_scr, cb_part, self.E, self.k] + lg_ct
        rd_ct += (
            ttnn.TensorAccessorArgs(ids).get_compile_time_args()
            + ttnn.TensorAccessorArgs(scores).get_compile_time_args()
        )
        wr_ct = [cb_lg, cb_part, self.E, self.k] + lg_ct
        dm = ttnn.DataMovementConfigDescriptor
        defines = [("ROUTE_TILES", "1")] + ([("ROUTER_SIGB", "1")] if self.router == "sigb" else [])
        kernels = [
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "route_rows_reader.cpp"),
                core_ranges=cores,
                compile_time_args=rd_ct,
                runtime_args=rd_rt,
                config=dm(processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.NOC_0),
                defines=defines,
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "route_rows_writer.cpp"),
                core_ranges=cores,
                compile_time_args=wr_ct,
                runtime_args=wr_rt,
                config=dm(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_1),
                defines=defines,
            ),
        ]
        io = [logits, ids, scores] + ([bias] if bias is not None else [])
        ttnn.generic_op(io, ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))
        return scores, ids


def selection_bias_q23(bias):
    """GLM-4.5-Air e_score_correction_bias [E] -> the sigb kernels' Q23 int32 (mean-centred in float64: selection is
    invariant to a constant shift, and the bias sits near 16-20 with a ~0.02-0.05 spread)."""
    b = bias.reshape(-1).to(torch.float64)
    return torch.round((b - b.mean()) * (1 << 23)).to(torch.int32)


def upload_selection_bias(mesh_device, bias):
    """The Q23 bias as one DRAM page ([1, 1, 1, E] INT32 row-major: page 0 in bank 0, where the kernels read it)."""
    multi = getattr(mesh_device, "get_num_devices", lambda: 1)() > 1
    return ttnn.from_torch(
        selection_bias_q23(bias).reshape(1, 1, 1, -1),
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        **({"mesh_mapper": ttnn.ReplicateTensorToMesh(mesh_device)} if multi else {}),
    )


def _kch(v, kt, ring, banks=None):
    """K-row chunks per pushed block: the largest n <= v with kt / n even (custom_mm), the reads in flight within the
    transaction ids, and (banks: the remainder pairs' K-split blocks) whole banks per chunk."""
    return max(n for n in range(1, int(v) + 1)
               if kt % n == 0 and (kt // n) % 2 == 0 and ring * n - n <= 14 and (banks is None or banks % n == 0))


class ExpertGateUpRows:
    """(x, table, identity, up_gate, shared_up_gate, act) -> act. readers: cores per column group (experts.gu_layout);
    union slot s goes to core s % readers."""

    def __init__(
        self,
        mesh_device,
        hidden,
        inter,
        num_experts,
        k,
        weight_dtype,
        rows,
        readers=None,
        buffers=5,
        shared=True,
        ep_experts=None,
        gate_first=False,
        subs=1,
        rem=0,
        pairs_total=None,
    ):
        """rem > 0: `rem` more gate|up column pairs past the `inter` columns, each pair's [kt, 2] block split over the
        banks by K (bank b: K tiles [b kt / banks, (b + 1) kt / banks) of the pair's gate | up columns, one contiguous
        run per expert in a [1, E, rem * kt * 2 / banks * 32, banks * 32] tensor rem_w, tile (t, b)): streamed from every
        bank by the pair's own cores (no column padding to a multiple of the banks). pairs_total: the activation pairs per slot
        (>= inter / 32 + rem: the extra pairs are written as zeros, e.g. an even down K).
        subs > 1: each expert stored as `subs` sub-shards of `inter` columns (virtual experts e * subs + s, weights
        stacked [E * subs, ...]; kernels' MOE_SUBS), one program for all of them.
        shared False: routed experts only (no union slot 0). ep_experts / gate_first: as ExpertGateUp. readers None: 4 cores per column pair. At 8 rows a core spends ~2.6 us per union slot (unpack 1.7 +
        SFPU 0.7, measured), yet 6-22 cores per pair measured slower: the extra cores sit off the banks' rows and
        their reads collapse (8 rows, 4-chip shapes: 51 us median at 4 per pair, 89 at 22, 217 max at 8)."""
        self.banks = len(bank_workers(mesh_device))
        self.kt = hidden // TILE
        self.n_tiles = 2 * inter // TILE
        self.pairs = self.n_tiles // 2
        self.m, self.nb, self.ngroups, self.group_pairs = gu_layout(self.n_tiles, self.banks)
        self.E, self.k, self.inter, self.rows = num_experts, k, inter, rows
        self.subs = subs
        assert subs == 1 or (not ep_experts and not shared)
        self.shared, self.ep_experts, self.gate_first = shared, ep_experts, gate_first
        self.trows, self.groups = tile_rows(rows), row_groups(rows)
        self.max_slots = min(num_experts * subs, rows * k * subs) + 1
        # split-pair (QWEN35_MOE_SPLIT_PAIR=1, m = 1: a pair's up / gate columns in banks q / q + banks / 2): one bank per
        # core, `readers` cores per bank on its row; the gate cores send w * silu(g) to their up partners (gur_writer.cpp)
        self.split = (
            os.environ.get("QWEN35_MOE_SPLIT_PAIR", "0") == "1"
            and self.m == 1
            and gu_layout(self.n_tiles, self.banks)[1] == 2
            and not gate_first
        )
        self.ring = int(os.environ.get("QWEN35_MOE_SPLIT_RING", "8"))
        if readers is None:
            # QWEN35_MOE_GU_READERS: a per-model override of the 4 cores per column group (Inkling-Small, H 4096, 512
            # columns per die: 2 readers, with QWEN35_MOE_DN_OFFROW=1 / DN_CHUNK=2 the whole MoE 270 -> 130 us per layer;
            # bench_moe_rows_inkling.py; outputs bit-identical)
            readers = (
                int(os.environ.get("QWEN35_MOE_SPLIT_READERS", "4"))
                if self.split
                # QWEN35_MOE_REM_GU_READERS: with remainder pairs (Inkling-2-Mini TP2 / TP4: 2 vs 3, 4: -6 / -30 %)
                else int(os.environ.get("QWEN35_MOE_REM_GU_READERS", "2")) if rem
                else int(os.environ.get("QWEN35_MOE_GU_READERS", "4"))
            )
        self.readers = readers
        self.tile_bytes = TILE_BYTES[weight_dtype]
        self.weight_dtype = weight_dtype
        # nb = 2 (odd m: a slot is two consecutive ring blocks the compute indexes together): an even ring
        self.buffers = gu_ring(self.kt, self.m, self.nb, self.tile_bytes, buffers, even=True)
        # QWEN35_MOE_GU_KCH: blocks pushed as this many K-row chunks (the compute starts on a block's first rows; bitwise)
        self.kch = 1 if self.split else _kch(os.environ.get("QWEN35_MOE_GU_KCH", "1"), self.kt, self.buffers)
        self.page_bytes = _page_bytes(self.kt * self.m // self.kch, self.tile_bytes)
        self.dst_full = self.m * self.nb + 1 > 4  # the matmul tiles + the weight tile; FP32 DST: 4 half-sync, 8 full
        assert self.m * self.nb + 1 <= 8, f"{self.m} up|gate columns per bank: more than 8 DST tiles"
        if self.split:
            # banks whose optimal reader shares a row with an earlier bank's (banks 0 / 4 on row 9 here): readers on the
            # next row inward, reading on NOC1 (QWEN35_MOE_SPLIT_OFFROW=1)
            workers = bank_workers(mesh_device, rows=True)
            self.noc1_banks = set()
            if os.environ.get("QWEN35_MOE_SPLIT_OFFROW", "1") == "1":
                seen = set()
                for b, w in enumerate(workers):
                    if w.y in seen:
                        self.noc1_banks.add(b)
                    seen.add(w.y)
            grid = mesh_device.compute_with_storage_grid_size()
            used = set()
            placed = {}
            for b in range(self.banks):
                if b in self.noc1_banks:
                    w = workers[b]
                    cand = offrow_candidates(mesh_device, w, rows=True)
                    cores = []
                    for x, yy in cand:
                        if len(cores) == readers:
                            break
                        if 0 <= x < grid.x and (x, yy) not in used:
                            used.add((x, yy))
                            cores.append(ttnn.CoreCoord(x, yy))
                    assert len(cores) == readers, f"no room for {readers} cores near bank {b}"
                    placed[b] = cores
            placed.update(place_along_rows(mesh_device, {b: readers for b in range(self.banks) if b not in self.noc1_banks},
                                           used=used, rows=True))
            self.assign = [(c, b, r) for b in range(self.banks) for r, c in enumerate(placed[b])]
            half = self.banks // 2
            self.partner = {
                (b, r): placed[(b + half) % self.banks][r] for b in range(self.banks) for r in range(readers)
            }
            self.mesh = mesh_device
        else:
            # a two-bank group (nb = 2) whose banks' optimal readers share a NOC row (banks 0 / 4 on row 9 of these chips)
            # streams both banks' responses over that one row's links: its readers ran ~45% longer than the other groups'
            # (M2.7 shapes, 4 rows: 103 vs 60-72 us per core) and set the op's time. QWEN35_MOE_GU_OFFROW: such a group's
            # readers read on NOC1 (Y first), 2: spread over the rows free of bank readers, so the two banks' responses
            # spread over the rows (M2.7, union 23: gate|up 100.4 -> 76.7 us, bit-identical); 1: all on the next row
            # inward (no better: the responses converge on that row instead); 0 (default; untested on the other models'
            # shapes): on the bank row (the old placement)
            self.offrow = set()
            used = set()
            placed = {}
            mode = os.environ.get("QWEN35_MOE_GU_OFFROW", "0")
            # QWEN35_MOE_GU_OFFROW1=1: one-bank groups (nb = 1) whose bank's reader row is an earlier bank's (banks 0 / 4)
            # on the next row inward, reading on NOC1 (the down's DN_OFFROW placement)
            if self.nb == 1 and os.environ.get("QWEN35_MOE_GU_OFFROW1", "0") == "1":
                placed, self.offrow = place_banks_offrow(mesh_device, {q: readers for q in range(self.ngroups)}, used=used)
            if self.nb == 2 and mode in ("1", "2"):
                workers = bank_workers(mesh_device)
                grid = mesh_device.compute_with_storage_grid_size()
                bank_rows = {w.y for w in workers}
                for q in range(self.ngroups):
                    w, w2 = workers[q], workers[q + self.ngroups]
                    if w.y != w2.y:
                        continue
                    if mode == "1":  # all on the next row inward
                        rows = [w.y - 1 if w.y > 0 else w.y + 1] * readers
                    else:  # one reader per free row (no bank's reader row), nearest first: the responses spread
                        free = sorted((y for y in range(grid.y) if y not in bank_rows), key=lambda y: abs(y - w.y))
                        if not free:
                            continue
                        rows = [free[i % len(free)] for i in range(readers)]  # (more readers than free rows: reuse)
                    # QWEN35_MOE_GU_OFFROW_X=east: those readers by the group's second (east) bank's reader column instead
                    # of beside the other groups' (west) readers (one die, banks 0 / 4: that group ended 6-10 us after
                    # the others; gate|up 103.3 / 236.9 -> 96.6 / 232.1 us at 1 / 4 rows)
                    wx = w2.x if os.environ.get("QWEN35_MOE_GU_OFFROW_X") == "east" else w.x
                    cores = []
                    for y in rows:
                        for d in range(grid.x):
                            hit = next(
                                (
                                    wx + dx
                                    for dx in ((d, -d) if d else (0,))
                                    if 0 <= wx + dx < grid.x and (wx + dx, y) not in used
                                ),
                                None,
                            )
                            if hit is not None:
                                used.add((hit, y))
                                cores.append(ttnn.CoreCoord(hit, y))
                                break
                    assert len(cores) == readers, f"no room for {readers} cores off bank {q}'s row"
                    placed[q] = cores
                    self.offrow.add(q)
            placed.update(
                place_along_rows(mesh_device, {q: readers for q in range(self.ngroups) if q not in placed}, used=used, rows=True)
            )
            self.assign = [(c, q, r) for q in range(self.ngroups) for r, c in enumerate(placed[q])]
        self.rem = rem
        self.pairs_total = pairs_total or self.pairs + rem
        assert self.pairs_total >= self.pairs + rem
        self.rem_assign = []
        if rem:
            assert not self.split and not ep_experts and not shared and self.kt % self.banks == 0
            # QWEN35_MOE_REM_READERS: cores per remainder pair (slots s % readers); off the bank readers' rows first
            rr = int(os.environ.get("QWEN35_MOE_REM_READERS", "8"))
            grid = mesh_device.compute_with_storage_grid_size()
            used = {(c.x, c.y) for c, _, _ in self.assign}
            busy_rows = {c.y for c, _, _ in self.assign}
            cand = sorted(((x, y) for y in range(grid.y) for x in range(grid.x) if (x, y) not in used),
                          key=lambda c: (c[1] in busy_rows, abs(c[0] - grid.x // 2), abs(c[1] - grid.y // 2)))
            assert len(cand) >= rem * rr, "no room for the remainder pairs' cores"
            self.rem_readers = rr
            self.rem_assign = [(ttnn.CoreCoord(*cand[p * rr + j]), p, j) for p in range(rem) for j in range(rr)]
            self.rem_buffers = gu_ring(self.kt, 2, 1, self.tile_bytes, buffers, even=True)
            # QWEN35_MOE_REM_KCH: the remainder blocks pushed as this many chunks of banks (K-row chunks)
            self.rem_kch = _kch(os.environ.get("QWEN35_MOE_REM_KCH", "1"), self.kt, self.rem_buffers,
                                banks=self.banks)
        self.cores = _range_set([c for c, _, _ in self.assign])

    def __call__(self, x, table, identity, up_gate, shared_up_gate, act, ep_offset=None, rem_w=None):
        cb_x, cb_w, cb_scr, cb_id, cb_sc, cb_tab, cb_wt, cb_act = 0, 1, 2, 3, 4, 5, 6, 16
        tb = 64 * self.trows
        g = self.groups
        buffers = self.buffers
        np_ = len(self.group_pairs[0])
        per_core = -(-self.max_slots // self.readers)
        cbs = [
            _cb(self.cores, cb_x, ttnn.bfloat16, tb, g * self.kt, tiny=True, rows=self.trows),
            _cb(self.cores, cb_w, self.weight_dtype, self.tile_bytes, buffers * self.kt * self.m),
            _cb(self.cores, cb_scr, ttnn.uint16, 64 * self.rows + 128, 1),  # + the EP offset (64-byte aligned)
            _cb(self.cores, cb_id, self.weight_dtype, self.tile_bytes, 2),
            # two slots' weight tiles (split-pair: SPLIT_RING slots, the up cores' ring for their partners' g')
            _cb(
                self.cores,
                cb_sc,
                ttnn.bfloat16,
                tb,
                2 * g * (self.ring if self.split else 2),
                tiny=True,
                rows=self.trows,
            ),
            _cb(self.cores, cb_tab, ttnn.uint16, 64 * self.rows + 128, 1),
            _cb(self.cores, cb_wt, ttnn.uint16, ((2 * per_core * self.rows + 15) // 16) * 16, 1),
            _cb(self.cores, cb_act, ttnn.bfloat16, tb, 2 * g * np_, tiny=True, rows=self.trows),
        ]
        assert (ep_offset is not None) == bool(self.ep_experts)
        ep_arg = [ep_offset.buffer_address()] if ep_offset is not None else []
        rd_rt, wr_rt = {}, {}
        for c, j, r in self.assign:
            s0 = r if self.shared else r + 1  # first union slot (no shared expert: slot 0 is never streamed)
            rd_rt[(c.x, c.y)] = [
                table.buffer_address(),
                up_gate.buffer_address(),
                shared_up_gate.buffer_address(),
                identity.buffer_address(),
                j,
                s0,
                (j + r) & 0x3,
            ] + ep_arg
            wr_rt[(c.x, c.y)] = [x.buffer_address(), act.buffer_address(), table.buffer_address(), j, s0] + ep_arg
        rd_ct = [cb_w, cb_scr, cb_id, self.kt, self.n_tiles, self.k, self.rows, self.readers, self.tile_bytes]
        rd_ct += [self.page_bytes, self.banks, buffers]
        rd_ct += (
            ttnn.TensorAccessorArgs(table).get_compile_time_args()
            + ttnn.TensorAccessorArgs(identity).get_compile_time_args()
        )
        wr_ct = [
            cb_x,
            cb_act,
            cb_sc,
            cb_tab,
            self.kt,
            self.k,
            self.rows,
            self.readers,
            self.pairs_total,
            self.trows,
            g,
            cb_wt,
            ACT_PAGE,
        ]
        wr_ct += [self.m, self.banks]
        for t in (x, act, table):
            wr_ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        cp_ct = [cb_x, cb_w, cb_act, cb_sc, cb_id, self.kt, g, self.m, self.banks]
        defines = [("ROWS", str(self.trows))] + _ep_defines(self.ep_experts, self.gate_first)
        defines += [("MOE_SUBS", str(self.subs))] if self.subs > 1 else []
        kch = [("GU_KCH", str(self.kch))] if self.kch > 1 else []
        # QWEN35_MOE_ZONES=1: device-profiler points in the rows kernels (kernels/moe_zones.hpp)
        defines += [("MOE_ZONES", "1")] if os.environ.get("QWEN35_MOE_ZONES") == "1" else []
        semaphores = []
        if self.split:
            half = self.banks // 2
            kernels = []
            for gate, noc1 in ((False, False), (True, False), (False, True), (True, True)):
                sel = [(c, b, r) for c, b, r in self.assign if (b >= half) == gate and (b in self.noc1_banks) == noc1]
                if not sel:
                    continue
                keys = {(c.x, c.y) for c, _, _ in sel}
                rd_g = {k: v for k, v in rd_rt.items() if k in keys}
                wr_g = {}
                for c, b, r in sel:
                    pc = self.mesh.worker_core_from_logical_core(self.partner[(b, r)])
                    args = list(wr_rt[(c.x, c.y)])
                    args[3] = b % half  # the pair (up column) index
                    wr_g[(c.x, c.y)] = args + [pc.x, pc.y]
                for c, b, r in sel:
                    rd_g[(c.x, c.y)][4] = b  # the reader's bank
                cores = _range_set([c for c, _, _ in sel])
                d = (
                    defines
                    + [("SPLIT_PAIR", "1"), ("SPLIT_RING", str(self.ring))]
                    + ([("ROLE_GATE", "1")] if gate else [])
                )
                kernels += _kernels(
                    cores,
                    "gur",
                    (rd_ct, rd_g),
                    (wr_ct, wr_g),
                    (cp_ct, None),
                    d,
                    dst_full=self.dst_full,
                    reader_noc=ttnn.NOC.NOC_1 if noc1 else None,
                )
            semaphores = [ttnn.SemaphoreDescriptor(i, ttnn.CoreType.WORKER, self.cores, 0) for i in (0, 1)]
        else:
            kernels = []
            for noc1 in (False, True):
                sel = [c for c, q, _ in self.assign if (q in self.offrow) == noc1]
                if not sel:
                    continue
                keys = {(c.x, c.y) for c in sel}
                kernels += _kernels(
                    _range_set(sel),
                    "gur",
                    (rd_ct, {k: v for k, v in rd_rt.items() if k in keys}),
                    (wr_ct, {k: v for k, v in wr_rt.items() if k in keys}),
                    (cp_ct, None),
                    defines + kch,
                    dst_full=self.dst_full,
                    reader_noc=ttnn.NOC.NOC_1 if noc1 else None,
                )
        io = [up_gate, shared_up_gate, identity, x, table, act] + ([ep_offset] if ep_offset is not None else [])
        if self.rem:
            assert rem_w is not None
            kernels, cbs = kernels + self._rem_kernels(x, table, identity, act, rem_w, defines, wr_ct), cbs + self._rem_cbs
            io.append(rem_w)
        program = ttnn.ProgramDescriptor(kernels=kernels, semaphores=semaphores, cbs=cbs)
        return ttnn.generic_op(io, program)

    def _rem_kernels(self, x, table, identity, act, rem_w, defines, wr_ct_main):
        """The remainder pairs' cores: the gur kernels at m = 2 (one pair), the reader gathering the pair's K chunks from
        every bank (GU_REM), the writer at pair index inter / 32 + p (+ the zero pairs up to pairs_total)."""
        cb_x, cb_w, cb_scr, cb_id, cb_sc, cb_tab, cb_wt, cb_act = 0, 1, 2, 3, 4, 5, 6, 16
        tb = 64 * self.trows
        g = self.groups
        rr = self.rem_readers
        cores = _range_set([c for c, _, _ in self.rem_assign])
        per_core = -(-self.max_slots // rr)
        self._rem_cbs = [
            _cb(cores, cb_x, ttnn.bfloat16, tb, g * self.kt, tiny=True, rows=self.trows),
            _cb(cores, cb_w, self.weight_dtype, self.tile_bytes, self.rem_buffers * self.kt * 2),
            _cb(cores, cb_scr, ttnn.uint16, 64 * self.rows + 128, 1),
            _cb(cores, cb_id, self.weight_dtype, self.tile_bytes, 2),
            _cb(cores, cb_sc, ttnn.bfloat16, tb, 4 * g, tiny=True, rows=self.trows),
            _cb(cores, cb_tab, ttnn.uint16, 64 * self.rows + 128, 1),
            _cb(cores, cb_wt, ttnn.uint16, ((2 * per_core * self.rows + 15) // 16) * 16, 1),
            _cb(cores, cb_act, ttnn.bfloat16, tb, 2 * g, tiny=True, rows=self.trows),
        ]
        rd_rt, wr_rt = {}, {}
        for c, p, j in self.rem_assign:
            rd_rt[(c.x, c.y)] = [table.buffer_address(), rem_w.buffer_address(), rem_w.buffer_address(),
                                 identity.buffer_address(), p, j + 1, (p + j) & 0x3]
            wr_rt[(c.x, c.y)] = [x.buffer_address(), act.buffer_address(), table.buffer_address(), p, j + 1]
        rd_ct = [cb_w, cb_scr, cb_id, self.kt, 2 * self.banks, self.k, self.rows, rr, self.tile_bytes]
        rd_ct += [_page_bytes(self.kt // self.banks * 2, self.tile_bytes), self.banks, self.rem_buffers]
        rd_ct += ttnn.TensorAccessorArgs(table).get_compile_time_args() + ttnn.TensorAccessorArgs(identity).get_compile_time_args()
        wr_ct = list(wr_ct_main)
        wr_ct[7] = rr  # step
        wr_ct[13] = 2  # m
        cp_ct = [cb_x, cb_w, cb_act, cb_sc, cb_id, self.kt, g, 2, self.banks]
        d = list(defines) + [("GU_KCH", str(self.rem_kch))] * (self.rem_kch > 1) + [
                             ("GU_REM", str(self.pairs)), ("GU_REM_PAIRS", str(self.rem)),
                             ("GU_REM_PAD", str(self.pairs_total - self.pairs - self.rem))]
        return _kernels(cores, "gur", (rd_ct, rd_rt), (wr_ct, wr_rt), (cp_ct, None), d, dst_full=False)


class ExpertDownRows:
    """(act, table, down, shared_down, out) -> out. readers: cores per DRAM bank; chunk: union slots per weight
    block (the cb holds two)."""

    def __init__(
        self,
        mesh_device,
        hidden,
        inter,
        num_experts,
        k,
        weight_dtype,
        rows,
        readers=None,
        chunk=8,
        shared=True,
        ep_experts=None,
        subs=1,
        split=None,
        pad=0,
    ):
        """shared False: routed experts only (shared_down is not read). ep_experts: as ExpertDown. split: K tiles of the
        first of two accumulation chains per slot, each packed to BF16 and the two added (two sub-shards' numerics).
        pad: the last `pad` K tiles of every slot are zero weights (an even K): not read, zeroed in L1 once."""
        self.pad = pad
        assert not ep_experts or not shared
        self.shared, self.ep_experts = shared, ep_experts
        self.banks = len(bank_workers(mesh_device))
        self.kt = inter // TILE
        self.m = hidden // TILE // self.banks
        self.trows, self.groups = tile_rows(rows), row_groups(rows)
        # > 1 row group: full-sync DST (8 FP32 tiles; the kernel packs once, at the end): groups x cols <= 8, so fewer
        # cores each read the whole activation block
        self.dst_full = self.groups > 1
        # QWEN35_MOE_DN_READERS / QWEN35_MOE_DN_CHUNK: per-model overrides (MiniMax-M2.7, 4 rows, H 3072, 384 per die:
        # 4 readers per bank with 4-slot chunks 137 vs 157 us per MoE call, hybrid_moe/bench_moe_rows.py; 6 readers with
        # 2-slot chunks: the down op 48.8 vs 57.5 us at a 23-expert union, 12 readers do not fit near the banks)
        # QWEN35_MOE_DN_READERS_WIDE: the reader count for > 2 row groups (> 16 rows), where the DST budget allows fewer
        # columns per core (Inkling-2-Mini: hidden 2304 = 9 columns per bank -> 9 readers, one column each, at 32 rows)
        if readers is None and self.groups > 2 and os.environ.get("QWEN35_MOE_DN_READERS_WIDE"):
            readers = int(os.environ["QWEN35_MOE_DN_READERS_WIDE"])
        if readers is None and os.environ.get("QWEN35_MOE_DN_READERS"):
            readers = int(os.environ["QWEN35_MOE_DN_READERS"])
        if os.environ.get("QWEN35_MOE_DN_CHUNK"):
            chunk = int(os.environ["QWEN35_MOE_DN_CHUNK"])
        if readers is None:  # 2 per bank (4 / 8 measured slower at 8 rows: 33 vs 24 us)
            readers = max(2 if self.m % 2 == 0 else 1, self.m // max(1, (8 if self.dst_full else 4) // self.groups))
        assert self.m * self.banks * TILE == hidden and self.m % readers == 0
        # (one row group only: two chains of > 1 group would need fewer columns per core, ~10 % slower at 16 rows)
        self.split = split if self.groups == 1 else None
        if self.split:  # two chains of cols DST tiles: <= 8
            self.dst_full = True
            while 2 * self.groups * (self.m // readers) > 8:
                readers = next(r for r in range(readers + 1, self.m + 1) if self.m % r == 0)
        self.cols = self.m // readers
        assert self.groups * self.cols <= (8 if self.dst_full else 4), (self.groups, self.cols)
        self.tile_bytes = TILE_BYTES[weight_dtype]
        # two chunks of [chunk * K, cols] weight tiles in cb_w: <= RING_BYTES (GLM-4.5-Air: hidden 4096, 8 slots = 835 KB)
        chunk = max(2, min(chunk, RING_BYTES // (2 * self.kt * self.cols * self.tile_bytes)))
        self.E, self.k, self.inter, self.rows, self.chunk = num_experts, k, inter, rows, chunk
        self.subs = subs  # (as ExpertGateUpRows: the slots of every sub-shard summed in one program)
        assert subs == 1 or (not ep_experts and not shared)
        self.max_slots = min(num_experts * subs, rows * k * subs) + 1
        self.weight_dtype = weight_dtype
        # QWEN35_MOE_DN_OFFROW=1: a bank whose optimal reader row is an earlier bank's (banks 0 / 4 on row 9 here) gets its
        # readers on the next row inward, reading on NOC1 (place_banks_offrow, as DecodeLinear): on the shared row the
        # two banks' readers ran 44 / 49 vs 34-37 us per core (M2.7 shapes, 6 readers per bank, 4 rows). 0 (default;
        # untested on the other models' shapes): every bank on its row
        self.noc1_banks = set()
        if os.environ.get("QWEN35_MOE_DN_OFFROW", "0") == "1":
            placed, self.noc1_banks = place_banks_offrow(mesh_device, {b: readers for b in range(self.banks)})
        else:
            placed = place_along_rows(mesh_device, {b: readers for b in range(self.banks)}, rows=True)
        self.assign = [(c, b, r) for b in range(self.banks) for r, c in enumerate(placed[b])]
        self.cores = _range_set([c for c, _, _ in self.assign])

    def __call__(self, act, table, down, shared_down, out, ep_offset=None):
        assert (ep_offset is not None) == bool(self.ep_experts)
        ep_arg = [ep_offset.buffer_address()] if ep_offset is not None else []
        cb_in0, cb_w, cb_scr, cb_tab, cb_out = 0, 1, 3, 5, 16
        tb = 64 * self.trows
        cbs = [
            _cb(
                self.cores,
                cb_in0,
                ttnn.bfloat16,
                tb,
                2 * self.groups * self.chunk * self.kt,
                tiny=True,
                rows=self.trows,
            ),
            _cb(self.cores, cb_w, self.weight_dtype, self.tile_bytes, 2 * self.chunk * self.kt * self.cols),
            _cb(self.cores, cb_scr, ttnn.uint16, 64 * self.rows + 128, 1),
            _cb(self.cores, cb_tab, ttnn.uint16, 64 * self.rows + 128, 1),
            _cb(self.cores, cb_out, ttnn.bfloat16, tb, self.groups * self.cols, tiny=True, rows=self.trows),
        ]
        if self.split:
            cbs.append(_cb(self.cores, 6, ttnn.bfloat16, tb, 2 * self.groups * self.cols, tiny=True, rows=self.trows))
        rd_rt, wr_rt = {}, {}
        for c, b, r in self.assign:
            c0 = r * self.cols
            rd_rt[(c.x, c.y)] = [
                down.buffer_address(),
                shared_down.buffer_address(),
                table.buffer_address(),
                b,
                c0,
                (b + r) & 0x3,
            ] + ep_arg
            wr_rt[(c.x, c.y)] = [act.buffer_address(), out.buffer_address(), table.buffer_address(), b, c0] + ep_arg
        rd_ct = [cb_w, cb_scr, self.kt, self.m, self.cols, self.k, self.tile_bytes, self.rows, self.chunk]
        rd_ct += ttnn.TensorAccessorArgs(table).get_compile_time_args()
        wr_ct = [
            cb_in0,
            cb_out,
            cb_tab,
            self.kt,
            self.cols,
            self.banks,
            self.k,
            self.rows,
            self.trows,
            self.groups,
            self.chunk,
            ACT_PAGE,
        ]
        for t in (act, out, table):
            wr_ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        cp_ct = [cb_in0, cb_w, cb_out, self.kt, self.cols, self.chunk, self.groups]
        defines = [("ROWS", str(self.trows))] + ([("DST_FULL", "1")] if self.dst_full else [])
        defines += [("MOE_SUBS", str(self.subs))] if self.subs > 1 else []
        defines += [("DN_SPLIT", str(self.split)), ("DN_TMP", "6")] if self.split else []
        defines += [("DN_PAD", str(self.pad))] if self.pad else []
        # QWEN35_MOE_DN_SLOT=1 (one row group): weight slots pushed and multiplied one by one as they land (bitwise)
        defines += [("DN_SLOT", "1")] if os.environ.get("QWEN35_MOE_DN_SLOT") == "1" and self.groups == 1 else []
        defines += [("MOE_ZONES", "1")] if os.environ.get("QWEN35_MOE_ZONES") == "1" else []
        if not self.shared:
            defines.append(("NO_SHARED", "1"))
        defines += _ep_defines(self.ep_experts)
        kernels = []
        for noc1 in (False, True):
            sel = [c for c, b, _ in self.assign if (b in self.noc1_banks) == noc1]
            if not sel:
                continue
            keys = {(c.x, c.y) for c in sel}
            kernels += _kernels(
                _range_set(sel),
                "dnr",
                (rd_ct, {k: v for k, v in rd_rt.items() if k in keys}),
                (wr_ct, {k: v for k, v in wr_rt.items() if k in keys}),
                (cp_ct, None),
                defines,
                dst_full=self.dst_full,
                reader_noc=ttnn.NOC.NOC_1 if noc1 else None,
            )
        program = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
        io = [down, shared_down, act, table, out] + ([ep_offset] if ep_offset is not None else [])
        return ttnn.generic_op(io, program)
