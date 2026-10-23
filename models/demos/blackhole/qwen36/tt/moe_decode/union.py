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

import torch

import ttnn

from .experts import (
    KERNEL_DIR, TILE, TILE_BYTES, _cb, _kernels, _page_bytes, _range_set, bank_workers, gu_layout, gu_ring, place_along_rows,
)

TILE_ROWS = (1, 2, 4, 8)  # custom_mm activation tile heights


def tile_rows(m):
    """[rows, 32] activation tile height for m decode rows (> 8 rows: groups of 8)."""
    return next(r for r in TILE_ROWS if r >= min(m, 8))


def row_groups(m):
    return -(-m // tile_rows(m))


ACT_PAGE = 64 * 8  # one [8, 32] BF16 tile: every width's [Mp, 32] tile fits a page


def act_buffer(mesh_device, inter, num_experts, k, rows):
    """The activations of a `rows`-row step (one [Mp, 32] tile per ACT_PAGE page: union slot, K tile, row group), L1
    interleaved: every down core reads all of it, which from DRAM cost more than the weights at 32 rows (2.7 MB x 64
    cores per device on 2 chips). Allocated per step on the device (ttnn.empty: no host upload, trace-safe) and freed
    after the down op: a persistent buffer of the 32-row size (2.7 MB per device on 2 chips) left too little L1 for the
    model's other programs' circular buffers. Every page the down op reads is written by the gate|up op first."""
    slots = min(num_experts, rows * k) + 1
    return ttnn.empty(
        [slots * inter // TILE * row_groups(rows), ACT_PAGE // 2], ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, mesh_device,
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
        self.table = ttnn.from_torch(torch.zeros(rows, 32, dtype=torch.int32), dtype=ttnn.uint16, layout=ttnn.ROW_MAJOR_LAYOUT, **l1)
        self.act = act
        self.out = ttnn.from_torch(
            torch.zeros(1, 1, rows, hidden, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
            device=mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG, **rep,
        )
        self.identity = ttnn.from_torch(
            torch.eye(TILE)[None, None], dtype=weight_dtype, layout=ttnn.TILE_LAYOUT, device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG, **rep,
        )


class RouteRows:
    """(logits [M, E + 32], table) -> table, one core per row (NCRISC and BRISC each take half the logits)."""

    def __init__(self, mesh_device, num_experts, k, rows):
        self.E, self.k, self.rows = num_experts, k, rows
        gx = mesh_device.compute_with_storage_grid_size().x
        self.assign = [ttnn.CoreCoord(r % gx, r // gx) for r in range(rows)]
        self.cores = _range_set(self.assign)

    def __call__(self, logits, table):
        cb_scr, cb_part, cb_lg = 0, 1, 2
        lg_bytes = 64 * (self.E // TILE + 1)
        cbs = [
            _cb(self.cores, cb_scr, ttnn.uint16, lg_bytes + 64, 1),
            _cb(self.cores, cb_part, ttnn.uint32, 8 * self.k, 1),
            _cb(self.cores, cb_lg, ttnn.bfloat16, lg_bytes, 1),
        ]
        rd_rt, wr_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for r, c in enumerate(self.assign):
            rd_rt[c.x][c.y] = [logits.buffer_address(), table.buffer_address(), r]
            wr_rt[c.x][c.y] = [logits.buffer_address(), r]
        lg_ct = ttnn.TensorAccessorArgs(logits).get_compile_time_args()
        rd_ct = [cb_scr, cb_part, self.E, self.k] + lg_ct + ttnn.TensorAccessorArgs(table).get_compile_time_args()
        wr_ct = [cb_lg, cb_part, self.E, self.k] + lg_ct
        dm = ttnn.DataMovementConfigDescriptor
        kernels = [
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "route_rows_reader.cpp"), core_ranges=self.cores,
                compile_time_args=rd_ct, runtime_args=rd_rt,
                config=dm(processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.NOC_0),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "route_rows_writer.cpp"), core_ranges=self.cores,
                compile_time_args=wr_ct, runtime_args=wr_rt,
                config=dm(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_1),
            ),
        ]
        ttnn.generic_op([logits, table], ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))
        return table


class ExpertGateUpRows:
    """(x, table, identity, up_gate, shared_up_gate, act) -> act. readers: cores per column group (experts.gu_layout);
    union slot s goes to core s % readers."""

    def __init__(self, mesh_device, hidden, inter, num_experts, k, weight_dtype, rows, readers=None, buffers=5):
        """readers None: 4 cores per column pair. At 8 rows a core spends ~2.6 us per union slot (unpack 1.7 +
        SFPU 0.7, measured), yet 6-22 cores per pair measured slower: the extra cores sit off the banks' rows and
        their reads collapse (8 rows, 4-chip shapes: 51 us median at 4 per pair, 89 at 22, 217 max at 8)."""
        self.banks = len(bank_workers(mesh_device))
        self.kt = hidden // TILE
        self.n_tiles = 2 * inter // TILE
        self.pairs = self.n_tiles // 2
        self.m, self.nb, self.ngroups, self.group_pairs = gu_layout(self.n_tiles, self.banks)
        self.E, self.k, self.inter, self.rows = num_experts, k, inter, rows
        self.trows, self.groups = tile_rows(rows), row_groups(rows)
        self.max_slots = min(num_experts, rows * k) + 1
        if readers is None:
            readers = 4
        self.readers = readers
        self.tile_bytes = TILE_BYTES[weight_dtype]
        self.weight_dtype = weight_dtype
        # nb = 2 (odd m: a slot is two consecutive ring blocks the compute indexes together): an even ring
        self.buffers = gu_ring(self.kt, self.m, self.nb, self.tile_bytes, buffers, even=True)
        self.page_bytes = _page_bytes(self.kt * self.m, self.tile_bytes)
        self.dst_full = self.m * self.nb + 1 > 4  # the matmul tiles + the weight tile; FP32 DST: 4 half-sync, 8 full
        assert self.m * self.nb + 1 <= 8, f"{self.m} up|gate columns per bank: more than 8 DST tiles"
        placed = place_along_rows(mesh_device, {q: readers for q in range(self.ngroups)})
        self.assign = [(c, q, r) for q in range(self.ngroups) for r, c in enumerate(placed[q])]
        self.cores = _range_set([c for c, _, _ in self.assign])

    def __call__(self, x, table, identity, up_gate, shared_up_gate, act):
        cb_x, cb_w, cb_scr, cb_id, cb_sc, cb_tab, cb_wt, cb_act = 0, 1, 2, 3, 4, 5, 6, 16
        tb = 64 * self.trows
        g = self.groups
        buffers = self.buffers
        np_ = len(self.group_pairs[0])
        per_core = -(-self.max_slots // self.readers)
        cbs = [
            _cb(self.cores, cb_x, ttnn.bfloat16, tb, g * self.kt, tiny=True, rows=self.trows),
            _cb(self.cores, cb_w, self.weight_dtype, self.tile_bytes, buffers * self.kt * self.m),
            _cb(self.cores, cb_scr, ttnn.uint16, 64 * self.rows, 1),
            _cb(self.cores, cb_id, self.weight_dtype, self.tile_bytes, 2),
            _cb(self.cores, cb_sc, ttnn.bfloat16, tb, 2 * 2 * g, tiny=True, rows=self.trows),  # two slots' weight tiles
            _cb(self.cores, cb_tab, ttnn.uint16, 64 * self.rows, 1),
            _cb(self.cores, cb_wt, ttnn.uint16, ((2 * per_core * self.rows + 15) // 16) * 16, 1),
            _cb(self.cores, cb_act, ttnn.bfloat16, tb, 2 * g * np_, tiny=True, rows=self.trows),
        ]
        rd_rt, wr_rt = {}, {}
        for c, j, r in self.assign:
            rd_rt[(c.x, c.y)] = [table.buffer_address(), up_gate.buffer_address(), shared_up_gate.buffer_address(),
                                 identity.buffer_address(), j, r, (j + r) & 0x3]
            wr_rt[(c.x, c.y)] = [x.buffer_address(), act.buffer_address(), table.buffer_address(), j, r]
        rd_ct = [cb_w, cb_scr, cb_id, self.kt, self.n_tiles, self.k, self.rows, self.readers, self.tile_bytes]
        rd_ct += [self.page_bytes, self.banks, buffers]
        rd_ct += ttnn.TensorAccessorArgs(table).get_compile_time_args() + ttnn.TensorAccessorArgs(identity).get_compile_time_args()
        wr_ct = [cb_x, cb_act, cb_sc, cb_tab, self.kt, self.k, self.rows, self.readers, self.pairs, self.trows, g, cb_wt, ACT_PAGE]
        wr_ct += [self.m, self.banks]
        for t in (x, act, table):
            wr_ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        cp_ct = [cb_x, cb_w, cb_act, cb_sc, cb_id, self.kt, g, self.m, self.banks]
        defines = [("ROWS", str(self.trows))]
        kernels = _kernels(self.cores, "gur", (rd_ct, rd_rt), (wr_ct, wr_rt), (cp_ct, None), defines, dst_full=self.dst_full)
        program = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
        return ttnn.generic_op([up_gate, shared_up_gate, identity, x, table, act], program)


class ExpertDownRows:
    """(act, table, down, shared_down, out) -> out. readers: cores per DRAM bank; chunk: union slots per weight
    block (the cb holds two)."""

    def __init__(self, mesh_device, hidden, inter, num_experts, k, weight_dtype, rows, readers=None, chunk=8):
        self.banks = len(bank_workers(mesh_device))
        self.kt = inter // TILE
        self.m = hidden // TILE // self.banks
        self.trows, self.groups = tile_rows(rows), row_groups(rows)
        # > 1 row group: full-sync DST (8 FP32 tiles; the kernel packs once, at the end): groups x cols <= 8, so fewer
        # cores each read the whole activation block
        self.dst_full = self.groups > 1
        if readers is None:  # 2 per bank (4 / 8 measured slower at 8 rows: 33 vs 24 us)
            readers = max(2 if self.m % 2 == 0 else 1, self.m // max(1, (8 if self.dst_full else 4) // self.groups))
        assert self.m * self.banks * TILE == hidden and self.m % readers == 0
        self.cols = self.m // readers
        assert self.groups * self.cols <= (8 if self.dst_full else 4), (self.groups, self.cols)
        self.E, self.k, self.inter, self.rows, self.chunk = num_experts, k, inter, rows, chunk
        self.max_slots = min(num_experts, rows * k) + 1
        self.tile_bytes = TILE_BYTES[weight_dtype]
        self.weight_dtype = weight_dtype
        placed = place_along_rows(mesh_device, {b: readers for b in range(self.banks)})
        self.assign = [(c, b, r) for b in range(self.banks) for r, c in enumerate(placed[b])]
        self.cores = _range_set([c for c, _, _ in self.assign])

    def __call__(self, act, table, down, shared_down, out):
        cb_in0, cb_w, cb_scr, cb_tab, cb_out = 0, 1, 3, 5, 16
        tb = 64 * self.trows
        cbs = [
            _cb(self.cores, cb_in0, ttnn.bfloat16, tb, 2 * self.groups * self.chunk * self.kt, tiny=True, rows=self.trows),
            _cb(self.cores, cb_w, self.weight_dtype, self.tile_bytes, 2 * self.chunk * self.kt * self.cols),
            _cb(self.cores, cb_scr, ttnn.uint16, 64 * self.rows, 1),
            _cb(self.cores, cb_tab, ttnn.uint16, 64 * self.rows, 1),
            _cb(self.cores, cb_out, ttnn.bfloat16, tb, self.groups * self.cols, tiny=True, rows=self.trows),
        ]
        rd_rt, wr_rt = {}, {}
        for c, b, r in self.assign:
            c0 = r * self.cols
            rd_rt[(c.x, c.y)] = [down.buffer_address(), shared_down.buffer_address(), table.buffer_address(), b, c0, (b + r) & 0x3]
            wr_rt[(c.x, c.y)] = [act.buffer_address(), out.buffer_address(), table.buffer_address(), b, c0]
        rd_ct = [cb_w, cb_scr, self.kt, self.m, self.cols, self.k, self.tile_bytes, self.rows, self.chunk]
        rd_ct += ttnn.TensorAccessorArgs(table).get_compile_time_args()
        wr_ct = [cb_in0, cb_out, cb_tab, self.kt, self.cols, self.banks, self.k, self.rows, self.trows, self.groups, self.chunk, ACT_PAGE]
        for t in (act, out, table):
            wr_ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        cp_ct = [cb_in0, cb_w, cb_out, self.kt, self.cols, self.chunk, self.groups]
        defines = [("ROWS", str(self.trows))] + ([("DST_FULL", "1")] if self.dst_full else [])
        kernels = _kernels(self.cores, "dnr", (rd_ct, rd_rt), (wr_ct, wr_rt), (cp_ct, None), defines, dst_full=self.dst_full)
        program = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
        return ttnn.generic_op([down, shared_down, act, table, out], program)
