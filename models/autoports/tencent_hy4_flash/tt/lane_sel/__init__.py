# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Batched (lanes) decode selection programs (tt/decode_batch HY4_IDX_LANEX, tt/indexer step_lanes):

compact: scores [1, 1, 32, Ws] bf16 RM (indexer_score_dsa over every lane's group keys) -> [1, 1, 32, Wc] bf16 RM,
    row r = its own lane's visible scores sc[r, lo_r : lo_r + vis_r], then -inf (rows that do not select: all -inf).
    Replaces the full-width mask (slice, tilize, typecast, iota lt / ge / or, where, typecast, untilize) and lets the
    top-512 search Wc (one lane, valid length = the largest vis) instead of every lane's columns.
expand: top-512 lane-relative group ids [1, 1, 32, 512] uint32 RM -> index lists [1, 1, 32, K] uint32 RM: selecting rows
    [4 id + j + base_r (511 groups) | tail_r], the other rows their dense list (replaces to_layout, typecast, slice,
    expand x3, add delta, concat, multiply nmask, add dense, typecast, to_layout).
score: the indexer score of each row against its own lane's visible keys only, straight into compact()'s format
    (replaces indexer_score_dsa over every lane's keys + compact; bit-identical scores). kernels/ls_*.cpp.
lmeta [1, 1, 32, 16] uint32 RM: per row [lane group base lo, visible groups vis, cache base, selects]. compact /
expand: integer-exact, one core per row, the two RISCs split the row (kernels/lane_*.cpp).
Selection vs the full-width path (HY4_IDX_LANEX_RUN=0): the same scores, but top-512 runs over the compact rows (valid
length = the largest vis) instead of every lane's columns, so tied scores at the 512 boundary and the list order can
come out differently (tie order only)."""

from __future__ import annotations

from pathlib import Path

import ttnn

KDIR = Path(__file__).parent / "kernels"


class LaneSelOp:
    _OPS = {}

    @classmethod
    def get(cls, mesh):
        if id(mesh) not in cls._OPS:
            cls._OPS[id(mesh)] = cls(mesh)
        return cls._OPS[id(mesh)]

    def __init__(self, mesh):
        self.mesh = mesh
        g = mesh.compute_with_storage_grid_size()
        self.grid = [ttnn.CoreCoord(x, y) for y in range(g.y) for x in range(g.x)]

    def _run(self, name, ios, ct, rt_of, cbsz, rows=32):
        cores = self.grid[:rows]
        crs = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])
        cbsz = -(-cbsz // 64) * 64
        cbs = [ttnn.CBDescriptor(total_size=cbsz, core_ranges=crs, format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.uint32, page_size=cbsz)]) for i in (0, 1)]
        dm = ttnn.DataMovementConfigDescriptor
        kernels = []
        for part, (proc, noc) in enumerate(((ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.NOC_0),
                                            (ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.NOC_1))):
            a = ttnn.RuntimeArgs()
            for row, c in enumerate(cores):
                a[c.x][c.y] = rt_of(row, part)
            kernels.append(ttnn.KernelDescriptor(kernel_source=str(KDIR / name), core_ranges=crs, compile_time_args=ct,
                                                 runtime_args=a, config=dm(processor=proc, noc=noc)))
        ttnn.generic_op(ios, ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))

    def compact(self, sc, lmeta, Wc):
        assert sc.dtype == ttnn.bfloat16 and sc.layout == ttnn.ROW_MAJOR_LAYOUT and Wc % 64 == 0
        rows, Ws = int(sc.shape[-2]), int(sc.shape[-1])
        out = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, rows, Wc]), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT,
                                             self.mesh, ttnn.DRAM_MEMORY_CONFIG)
        acc = lambda t: ttnn.TensorAccessorArgs(t).get_compile_time_args()  # noqa: E731
        ct = [Ws, Wc] + acc(sc) + acc(lmeta) + acc(out)
        half = Wc // 2
        self._run("lane_compact.cpp", [sc, lmeta, out], ct,
                  lambda row, part: [sc.buffer_address(), lmeta.buffer_address(), out.buffer_address(), row,
                                     part * half, (part + 1) * half], half * 2 + 256 + 64, rows)
        return out

    def score(self, q, gk, w, lmeta, Wc):
        """Lane-aware indexer score straight into compact()'s format: q [1, 16, 32, 128] bf16 TILE, gk the lanes' group
        keys [1, 1, L gcap + 64, 128] bf16 TILE (lane l's groups at lo = l gcap), w [1, 1, 32, 16] bf16 TILE -> [1, 1,
        32, Wc] bf16 RM, row r = sum_h w[r, h] relu(q[h, r] . gk[lo_r + c]) for c < vis_r, -inf after, up to the
        largest selecting vis rounded up to a tile (top-k's valid length stops there; the rest is not written). Work:
        only each row's visible key tiles, 16 heads x 2 rows packed per matmul (rows 2 h + (r & 1), read straight from
        q's tiles; indexer_score_dsa over every lane's
        keys: 32 rows x L gcap per head). Bit-identical scores (kernels/ls_compute.cpp)."""
        assert q.dtype == ttnn.bfloat16 and q.layout == ttnn.TILE_LAYOUT and tuple(q.shape) == (1, 16, 32, 128)
        assert w.dtype == ttnn.bfloat16 and w.layout == ttnn.TILE_LAYOUT and tuple(w.shape)[-2:] == (32, 16)
        assert gk.dtype == ttnn.bfloat16 and gk.layout == ttnn.TILE_LAYOUT and int(gk.shape[-1]) == 128
        assert self.score_fits(Wc), Wc
        cores = self.grid
        NC = len(cores)
        crs = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])
        out = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, 32, Wc]), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT,
                                             self.mesh, ttnn.DRAM_MEMORY_CONFIG)

        def cb(i, n, page=2048, fmt=ttnn.bfloat16):
            return ttnn.CBDescriptor(total_size=n * page, core_ranges=crs, format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=i, data_format=fmt, page_size=page)])

        # k (2 key tiles in flight), q (2 sub-runs), w bcast, P, T_h (2 units), acc, compute descriptor, scratch x2
        cbs = [cb(0, 8), cb(1, 8), cb(2, 16), cb(3, 2), cb(4, 32), cb(5, 2), cb(6, 1, fmt=ttnn.uint32),
               cb(7, 1, 4096), cb(8, 1, 4096 + 64)]
        acc = lambda t: ttnn.TensorAccessorArgs(t).get_compile_time_args()  # noqa: E731
        dm = ttnn.DataMovementConfigDescriptor
        # addresses as common runtime args (one multicast); each core derives its index from its logical coordinates
        defs = [("LS_GX", str(self.mesh.compute_with_storage_grid_size().x))]
        # indexer_score_dsa's decode configuration: HiFi2 (bf16 operands), bf16 DEST, half sync
        cfg = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=False,
                                           math_approx_mode=False, dst_full_sync_en=False)
        kernels = [
            ttnn.KernelDescriptor(kernel_source=str(KDIR / "ls_reader.cpp"), core_ranges=crs, defines=defs,
                                  compile_time_args=[NC] + acc(q) + acc(w) + acc(gk) + acc(lmeta),
                                  common_runtime_args=[q.buffer_address(), w.buffer_address(), gk.buffer_address(),
                                                       lmeta.buffer_address()],
                                  config=dm(processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.NOC_1)),
            ttnn.KernelDescriptor(kernel_source=str(KDIR / "ls_writer.cpp"), core_ranges=crs, defines=defs,
                                  compile_time_args=[NC] + acc(lmeta) + acc(out),
                                  common_runtime_args=[lmeta.buffer_address(), out.buffer_address()],
                                  config=dm(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_0)),
            ttnn.KernelDescriptor(kernel_source=str(KDIR / "ls_compute.cpp"), core_ranges=crs, config=cfg),
        ]
        ttnn.generic_op([q, w, gk, lmeta, out], ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))
        return out

    def score_fits(self, Wc):
        """A core's compute units (<= 32 sub-runs x Wc / 32 key tiles over the grid) fit its 2 KB descriptor page."""
        return 32 * (Wc // 32) // len(self.grid) + 2 < 500

    def expand(self, ids, lmeta, tail, dense, r=4, n_sel=511):
        rows, K = int(ids.shape[-2]), int(ids.shape[-1])
        tw, KD = int(tail.shape[-1]), int(dense.shape[-1])
        assert KD == n_sel * r + tw
        idx = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, rows, KD]), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT,
                                             self.mesh, ttnn.DRAM_MEMORY_CONFIG)
        acc = lambda t: ttnn.TensorAccessorArgs(t).get_compile_time_args()  # noqa: E731
        ct = [K, n_sel, r, tw, KD] + acc(ids) + acc(lmeta) + acc(tail) + acc(dense) + acc(idx)
        half = (n_sel + 1) // 2
        self._run("lane_expand.cpp", [ids, lmeta, tail, dense, idx], ct,
                  lambda row, part: [ids.buffer_address(), lmeta.buffer_address(), tail.buffer_address(),
                                     dense.buffer_address(), idx.buffer_address(), row,
                                     0 if part == 0 else half, half if part == 0 else n_sel, part],
                  K * 4 + KD * 4 + 256 + 64, rows)
        return idx
