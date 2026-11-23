# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Gather-matmul of the expert-sorted prefill MoE (tt/prefill_moe.py, HY4_PF_GATHER_MM): the gate|up grouped sparse
matmul reading its input rows by index (tt-metal PR 61548 row 35) instead of moe_gather_rows writing the [G, 32, L]
copy (~114 MB per die per layer at 2K rows) that the matmul reads back.

One generic_op = the grouped factory's program (sparse_matmul_multicore_reuse_grouped.cpp: resident in1, padding groups
skipped) with its in0 reader replaced: kernels/gfm_reader.cpp reads each real group's 32 rows x_rm[src_tok[32 g + r]]
(row-major, as moe_gather_rows) and the compute kernel (kernels/gfm_compute.cpp = bmm_grouped_resident_in1.cpp +
IN0_TILIZE) fast-tilizes them into cb_in0 before the group's K-blocks. The in1 reader / output writer is the factory's
own kernel. Same in0 tiles (a bf16 tilize is a copy), same K-blocking, subblocks, fidelity and DEST mode as the
sparse_matmul call it replaces: bit-identical output (gm_bench 2026-11-23: every real group, 64 / 512 / 2048 rows)."""

from __future__ import annotations

from pathlib import Path

import ttnn

KDIR = Path(__file__).parent / "kernels"
W_KERNEL = "ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_writer_bmm_tile_layout_in1_indexed.cpp"
TILE_BYTES = {ttnn.bfloat4_b: 576, ttnn.bfloat8_b: 1088, ttnn.bfloat16: 2048}
DRAM_ALIGN = 64


def _align(n: int, a: int = DRAM_ALIGN) -> int:
    return -(-n // a) * a


class GatherMM:
    _OPS = {}

    @classmethod
    def get(cls, mesh):
        if id(mesh) not in cls._OPS:
            cls._OPS[id(mesh)] = cls(mesh)
        return cls._OPS[id(mesh)]

    def __init__(self, mesh):
        self.mesh = mesh
        g = mesh.compute_with_storage_grid_size()
        self.grid = [ttnn.CoreCoord(x, y) for y in range(g.y) for x in range(g.x)]  # row-wise, as the factory

    @staticmethod
    def supports(pc, w, x_cols: int) -> bool:
        """The program config / weights this op reproduces (the grouped factory's resident-in1 path, one tile row per
        group, no untilize_out)."""
        if w.dtype not in TILE_BYTES or x_cols % 32:
            return False
        if pc.per_core_M != 1 or pc.out_subblock_h != 1 or getattr(pc, "untilize_out", False):
            return False
        return (x_cols // 32) % pc.in0_block_w == 0

    def __call__(self, x_rm, src_tok, group_ids, w, pc, ckc, num_experts: int, mem=None):
        """x_rm [T, K] bf16 ROW_MAJOR, src_tok [1, Rd] uint32 ROW_MAJOR, group_ids [1, G] uint16 ROW_MAJOR, w [1, E, K, N]
        (tiled, interleaved) -> [1, G, 32, N] bf16 tiled (real groups only written, as sparse_matmul)."""
        K = int(x_rm.shape[-1])
        G = int(group_ids.shape[-1])
        E = int(num_experts)
        N = int(w.shape[-1])
        Kt, Nt, Mt = K // 32, N // 32, 1
        ib, sw, sh = int(pc.in0_block_w), int(pc.out_subblock_w), int(pc.out_subblock_h)
        assert int(pc.per_core_N) == Nt and Kt % ib == 0 and Nt % sw == 0
        nblk = Kt // ib
        wtb = TILE_BYTES[w.dtype]
        in1_blk = Nt * ib
        out = ttnn.allocate_tensor_on_device(ttnn.Shape([1, G, 32, N]), ttnn.bfloat16, ttnn.TILE_LAYOUT, self.mesh,
                                             mem or ttnn.DRAM_MEMORY_CONFIG)
        # consecutive whole groups per core (the factory's split; the in1 writer's same-expert runs need consecutive)
        nc = min(len(self.grid), G)
        base, rem = divmod(G, nc)
        cores = self.grid[:nc]
        starts, counts, s = [], [], 0
        for i in range(nc):
            n = base + (1 if i < rem else 0)
            starts.append(s)
            counts.append(n)
            s += n
        crs = lambda cs: ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cs])  # noqa: E731
        all_crs = crs(cores)
        idx_pg = _align(G * 2)
        src_pg = _align(int(src_tok.shape[-1]) * 4)

        def cb(i, pages, page, fmt, extra=None):
            fds = [ttnn.CBFormatDescriptor(buffer_index=i, data_format=fmt, page_size=page)]
            if extra is not None:
                fds.append(ttnn.CBFormatDescriptor(buffer_index=extra, data_format=fmt, page_size=page))
            return ttnn.CBDescriptor(total_size=pages * page, core_ranges=all_crs, format_descriptors=fds)

        cbs = [
            cb(0, Kt, 2048, ttnn.bfloat16),            # cb_in0: a group's tilized K tiles
            cb(1, nblk * in1_blk, wtb, w.dtype),       # cb_in1: an expert's whole [K, N] block (resident)
            cb(4, Mt * Nt, 2048, ttnn.bfloat16, 5),    # cb_out / cb_intermed0 (bf16 DEST, no L1 acc: one buffer)
            cb(6, 1, 128, ttnn.uint32),                # reader: a group's 32 source-row ids
            cb(7, 1, idx_pg, ttnn.uint16),             # writer: the group id stick
            cb(8, 1, idx_pg, ttnn.uint16),             # reader: its copy
            cb(9, 2 * Kt, 2048, ttnn.bfloat16),        # cb_in0_rm: 2 groups of 32 gathered rows
        ]
        acc = lambda t: ttnn.TensorAccessorArgs(t).get_compile_time_args()  # noqa: E731

        def rt(args, cs=cores):
            r = ttnn.RuntimeArgs()
            for c, a in zip(cs, args):
                r[c.x][c.y] = a
            return r

        dm = ttnn.DataMovementConfigDescriptor
        reader = ttnn.KernelDescriptor(
            kernel_source=str(KDIR / "gfm_reader.cpp"), core_ranges=all_crs,
            compile_time_args=[E, K * 2, Kt, idx_pg, src_pg, 9, 6, 8] + acc(x_rm) + acc(src_tok) + acc(group_ids),
            runtime_args=rt([[x_rm.buffer_address(), src_tok.buffer_address(), group_ids.buffer_address(), s0, n]
                             for s0, n in zip(starts, counts)]),
            config=dm(processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.NOC_1))
        # the factory's in1 reader / output writer, its compile-time args (sparse_matmul_multicore_reuse_grouped.cpp)
        n_sub = Nt // sw
        w_ct = [1, Nt, ib * Nt, Nt, ib, in1_blk, nblk, 0, Kt * Nt, 1, Nt, sw, sh * Nt, sw, sh, sw * sh, n_sub, Mt // sh,
                Mt * Nt, idx_pg, G, E] + acc(w) + acc(out) + acc(group_ids)
        writer = ttnn.KernelDescriptor(
            kernel_source=W_KERNEL, core_ranges=all_crs, compile_time_args=w_ct,
            named_compile_time_args=[("cb_in1", 1), ("cb_out", 4), ("cb_idx", 7)],
            defines=[("SKIP_PADDING_GROUPS", "1"), ("RESIDENT_IN1", "1")],
            runtime_args=rt([[w.buffer_address(), n, out.buffer_address(), s0 * Mt * Nt, group_ids.buffer_address(), s0]
                             for s0, n in zip(starts, counts)]),
            config=dm(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_0))
        cfg = ttnn.ComputeConfigDescriptor(math_fidelity=ckc.math_fidelity, fp32_dest_acc_en=False,
                                           math_approx_mode=bool(ckc.math_approx_mode))
        named = [("cb_in0", 0), ("cb_in1", 1), ("cb_bias", 3), ("cb_out", 4), ("cb_intermed0", 5),
                 ("cb_in0_transposed", 10), ("cb_in0_rm", 9)]
        computes = []
        for n_here in sorted(set(counts), reverse=True):
            cs = [c for c, n in zip(cores, counts) if n == n_here]
            c_ct = [ib, Mt // sh, Mt * ib, sh * ib, n_sub, in1_blk, sw * n_sub, nblk, 1, 1, sh, sw, sh * sw, n_here,
                    Mt * Nt, 0, 1, 0]
            computes.append(ttnn.KernelDescriptor(
                kernel_source=str(KDIR / "gfm_compute.cpp"), core_ranges=crs(cs), compile_time_args=c_ct,
                named_compile_time_args=named, defines=[("FUSE_ACTIVATION", "0"), ("IN0_TILIZE", "1")], config=cfg))
        ttnn.generic_op([x_rm, src_tok, group_ids, w, out],
                        ttnn.ProgramDescriptor(kernels=[reader, writer] + computes, semaphores=[], cbs=cbs))
        return out
