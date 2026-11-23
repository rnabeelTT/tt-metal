# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Fused router post-processing of the decode MoE (tt/moe.py router_row): one generic_op instead of add + sigmoid + add
+ concat + slice. a, b: the hi / lo halves of the streamed router's FP32 logits [1, 1, 32, E]; bias_rows: the
selection bias [1, 1, 32, E] FP32 (row-replicated) -> row [1, 1, T, 2E] FP32 = [s + bias | s], s = sigmoid(a + b).
The same SFPU arithmetic as the ttnn ops (FP32 add, sigmoid_tile<RC, accurate>, FP32 add). One output column tile of
each half per core. kernels/rp_*.cpp."""

from __future__ import annotations

from pathlib import Path

import ttnn

KDIR = Path(__file__).parent / "kernels"
TILE = 32


class RouterPostOp:
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

    def __call__(self, a, b, bias_rows, T, mem=None, route=None, wexp=0, groups=None):
        """route: a route table (UINT16 row-major L1, one REC_BYTES page per row, e.g. the qwen36 fused MoE's one-row
        route buffer or RowsBuffers.table): also take each real row's top-k + weights here (RP_ROUTE, rp_writer.cpp);
        wexp: the weights x 2^wexp (exact; the routed scale folded in). groups (route only; default HY4_RP_GROUPS,
        "2", capped at T and the grid): that many copies of the 16 key-tile cores, group g taking the rows r = g mod
        groups (the per-row top-k on the data-movement cores is serial: ~1-2 us a row)."""
        E = int(a.shape[-1])
        n = E // TILE
        nc = min(n, 16)
        per = -(-n // nc)
        nc = -(-n // per)
        G = 1
        if route is not None:
            import os

            G = groups or int(os.environ.get("HY4_RP_GROUPS", "2"))
            G = max(1, min(G, T, len(self.grid) // nc))
        cores = self.grid[:nc * G]
        crs = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])
        y = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, T, 2 * E]), ttnn.float32, ttnn.TILE_LAYOUT, self.mesh,
                                           mem or ttnn.L1_MEMORY_CONFIG)

        def cb(i, k):
            fmt = ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.float32, page_size=4096)
            return ttnn.CBDescriptor(total_size=4096 * k, core_ranges=crs, format_descriptors=[fmt])

        cbs = [cb(0, 2), cb(1, 2), cb(2, 2), cb(16, 2), cb(17, 2)]
        if route is not None:
            assert per == 1 and E == 512, (E, per)  # one key tile per core; top-16 route records (ROUTE_WIDE)
            lb = 3 * 4 * 16
            tg = -(-T // G)  # rows per group (max)
            scr = (-(-tg // nc)) * nc * lb + tg * lb + 128
            cbs.append(ttnn.CBDescriptor(total_size=scr, core_ranges=crs, format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=3, data_format=ttnn.uint32, page_size=scr)]))
        acc = lambda t: ttnn.TensorAccessorArgs(t).get_compile_time_args()  # noqa: E731
        rng = [(k * per, min(n, (k + 1) * per)) for k in range(nc)] * G

        def rt(args):
            r = ttnn.RuntimeArgs()
            for c, x in zip(cores, args):
                r[c.x][c.y] = x
            return r

        cfg = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True,
                                           math_approx_mode=False)
        modes = [ttnn.UnpackToDestMode.Default] * 64
        modes[0] = modes[1] = modes[2] = ttnn.UnpackToDestMode.UnpackToDestFp32
        cfg.unpack_to_dest_mode = modes
        dm = ttnn.DataMovementConfigDescriptor
        wr_ct = [n] + acc(y)
        wr_args = [[y.buffer_address(), s, e] for s, e in rng]
        wr_def, sems = [], []
        if route is not None:
            wr_ct += acc(route)
            xy = [self.mesh.worker_core_from_logical_core(c) for c in cores]
            own = [p.x | (p.y << 8) for p in xy]
            wr_ct += own  # (compile time: per-core runtime args cost dispatch time per core)
            wr_args = [a_ + [route.buffer_address(), i % nc, i // nc] for i, a_ in enumerate(wr_args)]
            wr_def = [("RP_ROUTE", "1"), ("RP_ROWS", str(T)), ("RP_GROUPS", str(G)), ("ROUTE_WIDE", "1"),
                      ("ROUTER_SIGB", "1"),
                      ("ROUTER_KEYS", "1")] + ([("RP_NLIVE", "1")] if T == 1 else [])
            if wexp:
                wr_def.append(("RP_WEXP", str(int(wexp))))
            sems = [ttnn.SemaphoreDescriptor(0, ttnn.CoreType.WORKER, crs, 0)]
        kernels = [
            ttnn.KernelDescriptor(kernel_source=str(KDIR / "rp_reader.cpp"), core_ranges=crs,
                                  compile_time_args=acc(a) + acc(b) + acc(bias_rows),
                                  runtime_args=rt([[a.buffer_address(), b.buffer_address(), bias_rows.buffer_address(),
                                                    s, e] for s, e in rng]),
                                  config=dm(processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.NOC_1)),
            ttnn.KernelDescriptor(kernel_source=str(KDIR / "rp_writer.cpp"), core_ranges=crs,
                                  compile_time_args=wr_ct, runtime_args=rt(wr_args), defines=wr_def,
                                  config=dm(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_0)),
            ttnn.KernelDescriptor(kernel_source=str(KDIR / "rp_compute.cpp"), core_ranges=crs,
                                  compile_time_args=[], runtime_args=rt([[e - s] for s, e in rng]), config=cfg),
        ]
        io = [a, b, bias_rows, y] + ([route] if route is not None else [])
        ttnn.generic_op(io, ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems, cbs=cbs))
        return y
