# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Decode attention glue as data-movement programs (HY4_ATTN_GLUE=1), for passes of R <= 16 real rows:

qprep: [q_lat | q_rope | q_sink] (TILE, L1) -> the persistent row-major sparse_sdpa query [1, Hp, 32, 608] (DRAM),
    only the real rows of the real heads written (one program instead of concat + concat(head pad) + to_layout). The
    buffer is zeroed once: padding heads / rows stay zero (finite; padding rows attend the sink only).
otile: the sparse_sdpa output [1, Hp, 32, V] RM -> [1, Hd, 32, V] TILE, real rows placed, others zero (one program
    instead of slice + to_layout). Same values: pure data movement."""

from __future__ import annotations

from pathlib import Path

import torch
import ttnn

KDIR = Path(__file__).parent / "kernels"
TILE = 32


class AttnGlue:
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
        self.qbuf = {}

    def qbuffer(self, Hp, W):
        """The persistent zeroed query buffer (create it outside any trace capture: the eager pass)."""
        key = (Hp, W)
        if key not in self.qbuf:
            rep = {"mesh_mapper": ttnn.ReplicateTensorToMesh(self.mesh)} if self.mesh.get_num_devices() > 1 else {}
            self.qbuf[key] = ttnn.from_torch(torch.zeros(1, Hp, TILE, W, dtype=torch.bfloat16), dtype=ttnn.bfloat16,
                                             layout=ttnn.ROW_MAJOR_LAYOUT, device=self.mesh,
                                             memory_config=ttnn.DRAM_MEMORY_CONFIG, **rep)
        return self.qbuf[key]

    @staticmethod
    def _rt(cores, args):
        r = ttnn.RuntimeArgs()
        for c, a in zip(cores, args):
            r[c.x][c.y] = a
        return r

    def qprep(self, q_lat, q_rope, q_sink, Hp, R):
        Hd, LT, RT = int(q_lat.shape[1]), int(q_lat.shape[-1]) // TILE, int(q_rope.shape[-1]) // TILE
        W = int(q_lat.shape[-1]) + int(q_rope.shape[-1]) + int(q_sink.shape[-1])
        out = self.qbuffer(Hp, W)
        cores = self.grid[:Hd]
        crs = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])
        fmt = ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.bfloat16, page_size=W * 2)
        cbs = [ttnn.CBDescriptor(total_size=W * 2, core_ranges=crs, format_descriptors=[fmt])]
        acc = lambda t: ttnn.TensorAccessorArgs(t).get_compile_time_args()  # noqa: E731
        ct = [LT, RT, W * 2, TILE, 0] + acc(q_lat) + acc(q_rope) + acc(q_sink) + acc(out)
        rt = [[q_lat.buffer_address(), q_rope.buffer_address(), q_sink.buffer_address(), out.buffer_address(), h, R]
              for h in range(Hd)]
        k = ttnn.KernelDescriptor(kernel_source=str(KDIR / "qprep.cpp"), core_ranges=crs, compile_time_args=ct,
                                  runtime_args=self._rt(cores, rt),
                                  config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0,
                                                                           noc=ttnn.NOC.NOC_0))
        ttnn.generic_op([q_lat, q_rope, q_sink, out], ttnn.ProgramDescriptor(kernels=[k], semaphores=[], cbs=cbs))
        return out

    def otile(self, o, Hd, R, mem):
        V = int(o.shape[-1])
        VT = V // TILE
        per_head = 2 if VT % 2 == 0 else 1
        NT = VT // per_head
        n = Hd * per_head
        cores = self.grid[:n]
        crs = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])
        out = ttnn.allocate_tensor_on_device(ttnn.Shape([1, Hd, TILE, V]), ttnn.bfloat16, ttnn.TILE_LAYOUT, self.mesh,
                                             mem)
        cbs = [ttnn.CBDescriptor(total_size=NT * 2048, core_ranges=crs, format_descriptors=[
                   ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.bfloat16, page_size=2048)]),
               ttnn.CBDescriptor(total_size=NT * 64, core_ranges=crs, format_descriptors=[
                   ttnn.CBFormatDescriptor(buffer_index=1, data_format=ttnn.bfloat16, page_size=NT * 64)])]
        acc = lambda t: ttnn.TensorAccessorArgs(t).get_compile_time_args()  # noqa: E731
        ct = [NT, VT, int(o.shape[-2]), 0, 1] + acc(o) + acc(out)
        rt = [[o.buffer_address(), out.buffer_address(), k // per_head, (k % per_head) * NT, R] for k in range(n)]
        kd = ttnn.KernelDescriptor(kernel_source=str(KDIR / "otile.cpp"), core_ranges=crs, compile_time_args=ct,
                                   runtime_args=self._rt(cores, rt),
                                   config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0,
                                                                            noc=ttnn.NOC.NOC_0))
        ttnn.generic_op([o, out], ttnn.ProgramDescriptor(kernels=[kd], semaphores=[], cbs=cbs))
        return out
