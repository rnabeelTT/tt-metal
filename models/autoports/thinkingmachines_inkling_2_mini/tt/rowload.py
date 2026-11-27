# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Many row loads into fp32 TILE tensors in one program (tt/kernels/rowload.cpp).

A job (src, j, dst, b) copies row j of a ROW_MAJOR fp32 DRAM tensor [.., R, C] into row b of an fp32 TILE DRAM tensor
[.., B, C]. Used at a prefill's end (BatchedDecoder.load_slot, I2M_ROWLOAD): every application's lane conv states into
the decode rows of slot b. Was a to_layout + 3 x (slice / slice / slice / concat / copy) per conv state: ~3,000 ops,
67 ms per request on 4 dies (served p300x2, I2M_PREFILL_PROF). Bitwise (a copy). The runtime-argument layout is fixed
(MAXJ job slots per core) so one program serves every call: compiled at warm-up (load_slot of every slot), before any
trace capture."""

from __future__ import annotations

import hashlib
import os

import ttnn

from .bconv import KDIR, _cb, _range_set, _rt
from .bconv import _ks

ON = os.environ.get("I2M_ROWLOAD", "1") == "1"
NCORES = 64
MAXJ = 24  # job slots per core (6 runtime args each)
BUF = 16384  # L1 staging bytes per core (one source row: C <= 4096 fp32)


class RowLoad:
    def __init__(self, mesh, proto):
        """proto: any DRAM interleaved tensor (the accessor arguments are the same for all of them)."""
        self.mesh = mesh
        grid = mesh.compute_with_storage_grid_size()
        self.cores = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(min(NCORES, grid.x * grid.y))]
        self.crs = _range_set(self.cores)
        self.ct = ttnn.TensorAccessorArgs(proto).get_compile_time_args()
        self.proto = proto

    def __call__(self, jobs):
        """jobs: [(src ROW_MAJOR fp32, row j, dst TILE fp32, row b)] (DRAM interleaved, same C)."""
        work = []
        for src, j, dst, b in jobs:
            for t in (src, dst):
                mc = t.memory_config()
                assert mc.buffer_type == ttnn.BufferType.DRAM and not t.is_sharded(), f"rowload: {mc} (DRAM only)"
                assert t.dtype == ttnn.float32, t.dtype
            assert src.layout == ttnn.ROW_MAJOR_LAYOUT and dst.layout == ttnn.TILE_LAYOUT
            C = int(dst.padded_shape[-1])
            assert C % 32 == 0 and int(src.padded_shape[-1]) == C and C * 4 <= BUF, (C, src.padded_shape)
            assert 0 <= b < int(dst.padded_shape[-2]), (b, dst.padded_shape)
            work.append((src.buffer_address(), int(j), C * 4, dst.buffer_address(), int(b), C // 32))
        if not work:
            return
        nc = len(self.cores)
        for i0 in range(0, len(work), nc * MAXJ):  # (more jobs than slots: several launches of the same program)
            chunk = work[i0:i0 + nc * MAXJ]
            per = [[] for _ in range(nc)]
            for i, w in enumerate(chunk):
                per[i % nc].append(w)
            rt = []
            for lst in per:
                a = [len(lst)]
                for w in lst:
                    a += list(w)
                rt.append(a + [0] * (1 + 6 * MAXJ - len(a)))
            k = ttnn.KernelDescriptor(
                kernel_source=_ks(KDIR, "rowload.cpp"), core_ranges=self.crs, compile_time_args=[0] + self.ct,
                runtime_args=_rt(self.cores, rt),
                config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0,
                                                         noc=ttnn.NOC.NOC_0))
            prog = ttnn.ProgramDescriptor(kernels=[k], semaphores=[], cbs=[_cb(self.crs, 0, ttnn.float32, 4096, BUF // 4096)])
            key = ("i2m_rowload", nc, MAXJ, BUF)
            prog.custom_program_hash = int.from_bytes(hashlib.sha1(repr(key).encode()).digest()[:8], "little") & ((1 << 63) - 1)
            ttnn.generic_op([self.proto, self.proto], prog)
