# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The sorted prefill MoE's un-dispatch over all sub-shards in one program (tt/kernels/undisp_*.cpp): S row-major down
outputs [Rd, H] bf16 and the dispatch's undispatch stick [1, T * k] uint32 -> [1, 1, T, H] bf16 TILE. Each sub-shard's
k rows summed in slot order (ttnn.moe_gather_add_rows), the sub-shard sums in order (the ttnn.add calls): bitwise."""

from __future__ import annotations

import hashlib

import ttnn

from .bconv import KDIR, _cb, _range_set, _rt
from .bconv import _ks

L1_BUDGET = 700_000


PCACHE = __import__("os").environ.get("I2M_PCACHE", "1") == "1"  # (see tt/gmm.py)


class Undispatch:
    def __init__(self, mesh):
        self.mesh = mesh
        g = mesh.compute_with_storage_grid_size()
        self.gx, self.ncores = g.x, g.x * g.y
        self._st = {}

    def _static(self, T, k, S, H):
        Wt, RT = H // 32, (T + 31) // 32
        best = None
        for cs in range(1, Wt + 1):
            if Wt % cs or 8 * (Wt // cs) * 2048 > L1_BUDGET:
                continue
            items = RT * cs
            cost = -(-items // self.ncores) * (Wt // cs + 4)  # per-core tiles + a per-item overhead
            if best is None or cost < best[0]:
                best = (cost, cs)
        cs = best[1]
        Wt_c, items = Wt // cs, RT * cs
        n = min(self.ncores, items)
        bounds = [items * i // n for i in range(n + 1)]
        cores = [ttnn.CoreCoord(i % self.gx, i // self.gx) for i in range(n)]
        return dict(Wt=Wt, Wt_c=Wt_c, cs=cs, cores=cores, crs=_range_set(cores),
                    rt=[[bounds[i], bounds[i + 1] - bounds[i]] for i in range(n)])

    def __call__(self, rows, und, T, k):
        S, H = len(rows), int(rows[0].shape[-1])
        key = (T, k, S, H)
        st = self._st.get(key)
        if st is None:
            st = self._st[key] = self._static(T, k, S, H)
        out = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, T, H]), ttnn.bfloat16, ttnn.TILE_LAYOUT, self.mesh,
                                             ttnn.DRAM_MEMORY_CONFIG)
        pk = (key, repr(rows[0].memory_config()), repr(und.memory_config()))
        hit = st.get(pk) if PCACHE else None
        if hit is None:
            hit = self._desc(st, key, rows, und, out, T, k, S, H)
            if PCACHE:
                st[pk] = hit
        rd, wr, cp, cbs, h = hit
        rd.common_runtime_args = [und.buffer_address()] + [r.buffer_address() for r in rows]
        wr.common_runtime_args = [out.buffer_address()]
        prog = ttnn.ProgramDescriptor(kernels=[rd, wr, cp], semaphores=[], cbs=cbs)
        prog.custom_program_hash = h
        ttnn.generic_op(list(rows) + [und, out], prog)
        return out

    def _desc(self, st, key, rows, und, out, T, k, S, H):
        """__call__'s kernels (common runtime args set per call), CBs and hash"""
        Wt, Wt_c, cs, cores, crs = st["Wt"], st["Wt_c"], st["cs"], st["cores"], st["crs"]
        bf = ttnn.bfloat16
        cb_in, cb_til, cb_aa, cb_ab, cb_ta, cb_tb, cb_out, cb_und = range(8)
        cbs = [_cb(crs, cb_in, bf, 2048, 2 * Wt_c)] + [_cb(crs, i, bf, 2048, Wt_c) for i in range(1, 7)]
        und_bytes = (32 * k * 4 + 63) // 64 * 64
        cbs.append(_cb(crs, cb_und, ttnn.uint32, und_bytes, 1))
        dm = ttnn.DataMovementConfigDescriptor
        rd = ttnn.KernelDescriptor(
            kernel_source=_ks(KDIR, "undisp_reader.cpp"), core_ranges=crs,
            compile_time_args=[T, k, S, Wt_c, cs, Wt_c * 64, H * 2, cb_in, cb_und]
            + ttnn.TensorAccessorArgs(rows[0]).get_compile_time_args() + ttnn.TensorAccessorArgs(und).get_compile_time_args(),
            runtime_args=_rt(cores, st["rt"]), config=dm(processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.NOC_0))
        rd.common_runtime_args = [und.buffer_address()] + [r.buffer_address() for r in rows]
        wr = ttnn.KernelDescriptor(
            kernel_source=_ks(KDIR, "undisp_writer.cpp"), core_ranges=crs,
            compile_time_args=[Wt, Wt_c, cs, cb_out] + ttnn.TensorAccessorArgs(out).get_compile_time_args(),
            runtime_args=_rt(cores, st["rt"]), config=dm(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_1))
        wr.common_runtime_args = [out.buffer_address()]
        cfg = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=False)
        cp = ttnn.KernelDescriptor(
            kernel_source=_ks(KDIR, "undisp_compute.cpp"), core_ranges=crs,
            compile_time_args=[Wt_c, k, S, cb_in, cb_til, cb_aa, cb_ab, cb_ta, cb_tb, cb_out],
            runtime_args=_rt(cores, [[r[1]] for r in st["rt"]]), config=cfg)
        hk = ("i2m_undisp", key, tuple(repr(t.memory_config()) for t in (rows[0], und)))
        return rd, wr, cp, cbs, int.from_bytes(hashlib.sha1(repr(hk).encode()).digest()[:8], "little") & ((1 << 63) - 1)
