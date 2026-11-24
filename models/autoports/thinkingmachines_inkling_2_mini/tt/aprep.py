# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Decode attention prep in one program (tt/kernels/aprep_*.cpp), B <= 32 users, one row each:

    pq [1, 1, B, q | r | sink | gate] fp32, k / v [1, 1, B, NKV * HD] bf16 (conv outputs)  ->
    q  [1, B, NH, HD]  bf16   l2-clipped (qk_clip) (* tau_b on global layers)
    r  [1, B, NH, 32]  bf16   l2-clipped (rel_clip) (* tau_b), DR (16) columns + zero padding (the SDPA's rq)
    k  [1, B, 32, HD]  bf16   l2-clipped, padded to 32 head rows, in the KV-write layout (height-sharded)
    v  [1, B, 32, HD]  bf16   unweighted RMS
    sk [1, B, NH, 32]  bf16   sink / scale in every column

Replaces ~35 ttnn ops per application (slices, reshapes, typecasts, the clip / rms chains, tau, pads, the sink
broadcast, the KV reshards). Units (user, tensor) are spread over the cores."""

from __future__ import annotations

import hashlib
import struct

import ttnn

from .bconv import KDIR, _cb, _range_set, _rt

F32, BF = 4096, 2048
# I2M_APREP_QG (default 1): decode q / k / v heads in groups (4 or 8 heads per core, as the grid allows: units q groups,
# r, k groups, v groups, sink per user); the per-head math is row-local, so the outputs are unchanged
QG = __import__("os").environ.get("I2M_APREP_QG", "1") == "1"


def _bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", float(x)))[0]


class AttnPrep:
    def __init__(self, w, B: int, kv_mc):
        """B: decode users (<= 32), or the rows of a prefill chunk (> 32: call with fill=; q / r / sink in DRAM)."""
        assert w.DR in (16, 32) and w.NH < 32
        self.w, self.B, self.kv_mc, self.mesh = w, B, kv_mc, w.mesh
        grid = self.mesh.compute_with_storage_grid_size()
        nu = lambda hg, kg: -(-w.NH // hg) + 2 * -(-w.NKV // kg) + 2  # noqa: E731
        # (heads per unit; a group never crosses a 16-row face)
        opts = [(4, 4), (8, 4), (8, 8)] if QG and B <= 32 else []
        self.HG, self.KG = next(((hg, kg) for hg, kg in opts if nu(hg, kg) * B <= grid.x * grid.y), (w.NH, w.NKV))
        self.QN, self.KN = -(-w.NH // self.HG), -(-w.NKV // self.KG)
        self.NU = nu(self.HG, self.KG)
        self.total = self.NU * B
        nc = min(self.total, grid.x * grid.y)
        self.cores = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(nc)]
        self.crs = _range_set(self.cores)
        self.HT = w.HD // 32

    def __call__(self, pq, k, v, tau=None, kv=None, fill=None):
        """kv = (k_cache, v_cache, positions int32 [B], block table int32 [B, max blocks]): the k / v rows written into
        the paged bf16 caches here (no k / v outputs: None). fill = (k_cache, v_cache, fill table int32 [1, B / 64]):
        a prefill chunk's rows written into the caches (row b -> block fill[b / 64], row b % 64)."""
        w, B, HT, mesh = self.w, self.B, self.HT, self.mesh
        L1 = ttnn.L1_MEMORY_CONFIG
        kf = vf = None
        if fill is not None:  # (the rows go to L1 [1, NKV, T, HD] tensors; paged_fill_cache moves them below)
            mk = lambda: ttnn.allocate_tensor_on_device(ttnn.Shape([1, w.NKV, B, w.HD]), ttnn.bfloat16,  # noqa: E731
                                                        ttnn.TILE_LAYOUT, mesh, L1)
            kf, vf = mk(), mk()
            kv = (kf, vf, fill[2], fill[2])
        prefill = B > 32
        k = k if (not k.is_sharded() and k.memory_config().buffer_type == ttnn.BufferType.L1) else \
            ttnn.to_memory_config(k, L1)
        v = v if (not v.is_sharded() and v.memory_config().buffer_type == ttnn.BufferType.L1) else \
            ttnn.to_memory_config(v, L1)
        glob = not w.sliding
        al = lambda shape, mc: ttnn.allocate_tensor_on_device(ttnn.Shape(shape), ttnn.bfloat16,  # noqa: E731
                                                               ttnn.TILE_LAYOUT, mesh, mc)
        omc = ttnn.DRAM_MEMORY_CONFIG if prefill else L1
        q_o = al([1, B, w.NH, w.HD], omc)
        r_o = al([1, B, w.NH, 32], omc)
        if kv is not None and prefill:  # (no k / v outputs: placeholders for the unused writer arguments)
            k_o = v_o = q_o
        else:
            k_o = al([1, B, 32, w.HD], self.kv_mc)
            v_o = al([1, B, 32, w.HD], self.kv_mc)
        s_o = al([1, B, w.NH, 32], omc)
        tau_t = tau if glob else pq  # (unused on sliding layers)
        KG = self.KG if kv is not None else w.NKV  # (k / v head groups only when the rows go to the caches here)
        KN = -(-w.NKV // KG)
        total = (self.QN + 2 * KN + 2) * B
        cb_xf, cb_xb, cb_tau, cb_ones, cb_sq, cb_f, cb_out = 0, 1, 2, 3, 4, 5, 6
        f32, bf = ttnn.float32, ttnn.bfloat16
        cbs = [_cb(self.crs, cb_xf, f32, F32, 2 * HT), _cb(self.crs, cb_xb, bf, BF, 2 * HT),
               _cb(self.crs, cb_tau, f32, F32, 2), _cb(self.crs, cb_ones, bf, BF, 1),
               _cb(self.crs, cb_sq, f32, F32, 2 * HT), _cb(self.crs, cb_f, f32, F32, 2),
               _cb(self.crs, cb_out, bf, BF, 2 * HT)]
        nc = len(self.cores)
        R0 = w.qd // 32
        S0 = (w.qd + w.rd) // 32
        rd_ct = [cb_xf, cb_xb, cb_tau, cb_ones, w.NH, w.NKV, HT, w.DR, R0, S0, int(glob)] \
            + ttnn.TensorAccessorArgs(pq).get_compile_time_args() + ttnn.TensorAccessorArgs(k).get_compile_time_args() \
            + ttnn.TensorAccessorArgs(tau_t).get_compile_time_args()
        rd_rt = [[pq.buffer_address(), k.buffer_address(), v.buffer_address(), tau_t.buffer_address(), i, nc,
                  total] for i in range(nc)]
        cp_ct = [cb_xf, cb_xb, cb_tau, cb_ones, cb_sq, cb_f, cb_out, HT, w.DR // 32, int(glob), _bits(w.qk_clip),
                 _bits(w.rel_clip), _bits(1.0 / w.HD), _bits(w.eps), _bits(1.0 / w.scale)]
        cp_rt = [[i, nc, total] for i in range(nc)]
        wr_ct = [cb_out, HT] + ttnn.TensorAccessorArgs(q_o).get_compile_time_args() \
            + ttnn.TensorAccessorArgs(r_o).get_compile_time_args() + ttnn.TensorAccessorArgs(k_o).get_compile_time_args()
        wr_extra, wr_defs = [], [("APREP_KV", "0")]
        if kv is not None:
            kc, vc, wpos, tab = kv
            cb_st = 7
            # (positions page, table chunk, then every head's row of every k / v tile: 2 x 64 B per head and tile)
            cbs.append(_cb(self.crs, cb_st, bf, 512 + HT * w.NKV * 128, 1))
            wr_ct += ttnn.TensorAccessorArgs(kc).get_compile_time_args() \
                + ttnn.TensorAccessorArgs(wpos).get_compile_time_args() + ttnn.TensorAccessorArgs(tab).get_compile_time_args()
            wr_extra = [kc.buffer_address(), vc.buffer_address(), wpos.buffer_address(), tab.buffer_address()]
            pos_page = 64 if fill is not None else int(wpos.shape[-1]) * 4
            assert pos_page <= 256 and kc.dtype == ttnn.bfloat16 and int(kc.shape[1]) == w.NKV
            fill_defs = [("APREP_RT", str(B // 32))] if fill is not None else []
            assert fill is None or int(tab.shape[-1]) * 4 <= 1024
            al = 64 if tab.memory_config().buffer_type == ttnn.BufferType.DRAM else 16
            tab_page = -(-int(tab.shape[-1]) * 4 // al) * al  # (interleaved pages sit at the aligned page size)
            wr_defs = [("APREP_KV", "1"), ("APREP_CB_STAGE", str(cb_st)), ("APREP_NKV", str(w.NKV)),
                       ("APREP_BLK", str(int(kc.shape[2]))), ("APREP_POS_PAGE", str(pos_page)),
                       ("APREP_TAB_PAGE", str(tab_page)), ("APREP_FILL", str(int(fill is not None)))] + fill_defs
        wr_rt = [[q_o.buffer_address(), r_o.buffer_address(), k_o.buffer_address(), v_o.buffer_address(),
                  s_o.buffer_address(), i, nc, total] + wr_extra for i in range(nc)]
        u2d = [ttnn.UnpackToDestMode.Default] * 64
        for i in (cb_xf, cb_f):
            u2d[i] = ttnn.UnpackToDestMode.UnpackToDestFp32
        cfg = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True,
                                           dst_full_sync_en=True, math_approx_mode=False)
        cfg.unpack_to_dest_mode = u2d
        ud = [("APREP_QN", str(self.QN)), ("APREP_HG", str(self.HG)), ("APREP_KN", str(KN)),
              ("APREP_KG", str(KG)), ("APREP_NH", str(w.NH)), ("APREP_NKVH", str(w.NKV))]
        dm = lambda src, ct, rt, riscv, noc, defs=(): ttnn.KernelDescriptor(  # noqa: E731
            kernel_source=str(KDIR / src), core_ranges=self.crs, compile_time_args=ct, runtime_args=_rt(self.cores, rt),
            defines=list(defs) + ud, config=ttnn.DataMovementConfigDescriptor(processor=riscv, noc=noc))
        kernels = [
            dm("aprep_reader.cpp", rd_ct, rd_rt, ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.NOC_0,
               [("APREP_PQT", str(int(pq.shape[-1]) // 32)), ("APREP_KVT", str(int(k.shape[-1]) // 32))]
               if prefill else []),
            dm("aprep_writer.cpp", wr_ct, wr_rt, ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.NOC_1, wr_defs),
            ttnn.KernelDescriptor(kernel_source=str(KDIR / "aprep_compute.cpp"), core_ranges=self.crs,
                                  compile_time_args=cp_ct, runtime_args=_rt(self.cores, cp_rt), config=cfg, defines=ud),
        ]
        prog = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
        key = ("i2m_aprep", B, nc, self.HG, KG, w.NH, w.NKV, w.HD, glob, w.qk_clip, w.rel_clip, repr(pq.memory_config()),
               repr(k.memory_config()), repr(tau_t.memory_config()), repr(self.kv_mc), int(pq.shape[-1]),
               tuple(wr_defs), tuple((repr(t.memory_config()), tuple(t.shape)) for t in (kv or ())), prefill,
               tuple(pq.shape), tuple(k.shape))
        prog.custom_program_hash = int.from_bytes(hashlib.sha1(repr(key).encode()).digest()[:8], "little") & ((1 << 63) - 1)
        io, seen = [], set()
        for t in [pq, k, v, tau_t, q_o, r_o, k_o, v_o, s_o] + list(kv or ()):
            if id(t) not in seen:
                seen.add(id(t))
                io.append(t)
        ttnn.generic_op(io, prog)
        if fill is not None:
            for cache, t in ((fill[0], kf), (fill[1], vf)):
                ttnn.experimental.paged_fill_cache(cache, t, fill[2], batch_idx=0)
                ttnn.deallocate(t)
            return q_o, r_o, None, None, s_o
        if kv is not None:
            if not prefill:
                ttnn.deallocate(k_o)
                ttnn.deallocate(v_o)
            return q_o, r_o, None, None, s_o
        return q_o, r_o, k_o, v_o, s_o
