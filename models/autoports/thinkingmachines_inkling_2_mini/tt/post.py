# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The decode post-attention section in ONE program (tt/kernels/post_*.cpp), <= 16 rows:

    y = bf16(conv(att)) ; resid = x + y (or y at a block boundary) ; cur = rms(resid)
    mi = mix(blocks + [cur]) (the blocks' logits from MixPlan) ; out = bf16(rms(mi) * mlp_norm)   (or mi: loop layers)

C cores own Ht / C hidden tiles each; two cross-core rounds (each core sends its partial tiles to every core: no root).
Replaces the conv step, typecasts, the add, two sharded norms with their reshards, the logits matmul and the mix
program (~12 programs)."""

from __future__ import annotations

import hashlib
import struct

import torch

import ttnn

from .bconv import KDIR, _cb, _range_set, _rt

F32, BF = 4096, 2048
CX, CY = [int(v) for v in __import__("os").environ.get("I2M_POST_GRID", "9,4").split(",")]  # the core rectangle: Ht / C tiles each
C = CX * CY
# the router round (a root-only gather of 2 E / 32 parts) at <= 4 rows: at 8 the round costs more than the separate
# router matmul (B = 8 29.0 vs 28.8 ms). I2M_POST_ROUTER_ROWS
ROUTER_ROWS = int(__import__("os").environ.get("I2M_POST_ROUTER_ROWS", "4"))


def _bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", float(x)))[0]


class PostKernel:
    def __init__(self, mesh, H: int, eps: float):
        self.mesh, self.H, self.Ht, self.eps = mesh, H, H // 32, eps
        assert self.Ht % C == 0
        self.PER = self.Ht // C
        grid = mesh.compute_with_storage_grid_size()
        self.cores = [ttnn.CoreCoord(i % CX, i // CX) for i in range(C)]
        self.crs = _range_set(self.cores)
        lo = mesh.worker_core_from_logical_core(ttnn.CoreCoord(0, 0))
        hi = mesh.worker_core_from_logical_core(ttnn.CoreCoord(CX - 1, CY - 1))
        self.mc = [lo.x, lo.y, hi.x, hi.y]
        self.sel = {}
        rep = ttnn.ReplicateTensorToMesh(mesh) if mesh.get_num_devices() > 1 else None
        self.ones = ttnn.from_torch(torch.ones(1, 1, 32, 32), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh,
                                    mesh_mapper=rep)
        self.rep = rep
        self.zeros = ttnn.from_torch(torch.zeros(1, 1, 32, 32), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh,
                                     mesh_mapper=rep)

    def _sel(self, T):
        """NQ 0 / 1 tiles A_k: (sum_k A_k @ Q_k)[t] = sum over the cores of row t (Q row c T + t = core c's row t)."""
        if T not in self.sel:
            nq = -(-C * T // 32)
            a = torch.zeros(nq, 32, 32)
            for rho in range(C * T):
                a[rho // 32, rho % T, rho % 32] = 1.0
            self.sel[T] = (nq, ttnn.from_torch(a.reshape(1, 1, nq * 32, 32).transpose(-1, -1).contiguous()
                                               .reshape(nq, 32, 32).permute(1, 0, 2).reshape(1, 1, 32, nq * 32),
                                               dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.mesh,
                                               mesh_mapper=self.rep))
        return self.sel[T]

    def rows_weight(self, w, col=None):
        """A vector over H ([1, 1, 1, H] fp32, or column `col` of a [1, 1, H, 32] query) -> every tile row holding it
        ([1, 1, 32, H] fp32: the kernel multiplies tiles elementwise)."""
        t = ttnn.to_torch(ttnn.get_device_tensors(w)[0] if self.mesh.get_num_devices() > 1 else w).float()
        t = t.reshape(-1, 32)[:, col] if col is not None else t.reshape(-1)
        return ttnn.from_torch(t.reshape(1, 1, 1, -1).expand(1, 1, 32, -1).contiguous(), dtype=ttnn.float32,
                               layout=ttnn.TILE_LAYOUT, device=self.mesh, mesh_mapper=self.rep)

    def __call__(self, att, x, conv, q, blocks, logits, nw=None, out_mc=ttnn.L1_MEMORY_CONFIG, router=None, dup=False):
        """att / x [1, 1, T, H] bf16 (x None: a block boundary, resid = y), conv: the BatchedConv (states / taps),
        q: the query as rows ([1, 1, 32, H] fp32, rows_weight), blocks n fp32 [1, 1, T, H] with logits [(tensor, tile)], nw: the norm weight rows
        ([1, 1, 32, H] fp32; None: return the mix output) -> (resid fp32, out bf16 | fp32)."""
        from .ccl import PackedAR

        packed = isinstance(att, PackedAR)  # (att: the packed all-reduce rows, read in place)
        T = int(att.shape[-2])
        att = att.t if packed else att
        assert T <= 16 and att.dtype == ttnn.bfloat16 and len(blocks) == len(logits) >= 1
        NB, PER, add, norm = len(blocks), self.PER, x is not None, nw is not None
        # router (W hi, W lo) bf16 [1, 1, H, E] with nw, <= ROUTER_ROWS rows: the MoE router logits too (a root-only round) ->
        # [1, 1, max(T, 2), E] fp32; dup (T = 1): row 0 of out / logits also into row 1 (the MoE's two-row path)
        rt = router is not None and norm and T <= ROUTER_ROWS
        dup = dup and T == 1 and norm
        ET = int(router[0].shape[-1]) // 32 if rt else 0
        resid = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, T, self.H]), ttnn.float32, ttnn.TILE_LAYOUT, self.mesh,
                                               out_mc)
        odt = ttnn.bfloat16 if norm else ttnn.float32
        out = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, T, self.H]), odt, ttnn.TILE_LAYOUT, self.mesh, out_mc)
        (cb_att, cb_x, cb_s, cb_w, cb_q, cb_ones, cb_d, cb_blk, cb_nw, cb_y16, cb_so, cb_res, cb_rmm, cb_rout, cb_sq,
         cb_part, cb_gath, cb_red, cb_wt, cb_mi, cb_part2, cb_gath2, cb_out, cb_sel, cb_tout, cb_tot, cb_tot2, cb_zero) = range(28)
        NQ, sel = self._sel(T)
        f32, bf, crs = ttnn.float32, ttnn.bfloat16, self.crs
        cbs = [_cb(crs, cb_att, bf, BF, PER), _cb(crs, cb_x, bf, BF, PER), _cb(crs, cb_s, f32, F32, 3 * PER),
               _cb(crs, cb_w, f32, F32, 4 * PER), _cb(crs, cb_q, f32, F32, PER), _cb(crs, cb_ones, bf, BF, 1),
               _cb(crs, cb_d, f32, F32, NB), _cb(crs, cb_blk, f32, F32, NB * PER), _cb(crs, cb_nw, f32, F32, PER),
               _cb(crs, cb_y16, bf, BF, 1), _cb(crs, cb_so, f32, F32, 3 * PER), _cb(crs, cb_res, f32, F32, PER),
               _cb(crs, cb_rmm, f32, F32, PER), _cb(crs, cb_rout, f32, F32, PER), _cb(crs, cb_sq, f32, F32, 4 * PER),
               _cb(crs, cb_part, f32, F32, 4), _cb(crs, cb_gath, f32, F32, 4 * NQ), _cb(crs, cb_red, f32, F32, 2),
               _cb(crs, cb_wt, f32, F32, NB + 1), _cb(crs, cb_mi, f32, F32, PER), _cb(crs, cb_part2, f32, F32, 2),
               _cb(crs, cb_gath2, f32, F32, 2 * NQ), _cb(crs, cb_out, odt, BF if norm else F32, 2),
               _cb(crs, cb_sel, bf, BF, NQ), _cb(crs, cb_tout, f32, F32, 2), _cb(crs, cb_tot, f32, F32, 2),
               _cb(crs, cb_tot2, f32, F32, 1), _cb(crs, cb_zero, bf, BF, 1)]
        cb_rw, cb_xb, cb_p3, cb_g3, cb_t3 = range(28, 33)
        if rt:
            cbs += [_cb(crs, cb_rw, bf, BF, 2 * PER * ET), _cb(crs, cb_xb, bf, BF, PER), _cb(crs, cb_p3, f32, F32, 2 * ET),
                    _cb(crs, cb_g3, f32, F32, 2 * ET * NQ), _cb(crs, cb_t3, f32, F32, ET)]
        lg = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, max(T, 2), 32 * ET]), ttnn.float32, ttnn.TILE_LAYOUT,
                                            self.mesh, out_mc) if rt else None
        rdefs = [("POST_RT", str(ET)), ("POST_DUP", str(int(dup))), ("POST_CB_RW", str(cb_rw)),
                 ("POST_CB_XB", str(cb_xb)), ("POST_CB_P3", str(cb_p3)), ("POST_CB_G3", str(cb_g3)),
                 ("POST_CB_T3", str(cb_t3))]
        k = conv.kernel
        taps = list(k.w) + [k.wc]
        xa = x if x is not None else att
        rd_ct = [cb_att, cb_x, cb_s, cb_w, cb_q, cb_ones, cb_d, cb_blk, cb_nw, PER, NB, int(add), int(norm), cb_sel, NQ]
        for t in (att, xa, conv.s[0], taps[0], blocks[0]):
            rd_ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        base = [att.buffer_address(), xa.buffer_address()] + [s.buffer_address() for s in conv.s] \
            + [t.buffer_address() for t in taps] + [q.buffer_address(), self.ones.buffer_address(),
                                                    (nw if norm else q).buffer_address(), sel.buffer_address()]
        tail = [v for b, (lg, off) in zip(blocks, logits) for v in (b.buffer_address(), lg.buffer_address(), int(off))]
        rd_common = base + tail + ([router[0].buffer_address(), router[1].buffer_address()] if rt else [])
        from . import bconv

        sdefs, scbs, acc = bconv.seq_parts(rd_ct, len(rd_common), 33, crs)
        cbs += scbs
        rd_common += [acc.buffer_address()] if acc else []
        wr_ct = [cb_so, cb_rout, cb_part, cb_gath, cb_part2, cb_gath2, cb_out, PER, C, int(norm), BF if norm else F32,
                 T, NQ, cb_tout, cb_tot, cb_tot2, cb_zero]
        for t in (conv.s[0], resid, out, self.zeros) + ((lg,) if rt else ()):
            wr_ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        wr_common = [s.buffer_address() for s in conv.s] + [resid.buffer_address(), out.buffer_address()] + self.mc \
            + self.mc[:2] + [self.zeros.buffer_address()] + ([lg.buffer_address()] if rt else [])
        cp_ct = [cb_att, cb_x, cb_s, cb_w, cb_q, cb_ones, cb_d, cb_blk, cb_nw, cb_y16, cb_so, cb_res, cb_rmm, cb_rout,
                 cb_sq, cb_part, cb_gath, cb_red, cb_wt, cb_mi, cb_part2, cb_gath2, cb_out, PER, NB, C, int(add),
                 int(norm), _bits(1.0 / self.H), _bits(self.eps), cb_sel, NQ, cb_tout, cb_tot, cb_tot2]
        u2d = [ttnn.UnpackToDestMode.Default] * 64
        for i in (cb_s, cb_w, cb_q, cb_d, cb_blk, cb_nw, cb_res, cb_red, cb_wt, cb_mi, cb_tot, cb_tot2):
            u2d[i] = ttnn.UnpackToDestMode.UnpackToDestFp32
        cfg = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True,
                                           dst_full_sync_en=True, math_approx_mode=False)
        cfg.unpack_to_dest_mode = u2d
        defs = [("POST_CX", str(CX)), ("POST_ROWS", str(T)), ("POST_ATT_PACKED", str(int(packed)))] + rdefs + sdefs

        def dm(src, ct, common, riscv, noc):
            kd = ttnn.KernelDescriptor(kernel_source=str(KDIR / src), core_ranges=crs, compile_time_args=ct,
                                       defines=defs, config=ttnn.DataMovementConfigDescriptor(processor=riscv, noc=noc))
            kd.common_runtime_args = common
            return kd

        kernels = [
            dm("post_reader.cpp", rd_ct, rd_common, ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.NOC_1),
            dm("post_writer.cpp", wr_ct, wr_common, ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.NOC_0),
            ttnn.KernelDescriptor(kernel_source=str(KDIR / "post_compute.cpp"), core_ranges=crs, compile_time_args=cp_ct,
                                  runtime_args=_rt(self.cores, [[i] for i in range(C)]), config=cfg,
                                  defines=[("ROWS", str(T))] + rdefs),
        ]
        sems = [ttnn.SemaphoreDescriptor(i, ttnn.CoreType.WORKER, crs, 0) for i in range(4)]
        prog = ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems, cbs=cbs)
        key = ("i2m_post", T, NB, add, norm, self.H, ET, dup, packed, bconv.SEQ[:2] if acc else None, tuple(repr(t.memory_config()) for t in
                                                            (att, xa, conv.s[0], taps[0], blocks[0], resid, out)))
        prog.custom_program_hash = int.from_bytes(hashlib.sha1(repr(key).encode()).digest()[:8], "little") & ((1 << 63) - 1)
        io, seen = [], set()
        for t in [att, xa] + list(conv.s) + taps + [q, self.ones, sel, self.zeros] + ([nw] if norm else []) + list(blocks) \
                + [g for g, _ in logits] + [resid, out] + (list(router) + [lg] if rt else []) + ([acc] if acc else []):
            if id(t) not in seen:
                seen.add(id(t))
                io.append(t)
        ttnn.generic_op(io, prog)
        if router is not None:
            return resid, out, lg
        return resid, out
