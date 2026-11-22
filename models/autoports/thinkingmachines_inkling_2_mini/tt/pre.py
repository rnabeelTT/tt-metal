# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The decode pre-attention section in ONE program (tt/kernels/pre_*.cpp), <= 16 rows: the previous application's
tail (MoE-output conv + residual -> x, TAIL), the input norm, the block mix with NQ queries (1 sliding / 3 global; the
blocks' logits from MixPlan), and the q / k / v input norms (NORM; loop passes with message gates return the mixes).
Same core rectangle / rounds as tt/post.py."""

from __future__ import annotations

import hashlib

import ttnn

from .bconv import KDIR, _cb, _range_set, _rt
from .post import BF, CX, CY, C, F32, _bits


class PreKernel:
    def __init__(self, post):
        self.post = post  # (shares its cores, ones, zeros and selection tiles)
        self.mesh, self.H, self.PER, self.eps = post.mesh, post.H, post.PER, post.eps

    def __call__(self, x, blocks, logits, qrows, nws=None, tail=None, bound=False, out_mc=ttnn.L1_MEMORY_CONFIG):
        """x [1, 1, T, H] bf16 (the layer input) or, with tail = (mo, resid, conv), the previous application's MoE
        output (bf16) + residual (fp32) + its MoE-output BatchedConv; blocks n fp32 with logits [(tensor, first tile)]
        (NQ consecutive tiles each); qrows NQ query rows ([1, 1, 32, H] fp32); nws NQ norm weight rows (None: return
        the fp32 mixes). -> (x (bf16; the input when no tail), xn (fp32 or None), [NQ outputs])."""
        pk, PER = self.post, self.PER
        T = int(x.shape[-2])
        NB, NQ = len(blocks), len(qrows)
        norm = nws is not None
        assert T <= 16 and 1 <= NB and NQ <= 3 and len(logits) == NB
        NSEL, sel = pk._sel(T)
        shape = ttnn.Shape([1, 1, T, self.H])
        alloc = lambda dt: ttnn.allocate_tensor_on_device(shape, dt, ttnn.TILE_LAYOUT, self.mesh, out_mc)  # noqa: E731
        xo = alloc(ttnn.bfloat16) if tail is not None else x
        xn = alloc(ttnn.float32) if bound else None
        odt = ttnn.bfloat16 if norm else ttnn.float32
        outs = [alloc(odt) for _ in range(NQ)]
        K1 = 1 + NQ
        (cb_in, cb_rp, cb_s, cb_w, cb_q, cb_ones, cb_d, cb_blk, cb_nw, cb_sel, cb_so, cb_xo, cb_x, cb_sq, cb_sum, cb_part,
         cb_gath, cb_tout, cb_tot, cb_red, cb_wt, cb_h, cb_xn, cb_part2, cb_gath2, cb_tot2, cb_out, cb_zero) = range(28)
        f32, bf, crs = ttnn.float32, ttnn.bfloat16, pk.crs
        n = NB + 1
        cbs = [_cb(crs, cb_in, bf, BF, PER), _cb(crs, cb_rp, f32, F32, PER), _cb(crs, cb_s, f32, F32, 3 * PER),
               _cb(crs, cb_w, f32, F32, 4 * PER), _cb(crs, cb_q, f32, F32, NQ * PER), _cb(crs, cb_ones, bf, BF, 1),
               _cb(crs, cb_d, f32, F32, NB * NQ), _cb(crs, cb_blk, f32, F32, NB * PER),
               _cb(crs, cb_nw, f32, F32, max(1, NQ * PER)), _cb(crs, cb_sel, bf, BF, NSEL),
               _cb(crs, cb_so, f32, F32, 3 * PER), _cb(crs, cb_xo, bf, BF, PER), _cb(crs, cb_x, f32, F32, PER),
               _cb(crs, cb_sq, f32, F32, 2 * PER), _cb(crs, cb_sum, f32, F32, 1), _cb(crs, cb_part, f32, F32, 2 * K1),
               _cb(crs, cb_gath, f32, F32, 2 * K1 * NSEL), _cb(crs, cb_tout, f32, F32, K1),
               _cb(crs, cb_tot, f32, F32, K1), _cb(crs, cb_red, f32, F32, K1), _cb(crs, cb_wt, f32, F32, NQ * n),
               _cb(crs, cb_h, f32, F32, NQ * PER), _cb(crs, cb_xn, f32, F32, PER),
               _cb(crs, cb_part2, f32, F32, 2 * NQ), _cb(crs, cb_gath2, f32, F32, 2 * NQ * NSEL),
               _cb(crs, cb_tot2, f32, F32, NQ), _cb(crs, cb_out, odt, BF if norm else F32, 2),
               _cb(crs, cb_zero, bf, BF, 1)]
        mo, resid, conv = tail if tail is not None else (x, x, None)
        if conv is not None:
            states, taps = list(conv.s), list(conv.kernel.w) + [conv.kernel.wc]
        else:
            states, taps = [pk.ones] * 3, [pk.ones] * 4  # (unread)
        nwl = list(nws) if norm else list(qrows)  # (unread when not norm)
        rd_ct = [cb_in, cb_rp, cb_s, cb_w, cb_q, cb_ones, cb_d, cb_blk, cb_nw, cb_sel, PER, NB, NQ, int(tail is not None),
                 int(norm), NSEL]
        for t in (mo, resid, states[0], taps[0], blocks[0]):
            rd_ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        rd_common = [mo.buffer_address(), resid.buffer_address()] + [s.buffer_address() for s in states] \
            + [t.buffer_address() for t in taps] + [pk.ones.buffer_address(), sel.buffer_address()] \
            + [q.buffer_address() for q in qrows] + [w.buffer_address() for w in nwl] \
            + [v for b, (lg, off) in zip(blocks, logits) for v in (b.buffer_address(), lg.buffer_address(), int(off))]
        wr_ct = [cb_so, cb_xo, cb_part, cb_gath, cb_tout, cb_tot, cb_xn, cb_part2, cb_gath2, cb_tot2, cb_out, cb_zero,
                 PER, C, NQ, int(tail is not None), int(norm), int(bound), NSEL]
        for t in (states[0], outs[0], pk.zeros):
            wr_ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        wr_common = [s.buffer_address() for s in states] + [xo.buffer_address(), (xn or xo).buffer_address()] \
            + [o.buffer_address() for o in outs] + pk.mc + [pk.zeros.buffer_address()]
        cp_ct = [cb_in, cb_rp, cb_s, cb_w, cb_q, cb_ones, cb_d, cb_blk, cb_nw, cb_sel, cb_so, cb_xo, cb_x, cb_sq, cb_sum,
                 cb_part, cb_gath, cb_tout, cb_tot, cb_red, cb_wt, cb_h, cb_xn, cb_part2, cb_gath2, cb_tot2, cb_out, PER,
                 NB, NQ, int(tail is not None), int(norm), int(bound), NSEL, _bits(1.0 / self.H), _bits(self.eps)]
        u2d = [ttnn.UnpackToDestMode.Default] * 64
        for i in (cb_rp, cb_s, cb_w, cb_q, cb_d, cb_blk, cb_nw, cb_x, cb_tot, cb_red, cb_wt, cb_h, cb_tot2):
            u2d[i] = ttnn.UnpackToDestMode.UnpackToDestFp32
        cfg = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True,
                                           dst_full_sync_en=True, math_approx_mode=False)
        cfg.unpack_to_dest_mode = u2d
        defs = [("PRE_CX", str(CX)), ("PRE_ROWS", str(T))]

        def dm(src, ct, common, riscv, noc):
            kd = ttnn.KernelDescriptor(kernel_source=str(KDIR / src), core_ranges=crs, compile_time_args=ct,
                                       defines=defs, config=ttnn.DataMovementConfigDescriptor(processor=riscv, noc=noc))
            kd.common_runtime_args = common
            return kd

        kernels = [
            dm("pre_reader.cpp", rd_ct, rd_common, ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.NOC_1),
            dm("pre_writer.cpp", wr_ct, wr_common, ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.NOC_0),
            ttnn.KernelDescriptor(kernel_source=str(KDIR / "pre_compute.cpp"), core_ranges=crs, compile_time_args=cp_ct,
                                  runtime_args=_rt(pk.cores, [[i] for i in range(C)]), config=cfg,
                                  defines=[("ROWS", str(T))]),
        ]
        sems = [ttnn.SemaphoreDescriptor(i, ttnn.CoreType.WORKER, crs, 0) for i in range(4)]
        prog = ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems, cbs=cbs)
        key = ("i2m_pre", T, NB, NQ, tail is not None, norm, bound, self.H, CX, CY,
               tuple(repr(t.memory_config()) for t in (mo, resid, states[0], taps[0], blocks[0], outs[0])))
        prog.custom_program_hash = int.from_bytes(hashlib.sha1(repr(key).encode()).digest()[:8], "little") & ((1 << 63) - 1)
        io, seen = [], set()
        for t in [mo, resid] + states + taps + [pk.ones, sel, pk.zeros] + list(qrows) + nwl + list(blocks) \
                + [lg for lg, _ in logits] + [xo] + ([xn] if xn is not None else []) + outs:
            if id(t) not in seen:
                seen.add(id(t))
                io.append(t)
        ttnn.generic_op(io, prog)
        return xo, xn, outs
