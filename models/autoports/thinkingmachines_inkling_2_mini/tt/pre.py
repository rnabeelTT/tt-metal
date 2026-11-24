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
from .post import BF, CX, CY, C, F32, _bits, cbt, red_k, tile_rows


class PreKernel:
    def __init__(self, post):
        self.post = post  # (shares its cores, ones, zeros and selection tiles)
        self.mesh, self.H, self.PER, self.eps = post.mesh, post.H, post.PER, post.eps

    def __call__(self, x, blocks, logits, qrows, nws=None, tail=None, bound=False, out_mc=ttnn.L1_MEMORY_CONFIG,
                 rows=None):
        """x [1, 1, T, H] bf16 (the layer input) or, with tail = (mo, resid, conv), the previous application's MoE
        output (bf16) + residual (fp32) + its MoE-output BatchedConv; blocks n fp32 with logits [(tensor, first tile)]
        (NQ consecutive tiles each); qrows NQ query rows ([1, 1, 32, H] fp32); nws NQ norm weight rows (None: return
        the fp32 mixes). -> (x (bf16; the input when no tail), xn (fp32 or None), [NQ outputs])."""
        from .ccl import PackedAR

        pk, PER = self.post, self.PER
        T = rows or int(x.shape[-2])  # (rows: x may be a wider buffer whose first rows are live, e.g. the MoE's)
        packed = tail is not None and isinstance(tail[0], PackedAR)  # (mo: the packed all-reduce rows, in place)
        if packed:
            tail = (tail[0].t,) + tuple(tail[1:])
        NB, NQ = len(blocks), len(qrows)
        norm = nws is not None
        assert T <= 16 and 1 <= NB and NQ <= 3 and len(logits) == NB
        NSEL, sel = pk.red(T)
        shape = ttnn.Shape([1, 1, T, self.H])
        alloc = lambda dt: ttnn.allocate_tensor_on_device(shape, dt, ttnn.TILE_LAYOUT, self.mesh, out_mc)  # noqa: E731
        xo = alloc(ttnn.bfloat16) if tail is not None else x
        xn = alloc(ttnn.float32) if bound else None
        odt = ttnn.bfloat16 if norm else ttnn.float32
        outs = [alloc(odt) for _ in range(NQ)]
        K1 = 1 + NQ
        (cb_in, cb_rp, cb_s, cb_w, cb_q, cb_ones, cb_d, cb_blk, cb_nw, cb_sel, cb_so, cb_xo, cb_x, cb_sq, cb_sum, cb_part,
         cb_gath, cb_tout, cb_tot, cb_red, cb_wt, cb_h, cb_xn, cb_part2, cb_gath2, cb_tot2, cb_out, cb_zero) = range(28)
        f32, bf, crs, th = ttnn.float32, ttnn.bfloat16, pk.crs, tile_rows(T)
        r = lambda i, dt, n: cbt(crs, i, dt, n, th)  # noqa: E731  (token rows: th x 32 tiles)
        n = NB + 1
        cbs = [r(cb_in, bf, PER), r(cb_rp, f32, PER), r(cb_s, f32, 3 * PER),
               r(cb_w, f32, 4 * PER), r(cb_q, f32, NQ * PER), _cb(crs, cb_ones, bf, BF, 1),
               r(cb_d, f32, NB * NQ), r(cb_blk, f32, NB * PER),
               r(cb_nw, f32, max(1, NQ * PER)), r(cb_sel, bf, NSEL),
               r(cb_so, f32, 3 * PER), r(cb_xo, bf, PER), r(cb_x, f32, PER),
               r(cb_sq, f32, 2 * PER), r(cb_sum, f32, 1), r(cb_part, f32, 2 * K1),
               _cb(crs, cb_gath, f32, F32, 2 * K1 * NSEL), r(cb_tout, f32, K1),
               r(cb_tot, f32, K1), r(cb_red, f32, K1), r(cb_wt, f32, NQ * n),
               r(cb_h, f32, NQ * PER), r(cb_xn, f32, PER),
               r(cb_part2, f32, 2 * NQ),  # (round 2 gathers into cb_gath too)
               r(cb_tot2, f32, NQ), r(cb_out, odt, 2),
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
        from . import bconv

        sdefs, scbs, acc = bconv.seq_parts(rd_ct, len(rd_common), 28, crs) if conv is not None else ([], [], None)
        cbs += scbs
        rd_common += [acc.buffer_address()] if acc else []
        wr_ct = [cb_so, cb_xo, cb_part, cb_gath, cb_tout, cb_tot, cb_xn, cb_part2, cb_gath2, cb_tot2, cb_out, cb_zero,
                 PER, C, NQ, int(tail is not None), int(norm), int(bound), NSEL]
        for t in (states[0], outs[0], pk.zeros):
            wr_ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        wr_common = [s.buffer_address() for s in states] + [xo.buffer_address(), (xn or xo).buffer_address()] \
            + [o.buffer_address() for o in outs] + pk.mc + [pk.zeros.buffer_address()] + pk.roots[:2 * T]
        cp_ct = [cb_in, cb_rp, cb_s, cb_w, cb_q, cb_ones, cb_d, cb_blk, cb_nw, cb_sel, cb_so, cb_xo, cb_x, cb_sq, cb_sum,
                 cb_part, cb_gath, cb_tout, cb_tot, cb_red, cb_wt, cb_h, cb_xn, cb_part2, cb_gath2, cb_tot2, cb_out, PER,
                 NB, NQ, int(tail is not None), int(norm), int(bound), NSEL, _bits(1.0 / self.H), _bits(self.eps)]
        u2d = [ttnn.UnpackToDestMode.Default] * 64
        for i in (cb_rp, cb_s, cb_w, cb_q, cb_d, cb_blk, cb_nw, cb_x, cb_tot, cb_red, cb_wt, cb_h, cb_tot2):
            u2d[i] = ttnn.UnpackToDestMode.UnpackToDestFp32
        cfg = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True,
                                           dst_full_sync_en=True, math_approx_mode=False)
        cfg.unpack_to_dest_mode = u2d
        defs = [("PRE_RK", str(red_k(T))), ("PRE_CX", str(CX)), ("PRE_ROWS", str(T)), ("PRE_IN_PACKED", str(int(packed))),
                ("PRE_TH", str(th))] + sdefs

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
        sems = [ttnn.SemaphoreDescriptor(i, ttnn.CoreType.WORKER, crs, 0) for i in range(5)]
        prog = ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems, cbs=cbs)
        key = ("i2m_pre", T, th, NB, NQ, tail is not None, norm, bound, self.H, CX, CY, packed, bconv.SEQ[:2] if acc else None,
               tuple(repr(t.memory_config()) for t in (mo, resid, states[0], taps[0], blocks[0], outs[0])))
        prog.custom_program_hash = int.from_bytes(hashlib.sha1(repr(key).encode()).digest()[:8], "little") & ((1 << 63) - 1)
        io, seen = [], set()
        for t in [mo, resid] + states + taps + [pk.ones, sel, pk.zeros] + list(qrows) + nwl + list(blocks) \
                + [lg for lg, _ in logits] + [xo] + ([xn] if xn is not None else []) + outs + ([acc] if acc else []):
            if id(t) not in seen:
                seen.add(id(t))
                io.append(t)
        ttnn.generic_op(io, prog)
        return xo, xn, outs
