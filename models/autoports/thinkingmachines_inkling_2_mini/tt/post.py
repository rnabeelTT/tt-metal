# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The decode post-attention section in ONE program (tt/kernels/post_*.cpp), <= 16 rows:

    y = bf16(conv(att)) ; resid = x + y (or y at a block boundary) ; cur = rms(resid)
    mi = mix(blocks + [cur]) (the blocks' logits from MixPlan) ; out = bf16(rms(mi) * mlp_norm)   (or mi: loop layers)

C cores own Ht / C hidden tiles each; two cross-core rounds, one root core per row (core t reduces row t).
Replaces the conv step, typecasts, the add, two sharded norms with their reshards, the logits matmul and the mix
program (~12 programs)."""

from __future__ import annotations

import hashlib
import struct

import torch

import ttnn

from .bconv import KDIR, _cb, _range_set, _rt
from .bconv import _ks

F32, BF = 4096, 2048
CX, CY = [int(v) for v in __import__("os").environ.get("I2M_POST_GRID", "9,4").split(",")]  # the core rectangle: Ht / C tiles each
C = CX * CY
# the router round (a root-only gather of 2 E / 32 parts) at <= 4 rows: at 8 the round costs more than the separate
# router matmul (B = 8 29.0 vs 28.8 ms). I2M_POST_ROUTER_ROWS
# cross-core reductions: at most red_k(T) partials per 8-row group of a gather tile (tt/kernels/red.hpp). 2 with
# several rows (8 there: decode eval B = 4 KL 0.0123 -> 0.0173), 8 with one (B = 1 eval within noise, 0.4 ms faster).
# I2M_RED_K forces one
_RK = __import__("os").environ.get("I2M_RED_K")


def red_k(T):
    return int(_RK) if _RK else (2 if T > 1 else 8)

ROUTER_ROWS = int(__import__("os").environ.get("I2M_POST_ROUTER_ROWS", "4"))
# the reader pushes the round-1 inputs, then the mix inputs, then the router weights (needed only after round 2), each
# after its own barrier, instead of all after one. I2M_RD_STAGED (pre / post)
RD_STAGED = __import__("os").environ.get("I2M_RD_STAGED", "1") == "1"
# the roots zero the later rounds' gather rows after round 1, not before it (red.hpp). I2M_RED_ZLAZY
ZLAZY = __import__("os").environ.get("I2M_RED_ZLAZY", "1") == "1"
# the block-mix softmax with one SFPU init per op kind over all sources (pre / post). I2M_SOFT_BATCH
SOFT_BATCH = __import__("os").environ.get("I2M_SOFT_BATCH", "1") == "1"
# the router round reduced by core t E / 32 + e for logit tile e of row t, not all by root t (post / gate2). I2M_ROUTER_SPLIT
RSPLIT = __import__("os").environ.get("I2M_ROUTER_SPLIT", "1") == "1"
# at T > 1 rows, a round's hi / lo pairs of row t reduced by their own roots (core t P + j for pair j; the same matmuls,
# so the same sums) instead of all by root t. I2M_RED_MR: the programs using it (post, pre, gate2, multi; 1 all, 0 none;
# default pre: T = 4 pre NQ = 3 43.5 -> 41.8 us; post / gate2 no gain, multi 32.6 -> 35.6)
_MR = __import__("os").environ.get("I2M_RED_MR", "pre")
MR = {"post", "pre", "gate2", "multi"} if _MR == "1" else set() if _MR == "0" else set(_MR.split(","))
# the rounds' gather zeroing and partial sends as stateful one-packet NOC writes (red.hpp; post / pre / gate2). I2M_RED_WS
WS = __import__("os").environ.get("I2M_RED_WS", "1") == "1"
# a round-1 root zeroes its gather's unused rows after its own send, not before it (red.hpp rows: the partials never
# land there; post / pre / gate2). I2M_RED_ZDEFER
ZDEFER = __import__("os").environ.get("I2M_RED_ZDEFER", "0") == "1"
# post compute: round 1's tail (r, lg) with the softmax, round 2's tail with the normed output, each one DST block
# (no CB round trip; the same ops). I2M_POST_FUSE
POST_FUSE = __import__("os").environ.get("I2M_POST_FUSE", "0") == "1"
# the mix block also squares mi (round 2's products: no mi round trip). I2M_POST_MSQ
POST_MSQ = __import__("os").environ.get("I2M_POST_MSQ", "0") == "1"
# round 1's products (resid^2, resid q) in the conv's DST block, the query rows read with the conv inputs. I2M_POST_CSQ
CSQ = __import__("os").environ.get("I2M_POST_CSQ", "0") == "1"
# rounds of 1 < T rows with red_k(T) T <= 8: every core's T rows in its gather group's spare rows, one root (core 0)
# reducing all rows at once (red.hpp TR): one write per part and face, no relay. I2M_RED_PACK: the programs (pre, postl:
# post without a router round, gate: gate2 / multi without one; 1 all, 0 none)
_PK = __import__("os").environ.get("I2M_RED_PACK", "1")
PACK = {"pre", "postl", "gate"} if _PK == "1" else set() if _PK == "0" else set(_PK.split(","))


def red_pack(T, kind):
    return kind in PACK and T > 1 and red_k(T) * T <= 8
# the token-row CB tiles 8 / 16 rows tall (T <= 8 / 16): the fp32 unpacks / packs and the matmul passes scale with the
# rows; the cross-core gather tiles (a row per core) stay 32 x 32. I2M_TINY=0: whole tiles
TINY = __import__("os").environ.get("I2M_TINY", "1") == "1"


def tile_rows(T):
    return (8 if T <= 8 else 16) if TINY and T <= 16 else 32


def cbt(cores, index, dtype, tiles, th):
    """A CB of `tiles` th x 32 tiles (th < 32: CBFormatDescriptor tile shape)."""
    page = th * (128 if dtype == ttnn.float32 else 64)
    fmt = ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page,
                                  tile=ttnn.TileDescriptor(th, 32) if th < 32 else None)
    return ttnn.CBDescriptor(total_size=page * tiles, core_ranges=cores, format_descriptors=[fmt])


def red_sel(mesh, rep, C, K, TR=1):
    """0 / 1 tiles summing the C gathered rows (red.hpp layout: core c at tile c // (4 K), row 8 (c % 4 K // K) + c % K;
    TR > 1: its TR rows from there on) into row 0 (row t: their row t) -> (tiles, tensor [1, 1, 32, 32 tiles] bf16)."""
    nq = -(-C // (4 * K))
    a = torch.zeros(nq, 32, 32)
    for c in range(C):
        for t in range(TR):
            a[c // (4 * K), t, 8 * (c % (4 * K) // K) + (c % K) * TR + t] = 1.0
    return nq, ttnn.from_torch(a.permute(1, 0, 2).reshape(1, 1, 32, nq * 32), dtype=ttnn.bfloat16,
                               layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep)


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
        self.roots = [v for cc in self.cores[:16] for w in [mesh.worker_core_from_logical_core(cc)] for v in (w.x, w.y)]
        rep = ttnn.ReplicateTensorToMesh(mesh) if mesh.get_num_devices() > 1 else None
        self.ones = ttnn.from_torch(torch.ones(1, 1, 32, 32), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh,
                                    mesh_mapper=rep)
        self.rep = rep
        self.reds = {}
        self.zeros = ttnn.from_torch(torch.zeros(1, 1, 32, 32), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh,
                                     mesh_mapper=rep)

    def red(self, T, pack=False):
        """(tiles, tensor): the 0 / 1 tiles of a root per row (red_sel; pack: one root for the T rows)."""
        K, TR = red_k(T), T if pack else 1
        if (K, TR) not in self.reds:
            self.reds[(K, TR)] = red_sel(self.mesh, self.rep, C, K, TR)
        return self.reds[(K, TR)]

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
        assert T <= min(16, C) and att.dtype == ttnn.bfloat16 and len(blocks) == len(logits) >= 1
        NB, PER, add, norm = len(blocks), self.PER, x is not None, nw is not None
        # router (W hi, W lo) bf16 [1, 1, H, E] with nw, <= ROUTER_ROWS rows: the MoE router logits too (a root-only round) ->
        # [1, 1, max(T, 2), E] fp32; dup (T = 1): row 0 of out / logits also into row 1 (the MoE's two-row path)
        rt = router is not None and norm and T <= ROUTER_ROWS
        dup = dup and T == 1 and norm
        ET = int(router[0].shape[-1]) // 32 if rt else 0
        pack = not rt and red_pack(T, "postl")
        mr = "post" in MR and 1 < T <= 8 and not pack  # (2 T roots among the first 16 cores)
        assert not (rt and RSPLIT) or T * ET <= 16, (T, ET)
        resid = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, T, self.H]), ttnn.float32, ttnn.TILE_LAYOUT, self.mesh,
                                               out_mc)
        odt = ttnn.bfloat16 if norm else ttnn.float32
        out = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, T, self.H]), odt, ttnn.TILE_LAYOUT, self.mesh, out_mc)
        (cb_att, cb_x, cb_s, cb_w, cb_q, cb_ones, cb_d, cb_blk, cb_nw, cb_y16, cb_so, cb_res, cb_rmm, cb_rout, cb_sq,
         cb_part, cb_gath, cb_red, cb_wt, cb_mi, cb_part2, cb_gath2, cb_out, cb_sel, cb_tout, cb_tot, cb_tot2, cb_zero) = range(28)
        NQ, sel = self.red(T, pack)
        f32, bf, crs, th = ttnn.float32, ttnn.bfloat16, self.crs, tile_rows(T)
        r = lambda i, dt, n: cbt(crs, i, dt, n, th)  # noqa: E731  (token rows: th x 32 tiles)
        cbs = [r(cb_att, bf, PER), r(cb_x, bf, PER), r(cb_s, f32, 3 * PER),
               r(cb_w, f32, 4 * PER), r(cb_q, f32, PER), _cb(crs, cb_ones, bf, BF, 1),
               r(cb_d, f32, NB), r(cb_blk, f32, NB * PER), r(cb_nw, f32, PER),
               r(cb_y16, bf, 1), r(cb_so, f32, 3 * PER), r(cb_res, f32, PER),
               r(cb_rmm, f32, PER), r(cb_rout, f32, PER), r(cb_sq, f32, 4 * PER),
               r(cb_part, f32, 4), _cb(crs, cb_gath, f32, F32, 4 * NQ), r(cb_red, f32, 2),
               r(cb_wt, f32, NB + 1), r(cb_mi, f32, PER), r(cb_part2, f32, 2),
               _cb(crs, cb_gath2, f32, F32, 2 * NQ), r(cb_out, odt, 2),
               r(cb_sel, bf, NQ), r(cb_tout, f32, 2), r(cb_tot, f32, 2),
               r(cb_tot2, f32, 1), _cb(crs, cb_zero, bf, BF, 1)]
        cb_rw, cb_xb, cb_p3, cb_g3, cb_t3 = range(28, 33)
        if rt:
            cbs += [_cb(crs, cb_rw, bf, BF, 2 * PER * ET), r(cb_xb, bf, PER), r(cb_p3, f32, 2 * ET),
                    _cb(crs, cb_g3, f32, F32, 2 * ET * NQ), r(cb_t3, f32, ET)]
        lg = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, max(T, 2), 32 * ET]), ttnn.float32, ttnn.TILE_LAYOUT,
                                            self.mesh, out_mc) if rt else None
        rdefs = [("POST_RT", str(ET)), ("POST_RD_STAGED", str(int(RD_STAGED))), ("POST_ZLAZY", str(int(ZLAZY))), ("POST_RSPLIT", str(int(RSPLIT and rt))), ("POST_DUP", str(int(dup))), ("POST_MR", str(int(mr))), ("POST_PACK", str(int(pack))), ("RED_WS", str(int(WS))), ("RED_ZDEFER", str(int(ZDEFER))), ("POST_CSQ", str(int(CSQ))), ("POST_CB_RW", str(cb_rw)),
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
            + [self.zeros.buffer_address(), lg.buffer_address() if rt else 0] \
            + self.roots[:2 * max(T * ET if RSPLIT and rt else T, 2 * T if mr else T)]
        cp_ct = [cb_att, cb_x, cb_s, cb_w, cb_q, cb_ones, cb_d, cb_blk, cb_nw, cb_y16, cb_so, cb_res, cb_rmm, cb_rout,
                 cb_sq, cb_part, cb_gath, cb_red, cb_wt, cb_mi, cb_part2, cb_gath2, cb_out, PER, NB, C, int(add),
                 int(norm), _bits(1.0 / self.H), _bits(self.eps), cb_sel, NQ, cb_tout, cb_tot, cb_tot2]
        u2d = [ttnn.UnpackToDestMode.Default] * 64
        for i in (cb_s, cb_w, cb_q, cb_d, cb_blk, cb_nw, cb_res, cb_red, cb_wt, cb_mi, cb_tot, cb_tot2):
            u2d[i] = ttnn.UnpackToDestMode.UnpackToDestFp32
        cfg = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True,
                                           dst_full_sync_en=True, math_approx_mode=False)
        cfg.unpack_to_dest_mode = u2d
        defs = [("POST_RK", str(red_k(T))), ("POST_CX", str(CX)), ("POST_ROWS", str(T)), ("POST_ATT_PACKED", str(int(packed))),
                ("POST_TH", str(th))] + rdefs + sdefs

        def dm(src, ct, common, riscv, noc):
            kd = ttnn.KernelDescriptor(kernel_source=_ks(KDIR, src), core_ranges=crs, compile_time_args=ct,
                                       defines=defs, config=ttnn.DataMovementConfigDescriptor(processor=riscv, noc=noc))
            kd.common_runtime_args = common
            return kd

        kernels = [
            dm("post_reader.cpp", rd_ct, rd_common, ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.NOC_1),
            dm("post_writer.cpp", wr_ct, wr_common, ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.NOC_0),
            ttnn.KernelDescriptor(kernel_source=_ks(KDIR, "post_compute.cpp"), core_ranges=crs, compile_time_args=cp_ct,
                                  runtime_args=_rt(self.cores, [[i] for i in range(C)]), config=cfg,
                                  defines=[("ROWS", str(T)), ("SOFT_BATCH", str(int(SOFT_BATCH))), ("POST_FUSE", str(int(POST_FUSE))), ("POST_MSQ", str(int(POST_MSQ)))] + rdefs),
        ]
        sems = [ttnn.SemaphoreDescriptor(i, ttnn.CoreType.WORKER, crs, 0) for i in range(5)]
        prog = ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems, cbs=cbs)
        key = ("i2m_post", T, th, NB, add, norm, self.H, ET, RD_STAGED, ZLAZY, SOFT_BATCH, RSPLIT, mr, pack, WS, ZDEFER, POST_FUSE, POST_MSQ, CSQ, dup, packed, bconv.SEQ[:2] + (bconv.SEQ_ONE,) if acc else None, tuple(repr(t.memory_config()) for t in
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
