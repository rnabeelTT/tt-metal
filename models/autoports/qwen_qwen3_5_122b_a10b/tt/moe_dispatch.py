# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Sorted expert dispatch for MoE prefill (device-only index math, static shapes).

The masked prefill path computes, for every 32/64-token group, the union of the experts any token in the group selected
(~220 of 256 experts at 64 tokens), i.e. ~27x the useful expert work. This module sorts the (token, expert) pairs by expert
on device, pads every expert's block of rows to a multiple of 32, and hands the routed experts ONE ``ttnn.sparse_matmul``
call per projection in which every 32-row group belongs to exactly one expert (``nnz`` = number of groups, known on the
host). All shapes are static upper bounds (``rows_dispatched = roundup(T*k, 32) + 31*E + 32``); the rows that no expert
uses are computed with expert 0 and weight 0.

Pipeline (all on device, fp32 index arithmetic, exact for counts < 2^24):
  ids [T,k] -> flat [R] -> sort -> perm, sorted_ids
  one-hot(sorted_ids) [R,E] -> counts c[E]; padded counts c'[E] = roundup(c,32); starts = excl-cumsum(c); offsets = excl-cumsum(c')
  destination row d_i = offsets[e_i] + (i - starts[e_i]) for sorted position i
  dispatch maps (scatter into the [rows_dispatched] axis): source token, routing weight, group expert id
  gather rows: X_disp = embedding(source_token, X)                      -> [rows_dispatched, H]
  un-dispatch: for slot j < k: Y_t += embedding(d[t, j], Y_disp)       -> [T, H]
"""
from __future__ import annotations

import ttnn

TILE = 32
DRAM = ttnn.DRAM_MEMORY_CONFIG
# Largest dispatched-row count the index math is exact for: sort indices are uint16 and the start/offset table is split into
# two exact bytes (< 65536). With k=8, E=256, 32-row groups this admits T <= 7196 tokens; the decoder caps lower (4096).
MAX_DISPATCH_ROWS = 65536
CONST_CACHE_ENTRIES = 8


def dispatch_rows(T: int, k: int, E: int, group: int = TILE) -> int:
    """Static number of dispatched rows: padded pairs + worst-case per-expert padding + one all-padding group.
    ``group`` = rows per expert block (multiple of 32): larger blocks read each expert's weights fewer times but pad more.
    """
    R = T * k
    Rp = -(-R // TILE) * TILE
    return Rp + (-(-(group - 1) * E // group)) * group + group


class SortedDispatch:
    """Builds the dispatch/un-dispatch index tensors for one prefill call."""

    def __init__(self, device, E: int, k: int, group: int = TILE):
        assert group % TILE == 0
        self.device = device
        self.E = E
        self.k = k
        self.group = group
        # constants: expert ids as a row [1,1,1,E] (fp32) for one-hot comparisons
        self.arange_E = ttnn.arange(0, E, 1, dtype=ttnn.float32, device=device, layout=ttnn.TILE_LAYOUT)
        self.arange_E = ttnn.reshape(self.arange_E, (1, 1, 1, E))
        self._const = {}  # per-T constant tensors (ones row, positions, zero templates): created once, reused

    def _consts(self, T):
        c = self._const.get(T)
        if c is None:
            k, E, dev, gs = self.k, self.E, self.device, self.group
            R = T * k
            Rp = -(-R // TILE) * TILE
            Rd = dispatch_rows(T, k, E, gs)
            G = Rd // gs
            f32, u32, RM = ttnn.float32, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT
            c = {
                "ones_row": ttnn.full((1, 1, TILE, Rp), 1.0, dtype=f32, layout=ttnn.TILE_LAYOUT, device=dev),
                "i_f": ttnn.reshape(
                    ttnn.arange(0, Rp, 1, dtype=f32, device=dev, layout=ttnn.TILE_LAYOUT), (1, 1, 1, Rp)
                ),
                "zeros_rd_u": ttnn.zeros((1, 1, 1, Rd), dtype=u32, layout=RM, device=dev),
                "zeros_rd_b": ttnn.zeros((1, 1, 1, Rd), dtype=ttnn.bfloat16, layout=RM, device=dev),
                "zeros_g": ttnn.zeros((1, 1, 1, G), dtype=u32, layout=RM, device=dev),
                "zeros_rp": ttnn.zeros((1, 1, 1, Rp), dtype=u32, layout=RM, device=dev),
            }
            if len(self._const) >= CONST_CACHE_ENTRIES:  # bounded: drop the oldest length's constants
                oldest = next(iter(self._const))
                for t in self._const.pop(oldest).values():
                    ttnn.deallocate(t)
            self._const[T] = c
        return c

    @staticmethod
    def _f32(t):
        return t if t.dtype == ttnn.float32 else ttnn.typecast(t, ttnn.float32)

    @staticmethod
    def _u32(t):
        return t if t.dtype == ttnn.uint32 else ttnn.typecast(t, ttnn.uint32)

    @staticmethod
    def _tiled(t):
        return t if t.layout == ttnn.TILE_LAYOUT else ttnn.to_layout(t, ttnn.TILE_LAYOUT)

    def build(self, ids, scores, T: int):
        """ids [1,1,T,k] (uint16/uint32/int32, any layout), scores [1,1,T,k] bf16 tiled.

        Returns dict: src_tok (uint32 ROW_MAJOR [1, Rd]), weights (bf16 tiled [1,1,Rd,1]), group_mask (bf16 ROW_MAJOR
        [1,G,1,E], exactly one 1 per group), undispatch (uint32 tiled [1,1,T,k] destination rows), Rd, G.
        """
        k, E, dev = self.k, self.E, self.device
        gs = self.group
        R = T * k
        Rp = -(-R // TILE) * TILE
        Rd = dispatch_rows(T, k, E, gs)
        G = Rd // gs
        assert (
            Rd <= MAX_DISPATCH_ROWS
        ), f"sorted dispatch admits at most {MAX_DISPATCH_ROWS} dispatched rows (T={T}: {Rd})"
        f32, u32 = ttnn.float32, ttnn.uint32

        # ---- flat, token-major (token t, slot j) -> position t*k + j
        ids_f = self._f32(self._u32(self._tiled(ids)))  # [1,1,T,k]
        ids_flat = ttnn.reshape(ids_f, (1, 1, 1, R))
        sc_flat = ttnn.reshape(self._tiled(scores), (1, 1, 1, R))
        if Rp != R:
            ids_flat = ttnn.pad(ids_flat, [(0, 0), (0, 0), (0, 0), (0, Rp - R)], value=float(E))  # sentinel expert
            sc_flat = ttnn.pad(sc_flat, [(0, 0), (0, 0), (0, 0), (0, Rp - R)], value=0.0)
        # ---- sort pairs by expert
        sorted_ids, perm = ttnn.sort(ids_flat, dim=-1)  # fp32 values, uint16 indices
        perm_u32 = self._u32(perm)
        perm_f = self._f32(perm_u32)
        sc_sorted = ttnn.gather(sc_flat, dim=3, index=perm_u32)  # [1,1,1,Rp] bf16
        tok_sorted = ttnn.floor(ttnn.multiply(perm_f, 1.0 / k))  # source token of each sorted row
        # ---- per-expert counts from the one-hot rows (exact fp32 sums)
        oh = ttnn.eq(ttnn.reshape(sorted_ids, (1, 1, Rp, 1)), self.arange_E)  # [1,1,Rp,E] fp32 0/1 (sentinel: all 0)
        consts = self._consts(T)
        ones_row = consts["ones_row"]
        f32_ckc = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False
        )
        counts_blk = ttnn.matmul(ones_row, oh, compute_kernel_config=f32_ckc, dtype=f32)  # [1,1,32,E], every row = c
        c = ttnn.slice(counts_blk, (0, 0, 0, 0), (1, 1, 1, E))  # [1,1,1,E]
        ttnn.deallocate(counts_blk)
        c_pad = ttnn.multiply(ttnn.floor(ttnn.multiply(ttnn.add(c, float(gs - 1)), 1.0 / gs)), float(gs))
        starts = ttnn.subtract(ttnn.cumsum(c, dim=3), c)  # exclusive prefix sums
        offsets = ttnn.subtract(ttnn.cumsum(c_pad, dim=3), c_pad)
        ttnn.deallocate(c_pad)
        ttnn.deallocate(c)
        # per sorted row: starts[e_i], offsets[e_i] selected through the one-hot rows. The matmul datapath rounds fp32
        # operands to ~8 mantissa bits, so the table is split into high/low bytes (each < 256, exact) and recombined.
        table = ttnn.concat(
            [ttnn.reshape(starts, (1, 1, E, 1)), ttnn.reshape(offsets, (1, 1, E, 1))], dim=3
        )  # [1,1,E,2]
        ttnn.deallocate(starts)
        ttnn.deallocate(offsets)
        t_hi = ttnn.floor(ttnn.multiply(table, 1.0 / 256.0))
        t_lo = ttnn.subtract(table, ttnn.multiply(t_hi, 256.0))
        ttnn.deallocate(table)
        sel_hi = ttnn.matmul(oh, t_hi, compute_kernel_config=f32_ckc, dtype=f32)  # [1,1,Rp,2]
        sel_lo = ttnn.matmul(oh, t_lo, compute_kernel_config=f32_ckc, dtype=f32)
        ttnn.deallocate(t_hi)
        ttnn.deallocate(t_lo)
        ttnn.deallocate(oh)
        sel = ttnn.add(ttnn.multiply(sel_hi, 256.0), sel_lo)  # sentinel rows -> 0,0
        ttnn.deallocate(sel_hi)
        ttnn.deallocate(sel_lo)
        start_i = ttnn.reshape(ttnn.slice(sel, (0, 0, 0, 0), (1, 1, Rp, 1)), (1, 1, 1, Rp))
        off_i = ttnn.reshape(ttnn.slice(sel, (0, 0, 0, 1), (1, 1, Rp, 2)), (1, 1, 1, Rp))
        ttnn.deallocate(sel)
        i_f = consts["i_f"]
        d_real = ttnn.add(off_i, ttnn.subtract(i_f, start_i))
        ttnn.deallocate(start_i)
        ttnn.deallocate(off_i)
        if Rp != R:
            # sentinel rows (positions >= R) go to the trailing all-padding group
            is_real = ttnn.lt(i_f, float(R))
            d_f = ttnn.where(is_real, d_real, ttnn.add(i_f, float(Rd - Rp)))
            ttnn.deallocate(is_real)
            ttnn.deallocate(d_real)
            src_ids = ttnn.where(ttnn.lt(i_f, float(R)), sorted_ids, 0.0)
        else:
            d_f = d_real
            src_ids = sorted_ids
        # ---- dispatch maps over the Rd rows. ttnn.scatter rejects fp32 tiled operands and int32 tiled rows > 256,
        # so the scatters run on ROW_MAJOR uint32 (integers: exact) / bf16 (routing weights) tensors.
        RM = ttnn.ROW_MAJOR_LAYOUT

        def _rm_u32(t):
            return ttnn.to_layout(self._u32(t), RM)

        d_rm = _rm_u32(d_f)
        src_tok = ttnn.scatter(consts["zeros_rd_u"], 3, d_rm, _rm_u32(tok_sorted))  # uint32 RM [1,1,1,Rd]
        ttnn.deallocate(tok_sorted)
        w_disp = ttnn.scatter(consts["zeros_rd_b"], 3, d_rm, ttnn.to_layout(sc_sorted, RM))  # bf16 RM
        ttnn.deallocate(sc_sorted)
        grp_of_row = _rm_u32(ttnn.floor(ttnn.multiply(d_f, 1.0 / gs)))
        grp_expert = ttnn.scatter(
            consts["zeros_g"], 3, grp_of_row, _rm_u32(src_ids)
        )  # expert of every group (0 = padding)
        ttnn.deallocate(grp_of_row)
        if src_ids is not sorted_ids:
            ttnn.deallocate(src_ids)
        grp_f = self._f32(ttnn.to_layout(grp_expert, ttnn.TILE_LAYOUT))
        ttnn.deallocate(grp_expert)
        group_mask = ttnn.eq(ttnn.reshape(grp_f, (1, G, 1, 1)), self.arange_E)  # [1,G,1,E] fp32 one-hot
        ttnn.deallocate(grp_f)
        group_mask = ttnn.to_layout(ttnn.typecast(group_mask, ttnn.bfloat16), RM)
        # ---- un-dispatch: destination row of (token t, slot j) in token-major order
        d_tokmajor = ttnn.scatter(consts["zeros_rp"], 3, ttnn.to_layout(perm_u32, RM), d_rm)
        d_tokmajor = ttnn.to_layout(d_tokmajor, ttnn.TILE_LAYOUT)
        if Rp != R:
            d_tokmajor = ttnn.slice(d_tokmajor, (0, 0, 0, 0), (1, 1, 1, R))
        undispatch = ttnn.reshape(d_tokmajor, (1, 1, T, k))  # uint32 tiled
        for t in (perm, perm_u32, perm_f, sorted_ids, d_f, d_rm, ids_flat, sc_flat):
            try:
                ttnn.deallocate(t)
            except Exception:
                pass
        src_tok_rm = ttnn.reshape(src_tok, (1, Rd))
        weights = ttnn.reshape(ttnn.to_layout(w_disp, ttnn.TILE_LAYOUT), (1, 1, Rd, 1))
        return {
            "src_tok": src_tok_rm,
            "weights": weights,
            "group_mask": group_mask,
            "undispatch": undispatch,
            "Rd": Rd,
            "G": G,
        }


def gather_rows(x4, src_tok_rm, Rd: int, group: int = TILE):
    """X_disp [1, G, group, H] tiled bf16 from x4 [1,1,T,H] tiled bf16 and the uint32 row index [1, Rd]."""
    T, H = x4.shape[2], x4.shape[3]
    x_rm = ttnn.to_layout(ttnn.reshape(x4, (T, H)), ttnn.ROW_MAJOR_LAYOUT)
    rows = ttnn.embedding(src_tok_rm, x_rm, layout=ttnn.TILE_LAYOUT, memory_config=DRAM)  # [1, Rd, H]
    ttnn.deallocate(x_rm)
    return ttnn.reshape(rows, (1, Rd // group, group, H))


def undispatch_rows(y_disp4, weights, undispatch_u32, T: int, k: int, H: int):
    """Sum the k dispatched rows of every token: y_disp4 [1,G,32,H] tiled -> [1,1,T,H] tiled bf16."""
    Rd = y_disp4.shape[1] * y_disp4.shape[2]
    if weights is not None:
        y = ttnn.multiply(ttnn.reshape(y_disp4, (1, 1, Rd, H)), weights, memory_config=DRAM)
        y_rm = ttnn.to_layout(ttnn.reshape(y, (Rd, H)), ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(y)
    else:
        y_rm = ttnn.to_layout(ttnn.reshape(y_disp4, (Rd, H)), ttnn.ROW_MAJOR_LAYOUT)
    acc = None
    for j in range(k):
        idx = ttnn.to_layout(
            ttnn.reshape(ttnn.slice(undispatch_u32, (0, 0, 0, j), (1, 1, T, j + 1)), (1, T)), ttnn.ROW_MAJOR_LAYOUT
        )
        part = ttnn.embedding(idx, y_rm, layout=ttnn.TILE_LAYOUT, memory_config=DRAM)  # [1, T, H]
        ttnn.deallocate(idx)
        if acc is None:
            acc = part
        else:
            acc = ttnn.add(acc, part, output_tensor=acc)
            ttnn.deallocate(part)
    ttnn.deallocate(y_rm)
    return ttnn.reshape(acc, (1, 1, T, H))
