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

import os

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


# How SortedDispatch.build orders the dispatched rows by expert (QWEN35_DISPATCH_MODE):
#   "counting_h" (default): counting sort from the [Rp, E] one-hot with a dim-2 cumsum - no ttnn.sort at all
#   (prefill 2048 host wall 133.0 / 128.2 ms vs 122.7 / 117.9 with the deadlocking sort);
#   "counting": the transposed [E, Rp] one-hot with a last-dim cumsum (measured slower: 140.8 / 136.2 ms);
#   "sort_u16": ttnn.sort on uint16 ids (DRAM multi-core factory; hang-free but ~4 ms per 512-token chunk);
#   "sort_fp32": the original fp32 ttnn.sort (cross-core-exchange factory; its sort_barrier deadlocks intermittently on
#   one device of the (1,4) mesh - the 48-layer warm-up hang, doc/full_model/triage/warm_hang_*.txt).
DISPATCH_MODE = os.environ.get("QWEN35_DISPATCH_MODE", "counting_h")  # "counting" | "sort_u16" | "sort_fp32"
SORT_ON_UINT16 = DISPATCH_MODE == "sort_u16"


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
                "ones_col": ttnn.full((1, 1, Rp, TILE), 1.0, dtype=f32, layout=ttnn.TILE_LAYOUT, device=dev),
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
        # reshape/pad run on the fp32 copy (reshape_view accepts bf16/fp32/int32/uint32 only)
        ids_f = self._f32(self._u32(self._tiled(ids)))  # [1,1,T,k]
        ids_flat = ttnn.reshape(ids_f, (1, 1, 1, R))
        sc_flat = ttnn.reshape(self._tiled(scores), (1, 1, 1, R))
        if Rp != R:
            ids_flat = ttnn.pad(ids_flat, [(0, 0), (0, 0), (0, 0), (0, Rp - R)], value=float(E))  # sentinel expert
            sc_flat = ttnn.pad(sc_flat, [(0, 0), (0, 0), (0, 0), (0, Rp - R)], value=0.0)
        consts = self._consts(T)
        i_f = consts["i_f"]
        f32_ckc = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False
        )
        if DISPATCH_MODE in ("counting", "counting_h"):
            # ---- counting sort (no ttnn.sort): every token-major row i keeps its position; its destination is
            # offsets[e_i] + (rank of i among the rows of expert e_i). The rank is the running count of the one-hot
            # along the token axis, picked through the one-hot. All integers stay exact: counts/cumsum in fp32,
            # selections through 0/1 masks with the values split into < 256 hi/lo parts (the reduce/matmul datapaths
            # round operands to ~8 mantissa bits).
            # "counting_h" (default) uses the [1,1,Rp,E] one-hot with a dim-2 cumsum (permutes in and out; ~2.7 ms per
            # 512-token chunk, Tracy 2026-09-24 22:14); "counting" keeps the one-hot TRANSPOSED, [1,1,E,Rp], with the
            # cumsum along the last dim and the picks as reductions over the E rows — measured slower on the host wall.
            if DISPATCH_MODE == "counting":
                oh_t = ttnn.eq(
                    ids_flat, ttnn.reshape(self.arange_E, (1, 1, E, 1))
                )  # [1,1,E,Rp] fp32 0/1 (sentinel col: 0)
                counts_blk = ttnn.matmul(
                    oh_t, consts["ones_col"], compute_kernel_config=f32_ckc, dtype=f32
                )  # [1,1,E,32]
                c = ttnn.reshape(ttnn.slice(counts_blk, (0, 0, 0, 0), (1, 1, E, 1)), (1, 1, 1, E))  # [1,1,1,E]
                ttnn.deallocate(counts_blk)
                c_pad = ttnn.multiply(ttnn.floor(ttnn.multiply(ttnn.add(c, float(gs - 1)), 1.0 / gs)), float(gs))
                offsets = ttnn.subtract(ttnn.cumsum(c_pad, dim=3), c_pad)  # padded exclusive prefix sums [1,1,1,E]
                ttnn.deallocate(c_pad)
                ttnn.deallocate(c)
                off_col = ttnn.reshape(offsets, (1, 1, E, 1))
                o_hi = ttnn.floor(ttnn.multiply(off_col, 1.0 / 256.0))
                o_lo = ttnn.subtract(off_col, ttnn.multiply(o_hi, 256.0))
                ttnn.deallocate(off_col)
                ttnn.deallocate(offsets)
                # offsets[e_i] per column i: one nonzero per column, values < 256 -> exact through the H-reduction
                sel_hi = ttnn.sum(
                    ttnn.multiply(oh_t, o_hi), dim=2, keepdim=True, compute_kernel_config=f32_ckc
                )  # [1,1,1,Rp]
                sel_lo = ttnn.sum(ttnn.multiply(oh_t, o_lo), dim=2, keepdim=True, compute_kernel_config=f32_ckc)
                ttnn.deallocate(o_hi)
                ttnn.deallocate(o_lo)
                off_i = ttnn.add(ttnn.multiply(sel_hi, 256.0), sel_lo)
                ttnn.deallocate(sel_hi)
                ttnn.deallocate(sel_lo)
                cs = ttnn.cumsum(oh_t, dim=3, dtype=f32)  # running count per expert row along the tokens [1,1,E,Rp]
                cs_hi = ttnn.floor(ttnn.multiply(cs, 1.0 / 256.0))
                cs_lo = ttnn.subtract(cs, ttnn.multiply(cs_hi, 256.0))
                ttnn.deallocate(cs)
                r_hi = ttnn.sum(ttnn.multiply(oh_t, cs_hi), dim=2, keepdim=True, compute_kernel_config=f32_ckc)
                r_lo = ttnn.sum(ttnn.multiply(oh_t, cs_lo), dim=2, keepdim=True, compute_kernel_config=f32_ckc)
                ttnn.deallocate(cs_hi)
                ttnn.deallocate(cs_lo)
                ttnn.deallocate(oh_t)
                rank = ttnn.add(ttnn.multiply(r_hi, 256.0), r_lo)  # [1,1,1,Rp]; sentinel columns 0
                ttnn.deallocate(r_hi)
                ttnn.deallocate(r_lo)
                d_real = ttnn.subtract(ttnn.add(off_i, rank), 1.0)  # [1,1,1,Rp]
                ttnn.deallocate(off_i)
                ttnn.deallocate(rank)
            else:
                oh = ttnn.eq(
                    ttnn.reshape(ids_flat, (1, 1, Rp, 1)), self.arange_E
                )  # [1,1,Rp,E] fp32 0/1 (sentinel: all 0)
                counts_blk = ttnn.matmul(consts["ones_row"], oh, compute_kernel_config=f32_ckc, dtype=f32)  # [1,1,32,E]
                c = ttnn.slice(counts_blk, (0, 0, 0, 0), (1, 1, 1, E))  # [1,1,1,E]
                ttnn.deallocate(counts_blk)
                c_pad = ttnn.multiply(ttnn.floor(ttnn.multiply(ttnn.add(c, float(gs - 1)), 1.0 / gs)), float(gs))
                offsets = ttnn.subtract(ttnn.cumsum(c_pad, dim=3), c_pad)  # padded exclusive prefix sums [1,1,1,E]
                ttnn.deallocate(c_pad)
                ttnn.deallocate(c)
                off_col = ttnn.reshape(offsets, (1, 1, E, 1))
                o_hi = ttnn.floor(ttnn.multiply(off_col, 1.0 / 256.0))
                o_lo = ttnn.subtract(off_col, ttnn.multiply(o_hi, 256.0))
                ttnn.deallocate(off_col)
                ttnn.deallocate(offsets)
                sel_hi = ttnn.matmul(oh, o_hi, compute_kernel_config=f32_ckc, dtype=f32)  # [1,1,Rp,1]
                sel_lo = ttnn.matmul(oh, o_lo, compute_kernel_config=f32_ckc, dtype=f32)
                ttnn.deallocate(o_hi)
                ttnn.deallocate(o_lo)
                off_i = ttnn.add(ttnn.multiply(sel_hi, 256.0), sel_lo)  # [1,1,Rp,1]
                ttnn.deallocate(sel_hi)
                ttnn.deallocate(sel_lo)
                cs = ttnn.cumsum(oh, dim=2, dtype=f32)  # [1,1,Rp,E]
                cs_hi = ttnn.floor(ttnn.multiply(cs, 1.0 / 256.0))
                cs_lo = ttnn.subtract(cs, ttnn.multiply(cs_hi, 256.0))
                ttnn.deallocate(cs)
                r_hi = ttnn.sum(ttnn.multiply(oh, cs_hi), dim=3, keepdim=True, compute_kernel_config=f32_ckc)
                r_lo = ttnn.sum(ttnn.multiply(oh, cs_lo), dim=3, keepdim=True, compute_kernel_config=f32_ckc)
                ttnn.deallocate(cs_hi)
                ttnn.deallocate(cs_lo)
                ttnn.deallocate(oh)
                rank = ttnn.add(ttnn.multiply(r_hi, 256.0), r_lo)
                ttnn.deallocate(r_hi)
                ttnn.deallocate(r_lo)
                d_real = ttnn.reshape(ttnn.subtract(ttnn.add(off_i, rank), 1.0), (1, 1, 1, Rp))
                ttnn.deallocate(off_i)
                ttnn.deallocate(rank)
            if Rp != R:
                # sentinel rows (positions >= R) go to the trailing all-padding group; their source token is masked to
                # token 0 (in range) so gather_rows never reads past the T real rows (the group is never consumed)
                is_real = ttnn.lt(i_f, float(R))
                d_f = ttnn.where(is_real, d_real, ttnn.add(i_f, float(Rd - Rp)))
                ttnn.deallocate(d_real)
                src_ids = ttnn.where(is_real, ids_flat, 0.0)
                ttnn.deallocate(is_real)
            else:
                d_f = d_real
                src_ids = ids_flat
            tok_sorted = ttnn.floor(ttnn.multiply(i_f, 1.0 / k))  # source token of row i (token-major)
            if Rp != R:
                # sentinel rows: source token masked to 0 (in range) so gather_rows never reads past the T real rows
                is_real = ttnn.lt(i_f, float(R))
                tok_masked = ttnn.where(is_real, tok_sorted, 0.0)
                ttnn.deallocate(tok_sorted)
                ttnn.deallocate(is_real)
                tok_sorted = tok_masked
            sc_sorted = sc_flat  # scores already token-major
            perm_u32 = None
            sorted_ids = ids_flat
        else:
            if DISPATCH_MODE == "sort_u16":
                # ttnn.sort routes uint16 TILE rows wider than 32 tiles to its DRAM multi-core factory (SingleRowMultiCore);
                # the fp32 row of the same width (Wt 128..512) takes the cross-core-data-exchange factory whose sort_barrier
                # deadlocked intermittently on one device of the (1,4) mesh (doc/full_model/triage/warm_hang_*.txt).
                ids_sort_in = ttnn.typecast(self._u32(ids_flat), ttnn.uint16)
            else:
                ids_sort_in = ids_flat
            # ---- sort pairs by expert (values in the sort input dtype, uint16 indices)
            sorted_ids, perm = ttnn.sort(ids_sort_in, dim=-1)
            if ids_sort_in is not ids_flat:
                ttnn.deallocate(ids_sort_in)
            if sorted_ids.dtype != ttnn.float32:
                sorted_u = sorted_ids
                sorted_ids = self._f32(self._u32(sorted_u))
                ttnn.deallocate(sorted_u)
            perm_u32 = self._u32(perm)
            perm_f = self._f32(perm_u32)
            sc_sorted = ttnn.gather(sc_flat, dim=3, index=perm_u32)  # [1,1,1,Rp] bf16
            tok_sorted = ttnn.floor(ttnn.multiply(perm_f, 1.0 / k))  # source token of each sorted row
            # ---- per-expert counts from the one-hot rows (exact fp32 sums)
            oh = ttnn.eq(
                ttnn.reshape(sorted_ids, (1, 1, Rp, 1)), self.arange_E
            )  # [1,1,Rp,E] fp32 0/1 (sentinel: all 0)
            ones_row = consts["ones_row"]
            counts_blk = ttnn.matmul(
                ones_row, oh, compute_kernel_config=f32_ckc, dtype=f32
            )  # [1,1,32,E], every row = c
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
            d_real = ttnn.add(off_i, ttnn.subtract(i_f, start_i))
            ttnn.deallocate(start_i)
            ttnn.deallocate(off_i)
            if Rp != R:
                # sentinel rows (positions >= R) go to the trailing all-padding group; source token masked to 0
                is_real = ttnn.lt(i_f, float(R))
                d_f = ttnn.where(is_real, d_real, ttnn.add(i_f, float(Rd - Rp)))
                ttnn.deallocate(d_real)
                src_ids = ttnn.where(is_real, sorted_ids, 0.0)
                tok_masked = ttnn.where(is_real, tok_sorted, 0.0)
                ttnn.deallocate(tok_sorted)
                tok_sorted = tok_masked
                ttnn.deallocate(is_real)
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
        if perm_u32 is None:  # counting sort: d_f is already token-major
            d_tokmajor = self._u32(d_f)
        else:
            d_tokmajor = ttnn.scatter(consts["zeros_rp"], 3, ttnn.to_layout(perm_u32, RM), d_rm)
            d_tokmajor = ttnn.to_layout(d_tokmajor, ttnn.TILE_LAYOUT)
        if Rp != R:
            d_tokmajor = ttnn.slice(d_tokmajor, (0, 0, 0, 0), (1, 1, 1, R))
        undispatch = ttnn.reshape(d_tokmajor, (1, 1, T, k))  # uint32 tiled
        for t in (locals().get("perm"), perm_u32, locals().get("perm_f"), sorted_ids, d_f, d_rm, ids_flat, sc_flat):
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
