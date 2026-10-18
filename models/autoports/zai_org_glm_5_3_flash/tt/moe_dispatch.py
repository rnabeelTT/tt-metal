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

import torch

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
#   "counting_mm" (stage 12): counting sort whose running count is a BLOCK-TRIANGULAR MATMUL (fp32 accumulate, exact
#   integers) instead of ttnn.cumsum on the [Rp, E] one-hot: the dim-2 cumsum ran on ~4 cores and needed two full-tensor
#   permutes (25 of the 53 ms MoE device time per 2048-token layer, stage-12 profile); the matmuls use the whole grid.
#   "kernel" (default, perf pass turn 5): ONE single-core program, ttnn.experimental.moe_sorted_dispatch, reads the tiled
#   top-k outputs in place and emits the same row layout bit for bit (test_dispatch_modes_equivalent): the whole dispatch
#   build is ~1.7 ms device at 2048 tokens vs ~6 ms for counting_mm (MoE block 16.5 -> 11.2 ms per layer).
DISPATCH_MODE = os.environ.get(
    "QWEN35_DISPATCH_MODE", "kernel"
)  # "kernel" | "counting_mm" | "counting_h" | "counting" | "sort_u16" | "sort_fp32"
SORT_ON_UINT16 = DISPATCH_MODE == "sort_u16"
# Un-dispatch through the multi-core gather-add kernel (ttnn.experimental.moe_gather_add_rows) instead of k x
# (slice + embedding gather + add); only available with the kernel dispatch (it consumes the raw undispatch stick).
UNDISPATCH_KERNEL = DISPATCH_MODE == "kernel" and os.environ.get("QWEN35_UNDISPATCH_KERNEL", "1") == "1"


class SortedDispatch:
    """Builds the dispatch/un-dispatch index tensors for one prefill call."""

    def __init__(self, device, E: int, k: int, group: int = TILE, pad_group_id: int = 0):
        assert group % TILE == 0
        self.device = device
        self.E = E
        self.k = k
        self.group = group
        self.pad_group_id = pad_group_id  # kernel mode only: expert id of padding groups (0 or a sentinel >= E)
        # constants: expert ids as a row [1,1,1,E] (fp32) for one-hot comparisons
        self.arange_E = ttnn.arange(0, E, 1, dtype=ttnn.float32, device=device, layout=ttnn.TILE_LAYOUT)
        self.arange_E = ttnn.reshape(self.arange_E, (1, 1, 1, E))
        self._const = {}  # per-T constant tensors (ones row, positions, zero templates): created once, reused
        self._frozen = False  # after prepare(): every length's constants exist; a new length is an error

    def prepare(self, lengths):
        """Create the constants of every row count a pass will use (setup, before any measured pass or trace
        capture), then freeze: no constant is created or evicted inside a pass afterwards."""
        self._frozen = False
        for T in sorted(set(int(t) for t in lengths)):
            self._consts(T)
        self._frozen = True

    def _consts(self, T):
        c = self._const.get(T)
        if c is None and self._frozen:
            raise RuntimeError(
                f"SortedDispatch: constants for {T} rows were not prepared at setup ({sorted(self._const)})"
            )
        if c is None and DISPATCH_MODE == "kernel":
            # the dispatch kernel builds every map itself: none of the index-math constants below is read. They cost
            # ~6.3 MB per MoE layer per 2048-row length (fp32 [32, Rp] / [Rp, 32] / [1, Rp]), ~265 MB per die over
            # the 42 MoE layers of the full model (stage-6 memory budget)
            c = self._const[T] = {}
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
            if DISPATCH_MODE == "counting_mm":
                # block-triangular running count: rows in blocks of Bk (the smallest tile multiple giving at most 32
                # blocks, so the block prefix is one 32 x 32 tile); Rp is zero-padded up to nb * Bk when Bk does not
                # divide it (e.g. T=500: Rp=4000 -> Bk=128, 32 blocks, Rp_pad=4096). L = lower-triangular ones incl. the
                # diagonal (within-block inclusive count), P = strictly-lower-triangular ones over the (up to 32) blocks
                Bk = TILE * (-(-Rp // (32 * TILE)))
                nb = -(-Rp // Bk)
                assert nb <= 32 and nb * Bk >= Rp, (Rp, Bk, nb)
                c["Rp_pad"] = nb * Bk
                L = torch.tril(torch.ones(Bk, Bk, dtype=torch.float32))
                P = torch.tril(torch.ones(32, 32, dtype=torch.float32), diagonal=-1)
                c["Bk"] = Bk
                c["nb"] = nb
                c["L_tri"] = ttnn.from_torch(L.reshape(1, 1, Bk, Bk), dtype=f32, layout=ttnn.TILE_LAYOUT, device=dev)
                c["P_excl"] = ttnn.from_torch(P.reshape(1, 1, 32, 32), dtype=f32, layout=ttnn.TILE_LAYOUT, device=dev)
                c["ones_row_bk"] = ttnn.full((1, 1, TILE, Bk), 1.0, dtype=f32, layout=ttnn.TILE_LAYOUT, device=dev)
            if not self._frozen and len(self._const) >= CONST_CACHE_ENTRIES:  # bounded (eager use only)
                oldest = next(iter(self._const))
                for t in self._const.pop(oldest).values():
                    if not isinstance(t, int):
                        ttnn.deallocate(t)
            self._const[T] = c
        return c

    def _running_count_mm(self, oh, consts, Rp, E, f32_ckc):
        """Inclusive running count of the one-hot rows along the token axis (== ttnn.cumsum(oh, dim=2)) as matmuls:
        within each block of Bk rows ``L_tri @ oh_block`` (batched over the blocks), plus the exclusive prefix of the
        block totals (``P_excl @ totals``) broadcast over the block's rows. All operands are 0/1 or small integers, the
        accumulation is fp32, so every count is exact."""
        Bk, nb, Rp_pad = consts["Bk"], consts["nb"], consts["Rp_pad"]
        f32 = ttnn.float32
        dram = ttnn.DRAM_MEMORY_CONFIG
        oh_src = oh
        if Rp_pad > Rp:  # zero rows appended: they add nothing to any count and are sliced off below
            oh_src = ttnn.pad(oh, [(0, 0), (0, 0), (0, Rp_pad - Rp), (0, 0)], value=0.0)
        oh_b = ttnn.reshape(oh_src, (1, nb, Bk, E))  # tile-aligned view: Bk % 32 == 0
        cs_local = ttnn.matmul(consts["L_tri"], oh_b, compute_kernel_config=f32_ckc, dtype=f32, memory_config=dram)
        if nb > 1:
            tot = ttnn.matmul(consts["ones_row_bk"], oh_b, compute_kernel_config=f32_ckc, dtype=f32, memory_config=dram)
            tot = ttnn.slice(tot, (0, 0, 0, 0), (1, nb, 1, E))  # [1,nb,1,E]: block totals (row 0 of every 32-row block)
            tot_rows = ttnn.reshape(ttnn.to_layout(tot, ttnn.ROW_MAJOR_LAYOUT), (1, 1, nb, E))
            ttnn.deallocate(tot)
            if nb < 32:
                tot_rows = ttnn.pad(tot_rows, [(0, 0), (0, 0), (0, 32 - nb), (0, 0)], value=0.0)
            tot_rows = ttnn.to_layout(tot_rows, ttnn.TILE_LAYOUT)
            pre = ttnn.matmul(consts["P_excl"], tot_rows, compute_kernel_config=f32_ckc, dtype=f32, memory_config=dram)
            ttnn.deallocate(tot_rows)  # [1,1,32,E]: exclusive block prefix in rows 0..nb-1
            pre_rows = ttnn.reshape(
                ttnn.to_layout(ttnn.slice(pre, (0, 0, 0, 0), (1, 1, nb, E)), ttnn.ROW_MAJOR_LAYOUT), (1, nb, 1, E)
            )
            ttnn.deallocate(pre)
            pre_rows = ttnn.to_layout(pre_rows, ttnn.TILE_LAYOUT)
            cs = ttnn.add(cs_local, pre_rows, memory_config=dram)  # broadcast over the Bk rows of every block
            ttnn.deallocate(cs_local)
            ttnn.deallocate(pre_rows)
        else:
            cs = cs_local
        cs = ttnn.reshape(cs, (1, 1, Rp_pad, E))
        if Rp_pad > Rp:
            ttnn.deallocate(oh_src)
            full = cs
            cs = ttnn.slice(full, (0, 0, 0, 0), (1, 1, Rp, E), memory_config=dram)
            ttnn.deallocate(full)
        return cs

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
        if DISPATCH_MODE == "kernel":
            return self._build_kernel(ids, scores, T, R, Rd, G)

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
        if DISPATCH_MODE in ("counting", "counting_h", "counting_mm"):
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
                if DISPATCH_MODE == "counting_mm":
                    cs = self._running_count_mm(oh, consts, Rp, E, f32_ckc)  # [1,1,Rp,E] fp32, == cumsum(oh, dim=2)
                else:
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
        # expert id of every group as the uint16 ROW_MAJOR stick the group-parallel sparse matmul (`indices=`) reads;
        # padding groups carry expert 0 (their rows are zero-weighted and never read back)
        group_ids = ttnn.reshape(ttnn.typecast(grp_expert, ttnn.uint16), (1, G))  # uint32 RM -> uint16 RM [1, G]
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
            "group_ids": group_ids,
            "undispatch": undispatch,
            "Rd": Rd,
            "G": G,
        }


def _kernel_dispatch(self, ids, scores, T, R, Rd, G):
    """One single-core program (ttnn.experimental.moe_sorted_dispatch) computes every dispatch map: the same row
    layout as the counting sort (experts ascending, each padded to whole `group` blocks, one trailing padding group),
    so the outputs are bit-identical to the counting_* modes (test_dispatch_modes_equivalent)."""
    k, E, gs = self.k, self.E, self.group
    RM = ttnn.ROW_MAJOR_LAYOUT
    dram = ttnn.DRAM_MEMORY_CONFIG
    # the op reads the tiled [1,1,T,k] top-k outputs in place (uint16 ids, bf16 scores): no typecast / untilize / reshape
    ids_t = self._tiled(ids)
    sc_t = self._tiled(scores)
    op = getattr(ttnn, "moe_sorted_dispatch", None) or ttnn._ttnn.operations.experimental.moe_sorted_dispatch
    src_tok, w_disp, group_ids, und = op(ids_t, sc_t, E, k, gs, Rd, memory_config=dram, pad_group_id=self.pad_group_id)
    for t in (ids_t, sc_t):
        if t is not ids and t is not scores:
            ttnn.deallocate(t)
    weights = ttnn.reshape(ttnn.to_layout(ttnn.reshape(w_disp, (1, 1, 1, Rd)), ttnn.TILE_LAYOUT), (1, 1, Rd, 1))
    ttnn.deallocate(w_disp)
    # one-hot of the group experts (the sparse op still validates a [1,G,1,E] sparsity operand)
    grp_f = self._f32(
        ttnn.to_layout(ttnn.typecast(ttnn.reshape(group_ids, (1, 1, 1, G)), ttnn.uint32), ttnn.TILE_LAYOUT)
    )
    group_mask = ttnn.eq(ttnn.reshape(grp_f, (1, G, 1, 1)), self.arange_E)
    ttnn.deallocate(grp_f)
    group_mask = ttnn.to_layout(ttnn.typecast(group_mask, ttnn.bfloat16), RM)
    out = {
        "src_tok": ttnn.reshape(src_tok, (1, Rd)),
        "weights": weights,
        "group_mask": group_mask,
        "group_ids": ttnn.reshape(group_ids, (1, G)),
        "Rd": Rd,
        "G": G,
    }
    if UNDISPATCH_KERNEL:
        # the un-dispatch kernel reads the raw [1, T*k] stick; no tilized copy needed
        out["undispatch_rm"] = und
        out["undispatch"] = None
    else:
        out["undispatch"] = ttnn.reshape(
            ttnn.to_layout(ttnn.reshape(und, (1, 1, 1, R)), ttnn.TILE_LAYOUT), (1, 1, T, k)
        )
        ttnn.deallocate(und)
    return out


SortedDispatch._build_kernel = _kernel_dispatch


GATHER_KERNEL = os.environ.get("QWEN35_GATHER_KERNEL", "1") == "1" and hasattr(ttnn, "moe_gather_rows")


def gather_rows(x4, src_tok_rm, Rd: int, group: int = TILE, group_ids=None, num_experts: int = 0):
    """X_disp [1, G, group, H] tiled bf16 from x4 [1,1,T,H] tiled bf16 and the uint32 row index [1, Rd].

    With ``group_ids`` (uint16 ROW_MAJOR [1, G], the dispatch's expert per group; >= num_experts = padding) and the
    kernel enabled (``QWEN35_GATHER_KERNEL``, default on), the multi-core ``ttnn.moe_gather_rows`` gathers each REAL
    group's 32 rows straight into tiles and skips the padding groups (their tiles are never read: the grouped sparse
    matmul skips them too). The embedding-op gather it replaces moved every dispatched row (97 % padding at 16
    decode rows: 601 us for 25 MB) - decode-width profile 2026-10-01."""
    T, H = x4.shape[2], x4.shape[3]
    x_rm = ttnn.to_layout(ttnn.reshape(x4, (T, H)), ttnn.ROW_MAJOR_LAYOUT)
    if GATHER_KERNEL and group_ids is not None and group == TILE and num_experts > 0:
        rows = ttnn.moe_gather_rows(x_rm, src_tok_rm, group_ids, int(num_experts), memory_config=DRAM)  # [1,G,32,H]
        ttnn.deallocate(x_rm)
        return rows
    rows = ttnn.embedding(src_tok_rm, x_rm, layout=ttnn.TILE_LAYOUT, memory_config=DRAM)  # [1, Rd, H]
    ttnn.deallocate(x_rm)
    return ttnn.reshape(rows, (1, Rd // group, group, H))


def undispatch_rows(y_disp4, weights, undispatch_u32, T: int, k: int, H: int, undispatch_rm=None):
    """Sum the k dispatched rows of every token: y_disp4 [1,G,32,H] tiled -> [1,1,T,H] tiled bf16.

    `undispatch_rm` (uint32 ROW_MAJOR [1, T*k], the dispatch kernel's raw stick) selects the fused multi-core gather-add
    kernel; otherwise `undispatch_u32` (uint32 tiled [1,1,T,k]) drives k embedding gathers + adds."""
    Rd = y_disp4.shape[1] * y_disp4.shape[2]
    if undispatch_rm is not None:
        assert weights is None
        if y_disp4.layout == ttnn.ROW_MAJOR_LAYOUT:
            # the down projection wrote row-major rows (grouped factory `untilize_out`): no untilize op
            y_rm = ttnn.reshape(y_disp4, (Rd, H))
            out = ttnn.moe_gather_add_rows(y_rm, undispatch_rm, T, k, memory_config=DRAM)
            if y_rm is not y_disp4:
                ttnn.deallocate(y_rm)
            return out
        y_rm = ttnn.to_layout(ttnn.reshape(y_disp4, (Rd, H)), ttnn.ROW_MAJOR_LAYOUT)
        out = ttnn.moe_gather_add_rows(y_rm, undispatch_rm, T, k, memory_config=DRAM)
        ttnn.deallocate(y_rm)
        return out
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
