# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Expert-sorted routed experts for Hy4 prefill (borrowed from the M2.7 port's Qwen35MoEOptimized._routed_sorted).

The chunk-masked path ran every expert any of a 32-row chunk's tokens picked (with top-16 of 512 that is ~200 experts
per chunk, x 64 chunks per 2048 rows: ~27x the useful expert work and ~600 ops per layer). Here the (token, expert)
pairs are sorted by expert on the device (ttnn.experimental.moe_sorted_dispatch, one program), each expert's rows padded
to whole 32-row groups, and each projection runs as ONE grouped sparse matmul over the groups (padding groups skipped);
moe_gather_rows / moe_gather_add_rows move the rows in and out. Hy4 specifics: the experts run in the 1536 latent
(input xl = W_down x), the routing weights (x 3.998) multiply the expert intermediate (the down projection is linear),
and the SwiGLU is clamped (gate <= 10, |up| <= 10) as the reference.
"""

from __future__ import annotations

import os

import ttnn

from models.autoports.minimaxai_minimax_m2_7.tt.moe_dispatch import SortedDispatch, gather_rows, undispatch_rows
from models.autoports.tencent_hy4_flash.tt.swiglu_pf import SwigluPF

GS = 32
DRAM = ttnn.DRAM_MEMORY_CONFIG


def _dispatch_mc_ok(T, E, k, Rd, gs) -> bool:
    """HY4_PF_DISPATCH_MC=1 (default) and the multi-core dispatch's per-core L1 plan fits this chunk."""
    if os.environ.get("HY4_PF_DISPATCH_MC", "1") != "1":
        return False
    from models.demos.blackhole.qwen36.tt.moe_prefill.dispatch_mc import fits

    return fits(T, E, k, Rd, gs)


_RM_KERNELS = ("ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm_grouped_resident_in1.cpp",
               "ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_writer_bmm_tile_layout_in1_indexed.cpp")


def _rm_kernels_ok() -> bool:
    """The grouped factory's untilize_out kernels have the 8-tile pack-untilize chunk (`untilize_chunk_cap`): with the
    model's 16-bit DEST the old 16-tile chunk wrote tiles 10..15 of every chunk wrong. Checks the files the JIT build
    resolves (tt_metal resolve_path: the working directory, TT_METAL_KERNEL_PATH, then the root), so an image without
    the fix keeps the tile path."""
    import ttnn as _t

    roots = [os.getcwd(), os.environ.get("TT_METAL_KERNEL_PATH"), os.environ.get("TT_METAL_RUNTIME_ROOT"),
             os.environ.get("TT_METAL_HOME"), os.path.dirname(os.path.dirname(os.path.dirname(_t.__file__)))]
    for rel in _RM_KERNELS:
        path = next((os.path.join(r, rel) for r in roots if r and os.path.exists(os.path.join(r, rel))), None)
        try:
            if path is None or "untilize_chunk_cap" not in open(path).read():
                return False
        except OSError:
            return False
    return True


def _grouped_cfg(device, k_dim, n_tiles, w_tile_bytes, in0_cap=8, sub_w=4, fp32_acc=False):
    """The M2.7 grouped-factory config: whole 32-row groups per core, K in in0_block_w tiles, fitting its L1 budget."""
    grid = device.compute_with_storage_grid_size()
    k_tiles = k_dim // 32
    ib = max(1, min(int(in0_cap), k_tiles))
    while k_tiles % ib:
        ib -= 1
    m_tiles = GS // 32
    budget = 1_300_000
    fixed = m_tiles * n_tiles * (2048 + 2048)
    resident = os.environ.get("TT_SPARSE_GROUPED_RESIDENT_IN1", "1") != "0"
    resident = resident and 2 * m_tiles * 2048 + k_tiles * n_tiles * w_tile_bytes + fixed <= budget
    in1 = (lambda ib_: k_tiles * n_tiles * w_tile_bytes) if resident else (lambda ib_: 2 * n_tiles * ib_ * w_tile_bytes)
    while ib > 1 and 2 * m_tiles * ib * 2048 + in1(ib) + fixed > budget:
        ib -= 1
        while k_tiles % ib:
            ib -= 1
    sw = max(1, min(sub_w, n_tiles))
    while n_tiles % sw:
        sw -= 1
    if fp32_acc:
        sw = min(sw, 4)
    return ttnn.MatmulMultiCoreReuseProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(grid.x, grid.y),
        in0_block_w=ib,
        out_subblock_h=1,
        out_subblock_w=sw,
        per_core_M=GS // 32,
        per_core_N=n_tiles,
        skip_padding_groups=True,
    )


class SortedExperts:
    """Per-MoE-layer helper (shares the layer's weights); built lazily on the first prefill call."""

    TILE_BYTES = {ttnn.bfloat4_b: 576, ttnn.bfloat8_b: 1088}

    def __init__(self, moe):
        self.moe = moe
        mesh, E, k = moe.mesh, moe.E, moe.k
        self.dispatch = SortedDispatch(mesh, E, k, group=GS, pad_group_id=0xFFFF)
        L, Il = moe.L, moe.Il
        self.cfg_gu = _grouped_cfg(mesh, L, (2 * Il) // 32, self.TILE_BYTES.get(moe.up_gate.dtype, 2048))
        self.cfg_dn = _grouped_cfg(mesh, Il, L // 32, self.TILE_BYTES.get(moe.down.dtype, 2048))
        # HY4_PF_DOWN_RM (default 1): the down projection writes ROW_MAJOR rows (grouped factory untilize_out: the
        # resident compute kernel's pack-untilize) straight into the gather-add un-dispatch: no untilize op (~0.7 ms per
        # MoE layer at 2K rows). Same values (layout only; needs the 8-tile pack-untilize chunk of
        # bmm_grouped_resident_in1.cpp with a 16-bit DEST).
        self.cfg_dn_rm = None
        if hasattr(self.cfg_dn, "untilize_out") and _rm_kernels_ok():
            k_t, n_t, wb = Il // 32, L // 32, self.TILE_BYTES.get(moe.down.dtype, 2048)
            if 2 * 2048 + k_t * n_t * wb + n_t * 4096 <= 1_300_000:  # resident in1 (the untilize_out kernel)
                self.cfg_dn_rm = _grouped_cfg(mesh, Il, L // 32, wb)
                self.cfg_dn_rm.untilize_out = True

    _MASKS = {}

    def _build(self, ids, scores, T):
        """SortedDispatch.build (kernel mode) without its group one-hot: the grouped sparse matmul selects groups by
        `indices` (padding groups carry the sentinel id) and only validates the [1, G, 1, E] sparsity operand's shape,
        so a cached zero tensor stands in for the fp32 one-hot (a [G, 32, E] fp32 eq + typecast + untilize: ~2.6 ms per
        layer at 2K rows). HY4_PF_MASK_DUMMY=0 restores the built mask."""
        d = self.dispatch
        if os.environ.get("HY4_PF_MASK_DUMMY", "1") != "1":
            return d.build(ids, scores, T)
        from models.autoports.minimaxai_minimax_m2_7.tt import moe_dispatch as md

        k, E, gs = d.k, d.E, d.group
        R = T * k
        Rd = md.dispatch_rows(T, k, E, gs)
        G = Rd // gs
        assert Rd <= md.MAX_DISPATCH_ROWS, f"sorted dispatch admits at most {md.MAX_DISPATCH_ROWS} rows (T={T}: {Rd})"
        ids_t, sc_t = d._tiled(ids), d._tiled(scores)
        if _dispatch_mc_ok(T, E, k, Rd, gs):
            # HY4_PF_DISPATCH_MC (default 1): the qwen36 multi-core drop-in (generic_op, bit-identical outputs; the
            # single-core op walks the T*k pairs on one RISC-V: ~2 ms per layer at 2K rows, top-16 of 512)
            from models.demos.blackhole.qwen36.tt.moe_prefill.dispatch_mc import dispatch_mc

            src_tok, w_disp, group_ids, und = dispatch_mc(ids_t, sc_t, E, k, gs, Rd, pad_group_id=d.pad_group_id,
                                                          memory_config=DRAM)
        else:
            op = getattr(ttnn, "moe_sorted_dispatch", None) or ttnn._ttnn.operations.experimental.moe_sorted_dispatch
            src_tok, w_disp, group_ids, und = op(ids_t, sc_t, E, k, gs, Rd, memory_config=DRAM,
                                                 pad_group_id=d.pad_group_id)
        for t in (ids_t, sc_t):
            if t is not ids and t is not scores:
                ttnn.deallocate(t)
        weights = ttnn.reshape(ttnn.to_layout(ttnn.reshape(w_disp, (1, 1, 1, Rd)), ttnn.TILE_LAYOUT), (1, 1, Rd, 1))
        ttnn.deallocate(w_disp)
        key = (id(d.device), G, E)
        mask = SortedExperts._MASKS.get(key)
        if mask is None:
            if len(SortedExperts._MASKS) >= 8:
                ttnn.deallocate(SortedExperts._MASKS.pop(next(iter(SortedExperts._MASKS))))
            mask = SortedExperts._MASKS[key] = ttnn.zeros((1, G, 1, E), dtype=ttnn.bfloat16,
                                                           layout=ttnn.ROW_MAJOR_LAYOUT, device=d.device)
        out = {"src_tok": ttnn.reshape(src_tok, (1, Rd)), "weights": weights, "group_mask": mask, "mask_cached": True,
               "group_ids": ttnn.reshape(group_ids, (1, G)), "Rd": Rd, "G": G}
        if md.UNDISPATCH_KERNEL:
            out["undispatch_rm"], out["undispatch"] = und, None
        else:
            out["undispatch"] = ttnn.reshape(ttnn.to_layout(ttnn.reshape(und, (1, 1, 1, R)), ttnn.TILE_LAYOUT), (1, 1, T, k))
            ttnn.deallocate(und)
        return out

    def __call__(self, xl, scores, ids, T, mem=DRAM):
        """xl [1,1,T,L] bf16 tiled (the latent), scores [1,1,T,k] bf16 (normalised x 3.998), ids [1,1,T,k] uint16
        -> per-die partial routed output [1,1,T,L] (the sum over this die's slice of every expert's intermediate)."""
        moe = self.moe
        E, L, Il = moe.E, moe.L, moe.Il
        m = self._build(ids, scores, T)
        Rd, G = m["Rd"], m["G"]
        x_disp = gather_rows(xl, m["src_tok"], Rd, GS, group_ids=m["group_ids"], num_experts=E)  # [1,G,32,L]
        ttnn.deallocate(m["src_tok"])
        fused = ttnn.sparse_matmul(x_disp, moe.up_gate, sparsity=m["group_mask"], indices=m["group_ids"],
                                   memory_config=DRAM, program_config=self.cfg_gu, compute_kernel_config=moe.ckc_e,
                                   dtype=ttnn.bfloat16)  # [1,G,32,2Il] = [up | gate] per die
        ttnn.deallocate(x_disp)
        if os.environ.get("HY4_PF_SWIGLU", "1") == "1" and SwigluPF.supports(Il):
            # one program for slice x 2, clamp x 2, multiply(silu), routing-weight multiply (tt/swiglu_pf: the same
            # SFPU calls in the same order on a 16-bit DEST: bit-identical)
            din2 = SwigluPF.get(moe.mesh)(fused, m["weights"], float(moe.limit), DRAM)
            ttnn.deallocate(fused)
        else:
            up = ttnn.slice(fused, (0, 0, 0, 0), (1, G, GS, Il), memory_config=DRAM)
            gate = ttnn.slice(fused, (0, 0, 0, Il), (1, G, GS, 2 * Il), memory_config=DRAM)
            ttnn.deallocate(fused)
            din = moe._swiglu(up, gate, DRAM)
            ttnn.deallocate(up)
            ttnn.deallocate(gate)
            din2 = ttnn.multiply(ttnn.reshape(din, (1, 1, Rd, Il)), m["weights"], memory_config=DRAM)
            ttnn.deallocate(din)
        ttnn.deallocate(m["weights"])
        din2 = ttnn.reshape(din2, (1, G, GS, Il))
        rm = (self.cfg_dn_rm is not None and m.get("undispatch_rm") is not None
              and os.environ.get("HY4_PF_DOWN_RM", "1") == "1")
        kw = {}
        if rm:
            kw["optional_output_tensor"] = ttnn.allocate_tensor_on_device(
                ttnn.Shape([1, G, GS, L]), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, moe.mesh, DRAM)
        down = ttnn.sparse_matmul(din2, moe.down, sparsity=m["group_mask"], indices=m["group_ids"],
                                  is_input_a_sparse=False, is_input_b_sparse=True, memory_config=DRAM,
                                  program_config=self.cfg_dn_rm if rm else self.cfg_dn,
                                  compute_kernel_config=moe.ckc_e, dtype=ttnn.bfloat16,
                                  **kw)  # [1,G,32,L] (ROW_MAJOR with HY4_PF_DOWN_RM)
        ttnn.deallocate(din2)
        if not m.get("mask_cached"):
            ttnn.deallocate(m["group_mask"])
        ttnn.deallocate(m["group_ids"])
        y = undispatch_rows(down, None, m.get("undispatch"), T, moe.k, L, undispatch_rm=m.get("undispatch_rm"))
        if down.is_allocated():  # (the ROW_MAJOR path frees it through its reshape view)
            ttnn.deallocate(down)
        for key in ("undispatch", "undispatch_rm"):
            if m.get(key) is not None:
                ttnn.deallocate(m[key])
        return y
