# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Expert-sorted MoE prefill for the expert-parallel mesh (adapted from the Qwen3.5-122B-A10B autoport).

The dense prefill (``optimized_decoder.moe_prefill_optimized``) runs every 32-token chunk through ALL of this die's
experts with an all-ones sparsity (32 local experts x 32 tokens, of which ~2 per token are routed here: 16x the useful
expert work) and re-reads every local expert's weights once per chunk. This path does what the 122B port's prefill does,
per die:

  router: global top-8 on the replicated activation (every die computes the same routing, exactly as the dense path)
  ids -> LOCAL ids: id - 32 * die for the pairs routed to this die's experts, the sentinel E_local for the others
  ttnn.moe_sorted_dispatch: (token, expert) pairs sorted by expert, each expert's rows padded to whole 32-row groups,
     padding groups marked 0xFFFF; the sentinel pairs are not dispatched (their un-dispatch slot points at the trailing
     all-padding group)
  ttnn.moe_gather_rows: the real groups' token rows -> [1, G, 32, H] tiles
  two group-parallel ttnn.sparse_matmul calls (indices = the group's expert; padding groups skipped, never written)
  ttnn.moe_gather_add_rows: per token, the sum of its k slots -> [1, 1, T, H] (this die's partial sum; the layer's
     existing all-reduce adds the dies, unchanged)

The sentinel slots read the trailing padding group, which the skipping matmul never writes: the down projection writes
into a PERSISTENT buffer that is zero-filled once (on device) and reused by every layer, so that group stays exactly zero
while the real groups are overwritten each layer. One buffer per distinct group count (a small LRU).

Knobs: QWEN3_MOE_PREFILL_MODE=sorted|dense (default sorted: teacher forcing vs HF identical to dense at 8192 tokens, 0.86 / 0.96 / 1.00, and 0.91 vs 0.92 top-1 at 2048), QWEN3_MOE_SORTED_MIN_TOKENS (default 128; shorter
prompts take the dense path), QWEN3_MOE_SORTED_MAX_TOKENS (default 4096: the single-core dispatch kernel's L1 and the
65536-row dispatch cap; longer prompts run in slices of this many tokens).
"""

from __future__ import annotations

import os
from collections import OrderedDict

import torch

import ttnn

TILE = 32
GROUP = 32
DRAM = ttnn.DRAM_MEMORY_CONFIG
PAD_GROUP_ID = 0xFFFF  # padding groups: >= E_local, skipped by moe_gather_rows and the grouped sparse matmul
MAX_DISPATCH_ROWS = 65536
L1_CB_BUDGET = 1_300_000  # the grouped sparse-matmul factory's per-core CB cap (sparse_matmul_multicore_reuse_grouped)

MOE_PREFILL_MODE = os.environ.get("QWEN3_MOE_PREFILL_MODE", "sorted").strip().lower()
SORTED_MIN_TOKENS = int(os.environ.get("QWEN3_MOE_SORTED_MIN_TOKENS", "128") or 0)
SORTED_MAX_TOKENS = int(os.environ.get("QWEN3_MOE_SORTED_MAX_TOKENS", "4096") or 4096)
_TILE_BYTES = {ttnn.bfloat4_b: 576, ttnn.bfloat8_b: 1088, ttnn.bfloat16: 2048}


def use_sorted(seq_len: int) -> bool:
    return MOE_PREFILL_MODE == "sorted" and seq_len >= SORTED_MIN_TOKENS


# Decode rows from which the decode layer takes the sorted path (0 = never). 32: the per-pair decode path's B x 32-expert
# intermediates leave L1 above 18 rows and every (row, expert) pair streams its expert's weights again.
SORTED_DECODE_MIN_ROWS = int(os.environ.get("QWEN3_DECODE_SORTED_MIN_ROWS", "0") or 0)  # 32 once validated


def use_sorted_decode(rows: int) -> bool:
    return 0 < SORTED_DECODE_MIN_ROWS <= rows


def dispatch_rows(T: int, k: int, E: int) -> int:
    """Static dispatched-row count: padded pairs (worst case: every pair local) + per-expert padding + one all-padding
    trailing group."""
    Rp = -(-(T * k) // TILE) * TILE
    return Rp + (-(-(GROUP - 1) * E // GROUP)) * GROUP + GROUP


class _State:
    """Per-mesh constants and the persistent zeroed down-output buffers."""

    def __init__(self, mesh_device, num_devices: int, num_experts_local: int):
        self.device = mesh_device
        self.E = num_experts_local
        # this die's first global expert id, as a [1,1,1,1] fp32 tile broadcast (a device-varying constant: one value per
        # die, sharded on dim 0 - the same construction as the dense path's expert window)
        offs = torch.arange(num_devices, dtype=torch.float32).mul(num_experts_local).reshape(num_devices, 1, 1, 1)
        self.die_offset = ttnn.from_torch(
            offs,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=DRAM,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0) if num_devices > 1 else None,
        )
        self.arange_E = ttnn.reshape(
            ttnn.arange(0, num_experts_local, 1, dtype=ttnn.float32, device=mesh_device, layout=ttnn.TILE_LAYOUT),
            (1, 1, 1, num_experts_local),
        )
        self.zero_out = OrderedDict()  # (G, H, layout) -> persistent zero-initialised down output
        self.cfgs = {}

    def down_out(self, G: int, H: int, layout):
        key = (G, H, layout)
        t = self.zero_out.get(key)
        if t is not None:
            self.zero_out.move_to_end(key)
            return t
        while len(self.zero_out) >= 2:
            _, old = self.zero_out.popitem(last=False)
            ttnn.deallocate(old)
        e = ttnn.empty([1, G, GROUP, H], ttnn.bfloat16, layout, self.device, DRAM)
        t = ttnn.fill(e, 0.0, memory_config=DRAM)
        ttnn.deallocate(e)
        self.zero_out[key] = t
        return t


_STATES: dict = {}


def _state(mesh_device, num_devices, num_experts_local) -> _State:
    key = (id(mesh_device), int(num_devices), int(num_experts_local))
    s = _STATES.get(key)
    if s is None:
        s = _STATES[key] = _State(mesh_device, num_devices, num_experts_local)
    return s


def _grouped_config(device, k: int, n: int, w_dtype, cap_in0: int, subblock_w: int = 8):
    """MatmulMultiCoreReuseProgramConfig for the group-parallel sparse matmul (whole 32-row groups per core, all N tiles
    per core, K in in0_block_w steps), sized to the factory's L1 budget. Returns (config, resident)."""
    grid = device.compute_with_storage_grid_size()
    k_tiles, n_tiles = k // TILE, n // TILE
    w_tile = _TILE_BYTES.get(w_dtype, 2048)
    fixed = n_tiles * (2048 + 2048)  # one bf16 output block + bf16 partials block (per_core_M = 1)
    resident = 2 * 2048 + k_tiles * n_tiles * w_tile + fixed <= L1_CB_BUDGET
    in1 = (lambda ib: k_tiles * n_tiles * w_tile) if resident else (lambda ib: 2 * n_tiles * ib * w_tile)
    ib = max(1, min(int(cap_in0), k_tiles))
    while k_tiles % ib:
        ib -= 1
    while ib > 1 and 2 * ib * 2048 + in1(ib) + fixed > L1_CB_BUDGET:
        ib -= 1
        while k_tiles % ib:
            ib -= 1
    sw = max(1, min(int(subblock_w), n_tiles))
    while n_tiles % sw:
        sw -= 1
    pc = ttnn.MatmulMultiCoreReuseProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(grid.x, grid.y),
        in0_block_w=ib,
        out_subblock_h=1,
        out_subblock_w=sw,
        per_core_M=1,
        per_core_N=n_tiles,
        skip_padding_groups=True,
    )
    return pc, resident


def router_topk_local(x, w_router, st: _State, top_k: int):
    """Global top-k on fp32 router logits (replicated; identical on every die), softmax over the k survivors (equal to
    the dense path's renormalised routing because norm_topk_prob holds), and this die's LOCAL expert ids with the
    sentinel ``E_local`` for pairs routed to another die. Returns (ids_local uint32 TILE [1,1,T,k], scores bf16 TILE).
    """
    logits = ttnn.linear(x, w_router, dtype=ttnn.float32, memory_config=DRAM)
    top_l, top_i = ttnn.topk(logits, k=top_k, dim=-1, largest=True, sorted=True)
    ttnn.deallocate(logits)
    scores = ttnn.typecast(ttnn.softmax(top_l, dim=-1, numeric_stable=True), ttnn.bfloat16)
    ttnn.deallocate(top_l)
    ids_f = ttnn.typecast(ttnn.typecast(top_i, ttnn.uint32), ttnn.float32)
    ttnn.deallocate(top_i)
    loc = ttnn.sub(ids_f, st.die_offset)
    ttnn.deallocate(ids_f)
    valid = ttnn.logical_and(ttnn.ge(loc, 0.0), ttnn.lt(loc, float(st.E)))
    loc_s = ttnn.where(valid, loc, float(st.E))
    ttnn.deallocate(valid)
    ttnn.deallocate(loc)
    ids_local = ttnn.typecast(loc_s, ttnn.uint32)
    ttnn.deallocate(loc_s)
    return ids_local, scores


def _moe_slice(x, w_router, experts, st: _State, top_k, compute_config, act_dtype, gu_in0, dn_in0):
    """One slice of T (multiple of 32, <= SORTED_MAX_TOKENS) tokens: [1,1,T,H] -> this die's partial [1,1,T,H]."""
    T, H = int(x.shape[2]), int(x.shape[3])
    E = st.E
    inter2 = int(experts.gate_up_proj.shape[-1])
    inter = inter2 // 2
    Rd = dispatch_rows(T, top_k, E)
    G = Rd // GROUP
    assert Rd <= MAX_DISPATCH_ROWS, f"T={T}: {Rd} dispatched rows > {MAX_DISPATCH_ROWS}"
    key = (inter2, H, experts.gate_up_proj.dtype, experts.down_proj.dtype)
    if key not in st.cfgs:
        cfg_gu, _ = _grouped_config(st.device, H, inter2, experts.gate_up_proj.dtype, gu_in0)
        cfg_dn, dn_res = _grouped_config(st.device, inter, H, experts.down_proj.dtype, dn_in0)
        if dn_res and act_dtype == ttnn.bfloat16:
            cfg_dn.untilize_out = True  # row-major rows straight into the gather-add un-dispatch
        st.cfgs[key] = (cfg_gu, cfg_dn, bool(cfg_dn.untilize_out))
    cfg_gu, cfg_dn, rm_out = st.cfgs[key]

    ids, scores = router_topk_local(x, w_router, st, top_k)
    src_tok, w_disp, group_ids, und = ttnn.moe_sorted_dispatch(
        ids, scores, E, top_k, GROUP, Rd, memory_config=DRAM, pad_group_id=PAD_GROUP_ID
    )
    ttnn.deallocate(ids)
    ttnn.deallocate(scores)
    weights = ttnn.reshape(ttnn.to_layout(ttnn.reshape(w_disp, (1, 1, 1, Rd)), ttnn.TILE_LAYOUT), (1, 1, Rd, 1))
    ttnn.deallocate(w_disp)
    group_ids = ttnn.reshape(group_ids, (1, G))
    # the op still validates a [1,G,1,E] sparsity operand (unused in indexed mode)
    grp_f = ttnn.typecast(
        ttnn.to_layout(ttnn.typecast(ttnn.reshape(group_ids, (1, 1, 1, G)), ttnn.uint32), ttnn.TILE_LAYOUT),
        ttnn.float32,
    )
    group_mask = ttnn.to_layout(
        ttnn.typecast(ttnn.eq(ttnn.reshape(grp_f, (1, G, 1, 1)), st.arange_E), ttnn.bfloat16), ttnn.ROW_MAJOR_LAYOUT
    )
    ttnn.deallocate(grp_f)

    x_rm = ttnn.to_layout(ttnn.reshape(x, (T, H)), ttnn.ROW_MAJOR_LAYOUT)
    x_disp = ttnn.moe_gather_rows(x_rm, ttnn.reshape(src_tok, (1, Rd)), group_ids, E, memory_config=DRAM)
    ttnn.deallocate(x_rm)
    ttnn.deallocate(src_tok)

    out_gu = ttnn.empty([1, G, GROUP, inter2], act_dtype, ttnn.TILE_LAYOUT, st.device, DRAM)
    fused = ttnn.sparse_matmul(
        x_disp,
        experts.gate_up_proj,
        sparsity=group_mask,
        indices=group_ids,
        memory_config=DRAM,
        program_config=cfg_gu,
        optional_output_tensor=out_gu,
        compute_kernel_config=compute_config,
        dtype=act_dtype,
    )
    ttnn.deallocate(x_disp)
    # gate is the FIRST half (Qwen3MoeExperts: chunk(2, dim=-1)); ttnn.swiglu would treat it as the linear half
    gate = ttnn.slice(fused, [0, 0, 0, 0], [1, G, GROUP, inter], memory_config=DRAM)
    up = ttnn.slice(fused, [0, 0, 0, inter], [1, G, GROUP, inter2], memory_config=DRAM)
    ttnn.deallocate(fused)
    din = ttnn.multiply(up, gate, input_tensor_b_activations=[ttnn.UnaryOpType.SILU], memory_config=DRAM)
    ttnn.deallocate(gate)
    ttnn.deallocate(up)
    # routing weight on the expert intermediate (the down projection is linear); padding rows carry weight 0
    din = ttnn.multiply(ttnn.reshape(din, (1, 1, Rd, inter)), weights, memory_config=DRAM)
    ttnn.deallocate(weights)
    din = ttnn.reshape(din, (1, G, GROUP, inter))
    out_dn = st.down_out(G, H, ttnn.ROW_MAJOR_LAYOUT if rm_out else ttnn.TILE_LAYOUT)
    down = ttnn.sparse_matmul(
        din,
        experts.down_proj,
        sparsity=group_mask,
        indices=group_ids,
        optional_output_tensor=out_dn,
        is_input_a_sparse=False,
        is_input_b_sparse=True,
        memory_config=DRAM,
        program_config=cfg_dn,
        compute_kernel_config=compute_config,
        dtype=act_dtype,
    )
    ttnn.deallocate(din)
    ttnn.deallocate(group_mask)
    ttnn.deallocate(group_ids)
    if rm_out:
        y_rm = ttnn.reshape(down, (Rd, H))
        out = ttnn.moe_gather_add_rows(y_rm, und, T, top_k, memory_config=DRAM)
    else:
        y_rm = ttnn.to_layout(ttnn.reshape(down, (Rd, H)), ttnn.ROW_MAJOR_LAYOUT)
        out = ttnn.moe_gather_add_rows(y_rm, und, T, top_k, memory_config=DRAM)
        ttnn.deallocate(y_rm)
    ttnn.deallocate(und)
    return out  # [1,1,T,H] (the persistent `down` buffer is NOT deallocated)


def moe_prefill_sorted(
    x,
    w_router,
    experts,
    num_devices: int,
    num_experts_local: int,
    top_k: int,
    compute_config,
    act_dtype=ttnn.bfloat16,
    gate_up_in0: int = 16,
    down_in0: int = 4,
):
    """This die's partial MoE output ``[1, 1, S, H]`` for the replicated normed activation ``x`` ``[1, 1, S, H]``.
    Same contract as ``router_forward_multichip`` + ``moe_prefill_optimized`` together (the caller all-reduces)."""
    st = _state(x.device(), num_devices, num_experts_local)
    S, H = int(x.shape[2]), int(x.shape[3])
    Sp = -(-S // TILE) * TILE
    if Sp != S:
        x = ttnn.pad(x, [(0, 0), (0, 0), (0, Sp - S), (0, 0)], value=0.0)
    outs = []
    for s0 in range(0, Sp, SORTED_MAX_TOKENS):
        s1 = min(Sp, s0 + SORTED_MAX_TOKENS)
        xs = x if (s0 == 0 and s1 == Sp) else ttnn.slice(x, [0, 0, s0, 0], [1, 1, s1, H], memory_config=DRAM)
        outs.append(_moe_slice(xs, w_router, experts, st, top_k, compute_config, act_dtype, gate_up_in0, down_in0))
        if xs is not x:
            ttnn.deallocate(xs)
    out = outs[0] if len(outs) == 1 else ttnn.concat(outs, dim=2, memory_config=DRAM)
    if len(outs) > 1:
        for o in outs:
            ttnn.deallocate(o)
    if Sp != S:
        ttnn.deallocate(x)
        sl = ttnn.slice(out, [0, 0, 0, 0], [1, 1, S, H], memory_config=DRAM)
        ttnn.deallocate(out)
        out = sl
    return out
