# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Multi-user MTP verify pass (eager, first block of the multi-user turn).

    B live users, each with R = k + 1 candidate rows [t_u, d_u1 .. d_uk] at positions p_u .. p_u + k, laid out user-major
    in ONE decode-layout block of BW = rows_width(B * R) rows (row u * R + i = user u, candidate i). Attention rows run as
    the batched decode step with per-row positions / page tables and the per-row K/V write (the single-user rows mode
    already keys everything by row). The GDN layers run `forward_prefill_batched`: the batched chunk form treats the
    leading dim as independent sequences and writes rec_state[u] / conv_states[*][u] straight into the batched decode
    buffers (kernel cap: B <= 4 at TP = 4; larger batches need grouped launches - not yet).
    The MoE runs the decode path of the block's width (pairs <= 8 rows, masked 16 / 32).
"""
from __future__ import annotations

import torch

import ttnn
from models.autoports.qwen_qwen3_5_122b_a10b.tt.mtp_rows import rows_pos_masks_host, rows_width
from models.tt_transformers.tt.common import Mode

GDN_T = 32  # the batched chunk form's row block per user (one tile; the fused chunk kernel's native chunk)


def rebuild_batched_conv_carry(dn):
    """dn._batched_conv_carry [B, K-1, D] = the decode conv taps conv_states[1..K-1] ([1, B, D] each), oldest first."""
    B = dn.conv_states[1].shape[1]
    D = dn.conv_states[1].shape[-1]
    cols = [ttnn.reshape(dn.conv_states[m], (B, 1, D)) for m in range(1, dn.K)]
    carry = ttnn.concat(cols, dim=1)  # [B, K-1, D]
    if getattr(dn, "_batched_conv_carry", None) is not None:
        ttnn.deallocate(dn._batched_conv_carry)
    dn._batched_conv_carry = carry


def _placement(layer, B, R, BW):
    """One-hot matrices (cached on the layer): place [B*T, BW] puts block row u*R+i at u*T+i; gather [BW, B*T] is its
    transpose (block rows past B*R stay zero)."""
    cache = layer.__dict__.setdefault("_mtp_placement", {})
    key = (B, R, BW)
    if key not in cache:
        mesh = layer.attention.mesh
        P = torch.zeros(1, 1, B * GDN_T, BW, dtype=torch.bfloat16)
        for u in range(B):
            for i in range(R):
                P[0, 0, u * GDN_T + i, u * R + i] = 1.0
        mk = lambda t: ttnn.from_torch(
            t,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        cache[key] = (mk(P), mk(P.transpose(-1, -2).contiguous()))
    return cache[key]


def _gdn_rows_batched(layer, x, B, R, valid_lens):
    """GDN layer over the user-major rows block x [1,1,BW,dim/TP] (rows u*R+i real for i < valid_lens[u])."""
    args = layer.args
    dn = layer.attention
    BW = x.shape[2]
    dev = ttnn.DRAM_MEMORY_CONFIG
    # decode-form attention norm (gather-then-norm): replicated [1,1,BW,dim] - the batched chunk form takes a
    # replicated per-user [B, T, dim] input (its projection flattens the users)
    x_src = ttnn.clone(x, memory_config=dev)
    n = layer._decode_norm(layer.attention_norm, x_src, Mode.DECODE, args.get_norm_config("attn", Mode.DECODE))
    if x_src.is_allocated() and x_src is not n:
        ttnn.deallocate(x_src)
    H = n.shape[-1]
    # rows block [BW, dim] -> per-user T-row blocks [B*T, dim] by a one-hot placement matmul (fixed program; ttnn.pad on
    # a [B, R, dim] tensor hits "FillPad N_slices > 1 not supported"), and back with its transpose after the layer
    place, gather = _placement(layer, B, R, BW)
    # the decode-form norm returns an L1 width-sharded tensor: interleave it, or the matmul picks a sharded 1D config
    n2 = ttnn.to_memory_config(ttnn.reshape(n, (1, 1, BW, H)), dev)
    ttnn.deallocate(n)
    xb_flat = ttnn.matmul(place, n2, memory_config=dev)  # [1,1,B*T,dim]
    ttnn.deallocate(n2)
    xb = ttnn.reshape(xb_flat, (B, GDN_T, H))
    rebuild_batched_conv_carry(dn)
    out = dn.forward_prefill_batched(xb, chunk_size=int(args.gdn_chunk_size), valid_lens=list(valid_lens), carry=True)
    if xb_flat.is_allocated():
        ttnn.deallocate(xb_flat)
    # out: the users' T rows, reduce-scattered -> [B*T, dim/TP]; rows i < R of every user back into the block's rows
    Dtp = out.shape[-1]
    out_flat = ttnn.to_memory_config(ttnn.reshape(out, (1, 1, B * GDN_T, Dtp)), dev)
    attn_out = ttnn.matmul(gather, out_flat, memory_config=dev)  # [1,1,BW,dim/TP]
    ttnn.deallocate(out_flat)
    h = ttnn.add(x, attn_out)
    ttnn.deallocate(attn_out)
    ff_cfg = args.get_norm_config("attn", Mode.DECODE)
    ff_in = layer._decode_norm(layer.ffn_norm, h, Mode.DECODE, ff_cfg)
    ff_out = layer.feed_forward.forward(ff_in, mode="decode")
    ttnn.deallocate(ff_in)
    y = ttnn.add(h, ff_out)
    ttnn.deallocate(h)
    ttnn.deallocate(ff_out)
    return y


def verify_pass_multi(model, rows_per_user, positions_per_user, page_table_rows, *, bw=None):
    """rows_per_user: list over B users of R token ids; positions_per_user: list over B users of R positions;
    page_table_rows: torch [B, blocks]. Returns (hidden [1,1,BW,H] replicated, logits [1,1,BW,V/TP] vocab-sharded,
    R). Row u * R + i is user u's candidate i."""
    B = len(rows_per_user)
    R = len(rows_per_user[0])
    assert all(len(r) == R for r in rows_per_user), "every user carries R = k + 1 rows"
    BW = bw or rows_width(B * R)
    assert B * R <= BW <= 32
    mesh = model.mesh_device
    dev = ttnn.DRAM_MEMORY_CONFIG
    tok_list = [t for rows in rows_per_user for t in rows] + [0] * (BW - B * R)
    pos_list = [p for ps in positions_per_user for p in ps]
    pos_list += [pos_list[-1] + 1 + j for j in range(BW - B * R)]  # dummy rows above the last user's rows
    pt = torch.cat([page_table_rows[u : u + 1].repeat(R, 1) for u in range(B)], dim=0)
    if BW > B * R:
        pt = torch.cat([pt, page_table_rows[B - 1 : B].repeat(BW - B * R, 1)], dim=0)
    tokens_tt, cur_pos_tt, rot_idxs, pt_tt = model.prepare_decode_inputs_host(
        torch.tensor(tok_list, dtype=torch.int32), torch.tensor(pos_list, dtype=torch.int32), pt
    )
    tokens_tt, cur_pos_tt, rot_idxs, pt_tt = (
        ttnn.to_device(t, mesh, memory_config=dev) for t in (tokens_tt, cur_pos_tt, rot_idxs, pt_tt)
    )
    cos, sin = model._rope_from_idxs(rot_idxs)
    masks = ttnn.to_device(rows_pos_masks_host(pos_list, BW), mesh, memory_config=dev)
    tok = model._decode_tokens_view(tokens_tt, BW)
    x = model.embd(tok)
    x = ttnn.reshape(x, (1, 1, BW, x.shape[-1]))
    x = ttnn.to_memory_config(x, dev)
    for layer in model.layers:
        if layer.is_full_attention:
            att = layer.attention
            att.rows_kv_write = True
            att.rows_pos_masks = masks
            try:
                y = layer.forward(x, cos, sin, position_tensor=cur_pos_tt, page_table=pt_tt, mode="decode")
            finally:
                att.rows_kv_write = False
                att.rows_pos_masks = None
        else:
            y = _gdn_rows_batched(layer, x, B, R, [R] * B)
        ttnn.deallocate(x)
        x = y
    h = model._final_norm_decode(x)
    ttnn.deallocate(x)
    logits = ttnn.linear(h, model.lm_head_weight, compute_kernel_config=model._lm_head_ckc())
    for t in (tokens_tt, cur_pos_tt, rot_idxs, pt_tt, cos, sin, masks):
        ttnn.deallocate(t)
    return h, logits, R
