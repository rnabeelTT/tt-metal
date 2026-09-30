# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""MTP verify pass (v1, eager, host accept): the main model over R consecutive tokens of ONE user in one pass.

    rows      : tokens [t_P, d_1 .. d_k] at positions P .. P+k (R = k+1), the model's decode-layout block of BW rows
    per layer : `mtp_rows.layer_forward_rows` (attention: decode with the per-row K/V write; GDN: prefill form on a
                128-row padded block continuing from the user's state, which the caller binds into the B=1 scratch)
    output    : final-normed hidden [1,1,BW,H] (replicated) and vocab-sharded logits [1,1,BW,V/TP]; rows 0..R-1 real.

State: the GDN scratch is bound around the pass (`model._bind_gdn_prefill_scratch`), the user's slot state copied into
it first (`copy_slot_into_scratch`), and left in the scratch afterwards for the caller to commit (`write_slot`) once the
accepted count is known (v1: commit by re-running the pass on the accepted prefix, or copy back for full acceptance).
"""
from __future__ import annotations

import torch

import ttnn
from models.autoports.qwen_qwen3_5_122b_a10b.tt.mtp_rows import layer_forward_rows


def copy_slot_into_scratch(prev_bindings, slot: int):
    """After `_bind_gdn_prefill_scratch()` (which returned `prev_bindings`), copy decode slot `slot` of every GDN layer's
    batched buffers (saved in prev_bindings) into the bound B=1 scratch: rec_state[slot] and the K conv taps."""
    for dn, B_b, rec_b, conv_b, carry_b, zero0_b, stable_b in prev_bindings:
        rec = ttnn.slice(rec_b, (slot, 0, 0, 0), (slot + 1, rec_b.shape[1], rec_b.shape[2], rec_b.shape[3]))
        if rec.dtype != dn.rec_state.dtype:
            rec = ttnn.typecast(rec, dn.rec_state.dtype)
        ttnn.copy(rec, dn.rec_state)
        ttnn.deallocate(rec)
        for m in range(dn.K):
            tap = ttnn.slice(conv_b[m], (0, slot, 0), (1, slot + 1, conv_b[m].shape[-1]))
            if tap.dtype != dn.conv_states[m].dtype:
                tap = ttnn.typecast(tap, dn.conv_states[m].dtype)
            ttnn.copy(tap, dn.conv_states[m])
            ttnn.deallocate(tap)


def snapshot_scratch(model):
    """Device clones of the bound scratch (rec + taps per GDN layer) for `model._write_gdn_slot_device`."""
    return model._snapshot_gdn_scratch_device()


def verify_pass(model, tokens, positions, page_table_row, *, bw=None):
    """tokens: list of R ints (t_P, drafts...), positions: list of R ints (P .. P+R-1), page_table_row: torch [1, blocks].
    Returns (hidden [1,1,BW,H] replicated, logits [1,1,BW,V/TP] vocab-sharded, R). The caller binds the GDN scratch."""
    R = len(tokens)
    BW = bw or max(8, -(-R // 8) * 8)
    assert R <= BW <= 32
    mesh = model.mesh_device
    dev = ttnn.DRAM_MEMORY_CONFIG
    pos_list = list(positions) + [positions[-1] + 1 + j for j in range(BW - R)]  # dummy rows above the real ones
    tok_list = list(tokens) + [0] * (BW - R)
    tokens_tt, cur_pos_tt, rot_idxs, pt_tt = model.prepare_decode_inputs_host(
        torch.tensor(tok_list, dtype=torch.int32),
        torch.tensor(pos_list, dtype=torch.int32),
        page_table_row.repeat(BW, 1),
    )
    tokens_tt, cur_pos_tt, rot_idxs, pt_tt = (
        ttnn.to_device(t, mesh, memory_config=dev) for t in (tokens_tt, cur_pos_tt, rot_idxs, pt_tt)
    )
    cos, sin = model._rope_from_idxs(rot_idxs)
    tok = model._decode_tokens_view(tokens_tt, BW)
    x = model.embd(tok)
    x = ttnn.reshape(x, (1, 1, BW, x.shape[-1]))
    x = ttnn.to_memory_config(x, dev)
    for layer in model.layers:
        y = layer_forward_rows(layer, x, R, cos=cos, sin=sin, position_tensor=cur_pos_tt, page_table=pt_tt)
        ttnn.deallocate(x)
        x = y
    h = model._final_norm_decode(x)
    logits = ttnn.linear(h, model.lm_head_weight, compute_kernel_config=model._lm_head_ckc())
    for t in (tokens_tt, cur_pos_tt, rot_idxs, pt_tt, cos, sin):
        ttnn.deallocate(t)
    return h, logits, R


def greedy_accept(logits_rows: torch.Tensor, drafts: list[int]):
    """logits_rows [R, V] (row i = logits after token i of the R rows); drafts d_1..d_k = tokens[1:].
    Returns (accepted a, next token): the first a drafts equal argmax(row i) for i < a; the next token is argmax(row a).
    """
    am = logits_rows.argmax(-1).tolist()
    a = 0
    for d, pred in zip(drafts, am):
        if d != pred:
            break
        a += 1
    return a, am[a]
