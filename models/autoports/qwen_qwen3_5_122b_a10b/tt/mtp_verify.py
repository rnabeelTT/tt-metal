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


def commit_gdn(model, prev_bindings, slot, saved_inputs, accepted_rows):
    """Exact GDN commit after a verify pass: the accepted rows 0..a of the pass have exactly the activations the pass
    computed (causality), so only the GDN layers need replaying, on the pass's saved 128-row normed inputs with
    valid_len = accepted_rows, from the slot state copied into the bound scratch again."""
    from models.autoports.qwen_qwen3_5_122b_a10b.tt.mtp_rows import GDN_CHUNK, rebuild_conv_carry

    copy_slot_into_scratch(prev_bindings, slot)
    gdn_layers = [layer for layer in model.layers if not layer.is_full_attention]
    assert len(gdn_layers) == len(saved_inputs), (len(gdn_layers), len(saved_inputs))
    for layer, attn_in in zip(gdn_layers, saved_inputs):
        dn = layer.attention
        rebuild_conv_carry(dn)
        out = dn.forward_prefill(attn_in, chunk_size=GDN_CHUNK, valid_len=accepted_rows, capture_state=True)
        ttnn.deallocate(out)


def verify_pass(model, tokens, positions, page_table_row, *, bw=None, saved=None):
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
        y = layer_forward_rows(layer, x, R, cos=cos, sin=sin, position_tensor=cur_pos_tt, page_table=pt_tt, saved=saved)
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


def mtp_generate_eager(gen, head, prompt_ids, max_new_tokens, k, *, log=None, gdn_commit=False):
    """v1 MTP greedy loop (eager, host accept, exact): prefill (eager all-logits path, captures the bucket hidden for the
    head's own prefill), then per pass: draft k tokens with the head, run the verify pass over [t, d_1..d_k], accept the
    matching prefix a, commit the GDN state by re-running the pass on the accepted a+1 rows (exact but doubles the
    pass cost; a per-row prefix-state commit replaces this later), feed the head the accepted rows. Returns the tokens.
    """
    from models.tt_transformers.tt.common import Mode

    model = gen.model
    mesh = model.mesh_device
    dev = ttnn.DRAM_MEMORY_CONFIG
    H = gen.args.dim
    page_table = gen.page_table[:1]
    captured = {}
    orig_bucket_hidden = model._run_bucket_hidden

    def capturing(bi, bucket, chunk_start):
        x = orig_bucket_hidden(bi, bucket, chunk_start)
        captured["x_all"] = ttnn.clone(x, memory_config=dev)
        return x

    model._run_bucket_hidden = capturing
    try:
        gen.reset()
        P = len(prompt_ids)
        logits0 = gen.prefill_forward(
            torch.tensor(prompt_ids).reshape(1, -1), page_table=page_table, prompt_lens=[P], return_all_logits=True
        )
    finally:
        model._run_bucket_hidden = orig_bucket_hidden
    tok0 = int(torch.argmax(logits0[0, -1]).item())
    x_all = captured.pop("x_all")
    h_all = model.norm(x_all, mode=Mode.PREFILL)  # replicated [1,1,bucket,H]
    ttnn.deallocate(x_all)

    def head_rows(tok_ids, hid_rows, positions):
        last = None
        for tk, hid, ps in zip(tok_ids, hid_rows, positions):
            tokens_tt, cur_pos_tt, rot_idxs, pt_tt = model.prepare_decode_inputs_host(
                torch.tensor([tk]), ps, page_table
            )
            tokens_tt, cur_pos_tt, rot_idxs, pt_tt = (
                ttnn.to_device(t, mesh, memory_config=dev) for t in (tokens_tt, cur_pos_tt, rot_idxs, pt_tt)
            )
            lg, last = head.decode_step(tokens_tt, hid, cur_pos_tt, rot_idxs, pt_tt)
            ttnn.deallocate(lg)
        return last

    def head_draft(tok, hid, pos, k):
        drafts, cur_tok, cur_hid, cur_pos = [], tok, hid, pos
        for _ in range(k):
            tokens_tt, cur_pos_tt, rot_idxs, pt_tt = model.prepare_decode_inputs_host(
                torch.tensor([cur_tok]), cur_pos, page_table
            )
            tokens_tt, cur_pos_tt, rot_idxs, pt_tt = (
                ttnn.to_device(t, mesh, memory_config=dev) for t in (tokens_tt, cur_pos_tt, rot_idxs, pt_tt)
            )
            lg, h_next = head.decode_step(tokens_tt, cur_hid, cur_pos_tt, rot_idxs, pt_tt)
            lgt = ttnn.to_torch(lg, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=-1)).float()
            d = int(lgt.reshape(-1, lgt.shape[-1])[0, : gen.args.vocab_size].argmax().item())
            drafts.append(d)
            cur_tok, cur_hid, cur_pos = d, h_next, cur_pos + 1
        return drafts

    # head prefill over the prompt: rows (token t+1, hidden t) at positions 1..P (row P takes tok0)
    rows_tok = [prompt_ids[t + 1] if t + 1 < P else tok0 for t in range(P)]
    rows_hid = [ttnn.slice(h_all, (0, 0, t, 0), (1, 1, t + 1, H)) for t in range(P)]
    head_rows(rows_tok, rows_hid, [t + 1 for t in range(P)])
    # clone: a slice may alias its source, which is freed next
    hid_last = ttnn.clone(ttnn.slice(h_all, (0, 0, P - 1, 0), (1, 1, P, H)), memory_config=dev)  # produced tok0
    for r in rows_hid:
        ttnn.deallocate(r)
    ttnn.deallocate(h_all)

    out_tokens = [tok0]
    pos = P  # position of the next token to consume (tok0)
    passes = 0
    while len(out_tokens) < max_new_tokens:
        t_cur = out_tokens[-1]
        drafts = head_draft(t_cur, hid_last, pos, k)
        rows = [t_cur] + drafts
        positions = [pos + i for i in range(k + 1)]
        prev = model._bind_gdn_prefill_scratch()
        try:
            copy_slot_into_scratch(prev, 0)
            saved = [] if gdn_commit else None
            h, logits, R = verify_pass(model, rows, positions, page_table, saved=saved)
            lg = ttnn.to_torch(logits, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=-1)).float()
            lg = lg.reshape(-1, lg.shape[-1])[:R, : gen.args.vocab_size]
            a, nxt = greedy_accept(lg, drafts)
            ttnn.deallocate(logits)
            if a < k:
                if gdn_commit:
                    # exact GDN-only commit on the pass's saved normed inputs (the accepted rows' activations are those
                    # of the pass); the pass's hidden rows 0..a stay valid for the head
                    commit_gdn(model, prev, 0, saved, a + 1)
                else:
                    # v1: re-run the whole pass over the accepted a+1 rows
                    ttnn.deallocate(h)
                    copy_slot_into_scratch(prev, 0)
                    h, logits2, _ = verify_pass(model, rows[: a + 1], positions[: a + 1], page_table)
                    ttnn.deallocate(logits2)
            if saved:
                for t in saved:
                    ttnn.deallocate(t)
            recs, convs = model._snapshot_gdn_scratch_device()
        finally:
            model._unbind_gdn_prefill_scratch(prev)
        model._write_gdn_slot_device(0, recs, convs)
        accepted_rows = rows[1 : a + 1]  # d_1..d_a
        # the head's KV for the accepted rows (row i = token rows[i], hidden row i-1), then the hidden for the next draft
        h_rows = [ttnn.slice(h, (0, 0, i, 0), (1, 1, i + 1, H)) for i in range(a + 1)]
        if a > 0:
            head_rows(accepted_rows, h_rows[:a], [pos + 1 + i for i in range(a)])
        ttnn.deallocate(hid_last)
        hid_last = ttnn.clone(h_rows[a], memory_config=dev)
        for r in h_rows:
            ttnn.deallocate(r)
        ttnn.deallocate(h)
        out_tokens.extend(accepted_rows)
        out_tokens.append(nxt)
        pos += a + 1
        passes += 1
        if log:
            log(f"pass {passes}: drafts {drafts} accepted {a} -> +{a + 1} tokens (total {len(out_tokens)})")
    return out_tokens[:max_new_tokens], passes
