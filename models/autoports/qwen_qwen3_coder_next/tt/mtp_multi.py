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
from models.autoports.qwen_qwen3_coder_next.tt.mtp_rows import rows_pos_masks_host, rows_width
from models.tt_transformers.tt.common import Mode

GDN_T = 32  # the batched chunk form's row block per user (one tile; the fused chunk kernel's native chunk)


class RowsLen:
    """Device-resident per-user row counts of one batched GDN rows block (the multi-user analogue of PrefillLen):
    mask_f32 / mask_q [B, T, 1] (1.0 for rows < len_u) and conv_sel [B, K-1, T+K-1] (one-hot rows selecting the last
    K-1 real conv inputs of every user). `stage(lens)` is a host -> device copy (no allocation: trace-safe)."""

    def __init__(self, mesh_device, B: int, T: int, conv_k: int, q_dtype=ttnn.bfloat16):
        self.mesh, self.B, self.bucket, self.K = mesh_device, int(B), int(T), int(conv_k)
        z = lambda *shape: torch.zeros(*shape, dtype=torch.float32)
        dev = lambda t, dt: ttnn.from_torch(
            t,
            dtype=dt,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        self.mask_f32 = dev(z(self.B, self.bucket, 1), ttnn.float32)
        self.mask_q = dev(z(self.B, self.bucket, 1), q_dtype)
        self.conv_sel = dev(z(self.B, self.K - 1, self.bucket + self.K - 1), ttnn.bfloat16)
        self.lens = None

    def stage(self, lens):
        lens = [int(v) for v in lens]
        assert len(lens) == self.B and all(1 <= v <= self.bucket for v in lens), (lens, self.B, self.bucket)
        m = torch.zeros(self.B, self.bucket, 1, dtype=torch.float32)
        sel = torch.zeros(self.B, self.K - 1, self.bucket + self.K - 1, dtype=torch.float32)
        for b, n in enumerate(lens):
            m[b, :n, 0] = 1.0
            for j in range(self.K - 1):
                sel[b, j, n + j] = 1.0  # x[i] sits at padded index (K-1)+i: the window x[n-K+1 .. n-1]
        host = lambda t, dt: ttnn.from_torch(t, dtype=dt, layout=ttnn.TILE_LAYOUT)
        ttnn.copy_host_to_device_tensor(host(m, ttnn.float32), self.mask_f32)
        ttnn.copy_host_to_device_tensor(host(m, self.mask_q.dtype), self.mask_q)
        ttnn.copy_host_to_device_tensor(host(sel, ttnn.bfloat16), self.conv_sel)
        self.lens = lens

    def __lt__(self, other):  # the adapter compares an int valid_len against T; a RowsLen takes the device path first
        return NotImplemented


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


def _gdn_rows_batched(layer, x, B, R, valid_lens, saved=None, gdn_index=0):
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
    vl = valid_lens if isinstance(valid_lens, RowsLen) else list(valid_lens)
    sink = None
    if saved is not None:
        # the commit re-runs only the conv FIR + chunk scan on this pass's scan inputs (commit_scan_batched): one dict
        # per GDN layer; an existing dict's persistent buffers are copied into (traced pass), a missing one is appended
        if len(saved) > gdn_index:
            sink = saved[gdn_index]
        else:
            sink = {}
            saved.append(sink)
    out = dn.forward_prefill_batched(xb, chunk_size=int(args.gdn_chunk_size), valid_lens=vl, carry=True, scan_sink=sink)
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


def commit_multi(model, snap, saved, accepted_lens):
    """Exact per-user commit: restore the pre-pass batched GDN snapshot, then ONE batched scan-only replay per GDN
    layer (conv FIR for the taps + chunk scan on the pass's saved q/k/v/beta/g) with per-row valid lengths a_u + 1 (a
    user that accepted every draft replays to the state the pass already produced)."""
    from models.autoports.qwen_qwen3_coder_next.tt.mtp_verify import restore_gdn_state

    restore_gdn_state(model, snap)
    gdn_layers = [layer for layer in model.layers if not layer.is_full_attention]
    assert len(gdn_layers) == len(saved)
    vl = accepted_lens if isinstance(accepted_lens, RowsLen) else list(accepted_lens)
    for layer, sink in zip(gdn_layers, saved):
        dn = layer.attention
        rebuild_batched_conv_carry(dn)
        dn.commit_scan_batched(sink, vl)


def free_saved_scan(saved):
    """Free the device tensors of the per-layer scan sinks produced by a rows pass."""
    for sink in saved:
        for key, t in sink.items():
            if key != "meta" and t.is_allocated():
                ttnn.deallocate(t)
        sink.clear()


def head_step_multi(model, head, toks, hidden, positions, page_table_rows, *, rows_kv=False, masks_pos=None):
    """One head step over B rows: toks [B] ids, hidden [1,1,B,H] replicated, positions [B], page_table_rows [B, blocks].
    Returns (argmax per row (list), head hidden [1,1,B,H]). With rows_kv the K/V write is per row (masks_pos: positions
    with -1 = skip) - the head's K/V rows for accepted tokens."""
    mesh = model.mesh_device
    dev = ttnn.DRAM_MEMORY_CONFIG
    B = len(toks)
    tokens_tt, cur_pos_tt, rot_idxs, pt_tt = model.prepare_decode_inputs_host(
        torch.tensor(toks, dtype=torch.int32),
        torch.tensor([max(p, 0) for p in positions], dtype=torch.int32),
        page_table_rows,
    )
    tokens_tt, cur_pos_tt, rot_idxs, pt_tt = (
        ttnn.to_device(t, mesh, memory_config=dev) for t in (tokens_tt, cur_pos_tt, rot_idxs, pt_tt)
    )
    att = head.layer.attention
    masks = None
    if rows_kv:
        masks = ttnn.to_device(
            rows_pos_masks_host(masks_pos if masks_pos is not None else positions, B), mesh, memory_config=dev
        )
        att.rows_kv_write = True
        att.rows_pos_masks = masks
    try:
        lg, h = head.decode_step(tokens_tt, hidden, cur_pos_tt, rot_idxs, pt_tt)
    finally:
        att.rows_kv_write = False
        att.rows_pos_masks = None
    lgt = ttnn.to_torch(lg, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=-1)).float()
    V = model.args.vocab_size
    am = lgt.reshape(-1, lgt.shape[-1])[:B, :V].argmax(-1).tolist()
    for t in (tokens_tt, cur_pos_tt, rot_idxs, pt_tt, lg):
        ttnn.deallocate(t)
    if masks is not None:
        ttnn.deallocate(masks)
    return am, h


def select_rows(h_block, row_ids, H):
    """[1,1,B,H] = rows row_ids of the replicated block h_block [1,1,BW,H] (one-hot matmul; B = len(row_ids))."""
    mesh = h_block.device() if hasattr(h_block, "device") else None
    BW = h_block.shape[2]
    B = len(row_ids)
    sel = torch.zeros(1, 1, max(B, 1), BW, dtype=torch.bfloat16)
    for b, r in enumerate(row_ids):
        sel[0, 0, b, r] = 1.0
    sel_tt = ttnn.from_torch(
        sel,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=h_block.device(),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(h_block.device()),
    )
    out = ttnn.matmul(sel_tt, h_block, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(sel_tt)
    return out


def verify_pass_multi(model, rows_per_user, positions_per_user, page_table_rows, *, bw=None, saved=None, rows_len=None):
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
    h, logits = rows_block_forward(model, x, cos, sin, cur_pos_tt, pt_tt, masks, B, R, rows_len, saved)
    for t in (tokens_tt, cur_pos_tt, rot_idxs, pt_tt, cos, sin, masks):
        ttnn.deallocate(t)
    return h, logits, R


def rows_block_forward(model, x, cos, sin, cur_pos_tt, pt_tt, masks, B, R, rows_len, saved):
    """The layers over an embedded user-major rows block x [1,1,BW,dim/TP] (device inputs given): returns
    (final-normed hidden [1,1,BW,H] replicated, vocab-sharded logits). Shared by the eager pass and the traced body."""
    gi = 0
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
            y = _gdn_rows_batched(
                layer, x, B, R, rows_len if rows_len is not None else [R] * B, saved=saved, gdn_index=gi
            )
            gi += 1
        ttnn.deallocate(x)
        x = y
    h = model._final_norm_decode(x)
    ttnn.deallocate(x)
    logits = ttnn.linear(h, model.lm_head_weight, compute_kernel_config=model._lm_head_ckc())
    return h, logits


def prefill_multi_with_head(gen, head, prompts):
    """Prefill B prompts into their slots (the generator's multi-slot path), running the head's prompt rows for each user
    from the bucket hidden the `_run_bucket_hidden` hook exposes (one call per user in that path). Returns
    (tok0 per user, hidden per user [1,1,1,H] replicated = the hidden that produced tok0, P per user)."""
    model = gen.model
    mesh = model.mesh_device
    dev = ttnn.DRAM_MEMORY_CONFIG
    H = gen.args.dim
    B = len(prompts)
    P = [len(p) for p in prompts]
    T = max(P)
    toks = torch.zeros(B, T, dtype=torch.int64)
    for u in range(B):
        toks[u, : P[u]] = torch.tensor(prompts[u])
    captured = []
    orig = model._run_bucket_hidden

    def hook(bi, bucket, chunk_start):
        x = orig(bi, bucket, chunk_start)
        captured.append(ttnn.clone(x, memory_config=dev))
        return x

    model._run_bucket_hidden = hook
    try:
        gen.reset()
        logits0 = gen.prefill_forward(toks, page_table=gen.page_table[:B], prompt_lens=P)
    finally:
        model._run_bucket_hidden = orig
    assert len(captured) == B, f"expected one bucket hidden per user, got {len(captured)} (multi-segment prompts?)"
    tok0 = [int(torch.argmax(logits0[u, -1]).item()) for u in range(B)]
    hids = []
    for u in range(B):
        h_all = model.norm(captured[u], mode=Mode.PREFILL)  # replicated [1,1,bucket,H]
        ttnn.deallocate(captured[u])
        pt_u = gen.page_table[u : u + 1]
        for t in range(P[u] - 1):  # rows (token t+1, hidden t) at positions 1..P-1
            hid = ttnn.slice(h_all, (0, 0, t, 0), (1, 1, t + 1, H))
            _, hn = head_step_multi(model, head, [prompts[u][t + 1]], hid, [t + 1], pt_u)
            ttnn.deallocate(hn)
            ttnn.deallocate(hid)
        hids.append(ttnn.clone(ttnn.slice(h_all, (0, 0, P[u] - 1, 0), (1, 1, P[u], H)), memory_config=dev))
        ttnn.deallocate(h_all)
    return tok0, hids, P


def mtp_generate_multi(gen, head, prompts, max_new_tokens, k, *, log=None, draft_fns=None, prefilled=None):
    """Eager multi-user MTP loop (greedy): every user drafts k tokens (batched head steps), one batched verify pass over
    all users' k + 1 rows, per-user accept, one batched commit with per-row accepted lengths, batched head K/V rows for
    the accepted tokens. Returns (tokens per user, passes). `draft_fns[u](tokens_u)` replaces the head for user u."""
    from models.autoports.qwen_qwen3_coder_next.tt.mtp_verify import free_gdn_snapshot, snapshot_gdn_state

    model = gen.model
    mesh = model.mesh_device
    dev = ttnn.DRAM_MEMORY_CONFIG
    H = gen.args.dim
    B = len(prompts)
    R = k + 1
    tok0, hids, P = prefilled if prefilled is not None else prefill_multi_with_head(gen, head, prompts)
    out = [[tok0[u]] for u in range(B)]
    pos = list(P)
    hid = ttnn.concat(hids, dim=2)  # [1,1,B,H]
    if prefilled is None:
        for h in hids:
            ttnn.deallocate(h)
    pt = gen.page_table[:B]
    passes = 0
    while min(len(o) for o in out) < max_new_tokens:
        # drafts: k batched head steps chained on the head's own hidden
        drafts = [[] for _ in range(B)]
        cur_tok = [o[-1] for o in out]
        cur_hid = hid
        for j in range(k):
            am, h_next = head_step_multi(model, head, cur_tok, cur_hid, [pos[u] + j for u in range(B)], pt)
            if draft_fns is not None:
                am = [int(draft_fns[u](out[u] + drafts[u])[0]) if draft_fns[u] is not None else am[u] for u in range(B)]
            for u in range(B):
                drafts[u].append(int(am[u]))
            if cur_hid is not hid:
                ttnn.deallocate(cur_hid)
            cur_hid, cur_tok = h_next, [int(a) for a in am]
        if cur_hid is not hid:
            ttnn.deallocate(cur_hid)
        rows = [[out[u][-1]] + drafts[u] for u in range(B)]
        positions = [[pos[u] + i for i in range(R)] for u in range(B)]
        snap = snapshot_gdn_state(model)
        saved = []
        h_blk, lg_tt, _ = verify_pass_multi(model, rows, positions, pt, saved=saved)
        BW = h_blk.shape[2]
        lg = ttnn.to_torch(lg_tt, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=-1)).float()
        lg = lg.reshape(-1, lg.shape[-1])[:, : gen.args.vocab_size]
        ttnn.deallocate(lg_tt)
        acc, nxt = [], []
        for u in range(B):
            preds = lg[u * R : (u + 1) * R].argmax(-1).tolist()
            a = 0
            for d, p in zip(drafts[u], preds):
                if d != p:
                    break
                a += 1
            acc.append(a)
            nxt.append(preds[a])
        if any(a < k for a in acc):
            commit_multi(model, snap, saved, [a + 1 for a in acc])
        free_saved_scan(saved)
        free_gdn_snapshot(snap)
        # head K/V rows for the accepted tokens: row u*R+i (i = 1..a_u) = (token rows[u][i], hidden row u*R+i-1)
        row_ids, toks_rows, pos_rows = [], [], []
        for u in range(B):
            for i in range(1, acc[u] + 1):
                row_ids.append(u * R + i - 1)
                toks_rows.append(rows[u][i])
                pos_rows.append(pos[u] + i)
        if row_ids:
            # pad the row set to a served decode width (positions -1 = the K/V write skips them)
            n = rows_width(len(row_ids))
            pad = n - len(row_ids)
            hid_rows = select_rows(h_blk, row_ids + [row_ids[-1]] * pad, H)  # [1,1,n,H]
            pt_rows = torch.cat([pt[r // R : r // R + 1] for r in row_ids + [row_ids[-1]] * pad], dim=0)
            _, hn = head_step_multi(
                model,
                head,
                toks_rows + [0] * pad,
                hid_rows,
                pos_rows + [0] * pad,
                pt_rows,
                rows_kv=True,
                masks_pos=pos_rows + [-1] * pad,
            )
            ttnn.deallocate(hn)
            ttnn.deallocate(hid_rows)
        # next drafts start from each user's accepted row hidden
        new_hid = select_rows(h_blk, [u * R + acc[u] for u in range(B)], H)
        ttnn.deallocate(h_blk)
        ttnn.deallocate(hid)
        hid = new_hid
        for u in range(B):
            out[u].extend(rows[u][1 : acc[u] + 1])
            out[u].append(nxt[u])
            pos[u] += acc[u] + 1
        passes += 1
        if log:
            log(f"pass {passes}: accepted {acc} drafts {drafts} -> lengths {[len(o) for o in out]}")
    ttnn.deallocate(hid)
    return [o[:max_new_tokens] for o in out], passes
