# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""MTP serving: the head's prompt rows during prefill, the traced draft / verify / commit step per decode.

    QWEN35_MTP=k (k >= 1) turns it on in generator_vllm; the plugin's variable-width output shim `mtp_plugin_shim`
    lets a step return 1..k+1 tokens per request. B = --max-num-seqs slots: B == 1 runs the single-user traced step
    (tt/mtp_traced.py), B > 1 the traced multi-user step over all slots (tt/mtp_traced_multi.py; B x (k+1) <= 32).
    Programs compile in `warm()` / `warm_prefill_path()` and the traces are captured in `capture()` - before any request,
    after every persistent allocation (the head is built at model construction, its KV allocated with the model's).

    Prefill: the masked-bucket prefill runs EAGERLY in this mode (no bucket traces: the head needs every segment's
    final-normed hidden, taken from the `_run_bucket_hidden` hook; the served multi-slot prefill runs the queued
    requests one after another, chunk_start 0 first for each), then the head writes its K/V for rows (token t+1,
    hidden t) at positions 1..P-1 in 32-row groups (a fixed program per bucket: rows selected by a one-hot matmul);
    hidden P-1 becomes the slot's first draft input.
    Decode: `step` (single) / `step_multi` (slots) -> per request the accepted tokens + the verified next token.
"""
from __future__ import annotations

import os
import time

import torch
from loguru import logger

import ttnn
from models.autoports.qwen_qwen3_5_122b_a10b.tt.mtp import Qwen35MtpHead
from models.autoports.qwen_qwen3_5_122b_a10b.tt.mtp_rows import rows_pos_masks_host
from models.autoports.qwen_qwen3_5_122b_a10b.tt.mtp_traced import MtpTraced, _host_i32, _host_u32
from models.tt_transformers.tt.common import Mode

_DRAM = ttnn.DRAM_MEMORY_CONFIG
ROWS = 32  # head prompt rows per call (one tile of decode rows)


class MtpServing:
    def __init__(self, gen, k: int):
        self.gen, self.k = gen, int(k)
        self.B = int(gen.max_batch_size)
        self.multi = self.B > 1
        self.model = gen.model
        self.mesh = self.model.mesh_device
        self.H = int(gen.args.dim)
        self.head = Qwen35MtpHead(self.model, gen.args.load_state_dict())
        self.runner = None
        self._pos = None  # single user: position of the token the next drafts extend (== the plugin's start_pos)
        self._tok = None
        self._stats = None
        self._sel = {}  # bucket -> persistent one-hot [1,1,ROWS,bucket] selector (head prompt rows)
        self._prompt = None
        self._prefill_queue = []  # (slot, prompt, page-table row) of the requests being prefilled, in order
        self._hid_rows = {}  # multi: slot -> device [1,1,1,H] hidden that produced the slot's first token
        self._slot_rows = {}  # multi: slot -> page-table row at prefill
        self._cur_slot = 0
        self._hook_installed = False
        self._hid_ready = False
        # QWEN35_MTP_TRACED_PREFILL=1 (default): the head's prompt rows run INSIDE the bucket program (one more layer per
        # segment, captured with the bucket traces: the eager-bucket path cost ~60 ms of TTFT at 128 tokens); 0 = the
        # eager-bucket path with the python hook and 32-row head groups
        self.in_trace = os.environ.get("QWEN35_MTP_TRACED_PREFILL", "1") == "1"
        self.hid_out = None  # [1,1,1,H] replicated: the hidden that produced the request's first token (in-trace mode)
        logger.info(
            f"MTP serving: k={self.k}, slots={self.B} ({'multi' if self.multi else 'single'}), head layer {self.head.layer_idx}"
        )

    # ------------------------------------------------------------------ allocation / warm-up
    def allocate_kv(self, kv_shape, dtype):
        self.head.allocate_kv_cache(kv_shape, dtype)

    def _ensure_runner(self):
        if self.runner is not None:
            return
        if self.multi:
            from models.autoports.qwen_qwen3_5_122b_a10b.tt.mtp_traced_multi import MtpTracedMulti

            self.runner = MtpTracedMulti(self.gen, self.head, self.k, self.B)
        else:
            self.runner = MtpTraced(self.gen, self.head, self.k)
        m = self.model
        dev = lambda t: ttnn.to_device(t, self.mesh, memory_config=_DRAM)
        pt = self.gen.page_table[:1]
        tok, pos, rot, ptt = m.prepare_decode_inputs_host(
            torch.zeros(ROWS, dtype=torch.int32), torch.zeros(ROWS, dtype=torch.int32), pt.repeat(ROWS, 1)
        )
        self.pr_tok, self.pr_pos, self.pr_rot, self.pr_pt = dev(tok), dev(pos), dev(rot), dev(ptt)
        self.pr_masks = dev(rows_pos_masks_host([0] * ROWS, ROWS))
        if self.in_trace:
            self.hid_out = ttnn.from_torch(
                torch.zeros(1, 1, 1, self.H, dtype=torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh,
                memory_config=_DRAM,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
            )
            m._mtp_bucket_hook = self._bucket_head  # device side, inside every bucket program (and its trace)
            m._mtp_stage_hook = self._bucket_stage  # host side, after the bucket's inputs are staged
        for b in self.model._PREFILL_MASK_BUCKETS:
            self._sel[b] = ttnn.from_torch(
                torch.zeros(1, 1, ROWS, b, dtype=torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh,
                memory_config=_DRAM,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
            )

    def warm(self):
        """Compile every MTP program (the step programs + the head prompt rows for every bucket)."""
        self._ensure_runner()
        self.runner.warm()
        for b in self.model._PREFILL_MASK_BUCKETS:
            h = ttnn.from_torch(
                torch.zeros(1, 1, b, self.H, dtype=torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh,
                memory_config=_DRAM,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
            )
            self._head_rows_group(h, b, 0, list(range(1, ROWS + 1)), [1] * ROWS)
            self._store_hidden(h, b, 0)
            self._drop_hid_rows()
            ttnn.deallocate(h)
        ttnn.synchronize_device(self.mesh)

    def _drop_hid_rows(self):
        for t in self._hid_rows.values():
            ttnn.deallocate(t)
        self._hid_rows = {}

    def warm_prefill_path(self):
        """Compile everything a served prefill runs in this mode: the eager masked bucket with the sharded logits, the
        hook's norm on the bucket hidden and the head's prompt rows / hidden selection - for every bucket at its full and
        a masked length (the program guard counts anything compiled after the captures as a fault)."""
        self._ensure_runner()
        pt = self.gen.page_table[:1]
        # a batched model prefills on the B = 1 GDN scratch (prefill_serving binds it per request); the eager single-user
        # prefill on the batched [B, ...] buffers would copy a [1, ...] state into them (shape mismatch)
        prev = self.model._bind_gdn_prefill_scratch() if self.multi else None
        try:
            for b in self.model._PREFILL_MASK_BUCKETS:
                for n in (b, max(2, b - 1)):
                    ids = [1000 + (i % 977) for i in range(n)]
                    self.begin_prefill(ids, pt[0], slot=0)
                    try:
                        if self.multi:
                            self.model._reset_gdn_state_for_new_sequence()
                        lg, sh = self.gen._prefill_single(
                            torch.tensor(ids, dtype=torch.int32).reshape(1, -1), pt, want_sharded=True
                        )
                    finally:
                        self.abort_prefill()
        finally:
            if prev is not None:
                self.model._unbind_gdn_prefill_scratch(prev)
        if not self.multi:
            self.model._reset_gdn_state_for_new_sequence()
        ttnn.synchronize_device(self.mesh)

    def capture(self):
        self._ensure_runner()
        self.runner.capture()
        logger.info("MTP traces captured (pass, commit, draft, head rows)")

    # ------------------------------------------------------------------ prefill side
    def _select_rows(self, h, bucket, j0):
        """[1,1,ROWS,H] = rows j0..j0+ROWS-1 of the replicated bucket hidden h [1,1,bucket,H] (one-hot matmul)."""
        sel = torch.zeros(1, 1, ROWS, bucket, dtype=torch.bfloat16)
        for r in range(ROWS):
            if j0 + r < bucket:
                sel[0, 0, r, j0 + r] = 1.0
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(sel, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT), self._sel[bucket]
        )
        return ttnn.matmul(self._sel[bucket], h, memory_config=_DRAM)

    def _head_rows_group(self, h, bucket, j0, tokens, positions):
        """Head K/V rows r = 0..ROWS-1: (token tokens[r], hidden h[j0 + r]) at positions[r] (-1 = skip the write)."""
        rows_h = self._select_rows(h, bucket, j0)
        S = self.pr_tok.shape[-1]
        w = ttnn.copy_host_to_device_tensor
        w(_host_u32(tokens, (1, 1, 1, S)), self.pr_tok)
        w(_host_i32([max(p, 0) for p in positions], (ROWS,)), self.pr_pos)
        w(_host_u32([max(p, 0) for p in positions], (1, ROWS)), self.pr_rot)
        w(rows_pos_masks_host(positions, ROWS), self.pr_masks)
        att = self.head.layer.attention
        att.rows_kv_write = True
        att.rows_pos_masks = self.pr_masks
        try:
            lg, hn = self.head.decode_step(self.pr_tok, rows_h, self.pr_pos, self.pr_rot, self.pr_pt)
        finally:
            att.rows_kv_write = False
            att.rows_pos_masks = None
        for t in (lg, hn, rows_h):
            ttnn.deallocate(t)

    def _store_hidden(self, h, bucket, j):
        """The hidden that produced the request's first sampled token (h[j]): single user -> the runner's draft hidden
        input; multi -> kept per slot until end_prefill hands it to the runner."""
        sel = torch.zeros(1, 1, ROWS, bucket, dtype=torch.bfloat16)
        sel[0, 0, 0, j] = 1.0
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(sel, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT), self._sel[bucket]
        )
        rows_h = ttnn.matmul(self._sel[bucket], h, memory_config=_DRAM)  # row 0 = h[j]
        one = ttnn.slice(rows_h, (0, 0, 0, 0), (1, 1, 1, self.H))
        if self.multi:
            if self._cur_slot in self._hid_rows:
                ttnn.deallocate(self._hid_rows[self._cur_slot])
            self._hid_rows[self._cur_slot] = ttnn.clone(one, memory_config=_DRAM)
        else:
            ttnn.copy(one, self.runner.d_hid)
        ttnn.deallocate(one)
        ttnn.deallocate(rows_h)

    def set_page_table(self, row):
        """The request's page-table row (vLLM block ids, host int32 [blocks]) into the persistent page-table inputs the
        request's prefill and (single user) step use; the multi-user runner keeps its own per-slot rows."""
        r = self.runner
        row = torch.as_tensor(row).to(torch.int32).reshape(1, -1)
        bufs = (self.pr_pt,) if self.multi else (r.p_pt, r.r_pt, self.pr_pt, r.d_pt)
        for buf in bufs:
            n = buf.shape[0]
            host = ttnn.from_torch(row.repeat(n, 1), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
            ttnn.copy_host_to_device_tensor(host, buf)

    # ---- in-trace head prefill (one extra layer per bucket segment) ----------------------------------------------
    def _bucket_stage(self, bi, bucket, cs, actual_len):
        """Host writes for the head's segment: the first-draft one-hot (row P-1 when it lies in this segment) and, at a
        request's first segment, the zero 'previous hidden'. A queued request is taken at chunk_start 0."""
        if cs == 0 and self._prefill_queue:
            if self.multi and self._hid_ready and self._prompt is not None:
                # several requests in one prefill call: harvest the previous request's first-draft hidden to host
                # before its hid_out is overwritten (the read waits for the previous replays)
                self._hid_rows[self._cur_slot] = (
                    ttnn.to_torch(ttnn.get_device_tensors(self.hid_out)[0]).reshape(-1)[: self.H].clone()
                )
            self._cur_slot, self._prompt, row = self._prefill_queue.pop(0)
            self._slot_rows[self._cur_slot] = row
            self._hid_ready = False
        P = len(self._prompt) if self._prompt is not None else 0
        sel = torch.zeros(1, 1, 32, bucket, dtype=torch.bfloat16)
        if P and cs <= P - 1 < cs + bucket:
            sel[0, 0, 0, P - 1 - cs] = 1.0
            self._hid_ready = True
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(sel, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT), bi["mtp_sel"]
        )
        if cs == 0:
            ttnn.copy_host_to_device_tensor(
                ttnn.from_torch(
                    torch.zeros(1, 1, 1, self.H, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
                ),
                bi["mtp_prev"],
            )

    def _bucket_head(self, bi, bucket, cs, x):
        """Device side (captured): the head over this segment's rows. Row j = (token at position cs + j, hidden of
        position cs + j - 1) written at position cs + j: the same positions / page tables / rope as the main layers (row 0
        of a request = token 0 with a zero previous hidden; the head's own rows start at position 1 as the eager path's).
        Also hands the hidden of the prompt's last position to the first draft (hid_out) and carries this segment's last
        hidden into the next segment's row 0."""
        m, head = self.model, self.head
        h = m.norm(ttnn.clone(x, memory_config=_DRAM), mode=Mode.PREFILL)  # replicated [1,1,bucket,H]
        # first-draft hidden: one-hot row 0 picks hidden P-1 (zeros when P-1 is not in this segment)
        picked = ttnn.matmul(bi["mtp_sel"], h, memory_config=_DRAM)  # [1,1,32,H]
        one = ttnn.slice(picked, (0, 0, 0, 0), (1, 1, 1, self.H))
        ttnn.copy(one, self.hid_out)
        ttnn.deallocate(one)
        ttnn.deallocate(picked)
        # shifted hidden rows [prev, h[0 .. bucket-2]]: a tile-layout concat of 1 + (bucket-1) rows is not tile aligned,
        # so shift in ROW_MAJOR and tilize the result
        last = ttnn.slice(h, (0, 0, bucket - 1, 0), (1, 1, bucket, self.H))
        h_rm = ttnn.to_layout(h, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(h)
        head_rows = ttnn.slice(h_rm, (0, 0, 0, 0), (1, 1, bucket - 1, self.H))
        ttnn.deallocate(h_rm)
        prev_rm = ttnn.to_layout(bi["mtp_prev"], ttnn.ROW_MAJOR_LAYOUT)
        cat_rm = ttnn.concat([prev_rm, head_rows], dim=2, memory_config=_DRAM)  # [1,1,bucket,H]
        h_shift = ttnn.to_layout(cat_rm, ttnn.TILE_LAYOUT)
        ttnn.copy(last, bi["mtp_prev"])
        for t in (head_rows, last, prev_rm, cat_rm):
            ttnn.deallocate(t)
        # mix: fc(cat(norm_e(embed(tok)), norm_h(hidden))) in prefill form -> the layer's fractured input
        e = m.embd(bi["tok"])  # [1, bucket, H/TP]
        e = ttnn.reshape(e, (1, 1, bucket, e.shape[-1]))
        e = ttnn.to_memory_config(e, _DRAM)
        e_n = head.pre_fc_norm_embedding(e, mode=Mode.PREFILL)  # replicated [1,1,bucket,H]
        ttnn.deallocate(e)
        h_n = head.pre_fc_norm_hidden(h_shift, mode=Mode.PREFILL)
        ttnn.deallocate(h_shift)
        xin = ttnn.concat([e_n, h_n], dim=-1, memory_config=_DRAM)  # [1,1,bucket,2H]
        ttnn.deallocate(e_n)
        ttnn.deallocate(h_n)
        x_mix = ttnn.linear(xin, head.fc_weight, memory_config=_DRAM, dtype=ttnn.bfloat16)  # [1,1,bucket,H/TP]
        ttnn.deallocate(xin)
        out = head.layer.forward(
            x_mix,
            cos=bi["cos"],
            sin=bi["sin"],
            mode="prefill",
            page_table=bi["full_pt"],
            chunk_page_table=bi["chunk_pt"],
            chunk_start_idx=cs,
            chunk_start_idx_tensor=bi["csi"],
            valid_len=bi["len"],
        )
        ttnn.deallocate(x_mix)
        ttnn.deallocate(out)

    def begin_prefill(self, prompt_ids, page_table_row, slot: int = 0):
        """Register one request's prefill (slot, prompt, page-table row) and install the bucket-hidden hook; the served
        prefill runs the queued requests in order (one bucket call per segment, chunk_start 0 first for each) and
        `end_prefill` hands every slot's first token / hidden to the runner."""
        self._prefill_queue.append((int(slot), list(prompt_ids), torch.as_tensor(page_table_row).to(torch.int32)))
        if self.in_trace:
            self._hid_ready = False
            return  # the bucket program carries the head rows; _bucket_stage takes the queue at chunk_start 0
        if self._hook_installed:
            return
        self._hid_ready = False
        model = self.model
        orig = model._run_bucket_hidden
        mtp = self

        def hook(bi, bucket, chunk_start):
            x = orig(bi, bucket, chunk_start)
            if chunk_start == 0:  # the next queued request starts
                mtp._cur_slot, mtp._prompt, row = mtp._prefill_queue.pop(0)
                mtp._slot_rows[mtp._cur_slot] = row
                mtp.set_page_table(row)
                mtp._hid_ready = False
            h = model.norm(ttnn.clone(x, memory_config=_DRAM), mode=Mode.PREFILL)  # replicated [1,1,bucket,H]
            mtp._prompt_rows(h, bucket, chunk_start)
            ttnn.deallocate(h)
            return x

        model._run_bucket_hidden = hook
        self._orig_hook = orig
        self._hook_installed = True

    def _prompt_rows(self, h, bucket, cs):
        """Head rows for the prompt positions in this segment: row j (hidden of prompt position cs + j) pairs with the
        token at cs + j + 1 and is written at position cs + j + 1 (only while cs + j + 1 <= P - 1)."""
        P = len(self._prompt)
        n_seg = min(bucket, P - cs)
        for j0 in range(0, n_seg, ROWS):
            toks, poss = [], []
            for r in range(ROWS):
                j = j0 + r
                p = cs + j + 1
                if j < n_seg and p <= P - 1:
                    toks.append(self._prompt[p])
                    poss.append(p)
                else:
                    toks.append(0)
                    poss.append(-1)
            if any(p >= 0 for p in poss):
                self._head_rows_group(h, bucket, j0, toks, poss)
        if cs <= P - 1 < cs + bucket:
            self._store_hidden(h, bucket, P - 1 - cs)
            self._hid_ready = True

    def abort_prefill(self):
        if not self.in_trace:
            self.model._run_bucket_hidden = self._orig_hook
        self._hook_installed = False
        self._prefill_queue = []
        self._prompt = None
        self._drop_hid_rows()

    def _log_stats(self, why):
        st = self._stats
        if st and st["passes"]:
            ms = st["wall"] / st["passes"] * 1e3
            logger.info(
                f"MTP_STATS ({why}): {st['passes']} passes, {st['tokens']} tokens = {st['tokens'] / st['passes']:.2f} per pass, "
                f"acceptance hist {st['hist']}, {ms:.1f} ms per pass -> {st['tokens'] / st['wall']:.1f} tok/s (device steps only)"
            )

    def end_prefill(self, tok0, P, slots=None):
        """Single user: (tok0, P) scalars. Multi: lists per prefilled request with their slots."""
        if not self.in_trace:
            self.model._run_bucket_hidden = self._orig_hook
        self._hook_installed = False
        assert not self._prefill_queue, f"{len(self._prefill_queue)} queued request(s) were not prefilled"
        assert self._hid_ready, "the prefill hook did not see the last prompt position"
        self._prompt = None
        if self.multi:
            for t0, p, slot in zip(tok0, P, slots):
                slot = int(slot)
                if self.in_trace:
                    # the last request's hidden is still in hid_out; earlier ones were harvested by _bucket_stage
                    row = self._hid_rows.pop(slot, None)
                    if row is None:
                        row = ttnn.to_torch(ttnn.get_device_tensors(self.hid_out)[0]).reshape(-1)[: self.H].clone()
                    self.runner.set_slot(slot, int(t0), int(p), self._slot_rows[slot], row)
                    continue
                row = self._hid_rows.pop(slot)
                self.runner.set_slot(slot, int(t0), int(p), self._slot_rows[slot], row)
                ttnn.deallocate(row)
            if self._stats is None:
                self._stats = {"passes": 0, "tokens": 0, "hist": [0] * (self.k + 1), "wall": 0.0}
            return
        self._log_stats("previous request")
        self._stats = {"passes": 0, "tokens": 0, "hist": [0] * (self.k + 1), "wall": 0.0}
        self._sampler = None  # the next request gets its own sampler / RNG
        if self.in_trace:
            ttnn.copy(self.hid_out, self.runner.d_hid)
        self._pos = int(P)
        self._tok = int(tok0)

    # ------------------------------------------------------------------ decode side
    def step(self, tok: int, pos: int, page_table_row=None, host_authoritative=False, sampling=None) -> list[int]:
        """Single user. Returns the accepted drafts followed by the verified next token (1..k+1 ids). The model owns the
        continuation (its last emitted token / position and the head's hidden for it); the plugin's `tok` / `pos` are
        only checked when it flags the host inputs authoritative (`reload_inputs` / `reset_batch`: the first decode after
        a prefill) - on the steady async path its host token can lag the device by a step."""
        t0 = time.perf_counter()
        r = self.runner
        if page_table_row is not None:
            self.set_page_table(page_table_row)  # vLLM may have grown the request's block list
        if host_authoritative and self._pos is not None and (pos != self._pos or tok != self._tok):
            logger.warning(
                f"MTP step input (tok {tok}, pos {pos}) differs from the model's (tok {self._tok}, pos {self._pos})"
            )
        tok, pos = self._tok if self._tok is not None else tok, self._pos if self._pos is not None else pos
        # sampling = (temperature, top_k, top_p, seed) of the request: temperature > 0 -> exact speculative sampling
        # (sampled drafts, rejection-sampling accept on the read-back logits rows); else greedy (argmax compare)
        sampler = self._sampler_for(sampling)
        if sampler is not None and not sampler.greedy:
            drafts, qs = r.draft(tok, pos, sampler=sampler)
            rows = [tok] + drafts
            preds, lg = r.verify(rows, pos, want_logits=True)
            a, nxt = sampler.accept(lg, drafts, qs)
        else:
            drafts = r.draft(tok, pos)
            rows = [tok] + drafts
            preds = r.verify(rows, pos)
            a = 0
            for d, p in zip(drafts, preds):
                if d != p:
                    break
                a += 1
            nxt = preds[a]
        if a < self.k:
            r.commit(a + 1)
        if a > 0:
            r.head_rows(rows[1 : a + 1], pos)
        r._select_hidden(a)
        out = rows[1 : a + 1] + [nxt]
        self._pos = pos + a + 1
        self._tok = nxt
        st = self._stats
        if st is not None:
            st["passes"] += 1
            st["tokens"] += len(out)
            st["hist"][a] += 1
            st["wall"] += time.perf_counter() - t0
            if st["passes"] % 32 == 0:
                self._log_stats("running")
        return out

    def _sampler_for(self, sampling):
        """One SpecSampler per request (its RNG persists across the request's steps); None = greedy."""
        if sampling is None:
            return None
        from models.autoports.qwen_qwen3_5_122b_a10b.tt.mtp_sampling import SpecSampler

        key = tuple(sampling)
        cur = getattr(self, "_sampler", None)
        if cur is None or cur[0] != key:  # a new request (end_prefill clears it) or changed params: fresh RNG
            t, k, p, seed = sampling
            cur = (key, SpecSampler(t, k, p, seed))
            self._sampler = cur
        return cur[1]

    def step_multi(self, positions, page_table, slot_remap=None):
        """B slots. positions: the plugin's [B_pad] start positions (-1 = no request in that slot); page_table:
        [B_pad, W] host rows. Returns per slot the emitted ids ([] for slots without a request)."""
        t0 = time.perf_counter()
        r = self.runner
        if slot_remap is not None:
            r.remap_slots([int(x) for x in torch.as_tensor(slot_remap).reshape(-1).tolist()])
        pos = torch.as_tensor(positions).reshape(-1).tolist()
        pt = page_table if isinstance(page_table, torch.Tensor) else ttnn.to_torch(page_table)
        for u in range(self.B):
            if u >= len(pos) or int(pos[u]) < 0:
                r.live[u] = False
            elif not r.live[u]:
                # the plugin's warm-up dummy decode (and any slot it considers live that we never prefilled): run it as a
                # well-formed slot with token 0 and a zero hidden so the step emits its rows (garbage, never served)
                if not getattr(self, "_phantom_logged", False):
                    self._phantom_logged = True
                    logger.info(
                        f"MTP: slot {u} live for the plugin (pos {pos[u]}) without a prefill here: warm-up dummy step"
                    )
                r.set_slot(u, 0, int(pos[u]), pt[u], torch.zeros(self.H, dtype=torch.bfloat16))
        emitted = r.step(page_table_rows=[pt[u] for u in range(self.B)])
        st = self._stats
        if st is not None:
            st["passes"] += 1
            st["tokens"] += sum(len(e) for e in emitted)
            for e in emitted:
                if e:
                    st["hist"][len(e) - 1] += 1
            st["wall"] += time.perf_counter() - t0
            if st["passes"] % 32 == 0:
                self._log_stats("running")
        return emitted
