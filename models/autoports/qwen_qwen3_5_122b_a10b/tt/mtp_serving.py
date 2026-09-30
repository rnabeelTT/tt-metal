# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Single-user MTP serving: the head's prompt rows during prefill, the traced draft / verify / commit step per decode.

    QWEN35_MTP=k (k >= 1) turns it on in generator_vllm (`--max-num-seqs 1`; greedy drafts / accept; the plugin's
    variable-width output shim `mtp_plugin_shim` lets a step return 1..k+1 tokens). Programs compile in `warm()` and
    the four traces are captured in `capture()` - both before any request, after every persistent allocation (the head
    is built at model construction and its KV allocated with the model's KV).

    Prefill: the masked-bucket prefill runs EAGERLY in this mode (no bucket traces: the head needs every segment's
    final-normed hidden, taken from the `_run_bucket_hidden` hook), then the head writes its K/V for rows
    (token t+1, hidden t) at positions 1..P-1 in 32-row groups (a fixed head "rows" program per bucket: the rows are
    selected by a one-hot matmul, never a per-offset slice); hidden P-1 becomes the first draft's input.
    Decode: `step(tok, pos)` -> the accepted tokens + the verified next token (1..k+1 ids).
"""
from __future__ import annotations

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
        assert gen.max_batch_size == 1, "MTP serving is single-user (--max-num-seqs 1)"
        self.gen, self.k = gen, int(k)
        self.model = gen.model
        self.mesh = self.model.mesh_device
        self.H = int(gen.args.dim)
        self.head = Qwen35MtpHead(self.model, gen.args.load_state_dict())
        self.runner = None
        self._pos = None  # position of the token the next drafts extend (== the plugin's start_pos)
        self._tok = None
        self._stats = None  # per-request: passes, tokens, acceptance histogram, wall
        self._hid_ready = False
        self._sel = {}  # bucket -> persistent one-hot [1,1,ROWS,bucket] selector (head prompt rows)
        self._prompt = None
        logger.info(f"MTP serving: k={self.k}, head layer {self.head.layer_idx}")

    # ------------------------------------------------------------------ allocation / warm-up
    def allocate_kv(self, kv_shape, dtype):
        self.head.allocate_kv_cache(kv_shape, dtype)

    def _ensure_runner(self):
        if self.runner is None:
            self.runner = MtpTraced(self.gen, self.head, self.k)
            m = self.model
            dev = lambda t: ttnn.to_device(t, self.mesh, memory_config=_DRAM)
            pt = self.gen.page_table[:1]
            tok, pos, rot, ptt = m.prepare_decode_inputs_host(
                torch.zeros(ROWS, dtype=torch.int32), torch.zeros(ROWS, dtype=torch.int32), pt.repeat(ROWS, 1)
            )
            self.pr_tok, self.pr_pos, self.pr_rot, self.pr_pt = dev(tok), dev(pos), dev(rot), dev(ptt)
            self.pr_masks = dev(rows_pos_masks_host([0] * ROWS, ROWS))
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
        """Compile every MTP program (the four step programs + the head prompt rows for every bucket)."""
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
            ttnn.deallocate(h)
        ttnn.synchronize_device(self.mesh)

    def warm_prefill_path(self):
        """Compile everything a served prefill runs in this mode: the eager masked bucket with the sharded logits, the
        hook's norm on the bucket hidden and the head's prompt rows / hidden selection - for every bucket at its full and
        a masked length (the program guard counts anything compiled after the captures as a fault)."""
        self._ensure_runner()
        pt = self.gen.page_table[:1]
        for b in self.model._PREFILL_MASK_BUCKETS:
            for n in (b, max(2, b - 1)):
                ids = [1000 + (i % 977) for i in range(n)]
                self.begin_prefill(ids, pt[0])
                try:
                    lg, sh = self.gen._prefill_single(
                        torch.tensor(ids, dtype=torch.int32).reshape(1, -1), pt, want_sharded=True
                    )
                    ttnn.deallocate(sh) if hasattr(sh, "is_allocated") and sh.is_allocated() else None
                finally:
                    self.abort_prefill()
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
        self.runner._write(self.pr_tok, _host_u32(tokens, (1, 1, 1, S)))
        self.runner._write(self.pr_pos, _host_i32([max(p, 0) for p in positions], (ROWS,)))
        self.runner._write(self.pr_rot, _host_u32([max(p, 0) for p in positions], (1, ROWS)))
        self.runner._write(self.pr_masks, rows_pos_masks_host(positions, ROWS))
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
        """d_hid <- h[j] (the hidden that produced the first sampled token) via the one-hot selector."""
        sel = torch.zeros(1, 1, ROWS, bucket, dtype=torch.bfloat16)
        sel[0, 0, 0, j] = 1.0
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(sel, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT), self._sel[bucket]
        )
        rows_h = ttnn.matmul(self._sel[bucket], h, memory_config=_DRAM)  # row 0 = h[j]
        one = ttnn.slice(rows_h, (0, 0, 0, 0), (1, 1, 1, self.H))
        ttnn.copy(one, self.runner.d_hid)
        ttnn.deallocate(one)
        ttnn.deallocate(rows_h)

    def set_page_table(self, row):
        """The request's page-table row (vLLM block ids, host int32 [blocks]) into every persistent page-table input
        (pass rows, draft, head rows, head prompt rows); the widths match the generator's page table."""
        r = self.runner
        row = torch.as_tensor(row).to(torch.int32).reshape(1, -1)
        for buf in (r.p_pt, r.r_pt, self.pr_pt, r.d_pt):
            n = buf.shape[0]
            host = ttnn.from_torch(row.repeat(n, 1), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
            ttnn.copy_host_to_device_tensor(host, buf)

    def begin_prefill(self, prompt_ids: list[int], page_table_row):
        """Install the bucket-hidden hook for one request's prefill; `end_prefill(tok0)` removes it."""
        self._prompt = list(prompt_ids)
        self._hid_ready = False
        self.set_page_table(page_table_row)
        model = self.model
        orig = model._run_bucket_hidden
        mtp = self

        def hook(bi, bucket, chunk_start):
            x = orig(bi, bucket, chunk_start)
            h = model.norm(ttnn.clone(x, memory_config=_DRAM), mode=Mode.PREFILL)  # replicated [1,1,bucket,H]
            mtp._prompt_rows(h, bucket, chunk_start)
            ttnn.deallocate(h)
            return x

        model._run_bucket_hidden = hook
        self._orig_hook = orig

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
        self.model._run_bucket_hidden = self._orig_hook

    def _log_stats(self, why):
        st = self._stats
        if st and st["passes"]:
            ms = st["wall"] / st["passes"] * 1e3
            logger.info(
                f"MTP_STATS ({why}): {st['passes']} passes, {st['tokens']} tokens = {st['tokens'] / st['passes']:.2f} per pass, "
                f"acceptance hist {st['hist']}, {ms:.1f} ms per pass -> {st['tokens'] / st['wall']:.1f} tok/s (device steps only)"
            )

    def end_prefill(self, tok0: int, P: int):
        self.model._run_bucket_hidden = self._orig_hook
        assert self._hid_ready, "the prefill hook did not see the last prompt position"
        self._log_stats("previous request")
        self._stats = {"passes": 0, "tokens": 0, "hist": [0] * (self.k + 1), "wall": 0.0}
        self._pos = P
        self._tok = int(tok0)

    # ------------------------------------------------------------------ decode side
    def step(self, tok: int, pos: int, page_table_row=None, host_authoritative=False) -> list[int]:
        """One MTP step: returns the accepted drafts followed by the verified next token (1..k+1 ids). The model owns the
        continuation (its last emitted token / position and the head's hidden for it); the plugin's `tok` / `pos` are
        only checked when it flags the host inputs authoritative (`reload_inputs` / `reset_batch`: the first decode after
        a prefill) - on the steady async path its host token can lag the device by a step."""
        import time

        t0 = time.perf_counter()
        r = self.runner
        if page_table_row is not None:
            self.set_page_table(page_table_row)  # vLLM may have grown the request's block list
        if host_authoritative and self._pos is not None and (pos != self._pos or tok != self._tok):
            logger.warning(
                f"MTP step input (tok {tok}, pos {pos}) differs from the model's (tok {self._tok}, pos {self._pos})"
            )
        tok, pos = self._tok if self._tok is not None else tok, self._pos if self._pos is not None else pos
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
        if self._stats is not None:
            st = self._stats
            st["passes"] += 1
            st["tokens"] += len(out)
            st["hist"][a] += 1
            st["wall"] += time.perf_counter() - t0
            if st["passes"] % 32 == 0:
                self._log_stats("running")
        return out
