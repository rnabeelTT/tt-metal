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
from models.autoports.qwen_qwen3_5_122b_a10b.tt.mtp_rows import rebuild_conv_carry, rows_pos_masks_host
from models.autoports.qwen_qwen3_5_122b_a10b.tt.mtp_traced import MtpTraced, _host_i32, _host_u32
from models.tt_transformers.tt.common import Mode

_DRAM = ttnn.DRAM_MEMORY_CONFIG
ROWS = 32  # head prompt rows per call (one tile of decode rows)


class MtpServing:
    def __init__(self, gen, k: int):
        self.gen, self.k = gen, int(k)
        self.B = int(gen.max_batch_size)
        # B > 1 slots: ADAPTIVE (default) = the single-user step whenever exactly one request is live, the plain batched
        # decode otherwise, state handed over at every switch (doc/perf_pass/adaptive_mtp_plan.md); QWEN35_MTP_ADAPTIVE=0
        # = the batched multi-user step over every live slot (measured slower than plain decode at 2-4 users)
        self.adaptive = self.B > 1 and os.environ.get("QWEN35_MTP_ADAPTIVE", "1") == "1"
        self.multi = self.B > 1 and not self.adaptive
        self.per_slot = self.B > 1  # prefill bookkeeping per slot (several requests per prefill call)
        self._phase_slot = None  # adaptive: the slot whose GDN state lives in the MTP scratch (None = plain decode)
        self._scratch_prev = None
        self._suspended = False  # adaptive: phase kept (scratch content) but unbound across a prefill
        self._force_reload = False  # adaptive: the first plain step after a phase must be host-authoritative
        self.hid_prev = None  # adaptive: [1,1,B,H] final hidden of every slot's last decoded token (device, in-trace)
        self._place = {}  # adaptive: width -> one-hot [1,1,B,w] (rows [0,w) into the B-row buffer)
        self._keep = {}  # adaptive: width -> [1,1,B,1] mask (1 for rows >= w)
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
        # slots admitted by the batched short-prompt prefill (no head rows over their prompt: the head's KV of the prompt
        # positions is missing, so drafts would be poor) - they stay on the plain decode until a new request takes the
        # slot (begin_prefill clears it)
        self.no_head_slots = set()
        # QWEN35_MTP_TRACED_PREFILL=1 (default): the head's prompt rows run INSIDE the bucket program (one more layer per
        # segment, captured with the bucket traces: the eager-bucket path cost ~60 ms of TTFT at 128 tokens); 0 = the
        # eager-bucket path with the python hook and 32-row head groups
        self.in_trace = os.environ.get("QWEN35_MTP_TRACED_PREFILL", "1") == "1"
        self.hid_out = None  # [1,1,1,H] replicated: the hidden that produced the request's first token (in-trace mode)
        logger.info(
            f"MTP serving: k={self.k}, slots={self.B} "
            f"({'adaptive single-user' if self.adaptive else 'multi' if self.multi else 'single'}), "
            f"head layer {self.head.layer_idx}"
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
        elif self.adaptive:
            # the single-user runner's state snapshots / scan sinks are shaped by the bound GDN state: build it on the
            # B = 1 MTP scratch (bound again for warm-up, capture and every phase)
            prev = self.model._bind_gdn_mtp_scratch()
            try:
                self.runner = MtpTraced(self.gen, self.head, self.k)
            finally:
                self.model._unbind_gdn_mtp_scratch(prev)
        else:
            self.runner = MtpTraced(self.gen, self.head, self.k)
        m = self.model
        if self.adaptive:
            z = lambda shape: ttnn.from_torch(
                torch.zeros(*shape, dtype=torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh,
                memory_config=_DRAM,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
            )
            self.hid_prev = z((1, 1, self.B, self.H))
            self._e = z((1, 1, self.B, 1))  # one-hot column of the slot being written
            self._ne = z((1, 1, self.B, 1))  # its complement
            self._pick = z((1, 1, 1, self.B))  # one-hot row of the slot being read
            self._perm = z((1, 1, self.B, self.B))  # slot permutation (batch condense)
            self._rowbuf = z((1, 1, 1, self.H))  # host hidden rows land here (no request-time allocation)
            if (
                os.environ.get("QWEN35_MTP_HOOKS", "1") == "1"
            ):  # diagnostic knob: 0 = no head bookkeeping in plain steps
                m._mtp_decode_hook = self._decode_hook  # inside every plain decode step (and its traces)
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

    def _bound(self):
        """Context: the MTP scratch bound onto the GDN layers (adaptive), a no-op otherwise."""
        import contextlib

        if not self.adaptive:
            return contextlib.nullcontext()

        @contextlib.contextmanager
        def _cm():
            prev = self.model._bind_gdn_mtp_scratch()
            try:
                yield
            finally:
                self.model._unbind_gdn_mtp_scratch(prev)

        return _cm()

    def warm(self):
        """Compile every MTP program (the step programs + the head prompt rows for every bucket)."""
        self._ensure_runner()
        with self._bound():
            self.runner.warm()
        if self.adaptive:
            self.warm_adaptive()
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
            if not isinstance(t, torch.Tensor):
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
        prev = self.model._bind_gdn_prefill_scratch() if self.per_slot else None
        try:
            for b in self.model._PREFILL_MASK_BUCKETS:
                for n in (b, max(2, b - 1)):
                    ids = [1000 + (i % 977) for i in range(n)]
                    logger.debug(f"MTP warm prefill path: bucket {b} len {n}")
                    self.begin_prefill(ids, pt[0], slot=0)
                    try:
                        if self.per_slot:
                            self.model._reset_gdn_state_for_new_sequence()
                        lg, sh = self.gen._prefill_single(
                            torch.tensor(ids, dtype=torch.int32).reshape(1, -1), pt, want_sharded=True
                        )
                    finally:
                        self.abort_prefill()
        finally:
            if prev is not None:
                self.model._unbind_gdn_prefill_scratch(prev)
        if not self.per_slot:
            self.model._reset_gdn_state_for_new_sequence()
        ttnn.synchronize_device(self.mesh)

    def capture(self):
        self._ensure_runner()
        with self._bound():
            self.runner.capture()
        logger.info("MTP traces captured (pass, commit, draft, head rows)")

    # ------------------------------------------------------------------ adaptive: hooks, enter / leave
    def _decode_hook(self, stage, *a):
        """Inside the plain decode step (captured in its traces). "pre": the head's K/V rows of (this step's tokens,
        hid_prev) at the step's positions / page tables; "post": hid_prev rows [0, w) <- this step's final hidden."""
        if stage == "pre":
            tok_view, cur_pos, page_table, cos, sin = a
            w = int(cur_pos.shape[-1])
            hid = self.hid_prev if w == self.B else ttnn.slice(self.hid_prev, (0, 0, 0, 0), (1, 1, w, self.H))
            self.head.kv_rows(tok_view, hid, cur_pos, cos, sin, page_table)
            if hid is not self.hid_prev:
                ttnn.deallocate(hid)
            return
        (x,) = a  # [1,1,w,H] replicated final-normed hidden (the LM head's input; not consumed here)
        w = int(x.shape[2])
        # a DRAM copy (ttnn.to_memory_config can hand back a new handle on the SAME buffer: freeing it freed x)
        xd = ttnn.clone(x, memory_config=_DRAM)
        if w == self.B:
            ttnn.copy(xd, self.hid_prev)
        else:
            if w not in self._place:
                pl = torch.zeros(1, 1, self.B, w, dtype=torch.bfloat16)
                kp = torch.ones(1, 1, self.B, 1, dtype=torch.bfloat16)
                for i in range(w):
                    pl[0, 0, i, i] = 1.0
                    kp[0, 0, i, 0] = 0.0
                dev = lambda t: ttnn.from_torch(
                    t,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.mesh,
                    memory_config=_DRAM,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
                )
                self._place[w], self._keep[w] = dev(pl), dev(kp)
            ph = ttnn.matmul(self._place[w], xd, memory_config=_DRAM)  # [1,1,B,H], rows >= w zero
            kept = ttnn.multiply(self.hid_prev, self._keep[w], memory_config=_DRAM)
            new = ttnn.add(kept, ph, memory_config=_DRAM)
            ttnn.copy(new, self.hid_prev)
            for t in (ph, kept, new):
                ttnn.deallocate(t)
        ttnn.deallocate(xd)

    def _put_row(self, slot, row):
        """hid_prev[slot] <- row (device [1,1,1,H] or host torch [H])."""
        if isinstance(row, torch.Tensor):
            ttnn.copy_host_to_device_tensor(
                ttnn.from_torch(
                    row.reshape(1, 1, 1, self.H).to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
                ),
                self._rowbuf,
            )
            row = self._rowbuf
        e = torch.zeros(1, 1, self.B, 1, dtype=torch.bfloat16)
        e[0, 0, slot, 0] = 1.0
        host = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        ttnn.copy_host_to_device_tensor(host(e), self._e)
        ttnn.copy_host_to_device_tensor(host(1.0 - e), self._ne)
        placed = ttnn.matmul(self._e, row, memory_config=_DRAM)  # [1,1,B,H]: the row at slot, zeros elsewhere
        kept = ttnn.multiply(self.hid_prev, self._ne, memory_config=_DRAM)
        new = ttnn.add(kept, placed, memory_config=_DRAM)
        ttnn.copy(new, self.hid_prev)
        for t in (placed, kept, new):
            ttnn.deallocate(t)

    def _pick_row(self, slot):
        """runner.d_hid <- hid_prev[slot]."""
        sel = torch.zeros(1, 1, 1, self.B, dtype=torch.bfloat16)
        sel[0, 0, 0, slot] = 1.0
        ttnn.copy_host_to_device_tensor(ttnn.from_torch(sel, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT), self._pick)
        picked = ttnn.matmul(self._pick, self.hid_prev, memory_config=_DRAM)  # [1,1,1,H]
        ttnn.copy(picked, self.runner.d_hid)
        ttnn.deallocate(picked)

    def remap(self, remap):
        """Batch condense: hid_prev row i <- old row remap[i] (the GDN slots are remapped by the caller)."""
        idx = [int(remap[i]) for i in range(self.B)]
        if all(idx[i] == i for i in range(self.B)):
            return
        assert self._phase_slot is None, "leave the MTP phase before a slot remap"
        self.no_head_slots = {i for i in range(self.B) if idx[i] in self.no_head_slots}
        P = torch.zeros(1, 1, self.B, self.B, dtype=torch.bfloat16)
        for i, j in enumerate(idx):
            P[0, 0, i, j] = 1.0
        ttnn.copy_host_to_device_tensor(ttnn.from_torch(P, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT), self._perm)
        new = ttnn.matmul(self._perm, self.hid_prev, memory_config=_DRAM)
        ttnn.copy(new, self.hid_prev)
        ttnn.deallocate(new)

    @property
    def in_phase(self):
        return self._phase_slot is not None

    def enter(self, slot, tok, pos, page_table_row):
        """Plain decode -> MTP phase for `slot`: its GDN state into the MTP scratch (bound), the head hidden of its last
        token into the runner, the runner's token / position from the caller (authoritative)."""
        assert self.adaptive and self._phase_slot is None
        t0 = time.perf_counter()
        m = self.model
        self._scratch_prev = m._bind_gdn_mtp_scratch()
        for (dn, rec, conv, _carry, _z), (_dn, _Bb, rec_b, conv_b, *_rest) in zip(
            m._gdn_mtp_scratch, self._scratch_prev
        ):
            src = dn._slice_along(rec_b, 0, slot, slot + 1)
            ttnn.copy(src, rec)
            ttnn.deallocate(src)
            for j in range(dn.K):
                src = dn._slice_along(conv_b[j], 1, slot, slot + 1)
                ttnn.copy(src, conv[j])
                ttnn.deallocate(src)
            rebuild_conv_carry(dn)
        self._phase_slot = int(slot)
        self.runner.rope_delta = self.model.slot_rope_delta(slot)  # image prompts decode at pos + M-RoPE delta
        self.set_page_table(page_table_row)
        self._pick_row(slot)
        self._tok, self._pos = int(tok), int(pos)
        self._sampler = None
        self._log_stats("previous phase")
        self._stats = {"passes": 0, "tokens": 0, "hist": [0] * (self.k + 1), "wall": 0.0}
        self._phase_t0 = time.perf_counter()
        logger.info(f"MTP phase enter slot {slot} at pos {pos}: {(time.perf_counter() - t0) * 1e3:.1f} ms")

    def suspend(self):
        """Keep the phase but unbind the scratch (a prefill is about to run; the next decode decides: resume / drop /
        leave with the write-back)."""
        if self._phase_slot is None or self._suspended:
            return
        self.model._unbind_gdn_mtp_scratch(self._scratch_prev)
        self._scratch_prev = None
        self._suspended = True

    def resume(self):
        assert self._phase_slot is not None and self._suspended
        self._scratch_prev = self.model._bind_gdn_mtp_scratch()
        self._suspended = False

    def drop(self):
        """End the phase without writing the state back (the phase's request finished)."""
        if self._phase_slot is None:
            return
        if not self._suspended:
            self.model._unbind_gdn_mtp_scratch(self._scratch_prev)
            self._scratch_prev = None
        self._suspended = False
        self._phase_slot = None
        self._tok = self._pos = None
        self._log_stats("phase end (request finished)")

    def leave(self):
        """MTP phase -> plain decode: the runner's hidden back into hid_prev[slot], the scratch state back into the
        slot's row of the batched GDN buffers; the next plain step must take the host token / position for the slot."""
        slot = self._phase_slot
        if slot is None:
            return None
        t0 = time.perf_counter()
        m = self.model
        self._put_row(slot, self.runner.d_hid)
        if not self._suspended:
            m._unbind_gdn_mtp_scratch(self._scratch_prev)
            self._scratch_prev = None
        self._suspended = False
        for dn, rec, conv, _carry, _z in m._gdn_mtp_scratch:
            dn.write_slot(slot, ttnn.clone(rec), [ttnn.clone(c) for c in conv])
        self._phase_slot = None
        self._force_reload = True
        last = (self._tok, self._pos)
        self._tok = self._pos = None
        self._log_stats("phase end")
        logger.info(f"MTP phase leave slot {slot} at pos {last[1]}: {(time.perf_counter() - t0) * 1e3:.1f} ms")
        return last

    def consume_force_reload(self):
        f = self._force_reload
        self._force_reload = False
        return f

    def warm_adaptive(self):
        """Compile the enter / leave programs for every slot (slot slices, slot writes, row picks / puts) before any
        capture; the state content is copied out and back, so nothing changes."""
        pt = self.gen.page_table[:1]
        for slot in range(self.B):
            self.enter(slot, 0, 0, pt[0])
            self.leave()
        self._force_reload = False
        self._stats = None
        self.remap([(i + 1) % self.B for i in range(self.B)])
        self.remap([(i - 1) % self.B for i in range(self.B)])
        ttnn.synchronize_device(self.mesh)

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
        if self.per_slot:
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
        # the plugin hands the row every step; at the full context it is 4096 blocks wide (4 buffers x BW rows of int32
        # per write = ~1.5 ms per pass measured at 32 slots): write only when the row changed
        last = getattr(self, "_last_pt_row", None)
        if last is not None and last.shape == row.shape and bool(torch.equal(last, row)):
            return
        self._last_pt_row = row.clone()
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
            if self.per_slot and self._hid_ready and self._prompt is not None:
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
        self.no_head_slots.discard(int(slot))
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

    def note_batched(self, slot, hid_row, page_table_row):
        """A request admitted by the batched short-prompt pass (tt/prefill_batched.py): it leaves the prefill queue
        (no bucket program runs for it), its first-draft hidden is the pass's final-normed last-position row, and the
        slot is marked head-less (no MTP phase) until a new request takes it."""
        slot = int(slot)
        for i, (s, _p, _r) in enumerate(self._prefill_queue):
            if s == slot:
                self._prefill_queue.pop(i)
                break
        else:
            raise AssertionError(f"slot {slot} was not registered with begin_prefill")
        old = self._hid_rows.pop(slot, None)
        if old is not None and not isinstance(old, torch.Tensor):
            ttnn.deallocate(old)
        self._hid_rows[slot] = torch.as_tensor(hid_row).reshape(-1).clone()
        self._slot_rows[slot] = torch.as_tensor(page_table_row).to(torch.int32)
        self.no_head_slots.add(slot)
        self._hid_ready = True  # a following single request re-arms this through its own bucket stage

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
        if self.adaptive:
            # per slot: the hidden that produced the first token goes to hid_prev[slot] (the plain decode's head rows
            # and a later enter() read it there); token / position come from the plugin at the slot's first decode
            assert not self.in_phase or self._suspended, "a prefill ran inside a bound MTP phase (suspend() first)"
            if self._phase_slot is not None and self._phase_slot in [int(x) for x in slots]:
                # vLLM reused the phase's slot for a new request: the phase's request is gone (no write-back)
                self.drop()
            for slot in slots:
                slot = int(slot)
                if self.in_trace:
                    row = self._hid_rows.pop(slot, None)
                    if row is None:
                        row = ttnn.to_torch(ttnn.get_device_tensors(self.hid_out)[0]).reshape(-1)[: self.H].clone()
                    self._put_row(slot, row)
                else:
                    row = self._hid_rows.pop(slot)
                    self._put_row(slot, row)
                    if not isinstance(row, torch.Tensor):
                        ttnn.deallocate(row)
            return
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
                if not isinstance(row, torch.Tensor):
                    ttnn.deallocate(row)
            if self._stats is None:
                self._stats = {"passes": 0, "tokens": 0, "hist": [0] * (self.k + 1), "wall": 0.0}
            return
        self._log_stats("previous request")
        self._stats = {"passes": 0, "tokens": 0, "hist": [0] * (self.k + 1), "wall": 0.0}
        self._sampler = None  # the next request gets its own sampler / RNG
        if self.in_trace:
            ttnn.copy(self.hid_out, self.runner.d_hid)
        self.runner.rope_delta = self.model.slot_rope_delta(0)
        self._pos = int(P)
        self._tok = int(tok0)

    # ------------------------------------------------------------------ decode side
    @property
    def suspended(self):
        return self._suspended

    def step(self, tok: int, pos: int, page_table_row=None, host_authoritative=False, sampling=None) -> list[int]:
        """Single user. Returns the accepted drafts followed by the verified next token (1..k+1 ids). The model owns the
        continuation (its last emitted token / position and the head's hidden for it); the plugin's `tok` / `pos` are
        only checked when it flags the host inputs authoritative (`reload_inputs` / `reset_batch`: the first decode after
        a prefill) - on the steady async path its host token can lag the device by a step."""
        t0 = time.perf_counter()
        r = self.runner
        assert not self.adaptive or (self.in_phase and not self._suspended), "adaptive MTP: step() outside a phase"
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
            preds, target = r.verify(rows, pos, want_logits=True, sampler=sampler)
            a, nxt = sampler.accept(target, drafts, qs)
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
