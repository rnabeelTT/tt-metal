# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Batched short-prompt prefill: up to B = 4 prompts that all fit ONE bucket run in a single pass.

Why (perf pass, 2026-10-01): a served 128-token prefill costs ~238 ms, of which ~193 ms is the bucket program itself,
op-launch bound (~3.7 ms per layer at any length <= 512). The plugin hands a burst of waiting requests to one
``prefill_forward`` call, so grouping the final-chunk short prompts in fours and running one pass per group turns
4 x 238 ms into roughly one 512-row bucket (~200 ms) plus the slot writes.

How: embedding over the ``[B, T]`` token block (``B*T`` rows back to back, user u = rows ``u*T .. (u+1)*T-1``); the GDN
layers through the batched chunk form (``TPGatedDeltaNet.forward_prefill_batched``, one (user, head) scan row per core,
hence ``B * Nv_tp <= 32`` -> B <= 4 at TP = 4) with the trace-safe per-user lengths ``mtp_multi.RowsLen``, writing the
users' states straight into a dedicated B-row GDN scratch (bound like the B = 1 prefill scratch); the attention layers
through ``TPAttention.forward_prefill_paged_batched`` (one QKV / rope / output projection over ``B*T`` rows, per-user K/V
fill and causal SDPA against the user's own page-table row); norms and the MoE on ``B*T`` rows unchanged; the LM head on
the B selected last rows (one-hot ``[32, B*T]``, rows 0..B-1 = users, the decode sampler's input spec). Afterwards
``write_slots`` copies scratch row u into the request's decode slot (``TPGatedDeltaNet.write_slot``).

Scope: text-only prompts (every user's rope positions are 0..T-1: image prompts with M-RoPE deltas stay on the
single-user path), first chunk only (``chunk_start == 0``), no MTP head rows (batched admissions are MTP-ineligible).
Persistent per-(bucket, B) inputs + host->device staging (no allocation per request) so a later step can capture one
trace per bucket exactly like the single-user buckets.
"""
import torch

import ttnn
from models.autoports.qwen_qwen3_coder_next.tt.mtp_multi import RowsLen, rebuild_batched_conv_carry
from models.tt_transformers.tt.common import Mode

GDN_BATCH_CAP = 4  # the chunk-scan kernel maps one (user, head) row per core: B * Nv_tp <= 32 at TP = 4
MAX_BATCHED_BUCKET = 512  # longest bucket the batched pass serves (the launch-bound regime; longer prompts go single)


class BatchedPrefill:
    """Batched short-prompt prefill driver bound to one model (see the module docstring)."""

    def __init__(self, model, B=GDN_BATCH_CAP):
        self.model = model
        self.B = int(B)
        assert 2 <= self.B <= GDN_BATCH_CAP, (self.B, GDN_BATCH_CAP)
        self._scratch = (
            None  # [(dn, rec, conv_states, conv_carry, zero_conv0, zero_rec, zero_conv_carry)] per GDN layer
        )
        self._inputs = {}  # bucket -> persistent device inputs
        self.traces = {}  # bucket -> (trace id, logits_sh) once captured (a later step)

    # ---------------------------------------------------------------- GDN scratch (B rows) ------------------------
    def _ensure_scratch(self):
        """Allocate the persistent B-row GDN scratch once (like ``_ensure_gdn_prefill_scratch`` with ``dn.B = B``):
        rec_state [B, Nv, Dk, Dv], conv_states[*] [1, B, D] and the matching zero sources."""
        if self._scratch is not None:
            return
        scratch = []
        for layer in self.model.layers:
            if layer.is_full_attention:
                continue
            dn = layer.attention
            saved = (
                dn.B,
                dn.rec_state,
                dn.conv_states,
                dn.conv_carry,
                dn._zero_conv0,
                dn._zero_rec,
                dn._zero_conv_carry,
                dn._stable_state,
                getattr(dn, "_batched_conv_carry", None),
            )
            dn._batched_conv_carry = None  # reset_state would free the bound one
            dn.B = self.B
            dn.reset_state()
            scratch.append(
                (dn, dn.rec_state, dn.conv_states, dn.conv_carry, dn._zero_conv0, dn._zero_rec, dn._zero_conv_carry)
            )
            (
                dn.B,
                dn.rec_state,
                dn.conv_states,
                dn.conv_carry,
                dn._zero_conv0,
                dn._zero_rec,
                dn._zero_conv_carry,
                dn._stable_state,
                dn._batched_conv_carry,
            ) = saved
        self._scratch = scratch

    def bind(self):
        """Bind the B-row scratch onto every GDN layer; returns the previous bindings for :meth:`unbind`."""
        self._ensure_scratch()
        prev = []
        for dn, rec, conv, carry, zero0, zero_rec, zero_carry in self._scratch:
            prev.append(
                (
                    dn,
                    dn.B,
                    dn.rec_state,
                    dn.conv_states,
                    dn.conv_carry,
                    dn._zero_conv0,
                    dn._zero_rec,
                    dn._zero_conv_carry,
                    dn._stable_state,
                    getattr(dn, "_batched_conv_carry", None),
                )
            )
            dn.B = self.B
            dn.rec_state = rec
            dn.conv_states = conv
            dn.conv_carry = carry
            dn._zero_conv0 = zero0
            dn._zero_rec = zero_rec
            dn._zero_conv_carry = zero_carry
            dn._stable_state = True
            dn._batched_conv_carry = None
        return prev

    def unbind(self, prev):
        for dn, B_b, rec_b, conv_b, carry_b, zero0_b, zero_rec_b, zero_carry_b, stable_b, bcc_b in prev:
            # the pass leaves its last conv left-context on the layer: free it after an eager pass (a traced pass leaves
            # a trace-owned handle, or None after a replay: nothing to free)
            bcc = getattr(dn, "_batched_conv_carry", None)
            if bcc is not None and bcc is not bcc_b:
                try:
                    if bcc.is_allocated():
                        ttnn.deallocate(bcc)
                except RuntimeError:  # trace-owned buffer
                    pass
            dn.B = B_b
            dn.rec_state = rec_b
            dn.conv_states = conv_b
            dn.conv_carry = carry_b
            dn._zero_conv0 = zero0_b
            dn._zero_rec = zero_rec_b
            dn._zero_conv_carry = zero_carry_b
            dn._stable_state = stable_b
            dn._batched_conv_carry = bcc_b

    def reset_scratch(self):
        """Zero the bound scratch in place (sequence start for all B rows)."""
        for layer in self.model.layers:
            if not layer.is_full_attention:
                layer.attention.reset_state_inplace()

    def write_slots(self, slots):
        """Scratch row u -> decode slot ``slots[u]`` of the batched GDN buffers (call AFTER :meth:`unbind`)."""
        assert len(slots) <= self.B, (slots, self.B)
        for dn, rec, conv, *_ in self._scratch:
            Nv, Dk, Dv = rec.shape[1], rec.shape[2], rec.shape[3]
            D = conv[0].shape[-1]
            for u, slot in enumerate(slots):
                rec_u = ttnn.slice(rec, (u, 0, 0, 0), (u + 1, Nv, Dk, Dv))
                conv_u = [ttnn.slice(conv[m], (0, u, 0), (1, u + 1, D)) for m in range(dn.K)]
                dn.write_slot(int(slot), rec_u, conv_u)  # consumes the slices

    # ---------------------------------------------------------------- persistent inputs --------------------------
    def inputs(self, bucket, max_blocks=None):
        bi = self._inputs.get(bucket)
        if bi is not None:
            return bi
        model, B = self.model, self.B
        dev = model.device
        rep = ttnn.ReplicateTensorToMesh(dev)
        bs = model._prefill_block_size()
        rd = model.args.rope_head_dim
        conv_k = next(l.attention.K for l in model.layers if not l.is_full_attention)
        if max_blocks is None:
            max_blocks = getattr(model, "_bucket_max_blocks", None)
        assert max_blocks is not None, "first inputs() call must give the page-table row width"
        rm = lambda t, dt: ttnn.from_torch(t, dtype=dt, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, mesh_mapper=rep)
        tl = lambda t, dt: ttnn.from_torch(
            t, dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=rep
        )
        bi = {
            "tok": rm(torch.zeros(B, bucket, dtype=torch.int32), ttnn.uint32),
            "cos": tl(torch.zeros(1, 1, B * bucket, rd, dtype=torch.bfloat16), ttnn.bfloat16),
            "sin": tl(torch.zeros(1, 1, B * bucket, rd, dtype=torch.bfloat16), ttnn.bfloat16),
            "full_pt": [rm(torch.zeros(1, int(max_blocks), dtype=torch.int32), ttnn.int32) for _ in range(B)],
            "chunk_pt": [rm(torch.zeros(1, bucket // bs, dtype=torch.int32), ttnn.int32) for _ in range(B)],
            "len": RowsLen(dev, B, bucket, conv_k),
            "logit_sel": tl(torch.zeros(1, 1, 32, B * bucket, dtype=torch.bfloat16), ttnn.bfloat16),
            # final-normed hidden of every user's last prompt position (rows 0..B-1): the MTP head's "hidden that
            # produced the first token" (MtpServing.note_batched), written inside the pass
            "hid": tl(torch.zeros(1, 1, 32, model.args.dim, dtype=torch.bfloat16), ttnn.bfloat16),
            "staged": None,
        }
        self._inputs[bucket] = bi
        return bi

    def stage(self, bi, tokens, lens, page_table, bucket):
        """Host -> device copies of one group's inputs: ``tokens`` [n, >= max(lens)] int, ``lens`` the n real lengths,
        ``page_table`` [n, W] (row u = request u's blocks). Rows n..B-1 of a short group are idle: one pad token, page
        tables pointing at the reserved KV scratch block only; their logits / scratch state are never read."""
        model, B = self.model, self.B
        n = len(lens)
        assert 1 <= n <= B, (n, B)
        assert all(1 <= int(l) <= bucket for l in lens), (lens, bucket)
        dev = model.device
        rep = ttnn.ReplicateTensorToMesh(dev)
        bs = model._prefill_block_size()
        scratch = int(model._kv_scratch_block)
        max_blocks = int(bi["full_pt"][0].shape[-1])
        nblk = bucket // bs
        tok = torch.zeros(B, bucket, dtype=torch.int32)
        pt_full = torch.full((B, max_blocks), scratch, dtype=torch.int32)
        cpt = torch.full((B, nblk), scratch, dtype=torch.int32)
        pt = page_table.to(torch.int32)
        assert pt.shape[1] <= max_blocks, (pt.shape, max_blocks)
        lens_b = []
        for u in range(B):
            if u < n:
                L = int(lens[u])
                tok[u, :L] = tokens[u, :L].to(torch.int32)
                pt_full[u, : pt.shape[1]] = pt[u]
                n_real = min(-(-L // bs), nblk, pt.shape[1])
                cpt[u, :n_real] = pt[u, :n_real]
                lens_b.append(L)
            else:
                lens_b.append(1)  # idle row: one pad token, K/V into the scratch block only
        host_rm = lambda t, dt: ttnn.from_torch(t, dtype=dt, layout=ttnn.ROW_MAJOR_LAYOUT, device=None, mesh_mapper=rep)
        host_tl = lambda t, dt: ttnn.from_torch(t, dtype=dt, layout=ttnn.TILE_LAYOUT, device=None, mesh_mapper=rep)
        ttnn.copy_host_to_device_tensor(host_rm(tok, ttnn.uint32), bi["tok"])
        for u in range(B):
            ttnn.copy_host_to_device_tensor(host_rm(pt_full[u : u + 1], ttnn.int32), bi["full_pt"][u])
            ttnn.copy_host_to_device_tensor(host_rm(cpt[u : u + 1], ttnn.int32), bi["chunk_pt"][u])
        # text-only: every user at positions 0..bucket-1 (clear any staged M-RoPE table first)
        model._build_request_rope(tok[:1, : lens_b[0]], None)
        cos_t, sin_t = model._rope_tp_cos_sin_torch(0, bucket)  # [1,1,bucket,rd]
        ttnn.copy_host_to_device_tensor(host_tl(cos_t.repeat(1, 1, B, 1), ttnn.bfloat16), bi["cos"])
        ttnn.copy_host_to_device_tensor(host_tl(sin_t.repeat(1, 1, B, 1), ttnn.bfloat16), bi["sin"])
        bi["len"].stage(lens_b)
        sel = torch.zeros(1, 1, 32, B * bucket, dtype=torch.float32)
        for u in range(B):
            sel[0, 0, u, u * bucket + lens_b[u] - 1] = 1.0
        ttnn.copy_host_to_device_tensor(host_tl(sel, ttnn.bfloat16), bi["logit_sel"])
        bi["staged"] = (tuple(lens_b),)

    # ---------------------------------------------------------------- the pass -----------------------------------
    def _attn_layer(self, layer, x, bi, B):
        args = layer.args
        cfg = {
            **args.get_norm_config("attn", Mode.PREFILL),
            "distributed_output_mem_config": ttnn.L1_MEMORY_CONFIG,
        }
        n = layer._decode_norm(
            layer.attention_norm,
            x,
            Mode.PREFILL,
            cfg,
            keep_sharded=bool(getattr(args, "attn_norm_out_sharded", False)),
        )
        a = layer.attention.forward_prefill_paged_batched(
            n, bi["cos"], bi["sin"], bi["full_pt"], chunk_page_tables=bi["chunk_pt"], user_ids=[0] * B
        )
        if n.is_allocated():
            ttnn.deallocate(n)
        return self._ffn(layer, x, a)

    def _gdn_layer(self, layer, x, bi, B, T):
        args = layer.args
        dn = layer.attention
        cfg = {
            **args.get_norm_config("attn", Mode.PREFILL),
            "distributed_output_mem_config": ttnn.L1_MEMORY_CONFIG,
        }
        n = layer._decode_norm(layer.attention_norm, x, Mode.PREFILL, cfg)
        xb = ttnn.reshape(n, (B, T, n.shape[-1]))  # the batched chunk form takes [B, T, dim(/tp)] (flattens it again)
        rebuild_batched_conv_carry(dn)  # [B, K-1, D] from the (zeroed) decode taps: device-only, trace-safe
        out = dn.forward_prefill_batched(xb, chunk_size=int(args.gdn_chunk_size), valid_lens=bi["len"], carry=True)
        if n.is_allocated():
            ttnn.deallocate(n)
        a = ttnn.reshape(out, (1, 1, B * T, out.shape[-1]))  # [1,B,T,dim/TP] -> rows
        return self._ffn(layer, x, a)

    @staticmethod
    def _ffn(layer, x, attn_out):
        args = layer.args
        h = ttnn.add(x, attn_out)
        ttnn.deallocate(attn_out)
        ff_in = layer._decode_norm(layer.ffn_norm, h, Mode.PREFILL, args.get_norm_config("ff", Mode.PREFILL))
        ff_out = layer.feed_forward.forward(ff_in, mode="prefill")
        ttnn.deallocate(ff_in)
        y = ttnn.add(h, ff_out)
        ttnn.deallocate(h)
        ttnn.deallocate(ff_out)
        return y

    def run(self, bi, bucket):
        """The fixed batched program over the staged inputs (scratch bound + zeroed by the caller). Returns
        ``(logits [1, 1, 32, vocab] replicated, logits_sharded [1, 1, 32, vocab/TP])``: row u = user u's next-token
        logits (rows >= B zero)."""
        model, B, T = self.model, self.B, bucket
        x = model.embd(bi["tok"])  # [B, T, dim/TP]
        x = ttnn.reshape(x, (1, 1, B * T, x.shape[-1]))
        x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
        for layer in model.layers:
            if layer.is_full_attention:
                x_new = self._attn_layer(layer, x, bi, B)
            else:
                x_new = self._gdn_layer(layer, x, bi, B, T)
            ttnn.deallocate(x)
            x = x_new
        x_last = ttnn.matmul(bi["logit_sel"], x)  # [1,1,32,dim/TP]: rows 0..B-1 = the users' last hidden rows
        ttnn.deallocate(x)
        x_last = ttnn.to_memory_config(x_last, ttnn.DRAM_MEMORY_CONFIG)
        x_last = model.norm(x_last, mode=Mode.PREFILL)  # replicated [1,1,32,dim]
        ttnn.copy(x_last, bi["hid"])
        logits_sh = ttnn.linear(x_last, model.lm_head_weight, compute_kernel_config=model._lm_head_ckc())
        if model._lmhead_vocab_sharded:
            from models.tt_transformers.tt.ccl import tt_all_gather

            logits = tt_all_gather(
                ttnn.clone(logits_sh, memory_config=ttnn.DRAM_MEMORY_CONFIG),
                model.mesh_device,
                model.tt_ccl,
                cluster_axis=None,
                dim=3,
                topology=model.args.ccl_topology(),
            )
        else:
            logits = ttnn.clone(logits_sh, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return logits, logits_sh

    # ---------------------------------------------------------------- entry point, warm-up, traces ----------------
    def prefill(self, tokens, lens, page_table, slots, bucket=None):
        """Batched prefill of n <= B text prompts (first chunk, all fitting ``bucket``): stage, run (the bucket's trace
        when captured, else eager) on the bound and zeroed scratch, write the slots. Returns ``(logits [n, 1, vocab]
        host, logits_sharded [1, 1, 32, vocab/TP] device (row u = request u, the on-device sampler's input),
        hidden [n, dim] host final-normed last-position hidden rows)``."""
        model = self.model
        n = len(lens)
        bucket = bucket or model._mask_bucket_for(max(int(l) for l in lens))
        assert bucket <= MAX_BATCHED_BUCKET, (bucket, MAX_BATCHED_BUCKET)
        pt = page_table if isinstance(page_table, torch.Tensor) else torch.as_tensor(page_table)
        bi = self.inputs(bucket, max_blocks=pt.shape[-1])
        self.stage(bi, tokens, lens, pt, bucket)
        prev = self.bind()
        try:
            self.reset_scratch()
            tr = self.traces.get(bucket)
            if tr is not None:
                ttnn.execute_trace(model.device, tr[0], cq_id=0, blocking=False)
                logits, logits_sh = tr[1], tr[2]
            else:
                logits, logits_sh = self.run(bi, bucket)
                ttnn.synchronize_device(model.device)
        finally:
            self.unbind(prev)
        self.write_slots([int(s) for s in slots])
        host = ttnn.to_torch(ttnn.get_device_tensors(logits)[0]).float()  # [1,1,32,V] (one replica)
        hid = ttnn.to_torch(ttnn.get_device_tensors(bi["hid"])[0])[0, 0, :n].clone()  # [n, dim] bf16
        if tr is None:
            ttnn.deallocate(logits)
        return host[0, 0, :n].reshape(n, 1, -1), logits_sh, hid

    @staticmethod
    def _warm_lens(bucket, B):
        return [bucket, max(1, bucket // 2), 1, bucket - 1][:B]  # full, masked, one token, masked by one

    def warm(self, page_table, buckets=None):
        """Compile every program of the batched pass and the scratch-row slot writes (eager, before any trace is
        captured: #48536 rule) for the batched buckets."""
        B = self.B
        buckets = [b for b in (buckets or self.model._PREFILL_MASK_BUCKETS) if b <= MAX_BATCHED_BUCKET]
        pt = page_table if isinstance(page_table, torch.Tensor) else torch.as_tensor(page_table)
        for b in sorted(buckets):
            lens = self._warm_lens(b, B)
            toks = torch.zeros(B, b, dtype=torch.int32)
            for u in range(B):
                toks[u, : lens[u]] = 1000 + torch.arange(lens[u]) % 977
            self.prefill(toks, lens, pt[:B], list(range(B)), bucket=b)
        ttnn.synchronize_device(self.model.device)

    def capture_traces(self, page_table, buckets=None):
        """One trace per batched bucket on the persistent inputs (after every program is warm and every persistent
        allocation exists). Returns the captured bucket list."""
        model, B = self.model, self.B
        buckets = [b for b in (buckets or model._PREFILL_MASK_BUCKETS) if b <= MAX_BATCHED_BUCKET]
        pt = page_table if isinstance(page_table, torch.Tensor) else torch.as_tensor(page_table)
        for b in sorted(buckets):
            if b in self.traces:
                continue
            bi = self.inputs(b, max_blocks=pt.shape[-1])
            lens = self._warm_lens(b, B)
            toks = torch.zeros(B, b, dtype=torch.int32)
            self.stage(bi, toks, lens, pt[:B], b)
            prev = self.bind()
            try:
                self.reset_scratch()
                ttnn.synchronize_device(model.device)
                tid = ttnn.begin_trace_capture(model.device, cq_id=0)
                logits, logits_sh = self.run(bi, b)
                ttnn.end_trace_capture(model.device, tid, cq_id=0)
                self.traces[b] = (tid, logits, logits_sh)
                self.reset_scratch()
            finally:
                self.unbind(prev)
        ttnn.synchronize_device(model.device)
        return sorted(self.traces)

    def release_traces(self):
        for tid, *_ in self.traces.values():
            ttnn.release_trace(self.model.device, tid)
        self.traces = {}
