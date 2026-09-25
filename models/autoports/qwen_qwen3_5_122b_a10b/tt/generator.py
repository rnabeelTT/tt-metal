# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Readiness generator for Qwen/Qwen3.5-122B-A10B on the (1,4) mesh (stage 6).

``build_generator(model_dir, mesh_device, **kwargs)`` -> :class:`Qwen35MoeGenerator`, a ``readiness_check.contract.Generator``:

* low level: ``prefill_forward(tokens, *, page_table, kv_cache, prompt_lens, return_all_logits=False)`` and
  ``decode_forward(tokens, start_pos, *, page_table, kv_cache, enable_trace=True, on_device_sampling=True, ...)``
  with explicit cache / page-table / position / batch state (the generator owns a paged KV pool + page table for
  ``max_batch_size`` users when the caller does not pass one);
* high level: ``prefill_logits(prompt)`` (logits at every position) and ``generate(prompt, max_new_tokens, *,
  next_input=None, enable_trace=True)`` (greedy; teacher forcing through ``next_input``).

Decode is the stock ``models.tt_transformers.tt.generator.Generator`` traced path around ``Qwen35MoeModel``: model trace up
to vocab-sharded logits + the common ``models/common/sampling`` ``SamplingGenerator`` trace (top-k candidates per shard,
all-gather, semantically greedy params), ``tt_out_tok`` bound to the persistent decode token input, positions advanced on
device (``plus_one`` in the model trace), page table copied only when it changes. Host sampling is the explicit
compatibility mode ``on_device_sampling=False`` (logits ``[B, vocab]`` read back).
"""

from __future__ import annotations

import json
import math
import os
import time

import torch
from loguru import logger
from readiness_check.contract import Generator as ReadinessGenerator

import ttnn
from models.autoports.qwen_qwen3_5_122b_a10b.tt.model import create_model
from models.common.sampling.sampling_params import SamplingParams
from models.tt_transformers.tt.generator import Generator as TTGenerator

BLOCK_SIZE = 64
GREEDY = SamplingParams(temperature=0.0, top_k=1, top_p=1.0)


class PendingTokens:
    """Deferred readback handle of one decode step (see ``Qwen35MoeGenerator.decode_step_async``)."""

    def __init__(self, gen, host, events, B):
        self._gen, self._host, self._events, self._B = gen, host, events, B
        self._result = None

    def result(self):
        if self._result is None:
            for ev in self._events:
                ttnn.event_synchronize(ev)
            toks = self._gen._tt.process_decode_output_host(self._host, is_tokens=True)
            self._result = torch.as_tensor(toks).reshape(-1)[: self._B].to(torch.int64)
        return self._result


class Qwen35MoeGenerator(ReadinessGenerator):
    def __init__(
        self,
        mesh_device,
        *,
        max_batch_size=1,
        max_seq_len=None,
        layer_indices=None,
        block_size=BLOCK_SIZE,
        warm_decode_trace=True,
        warm_host_trace=False,
        warm_prefill=True,
        warm_prefill_trace=True,
    ):
        self.mesh_device = mesh_device
        self.args, self.model = create_model(
            mesh_device, max_batch_size=max_batch_size, max_seq_len=max_seq_len, layer_indices=layer_indices
        )
        self.tokenizer = self.args.tokenizer
        self.max_batch_size = int(max_batch_size)
        self.max_seq_len = int(self.args.max_seq_len)
        self.block_size = int(block_size)
        self._tt = TTGenerator([self.model], [self.args], mesh_device)
        # Split-sampling token feedback: the sampler writes the sampled token straight into the persistent decode token
        # input. The force-argmax path (ttnn.argmax) accepts the [B,1] token input as its output tensor; the top-k
        # candidate path (ttnn.sampling) requires a rank-4 [1,1,1,B] output -> hand it a zero-copy view of the same buffer.
        self._tt._decode_token_feedback_buffer = self._token_feedback_buffer
        # paged KV pool: one local KV head per device (kv_replication), contiguous blocks per user
        self.blocks_per_seq = math.ceil(self.max_seq_len / self.block_size)
        # + 1 reserved scratch block: the persistent-input masked-bucket prefill fills the KV rows of the whole bucket,
        # and the pad rows past the real tokens are routed to this block (never read: decode overwrites a position
        # before it attends to it, and the causal prefill mask hides pad keys from real queries)
        num_blocks = math.ceil((self.blocks_per_seq * self.max_batch_size + 1) / 32) * 32
        self.kv_scratch_block = self.blocks_per_seq * self.max_batch_size
        self.model._kv_scratch_block = self.kv_scratch_block
        self.kv_cache_shape = (num_blocks, int(self.args.n_local_kv_heads), self.block_size, int(self.args.head_dim))
        self.kv_dtype = self.args.kv_dtype
        self.kv_cache = self.model.allocate_kv_caches(
            self.kv_cache_shape, self.kv_dtype, batch_size=self.max_batch_size
        )
        self.page_table = torch.arange(self.blocks_per_seq * self.max_batch_size, dtype=torch.int32).reshape(
            self.max_batch_size, self.blocks_per_seq
        )
        self.positions = torch.zeros(self.max_batch_size, dtype=torch.int64)
        logger.info(
            f"Qwen35MoeGenerator: B={self.max_batch_size} max_seq_len={self.max_seq_len} kv {self.kv_cache_shape} {self.kv_dtype} "
            f"({len(self.kv_cache)} attention layers), layers={len(self.model.layers)}"
        )
        # ORDER MATTERS (qwen36 #48536 rule): every prefill program and every persistent prefill allocation (masked-bucket
        # programs, expert-sorted dispatch constants, GDN conv carry, fused out-proj buffers) must exist BEFORE the decode
        # trace is captured. A prefill that first runs after the capture allocates into the trace's freed scratch range;
        # the next decode replays overwrite those buffers and the following prefill hangs on corrupted indices
        # (stage-6 work_log 17:55).
        self._host_trace_warm = False
        if os.environ.get("QWEN35_WARM_LOG") == "1":
            self._memory_report()  # footprint before any prefill program / trace exists
        if warm_prefill:
            self._warm_prefill()
        if warm_decode_trace:
            self._warm_decode_trace(warm_host_trace)
        # snapshot BEFORE the bucket captures: a program first compiled between the decode capture and the bucket
        # captures would otherwise hide inside the guard baseline (review round 3)
        self._program_cache_before_bucket_traces = self.program_cache_entries()
        if warm_prefill_trace:
            # LAST: every persistent allocation exists; a bucket trace's intermediates then only alias freed scratch
            self._warm_prefill_traces()
        self._program_cache_after_warmup = self.program_cache_entries()
        if self._program_cache_after_warmup != self._program_cache_before_bucket_traces:
            logger.warning(
                f"{self._program_cache_after_warmup - self._program_cache_before_bucket_traces} program(s) compiled during "
                "the bucket-trace captures (compile-after-capture risk: warm them before the decode capture)"
            )
        logger.info(
            f"program cache entries after warm-up: {self._program_cache_after_warmup} "
            f"(before the bucket captures: {self._program_cache_before_bucket_traces})"
        )
        # QWEN35_PROGRAM_GUARD=1: assert after every prefill / decode call that no program was compiled since warm-up
        self._program_guard = os.environ.get("QWEN35_PROGRAM_GUARD", "0") == "1"

    # ---- setup -------------------------------------------------------------------------------------------------------
    def _token_feedback_buffer(self, model, device_inputs):
        # the persistent token buffer has the sampler's rank-4 [1,1,1,S] shape (model.prepare_decode_inputs_host), so both
        # greedy strategies (ttnn.argmax and ttnn.sampling) write the sampled tokens straight into it
        return device_inputs[0]

    def _warm_prefill(self):
        """Compile the masked fixed-bucket prefill programs (128..2048, masked and full) and allocate their persistent
        buffers, then zero the state the dummy forwards touched."""
        t0 = time.time()
        if os.environ.get("QWEN35_WARM_LOG") == "1":
            # diagnostic: log every (bucket, actual_len, layer) with a device sync per layer, so a device hang inside the
            # warm-up names its layer and bucket (48-layer bring-up, 2026-09-24)
            state = {"bucket": None}
            orig_bucket = self.model.prefill_masked_bucket

            def logged_bucket(tokens, page_table, actual_len=None, bucket=None, **kw):
                state["bucket"] = (bucket, actual_len)
                logger.info(f"WARM bucket={bucket} actual_len={actual_len} start")
                out = orig_bucket(tokens, page_table, actual_len=actual_len, bucket=bucket, **kw)
                logger.info(f"WARM bucket={bucket} actual_len={actual_len} done")
                return out

            self.model.prefill_masked_bucket = logged_bucket
            for li, layer in enumerate(self.model.layers):
                orig_fwd = layer.forward

                def make(li_, orig_fwd_):
                    def fwd(*a, **k):
                        logger.info(f"WARM {state['bucket']} layer {li_} start")
                        r = orig_fwd_(*a, **k)
                        ttnn.synchronize_device(self.mesh_device)
                        logger.info(f"WARM {state['bucket']} layer {li_} done")
                        return r

                    return fwd

                layer.forward = make(li, orig_fwd)
        if self.max_batch_size > 1:
            # batched model: the masked-bucket prefill programs run B=1 on the model's persistent GDN prefill scratch
            # (allocated here, before any trace); bind it for the warm-up exactly as prefill does
            prev = self.model._bind_gdn_prefill_scratch()
            try:
                self.model.warmup_prefill_masked_buckets(self.page_table[:1])
                self._warm_all_logits()
            finally:
                self.model._unbind_gdn_prefill_scratch(prev)
            self._alloc_batched_zero_templates()  # B-sized zero templates for reset(), allocated before any trace
            self._warm_slot_writes()  # every slot's GDN state write (slice/concat programs are slot-specific)
        else:
            self.model.warmup_prefill_masked_buckets(self.page_table[:1])
            self._warm_all_logits()
        ttnn.synchronize_device(self.mesh_device)
        self._zero_gdn_state()
        logger.info(f"prefill buckets warmed in {time.time() - t0:.1f}s")

    def _warm_slot_writes(self):
        """Compile the per-slot GDN state writes of the batched decode buffers (`_write_gdn_slot`: slice / concat programs
        whose shapes depend on the slot index) before any trace is captured — the first batched prefill after the
        captures would otherwise compile them (caught by the program guard in the stage-7 generator tests)."""
        comp = ttnn.ConcatMeshToTensor(self.mesh_device, dim=0)
        prev = self.model._bind_gdn_prefill_scratch()
        try:
            self.model._reset_gdn_state_for_new_sequence()
            dn_layers = [layer.attention for layer in self.model.layers if not layer.is_full_attention]
            rec = [ttnn.to_torch(dn.rec_state, mesh_composer=comp) for dn in dn_layers]
            conv = [[ttnn.to_torch(c, mesh_composer=comp) for c in dn.conv_states] for dn in dn_layers]
        finally:
            self.model._unbind_gdn_prefill_scratch(prev)
        for u in range(self.max_batch_size):
            self.model._write_gdn_slot(u, rec, conv)
        ttnn.synchronize_device(self.mesh_device)

    def _warm_all_logits(self):
        """Compile the all-positions logits programs (readiness prefill check) before any trace is parked: a program
        compiled after a trace capture can have its binaries overwritten by a later replay (README section 4)."""
        if getattr(self.model, "_kv_scratch_block", None) is not None and hasattr(
            self.model, "warmup_all_logits_programs"
        ):
            self.model.warmup_all_logits_programs(self.page_table[:1])

    def program_cache_entries(self):
        """Number of compiled programs on the mesh (None when the binding is unavailable). The compile-after-capture
        guard: this number must not grow after warm-up in a session that replays traces (README section 4)."""
        for name in ("num_program_cache_entries",):
            fn = getattr(self.mesh_device, name, None)
            if fn is not None:
                try:
                    return int(fn())
                except Exception:  # noqa: BLE001
                    return None
        return None

    def assert_no_new_programs(self, where=""):
        """Raise if a program was compiled since warm-up (a later replay could overwrite its binaries)."""
        n = self.program_cache_entries()
        ref = getattr(self, "_program_cache_after_warmup", None)
        if n is not None and ref is not None and n != ref:
            raise RuntimeError(f"programs compiled after the trace captures ({ref} -> {n}) {where}")
        return n

    def _warm_prefill_traces(self):
        """Capture one trace per masked prefill bucket on the persistent bucket inputs (``Qwen35MoeModel.
        capture_prefill_bucket_traces``); a request's segments then replay them. Batched models capture on the B=1 GDN
        prefill scratch they prefill with."""
        t0 = time.time()
        max_bucket = int(os.environ.get("QWEN35_PREFILL_TRACE_MAX_BUCKET", "0")) or None
        buckets = [b for b in self.model._PREFILL_MASK_BUCKETS if max_bucket is None or b <= max_bucket]
        if self.max_batch_size > 1:
            prev = self.model._bind_gdn_prefill_scratch()
            try:
                self.model.capture_prefill_bucket_traces(self.page_table[:1], buckets=buckets)
            finally:
                self.model._unbind_gdn_prefill_scratch(prev)
        else:
            self.model.capture_prefill_bucket_traces(self.page_table[:1], buckets=buckets)
        ttnn.synchronize_device(self.mesh_device)
        self._zero_gdn_state()
        self._prefill_traces_warm = True
        logger.info(f"prefill bucket traces captured in {time.time() - t0:.1f}s: {sorted(self.model._bucket_traces)}")
        self._memory_report()  # footprint with every trace parked (decode + buckets): the context-contract numbers

    def _precompile_sampler_variants(self, logits0):
        """Compile every program a later sampling-trace capture can record, while no trace is live.

        ``SamplingGenerator.precompile(all_configs=True)`` runs the sampler for every (penalties, log-probs,
        force-argmax) key but with ``count_tokens=False`` and a sampler-allocated token output. The traced penalties
        variant additionally records the token-count update (``TTPenalties.update_output_tokens``: scatter_add, tilize,
        add, slice, gt) on the decode token buffer, so a penalties request whose slot is captured after the decode trace
        is parked compiled those ops inside the capture ("Cannot load new binaries during trace capture", fatal; found
        by test_generator_sampler_variants_after_replays). Run them here on a scratch token buffer with the exact spec
        of the trace's token feedback buffer (``prepare_decode_inputs_host`` -> uint32 [1,1,1,S] ROW_MAJOR on device),
        then zero the counters again."""
        sampling = self._tt.model[0].sampling
        sampling.precompile(logits0, tt_out_tok=None, all_configs=True)
        B = self.max_batch_size
        host = self._tt.model[0].prepare_decode_inputs_host(
            torch.zeros(B, 1, dtype=torch.int64), torch.zeros(B, dtype=torch.int64), page_table=None
        )
        scratch_tok = ttnn.to_device(host[0], device=self.mesh_device)
        pen = sampling.tt_penalties
        saved = sampling._penalties_active
        try:
            sampling._penalties_active = True
            # the counted penalties step exactly as capture_trace records it (penalties -> sampler -> count update)
            scratch = ttnn.clone(logits0)
            sampling._run_sampling(scratch, penalties_on=True, tt_out_tok=scratch_tok, count_tokens=True)
            ttnn.deallocate(scratch)
        finally:
            sampling._penalties_active = saved
            pen.reset_output_tokens()
        # update_output_tokens deallocates its (view of the) token tensor; free the scratch if still allocated
        try:
            ttnn.deallocate(scratch_tok)
        except Exception:
            pass

    def _warm_decode_trace(self, warm_host_trace=False):
        """Compile + capture the decode trace (model + sampler) at position 0 with dummy tokens, then reset the state the
        forwards touched. Trace capture runs the step TWICE on device (the framework's compile pass executes, the capture
        records, the first replay executes again), which advances the GDN recurrent/conv state twice: a trace captured in
        the middle of a generation therefore produces wrong logits for that step (stage-6 mode probe: the host-logits trace
        captured at the first real decode step gave a different top-1 than the eager step and the prefill truth). Both trace
        variants are captured here, on dummy inputs, and the state is reset; the host-logits variant is optional
        (``warm_host_trace``): without it the compatibility mode runs untraced."""
        B = self.max_batch_size
        tokens = torch.zeros(B, 1, dtype=torch.int64)
        pos = torch.zeros(B, dtype=torch.int64)
        # 1) eager compile pass with the sampling deferred: its logits have the spec the traced sampler will see, and NO
        #    trace is live yet -> every sampling-trace variant (penalties / log-probs / force-argmax where allowed) is
        #    compiled here (SamplingGenerator.precompile all_configs). A variant first compiled at request time would be
        #    a compile after a parked trace (README section 4); the sampler docstring asks for exactly this ordering.
        logits = self._tt.decode_forward(
            tokens,
            pos,
            page_table=self.page_table,
            kv_cache=self.kv_cache,
            enable_trace=False,
            read_from_device=False,
            defer_device_sampling=True,
            reset_batch=True,
        )
        logits0 = logits[0] if isinstance(logits, (tuple, list)) else logits
        logits0 = logits0[0] if isinstance(logits0, (tuple, list)) else logits0
        self._precompile_sampler_variants(logits0)
        ttnn.synchronize_device(self.mesh_device)
        self.reset()
        logger.info("all sampling-trace variants precompiled (before any trace)")
        # 2) capture the decode trace (model + default sampler)
        self._tt.decode_forward(
            tokens,
            pos,
            page_table=self.page_table,
            kv_cache=self.kv_cache,
            enable_trace=True,
            read_from_device=True,
            sampling_params=GREEDY,
            reset_batch=True,
        )
        ttnn.synchronize_device(self.mesh_device)
        self.reset()
        logger.info("decode trace (model + on-device sampling) captured")
        if warm_host_trace:
            self._tt.decode_forward(
                tokens,
                pos,
                page_table=self.page_table,
                kv_cache=self.kv_cache,
                enable_trace=True,
                read_from_device=True,
                sampling_params=None,
                reset_batch=True,
            )
            ttnn.synchronize_device(self.mesh_device)
            self.reset()
            self._host_trace_warm = True
            logger.info("decode trace (host-logits compatibility mode) captured")
        self.memory_report = self._memory_report()

    def _memory_report(self):
        """Measured per-device footprint after build + warm-up (weights, KV pool, GDN state, sampler, trace): the numbers
        the full-stack context contract is derived from. Logged as MEMORY_REPORT."""
        rep = {}
        try:
            for name, bt in (
                ("dram", ttnn.BufferType.DRAM),
                ("l1", ttnn.BufferType.L1),
                ("trace", ttnn.BufferType.TRACE),
            ):
                mv = ttnn.get_memory_view(self.mesh_device, bt)
                rep[name] = {
                    "total_bytes_per_bank": int(mv.total_bytes_per_bank),
                    "allocated_bytes_per_bank": int(mv.total_bytes_allocated_per_bank),
                    "banks": int(mv.num_banks),
                }
            d = rep["dram"]
            rep["dram_allocated_gib_per_device"] = round(d["allocated_bytes_per_bank"] * d["banks"] / 2**30, 3)
            rep["dram_total_gib_per_device"] = round(d["total_bytes_per_bank"] * d["banks"] / 2**30, 3)
        except Exception as e:  # API drift: report what we can
            rep["error"] = str(e)
        logger.info("MEMORY_REPORT " + json.dumps(rep))
        return rep

    # ---- readiness contract ------------------------------------------------------------------------------------------
    def reset(self):
        """New sequence(s): zero the GDN recurrent/conv state in place (trace addresses preserved) and the positions. The
        paged KV blocks need no zeroing: attention reads only [0, current_pos) through the page table."""
        self._zero_gdn_state()
        self.positions.zero_()
        self._first_decode = True

    def _zero_gdn_state(self):
        """Zero the (decode-bound) GDN state in place. The model's in-place reset copies from its B=1 zero templates (the
        prefill-scratch shape); a batched model's decode buffers are [B,...], so they are zeroed from B-sized templates
        allocated once at warm-up (before any trace)."""
        if self.max_batch_size == 1:
            self.model._reset_gdn_state_for_new_sequence()
            return
        for dn, zero_rec, zero_convs, zero_carry in self._zero_batched:
            ttnn.copy(zero_rec, dn.rec_state)
            for z, cs in zip(zero_convs, dn.conv_states):
                ttnn.copy(z, cs)
            if zero_carry is not None and getattr(dn, "conv_carry", None) is not None:
                ttnn.copy(zero_carry, dn.conv_carry)

    def _alloc_batched_zero_templates(self):
        self._zero_batched = []
        for layer in self.model.layers:
            if layer.is_full_attention:
                continue
            dn = layer.attention
            zero_rec = ttnn.zeros_like(dn.rec_state)
            zero_convs = [ttnn.zeros_like(c) for c in dn.conv_states]
            zero_carry = ttnn.zeros_like(dn.conv_carry) if getattr(dn, "conv_carry", None) is not None else None
            self._zero_batched.append((dn, zero_rec, zero_convs, zero_carry))

    def teardown(self):
        # close a mesh this generator re-opened (see build_generator); the runner's stale handle then closes as a no-op
        if getattr(self, "_owns_mesh", False):
            ttnn.close_mesh_device(self.mesh_device)
            self._owns_mesh = False

    def prefill_forward(
        self, tokens, *, page_table=None, kv_cache=None, prompt_lens=None, return_all_logits=False, **kwargs
    ):
        """``tokens`` [B, T] int; returns logits ``[B, 1, vocab]`` (or ``[B, T, vocab]`` with ``return_all_logits``, B=1)."""
        page_table = self.page_table if page_table is None else page_table
        B, T = tokens.shape
        lens = [int(prompt_lens[u]) for u in range(B)] if prompt_lens is not None else [T] * B
        if return_all_logits:
            assert B == 1, "all-position logits are a single-sequence path"
            out = self.model.prefill_all_logits(tokens[:1, : lens[0]].to(torch.int32), page_table[:1])
            self.positions[0] = lens[0]
            return out
        assert B <= self.max_batch_size, f"prefill batch {B} > max_batch_size {self.max_batch_size}"
        if self.max_batch_size == 1:
            logits = self._prefill_single(tokens[:1, : lens[0]].to(torch.int32), page_table[:1])
        else:
            logits = self._prefill_slots(
                tokens, page_table, lens
            )  # any B on a batched model: per-slot on the GDN scratch
        for u in range(B):
            self.positions[u] = lens[u]
        if getattr(self, "_program_guard", False):
            self.assert_no_new_programs("(prefill_forward)")
        # The stock Generator's first decode after a prefill (reset_batch) keeps the DEVICE token/position for every slot
        # whose device position chain looks continuous with the host view (vLLM async-scheduling rule) unless the slot
        # is marked as freshly prefilled. Every slot prefilled here is fresh: mark them, or a request whose prompt length
        # equals the previous request's prompt + generated length (or that minus one) would decode from the stale token.
        if not hasattr(self._tt, "_slots_prefilled_since_decode"):
            self._tt._slots_prefilled_since_decode = set()
        self._tt._slots_prefilled_since_decode.update(range(B))
        self._first_decode = True
        return logits

    def _prefill_slots(self, tokens, page_table, lens):
        """B > 1 prefill through the WARMED masked-bucket programs (never the exact-length paged path, which would compile
        after the decode trace was parked): bind the model's B=1 GDN prefill scratch, run each user through the same
        segmented masked-bucket path as B=1 against its own page-table row, snapshot its GDN state and write it into the
        user's decode slot (qwen36 ``prefill_paged_slots`` machinery). Returns host logits ``[B, 1, vocab]``."""
        B = tokens.shape[0]
        comp = ttnn.ConcatMeshToTensor(self.mesh_device, dim=0)
        dn_layers = [layer.attention for layer in self.model.layers if not layer.is_full_attention]
        prev = self.model._bind_gdn_prefill_scratch()
        logits, rec_snaps, conv_snaps = [], [], []
        try:
            for u in range(B):
                self.model._reset_gdn_state_for_new_sequence()  # zero the scratch for this user's from-scratch prefill
                logits.append(self._prefill_single(tokens[u : u + 1, : lens[u]].to(torch.int32), page_table[u : u + 1]))
                rec_snaps.append([ttnn.to_torch(dn.rec_state, mesh_composer=comp) for dn in dn_layers])
                conv_snaps.append([[ttnn.to_torch(c, mesh_composer=comp) for c in dn.conv_states] for dn in dn_layers])
        finally:
            self.model._unbind_gdn_prefill_scratch(prev)
        for u in range(B):
            self.model._write_gdn_slot(u, rec_snaps[u], conv_snaps[u])
        return torch.cat(logits, dim=0)

    def _host_logits(self, lg):
        """One replica of a (host or device) mesh logits tensor -> torch ``[1, 1, vocab]`` float."""
        if isinstance(lg, torch.Tensor):
            return lg.float().reshape(1, 1, -1)
        t = ttnn.to_torch(lg, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=0))
        return t.float().reshape(-1, self.args.vocab_size)[:1].reshape(1, 1, -1)

    def _prefill_single(self, tokens, page_table, segment=2048):
        """B=1 prefill of any logical length through the model's masked fixed-bucket path (pads each segment to one of
        the pre-warmed bucket lengths, masks the GDN state to the exact valid length): full 2048-token segments carry the
        GDN/KV state forward through ``chunk_start``; the tail runs at the smallest bucket >= its length. Returns host
        logits ``[1, 1, vocab]`` after the last real token."""
        T = tokens.shape[1]
        logits = None
        for cs in range(0, T, segment):
            seg = tokens[:, cs : min(cs + segment, T)]
            lg = self.model.prefill_masked_bucket(seg, page_table, actual_len=seg.shape[1], chunk_start=cs)
            logits = self._host_logits(lg)
        return logits

    def decode_forward(
        self,
        tokens,
        start_pos,
        *,
        page_table=None,
        kv_cache=None,
        enable_trace=True,
        on_device_sampling=True,
        reload_inputs=None,
        read_from_device=True,
        **kwargs,
    ):
        """One decode step. ``tokens`` [B, 1], ``start_pos`` [B]. On-device greedy sampling returns the sampled tokens
        ``[B]``; ``on_device_sampling=False`` (host compatibility mode) returns logits ``[B, vocab]``.

        ``read_from_device=False`` is the serving contract (vLLM async decode): the step is submitted with nonblocking
        trace replays and the DEVICE output handle is returned without any host sync. The caller then reads it with
        :meth:`read_decode_output` (``async_read=True`` -> ``(host_tensors, read_events)``) and converts it with
        :meth:`process_decode_output_host`, so the readback of step i overlaps the replay of step i+1."""
        page_table = self.page_table if page_table is None else page_table
        tokens = torch.as_tensor(tokens).reshape(-1, 1).to(torch.int64)
        start_pos = torch.as_tensor(start_pos).reshape(-1).to(torch.int64)
        B = tokens.shape[0]
        Bmax = self.max_batch_size
        assert B <= Bmax, f"decode batch {B} > max_batch_size {Bmax}"
        if B < Bmax:
            # the decode trace is captured at max_batch_size: pad the host view to it. Inactive rows carry position -1
            # (ttnn.plus_one skips negative entries, the paged update writes nothing for them) and token 0.
            tokens = torch.cat([tokens, torch.zeros(Bmax - B, 1, dtype=torch.int64)], dim=0)
            start_pos = torch.cat([start_pos, torch.full((Bmax - B,), -1, dtype=torch.int64)], dim=0)
            page_table = (
                self.page_table.clone()
                if page_table is None
                else torch.cat([page_table[:B], self.page_table[B:Bmax]], dim=0)
            )
        if reload_inputs is None:
            reload_inputs = self._first_decode  # host tokens/positions are authoritative right after prefill
        if enable_trace and not on_device_sampling and not self._host_trace_warm:
            # A trace captured mid-generation executes the step twice (compile pass + replay) and corrupts the GDN state
            # for that step; the compatibility mode runs eagerly unless its trace was captured at build time.
            if not getattr(self, "_host_untraced_logged", False):
                logger.info(
                    "host-sampling compatibility mode runs untraced (build with warm_host_trace=True to trace it)"
                )
                self._host_untraced_logged = True
            enable_trace = False
        out = self._tt.decode_forward(
            tokens,
            start_pos,
            page_table=page_table[:Bmax],
            kv_cache=self.kv_cache,
            enable_trace=enable_trace,
            read_from_device=read_from_device,
            sampling_params=GREEDY if on_device_sampling else None,
            reset_batch=self._first_decode,
            reload_inputs=reload_inputs,
        )
        self._first_decode = False
        if getattr(self, "_program_guard", False):
            self.assert_no_new_programs("(decode_forward)")
        if not read_from_device:
            self._pending_decode_B = B
            return out  # device handle: list of per-rank (tokens, log_probs) or logits tensors
        return self._finish_decode_output(out, B, on_device_sampling)

    def _finish_decode_output(self, out, B, on_device_sampling):
        if isinstance(out, (tuple, list)):  # host path returns (logits, log_probs)
            out = out[0]
        if on_device_sampling:
            return torch.as_tensor(out).reshape(-1)[:B].to(torch.int64)
        return torch.as_tensor(out).float().reshape(self.max_batch_size, -1)[:B]

    # --- serving contract: split submit / readback / host conversion (the vLLM plugin calls these three by name) ---
    def read_decode_output(self, tt_out, async_read=False):
        """Read the device output of a ``read_from_device=False`` step. ``async_read=True`` returns
        ``(host_tensors, read_events)`` and does not block; synchronize the events before converting."""
        return self._tt.read_decode_output(tt_out, async_read=async_read)

    def process_decode_output_host(self, tt_out, is_tokens=True):
        """Convert host tensors from :meth:`read_decode_output` to torch (``[Bmax]`` tokens or ``[Bmax, vocab]`` logits;
        the caller slices its active rows)."""
        return self._tt.process_decode_output_host(tt_out, is_tokens=is_tokens)

    def decode_step_async(self, tokens, start_pos, **kwargs):
        """Submit one on-device-sampled decode step and return a :class:`PendingTokens` whose ``result()`` gives the
        ``[B]`` tokens. Reading is deferred: call ``result()`` after submitting the next step to overlap the readback.
        """
        out = self.decode_forward(tokens, start_pos, read_from_device=False, **kwargs)
        host, events = self._tt.read_decode_output(out, async_read=True)
        return PendingTokens(self, host, events, self._pending_decode_B)

    def prefill_logits(self, prompt_token_ids):
        self.reset()
        tokens = torch.tensor(list(prompt_token_ids), dtype=torch.int64).reshape(1, -1)
        return self.prefill_forward(
            tokens, page_table=self.page_table[:1], prompt_lens=[tokens.shape[1]], return_all_logits=True
        )

    def generate(
        self, prompt_token_ids, max_new_tokens, *, next_input=None, enable_trace=True, on_device_sampling=True, **kwargs
    ):
        """Greedy generation; ``next_input(i, pred_i)`` may override the token fed at step i (teacher forcing). Returns
        the model's own ``max_new_tokens`` predictions."""
        self.reset()
        tokens = torch.tensor(list(prompt_token_ids), dtype=torch.int64).reshape(1, -1)
        P = tokens.shape[1]
        logits = self.prefill_forward(tokens, page_table=self.page_table[:1], prompt_lens=[P])
        pred = int(
            torch.argmax(logits[0, -1]).item()
        )  # first prediction from the prefill logits (host argmax of one row)
        preds = []
        pos = P
        for i in range(int(max_new_tokens)):
            preds.append(pred)
            forced = next_input(i, pred) if next_input is not None else None
            tok = pred if forced is None else int(forced)
            # host authority when a token is forced (teacher forcing) or at the first step after prefill; otherwise the
            # device feeds its sampled token (tt_out_tok) and advances the positions itself
            reload = self._first_decode or (forced is not None and tok != pred) or not on_device_sampling
            out = self.decode_forward(
                torch.tensor([[tok]]),
                torch.tensor([pos]),
                enable_trace=enable_trace,
                on_device_sampling=on_device_sampling,
                reload_inputs=reload,
            )
            pred = int(out[0].item()) if on_device_sampling else int(torch.argmax(out[0]).item())
            pos += 1
        return preds


L1_SMALL_SIZE = 24576  # the GDN prefill depthwise conv1d (GDN_CONV1D_L1_SMALL_SIZE)
TRACE_REGION_SIZE = int(os.environ.get("QWEN35_TRACE_REGION", 512 * 1024 * 1024))


def _check_mesh(mesh_device):
    """The decode trace needs a trace region and the GDN prefill conv1d needs the small-L1 region; the bare
    ``ttnn.open_mesh_device(mesh_shape=...)`` the readiness runners use gives neither, and a FABRIC_1D mesh cannot be
    closed and re-opened inside one process on this board (hang, stage-6 work_log). Fail fast with the fix."""
    try:
        trace = ttnn.get_memory_view(mesh_device, ttnn.BufferType.TRACE).total_bytes_per_bank
        small = ttnn.get_memory_view(mesh_device, ttnn.BufferType.L1_SMALL).total_bytes_per_bank
    except Exception as e:  # API drift: cannot check
        logger.warning(f"memory view unavailable ({e}); assuming the mesh was opened with a trace region")
        return
    if trace <= 0 or small < L1_SMALL_SIZE:
        raise RuntimeError(
            f"mesh opened with trace_region_size={trace} l1_small_size={small}; this model needs "
            f"trace_region_size>={TRACE_REGION_SIZE} and l1_small_size>={L1_SMALL_SIZE}. Open the mesh with "
            "models/autoports/qwen_qwen3_5_122b_a10b/tools/readiness.py (wraps the readiness runners) or "
            "ttnn.open_mesh_device(MeshShape(1,4), l1_small_size=24576, trace_region_size=...)."
        )


def build_generator(model_dir, mesh_device, **kwargs):
    _check_mesh(mesh_device)
    if os.environ.get("QWEN35_WARM_HOST_TRACE") == "1":
        kwargs.setdefault("warm_host_trace", True)
    owns_mesh = False
    max_seq_len = kwargs.get("max_seq_len") or int(os.environ.get("QWEN35_MAX_SEQ_LEN", "0")) or None
    layer_indices = kwargs.get("layer_indices")
    env_layers = os.environ.get("QWEN35_LAYER_INDICES")
    if layer_indices is None and env_layers:
        layer_indices = [int(s) for s in env_layers.split(",")]
    gen = Qwen35MoeGenerator(
        mesh_device,
        max_batch_size=int(kwargs.get("max_batch_size", os.environ.get("QWEN35_MAX_BATCH", 1))),
        max_seq_len=max_seq_len,
        layer_indices=layer_indices,
        block_size=int(kwargs.get("block_size", BLOCK_SIZE)),
        warm_decode_trace=bool(kwargs.get("warm_decode_trace", True)),
        warm_host_trace=bool(kwargs.get("warm_host_trace", False)),
        warm_prefill=bool(kwargs.get("warm_prefill", True)),
        warm_prefill_trace=bool(kwargs.get("warm_prefill_trace", os.environ.get("QWEN35_PREFILL_TRACE", "1") == "1")),
    )
    gen._owns_mesh = owns_mesh
    return gen
