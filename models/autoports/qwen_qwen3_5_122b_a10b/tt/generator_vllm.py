# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""vLLM adapter for the Qwen/Qwen3.5-122B-A10B autoport on a (1,4) Blackhole mesh (text only).

A thin ``tt_transformers`` Generator subclass (through the qwen36 adapter) that delegates to the autoport's low-level
generator (``tt/generator.py``): the model is built by ``create_model`` (so the datatype-sweep precision policy
``doc/datatype_sweep/selected_precision_config.json`` is applied and asserted), vLLM owns the paged KV cache
(``allocate_kv_cache`` -> ``Qwen35MoeGenerator.bind_kv_cache``, one scratch block reserved past vLLM's blocks), the GDN
recurrent / conv state lives inside the model per decode slot, prefill runs the warmed masked-bucket traces per request
into its slot with the request's sampling params applied to the on-device sampler (``prefill_serving``), and decode is the
base Generator's traced decode with the canonical split sampling (``decode_forward(read_from_device=False)`` ->
``read_decode_output(async_read=True)`` -> ``process_decode_output_host``). Warm-up follows the plugin's two phases:
phase 1 (no trace) compiles every prefill program, the eager decode and every sampler variant; phase 2 captures the
decode trace and then the prefill bucket traces (compile-after-capture rule, doc/optimized_full_model/README.md).

Registration: ``vllm_metadata.json`` next to this package (EXTRA_MODELS_DIR bundle) or
``platform.py::register_tt_models`` ("TTQwen3_5MoeForConditionalGeneration").
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Mapping, Optional

import torch
from loguru import logger
from vllm.model_executor.models.qwen3_5 import (
    Qwen3_5MoeProcessingInfo,
    Qwen3VLDummyInputsBuilder,
    Qwen3VLMultiModalProcessor,
)
from vllm.multimodal import MULTIMODAL_REGISTRY

import ttnn
from models.autoports.qwen_qwen3_5_122b_a10b.tt.generator import BLOCK_SIZE, Qwen35MoeGenerator
from models.autoports.qwen_qwen3_5_122b_a10b.tt.model import create_model
from models.common.sampling.generator import broadcast_sampling_params, format_sampling_params
from models.demos.blackhole.qwen36.tt.qwen36_vllm import Qwen36ForCausalLM
from models.tt_transformers.tt.generator import Generator

MODEL_DIR = Path(__file__).resolve().parents[1]
HF_MODEL = "Qwen/Qwen3.5-122B-A10B"


def _context_contract():
    with (MODEL_DIR / "doc" / "context_contract.json").open() as f:
        return json.load(f)


class TT_Qwen3_5MoeProcessingInfo(Qwen3_5MoeProcessingInfo):
    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        # text-only scope of this bringup: the architecture is multimodal, the served model takes no images / videos
        return {"image": 0, "video": 0}


@MULTIMODAL_REGISTRY.register_processor(
    Qwen3VLMultiModalProcessor, info=TT_Qwen3_5MoeProcessingInfo, dummy_inputs=Qwen3VLDummyInputsBuilder
)
class Qwen35MoeForConditionalGeneration(Qwen36ForCausalLM):
    """vLLM bridge for the Qwen3.5-122B-A10B autoport (HF arch Qwen3_5MoeForConditionalGeneration)."""

    # One decode trace at max_batch_size (no decode bucketing); the generator's persistent decode inputs are refreshed in
    # place, so the base Generator's trace I/O reuse annotation stays on.
    _tt_allow_decode_trace_buffer_reuse = True

    # supports_async_decode: the traced decode keeps token + position on device (the sampler writes the next token into
    # the persistent decode input, ttnn.plus_one advances the position) and the split submit / read / host-format contract
    # is implemented (Qwen35MoeGenerator.decode_forward(read_from_device=False) -> read_decode_output -> process_...).
    # Overlap stays off until the page-growth refresh of the plugin revision is proven (doc/vllm_integration/README.md).
    model_capabilities = {
        "supports_prefix_caching": False,
        "supports_async_decode": True,
        "supports_async_decode_overlap": False,
        "supports_sample_on_device": True,
        "supports_chunked_prefill": False,
    }

    # ------------------------------------------------------------------ construction
    @classmethod
    def initialize_vllm_model(
        cls, hf_config, mesh_device, max_batch_size, max_seq_len=None, tt_data_parallel=1, optimizations=None, **kwargs
    ):
        if int(tt_data_parallel) != 1:
            raise ValueError(f"{cls.__name__} serves one (1,4) mesh; got tt_data_parallel={tt_data_parallel}")
        contract = _context_contract()
        context = int(contract["datatype_sweep_stage8"]["current_supported_context"])
        max_seq_len = int(max_seq_len or context)
        if max_seq_len > context:
            raise ValueError(f"max_model_len {max_seq_len} exceeds the context contract ({context})")
        os.environ.setdefault("HF_MODEL", HF_MODEL)
        B = int(max_batch_size)
        layer_indices = os.environ.get("QWEN35_LAYER_INDICES")  # reduced serving target for the adapter bring-up loop
        layer_indices = [int(x) for x in layer_indices.split(",")] if layer_indices else None
        args, model = create_model(mesh_device, max_batch_size=B, max_seq_len=max_seq_len, layer_indices=layer_indices)
        self = cls([model], [args], mesh_device)
        # the low-level generator shares THIS Generator object (trace ids, sampling state, token feedback hook)
        self._gen = Qwen35MoeGenerator(
            mesh_device,
            max_batch_size=B,
            max_seq_len=max_seq_len,
            external_kv=True,
            tt_generator=self,
            prebuilt=(args, model),
        )
        self._decode_token_feedback_buffer = self._gen._token_feedback_buffer
        # Seeded requests reproduce regardless of batch composition: the common sampler salts the device seed of a request
        # whose seed another live slot already holds (smallest free salt), so a seed-42 request's stream depended on how
        # many seed-42 siblings were admitted before it in that batch (readiness smoke test test_mixed_params_batch, two
        # runs with different admission order). With salting off, (seed, position) alone defines the stream; identical
        # seeded requests in one batch then produce identical text - the semantics a seed promises.
        if getattr(model, "sampling", None) is not None:
            model.sampling.seed_manager.salt_duplicate_seeds = False
        self._context = max_seq_len
        self._warm_phase = {"prefill_programs": False, "decode_programs": False, "traces": False}
        self._live_slots = set()  # slots with a running request as of the last decode (start_pos >= 0) or prefill
        logger.info(
            f"{cls.__name__}: B={B} max_model_len={max_seq_len} layers={len(model.layers)} policy="
            f"{getattr(model, 'precision_policy', {}).get('config_id')} (vLLM owns the KV cache)"
        )
        # positive evidence in the server log that the compile-after-capture guard is armed inside the engine
        logger.info(
            f"PROGRAM_GUARD {'active' if getattr(self._gen, '_program_guard', False) else 'OFF'} "
            f"(QWEN35_PROGRAM_GUARD={os.environ.get('QWEN35_PROGRAM_GUARD', '<unset>')}): every served prefill / decode "
            "asserts that no program was compiled after the trace captures"
        )
        self._decode_steps = 0
        return self

    def _log_program_cache(self, where):
        """PROGRAM_CACHE evidence line: entries now vs. after the warm-up captures (equal == nothing compiled since)."""
        now = self._gen.program_cache_entries()
        base = self._gen._program_cache_after_warmup
        logger.info(
            f"PROGRAM_CACHE {now} entries {where} (after the warm-up captures: {base}; new: {now - base if base is not None else 'n/a'})"
        )

    # Per-device DRAM budget of the serving build (GiB), measured: the stage-8 context contract gives the non-KV footprint
    # of the B=1 build with every trace parked (17.660) and the total (31.331); the first 48-layer server build at
    # max_num_seqs 32 measured 30.434 GiB allocated with a 11.64 GiB KV pool, i.e. a non-KV footprint of 18.79 GiB ->
    # +36.5 MB per additional decode slot (GDN recurrent + conv state 19.8 MB, sampler / penalty rows, B-sized decode buffers
    # and templates). The masked-bucket prefill programs allocate transients while a request runs (the expert-sorted
    # dispatch of the 2048-token bucket asked for one 512 MiB buffer during its trace capture) -> 1.25 GiB headroom.
    _PER_SLOT_GIB = 36.5 / 1024
    _TRANSIENT_HEADROOM_GIB = 1.25

    @classmethod
    def get_max_tokens_all_users(
        cls, model_name="", num_devices=1, tt_data_parallel=1, max_model_len=None, max_num_seqs=None, **kwargs
    ):
        """The shared paged-KV token pool vLLM may allocate on each device = DRAM total - non-KV footprint of the build at
        this max_num_seqs - prefill transient headroom, in the policy's KV bytes/token (context contract
        datatype_sweep_stage8), rounded down to blocks. QWEN35_MAX_TOKENS_ALL_USERS overrides. Independent of
        max_model_len x max_num_seqs (8.4 M tokens = 51 GiB per device at 262144 x 32 would never fit); one user still
        reaches the full 262144 context, 32 users share the pool."""
        override = os.environ.get("QWEN35_MAX_TOKENS_ALL_USERS")
        if override:
            return int(override)
        c = _context_contract()["datatype_sweep_stage8"]
        B = int(max_num_seqs or 1)
        nonkv = float(c["non_kv_footprint_gib"]) + (B - 1) * cls._PER_SLOT_GIB
        pool_gib = float(c["dram_total_gib_per_device"]) - nonkv - cls._TRANSIENT_HEADROOM_GIB
        tokens = int(pool_gib * 2**30 / int(c["kv_bytes_per_token_per_device"]))
        tokens = max(tokens // BLOCK_SIZE * BLOCK_SIZE, BLOCK_SIZE * B)
        logger.info(
            f"KV token pool for {B} slots: {pool_gib:.3f} GiB = {tokens} tokens ({tokens / max(B, 1):.0f} per user at full "
            f"occupancy; non-KV footprint {nonkv:.3f} GiB incl. slots, headroom {cls._TRANSIENT_HEADROOM_GIB} GiB)"
        )
        return tokens

    def allocate_kv_cache(self, kv_cache_shape, dtype, num_layers):
        """vLLM's paged KV cache: ``kv_cache_shape`` = (num_blocks, heads, block_size, head_dim). One scratch block is
        reserved past vLLM's blocks (id = num_blocks, 32-aligned allocation) for the pad rows of the masked-bucket
        prefill; the cache dtype is the precision policy's ``kv_cache_dtype`` (bfp8), not vLLM's request."""
        num_blocks, heads, block, hd = (int(d) for d in kv_cache_shape)
        assert block == BLOCK_SIZE, f"block size {block} != the generator's {BLOCK_SIZE}"
        n_model = len(self._gen.model._attention_layer_indices)
        if num_layers != n_model:
            # vLLM counts the HF config's attention layers (12); the reduced serving target (QWEN35_LAYER_INDICES) has
            # fewer - allocate for the layers the model actually has, vLLM's per-layer budget is host bookkeeping only
            logger.warning(
                f"vLLM asked for {num_layers} KV layers, the built model has {n_model} attention layers (reduced target)"
            )
        alloc_blocks = math.ceil((num_blocks + 1) / 32) * 32
        shape = (alloc_blocks, int(self._gen.args.n_local_kv_heads), block, int(self._gen.args.head_dim))
        kv = self._gen.model.allocate_kv_caches(shape, self._gen.kv_dtype, batch_size=self._gen.max_batch_size)
        width = min(math.ceil(self._context / block), num_blocks)  # == the plugin's max_num_blocks_per_req
        self._gen.bind_kv_cache(kv, num_blocks, width)
        logger.info(
            f"KV cache: vLLM {num_blocks} blocks (+1 scratch, {alloc_blocks} allocated) x {block} tokens, "
            f"{self._gen.kv_dtype}, {num_layers} attention layers, page-table width {width}"
        )
        return kv

    # ------------------------------------------------------------------ warm-up (plugin phases)
    def warmup_model_prefill(self, kv_cache, enable_trace, *args, **kwargs):
        if not enable_trace:
            if not self._warm_phase["prefill_programs"]:
                assert self._gen.kv_cache is not None, "allocate_kv_cache must run before the warm-up"
                self._gen._warm_prefill()
                self._warm_phase["prefill_programs"] = True
            return
        # phase 2 for prefill is folded into warmup_model_decode(enable_trace=True): the bucket traces are captured AFTER
        # the decode trace (every persistent allocation exists by then)
        self.already_warmed_up_prefill = True

    def warmup_model_decode(self, *args, **kwargs):
        kwargs.pop("non_greedy_decoding_on_device", None)
        self._validate_device_sampling_request(kwargs.get("can_sample_on_device", False))
        if not kwargs.get("enable_trace", False):
            if not self._warm_phase["decode_programs"]:
                if not self._warm_phase["prefill_programs"]:
                    self._gen._warm_prefill()
                    self._warm_phase["prefill_programs"] = True
                self._gen._warm_decode_eager()
                self._gen._program_cache_before_captures = self._gen.program_cache_entries()
                self._warm_phase["decode_programs"] = True
            return
        if not self._warm_phase["traces"]:
            self._gen.capture_traces()
            self._warm_phase["traces"] = True

    # ------------------------------------------------------------------ prefill
    def prefill_forward(self, tokens, page_table, kv_cache, prompt_lens, **kwargs):
        """vLLM prefill of N requests into their decode slots. ``start_pos`` must be 0 (chunked prefill / prefix caching
        are not supported: full prompts, any length <= max_model_len). Returns ``(tokens [N,1] int64, rope_deltas [N])``
        when the plugin asked for on-device sampling (``sampling_params``), else ``(logits [N,1,vocab], rope_deltas)``.
        """
        N = tokens.shape[0]
        start_pos = kwargs.get("start_pos")
        if start_pos is not None and int(torch.as_tensor(start_pos).max()) != 0:
            raise NotImplementedError("chunked / cached-prefix prefill is not supported (model_capabilities)")
        if self._has_visual(kwargs, "pixel_values") or self._has_visual(kwargs, "pixel_values_videos"):
            raise NotImplementedError("text-only serving: this bringup does not serve images or videos")
        plens = [int(prompt_lens[u]) for u in range(N)] if prompt_lens is not None else [tokens.shape[1]] * N
        slots = kwargs.get("empty_slots")
        slots = list(range(N)) if slots is None else [int(s) for s in slots]
        sampling_params = kwargs.get("sampling_params")
        pt = page_table if isinstance(page_table, torch.Tensor) else ttnn.to_torch(page_table)
        sampler = None
        if sampling_params is not None:
            sampling_module = self.model[0].sampling
            assert sampling_module is not None, "on-device sampling requested without a sampling module"
            total_batch = sampling_module.tt_sampling.max_batch_size * getattr(self.model[0], "sampling_dp", 1)
            # Seeded reproducibility: a request's device RNG stream is (seed, salt); the salt is the smallest free one
            # among the slots that still hold the same seed. Slots whose request finished keep their seed until the next
            # DECODE deactivates them, so a prefill that admits new requests before that decode would count those ghosts
            # and hand a same-seed request a different salt than on a fresh server. Drop the seed state of every slot
            # that is neither running (per the last decode's positions) nor being admitted now.
            sampling_module.seed_manager.deactivate_slots_except(sorted(self._live_slots - set(slots)))

            def sampler(logits_sh, u):
                # the request's params broadcast to every sampler row (row 0 carries the token); the counted output
                # token pollutes the other rows' penalty counters exactly as the base Generator's per-user prefill does -
                # the first decode after a layout change resets them from vLLM's prompt / output tokens
                per_request = format_sampling_params(
                    broadcast_sampling_params(sampling_params, u, slot_len=total_batch), total_batch
                )
                sampling_module.apply_prefill_state(
                    sampling_params=per_request,
                    prompt_tokens=tokens[u : u + 1, : plens[u]].to(torch.long).repeat(total_batch, 1),
                    empty_slots=[slots[u]],
                )
                tt_tokens, _ = sampling_module.sample(logits_sh, enable_trace=False)
                return int(ttnn.to_torch(ttnn.get_device_tensors(tt_tokens)[0]).reshape(-1)[0])

        logger.info(f"Prefilling {N} request(s) into slots {slots} (lengths {plens})")
        out = self._gen.prefill_serving(tokens, pt, plens, slots, sampler=sampler)
        self._live_slots |= set(slots)
        self._log_program_cache(
            f"after prefill of {N} request(s) into slots {slots} (decode steps so far: {getattr(self, '_decode_steps', 0)})"
        )
        return out, torch.zeros(N, dtype=torch.long)

    # ------------------------------------------------------------------ decode
    def decode_forward(self, *args, **kwargs):
        """Traced decode through the base Generator with the plugin's own kwargs (tokens, start_pos, page_table,
        sampling_params, reset_batch / reload_inputs, prompt_tokens / output_tokens, slot_remap, read_from_device):
        the model-internal GDN slot state follows a batch-condense ``slot_remap`` first."""
        if not getattr(self, "_decode_logged", False):
            self._decode_logged = True
            logger.info("Decode trace replay active (Qwen3.5-122B-A10B, split sampling, deferred readback)")
        if os.environ.get("QWEN35_VLLM_DEBUG_KWARGS") == "1" and getattr(self, "_decode_calls", 0) < 3:
            self._decode_calls = getattr(self, "_decode_calls", 0) + 1

            def desc(v):
                if isinstance(v, torch.Tensor):
                    return f"Tensor{tuple(v.shape)}{v.dtype} {v.reshape(-1)[:8].tolist()}"
                if hasattr(v, "__dataclass_fields__"):
                    return {f: desc(getattr(v, f)) for f in v.__dataclass_fields__}
                if isinstance(v, (list, tuple)):
                    return [desc(x) for x in v[:8]]
                return repr(v)[:80]

            logger.info(
                f"VLLM_DECODE_KWARGS args={[desc(a) for a in args]} kwargs={ {k: desc(v) for k, v in kwargs.items() if k != 'kv_cache'} }"
            )
        slot_remap = kwargs.get("slot_remap")
        if slot_remap is None and len(args) > 10:
            slot_remap = args[10]
        if slot_remap is not None:
            self.model[0]._remap_gdn_slots(slot_remap)
        start_pos = kwargs.get("start_pos", args[1] if len(args) > 1 else None)
        if start_pos is not None:
            self._live_slots = {i for i, p in enumerate(torch.as_tensor(start_pos).reshape(-1).tolist()) if int(p) >= 0}
        out = Generator.decode_forward(self, *args, **kwargs)
        if getattr(self._gen, "_program_guard", False):
            self._gen.assert_no_new_programs("(vllm decode_forward)")
        self._decode_steps = getattr(self, "_decode_steps", 0) + 1
        if self._decode_steps % 2000 == 0:
            self._log_program_cache(f"after {self._decode_steps} served decode steps")
        return out
