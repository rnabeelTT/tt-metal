# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Readiness-check / HF-style generator for Ornith-1.5-35B-A3B.

Implements the `readiness_check.contract.Generator` ABC so the packaged runners
(`run_prefill_check`, `run_autoregressive`, `run_teacher_forcing`) can drive this model
by convention from `<model_dir>/tt/generator.py::build_generator`.

The LOW level already exists: `models/demos/blackhole/qwen36` owns prefill dispatch,
paged KV + GDN state allocation and the decode trace machinery, and
`tt/ornith_vllm.py` owns model construction for serving. Everything genuinely new here
is the HIGH level -- owning the paged KV cache and page table, the greedy loop with the
teacher-forcing `next_input` callback, traced decode, and a `reset()` that zeroes state
without freeing device buffers, traces or weights.

Five constraints from this port's README are load-bearing and are honoured below.

1. `l1_small_size` (GDN_CONV1D_L1_SMALL_SIZE), `num_command_queues: 2` and a 1 GB
   `trace_region_size` are the device params every validated path on this port uses
   (bench_model.py, the text demo, all three serve profiles). The packaged readiness
   runners open the mesh with ttnn's defaults instead -- l1_small_size=0,
   trace_region_size=0, one command queue -- and there is no API to resize an open
   device's trace region, so `build_generator` RE-OPENS the mesh with the documented
   params (`_reopen_mesh_with_required_params`, disable with `reopen_device=False`).

   MEASURED, and it contradicts this port's README: on this tt-metal build the default
   params also work. Built with `reopen_device=False`, decode trace capture SUCCEEDS
   (`MeshTraceId(0)`) and traced decode measures ~33 t/s/u -- i.e. the same traced path,
   not a silent eager fallback (eager is ~3.4 t/s here). So the trace region appears to
   be grown on demand now, and the GDN conv does not need the L1-small reservation at
   these shapes. The re-open is kept as the DEFAULT anyway, because it is the only
   configuration this model's numbers were ever measured in and because the command-queue
   count differs; it is alignment with the port's device contract, not a hard requirement
   on this build.
2. A paged KV cache is mandatory for decode. The attention module only takes its decode
   branch when `use_paged_attention and T == 1`; without a page table it silently takes
   the CONCAT PREFILL branch even for one token, which is wrong for decode and dies
   inside trace capture on `!is_capturing_trace`.
3. The GDN chunk-prefill kernel needs T >= 128 and T a multiple of 32, and below that the
   single-device path is SILENTLY WRONG. Readiness prompts are shorter than that, so all
   prefill goes through the model's masked fixed-bucket path, which runs one of a few
   bucket-length programs (>= 128) and masks the GDN recurrent + conv state to the exact
   real length. It is numerically the exact-length path with a bounded program set.
4. Every masked-bucket program (and the all-logits LM-head width) is compiled at BUILD
   time, before any trace is parked. A compile after a trace is parked clobbers the trace
   (tenstorrent/tt-metal#48536) and hangs the next request.
5. A new generation must re-zero the GDN recurrent state, the fused conv state AND the
   conv window DERIVED from it (`drop_derived_conv_state()`); the derived window is not in
   the external-state list and otherwise carries the previous sequence's conv taps into
   the next one. `reset()` does all three, in place, so the decode trace's baked buffer
   addresses stay valid.

`max_num_seqs` is effectively 1 on this port (the GDN external-state hand-off is batch-1
and the batched path does not fit L1), so this generator is single-sequence. The
low-level `prefill_forward` / `decode_forward` pair keeps the caller-owned-cache shape of
the contract for `generator_vllm.py`, but batch > 1 is rejected rather than mis-served.

KNOWN, DOCUMENTED CAVEAT: this checkpoint's decode path is not bit-reproducible on this
stack (see NONDETERMINISM.md) and its decode-path logit PCC is materially worse than
prefill's. Greedy output here is therefore run-to-run variable. That is a property of the
model port, not of this generator.
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
from typing import Any, Callable, List, Optional

import torch
from loguru import logger

import ttnn


def _import_contract():
    """Import the readiness Generator ABC, adding the plugin runtime to sys.path if needed."""
    try:
        from readiness_check.contract import Generator as _Generator

        return _Generator
    except ImportError:
        pass
    root = os.environ.get("TT_MODEL_BRINGUP_ROOT")
    candidates = []
    if root:
        candidates.append(pathlib.Path(root) / "runtime")
    candidates += sorted(
        pathlib.Path.home().glob(".claude/plugins/cache/*/tt-model-bringup/*/runtime"),
        reverse=True,
    )
    for cand in candidates:
        if (cand / "readiness_check" / "contract.py").exists():
            sys.path.insert(0, str(cand))
            from readiness_check.contract import Generator as _Generator

            return _Generator
    raise ImportError(
        "readiness_check.contract is not importable. Put the tt-model-bringup plugin's "
        "`runtime` directory on PYTHONPATH (or set TT_MODEL_BRINGUP_ROOT) before importing "
        "this module."
    )


Generator = _import_contract()

from models.autoports.ornith_ai_ornith_1_5_35b_a3b.tt.model import OrnithModel  # noqa: E402
from models.autoports.ornith_ai_ornith_1_5_35b_a3b.tt.model_config import OrnithModelArgs  # noqa: E402
from models.autoports.ornith_ai_ornith_1_5_35b_a3b.tt.weight_mapping import load_ornith_state_dict  # noqa: E402
from models.demos.blackhole.qwen36.tt.generator_interface import prime_decode_trace  # noqa: E402
from models.demos.blackhole.qwen36.tt.model_config import GDN_CONV1D_L1_SMALL_SIZE  # noqa: E402
from models.tt_transformers.tt.common import Mode  # noqa: E402
from models.tt_transformers.tt.generator import Generator as TTGenerator  # noqa: E402

#: Paged KV block size. 64 is what the serve profiles and every demo path use.
BLOCK_SIZE = 64
#: Page-table sticks must be a multiple of 32 blocks for paged/chunked SDPA.
BLOCK_ALIGN = 32
#: Trace region. 1 GB is the MEASURED requirement from the serve profile, not a guess:
#: at the default 50 MB the decode trace buffers do not fit.
TRACE_REGION_SIZE = 1024 * 1024 * 1024
#: One fixed LM-head width for the all-positions prefill path, so `prefill_logits` adds
#: exactly one program shape however long the prompt is (constraint 4 above).
LOGITS_CHUNK = 128
#: Longest prompt this generator prefills. Beyond one masked bucket the model needs the
#: chunk-outer prefill TRACE (`prefill_traced_chunked`), which is a separate capture; the
#: readiness prompts are two orders of magnitude shorter, so it is refused explicitly
#: rather than silently routed onto an unwarmed path.
MAX_PREFILL_LEN = 2048


def _resolve_weights(model_dir: pathlib.Path, override: Optional[str] = None) -> str:
    """Local snapshot directory holding the checkpoint. Mirrors tt/ornith_vllm.py."""
    weights = (
        override
        or os.environ.get("MODEL_WEIGHTS_DIR")
        or os.environ.get("ORNITH_WEIGHTS")
        or os.environ.get("HF_MODEL")
    )
    if not weights:
        raise ValueError(
            "No checkpoint location. Set ORNITH_WEIGHTS (or MODEL_WEIGHTS_DIR / HF_MODEL) "
            "to the Ornith-1.5-35B-A3B snapshot, or pass hf_model=... to build_generator."
        )
    weights = os.path.expanduser(weights)
    if not os.path.isdir(weights):
        from huggingface_hub import snapshot_download

        offline = os.getenv("HF_HUB_OFFLINE") == "1" or os.getenv("CI") == "true"
        weights = snapshot_download(weights, local_files_only=offline)
    os.environ["ORNITH_WEIGHTS"] = weights
    os.environ.setdefault("HF_MODEL", weights)
    return weights


def _reopen_mesh_with_required_params(mesh_device):
    """Return a mesh device opened with this model's REQUIRED params.

    The readiness runners call `ttnn.open_mesh_device(mesh_shape=...)` with no device
    params, i.e. l1_small_size=0, trace_region_size=0, num_command_queues=1. This model
    cannot run there (see constraint 1 in the module docstring) and there is no API to
    resize an open device's trace region, so the mesh is closed and re-opened.

    The runner still holds the stale handle and closes it in its own `finally`, so
    `ttnn.close_mesh_device` is wrapped once to redirect that call to the device we
    actually opened. The runner's teardown therefore closes the right device and its exit
    status stays meaningful.
    """
    shape = tuple(int(d) for d in mesh_device.shape)
    params = {
        "l1_small_size": GDN_CONV1D_L1_SMALL_SIZE,
        "num_command_queues": int(os.environ.get("ORNITH_CQS", "2")),
        "trace_region_size": TRACE_REGION_SIZE,
    }
    logger.info(f"Re-opening mesh {shape} with Ornith's required device params {params}")
    ttnn.close_mesh_device(mesh_device)
    new_device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(*shape), **params)

    original_close = ttnn.close_mesh_device
    stale, live = mesh_device, new_device

    def _close_redirecting_stale_handle(device, *args, **kwargs):
        if device is stale:
            logger.info("close_mesh_device on the pre-reopen handle: closing the re-opened mesh instead")
            device = live
        return original_close(device, *args, **kwargs)

    ttnn.close_mesh_device = _close_redirecting_stale_handle
    return new_device


class OrnithGenerator(Generator):
    """Single-sequence generator for Ornith-1.5-35B-A3B on one Blackhole die."""

    def __init__(self, mesh_device, model, args, tokenizer, max_seq_len: int, kv_dtype=None):
        self.mesh_device = mesh_device
        self.model = model
        self.args = args
        self.tokenizer = tokenizer
        self.max_seq_len = int(max_seq_len)
        # Decode is driven through tt_transformers' Generator: it owns the decode trace
        # capture/replay, the persistent host->device input buffers and the output
        # readback. Prefill is model-owned (GDN forbids token padding and is not
        # position-general), exactly as in qwen36_vllm.
        self._inner = TTGenerator([model], [args], mesh_device)

        num_blocks = -(-self.max_seq_len // BLOCK_SIZE)
        num_blocks = max(BLOCK_ALIGN, -(-num_blocks // BLOCK_ALIGN) * BLOCK_ALIGN)
        self.num_blocks = num_blocks
        self.block_size = BLOCK_SIZE

        # KV dtype: bfloat16 is what every validated path uses (the serving adapter
        # hardcodes it). ORNITH_KV_DTYPE is honoured when explicitly set, because on this
        # model only 10 of 40 layers hold a KV cache so its precision is a sweep knob.
        if kv_dtype is None:
            override = os.environ.get("ORNITH_KV_DTYPE")
            kv_dtype = getattr(ttnn, override) if override else ttnn.bfloat16
        self.kv_dtype = kv_dtype
        self.kv_cache_shape = [num_blocks, args.n_local_kv_heads, BLOCK_SIZE, args.head_dim]
        self.kv_cache = model.allocate_kv_caches(self.kv_cache_shape, kv_dtype, batch_size=1)
        # Identity page table: one sequence owning every block.
        self.page_table = torch.arange(num_blocks, dtype=torch.int32).reshape(1, num_blocks)

        self._bind_inplace_gdn_state()
        self._alloc_kv_zero_buffer()
        self._warm_program_set()

        eos = getattr(tokenizer, "eos_token_id", None)
        self.eos_ids = set(eos if isinstance(eos, (list, tuple)) else ([eos] if eos is not None else []))
        cfg_eos = getattr(getattr(args, "hf_config", None), "eos_token_id", None)
        if isinstance(cfg_eos, int):
            self.eos_ids.add(cfg_eos)

        self._decode_trace_primed = False
        self._torn_down = False

    # ---- construction-time device state ---------------------------------------

    def _gdn_layers(self):
        return [layer for layer in self.model.layers if not layer.is_full_attention]

    def _bind_inplace_gdn_state(self):
        """Bind GDN to the persistent external buffers and make its resets IN PLACE.

        `allocate_kv_caches` installs external recurrent/conv buffers, but leaves the
        layers in the mode where `_reset_gdn_state_for_new_sequence()` calls
        `reset_state(1)`, which REALLOCATES `recurrent_state` and drops
        `fused_conv_state`. That unbinds the buffers a parked decode trace has baked
        addresses for, so the second prompt of a multi-prompt run would replay against
        freed/stale state. `_chunk_inplace_state` selects the in-place reset path
        (`_reset_dn_state_inplace`, zero-copy into the same buffers) instead. This is the
        same binding `capture_prefill_trace_chunked` performs for the serving path; it is
        done directly here because this generator does not need the chunk prefill trace.
        """
        model = self.model
        assert model._deltanet_external_states, "allocate_kv_caches must run before binding GDN state"
        for layer, (ext_rec, ext_conv) in zip(self._gdn_layers(), model._deltanet_external_states):
            dn = layer.attention
            dn.recurrent_state = ext_rec
            dn.fused_conv_state = ext_conv
            dn.drop_derived_conv_state()
            dn._chunk_inplace_state = True
        model._init_dn_zero_buffers()

    def _alloc_kv_zero_buffer(self):
        """One persistent zero block, reused to wipe every paged KV cache in `reset()`.

        `reset()` must zero the cache without freeing it (the decode trace and the vLLM
        contract both depend on the buffers surviving), so the zeroing is a device copy
        from this buffer rather than a realloc. One extra cache-sized block is a few MB.
        """
        self._kv_zero = ttnn.zeros(
            self.kv_cache_shape,
            dtype=self.kv_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _warm_program_set(self):
        """Compile every program a request can reach, BEFORE any trace is parked.

        Two families: the masked fixed-bucket prefill programs (one set per bucket, plus
        the per-fill-width paged_fill_cache programs), and the fixed-width LM head the
        all-positions `prefill_logits` path uses. A compile after the decode trace is
        parked clobbers it and hangs the next request (tt-metal#48536), which is exactly
        what a readiness run does: many prompts through one generator.
        """
        model = self.model
        limit = model._mask_bucket_for(min(self.max_seq_len, MAX_PREFILL_LEN))
        self._buckets = tuple(b for b in model._PREFILL_MASK_BUCKETS if b <= limit)
        logger.info(f"Warming masked-bucket prefill programs for buckets {self._buckets}")
        model.warmup_prefill_masked_buckets(self.page_table, buckets=self._buckets)
        logger.info(f"Warming the all-logits LM head at M={LOGITS_CHUNK}")
        self._warm_logits_chunk()
        ttnn.synchronize_device(self.mesh_device)

    def _warm_logits_chunk(self):
        dim = self.args.dim
        dummy = ttnn.zeros(
            [1, LOGITS_CHUNK, dim],
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        out = self.model._lm_head(self.model.norm(dummy, mode=Mode.PREFILL))
        ttnn.deallocate(out)
        ttnn.deallocate(dummy)

    # ---- prefill --------------------------------------------------------------

    def _check_prefill_len(self, length: int):
        if length < 1:
            raise ValueError("prompt must contain at least one token")
        if length > MAX_PREFILL_LEN:
            raise NotImplementedError(
                f"prompt of {length} tokens exceeds the single masked-bucket limit of "
                f"{MAX_PREFILL_LEN}. Longer prompts need the chunk-outer prefill trace "
                "(Qwen36Model.capture_prefill_trace_chunked + prefill_traced_chunked), "
                "which this generator does not park."
            )
        if length > self.max_seq_len:
            raise ValueError(f"prompt of {length} tokens exceeds max_seq_len={self.max_seq_len}")

    def _prefill_last_logits(self, token_ids: torch.Tensor, valid_len: int) -> torch.Tensor:
        """Masked fixed-bucket prefill. Returns host torch logits [1, 1, vocab]."""
        self._check_prefill_len(valid_len)
        logits = self.model.prefill_masked_bucket(token_ids, self.page_table, actual_len=valid_len)
        out = ttnn.to_torch(logits).float().reshape(1, 1, -1)
        if torch.isnan(out).any():
            raise RuntimeError("NaN in prefill logits")
        return out

    def _prefill_all_logits(self, token_ids: torch.Tensor, valid_len: int) -> torch.Tensor:
        """Masked fixed-bucket prefill returning logits at EVERY real position.

        Same forward as `prefill_masked_bucket` -- the model's own masked-bucket prologue
        (GDN re-zero, per-request RoPE staging, bucket padding) followed by
        `_forward_prefill_chunk_masked` -- but the norm + LM head are applied to all rows
        instead of the one-hot-selected last row. Positions >= valid_len are bucket
        padding and are dropped; the GDN state is masked to valid_len and attention is
        causal, so the kept rows are the exact-length answers.

        The head runs in fixed LOGITS_CHUNK-row slices so this path adds exactly one
        program shape (warmed at build) regardless of prompt length, and so the host never
        holds more than LOGITS_CHUNK x vocab floats at a time.
        """
        model = self.model
        self._check_prefill_len(valid_len)
        bucket = model._mask_bucket_for(valid_len)
        assert bucket in self._buckets, f"bucket {bucket} was not warmed at build time"

        real = token_ids[:, :valid_len].to(torch.int32)
        if bucket > valid_len:
            token_buf = torch.cat([real, torch.zeros(1, bucket - valid_len, dtype=torch.int32)], dim=1)
        else:
            token_buf = real

        model._reset_gdn_state_for_new_sequence()
        model._build_request_rope(real, None)
        model._set_vision_merge(token_buf, None, 0)
        hidden = model._forward_prefill_chunk_masked(token_buf, valid_len, 0, self.page_table, bucket)
        ttnn.synchronize_device(self.mesh_device)

        dim = hidden.shape[-1]
        rows: List[torch.Tensor] = []
        for start in range(0, bucket, LOGITS_CHUNK):
            if start >= valid_len:
                break
            piece = ttnn.slice(hidden, (0, start, 0), (1, start + LOGITS_CHUNK, dim))
            piece = ttnn.to_memory_config(piece, ttnn.DRAM_MEMORY_CONFIG)
            normed = model.norm(piece, mode=Mode.PREFILL)
            chunk_logits = model._lm_head(normed)
            rows.append(ttnn.to_torch(chunk_logits).float().reshape(LOGITS_CHUNK, -1))
            ttnn.deallocate(chunk_logits)
            ttnn.deallocate(piece)
        ttnn.deallocate(hidden)

        out = torch.cat(rows, dim=0)[:valid_len].unsqueeze(0)
        if torch.isnan(out).any():
            raise RuntimeError("NaN in prefill logits")
        return out

    # ---- low-level contract API (also the vLLM delegation target) -------------

    def prefill_forward(
        self,
        tokens: torch.Tensor,
        *,
        page_table: torch.Tensor,
        kv_cache: Any,
        prompt_lens: List[int],
        return_all_logits: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Run prefill, updating the model-bound paged KV + GDN state in place.

        `kv_cache` is accepted and ignored: on this hybrid model the KV caches and the GDN
        recurrent/conv state are bound to the modules (`set_paged_kv_caches`,
        `set_external_state`), which is the same contract qwen36_vllm ships.
        `page_table` IS honoured -- it selects the blocks prefill fills.
        """
        if tokens.shape[0] != 1:
            raise NotImplementedError(
                f"batch {tokens.shape[0]} prefill is not supported: max_num_seqs is 1 on this "
                "port (the GDN external-state hand-off is batch-1 and the batched decode "
                "kernel does not fit L1). See the README's batch>1 section."
            )
        valid_len = int(prompt_lens[0]) if prompt_lens is not None else int(tokens.shape[1])
        pt = page_table if page_table is not None else self.page_table
        saved_pt, self.page_table = self.page_table, pt
        try:
            if return_all_logits:
                return self._prefill_all_logits(tokens, valid_len)
            return self._prefill_last_logits(tokens, valid_len)
        finally:
            self.page_table = saved_pt

    def decode_forward(
        self,
        tokens: torch.Tensor,
        start_pos: torch.Tensor,
        *,
        page_table: torch.Tensor,
        kv_cache: Any,
        enable_trace: bool = True,
        **kwargs: Any,
    ) -> torch.Tensor:
        """One decode step. Returns host logits [batch, vocab].

        `kv_cache` is accepted and ignored for the same reason as in `prefill_forward`.
        """
        if tokens.shape[0] != 1:
            raise NotImplementedError("batch > 1 decode is not supported on this port (max_num_seqs=1)")
        out = self._inner.decode_forward(
            tokens,
            start_pos,
            page_table=page_table if page_table is not None else self.page_table,
            kv_cache=None,
            enable_trace=enable_trace,
            read_from_device=True,
            **kwargs,
        )
        logits = out[0] if isinstance(out, tuple) else out
        logits = logits.float().reshape(tokens.shape[0], -1)
        if torch.isnan(logits).any():
            raise RuntimeError("NaN in decode logits")
        return logits

    # ---- high-level contract API ---------------------------------------------

    def prefill_logits(self, prompt_token_ids: List[int]) -> torch.Tensor:
        """Logits at every position of one unpadded prompt: [1, prompt_len, vocab]."""
        self.reset()
        ids = torch.tensor([list(prompt_token_ids)], dtype=torch.long)
        return self.prefill_forward(
            ids,
            page_table=self.page_table,
            kv_cache=self.kv_cache,
            prompt_lens=[ids.shape[1]],
            return_all_logits=True,
        )

    def generate(
        self,
        prompt_token_ids: List[int],
        max_new_tokens: int,
        *,
        next_input: Optional[Callable[[int, int], int]] = None,
        enable_trace: bool = True,
        stop_on_eos: bool = True,
        **kwargs: Any,
    ) -> List[int]:
        """Greedy HF-style generation. Returns the model's OWN predictions.

        Sampling is argmax only. The returned list is always the model's predictions, even
        when `next_input` overrides what is fed next -- that is what the readiness check
        scores. With `next_input` set (teacher forcing) the loop never stops early, so the
        callback is invoked exactly `max_new_tokens` times.

        `enable_trace=True` runs every decode step through the captured trace and FAILS if
        the capture did not take; there is no eager fallback. `enable_trace=False` exists
        only as a model-local debug path.
        """
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be >= 1")
        prompt_token_ids = [int(t) for t in prompt_token_ids]
        prompt_len = len(prompt_token_ids)
        if prompt_len + max_new_tokens > self.max_seq_len:
            raise ValueError(
                f"prompt ({prompt_len}) + max_new_tokens ({max_new_tokens}) exceeds "
                f"max_seq_len={self.max_seq_len}; rebuild the generator with a larger max_seq_len"
            )

        self.reset()
        ids = torch.tensor([prompt_token_ids], dtype=torch.long)
        logits = self._prefill_last_logits(ids, prompt_len)
        predicted = int(torch.argmax(logits.reshape(-1)).item())

        predictions = [predicted]
        fed = next_input(0, predicted) if next_input is not None else predicted
        pos = prompt_len

        if max_new_tokens > 1:
            if enable_trace:
                self._ensure_decode_trace(fed, pos)
            for step in range(1, max_new_tokens):
                logits = self.decode_forward(
                    torch.tensor([[int(fed)]], dtype=torch.long),
                    torch.tensor([pos], dtype=torch.long),
                    page_table=self.page_table,
                    kv_cache=self.kv_cache,
                    enable_trace=enable_trace,
                )
                predicted = int(torch.argmax(logits.reshape(-1)).item())
                predictions.append(predicted)
                fed = next_input(step, predicted) if next_input is not None else predicted
                pos += 1
                # Free-running only: teacher forcing must cover the full reference length.
                if next_input is None and stop_on_eos and predicted in self.eos_ids:
                    break
        return predictions

    def _ensure_decode_trace(self, token: int, pos: int):
        """Capture the decode trace once, at the real post-prefill position and state.

        `prime_decode_trace` snapshots and restores the GDN recurrent + conv state around
        the capture. That is required, not defensive: the stock capture runs the forward
        twice (compile pass + capture pass) and GDN's state is a running accumulation, so
        without the snapshot the first real replay would start from a state advanced two
        extra steps.
        """
        if self._decode_trace_primed:
            return
        prime_decode_trace(
            self._inner,
            self.model,
            torch.tensor([[int(token)]], dtype=torch.long),
            torch.tensor([int(pos)], dtype=torch.long),
            self.page_table,
        )
        trace_ids = self._inner.trace_ids_decode[False]
        if not trace_ids:
            raise RuntimeError(
                "enable_trace=True but no decode trace was captured, so decode would run "
                "eager. This model REQUIRES a trace_region_size (1 GB in the serve profile); "
                "check that the mesh device was opened with one."
            )
        logger.info(f"Decode trace captured: {dict(trace_ids)}")
        self._decode_trace_primed = True

    def reset(self) -> None:
        """Wipe per-prompt state, keeping every device buffer, trace and weight alive.

        Three things carry between prompts on this model and all three are zeroed in
        place: the paged KV cache, the GDN recurrent + fused conv state, and the conv
        window DERIVED from the fused state. The last one is the subtle one -- it lives on
        the layer rather than in the external-state list and is only rebuilt "when it is
        None", so leaving it behind silently carries the previous sequence's conv taps
        into the next generation.

        Nothing here reallocates: the KV wipe is a copy from a persistent zero block and
        the GDN wipe goes through `_reset_dn_state_inplace`, so the decode trace's baked
        buffer addresses stay valid and the next prompt replays it instead of recapturing.
        """
        if getattr(self, "_torn_down", False):
            raise RuntimeError("generator has been torn down")
        for k_cache, v_cache in self.model._paged_kv_caches:
            ttnn.copy(self._kv_zero, k_cache)
            ttnn.copy(self._kv_zero, v_cache)
        self.model._reset_gdn_state_for_new_sequence()
        for layer in self._gdn_layers():
            layer.attention.drop_derived_conv_state()
        self.model.rope.rope_delta = 0
        ttnn.synchronize_device(self.mesh_device)

    # ---- optional teardown hook the runners call before closing the device ----

    def teardown(self) -> None:
        if self._torn_down:
            return
        self._torn_down = True
        for trace_store in (self._inner.trace_ids_decode,):
            for on_device_sampling in list(trace_store.keys()):
                trace_ids = trace_store[on_device_sampling] or {}
                for trace_id in list(trace_ids.values()):
                    try:
                        ttnn.release_trace(self.mesh_device, trace_id)
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(f"release_trace({trace_id}) failed: {exc}")
                trace_store[on_device_sampling] = None
        try:
            ttnn.deallocate(self._kv_zero)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"deallocating the KV zero block failed: {exc}")
        try:
            self.model.free_kv_caches()
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"free_kv_caches failed: {exc}")


def build_generator(model_dir, mesh_device, **kwargs) -> OrnithGenerator:
    """Readiness discovery entry point.

    kwargs (all optional):
        hf_model      checkpoint dir or hub id; else ORNITH_WEIGHTS / MODEL_WEIGHTS_DIR / HF_MODEL
        max_seq_len   prompt + generation budget (default 2048; see MAX_PREFILL_LEN)
        kv_dtype      ttnn dtype for the paged KV cache (default bfloat16)
        reopen_device re-open the mesh with this model's required device params (default True)
    """
    model_dir = pathlib.Path(model_dir)
    weights = _resolve_weights(model_dir, kwargs.pop("hf_model", None))
    max_seq_len = int(kwargs.pop("max_seq_len", MAX_PREFILL_LEN))
    kv_dtype = kwargs.pop("kv_dtype", None)
    if kwargs.pop("reopen_device", True):
        mesh_device = _reopen_mesh_with_required_params(mesh_device)
    if kwargs:
        logger.warning(f"build_generator ignoring unknown kwargs {sorted(kwargs)}")

    num_devices = mesh_device.get_num_devices()
    if num_devices != 1:
        raise NotImplementedError(
            f"this generator is single-die (mesh has {num_devices} devices). Experts are "
            "replicated on this port, so extra dies serve extra USERS via data parallelism "
            "rather than one bigger replica; open a (1,1) mesh."
        )

    args = OrnithModelArgs(mesh_device, max_batch_size=1, max_seq_len=max_seq_len)
    text_config = json.loads((pathlib.Path(weights) / "config.json").read_text())["text_config"]
    state = load_ornith_state_dict(weights, text_config)
    # A real cache path is required (the embedding builds its cache name unconditionally)
    # and is what makes a second build fast: 40 layers of expert weights reload from it.
    cache = pathlib.Path(args.weight_cache_path(args.routed_expert_dtype))
    cache.mkdir(parents=True, exist_ok=True)
    model = OrnithModel(mesh_device, args, state, tensor_cache_path=cache)

    tokenizer = getattr(args, "tokenizer", None)
    if tokenizer is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(weights, trust_remote_code=True)

    return OrnithGenerator(
        mesh_device=mesh_device,
        model=model,
        args=args,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        kv_dtype=kv_dtype,
    )
