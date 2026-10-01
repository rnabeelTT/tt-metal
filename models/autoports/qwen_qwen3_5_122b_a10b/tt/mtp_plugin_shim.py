# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Variable-width autoregressive output shim for the vLLM Tenstorrent plugin (speculative / MTP decode).

The plugin knows two output contracts today: the ordinary autoregressive path (exactly one token per request per decode
step) and the block-output path (``model_capabilities["output_tokens_per_step"] = K``: exactly K tokens every step,
one fixed canvas, neutralized sampling).  An MTP verify pass emits a *variable* number of tokens per request per step
(1..K+1, where K is the number of draft tokens), which neither contract accepts.  This module keeps
``output_tokens_per_step = 1`` and monkeypatches the plugin so that a decode step's device-sampled output may be an
int32 host tensor ``[B, W]`` (``W <= max_output_tokens_per_step``) whose rows carry ``n_i`` real token ids followed
by ``-1`` pads; the ordinary ``[B, 1]`` output keeps working unchanged.

Invariant relied upon (owned by the model / generator, not by this shim):
    the model appends the K/V of every emitted token itself; the next decode input is the last emitted token at
    position ``num_tokens - 1``.
The plugin already builds each decode input as ``token_ids_cpu[row, num_tokens - 1]`` at position ``num_tokens - 1``
(model_runner.py ``_prepare_model_inputs``, decode branch), so once ``num_tokens`` advances by ``n_i`` the next step
is automatically correct.  Positions ``num_tokens .. num_tokens - 1 + K`` are written by the verify pass, which is why
the scheduler must allocate K lookahead slots (below).

Patched (plugin revision at /home/raahem/vllm-tt-plugin, ``src/vllm_tt_plugin``):

* ``model_runner._coerce_output_block`` (model_runner.py:122): accepted width becomes ``1..max_width`` instead of
  ``== output_tokens_per_step``.
* ``TTModelRunner._get_output_tokens`` (model_runner.py:2118; the device-sampling branch at :2313-2319 raises unless
  ``shape[1] == output_tokens_per_step``): for a device-sampled decode step the read-back is reshaped to
  ``[sz, W]`` per DP rank, ``1 <= W <= max_width`` is enforced, pads must be trailing and every row must carry at
  least one token.  Logprobs, grammar bitmasks (structured outputs) and host-sampled decode steps are delegated to
  the original read-back: the adaptive MTP server never takes the MTP step for such requests (they run the plain
  one-token decode), so the stock contract holds for those steps.  Prefill steps are delegated unchanged.
* ``TTModelRunner._apply_sampled_tokens_to_state`` (model_runner.py:2427): writes only the ``n_i`` non-pad tokens of
  each row into ``token_ids_cpu`` / the request's ``output_token_ids`` and advances ``num_tokens`` by ``n_i``;
  a row that would cross ``max_model_len`` is clipped (with a warning) instead of raising, the scheduler's
  ``check_stop`` finishes it length-capped.
* ``TTModelRunner._build_runner_output`` (model_runner.py:2385): emits ragged ``sampled_token_ids`` lists (pads
  dropped); vLLM's ``ModelRunnerOutput.sampled_token_ids: list[list[int]]`` accepts ragged rows.
* ``TTModelRunner._prepare_model_inputs`` (model_runner.py:1077, nested ``_is_still_prefilling`` at :1164-1183):
  the closure classifies a cached row as prefill when ``num_computed + output_tokens_per_step < num_tokens``.  After
  an ``n``-token step the scheduler schedules those ``n`` already-emitted tokens, so with width 1 every MTP step
  would be dispatched as prompt work.  The closure cannot be patched directly, so the wrapper sets
  ``self._output_tokens_per_step = max_width`` for the duration of the call (that attribute is read nowhere else
  inside ``_prepare_model_inputs``): a row is a decode row when the outstanding gap past the prompt is at most one
  step's maximum width, prefill (preemption-resume replay) only when it is larger.  Sync scheduling leaves a gap of
  ``n <= max_width``; async scheduling a gap of ``n - 1`` (the runner's ``num_tokens`` lags by the same in-flight
  step), so the bound holds in both modes.  A chunked replay whose last chunk is at most ``max_width`` tokens would
  be misclassified, but this model disables chunked prefill (``supports_chunked_prefill = False``), so a resume
  replay is always a single ``resumed_req_ids`` step.
* ``TTScheduler._update_request_with_output`` (scheduler.py:589): for width 1 it defers to
  ``AsyncScheduler._update_request_with_output``, which subtracts ``len(new_token_ids)`` from the single reserved
  placeholder and asserts ``>= 0``.  The shim calls the upstream base ``Scheduler._update_request_with_output``
  (appends token by token, runs ``check_stop`` after each, trims at EOS / stop token / max_tokens / max_model_len),
  consumes exactly the one placeholder reserved for the frame (zero when the frame is empty, clamped at zero), and
  keeps the ``async_tokens_to_discard`` drain and ``kv_cache_manager.cache_blocks`` call of the async base.
* ``TTScheduler.__init__`` (scheduler.py:112): after construction ``num_lookahead_tokens = max_width - 1`` so
  ``allocate_slots`` covers the verify window (computed + 1 new + K lookahead = ``num_tokens + K`` slots).
  Upstream clamps ``num_tokens_need_slot`` at ``max_model_len`` (kv_cache_manager.py:430), so a pass whose verify
  window would cross ``max_model_len`` has no allocated slots for those positions; the model must not write them.

Unsupported under the shim (clear errors): block-output models (``output_tokens_per_step > 1`` keeps the original
behaviour), lane mode (``TTLaneInputBatch.extract_output`` is not patched). Host sampling, logprobs and structured
outputs on a decode step are delegated to the original read-back (the model must have run the plain one-token decode
for that step - the adaptive MTP server guarantees it).

``apply(max_output_tokens_per_step)`` is idempotent for the same width and returns ``undo()``, which restores every
original attribute; ``undo`` is safe to call twice.
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np
import torch
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import RequestStatus
from vllm_tt_plugin import model_runner as _mr
from vllm_tt_plugin import scheduler as _sched

logger = logging.getLogger(__name__)

PAD_TOKEN_ID = -1

# Active patch state: {"max_width": int, "originals": dict[str, object], "undo": callable} or None.
_STATE: dict | None = None

_RUNNER_ATTRS = (
    "_get_output_tokens",
    "_build_runner_output",
    "_apply_sampled_tokens_to_state",
    "_prepare_model_inputs",
)
_SCHEDULER_ATTRS = ("_update_request_with_output", "__init__")


def is_active() -> bool:
    return _STATE is not None


def max_output_width() -> int:
    """Widest row the patched plugin accepts (K + 1); raises when the shim is not applied."""
    if _STATE is None:
        raise RuntimeError("mtp_plugin_shim is not applied")
    return _STATE["max_width"]


# ----------------------------------------------------------------------------------------------------------------------
# row helpers


def _row_widths(rows: np.ndarray) -> np.ndarray:
    """Per-row count of real token ids in a ``[B, W]`` padded block; validates trailing pads and >= 1 token per row."""
    if rows.ndim != 2:
        raise ValueError(f"MTP shim: expected a 2-D [num_requests, width] block, got shape {rows.shape}")
    nonpad = rows != PAD_TOKEN_ID
    widths = nonpad.sum(axis=1)
    if rows.shape[0] == 0:
        return widths
    # Pads are trailing iff the first ``widths[i]`` entries of every row are all real tokens.
    prefix_ok = nonpad.cumsum(axis=1)[np.arange(rows.shape[0]), np.maximum(widths - 1, 0)] == widths
    if not bool(prefix_ok.all()):
        bad = int(np.nonzero(~prefix_ok)[0][0])
        raise ValueError(f"MTP shim: row {bad} has a non-trailing {PAD_TOKEN_ID} pad: {rows[bad].tolist()}")
    if bool((widths == 0).any()):
        bad = int(np.nonzero(widths == 0)[0][0])
        raise ValueError(f"MTP shim: row {bad} emitted no token (all pads); every decode row must emit >= 1 token")
    return widths


def _as_int32_np(sampled_token_ids: torch.Tensor) -> np.ndarray:
    arr = sampled_token_ids.detach().cpu().numpy()
    if arr.dtype != np.int32:
        arr = arr.astype(np.int32, copy=False)
    return arr


# ----------------------------------------------------------------------------------------------------------------------
# model_runner patches


def _shim_coerce_output_block(sampled_token_ids: torch.Tensor, num_reqs: int, width: int) -> torch.Tensor:
    """Replacement for ``model_runner._coerce_output_block``: any width in ``1..max_width`` is a valid block."""
    max_width = max_output_width()
    assert (
        sampled_token_ids.shape[0] == num_reqs
    ), f"Number of request outputs {sampled_token_ids.shape[0]} != number of requests in input batch {num_reqs}"
    if num_reqs == 0 and sampled_token_ids.numel() == 0:
        sampled_token_ids = sampled_token_ids.reshape(0, max(1, min(width, max_width)))
    if sampled_token_ids.dim() != 2 or not (1 <= sampled_token_ids.shape[1] <= max_width):
        raise ValueError(
            "Model output width violates the MTP shim contract: "
            f"got shape {tuple(sampled_token_ids.shape)}, expected [num_requests, 1..{max_width}]"
        )
    return sampled_token_ids


def _delegate_original_decode(
    self, tt_out, tt_log_probs, sampling_params, model_input, batch_size_per_dp, perform_device_sampling
):
    """The stock one-token-per-row decode read-back (the model took the plain path for this step)."""
    return _STATE["originals"]["runner._get_output_tokens"](
        self, tt_out, tt_log_probs, sampling_params, model_input, batch_size_per_dp, perform_device_sampling, True
    )


def _shim_get_output_tokens(
    self,
    tt_out,
    tt_log_probs,
    sampling_params,
    model_input,
    batch_size_per_dp,
    perform_device_sampling,
    is_decode,
):
    """Replacement for ``TTModelRunner._get_output_tokens``: variable-width device-sampled decode read-back."""
    if not is_decode:
        # Prefill emits one token per row (device- or host-sampled); the original contract holds.
        return _STATE["originals"]["runner._get_output_tokens"](
            self,
            tt_out,
            tt_log_probs,
            sampling_params,
            model_input,
            batch_size_per_dp,
            perform_device_sampling,
            is_decode,
        )
    if not perform_device_sampling:
        # a host-sampled decode step (logprobs, structured outputs, min_p / bad_words / logit_bias / allowed_token_ids /
        # min_tokens): the model ran the PLAIN decode and returned logits (the adaptive MTP step is never taken for
        # host-logits steps, generator_vllm._mtp_eligible), so the original one-token read-back applies
        return _delegate_original_decode(
            self, tt_out, tt_log_probs, sampling_params, model_input, batch_size_per_dp, perform_device_sampling
        )
    max_width = max_output_width()
    sampled_token_ids_per_dp: list[torch.Tensor] = []
    logprobs_per_dp: list = []
    enable_lp = sampling_params.enable_log_probs
    start = 0
    for dp_rank, sz in enumerate(batch_size_per_dp):
        if sz <= 0:
            sampled_token_ids_per_dp.append(torch.empty((0, 1), dtype=torch.int32))
            logprobs_per_dp.append(None)
            start += self.tt_per_lane_max_num_seqs  # fixed stride segments per DP rank for decode
            continue
        grammar_bitmask = model_input.grammar_bitmask[dp_rank] if model_input.grammar_bitmask is not None else None
        if grammar_bitmask is not None or (
            isinstance(enable_lp, torch.Tensor) and bool(enable_lp[start : start + sz].any())
        ):
            # structured outputs / logprobs: such requests run the plain one-token decode (never the MTP step), so
            # the original read-back (which applies the bitmask and gathers the logprobs) handles the whole batch
            return _delegate_original_decode(
                self, tt_out, tt_log_probs, sampling_params, model_input, batch_size_per_dp, perform_device_sampling
            )
        rows = torch.as_tensor(tt_out)[start : start + sz]
        next_token_ids = rows.reshape(sz, -1).to(torch.int32)
        width = next_token_ids.shape[1]
        if not (1 <= width <= max_width):
            raise ValueError(
                f"Model output width violates the MTP shim contract: {width} not in 1..{max_width} (DP rank {dp_rank})"
            )
        _row_widths(next_token_ids.numpy())  # trailing pads, >= 1 token per row
        sampled_token_ids_per_dp.append(next_token_ids)
        logprobs_per_dp.append(None)
        start += self.tt_per_lane_max_num_seqs
    return sampled_token_ids_per_dp, logprobs_per_dp


def _shim_build_runner_output(self, sampled_token_ids, logprobs=None, req_ids=None, req_id_to_index=None):
    """Replacement for ``TTModelRunner._build_runner_output``: ragged rows, pads dropped."""
    num_reqs = len(req_ids) if req_ids is not None else self.input_batch.num_reqs
    output_req_ids = list(req_ids) if req_ids is not None else list(self.input_batch.req_ids[:num_reqs])
    output_req_id_to_index = (
        dict(req_id_to_index)
        if req_id_to_index is not None
        else {req_id: idx for idx, req_id in enumerate(output_req_ids)}
    )
    sampled_token_ids = _shim_coerce_output_block(sampled_token_ids, num_reqs, 1)
    rows = _as_int32_np(sampled_token_ids)
    widths = _row_widths(rows)
    sampled_token_id_lists = [[int(t) for t in rows[i, : int(widths[i])]] for i in range(num_reqs)]
    return ModelRunnerOutput(
        req_ids=output_req_ids,
        req_id_to_index=output_req_id_to_index,
        sampled_token_ids=sampled_token_id_lists,
        logprobs=logprobs,
        prompt_logprobs_dict=dict.fromkeys(output_req_ids, None),
        pooler_output=[],
    )


def _shim_apply_sampled_tokens_to_state(self, sampled_token_ids, req_ids=None, skip_req_ids=None):
    """Replacement for ``TTModelRunner._apply_sampled_tokens_to_state``: append ``n_i`` tokens per row, clip at
    ``max_model_len``."""
    use_captured_req_ids = req_ids is not None
    num_reqs = len(req_ids) if req_ids is not None else self.input_batch.num_reqs
    sampled_token_ids = _shim_coerce_output_block(sampled_token_ids, num_reqs, 1)
    rows = _as_int32_np(sampled_token_ids)
    widths = _row_widths(rows)
    max_model_len = int(self.model_config.max_model_len)

    def _clipped_end(start_idx: int, n: int, req_id: str) -> int:
        end_idx = start_idx + n
        if end_idx > max_model_len:
            logger.warning(
                "MTP shim: %d-token row exceeds max_model_len=%d for request %s; keeping only the slice that fits",
                n,
                max_model_len,
                req_id,
            )
            end_idx = max(start_idx, max_model_len)
        return end_idx

    if not use_captured_req_ids:
        for req_idx in range(num_reqs):
            start_idx = int(self.input_batch.num_tokens[req_idx])
            end_idx = _clipped_end(start_idx, int(widths[req_idx]), self.input_batch.req_ids[req_idx])
            block = rows[req_idx][: end_idx - start_idx]
            self.input_batch.token_ids_cpu[req_idx, start_idx:end_idx] = block
            output_token_ids = self.input_batch.req_output_token_ids[req_idx]
            assert output_token_ids is not None
            output_token_ids.extend(int(t) for t in block)
            self.input_batch.num_tokens[req_idx] = end_idx
        return

    assert req_ids is not None
    for req_idx, req_id in enumerate(req_ids):
        if skip_req_ids is not None and req_id in skip_req_ids:
            continue
        req_state = self.requests.get(req_id)
        assert (
            req_state is not None
        ), f"captured request missing from runner state while applying sampled tokens: req_id={req_id!r}"
        n = int(widths[req_idx])
        current_row = self.input_batch.req_id_to_index.get(req_id)
        if current_row is not None:
            start_idx = int(self.input_batch.num_tokens[current_row])
            end_idx = _clipped_end(start_idx, n, req_id)
            block = rows[req_idx][: end_idx - start_idx]
            self.input_batch.token_ids_cpu[current_row, start_idx:end_idx] = block
            self.input_batch.num_tokens[current_row] = end_idx
        else:
            block = rows[req_idx][:n]
        req_state.output_token_ids.extend(int(t) for t in block)


def _shim_prepare_model_inputs(self, scheduler_output, grammar_output):
    """Wrapper for ``TTModelRunner._prepare_model_inputs``: widen the nested ``_is_still_prefilling`` threshold.

    The closure compares ``num_computed + self._output_tokens_per_step < num_tokens``; that attribute is read nowhere
    else inside ``_prepare_model_inputs`` (plugin model_runner.py:1181 is the only read), so it is set to the shim's
    maximum width for the duration of the call.  See the module docstring for why the bound holds.
    """
    saved = self._output_tokens_per_step
    self._output_tokens_per_step = max_output_width()
    try:
        return _STATE["originals"]["runner._prepare_model_inputs"](self, scheduler_output, grammar_output)
    finally:
        self._output_tokens_per_step = saved


# ----------------------------------------------------------------------------------------------------------------------
# scheduler patches


def _shim_update_request_with_output(self, request, new_token_ids):
    """Replacement for ``TTScheduler._update_request_with_output`` (width-1 configs only)."""
    if self._is_block_output_model:
        return _STATE["originals"]["scheduler._update_request_with_output"](self, request, new_token_ids)
    if request.async_tokens_to_discard > 0:
        # Force-preempted in reset_prefix_cache: drop one stale in-flight frame per call (AsyncScheduler semantics).
        request.async_tokens_to_discard -= 1
        return [], False
    max_width = max_output_width()
    if len(new_token_ids) > max_width:
        raise ValueError(
            f"Model output width violates the MTP shim contract: {len(new_token_ids)} tokens > {max_width} for "
            f"request {request.request_id}"
        )
    # One placeholder is reserved per scheduled frame (num_sampled_tokens_per_step == 1); an empty frame (a mid-prompt
    # chunk, a suppressed row) reserved none. Consume before trimming so an EOS-trimmed frame still frees its slot.
    consumed = 1 if new_token_ids else 0
    status_before_update = request.status
    new_token_ids, stopped = Scheduler._update_request_with_output(self, request, new_token_ids)
    request.num_output_placeholders = max(request.num_output_placeholders - consumed, 0)
    if status_before_update == RequestStatus.RUNNING:
        self.kv_cache_manager.cache_blocks(request, request.num_computed_tokens - request.num_output_placeholders)
    return new_token_ids, stopped


def _shim_scheduler_init(self, *args, **kwargs):
    """Wrapper for ``TTScheduler.__init__``: reserve K lookahead slots so ``allocate_slots`` covers the verify window."""
    _STATE["originals"]["scheduler.__init__"](self, *args, **kwargs)
    if not self._is_block_output_model:
        lookahead = max_output_width() - 1
        self.num_lookahead_tokens = max(int(self.num_lookahead_tokens), lookahead)
        logger.info("MTP shim: TTScheduler.num_lookahead_tokens = %d", self.num_lookahead_tokens)


# ----------------------------------------------------------------------------------------------------------------------
# apply / undo


def apply(max_output_tokens_per_step: int) -> Callable[[], None]:
    """Patch the plugin to accept ``1..max_output_tokens_per_step`` tokens per request per decode step.

    ``max_output_tokens_per_step`` is the widest row the model returns (K + 1 for K draft tokens). Idempotent for the
    same width; applying a different width while active raises. Returns ``undo``.
    """
    global _STATE
    width = int(max_output_tokens_per_step)
    if width < 1:
        raise ValueError(f"max_output_tokens_per_step must be >= 1, got {max_output_tokens_per_step!r}")
    if _STATE is not None:
        if _STATE["max_width"] != width:
            raise RuntimeError(
                f"mtp_plugin_shim already applied with max width {_STATE['max_width']}; undo() before re-applying "
                f"with {width}"
            )
        return _STATE["undo"]

    runner_cls = _mr.TTModelRunner
    sched_cls = _sched.TTScheduler
    originals: dict[str, object] = {"model_runner._coerce_output_block": _mr._coerce_output_block}
    for name in _RUNNER_ATTRS:
        originals[f"runner.{name}"] = runner_cls.__dict__[name]
    for name in _SCHEDULER_ATTRS:
        originals[f"scheduler.{name}"] = sched_cls.__dict__[name]

    def undo() -> None:
        global _STATE
        if _STATE is None or _STATE["undo"] is not undo:
            return
        _mr._coerce_output_block = originals["model_runner._coerce_output_block"]
        for name in _RUNNER_ATTRS:
            setattr(runner_cls, name, originals[f"runner.{name}"])
        for name in _SCHEDULER_ATTRS:
            setattr(sched_cls, name, originals[f"scheduler.{name}"])
        _STATE = None
        logger.info("MTP shim: plugin patches removed")

    _STATE = {"max_width": width, "originals": originals, "undo": undo}
    _mr._coerce_output_block = _shim_coerce_output_block
    runner_cls._get_output_tokens = _shim_get_output_tokens
    runner_cls._build_runner_output = _shim_build_runner_output
    runner_cls._apply_sampled_tokens_to_state = _shim_apply_sampled_tokens_to_state
    runner_cls._prepare_model_inputs = _shim_prepare_model_inputs
    sched_cls._update_request_with_output = _shim_update_request_with_output
    sched_cls.__init__ = _shim_scheduler_init
    logger.info("MTP shim: plugin patched for 1..%d tokens per request per decode step", width)
    return undo
