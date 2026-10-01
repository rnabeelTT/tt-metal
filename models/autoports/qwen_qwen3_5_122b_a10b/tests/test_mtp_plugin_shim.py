# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only tests of ``tt/mtp_plugin_shim.py`` against the installed vLLM Tenstorrent plugin (no device, no mesh).

Fixtures mirror the plugin's own host tests (``tests/test_block_model_runner.py``, ``tests/test_model_runner.py``,
``tests/test_block_scheduler.py``): a ``TTScheduler`` built from the plugin's local qwen2 model config with the model
resolution stubbed, and a ``SimpleNamespace`` runner around a real ``InputBatch``.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

pytest.importorskip("vllm")
vllm_tt_plugin = pytest.importorskip("vllm_tt_plugin")

from vllm.config import CacheConfig, DeviceConfig, ModelConfig, ParallelConfig, SchedulerConfig, VllmConfig
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.structured_output import StructuredOutputManager
from vllm.v1.worker.gpu_input_batch import CachedRequestState
from vllm_tt_plugin import model_runner as plugin_model_runner
from vllm_tt_plugin import scheduler as plugin_scheduler
from vllm_tt_plugin.config import store_tt_output_tokens_per_step
from vllm_tt_plugin.input_batch import InputBatch
from vllm_tt_plugin.model_runner import TTModelRunner
from vllm_tt_plugin.scheduler import TTScheduler

from models.autoports.qwen_qwen3_5_122b_a10b.tt import mtp_plugin_shim as shim

K = 3  # draft tokens per pass
MAX_WIDTH = K + 1
EOS = 2
PLUGIN_ROOT = Path(vllm_tt_plugin.__file__).resolve().parents[2]
LOCAL_MODEL_CONFIG = PLUGIN_ROOT / "tests" / "model_configs" / "qwen2"

# region fixtures

_TT_PLATFORM_CONFIG_ATTRS = (
    "_standard_dp_visible_device_groups",
    "_standard_dp_mesh_grids",
    "sample_on_device_mode",
    "always_compat_sampling",
    "_tt_vllm_config",
)


@pytest.fixture(autouse=True)
def _reset_tt_platform_class_state():
    """The platform hook records results on the class; restore them (mirrors the plugin's conftest)."""
    import vllm.v1.engine.async_llm as async_llm
    import vllm.v1.engine.core as engine_core
    import vllm.v1.engine.input_processor as input_processor
    from vllm_tt_plugin.platform import TTPlatform

    unset = object()
    saved = {n: TTPlatform.__dict__.get(n, unset) for n in _TT_PLATFORM_CONFIG_ATTRS}
    saved_mods = {
        (input_processor.InputProcessor, "process_inputs"): input_processor.InputProcessor.process_inputs,
        (async_llm.AsyncLLM, "_add_streaming_input_request"): async_llm.AsyncLLM._add_streaming_input_request,
        (engine_core.EngineCore, "reset_prefix_cache"): engine_core.EngineCore.reset_prefix_cache,
        (engine_core.EngineCore, "pause_scheduler"): engine_core.EngineCore.pause_scheduler,
        (engine_core.EngineCoreProc, "pause_scheduler"): engine_core.EngineCoreProc.pause_scheduler,
    }
    saved_markers = {
        (input_processor, "_tt_original_process_inputs"),
        (async_llm, "_tt_original_add_streaming_input_request"),
        (engine_core, "_tt_original_reset_prefix_cache"),
        (engine_core, "_tt_original_pause_scheduler"),
    }
    marker_values = {(m, a): m.__dict__.get(a, unset) for m, a in saved_markers}
    yield
    for (owner, attr), value in saved_mods.items():
        setattr(owner, attr, value)
    for (mod, attr), value in marker_values.items():
        if value is unset:
            mod.__dict__.pop(attr, None)
        else:
            setattr(mod, attr, value)
    for name, value in saved.items():
        if value is unset:
            if name in TTPlatform.__dict__:
                delattr(TTPlatform, name)
        else:
            setattr(TTPlatform, name, value)


@pytest.fixture
def applied():
    undo = shim.apply(MAX_WIDTH)
    yield undo
    undo()


class _StubModel:
    """No model_capabilities: the platform hook resolves output_tokens_per_step=1."""


@contextmanager
def _stub_model_resolution():
    with (
        patch("vllm_tt_plugin.platform.TTPlatform._tt_vllm_config", None),
        patch("vllm_tt_plugin.platform.register_tt_models"),
        patch("vllm_tt_plugin.platform._resolve_standard_dp_visible_device_groups", return_value=None),
        patch(
            "vllm.model_executor.models.registry.ModelRegistry.get_supported_archs",
            return_value=["TTQwen2ForCausalLM"],
        ),
        patch("vllm.model_executor.model_loader.utils.get_model_architecture", return_value=(_StubModel, None)),
    ):
        yield


def _scheduler(*, block_size: int = 16, max_model_len: int = 256) -> TTScheduler:
    if not LOCAL_MODEL_CONFIG.exists():
        pytest.skip(f"plugin test model config missing: {LOCAL_MODEL_CONFIG}")
    model_config = ModelConfig(model=str(LOCAL_MODEL_CONFIG), dtype="float16", seed=42, skip_tokenizer_init=True)
    model_config.max_model_len = max_model_len
    scheduler_config = SchedulerConfig(
        max_num_seqs=1,
        max_num_batched_tokens=max_model_len,
        max_model_len=max_model_len,
        enable_chunked_prefill=False,
        async_scheduling=False,
        is_encoder_decoder=model_config.is_encoder_decoder,
    )
    cache_config = CacheConfig(
        block_size=block_size, gpu_memory_utilization=0.9, cache_dtype="auto", enable_prefix_caching=False
    )
    with _stub_model_resolution():
        config = VllmConfig(
            scheduler_config=scheduler_config,
            model_config=model_config,
            cache_config=cache_config,
            parallel_config=ParallelConfig(),
            device_config=DeviceConfig(device="cpu"),
        )
    store_tt_output_tokens_per_step(config, 1)
    num_blocks = max_model_len // block_size + 2
    cache_config.num_gpu_blocks = num_blocks
    kv_cache_config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer"],
                FullAttentionSpec(block_size=block_size, num_kv_heads=1, head_size=1, dtype=torch.float32),
            )
        ],
    )
    return TTScheduler(
        vllm_config=config,
        kv_cache_config=kv_cache_config,
        block_size=block_size,
        log_stats=True,
        structured_output_manager=StructuredOutputManager(config),
    )


def _request(max_tokens: int, *, prompt_len: int = 30, ignore_eos: bool = False) -> Request:
    init_none_hash(sha256)
    sampling_params = SamplingParams(max_tokens=max_tokens, ignore_eos=ignore_eos)
    sampling_params.update_from_generation_config({}, eos_token_id=EOS)
    return Request(
        request_id="req-0",
        prompt_token_ids=[1] * prompt_len,
        sampling_params=sampling_params,
        pooling_params=None,
        block_hasher=get_request_block_hasher(16, sha256),
    )


def _runner_output(scheduler_output: SchedulerOutput, tokens: list[int]) -> ModelRunnerOutput:
    req_id = next(iter(scheduler_output.num_scheduled_tokens))
    return ModelRunnerOutput(
        req_ids=[req_id],
        req_id_to_index={req_id: 0},
        sampled_token_ids=[tokens],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )


VOCAB_SIZE = 64
BLOCK_SIZE = 16
MAX_MODEL_LEN = 32
MAX_NUM_SEQS = 4


def _batch_with_one_request(prompt_len: int, output_len: int, num_computed_tokens: int):
    batch = InputBatch(
        max_num_reqs=MAX_NUM_SEQS,
        max_model_len=MAX_MODEL_LEN,
        max_num_batched_tokens=MAX_MODEL_LEN,
        vocab_size=VOCAB_SIZE,
        block_sizes=[BLOCK_SIZE],
        kernel_block_sizes=[BLOCK_SIZE],
    )
    request = CachedRequestState(
        req_id="r",
        prompt_token_ids=list(range(prompt_len)),
        mm_features=None,
        sampling_params=SamplingParams(temperature=0.0),
        generator=None,
        block_ids=([0],),
        num_computed_tokens=num_computed_tokens,
        output_token_ids=list(range(prompt_len, prompt_len + output_len)),
    )
    batch.add_request(request)
    return batch, request


def _fake_runner(batch: InputBatch, request: CachedRequestState) -> SimpleNamespace:
    runner = SimpleNamespace(
        input_batch=batch,
        requests={"r": request},
        _output_tokens_per_step=1,
        _is_block_output_model=False,
        tt_per_lane_max_num_seqs=MAX_NUM_SEQS,
        tt_data_parallel_size=1,
        max_num_blocks_per_req=MAX_MODEL_LEN // BLOCK_SIZE,
        model_config=SimpleNamespace(is_multimodal_model=False, max_model_len=MAX_MODEL_LEN),
        check_perform_device_sampling=lambda **_: False,
        _block_tables_per_layer=lambda _: None,
        _alloc_prefill_state_slots=lambda row_req_ids: list(range(len(row_req_ids))),
        _decode_state_slot_remap=lambda row_req_ids: None,
        _sampling_params_for_padded_decode=lambda params, req_indices, n: params,
        _decode_layout_changed_since_last_decode=False,
        _build_host_generators=TTModelRunner._build_host_generators,
    )
    # Bound through the class so the patched (or restored) attribute is resolved at call time.
    runner._apply_sampled_tokens_to_state = lambda **kw: TTModelRunner._apply_sampled_tokens_to_state(runner, **kw)
    runner._build_runner_output = lambda **kw: TTModelRunner._build_runner_output(runner, **kw)
    return runner


def _prepare(runner, *rows):
    """Run _prepare_model_inputs for cached rows of (req_id, num_scheduled, num_computed, num_output)."""
    out = SchedulerOutput.make_empty()
    out.num_scheduled_tokens = {r: s for r, s, _, _ in rows}
    out.total_num_scheduled_tokens = sum(s for _, s, _, _ in rows)
    out.scheduled_cached_reqs = CachedRequestData(
        req_ids=[r for r, *_ in rows],
        resumed_req_ids=set(),
        new_token_ids=[[] for _ in rows],
        all_token_ids={},
        new_block_ids=[None for _ in rows],
        num_computed_tokens=[c for _, _, c, _ in rows],
        num_output_tokens=[o for *_, o in rows],
    )
    return TTModelRunner._prepare_model_inputs(runner, out, None)


def _apply_and_build(runner, block: torch.Tensor) -> ModelRunnerOutput:
    return TTModelRunner.apply_and_build_runner_output(runner, sampled_token_ids=block)


# endregion fixtures

# region runner: variable-width apply / build


def test_partial_row_appends_only_real_tokens(applied):
    """(a) ``[[t0, t1, -1, -1]]`` for K=3 appends 2 tokens, advances num_tokens by 2, emits ``[[t0, t1]]``."""
    prompt_len = 8
    batch, request = _batch_with_one_request(prompt_len, 0, prompt_len)
    runner = _fake_runner(batch, request)

    output = _apply_and_build(runner, torch.tensor([[11, 12, -1, -1]], dtype=torch.int32))

    assert batch.num_tokens[0] == prompt_len + 2
    assert batch.token_ids_cpu[0, prompt_len : prompt_len + 2].tolist() == [11, 12]
    assert batch.token_ids_cpu[0, prompt_len + 2 : prompt_len + 4].tolist() == [0, 0]
    assert request.output_token_ids == [11, 12]
    assert output.sampled_token_ids == [[11, 12]]


def test_full_width_row_appends_all(applied):
    """(b) a full-width row ``[[a, b, c, d]]`` appends 4."""
    prompt_len = 8
    batch, request = _batch_with_one_request(prompt_len, 0, prompt_len)
    runner = _fake_runner(batch, request)

    output = _apply_and_build(runner, torch.tensor([[21, 22, 23, 24]], dtype=torch.int32))

    assert batch.num_tokens[0] == prompt_len + 4
    assert request.output_token_ids == [21, 22, 23, 24]
    assert output.sampled_token_ids == [[21, 22, 23, 24]]


def test_width_one_row_still_works(applied):
    prompt_len = 8
    batch, request = _batch_with_one_request(prompt_len, 0, prompt_len)
    runner = _fake_runner(batch, request)

    output = _apply_and_build(runner, torch.tensor([[42]], dtype=torch.int32))

    assert batch.num_tokens[0] == prompt_len + 1
    assert output.sampled_token_ids == [[42]]


def test_row_wider_than_max_is_rejected(expect_error, applied):
    batch, request = _batch_with_one_request(8, 0, 8)
    runner = _fake_runner(batch, request)
    with expect_error(ValueError, "MTP shim contract"):
        _apply_and_build(runner, torch.zeros((1, MAX_WIDTH + 1), dtype=torch.int32))


def test_non_trailing_pad_and_empty_row_are_rejected(expect_error, applied):
    batch, request = _batch_with_one_request(8, 0, 8)
    runner = _fake_runner(batch, request)
    with expect_error(ValueError, "non-trailing"):
        _apply_and_build(runner, torch.tensor([[5, -1, 6, -1]], dtype=torch.int32))
    with expect_error(ValueError, "emitted no token"):
        _apply_and_build(runner, torch.tensor([[-1, -1, -1, -1]], dtype=torch.int32))


def test_row_crossing_max_model_len_is_clipped_not_raised(applied):
    prompt_len = MAX_MODEL_LEN - 2
    batch, request = _batch_with_one_request(prompt_len, 0, prompt_len)
    runner = _fake_runner(batch, request)

    output = _apply_and_build(runner, torch.tensor([[31, 32, 33, 34]], dtype=torch.int32))

    assert batch.num_tokens[0] == MAX_MODEL_LEN
    assert request.output_token_ids == [31, 32]
    # The engine still sees the whole row; the scheduler's check_stop trims it at max_model_len.
    assert output.sampled_token_ids == [[31, 32, 33, 34]]


def test_captured_req_ids_path_appends_per_row_width(applied):
    """The deferred (async) apply path resolves the live row from req_id_to_index and uses each row's own width."""
    outputs_a: list[int] = []
    outputs_b: list[int] = []
    runner = SimpleNamespace(
        requests={"a": SimpleNamespace(output_token_ids=outputs_a), "b": SimpleNamespace(output_token_ids=outputs_b)},
        input_batch=SimpleNamespace(
            req_id_to_index={"a": 0, "b": 1},
            num_tokens=np.array([4, 30], dtype=np.int32),
            token_ids_cpu=np.zeros((2, MAX_MODEL_LEN), dtype=np.int32),
        ),
        model_config=SimpleNamespace(max_model_len=MAX_MODEL_LEN),
    )
    block = torch.tensor([[1, 2, 3, -1], [7, 8, 9, 10]], dtype=torch.int32)

    TTModelRunner._apply_sampled_tokens_to_state(runner, block, req_ids=["a", "b"])

    assert runner.input_batch.num_tokens.tolist() == [7, MAX_MODEL_LEN]
    assert runner.input_batch.token_ids_cpu[0, 4:7].tolist() == [1, 2, 3]
    assert runner.input_batch.token_ids_cpu[1, 30:32].tolist() == [7, 8]
    assert outputs_a == [1, 2, 3]
    assert outputs_b == [7, 8]  # clipped at max_model_len


# endregion

# region runner: device-sampled read-back and prefill classification


def _device_readback_args(tt_out: torch.Tensor, *, enable_log_probs=None, grammar=None):
    n = tt_out.shape[0]
    sampling_params = SimpleNamespace(
        enable_log_probs=enable_log_probs if enable_log_probs is not None else torch.zeros(n, dtype=torch.bool)
    )
    model_input = SimpleNamespace(grammar_bitmask=[grammar], max_num_logprobs=[None], intermediate_prefill_mask=None)
    return dict(
        tt_out=tt_out,
        tt_log_probs=None,
        sampling_params=sampling_params,
        model_input=model_input,
        batch_size_per_dp=[1],
        perform_device_sampling=True,
        is_decode=True,
    )


def test_get_output_tokens_keeps_padded_width(applied):
    runner = SimpleNamespace(_output_tokens_per_step=1, _is_block_output_model=False, tt_per_lane_max_num_seqs=4)
    tt_out = torch.full((4, MAX_WIDTH), -1, dtype=torch.int32)
    tt_out[0, :2] = torch.tensor([11, 12])

    sampled, logprobs = TTModelRunner._get_output_tokens(runner, **_device_readback_args(tt_out))

    assert sampled[0].tolist() == [[11, 12, -1, -1]]
    assert logprobs == [None]


def test_get_output_tokens_contract_and_delegations(expect_error, applied):
    runner = SimpleNamespace(_output_tokens_per_step=1, _is_block_output_model=False, tt_per_lane_max_num_seqs=4)
    tt_out = torch.full((4, MAX_WIDTH), -1, dtype=torch.int32)
    tt_out[:, 0] = 5
    with expect_error(ValueError, "MTP shim contract"):
        TTModelRunner._get_output_tokens(
            runner, **_device_readback_args(torch.zeros((4, MAX_WIDTH + 1), dtype=torch.int32))
        )
    # logprobs / structured outputs / host sampling on a decode step: the model ran the plain one-token decode for
    # such requests (adaptive MTP never takes the step for them), so the shim hands the read-back to the original
    calls = []
    saved = shim._STATE["originals"]["runner._get_output_tokens"]
    shim._STATE["originals"]["runner._get_output_tokens"] = lambda self_, *a: calls.append(a) or ("orig", None)
    try:
        for kw in (
            _device_readback_args(tt_out, enable_log_probs=torch.tensor([True, False, False, False])),
            _device_readback_args(tt_out, grammar=torch.zeros(1, 2)),
            dict(_device_readback_args(tt_out), perform_device_sampling=False),
        ):
            assert TTModelRunner._get_output_tokens(runner, **kw) == ("orig", None)
    finally:
        shim._STATE["originals"]["runner._get_output_tokens"] = saved
    assert len(calls) == 3 and all(a[-1] is True for a in calls)  # is_decode=True on every delegation


def test_multi_token_gap_is_a_decode_step(applied):
    """After a 3-token step the scheduler schedules those 3 tokens; the row must still build a decode input."""
    prompt_len, emitted = 8, 3
    batch, request = _batch_with_one_request(prompt_len, emitted, prompt_len)
    model_input = _prepare(_fake_runner(batch, request), ("r", emitted, prompt_len, emitted))

    assert model_input.prompt_lens is None
    assert model_input.input_positions.tolist()[0] == prompt_len + emitted - 1
    assert model_input.input_tokens[0, 0] == batch.token_ids_cpu[0, prompt_len + emitted - 1]


def test_multi_token_gap_without_shim_is_misclassified_as_prefill():
    """Documents why the ``_is_still_prefilling`` wrapper is needed."""
    prompt_len, emitted = 8, 3
    batch, request = _batch_with_one_request(prompt_len, emitted, prompt_len)
    model_input = _prepare(_fake_runner(batch, request), ("r", emitted, prompt_len, emitted))
    assert model_input.prompt_lens is not None


def test_gap_wider_than_max_width_stays_prefill_with_chunked_prefill(applied):
    """With chunked prefill ON, more than one step's width outstanding past the prompt is uncomputed history."""
    batch, request = _batch_with_one_request(4, 6, 5)  # num_tokens 10, computed 5 -> gap 5 > MAX_WIDTH
    runner = _fake_runner(batch, request)
    runner.scheduler_config = SimpleNamespace(chunked_prefill_enabled=True)
    model_input = _prepare(runner, ("r", 5, 5, 6))
    assert model_input.prompt_lens is not None
    assert model_input.prompt_lens.tolist() == [10]


def test_wide_gap_is_a_decode_row_without_chunked_prefill(applied):
    """Chunked prefill OFF: a cached, non-resumed row past its prompt is a decode row whatever the outstanding gap
    (async scheduling leaves up to 2k + 1 emitted tokens outstanding after two in-flight multi-token steps)."""
    prompt_len, emitted = 4, 7  # gap 7 > MAX_WIDTH, computed past the prompt
    batch, request = _batch_with_one_request(prompt_len, emitted, prompt_len)
    model_input = _prepare(_fake_runner(batch, request), ("r", emitted, prompt_len, emitted))
    assert model_input.prompt_lens is None
    assert model_input.input_positions.tolist()[0] == prompt_len + emitted - 1


def test_uncomputed_prompt_tokens_are_still_prefill(applied):
    """The closure's first test is untouched: a resumed request (num_computed reset below its prompt) is prompt work."""
    batch, request = _batch_with_one_request(8, 2, 3)  # computed 3 < prompt 8
    model_input = _prepare(_fake_runner(batch, request), ("r", 7, 3, 2))
    assert model_input.prompt_lens is not None


def test_prepare_wrapper_restores_output_width(applied):
    batch, request = _batch_with_one_request(8, 0, 8)
    runner = _fake_runner(batch, request)
    _prepare(runner, ("r", 1, 8, 0))
    assert runner._output_tokens_per_step == 1


# endregion

# region scheduler


def test_scheduler_reserves_lookahead_for_the_verify_window(applied):
    scheduler = _scheduler(block_size=16)
    assert scheduler.num_lookahead_tokens == K
    request = _request(max_tokens=64, prompt_len=30)
    scheduler.add_request(request)
    scheduler.schedule()
    # 30 prompt tokens + K lookahead = 33 slots -> 3 blocks of 16 (2 without lookahead).
    assert len(scheduler.kv_cache_manager.get_block_ids(request.request_id)[0]) == 3


def test_eos_inside_a_row_trims_and_finishes(applied):
    """(c) EOS inside a row trims at the EOS (upstream check_stop) and the request finishes."""
    scheduler = _scheduler()
    request = _request(max_tokens=64)
    scheduler.add_request(request)
    prefill = scheduler.schedule()

    outputs = scheduler.update_from_output(prefill, _runner_output(prefill, [10, EOS, 11, 12]))

    assert outputs[0].outputs[0].new_token_ids == [10, EOS]
    assert list(request.output_token_ids) == [10, EOS]
    assert request.status == RequestStatus.FINISHED_STOPPED
    assert request.num_output_placeholders == 0
    assert not scheduler.running


def test_max_tokens_mid_row_trims(applied):
    """(d) max_tokens reached mid-row trims the row."""
    scheduler = _scheduler()
    request = _request(max_tokens=3, ignore_eos=True)
    scheduler.add_request(request)
    prefill = scheduler.schedule()

    outputs = scheduler.update_from_output(prefill, _runner_output(prefill, [10, 11, 12, 13]))

    assert outputs[0].outputs[0].new_token_ids == [10, 11, 12]
    assert list(request.output_token_ids) == [10, 11, 12]
    assert request.status == RequestStatus.FINISHED_LENGTH_CAPPED


def test_placeholder_bookkeeping_across_widths_1_3_2(applied):
    """(e) placeholders stay >= 0 across steps of widths 1, 3, 2 and the scheduler re-schedules exactly the
    emitted tokens (whose K/V the model already wrote)."""
    scheduler = _scheduler()
    request = _request(max_tokens=64, prompt_len=30)
    scheduler.add_request(request)

    step = scheduler.schedule()  # prefill
    assert step.num_scheduled_tokens == {"req-0": 30}
    assert request.num_output_placeholders == 1
    scheduler.update_from_output(step, _runner_output(step, [10]))
    assert request.num_output_placeholders == 0
    assert request.num_tokens == 31 and request.num_computed_tokens == 30

    step = scheduler.schedule()  # decode: the one emitted token
    assert step.num_scheduled_tokens == {"req-0": 1}
    assert request.num_output_placeholders == 1
    scheduler.update_from_output(step, _runner_output(step, [11, 12, 13]))
    assert request.num_output_placeholders == 0
    assert request.num_tokens == 34 and request.num_computed_tokens == 31

    step = scheduler.schedule()  # decode: the three emitted tokens
    assert step.num_scheduled_tokens == {"req-0": 3}
    assert request.num_output_placeholders == 1
    scheduler.update_from_output(step, _runner_output(step, [14, 15]))
    assert request.num_output_placeholders == 0
    assert request.num_tokens == 36 and request.num_computed_tokens == 34

    step = scheduler.schedule()
    assert step.num_scheduled_tokens == {"req-0": 2}
    assert list(request.output_token_ids) == [10, 11, 12, 13, 14, 15]
    assert request.status == RequestStatus.RUNNING


def test_scheduler_rejects_row_wider_than_max(expect_error, applied):
    scheduler = _scheduler()
    request = _request(max_tokens=64)
    scheduler.add_request(request)
    prefill = scheduler.schedule()
    with expect_error(ValueError, "MTP shim contract"):
        scheduler.update_from_output(prefill, _runner_output(prefill, [1, 2, 3, 4, 5]))


def test_width_one_without_shim_still_delegates_to_async_scheduler():
    """No import-time side effect: with the shim not applied the plugin's own width-1 path is untouched."""
    assert not shim.is_active()
    scheduler = _scheduler()
    assert scheduler.num_lookahead_tokens == 0
    request = _request(max_tokens=64)
    scheduler.add_request(request)
    prefill = scheduler.schedule()
    outputs = scheduler.update_from_output(prefill, _runner_output(prefill, [7]))
    assert outputs[0].outputs[0].new_token_ids == [7]
    assert request.num_output_placeholders == 0


# endregion

# region apply / undo


def test_undo_restores_original_functions(expect_error):
    """(f) ``undo()`` restores the original functions; ``apply`` is idempotent."""
    originals = {
        name: TTModelRunner.__dict__[name]
        for name in (
            "_get_output_tokens",
            "_build_runner_output",
            "_apply_sampled_tokens_to_state",
            "_prepare_model_inputs",
        )
    }
    sched_originals = {name: TTScheduler.__dict__[name] for name in ("_update_request_with_output", "__init__")}
    coerce_original = plugin_model_runner._coerce_output_block

    undo = shim.apply(MAX_WIDTH)
    try:
        assert shim.is_active() and shim.max_output_width() == MAX_WIDTH
        assert shim.apply(MAX_WIDTH) is undo  # idempotent
        with expect_error(RuntimeError, "already applied"):
            shim.apply(MAX_WIDTH + 1)
        for name, fn in originals.items():
            assert TTModelRunner.__dict__[name] is not fn
        for name, fn in sched_originals.items():
            assert TTScheduler.__dict__[name] is not fn
        assert plugin_model_runner._coerce_output_block is not coerce_original
    finally:
        undo()
    undo()  # second undo is a no-op

    assert not shim.is_active()
    for name, fn in originals.items():
        assert TTModelRunner.__dict__[name] is fn
    for name, fn in sched_originals.items():
        assert TTScheduler.__dict__[name] is fn
    assert plugin_model_runner._coerce_output_block is coerce_original
    assert plugin_scheduler.TTScheduler.__init__ is sched_originals["__init__"]

    # The original strict width contract is back.
    batch, request = _batch_with_one_request(8, 0, 8)
    runner = _fake_runner(batch, request)
    with expect_error(ValueError, "violates output_tokens_per_step"):
        _apply_and_build(runner, torch.tensor([[1, 2, -1, -1]], dtype=torch.int32))


# endregion


@pytest.mark.parametrize(
    "env,expected",
    [({}, "3"), ({"QWEN35_CHUNKED_PREFILL": "1"}, "0"), ({"QWEN35_MTP": "0"}, "0"), ({"QWEN35_MTP": "2"}, "2")],
    ids=["default_on", "chunked_turns_it_off", "explicit_off", "explicit_k2"],
)
def test_adapter_import_pins_mtp_env(env, expected):
    """The adapter decides k (default 3) and pins QWEN35_MTP for the model's bucket-input allocation: without the pin
    a server started with no env crashed at the MTP warm-up (KeyError 'mtp_sel', 2026-10-01)."""
    import os
    import subprocess
    import sys

    clean = {k: v for k, v in os.environ.items() if not k.startswith("QWEN35_MTP") and k != "QWEN35_CHUNKED_PREFILL"}
    clean.update(env)
    code = (
        "import os, models.autoports.qwen_qwen3_5_122b_a10b.tt.generator_vllm as g; "
        "print('PIN', os.environ['QWEN35_MTP'], g._MTP_K)"
    )
    out = subprocess.run([sys.executable, "-c", code], env=clean, capture_output=True, text=True, timeout=600)
    line = [l for l in out.stdout.splitlines() if l.startswith("PIN")]
    assert line, out.stdout[-500:] + out.stderr[-800:]
    assert line[0] == f"PIN {expected} {expected}", line[0]
