# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Loaded by every Python process started with this directory on PYTHONPATH (the vLLM API server and its EngineCore
subprocess). Registers the Hy4-flash config (`hy4_config.register`) the moment transformers' auto-config module is
imported, so `AutoConfig` / vLLM's `get_config` resolve `model_type: hy_v4_flash` in every process (the EngineCore
unpickles the config object too). Nothing is imported eagerly: processes that never touch transformers pay nothing.

It also registers the `hy_v4` reasoning and tool parsers (this directory's `hy_v4_reasoning_parser.py`,
`hy_v4_tool_parser.py`, ported from vLLM PR 68296) as lazy vLLM parsers when `vllm.reasoning` / `vllm.tool_parsers`
are imported, so `--reasoning-parser hy_v4 --tool-call-parser hy_v4` work in every process without plugin flags.

It also lets the TT plugin's model runner pass video inputs to the model (`_patch_tt_runner`: the plugin gathers images
only).

Chunked prefill (HY4_CHUNKED_PREFILL=1, default; tt/generator_vllm.py): the runner tells the model which prefill rows are
intermediate parts of their prompts (`_patch_tt_runner`); the scheduler runs one decode step of the running requests
after every prefill step while a prompt is in parts (`_patch_tt_scheduler`; the plugin's default runs every part back
to back); the budget (`_patch_tt_platform`): every prompt part at most HY4_CP_TOKENS (default 2048) tokens
(long_prefill_token_threshold), the step total unbounded (max_num_batched_tokens = max_model_len, as with chunked
prefill off: simultaneous short prompts still pack into one step).

With HYB_ENABLE=1 it also installs the hybrid-MoE overlay, as models/experimental/hybrid_moe/serve/sitecustomize.py
does (only one sitecustomize module is ever loaded)."""

import importlib.abc
import importlib.util
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))


def _here_on_path() -> None:
    if _HERE not in sys.path:
        sys.path.append(_HERE)


def _register_config(_module) -> None:
    _here_on_path()
    try:
        import hy4_config

        hy4_config.register()
    except Exception as e:  # noqa: BLE001 - surface loudly: without it vLLM cannot read the checkpoint's config
        sys.stderr.write(f"[hy4] config registration FAILED in pid {os.getpid()}: {e!r}\n")
        raise


def _register_arch(module) -> None:
    # vLLM 0.26 validates the checkpoint's architecture (ModelConfig) before the TT plugin maps it to its TT-prefixed
    # registration; it does not know HYV4FlashForConditionalGeneration, so register the plain name too (lazy: the same
    # adapter class the plugin registers from EXTRA_MODELS_DIR)
    try:
        if "HYV4FlashForConditionalGeneration" not in module.ModelRegistry.get_supported_archs():
            module.ModelRegistry.register_model(
                "HYV4FlashForConditionalGeneration",
                "hy4_vllm_stub:HYV4FlashForConditionalGeneration",
            )
    except Exception as e:  # noqa: BLE001
        print(f"hy4 sitecustomize: architecture registration failed: {e!r}", file=sys.stderr)


def _register_reasoning_parser(module) -> None:
    # lazy: the parser module (and its vLLM imports) loads only when "hy_v4" is asked for
    _here_on_path()
    module.ReasoningParserManager.register_lazy_module("hy_v4", "hy_v4_reasoning_parser", "HYV4ReasoningParser")


def _register_tool_parser(module) -> None:
    _here_on_path()
    module.ToolParserManager.register_lazy_module("hy_v4", "hy_v4_tool_parser", "HYV4ToolParser")


def _patch_tt_runner(module) -> None:
    # Video inputs: the TT plugin's TTModelRunner._gather_multi_modal_inputs passes images only (pixel_values /
    # image_grid_thw; any other modality raises "Only images are supported for now"). For a batch holding a video it
    # also gathers pixel_values_videos / video_grid_thw (per request a list per video, None for a text request); every
    # other batch goes through the plugin's own code, unchanged.
    R = getattr(module, "TTModelRunner", None)
    if R is None or getattr(R, "_hy4_video", False):
        return
    orig = R._gather_multi_modal_inputs

    def gather(self, req_indices=None):
        if req_indices is None:
            req_indices = list(range(self.input_batch.num_reqs))
        reqs = [self.requests[self.input_batch.req_ids[i]] for i in req_indices]
        if not any(f.modality == "video" for r in reqs for f in (r.mm_features or ())):
            return orig(self, req_indices)
        keys = {"image": ("pixel_values", "image_grid_thw"), "video": ("pixel_values_videos", "video_grid_thw")}
        out = {k: [] for ks in keys.values() for k in ks}
        for r in reqs:
            if not r.mm_features:
                for v in out.values():
                    v.append(None)
                continue
            rows = {k: [] for k in out}
            for f in r.mm_features:
                if f.modality not in keys:
                    raise NotImplementedError(f"Hy4: {f.modality} inputs are not supported")
                pk, gk = keys[f.modality]
                item = f.data
                rows[pk].append(None if item is None else item[pk].data)
                rows[gk].append(item[gk].data if item is not None and gk in item else None)
            for k, v in rows.items():
                out[k].append(v)
        return out

    R._gather_multi_modal_inputs = gather
    R._hy4_video = True
    if os.environ.get("HY4_CHUNKED_PREFILL", "1") == "1" and hasattr(R, "submit_prefill"):
        submit = R.submit_prefill

        def submit_prefill(self, model_input, *args, **kwargs):
            # the rows that are intermediate parts of their prompts (vLLM chunked prefill): no kwarg for it in the
            # plugin's prefill_forward call, so set on the model (tt/generator_vllm.py _parts reads and clears it)
            if hasattr(self.model, "_parts"):
                self.model._hy4_inter = getattr(model_input, "intermediate_prefill_mask", None)
            return submit(self, model_input, *args, **kwargs)

        R.submit_prefill = submit_prefill


def _is_hy4(model_class) -> bool:
    return "HYV4Flash" in getattr(model_class, "__name__", "") or "tencent_hy4_flash" in getattr(model_class,
                                                                                                "__module__", "")


def _patch_tt_platform(module) -> None:
    # Chunked prefill budget (above): per prompt part HY4_CP_TOKENS tokens, per step unbounded
    orig = getattr(module, "_apply_chunked_prefill_policy", None)
    if orig is None or os.environ.get("HY4_CHUNKED_PREFILL", "1") != "1":
        return

    def policy(vllm_config, model_capabilities, model_class):
        orig(vllm_config, model_capabilities, model_class)
        sc = vllm_config.scheduler_config
        if _is_hy4(model_class) and sc.enable_chunked_prefill:
            sc.max_num_batched_tokens = max(int(sc.max_num_batched_tokens), int(vllm_config.model_config.max_model_len))
            sc.long_prefill_token_threshold = int(os.environ.get("HY4_CP_TOKENS", "2048"))
            sys.stderr.write(f"[hy4] chunked prefill: parts of {sc.long_prefill_token_threshold} tokens, "
                             f"max_num_batched_tokens {sc.max_num_batched_tokens}\n")

    module._apply_chunked_prefill_policy = policy


def _patch_tt_scheduler(module) -> None:
    # While a prompt is prefilled in parts (a running request with is_prefill_chunk) and other requests decode: after
    # every prefill step one decode-only step, so the decoding users wait at most one part (the plugin's default policy
    # prefers prefill whenever prefill work is pending: every part back to back, the decoders frozen meanwhile).
    S = getattr(module, "TTScheduler", None)
    if (S is None or getattr(S, "_hy4_alt", False) or getattr(S, "interleaves_chunked_prefill", False)
            or os.environ.get("HY4_CHUNKED_PREFILL", "1") != "1"):
        return
    sched, pf_only = S._schedule_without_pending_output_resumes, S._schedule_prefill_only

    def schedule_prefill_only(self):
        res = pf_only(self)
        self._hy4_pf_ran = res.total_num_scheduled_tokens > 0
        return res

    def schedule(self, throttle_prefills=False):
        if (getattr(self, "_hy4_after_pf", False) and self._forced_mode == module.TTSchedulingMode.DEFAULT
                and any(r.is_prefill_chunk for r in self.running)
                and any(not r.is_prefill_chunk for r in self.running)):
            self._hy4_after_pf = False
            return self._finalize_scheduler_output(self._schedule_decode_only())
        self._hy4_pf_ran = False
        out = sched(self, throttle_prefills)
        self._hy4_after_pf = self._hy4_pf_ran
        return out

    S._schedule_prefill_only = schedule_prefill_only
    S._schedule_without_pending_output_resumes = schedule
    S._hy4_alt = True


# module -> callback run right after that module has executed (each at most once per process)
_HOOKS = {
    "vllm_tt_plugin.model_runner": _patch_tt_runner,
    "vllm_tt_plugin.scheduler": _patch_tt_scheduler,
    "vllm_tt_plugin.platform": _patch_tt_platform,
    "vllm.model_executor.models.registry": _register_arch,
    "transformers.models.auto.configuration_auto": _register_config,
    "vllm.reasoning": _register_reasoning_parser,
    "vllm.tool_parsers": _register_tool_parser,
}


class _PostImportFinder(importlib.abc.MetaPathFinder):
    """Runs `_HOOKS[name]` once module `name` has executed (wraps its loader's exec_module)."""

    def find_spec(self, fullname, path, target=None):
        hook = _HOOKS.pop(fullname, None)
        if hook is None:
            return None
        sys.meta_path.remove(self)  # resolve the real spec without recursing into ourselves
        try:
            spec = importlib.util.find_spec(fullname)
        finally:
            if _HOOKS:
                sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return spec
        exec_module = spec.loader.exec_module

        def exec_and_hook(module):
            exec_module(module)
            hook(module)

        spec.loader.exec_module = exec_and_hook
        return spec


for _name in [n for n in _HOOKS if n in sys.modules]:
    _HOOKS.pop(_name)(sys.modules[_name])
if _HOOKS:
    sys.meta_path.insert(0, _PostImportFinder())

if os.environ.get("HYB_ENABLE") == "1":
    sys.path.insert(0, os.environ.get("HYB_POC_DIR", "/poc"))
    try:
        import hybrid

        hybrid.install()
        sys.stderr.write(f"[hybrid] installed in pid {os.getpid()}\n")
    except Exception as e:  # noqa: BLE001 - surface loudly, then re-raise
        sys.stderr.write(f"[hybrid] install FAILED in pid {os.getpid()}: {e!r}\n")
        raise
