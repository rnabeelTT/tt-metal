# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""vLLM adapter for thinkingmachines/Inkling-Small (text) on a (1, 4) Blackhole mesh, through the hybrid-MoE overlay.

Single user (``--max-num-seqs 1``). Generation runs in the port's own engine (tt/inkling_serve.py): chunked prefill,
then speculative decoding with the checkpoint's native MTP drafter (``INKLING_SPEC_K`` drafts per verify pass, default 1;
0 = plain decode), verify / commit / drafter replayed as traces. The engine keeps every cache itself (the 7 global
layers' paged KV with an identity page table, the 35 sliding layers' rings, the 4 short-conv states per layer, the MTP
depths' caches), so the KV allocation vLLM asks for is a token budget only and prefix caching is off.

Plugin contract (vllm-tt-plugin at the M2.7 hybrid package's revision, 23b0736): the decode path is the M2.7 adapter's
external speculative step (``decode_forward`` -> ``_mtp_decode`` -> ``engine.step``, a host int32 ``[1, k + 1]`` row of
the accepted drafts + the verified token, -1 padded) with M2.7's ``mtp_plugin_shim`` for the variable-width output;
prefill returns the first token (on-device sampling mode) computed by the engine with the request's temperature /
top-k / top-p / seed (exact speculative sampling). Warm-up: phase 1 compiles every program (each prefill bucket, the
verify rows, the drafter), phase 2 captures the traces.

Registration: ``vllm_bundle/inkling_small/vllm_metadata.json`` (EXTRA_MODELS_DIR); vLLM itself knows the config
(``inkling_mm_model``), the chat template and the ``inkling`` reasoning / tool parsers.
"""

from __future__ import annotations

import math
import os
import time

import torch
from loguru import logger

from models.autoports.minimaxai_minimax_m2_7.tt.generator_vllm import MiniMaxM2ForCausalLM
from models.autoports.thinkingmachines_inkling_small.tt.model import create_model

HF_MODEL = "thinkingmachines/Inkling-Small"
BLOCK_SIZE = 64
SPEC_K = int(os.environ.get("INKLING_SPEC_K", "1"))
TRACE = os.environ.get("INKLING_TRACE", "1") == "1"

if SPEC_K > 0:
    # variable-width block output (1..k+1 tokens per step, -1 padded) through the plugin: import-time monkeypatches
    from models.autoports.minimaxai_minimax_m2_7.tt import mtp_plugin_shim

    mtp_plugin_shim.apply(SPEC_K + 1)


def _xgrammar_stop_ids(stop_ids):
    """Structured outputs: Inkling's tokenizer has no eos_token, so xgrammar guessed its stop token (200028, not the
    end of turn 200006): after a complete JSON value the grammar then allowed only 200028, which vLLM does not stop on,
    and the request failed ("grammar rejected tokens [200028]"). Every TokenizerInfo built in this process without
    stop ids gets the model's end-of-turn ids. Never fatal: on any surprise it logs and leaves xgrammar alone."""
    try:
        import xgrammar

        ti = xgrammar.TokenizerInfo
        raw = ti.__dict__.get("from_huggingface")
        if getattr(getattr(raw, "__func__", None), "_inkling_stop", False):
            return
        ids = [int(i) for i in stop_ids]

        def fill(tokenizer, kw):
            if kw.get("stop_token_ids") is None and getattr(tokenizer, "eos_token_id", None) is None:
                kw["stop_token_ids"] = list(ids)
            return kw

        if isinstance(raw, staticmethod):
            f = raw.__func__

            def from_huggingface(tokenizer, *a, **kw):
                return f(tokenizer, *a, **fill(tokenizer, kw))

            from_huggingface._inkling_stop = True
            ti.from_huggingface = staticmethod(from_huggingface)
        elif isinstance(raw, classmethod):
            f = raw.__func__

            def from_huggingface(klass, tokenizer, *a, **kw):
                return f(klass, tokenizer, *a, **fill(tokenizer, kw))

            from_huggingface._inkling_stop = True
            ti.from_huggingface = classmethod(from_huggingface)
        else:
            logger.warning(f"xgrammar TokenizerInfo.from_huggingface is a {type(raw).__name__}: stop ids not set")
    except Exception as e:  # noqa: BLE001 - structured outputs keep xgrammar's own guess
        logger.warning(f"xgrammar stop ids not set: {e!r}")


def _register_processor(cls):
    """vLLM resolves InklingForConditionalGeneration as a multimodal architecture and builds the processor registered
    on the model class: the qwen36 base's (Qwen3.5) rejects Inkling's config. Use vLLM's own Inkling processor (text
    requests pass through it unchanged; image / audio items reach prefill_forward as mm_items, tt/mm.py)."""
    try:
        from vllm.models.inkling.common.mm_preprocess import (
            InklingDummyInputsBuilder,
            InklingMultiModalProcessor,
            InklingProcessingInfo,
        )
        from vllm.multimodal import MULTIMODAL_REGISTRY
    except ImportError as e:  # standalone tools without vLLM
        logger.debug(f"vLLM Inkling processor not registered: {e}")
        return cls
    return MULTIMODAL_REGISTRY.register_processor(
        InklingMultiModalProcessor, info=InklingProcessingInfo, dummy_inputs=InklingDummyInputsBuilder
    )(cls)


@_register_processor
class InklingForConditionalGeneration(MiniMaxM2ForCausalLM):
    """vLLM bridge for Inkling-Small (HF arch InklingForConditionalGeneration): text, image and audio inputs."""

    supports_multimodal = True
    # the TT plugin passes every multimodal item (any modality, with its placeholder range and identifier) as the
    # mm_items prefill kwarg (vllm-tt-plugin branch inkling-mm)
    tt_generic_mm_items = True
    model_capabilities = {
        "supports_prefix_caching": False,
        "supports_async_decode": True,
        "supports_async_decode_overlap": True,
        "supports_sample_on_device": True,
        "supports_chunked_prefill": False,
        **({"max_output_tokens_per_step": SPEC_K + 1} if SPEC_K > 0 else {}),
    }
    decode_input_update_contract = 1

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        """Inkling's media placeholders (vLLM's models/inkling: the processor expands each into marker + rows); the
        qwen36 base's are Qwen's and refuse audio (chat requests with input_audio failed with 400)."""
        if modality.startswith("image"):
            return "<|content_image|>"
        if modality.startswith("audio"):
            return "<|content_audio_input|>"
        raise ValueError("Inkling takes image and audio inputs")

    @classmethod
    def initialize_vllm_model(
        cls, hf_config, mesh_device, max_batch_size, max_seq_len=None, tt_data_parallel=1, optimizations=None, **kwargs
    ):
        if int(tt_data_parallel) != 1 or int(max_batch_size) != 1:
            raise ValueError(
                f"{cls.__name__} serves one user on one (1,4) mesh (--max-num-seqs 1); got "
                f"max_batch_size={max_batch_size} tt_data_parallel={tt_data_parallel}"
            )
        max_seq_len = int(max_seq_len or 131072)
        os.environ.setdefault("HF_MODEL", HF_MODEL)
        # the engine's rows: verify width k + 1 <= the hybrid overlay's host-expert rows
        hyb_rows = int(os.environ.get("HYB_DECODE_ROWS", "1"))
        if os.environ.get("HYB_ENABLE") == "1" and hyb_rows < SPEC_K + 1:
            raise ValueError(
                f"HYB_DECODE_ROWS={hyb_rows} < the verify width {SPEC_K + 1}: rows past it would get no " "host experts"
            )
        # the context plus the verify window (positions past max_model_len are written, never read)
        args, model = create_model(mesh_device, max_batch_size=1, max_seq_len=max_seq_len + 2 * (SPEC_K + 1) + 64)
        self = cls([model], [args], mesh_device)
        from models.autoports.thinkingmachines_inkling_small.tt.inkling_serve import InklingSpecServing

        self._engine = InklingSpecServing(model, args, mesh_device, SPEC_K, trace=TRACE)
        _xgrammar_stop_ids(sorted(self._engine.eos))
        self._mtp = self._engine  # the inherited _mtp_decode drives it
        self._gen = None
        self._context = max_seq_len
        self._warmed = self._captured = False
        self._live_slots = set()
        self._decode_steps = 0
        if os.environ.get("QWEN35_FAULTHANDLER", "1") == "1":
            import faulthandler
            import signal

            faulthandler.register(signal.SIGUSR1, all_threads=True, chain=False)
        logger.info(
            f"{cls.__name__}: max_model_len={max_seq_len} layers={len(model.layers)} policy="
            f"{getattr(model, 'precision_policy', {}).get('config_id')} MTP k={SPEC_K} trace={TRACE} "
            "(engine-owned caches)"
        )
        return self

    # ------------------------------------------------------------------ KV: a token budget only
    @classmethod
    def get_max_tokens_all_users(
        cls, model_name="", num_devices=1, tt_data_parallel=1, max_model_len=None, max_num_seqs=None, **kwargs
    ):
        """One user's context plus the verify lookahead, in blocks. The engine's caches hold it (module doc)."""
        n = int(max_model_len or 131072) + 2 * (SPEC_K + 1) + BLOCK_SIZE
        return math.ceil(n / BLOCK_SIZE) * BLOCK_SIZE

    def allocate_kv_cache(self, kv_cache_shape, dtype, num_layers):
        logger.info(
            f"KV: vLLM asked for {num_layers} layers of {tuple(kv_cache_shape)}; the engine keeps its own "
            "caches (nothing allocated)"
        )
        return [(None, None)] * int(num_layers)

    def allocate_kv_cache_per_layer(self, per_layer_specs):
        logger.info(
            f"KV: vLLM asked for {len(per_layer_specs)} per-layer caches; the engine keeps its own caches "
            "(nothing allocated)"
        )
        return [(None, None)] * len(per_layer_specs)

    # ------------------------------------------------------------------ warm-up (plugin phases)
    def warmup_model_prefill(self, kv_cache, enable_trace, *args, **kwargs):
        if not self._warmed:
            self._engine.warm()
            self._warmed = True
        self.already_warmed_up_prefill = True

    def warmup_model_decode(self, *args, **kwargs):
        if not self._warmed:
            self._engine.warm()
            self._warmed = True
        if kwargs.get("enable_trace", False) and not self._captured:
            self._engine.capture()
            self._captured = True

    # ------------------------------------------------------------------ prefill
    def prefill_forward(self, tokens, page_table, kv_cache, prompt_lens, **kwargs):
        """A new request: the engine prefills it and picks its first token with the request's sampling params.
        Returns tokens [N, 1] (on-device sampling mode) or, when the plugin samples on the host, logits [N, 1, V]
        whose argmax is that token (greedy host sampling then agrees with the engine's continuation)."""
        N = int(tokens.shape[0])
        start_pos = kwargs.get("start_pos")
        if start_pos is not None and any(int(v) for v in torch.as_tensor(start_pos).reshape(-1).tolist()):
            raise NotImplementedError("prefix caching / chunked prefill are off for Inkling (start_pos must be 0)")

        def present(v):  # the plugin passes the multimodal kwargs on every prefill (None / empty lists for text)
            if v is None:
                return False
            if isinstance(v, (list, tuple)):
                return any(present(x) for x in v)
            return not (hasattr(v, "numel") and v.numel() == 0)

        if present(kwargs.get("pixel_values_videos")):
            raise NotImplementedError("Inkling serving takes text, image and audio inputs (no video)")
        mm_items = kwargs.get("mm_items")
        if mm_items is None and any(present(kwargs.get(k)) for k in ("pixel_values", "input_features", "audio_features")):
            raise NotImplementedError("image / audio inputs need the TT plugin's mm_items (vllm-tt-plugin inkling-mm)")
        plens = [int(prompt_lens[u]) for u in range(N)] if prompt_lens is not None else [int(tokens.shape[1])] * N
        sp = kwargs.get("sampling_params")
        out = []
        for u in range(N):
            ids = tokens[u, : plens[u]].tolist()
            sampling = self._mtp_sampling(sp, u)
            items = mm_items[u] if mm_items is not None and u < len(mm_items) else None
            mm = None
            if items:
                from models.autoports.thinkingmachines_inkling_small.tt import mm as mm_rows

                if getattr(self, "_towers", None) is None:
                    self._towers = mm_rows.InklingTowers(mm_rows.default_checkpoint())
                t_mm = time.time()
                mm = mm_rows.build(items, ids, self._towers)
                logger.info(
                    f"Inkling multimodal prompt: {len(items)} item(s), {int(mm['pos'].numel())} embedding rows "
                    f"({time.time() - t_mm:.2f} s on the host)"
                )
            out.append(self._engine.prefill(ids, sampling, mm))
        self._live_slots = set(range(N))
        if sp is not None:
            return torch.tensor(out, dtype=torch.int64).reshape(N, 1)
        # host sampling (the plugin sends no device params, e.g. logprobs): the real first-token logits (padded vocab
        # rows masked); a host pick that differs from the engine's re-anchors its continuation at the next decode
        self._engine.adopt_next = True  # the plugin samples the first token: its pick is the continuation
        V = int(getattr(self.model_args[0], "vocab_size", 201024))
        logits = torch.full((N, 1, V), float("-inf"), dtype=torch.float32)
        lg = self._engine.last_logits
        logits[:, 0, : lg.numel()] = lg
        return logits

    def _mtp_decode(self, args, kwargs):
        sp = kwargs.get("sampling_params", args[6] if len(args) > 6 else None)
        if sp is None:
            return self._host_sampled_decode(args, kwargs)
        out = super()._mtp_decode(args, kwargs)
        if self._decode_steps % 4096 == 0 and self._decode_steps:
            logger.info(f"Inkling engine stats: {self._engine.stats}")
        return out

    def _host_sampled_decode(self, args, kwargs):
        """The plugin samples on the host (logprobs, min_p, logit_bias, bad words, allowed ids, min_tokens, structured
        outputs): one plain decode step per call, the next-token logits [B_pad, 1, V] (the MTP shim hands host-sampled
        steps to the plugin's one-token read-back)."""
        tokens = torch.as_tensor(kwargs.get("tokens", args[0] if args else None)).reshape(-1)
        pos = torch.as_tensor(kwargs.get("start_pos", args[1] if len(args) > 1 else None)).reshape(-1)
        live = [i for i in range(pos.numel()) if int(pos[i]) >= 0]
        assert len(live) <= 1, "single-user Inkling step got several live rows"
        V = int(getattr(self.model_args[0], "vocab_size", 201024))
        logits = torch.full((tokens.numel(), 1, V), float("-inf"), dtype=torch.float32)
        for i in live:
            lg = self._engine.step_logits(int(tokens[i]), int(pos[i]))
            n = min(V, lg.numel(), self._engine.V)
            logits[i, 0, :n] = lg[:n].float()
        self._live_slots = set(live)
        self._decode_steps += 1
        return logits


__all__ = ["InklingForConditionalGeneration", "HF_MODEL"]
