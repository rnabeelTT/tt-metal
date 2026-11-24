# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""vLLM multimodal processor for Inkling-2-Mini images (vLLM 0.26 / transformers 5.12 know neither the architecture nor
its processor; transformers' Inkling2Processor + Inkling2ImageProcessor reimplemented: tt/vision.preprocess).

The chat template renders each image part as ``<|content_image|><|unused_200054|><|end_message|>``; the processor
expands the one placeholder into N = rows x cols placeholders (one per 40 x 40 px patch, transformers'
replace_image_token), and the TT plugin hands each image's pixel_values / image_grid_thw and its placeholder range to
prefill_forward (``mm_items``)."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from transformers.feature_extraction_utils import BatchFeature
from vllm.multimodal.inputs import MultiModalFieldConfig, MultiModalKwargsItems
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
    PromptUpdateDetails,
)

from models.autoports.thinkingmachines_inkling_2_mini.tt import vision

IMAGE_TOKEN = "<|unused_200054|>"
IMAGE_TOKEN_ID = 200054


def max_image_tokens(max_model_len: int) -> int:
    """The most tokens one image may take: the tower's cap (I2M_VISION_MAX_TOKENS) within the context."""
    return max(1, min(vision.max_tokens(), int(max_model_len) - 64))


class Inkling2ProcessingInfo(BaseProcessingInfo):
    def get_hf_processor(self, **kwargs: object):
        return None  # (no transformers processor: _call_hf_processor below does its work)

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"image": None}

    def max_tokens(self) -> int:
        return max_image_tokens(self.ctx.model_config.max_model_len)

    def get_mm_max_tokens_per_item(self, seq_len: int, mm_counts: Mapping[str, int]) -> Mapping[str, int]:
        return {"image": self.max_tokens()}


class Inkling2DummyInputsBuilder(BaseDummyInputsBuilder[Inkling2ProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return "<|content_image|>" + IMAGE_TOKEN if mm_counts.get("image", 0) else ""

    def get_dummy_mm_data(self, seq_len: int, mm_counts: Mapping[str, int], mm_options=None):
        n = mm_counts.get("image", 0)
        if not n:
            return {}
        tok = self.info.max_tokens()
        rows = max(1, int(math.isqrt(tok)))
        cols = max(1, tok // rows)
        return {"image": self._get_dummy_images(width=cols * vision.PATCH, height=rows * vision.PATCH, num_images=n)}


class Inkling2MultiModalProcessor(BaseMultiModalProcessor[Inkling2ProcessingInfo]):
    def _hf_processor_applies_updates(self, prompt_text, mm_items, hf_processor_mm_kwargs, tokenization_kwargs) -> bool:
        return False  # (one placeholder per image here; _get_prompt_updates expands it)

    def _call_hf_processor(self, prompt: str, mm_data: Mapping[str, object], mm_kwargs: Mapping[str, object],
                           tok_kwargs: Mapping[str, object]) -> BatchFeature:
        tokenizer = self.info.get_tokenizer()
        ids = tokenizer.encode(prompt, add_special_tokens=bool(tok_kwargs.get("add_special_tokens", False)))
        data: dict[str, Any] = {"input_ids": [ids]}
        images = list(mm_data.get("images") or [])
        if images:
            n_ph = sum(1 for t in ids if t == IMAGE_TOKEN_ID)
            if n_ph != len(images):
                raise ValueError(f"The prompt has {n_ph} image placeholder(s) for {len(images)} image(s).")
            cap = self.info.max_tokens()
            pvs, grids = [], []
            for i, img in enumerate(images):
                pv, grid = vision.preprocess(img)
                if pv.shape[0] > cap:
                    raise ValueError(
                        f"Image {i} is {img.size[0]} x {img.size[1]} px = {pv.shape[0]} image tokens (one per "
                        f"{vision.PATCH} x {vision.PATCH} px); this server takes at most {cap} per image. Send a smaller "
                        "image.")
                pvs.append(pv)
                grids.append(grid)
            data["pixel_values"] = torch.cat(pvs)
            data["image_grid_thw"] = torch.cat(grids)
        return BatchFeature(data=data, tensor_type=None)

    def _get_mm_fields_config(self, hf_inputs: BatchFeature, hf_processor_mm_kwargs: Mapping[str, object]
                              ) -> Mapping[str, MultiModalFieldConfig]:
        grid = hf_inputs.get("image_grid_thw", torch.empty((0, 3), dtype=torch.int64))
        return dict(
            pixel_values=MultiModalFieldConfig.flat_from_sizes("image", grid.prod(-1)),
            image_grid_thw=MultiModalFieldConfig.batched("image"),
        )

    def _get_prompt_updates(self, mm_items: Any, hf_processor_mm_kwargs: Mapping[str, Any],
                            out_mm_kwargs: MultiModalKwargsItems) -> Sequence[PromptUpdate]:
        grid = out_mm_kwargs.get_data().get("image_grid_thw")
        if grid is None or len(grid) == 0:
            return []

        def replacement(i: int):
            return PromptUpdateDetails.select_token_id([IMAGE_TOKEN_ID] * int(torch.as_tensor(grid[i]).prod()),
                                                       IMAGE_TOKEN_ID)

        return [PromptReplacement(modality="image", target=[IMAGE_TOKEN_ID], replacement=replacement)]


def register(cls):
    """The processor on the vLLM model class (no-op without vLLM, e.g. standalone tools)."""
    from vllm.multimodal import MULTIMODAL_REGISTRY

    return MULTIMODAL_REGISTRY.register_processor(
        Inkling2MultiModalProcessor, info=Inkling2ProcessingInfo, dummy_inputs=Inkling2DummyInputsBuilder)(cls)


__all__ = ["register", "max_image_tokens", "IMAGE_TOKEN_ID"]
