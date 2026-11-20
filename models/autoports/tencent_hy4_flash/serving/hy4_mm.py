# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""vLLM 0.26 multimodal processor for Hy4-flash images (vLLM and the image's transformers know no hy_v4_flash).

The chat template writes <image_start><image_pad><image_end> per image; this processor tokenizes the prompt itself,
preprocesses each image exactly as HF HYV4FlashImageProcessor (tt/vision_host.py: smart_resize to 32 px multiples
within [4096, HY4_IMAGE_MAX_PIXELS] px, PIL bicubic, (x / 255 - 0.5) / 0.5, row-major 16 x 16 patches) and vLLM expands
each pad to (gh / 2) x (gw / 2 + 1) pads (the merged rows plus one newline feature per row). Request data reaching the
TT model (plugin _gather_multi_modal_inputs): pixel_values [gh * gw, 768] and image_grid_thw [3] per image.
Registered on both the architecture stub (hy4_vllm_stub.py, what vLLM inspects) and the TT adapter (what the frontend
resolves after the plugin's TT-prefix rewrite). Video is not served (HY4 video: a stretch goal).
"""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from functools import partial

import numpy as np
import torch
from transformers import BatchFeature
from vllm.multimodal.inputs import MultiModalFieldConfig
from vllm.multimodal.processing import BaseDummyInputsBuilder, BaseMultiModalProcessor, BaseProcessingInfo
from vllm.multimodal.processing.processor import PromptReplacement, PromptUpdateDetails

try:  # the vision host helpers are plain python (no ttnn)
    from models.autoports.tencent_hy4_flash.tt import vision_host as vh
except ImportError:  # pragma: no cover - the repo root not on sys.path
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))
    from models.autoports.tencent_hy4_flash.tt import vision_host as vh

IMAGE_PLACEHOLDER = "<｜hy_image▁start:opensource｜><｜hy_image▁pad:opensource｜><｜hy_image▁end:opensource｜>"
MAX_IMAGES = int(os.environ.get("HY4_VISION_MAX_IMAGES", "4"))


class Hy4ProcessingInfo(BaseProcessingInfo):
    def get_hf_processor(self, **kwargs: object):
        return None  # no HF processor in this environment: Hy4MultiModalProcessor does the work

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"image": MAX_IMAGES}

    def get_image_size_with_most_features(self) -> tuple[int, int]:
        side = int((vh.max_pixels()) ** 0.5) // 32 * 32
        return side, side

    def get_mm_max_tokens_per_item(self, seq_len, mm_counts):
        w, h = self.get_image_size_with_most_features()
        return {"image": vh.num_image_tokens(vh.grid_for(h, w))}


class Hy4DummyInputsBuilder(BaseDummyInputsBuilder[Hy4ProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return IMAGE_PLACEHOLDER * mm_counts.get("image", 0)

    def get_dummy_mm_data(self, seq_len, mm_counts, mm_options=None):
        w, h = self.info.get_image_size_with_most_features()
        return {"image": self._get_dummy_images(width=w, height=h, num_images=mm_counts.get("image", 0))}


class Hy4MultiModalProcessor(BaseMultiModalProcessor[Hy4ProcessingInfo]):
    def _call_hf_processor(self, prompt, mm_data, mm_kwargs, tok_kwargs) -> BatchFeature:
        tok = self.info.get_tokenizer()
        kw = {k: v for k, v in dict(tok_kwargs).items() if k in ("add_special_tokens",)}
        kw.setdefault("add_special_tokens", False)
        ids = tok(prompt, **kw)["input_ids"] if prompt else []
        data = {"input_ids": [list(ids)]}
        images = mm_data.get("images") or []
        if not isinstance(images, (list, tuple)):
            images = [images]
        if images:
            pvs, grids = [], []
            for img in images:
                if isinstance(img, torch.Tensor) or isinstance(img, np.ndarray):
                    from PIL import Image

                    a = np.asarray(img)
                    if a.ndim == 3 and a.shape[0] in (1, 3) and a.shape[-1] not in (1, 3):
                        a = a.transpose(1, 2, 0)
                    img = Image.fromarray(a.astype(np.uint8))
                pv, g = vh.preprocess(img)
                pvs.append(torch.from_numpy(pv))
                grids.append(list(g))
            data["pixel_values"] = torch.cat(pvs)
            data["image_grid_thw"] = torch.tensor(grids, dtype=torch.long)
        return BatchFeature(data)

    def _hf_processor_applies_updates(self, prompt_text, mm_items, hf_processor_mm_kwargs, tokenization_kwargs):
        return False  # one pad per image in the processed ids: vLLM expands it (_get_prompt_updates)

    def _get_mm_fields_config(self, hf_inputs, hf_processor_mm_kwargs):
        grid = hf_inputs.get("image_grid_thw", torch.empty((0, 3), dtype=torch.long))
        return {
            "pixel_values": MultiModalFieldConfig.flat_from_sizes("image", grid.prod(-1)),
            "image_grid_thw": MultiModalFieldConfig.batched("image"),
        }

    def _get_prompt_updates(self, mm_items, hf_processor_mm_kwargs, out_mm_kwargs):
        def replacement(item_idx: int, modality: str):
            grid = out_mm_kwargs[modality][item_idx]["image_grid_thw"].data
            n = vh.num_image_tokens([int(v) for v in grid.tolist()])
            return PromptUpdateDetails.select_token_id([vh.IMAGE_PAD] * n, vh.IMAGE_PAD)

        return [PromptReplacement(modality="image", target=[vh.IMAGE_PAD], replacement=partial(replacement, modality="image"))]


def register(cls):
    """Class decorator: the Hy4 processor on a model class (+ the multimodal interface attributes)."""
    from vllm.multimodal import MULTIMODAL_REGISTRY

    cls.supports_multimodal = True
    cls.get_placeholder_str = classmethod(lambda c, modality, i: IMAGE_PLACEHOLDER if modality.startswith("image")
                                          else None)
    return MULTIMODAL_REGISTRY.register_processor(Hy4MultiModalProcessor, info=Hy4ProcessingInfo,
                                                  dummy_inputs=Hy4DummyInputsBuilder)(cls)


def enabled() -> bool:
    return os.environ.get("HY4_VISION", "1") == "1"
