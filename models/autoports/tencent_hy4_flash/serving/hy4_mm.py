# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""vLLM 0.26 multimodal processor for Hy4-flash images (vLLM and the image's transformers know no hy_v4_flash).

The chat template writes <image_start><image_pad><image_end> per image; this processor tokenizes the prompt itself,
preprocesses each image exactly as HF HYV4FlashImageProcessor (tt/vision_host.py: smart_resize to 32 px multiples
within [4096, HY4_IMAGE_MAX_PIXELS] px, PIL bicubic, (x / 255 - 0.5) / 0.5, row-major 16 x 16 patches) and vLLM expands
each pad to (gh / 2) x (gw / 2 + 1) pads (the merged rows plus one newline feature per row). Request data reaching the
TT model (plugin _gather_multi_modal_inputs): pixel_values [gh * gw, 768] and image_grid_thw [3] per image.
Registered on both the architecture stub (hy4_vllm_stub.py, what vLLM inspects) and the TT adapter (what the frontend
resolves after the plugin's TT-prefix rewrite).

Videos (Tencent's vLLM PR 68296 layout; the chat template writes <video_start><video_pad><video_end> per video): frames
from vLLM's loader with their metadata; a fully decoded clip (do_sample_frames) is resampled to 2 fps as the PR does
(vision_host.sample_frames), every frame resized by the PR's adaptive_resize (one budget for the whole clip) and
normalized as the PR; the pad becomes per frame "<{t:.1f} seconds>" <image_start> (gh / 2) x (gw / 2 + 1) video pads
<image_end>, t = frame index / fps. Request data reaching the TT model (serving/sitecustomize.py's plugin shim):
pixel_values_videos [T * gh * gw, 768] and video_grid_thw [3] per video.
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
VIDEO_PLACEHOLDER = "<｜hy_video▁start:opensource｜><｜hy_video▁pad:opensource｜><｜hy_video▁end:opensource｜>"
MAX_IMAGES = int(os.environ.get("HY4_VISION_MAX_IMAGES", "4"))
MAX_VIDEOS = int(os.environ.get("HY4_VISION_MAX_VIDEOS", "1"))
MAX_FRAMES = min(int(os.environ.get("HY4_VIDEO_MAX_FRAMES", "64")), vh.VIDEO_MAX_FRAMES)  # per video, after sampling


def _video_inputs(item):
    """One vLLM video item ((frames [T, H, W, 3], metadata) or frames) -> (pixel_values, grid, timestamps)."""
    frames, meta = item if isinstance(item, tuple) and len(item) == 2 else (item, None)
    if isinstance(frames, (list, tuple)):
        frames = np.stack([np.asarray(f) for f in frames])
    frames = np.asarray(frames)
    if frames.ndim == 4 and frames.shape[1] in (1, 3) and frames.shape[-1] not in (1, 3):
        frames = frames.transpose(0, 2, 3, 1)
    if frames.shape[-1] == 1:
        frames = np.repeat(frames, 3, axis=-1)
    meta = dict(meta or {})
    fps = float(meta.get("fps") or 24.0)  # the PR's default
    idx = list(meta.get("frames_indices") or range(len(frames)))
    total = int(meta.get("total_num_frames") or len(frames))
    if meta.get("do_sample_frames") and len(frames) == total:
        sel = vh.sample_frames(total, fps)
        frames, idx = frames[sel], [idx[i] for i in sel]
    if len(frames) > MAX_FRAMES:
        raise ValueError(f"Hy4: a video of {len(frames)} frames exceeds HY4_VIDEO_MAX_FRAMES={MAX_FRAMES}")
    return vh.video_preprocess(frames.astype(np.uint8), fps=fps, frames_indices=idx)


def _max_video_tokens() -> int:
    # pads <= the PR budget (max_pixels / 32^2 - reserve - 6 T, or 1 cell + newline per frame when it runs out) plus
    # per frame a timestamp (<= 8 tokens) and image start / end
    return vh.video_max_pixels() // 1024 + 12 * MAX_FRAMES


class Hy4ProcessingInfo(BaseProcessingInfo):
    def get_hf_processor(self, **kwargs: object):
        return None  # no HF processor in this environment: Hy4MultiModalProcessor does the work

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"image": MAX_IMAGES, "video": MAX_VIDEOS}

    def get_data_parser(self):
        from vllm.multimodal.parse import MultiModalDataParser

        # video metadata (fps, frame indices) kept: the timestamps need it (as the PR's HYV4VLDataParser)
        return MultiModalDataParser(expected_hidden_size=self._get_expected_hidden_size(), video_needs_metadata=True)

    def get_image_size_with_most_features(self) -> tuple[int, int]:
        side = int((vh.max_pixels()) ** 0.5) // 32 * 32
        return side, side

    def get_mm_max_tokens_per_item(self, seq_len, mm_counts):
        w, h = self.get_image_size_with_most_features()
        return {"image": vh.num_image_tokens(vh.grid_for(h, w)), "video": _max_video_tokens()}


class Hy4DummyInputsBuilder(BaseDummyInputsBuilder[Hy4ProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return IMAGE_PLACEHOLDER * mm_counts.get("image", 0) + VIDEO_PLACEHOLDER * mm_counts.get("video", 0)

    def get_dummy_mm_data(self, seq_len, mm_counts, mm_options=None):
        w, h = self.info.get_image_size_with_most_features()
        out = {"image": self._get_dummy_images(width=w, height=h, num_images=mm_counts.get("image", 0))}
        if mm_counts.get("video", 0):
            T = min(MAX_FRAMES, 32)
            vids = self._get_dummy_videos(width=vh.VIDEO_MAX_SIDE, height=vh.VIDEO_MAX_SIDE, num_frames=T,
                                          num_videos=mm_counts["video"])
            out["video"] = [(v, {"fps": 2.0, "total_num_frames": T, "frames_indices": list(range(T)),
                                 "do_sample_frames": False}) for v in vids]
        return out


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
        videos = mm_data.get("videos") or []
        if videos:
            pvs, grids, times = [], [], []
            for item in videos:
                pv, g, ts = _video_inputs(item)
                pvs.append(torch.from_numpy(pv))
                grids.append(list(g))
                times.append(torch.tensor(ts, dtype=torch.float64))
            data["pixel_values_videos"] = torch.cat(pvs)
            data["video_grid_thw"] = torch.tensor(grids, dtype=torch.long)
            data["video_timestamps"] = torch.cat(times)
        return BatchFeature(data)

    def _hf_processor_applies_updates(self, prompt_text, mm_items, hf_processor_mm_kwargs, tokenization_kwargs):
        return False  # one pad per image in the processed ids: vLLM expands it (_get_prompt_updates)

    def _get_mm_fields_config(self, hf_inputs, hf_processor_mm_kwargs):
        grid = hf_inputs.get("image_grid_thw", torch.empty((0, 3), dtype=torch.long))
        out = {
            "pixel_values": MultiModalFieldConfig.flat_from_sizes("image", grid.prod(-1)),
            "image_grid_thw": MultiModalFieldConfig.batched("image"),
        }
        if "video_grid_thw" in hf_inputs:
            vg = hf_inputs["video_grid_thw"]
            out.update({
                "pixel_values_videos": MultiModalFieldConfig.flat_from_sizes("video", vg.prod(-1)),
                "video_grid_thw": MultiModalFieldConfig.batched("video"),
                "video_timestamps": MultiModalFieldConfig.flat_from_sizes("video", vg[:, 0]),
            })
        return out

    def _get_prompt_updates(self, mm_items, hf_processor_mm_kwargs, out_mm_kwargs):
        def replacement(item_idx: int, modality: str):
            grid = out_mm_kwargs[modality][item_idx]["image_grid_thw"].data
            n = vh.num_image_tokens([int(v) for v in grid.tolist()])
            return PromptUpdateDetails.select_token_id([vh.IMAGE_PAD] * n, vh.IMAGE_PAD)

        def video_replacement(item_idx: int):
            it = out_mm_kwargs["video"][item_idx]
            grid = [int(v) for v in it["video_grid_thw"].data.tolist()]
            ts = it["video_timestamps"].data.tolist()
            toks = vh.video_frame_tokens(grid, ts, self.info.get_tokenizer())
            return PromptUpdateDetails.select_token_id(toks, vh.VIDEO_PAD)

        return [PromptReplacement(modality="image", target=[vh.IMAGE_PAD], replacement=partial(replacement, modality="image")),
                PromptReplacement(modality="video", target=[vh.VIDEO_PAD], replacement=video_replacement)]


def register(cls):
    """Class decorator: the Hy4 processor on a model class (+ the multimodal interface attributes)."""
    from vllm.multimodal import MULTIMODAL_REGISTRY

    cls.supports_multimodal = True
    cls.get_placeholder_str = classmethod(lambda c, modality, i: IMAGE_PLACEHOLDER if modality.startswith("image")
                                          else VIDEO_PLACEHOLDER if modality.startswith("video") else None)
    return MULTIMODAL_REGISTRY.register_processor(Hy4MultiModalProcessor, info=Hy4ProcessingInfo,
                                                  dummy_inputs=Hy4DummyInputsBuilder)(cls)


def enabled() -> bool:
    return os.environ.get("HY4_VISION", "1") == "1"
