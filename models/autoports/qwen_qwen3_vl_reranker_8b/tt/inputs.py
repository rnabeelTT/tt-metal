# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-side input preparation shared by the TT model and the HF reference.

Mirrors the official ``Qwen3VLReranker`` (QwenLM/Qwen3-VL-Embedding ``src/models/qwen3_vl_reranker.py``): the same
system prompt, ``<Instruct>/<Query>/<Document>`` layout (videos, then images, then text inside each part), the same
pixel budgets, and the same truncation that keeps special tokens and the last five template tokens. One pair is one
sequence; nothing is padded here (the TT side pads to its own buckets).

MRoPE position ids come from ``Qwen3VLModel.get_rope_index`` called on the config alone, so text and image positions
match HF exactly; for a text-only pair all three rows equal ``arange(T)``.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import torch
from PIL import Image

HF_MODEL = "Qwen/Qwen3-VL-Reranker-8B"
HF_REVISION = "b212dc8c91a8164aef1ea2de9c1a867611e75c04"  # the validated snapshot
YES_TOKEN = 9693
NO_TOKEN = 2152

MAX_LENGTH = 10240
IMAGE_FACTOR = 32
MIN_PIXELS = 4 * IMAGE_FACTOR * IMAGE_FACTOR
MAX_PIXELS = 1800 * IMAGE_FACTOR * IMAGE_FACTOR
FPS = 1.0
MAX_FRAMES = 64
FRAME_MAX_PIXELS = 768 * IMAGE_FACTOR * IMAGE_FACTOR
MAX_TOTAL_PIXELS = 10 * FRAME_MAX_PIXELS
DEFAULT_INSTRUCTION = "Given a search query, retrieve relevant candidates that answer the query."
SYSTEM_PROMPT = (
    "Judge whether the Document meets the requirements based on the Query and the Instruct provided. "
    'Note that the answer can only be "yes" or "no".'
)


@dataclass
class PairInputs:
    """One (query, document) pair, ready for either model."""

    input_ids: torch.Tensor  # [T] int64
    position_ids: torch.Tensor  # [3, T] int64 (MRoPE t/h/w)
    pixel_values: Optional[torch.Tensor] = None  # [n_patches, 1536] images
    image_grid_thw: Optional[torch.Tensor] = None  # [n_images, 3]
    pixel_values_videos: Optional[torch.Tensor] = None
    video_grid_thw: Optional[torch.Tensor] = None
    mm_token_type_ids: Optional[torch.Tensor] = None  # [T] 0 text / 1 image / 2 video
    extra: dict = field(default_factory=dict)

    @property
    def length(self) -> int:
        return int(self.input_ids.shape[-1])

    @property
    def has_vision(self) -> bool:
        return self.pixel_values is not None or self.pixel_values_videos is not None

    def hf_kwargs(self) -> dict[str, Any]:
        kw = {"input_ids": self.input_ids[None], "attention_mask": torch.ones_like(self.input_ids)[None]}
        if self.mm_token_type_ids is not None:
            kw["mm_token_type_ids"] = self.mm_token_type_ids[None]
        for k in ("pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"):
            v = getattr(self, k)
            if v is not None:
                kw[k] = v
        return kw


MAX_VIDEO_SECONDS = float(os.environ.get("QWEN3VL_RERANKER_MAX_VIDEO_SECONDS", "1800"))


def _read_video_pyav(ele: dict):
    """``qwen_vl_utils`` video reader on PyAV, with the torchvision reader's semantics (``smart_nframes`` over the
    decoded frame count and the stream fps, evenly spaced indices, real-fps metadata) -- torchvision >= 0.26 dropped
    ``io.read_video``. Two passes, so only the selected frames are ever converted to RGB (a long clip never sits in
    memory as full frames)."""
    import av
    import torch
    from qwen_vl_utils import vision_process as vp

    path = ele["video"]
    path = path[7:] if path.startswith("file://") else path
    with av.open(path) as c:
        s = c.streams.video[0]
        fps = float(s.average_rate or s.guessed_rate or 0) or 1.0
        total = sum(1 for _ in c.demux(s) if _.size)  # packets carrying a frame, no decode
        if total <= 0:
            raise ValueError(f"no video frames in {path}")
        if total / fps > MAX_VIDEO_SECONDS:
            raise ValueError(f"video is {total / fps:.0f} s long; the limit is {MAX_VIDEO_SECONDS:.0f} s")
    n = vp.smart_nframes(ele, total_frames=total, video_fps=fps)
    idx = torch.linspace(0, total - 1, n).round().long()
    want = {}
    for k, v in enumerate(idx.tolist()):
        want.setdefault(v, []).append(k)
    out = [None] * n
    with av.open(path) as c:
        for i, fr in enumerate(c.decode(video=0)):
            if i in want:
                a = torch.from_numpy(fr.to_ndarray(format="rgb24")).permute(2, 0, 1)
                for k in want[i]:
                    out[k] = a
            if i >= idx[-1]:
                break
    last = next((x for x in reversed(out) if x is not None), None)
    if last is None:
        raise ValueError(f"no frames decoded from {path}")
    out = [x if x is not None else last for x in out]  # a short tail decode: repeat the last frame
    meta = dict(fps=fps, frames_indices=idx, total_num_frames=total, video_backend="pyav")
    return torch.stack(out), meta, n / max(total, 1e-6) * fps


class VideoFile:
    """An encoded video on disk that the formatter reads through the official file path; the file is removed when
    the object is collected (the server writes request clips to temporary files)."""

    def __init__(self, path: str, fps: float | None = None, owned: bool = True):
        self.path, self.fps, self.owned = path, fps, owned

    def __del__(self):
        if self.owned:
            try:
                os.unlink(self.path)
            except OSError:
                pass


def install_video_reader():
    """Route ``qwen_vl_utils`` video files through ``_read_video_pyav`` (idempotent)."""
    from qwen_vl_utils import vision_process as vp

    vp.VIDEO_READER_BACKENDS["pyav"] = _read_video_pyav
    vp.get_video_reader_backend = lambda: "pyav"


def decode_video(path: str, fps: float = FPS, max_frames: int = MAX_FRAMES) -> list:
    """Frames of a video file as the official reader samples them (``fps``, ``max_frames``), as RGB PIL images."""
    v, _, _ = _read_video_pyav({"video": path, "fps": fps, "max_frames": max_frames})
    return [Image.fromarray(f.permute(1, 2, 0).numpy()) for f in v]


def _sample_frames(frames, max_segments):
    if len(frames) <= max_segments:
        return frames
    idx = np.linspace(0, len(frames) - 1, max_segments, dtype=int).tolist()
    return [frames[i] for i in idx]


def _as_list(x):
    if x is None:
        return []
    return list(x) if isinstance(x, (list, tuple)) else [x]


def _norm_image(img):
    if isinstance(img, Image.Image):
        return img
    if isinstance(img, str):
        return img if img.startswith(("http://", "https://", "file://", "data:")) else "file://" + img
    raise TypeError(f"unrecognized image type {type(img)}")


class PairFormatter:
    """Formats and tokenizes pairs exactly like the official reranker."""

    def __init__(
        self,
        model_path: str = HF_MODEL,
        max_length: int = MAX_LENGTH,
        min_pixels: int = MIN_PIXELS,
        max_pixels: int = MAX_PIXELS,
        total_pixels: int = MAX_TOTAL_PIXELS,
        fps: float = FPS,
        max_frames: int = MAX_FRAMES,
        processor=None,
        hf_config=None,
    ):
        from transformers import AutoConfig, AutoProcessor

        self.processor = processor or AutoProcessor.from_pretrained(model_path, padding_side="left")
        self.config = hf_config or AutoConfig.from_pretrained(model_path)
        self.max_length = max_length
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.total_pixels = total_pixels
        self.fps = fps
        self.max_frames = max_frames
        # every special added token (14: <|im_start|>, <|vision_start|>, <|image_pad|> ...), as tokenizer.all_special_ids
        # gave under the transformers 4.57 the official code ran on (5.x reports only the two eos/pad ids)
        tok = self.processor.tokenizer
        self.special_ids = {i for i, a in tok.added_tokens_decoder.items() if a.special} | set(tok.all_special_ids)
        install_video_reader()

    # ---- messages ---------------------------------------------------------------------------------------------
    def content(self, part: dict | str | None, prefix: str) -> list[dict]:
        if isinstance(part, str):
            part = {"text": part}
        part = part or {}
        texts = _as_list(part.get("text"))
        images = _as_list(part.get("image"))
        video = part.get("video")
        out = [{"type": "text", "text": prefix}]
        if not texts and not images and video is None:
            out.append({"type": "text", "text": "NULL"})
            return out
        if video is not None:
            if isinstance(
                video, (str, VideoFile)
            ):  # a video file: the official path (fps / max_frames, no total_pixels)
                fps = getattr(video, "fps", None) or part.get("fps", self.fps)
                v = video.path if isinstance(video, VideoFile) else video
                v = v if v.startswith(("http://", "https://", "file://")) else "file://" + v
                # total_pixels too (official applies it to frame lists only): an uncapped HD clip reaches ~23K visual
                # tokens, which the 10240-token truncation keeps as specials while dropping ALL text (query included)
                out.append(
                    {
                        "type": "video",
                        "video": v,
                        "fps": fps,
                        "max_frames": self.max_frames,
                        "total_pixels": self.total_pixels,
                    }
                )
            else:  # a list of frames: the official frame-list path
                frames = _sample_frames(list(video), self.max_frames)
                frames = [("file://" + f if isinstance(f, str) else f) for f in frames]
                out.append({"type": "video", "video": frames, "total_pixels": self.total_pixels})
        for img in images:
            out.append(
                {
                    "type": "image",
                    "image": _norm_image(img),
                    "min_pixels": self.min_pixels,
                    "max_pixels": self.max_pixels,
                }
            )
        for t in texts:
            out.append({"type": "text", "text": t})
        return out

    def messages(self, query, document, instruction: Optional[str] = None) -> list[dict]:
        contents = [{"type": "text", "text": "<Instruct>: " + (instruction or DEFAULT_INSTRUCTION)}]
        contents += self.content(query, "<Query>:")
        contents += self.content(document, "\n<Document>:")
        return [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": contents},
        ]

    # ---- tokenization -----------------------------------------------------------------------------------------
    def _truncate(self, ids: list[int]) -> list[int]:
        body, tail = ids[:-5], ids[-5:]
        if len(body) <= self.max_length:
            return ids
        n_special = sum(1 for t in body if t in self.special_ids)
        keep = self.max_length - n_special
        if keep <= 0:
            raise ValueError(
                f"the pair's images / video take {n_special} tokens, more than the {self.max_length}-token budget; "
                "the official truncation would drop all of its text (send fewer or smaller images)"
            )
        out, kept = [], 0
        for t in body:
            if t in self.special_ids:
                out.append(t)
            elif kept < keep:
                out.append(t)
                kept += 1
        return out + tail

    def __call__(self, query, document, instruction: Optional[str] = None) -> PairInputs:
        from qwen_vl_utils import process_vision_info

        msgs = self.messages(query, document, instruction)
        text = self.processor.apply_chat_template([msgs], tokenize=False, add_generation_prompt=True)
        images, videos, video_kwargs = process_vision_info(
            [msgs], image_patch_size=16, return_video_kwargs=True, return_video_metadata=True
        )
        metas = None
        if videos is not None:
            videos, metas = (list(x) for x in zip(*videos))
        enc = self.processor(
            text=text,
            images=images,
            videos=videos,
            video_metadata=metas,
            truncation=False,
            padding=False,
            do_resize=False,
            return_tensors="pt",
            **video_kwargs,
        )
        ids = enc["input_ids"][0].tolist()
        tids = self._truncate(ids)
        input_ids = torch.tensor(tids, dtype=torch.long)
        merge2 = self.config.vision_config.spatial_merge_size**2
        for tok_id, grid, what in (
            (self.config.image_token_id, enc.get("image_grid_thw"), "image"),
            (self.config.video_token_id, enc.get("video_grid_thw"), "video"),
        ):
            want = 0 if grid is None else int(grid.prod(-1).sum()) // merge2
            have = int((input_ids == tok_id).sum())
            if have != want:
                raise ValueError(f"the pair's {what} tokens do not survive truncation ({have} of {want} kept)")
        mm = enc.get("mm_token_type_ids")
        if mm is not None:
            mm = mm[0]
            if len(tids) != len(ids):  # truncation only drops text tokens: rebuild from the ids
                mm = torch.zeros_like(input_ids, dtype=mm.dtype)
                mm[input_ids == self.config.image_token_id] = 1
                mm[input_ids == self.config.video_token_id] = 2
        p = PairInputs(
            input_ids=input_ids,
            position_ids=None,
            pixel_values=enc.get("pixel_values"),
            image_grid_thw=enc.get("image_grid_thw"),
            pixel_values_videos=enc.get("pixel_values_videos"),
            video_grid_thw=enc.get("video_grid_thw"),
            mm_token_type_ids=mm,
        )
        p.position_ids = rope_positions(self.config, p)
        return p


def rope_positions(config, p: PairInputs) -> torch.Tensor:
    """[3, T] MRoPE positions from HF's ``get_rope_index`` (config only, no weights)."""
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel

    T = p.length
    if not p.has_vision:
        return torch.arange(T, dtype=torch.long)[None].expand(3, T).clone()
    mm = p.mm_token_type_ids
    if mm is None:
        mm = torch.zeros_like(p.input_ids)
        mm[p.input_ids == config.image_token_id] = 1
        mm[p.input_ids == config.video_token_id] = 2
    fake = Qwen3VLModel.__new__(Qwen3VLModel)  # methods only; get_rope_index reads nothing but config
    fake.__dict__["config"] = config
    pos, _ = fake.get_rope_index(
        input_ids=p.input_ids[None],
        mm_token_type_ids=mm[None].int(),
        image_grid_thw=p.image_grid_thw,
        video_grid_thw=p.video_grid_thw.clone() if p.video_grid_thw is not None else None,
        attention_mask=torch.ones_like(p.input_ids)[None],
    )
    return pos[:, 0].long()


def model_path() -> str:
    """Local snapshot path of the checkpoint (offline)."""
    from huggingface_hub import snapshot_download

    return os.environ.get("QWEN3VL_RERANKER_PATH") or snapshot_download(
        os.environ.get("HF_MODEL", HF_MODEL), revision=os.environ.get("HF_REVISION", HF_REVISION)
    )
