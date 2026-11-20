# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host side of Hy4 image and video inputs (no ttnn: also imported by the vLLM front-end processes).

- preprocess(): the HF HYV4FlashImageProcessor (PIL backend) exactly: smart_resize to multiples of 32 px within
  [min_pixels, max_pixels], PIL BICUBIC, x / 255, (x - 0.5) / 0.5, row-major 16 x 16 patches flattened (c, y, x).
- expand_image_tokens(): one <image_pad> per image in a chat-templated prompt -> (gh / 2) x (gw / 2 + 1) of them
  (every merged row ends with a newline feature; the tokens are all 120120).
- prompt_positions(): HF HYV4FlashModel.get_rope_index for one sequence: text tokens 1-D; an image block of H rows x
  (W + 1) columns starting at p gets (t, h, w) = (p, p + row, p + col); the text after it continues at
  p + max(H, W + 1). -> positions [3, L] and delta = max + 1 - L (decode positions = slot + delta).
- PosMap: slot index -> the 3 rope planes (positions past the prompt: slot + delta on every plane).
"""

from __future__ import annotations

import math
import os

import numpy as np
import torch

IMAGE_START, IMAGE_END, IMAGE_PAD, IMAGE_NEWLINE = 120118, 120119, 120120, 120121
VIDEO_START, VIDEO_END, VIDEO_PAD = 120122, 120123, 120124
PATCH, MERGE = 16, 2
MIN_PIXELS, MAX_PIXELS = 4096, 4194304  # processor_config.json (size.shortest_edge / longest_edge)
MEAN = STD = 0.5


def max_pixels() -> int:
    """Serving cap on the resized image area (HY4_IMAGE_MAX_PIXELS; default the checkpoint's 2048 x 2048)."""
    return int(os.environ.get("HY4_IMAGE_MAX_PIXELS", MAX_PIXELS))


def smart_resize(height, width, factor=PATCH * MERGE, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS):
    if max(height, width) / min(height, width) > 200:
        raise ValueError(f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}")
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def grid_for(height: int, width: int, maxp: int | None = None) -> tuple[int, int, int]:
    rh, rw = smart_resize(height, width, max_pixels=maxp or max_pixels())
    return 1, rh // PATCH, rw // PATCH


def num_image_tokens(grid) -> int:
    _, gh, gw = grid
    return (gh // MERGE) * (gw // MERGE + 1)


def preprocess(img, maxp: int | None = None):
    """PIL image -> (pixel_values float32 [gh * gw, 3 * 16 * 16], grid (1, gh, gw))."""
    from PIL import Image

    img = img.convert("RGB")
    rh, rw = smart_resize(img.height, img.width, max_pixels=maxp or max_pixels())
    if (rh, rw) != (img.height, img.width):
        img = img.resize((rw, rh), resample=Image.BICUBIC)
    a = np.asarray(img, dtype=np.float32).transpose(2, 0, 1)  # [3, H, W]
    a = a / np.float32(255)
    a = (a - np.float32(MEAN)) / np.float32(STD)
    gh, gw = rh // PATCH, rw // PATCH
    p = a.reshape(3, gh, PATCH, gw, PATCH).transpose(1, 3, 0, 2, 4).reshape(gh * gw, 3 * PATCH * PATCH)
    return np.ascontiguousarray(p), (1, gh, gw)


def expand_image_tokens(ids, grids) -> list[int]:
    """A tokenized prompt with ONE <image_pad> per image -> the pad expanded to each image's token count."""
    out, k = [], 0
    for t in ids:
        if int(t) == IMAGE_PAD:
            out += [IMAGE_PAD] * num_image_tokens(grids[k])
            k += 1
        else:
            out.append(int(t))
    assert k == len(grids), (k, len(grids))
    return out


def image_spans(ids, pad: int = IMAGE_PAD) -> list[tuple[int, int]]:
    """[(start, end)) of every run of image-pad (or `pad`) tokens."""
    ids = np.asarray(ids)
    m = ids == pad
    if not m.any():
        return []
    d = np.diff(np.concatenate([[0], m.astype(np.int8), [0]]))
    return list(zip(np.nonzero(d == 1)[0].tolist(), np.nonzero(d == -1)[0].tolist()))


def prompt_positions(ids, grids, videos=()):
    """-> (positions int64 [3, L], delta). Image-pad runs match `grids` in order, video-pad runs (one per frame) the
    frames of `videos` [(T, gh, gw)] in order (adjacent runs need their start / end tokens between them, as the chat
    template / the video layout write them). Every run is laid out as an image block."""
    L = len(ids)
    pos = np.zeros((3, L), dtype=np.int64)
    spans = image_spans(ids)
    vspans = image_spans(ids, VIDEO_PAD)
    vgrids = [(1, g[1], g[2]) for g in videos for _ in range(g[0])]
    assert len(spans) == len(grids) and len(vspans) == len(vgrids), (len(spans), len(grids), len(vspans), len(vgrids))
    runs = sorted(list(zip(spans, grids)) + list(zip(vspans, vgrids)))
    cur, i = 0, 0
    for (a, b), g in runs:
        pos[:, i:a] = np.arange(cur, cur + a - i)
        cur += a - i
        H, W1 = g[1] // MERGE, g[2] // MERGE + 1
        assert b - a == H * W1 * g[0], (b - a, g)
        hh, ww = np.meshgrid(np.arange(H), np.arange(W1), indexing="ij")
        pos[0, a:b] = cur
        pos[1, a:b] = cur + hh.reshape(-1)
        pos[2, a:b] = cur + ww.reshape(-1)
        cur = cur + max(H - 1, W1 - 1) + 1
        i = b
    pos[:, i:] = np.arange(cur, cur + L - i)
    cur += L - i
    return pos, int(cur - L)


class PosMap:
    """Rope positions of a sequence whose prompt holds images: slot index s -> [3] planes."""

    def __init__(self, pos3, delta: int):
        self.pos3 = torch.as_tensor(pos3, dtype=torch.long)
        self.L, self.delta = self.pos3.shape[1], int(delta)

    def __call__(self, slots) -> torch.Tensor:
        s = torch.as_tensor(slots, dtype=torch.long).reshape(-1)
        out = (s + self.delta).reshape(1, -1).repeat(3, 1)
        inside = (s >= 0) & (s < self.L)
        if inside.any():
            out[:, inside] = self.pos3[:, s[inside]]
        return out


# ---------------------------------------------------------------------------------------------- video (Tencent's vLLM
# PR 68296 format: per frame "<{t:.1f} seconds>" <image_start> pads <image_end>, inside <video_start> .. <video_end>)
VIDEO_TS_PER_FRAME, VIDEO_TS_RESERVE, VIDEO_MAX_SIDE = 6, 1000, 1024  # processor_config image_processor section


def video_max_pixels() -> int:
    """Total pixel budget of one video (the PR reads processor_config max_pixels: 4194304 = 4096 merged cells)."""
    return int(os.environ.get("HY4_VIDEO_MAX_PIXELS", MAX_PIXELS))


def adaptive_resize(height, width, num_frames, max_pixels=None, video_max_image_size=VIDEO_MAX_SIDE,
                    ts_per_frame=VIDEO_TS_PER_FRAME, ts_reserve=VIDEO_TS_RESERVE):
    factor = PATCH * MERGE
    max_pixels = max_pixels or video_max_pixels()
    frame_tokens = height * width // factor**2 + max(height, width) // factor
    budget = max(1, max_pixels // factor**2 - ts_reserve - ts_per_frame * num_frames)
    if frame_tokens * num_frames > budget:
        scale = math.sqrt(budget / (frame_tokens * num_frames))
    elif height * width > video_max_image_size**2:
        scale = video_max_image_size / math.sqrt(height * width)
    else:
        scale = 1.0
    return (max(factor, int(height * scale) // factor * factor), max(factor, int(width * scale) // factor * factor))


VIDEO_FPS, VIDEO_MIN_FRAMES, VIDEO_MAX_FRAMES = 2.0, 2, 1024  # the PR video processor (fps: class default)


def sample_frames(total: int, fps: float, target_fps: float = VIDEO_FPS, min_frames: int = VIDEO_MIN_FRAMES,
                  max_frames: int = VIDEO_MAX_FRAMES) -> list[int]:
    """The PR's HYV4VLVideoProcessor.sample_frames (do_sample_frames: a fully decoded clip of `total` frames at `fps`
    -> target_fps frames per second within [min_frames, max_frames], segment midpoints)."""
    if total <= 0 or not fps or fps <= 0:
        raise ValueError("Video metadata must contain positive frame count and fps")
    desired = math.floor(total / fps * target_fps) if target_fps > 0 else max_frames
    n = min(total, max(min_frames, min(max_frames, desired)))
    seg = (total - 1) / n
    return [min((int(np.round(seg * i)) + int(np.round(seg * (i + 1)))) // 2, total - 1) for i in range(n)]


def video_preprocess(frames, fps: float = 2.0, frames_indices=None, max_pixels=None):
    """frames uint8 [T, H, W, 3] (already sampled) -> (pixel_values float32 [T * gh * gw, 768], grid (T, gh, gw),
    timestamps [T] seconds = frame index / fps)."""
    import torchvision.transforms.v2.functional as tvF

    v = torch.as_tensor(np.asarray(frames))
    if v.shape[-1] == 3:
        v = v.permute(0, 3, 1, 2)
    T, _, H, W = v.shape
    if T == 0 or T > VIDEO_MAX_FRAMES:
        raise ValueError(f"a video needs 1 .. {VIDEO_MAX_FRAMES} frames, got {T}")
    h, w = adaptive_resize(H, W, T, max_pixels)
    if T * h * w > (max_pixels or video_max_pixels()):
        raise ValueError("Processed video exceeds its total max_pixels budget")
    if (h, w) != (H, W):
        v = tvF.resize(v.contiguous(), [h, w], interpolation=tvF.InterpolationMode.BICUBIC, antialias=True)
    # the PR's TorchvisionBackend.rescale_and_normalize fuses x / 255 into mean / std: (x - 127.5) / 127.5 in fp32
    x = v.float()
    x = (x - torch.tensor([MEAN * 255.0] * 3).view(3, 1, 1)) / torch.tensor([STD * 255.0] * 3).view(3, 1, 1)
    gh, gw = h // PATCH, w // PATCH
    p = x.reshape(T, 3, gh, PATCH, gw, PATCH).permute(0, 2, 4, 1, 3, 5).reshape(-1, 3 * PATCH * PATCH)
    idx = list(range(T)) if frames_indices is None else list(frames_indices)
    return p.contiguous().numpy(), (T, gh, gw), [i / fps for i in idx]


def num_video_frame_tokens(grid) -> int:
    _, gh, gw = grid
    return (gh // MERGE) * (gw // MERGE + 1)


def video_frame_tokens(grid, timestamps, tokenizer) -> list[int]:
    """The tokens replacing one <video_pad> (between <video_start> and <video_end>)."""
    out = []
    n = num_video_frame_tokens(grid)
    for t in timestamps:
        out += tokenizer.encode(f"<{float(t):.1f} seconds>", add_special_tokens=False)
        out += [IMAGE_START] + [VIDEO_PAD] * n + [IMAGE_END]
    return out


def expand_video_tokens(ids, videos, tokenizer) -> list[int]:
    """One <video_pad> per video -> its frames' tokens; videos [(grid, timestamps)] in order."""
    out, k = [], 0
    for t in ids:
        if int(t) == VIDEO_PAD:
            out += video_frame_tokens(videos[k][0], videos[k][1], tokenizer)
            k += 1
        else:
            out.append(int(t))
    assert k == len(videos), (k, len(videos))
    return out
