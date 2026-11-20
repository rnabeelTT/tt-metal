# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host side of Hy4 image inputs (no ttnn: also imported by the vLLM front-end processes).

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


def image_spans(ids) -> list[tuple[int, int]]:
    """[(start, end)) of every run of image-pad tokens."""
    ids = np.asarray(ids)
    m = ids == IMAGE_PAD
    if not m.any():
        return []
    d = np.diff(np.concatenate([[0], m.astype(np.int8), [0]]))
    return list(zip(np.nonzero(d == 1)[0].tolist(), np.nonzero(d == -1)[0].tolist()))


def prompt_positions(ids, grids):
    """-> (positions int64 [3, L], delta). Image runs must match the grids in order (adjacent images need their
    start / end tokens between them, as the chat template writes them)."""
    L = len(ids)
    pos = np.zeros((3, L), dtype=np.int64)
    spans = image_spans(ids)
    assert len(spans) == len(grids), (len(spans), len(grids))
    cur, i = 0, 0
    for (a, b), g in zip(spans, grids):
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
