# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Request-side decoding of rerank inputs (text / image / video) into what ``PairFormatter`` takes.

An input is a string (text) or an object ``{text?, image?, video?}``:

* ``image``: base64, a ``data:image/...;base64,`` URL, ``{"url": ...}``, or a list of those. http(s) URLs are fetched
  only when QWEN3VL_RERANKER_ALLOW_URLS=1.
* ``video``: ``{"frames": [<image>...], "fps"?}`` (or a bare list of frames), sampled evenly down to 64 frames; or an
  encoded clip (mp4 / webm ...) as ``{"data": "<base64>", "fps"?}``, ``{"url": ...}`` or a base64 / data-URL string,
  decoded with PyAV at ``fps`` (default 1) into at most 64 frames.

Everything that is wrong with an input raises ``RequestError`` (HTTP 400) naming the offending field.
"""
from __future__ import annotations

import base64
import binascii
import io
import math
import os
import tempfile
import urllib.request

from PIL import Image

from models.autoports.qwen_qwen3_vl_reranker_8b.tt import inputs as I

MAX_MEDIA_BYTES = int(os.environ.get("QWEN3VL_RERANKER_MAX_MEDIA_BYTES", str(32 * 1024 * 1024)))
MAX_IMAGE_PIXELS = 64 * 1024 * 1024  # decoded width * height, before the processor's own resize
MAX_IMAGES_PER_INPUT = 16
MAX_INPUT_FRAMES = 1024  # frames a request may send for one video (sampled down to MAX_FRAMES)
MAX_FRAME_PIXELS = 4096 * 2160  # per video frame (the processor resizes frames to <= 768 tokens anyway)
MAX_FRAMES = I.MAX_FRAMES
URL_TIMEOUT_S = 15
INPUT_KEYS = ("text", "image", "video")


class RequestError(ValueError):
    """A client error, rendered as an OpenAI-shaped error body."""

    def __init__(self, message, param=None, status=400, code=None, kind=None):
        super().__init__(message)
        self.message = message
        self.param = param
        self.status = status
        self.code = code
        self.kind = kind or ("invalid_request_error" if status < 500 else "server_error")


def urls_allowed() -> bool:
    return os.environ.get("QWEN3VL_RERANKER_ALLOW_URLS", "0") == "1"


def strip_data_url(value: str) -> str:
    if value.startswith("data:"):
        head, sep, data = value.partition(",")
        if not sep or ";base64" not in head:
            raise RequestError("data URLs must be base64 encoded")
        return data
    return value


def fetch_url(url: str, what: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "qwen3-vl-reranker-tt-server"})
    try:
        with urllib.request.urlopen(req, timeout=URL_TIMEOUT_S) as resp:
            data = resp.read(MAX_MEDIA_BYTES + 1)
    except Exception as exc:
        raise RequestError(f"{what}: could not fetch {url}: {exc}")
    if len(data) > MAX_MEDIA_BYTES:
        raise RequestError(f"{what}: {url} is larger than {MAX_MEDIA_BYTES} bytes")
    return data


def media_bytes(value, what: str) -> bytes:
    if isinstance(value, dict) and isinstance(value.get("url"), str):
        value = value["url"]
    if not isinstance(value, str) or not value:
        raise RequestError(f"{what} must be a non-empty base64 string or data URL")
    if value.startswith(("http://", "https://")):
        if not urls_allowed():
            raise RequestError(f"{what}: URL inputs are disabled on this server; send base64 or a data URL")
        return fetch_url(value, what)
    payload = strip_data_url(value)
    if len(payload) > (MAX_MEDIA_BYTES * 4) // 3 + 4:
        raise RequestError(f"{what} is larger than {MAX_MEDIA_BYTES} bytes")
    try:
        data = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError):
        raise RequestError(f"{what} is not valid base64, a data URL, or an http(s) URL")
    if not data:
        raise RequestError(f"{what} is empty")
    if len(data) > MAX_MEDIA_BYTES:
        raise RequestError(f"{what} is larger than {MAX_MEDIA_BYTES} bytes")
    return data


def decode_image(value, what: str = "image", max_pixels: int = None) -> Image.Image:
    data = media_bytes(value, what)
    max_pixels = max_pixels or MAX_IMAGE_PIXELS
    try:
        img = Image.open(io.BytesIO(data))
        if img.width * img.height > max_pixels:
            raise RequestError(f"{what} is {img.width}x{img.height}; the limit is {max_pixels} pixels")
        img.load()
        img = img.convert("RGB")
    except RequestError:
        raise
    except Exception as exc:
        raise RequestError(f"{what} is not a decodable image: {exc}")
    if min(img.size) < 1:
        raise RequestError(f"{what} has no pixels")
    return img


def _fps(value, what):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise RequestError(f"{what}.fps must be a positive number")
    return float(value)


def decode_video_frames(frames, what: str) -> list:
    if not frames:
        raise RequestError(f"{what}.frames must be a non-empty list of images")
    if len(frames) > MAX_INPUT_FRAMES:
        raise RequestError(f"{what} has {len(frames)} frames; the limit is {MAX_INPUT_FRAMES}")
    keep = I._sample_frames(list(range(len(frames))), MAX_FRAMES)  # same even sampling as the formatter
    out = [decode_image(frames[i], f"{what}.frames[{i}]", MAX_FRAME_PIXELS) for i in keep]
    if len({f.size for f in out}) != 1:
        raise RequestError(f"{what} frames must all have the same size")
    return out


def decode_video_clip(value, fps, what: str):
    """An encoded clip -> ``I.VideoFile`` (a temporary file the formatter reads through the official reader:
    ``fps`` / ``max_frames`` sampling and real timestamps). Probed here so a bad clip is a 400, not a 500 later."""
    data = media_bytes(value, what)
    fd, path = tempfile.mkstemp(prefix="qwen3vl-rerank-", suffix=".video")
    with os.fdopen(fd, "wb") as f:
        f.write(data)
    clip = I.VideoFile(path, fps or I.FPS)
    try:
        import av

        with av.open(path) as c:
            s = c.streams.video[0]
            if s.width * s.height > MAX_FRAME_PIXELS:
                raise RequestError(f"{what} frames are {s.width}x{s.height}; the limit is {MAX_FRAME_PIXELS} pixels")
            dur = float(c.duration / 1e6) if c.duration else 0.0
            if dur > I.MAX_VIDEO_SECONDS:
                raise RequestError(f"{what} is {dur:.0f} s long; the limit is {I.MAX_VIDEO_SECONDS:.0f} s")
    except RequestError:
        raise
    except Exception as exc:
        raise RequestError(f"{what} is not a decodable video: {type(exc).__name__}: {exc}")
    return clip


def decode_video(value, what: str = "video") -> list:
    """A list of RGB PIL frames (at most MAX_FRAMES), or an ``I.VideoFile`` for an encoded clip."""
    if isinstance(value, list):
        return decode_video_frames(value, what)
    if isinstance(value, str):
        return decode_video_clip(value, None, what)
    if not isinstance(value, dict):
        raise RequestError(f"{what} must be {{'frames': [...]}}, {{'data': '<base64>'}} or a base64 string")
    bad = sorted(set(value) - {"frames", "data", "url", "fps"})
    if bad:
        raise RequestError(f"{what} has unsupported fields {bad}")
    fps = _fps(value.get("fps"), what)
    sources = [k for k in ("frames", "data", "url") if value.get(k) is not None]
    if len(sources) != 1:
        raise RequestError(f"{what} needs exactly one of 'frames', 'data' or 'url'")
    if sources[0] == "frames":
        if not isinstance(value["frames"], list):
            raise RequestError(f"{what}.frames must be a list of images")
        return decode_video_frames(value["frames"], what)  # fps only matters for encoded clips
    return decode_video_clip(value[sources[0]], fps, what)


def _multimodal() -> bool:
    from models.autoports.qwen_qwen3_vl_reranker_8b.tt.spec import current

    return current().multimodal


def decode_input(item, what: str):
    """A rerank query / document as the formatter takes it: a string, or a dict with text / PIL images / frames."""
    if isinstance(item, str):
        return item
    if not isinstance(item, dict):
        raise RequestError(f"{what} must be a string or an object with text / image / video fields", param=what)
    bad = sorted(set(item) - set(INPUT_KEYS))
    if bad:
        raise RequestError(f"{what} has unsupported fields {bad}; use text, image, video", param=what)
    out = {}
    text = item.get("text")
    if text is not None:
        if not isinstance(text, str):
            raise RequestError(f"{what}.text must be a string", param=what)
        out["text"] = text
    if not _multimodal() and (item.get("image") is not None or item.get("video") is not None):
        raise RequestError(f"{what}: this model ranks text only; images and video are not accepted", param=what)
    image = item.get("image")
    if image is not None:
        images = image if isinstance(image, list) else [image]
        if not images:
            raise RequestError(f"{what}.image must not be an empty list", param=what)
        if len(images) > MAX_IMAGES_PER_INPUT:
            raise RequestError(f"{what} has {len(images)} images; the limit is {MAX_IMAGES_PER_INPUT}", param=what)
        try:
            out["image"] = [
                decode_image(x, f"{what}.image" if len(images) == 1 else f"{what}.image[{i}]")
                for i, x in enumerate(images)
            ]
        except RequestError as exc:
            exc.param = exc.param or what
            raise
    if item.get("video") is not None:
        try:
            out["video"] = decode_video(item["video"], f"{what}.video")
        except RequestError as exc:
            exc.param = exc.param or what
            raise
    return out  # {} -> the formatter writes "NULL", as the official reranker does
