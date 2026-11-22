# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The demo media library and the scoring calls behind the demo page (no gradio import here).

Library documents are handed to the live ``RerankService`` in the same decoded form ``/v1/rerank`` produces:
a string for text and code, ``{"image": [PIL]}`` for images, ``{"video": VideoFile}`` for encoded clips (read
through the official fps / max_frames path, exactly like a base64 clip sent to the API).
"""
from __future__ import annotations

import base64
import io
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncIterator, Optional, Sequence

from PIL import Image

from models.autoports.qwen_qwen3_vl_reranker_8b.server.service import sigmoid
from models.autoports.qwen_qwen3_vl_reranker_8b.tt import inputs as I

log = logging.getLogger("qwen3vl_reranker.demo")

LIBRARY_DIR = Path(os.environ.get("QWEN3VL_RERANKER_DEMO_LIBRARY") or Path(__file__).resolve().parent / "library")
KINDS = ("image", "text", "video", "code")
POSTER_SIDE = 320


@dataclass
class Entry:
    id: str
    kind: str
    title: str
    caption: str = ""
    file: Optional[str] = None
    lang: Optional[str] = None
    text: Optional[str] = None
    license: Optional[str] = None
    author: Optional[str] = None
    source_url: Optional[str] = None
    seconds: Optional[float] = None
    poster: Optional[str] = None  # data URL (videos)
    image: Optional[Image.Image] = field(default=None, repr=False)  # decoded RGB (images)

    def document(self, root: Path):
        """The decoded document, as ``service.decode`` would produce it from an API request."""
        if self.kind == "image":
            return {"image": [self.image]}
        if self.kind == "video":
            return {"video": I.VideoFile(str(root / self.file), I.FPS, owned=False)}
        return self.text


@dataclass
class Ranked:
    entry: Entry
    score: float  # raw logit(yes) - logit(no)

    @property
    def relevance(self) -> float:
        return sigmoid(self.score)


def poster_frame(path: Path, side: int = POSTER_SIDE) -> Optional[str]:
    """A JPEG data URL of the frame about a third into the clip (PyAV), or None if it cannot be decoded."""
    try:
        import av

        with av.open(str(path)) as c:
            stream = c.streams.video[0]
            dur = float(stream.duration * stream.time_base) if stream.duration else (c.duration or 0) / 1e6
            target = dur / 3
            if target > 0 and stream.time_base:
                c.seek(int(target / stream.time_base), stream=stream)  # nearest keyframe at or before target
            img = None
            for frame in c.decode(stream):
                img = frame.to_image()
                if frame.pts is None or float(frame.pts * stream.time_base) >= target:
                    break
        if img is None:
            return None
        img = img.convert("RGB")
        img.thumbnail((side, side))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=80)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception as exc:
        log.warning("no poster for %s: %r", path, exc)
        return None


class Library:
    """Loaded once at mount: decoded images, code text, video posters, example queries."""

    def __init__(self, root: Path = LIBRARY_DIR):
        self.root = Path(root)
        t0 = time.perf_counter()
        manifest = json.loads((self.root / "manifest.json").read_text())
        self.entries: list[Entry] = []
        for it in manifest["items"]:
            if it["kind"] not in KINDS:
                continue
            e = Entry(
                id=it["id"],
                kind=it["kind"],
                title=it.get("title", it["id"]),
                caption=it.get("caption", ""),
                file=it.get("file"),
                lang=it.get("lang"),
                text=it.get("text"),
                license=it.get("license"),
                author=it.get("author"),
                source_url=it.get("source_url"),
                seconds=it.get("seconds"),
            )
            if e.kind == "image":
                with Image.open(self.root / e.file) as img:
                    e.image = img.convert("RGB")
            elif e.kind == "code" and e.text is None:
                e.text = (self.root / e.file).read_text(encoding="utf-8")
            elif e.kind == "video":
                e.poster = poster_frame(self.root / e.file)
            self.entries.append(e)
        self.by_id = {e.id: e for e in self.entries}
        self.query_media = {q["file"]: q for q in manifest.get("query_media", []) if q.get("kind") == "image"}
        try:
            self.queries = json.loads((self.root / "queries.json").read_text()).get("queries", [])
        except (OSError, ValueError):
            self.queries = []
        self.load_s = round(time.perf_counter() - t0, 2)

    def counts(self) -> dict:
        out = {k: 0 for k in KINDS}
        for e in self.entries:
            out[e.kind] += 1
        return out

    def select(self, kinds: Sequence[str]) -> list[Entry]:
        kinds = set(kinds)
        return [e for e in self.entries if e.kind in kinds]

    def examples(self, tab: str, kind: str = "text", n: int = 8) -> list[tuple[str, list[str]]]:
        """(query, kinds to rank) from queries.json; the kinds are those of the expected documents. Image queries
        come back as absolute paths."""
        out = []
        for q in self.queries:
            if q.get("tab") != tab or q.get("query_kind") != kind:
                continue
            kinds = sorted({self.by_id[i].kind for i in q.get("expect", []) if i in self.by_id}, key=KINDS.index)
            value = q["query"]
            if kind == "image":
                if not (self.root / value).is_file():
                    continue
                value = str(self.root / value)
            if tab == "media" and set(kinds) <= {"image", "text"}:
                kinds = ["image", "text"]  # the page default: rank images and texts together
            out.append((value, kinds or ["image", "text"]))
        return out[:n]


def make_query(text: Optional[str], image: Optional[Image.Image]):
    """The decoded query: a string, or ``{"text"?, "image": [PIL]}``. None if both are empty."""
    text = (text or "").strip()
    if image is not None:
        q = {"image": [image.convert("RGB")]}
        if text:
            q["text"] = text
        return q
    return text or None


def chunk_size(service) -> int:
    """Documents per progress step: 16 per die (a step is ~1-3 s on silicon; the service fans it out over dies)."""
    return 16 * max(1, len(service.workers))


async def rank(
    service, query, docs: Sequence, instruction: Optional[str] = None, step: Optional[int] = None
) -> AsyncIterator[tuple[int, list[float]]]:
    """Score (query, doc) for every doc through ``service.score_pairs`` (the ``/v1/rerank`` path), in steps of
    ``step`` documents; yields (documents done, raw scores so far in doc order)."""
    instruction = (instruction or "").strip() or None
    step = step or len(docs) or 1
    scores: list[float] = []
    for start in range(0, len(docs), step):
        part = docs[start : start + step]
        res = await service.score_pairs(
            [(query, d) for d in part],
            instruction,
            labels=[f"documents[{start + i}]" for i in range(len(part))],
        )
        scores.extend(res.scores)
        yield len(scores), list(scores)
