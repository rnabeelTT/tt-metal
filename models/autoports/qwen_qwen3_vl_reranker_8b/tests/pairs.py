# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Fixed evaluation pair set: query groups over an openly licensed media library.

The library is the one bundled with ``tt-hous/embeddinggemma-2`` (``demo/library``: images, videos, multilingual
texts, code, each with a manifest entry and licence). Groups are deterministic (seeded distractors), so the HF goldens
and every TT run score the same pairs. A group is one query with its candidate documents, the unit for ranking
metrics (pairwise order agreement, NDCG).

Set QWEN3VL_RERANKER_LIBRARY to the library directory.
"""
from __future__ import annotations

import json
import os
import random
from pathlib import Path

DEFAULT_LIBRARY = (
    Path.home() / "qwen3-vl-reranker-bringup/refs/eg/code/models/autoports/google_embeddinggemma_2/demo/library"
)


def library_dir() -> Path:
    return Path(os.environ.get("QWEN3VL_RERANKER_LIBRARY", DEFAULT_LIBRARY))


def _doc(item, lib: Path, with_caption=False):
    kind = item["kind"]
    if kind == "image":
        d = {"image": str(lib / item["file"])}
        if with_caption:
            d["text"] = item["caption"]
        return d
    if kind == "video":
        return {"video": str(lib / item["file"])}
    if kind in ("text", "code"):
        text = item.get("text")
        if text is None and item.get("file"):
            text = (lib / item["file"]).read_text()
        return {"text": text}
    raise ValueError(kind)


def build_groups(lib: Path | None = None, seed: int = 0, include_video: bool = True) -> list[dict]:
    lib = lib or library_dir()
    manifest = json.loads((lib / "manifest.json").read_text())
    items = {x["id"]: x for x in manifest["items"]}
    qmedia = {x["file"]: x for x in manifest["query_media"]}
    queries = json.loads((lib / "queries.json").read_text())["queries"]
    rng = random.Random(seed)
    by_kind = {}
    for x in manifest["items"]:
        by_kind.setdefault(x["kind"], []).append(x["id"])

    def pick(pool, n, exclude):
        pool = [p for p in pool if p not in exclude]
        return rng.sample(pool, min(n, len(pool)))

    groups = []
    for qi, q in enumerate(queries):
        if q["query_kind"] == "audio" or q["tab"] in ("audio", "moments", "mrl"):
            continue
        expect = q["expect"] if isinstance(q["expect"], list) else [q["expect"]["id"]]
        expect = [e for e in expect if e in items and items[e]["kind"] != "audio"]
        if not expect:
            continue
        if q["tab"] == "code":
            cands = expect + pick(by_kind["code"], 6, expect) + pick(by_kind["text"], 1, expect)
        elif q["tab"] == "multilingual":
            cands = expect + pick(by_kind["text"], 6, expect) + pick(by_kind["image"], 1, expect)
        else:
            has_video = any(items[e]["kind"] == "video" for e in expect)
            n_vid = 1 if (include_video and has_video) else 0
            cands = (
                expect
                + pick(by_kind["image"], 5 if q["query_kind"] == "text" else 5, expect)
                + pick(by_kind["text"], 3, expect)
                + pick(by_kind["video"], n_vid, expect)
            )
            if not include_video:
                cands = [c for c in cands if items[c]["kind"] != "video"]
        if q["query_kind"] == "image":
            query = {"image": str(lib / q["query"])}
        else:
            query = {"text": q["query"]}
        groups.append(
            {
                "id": f"q{qi:02d}-{q['tab']}",
                "query": query,
                "docs": [_doc(items[c], lib) for c in cands],
                "doc_ids": cands,
                "relevant": [1 if c in expect else 0 for c in cands],
            }
        )

    # text + image documents
    imgs = by_kind["image"]
    for gi in range(3):
        chosen = rng.sample(imgs, 6)
        target = items[chosen[0]]
        groups.append(
            {
                "id": f"mm{gi}",
                "query": {"text": target["caption"]},
                "docs": [_doc(items[c], lib, with_caption=(i % 2 == 0)) for i, c in enumerate(chosen)],
                "doc_ids": chosen,
                "relevant": [1] + [0] * 5,
            }
        )

    # long text documents: concatenations of library texts (2-8K tokens)
    texts = [items[t] for t in by_kind["text"]]
    for gi, n_concat in enumerate((12, 30)):
        docs, ids = [], []
        for k in range(4):
            chunk = rng.sample(texts, n_concat)
            docs.append({"text": "\n\n".join(_doc(c, lib)["text"] for c in chunk)})
            ids.append("+".join(c["id"] for c in chunk[:3]))
        q = rng.choice(texts)
        groups.append(
            {
                "id": f"long{gi}",
                "query": {"text": f"Find the passage about {q['title'] if 'title' in q else q['id']}"},
                "docs": docs,
                "doc_ids": ids,
                "relevant": [0, 0, 0, 0],
            }
        )
    return groups


def flat_pairs(groups):
    for g in groups:
        for j, d in enumerate(g["docs"]):
            yield g["id"], j, g["query"], d
