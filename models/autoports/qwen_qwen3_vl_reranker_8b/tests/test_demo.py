# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Demo page tests against FakeEngines (no device). Skipped when gradio is not installed.

Run: pytest -q models/autoports/qwen_qwen3_vl_reranker_8b/tests/test_demo.py
"""
from __future__ import annotations

import asyncio
import os
import re
import subprocess
import sys

import pytest

os.environ.setdefault("HF_HUB_OFFLINE", "1")
gr = pytest.importorskip("gradio")

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from PIL import Image  # noqa: E402

from models.autoports.qwen_qwen3_vl_reranker_8b.demo.library import Library  # noqa: E402
from models.autoports.qwen_qwen3_vl_reranker_8b.demo.ui import MAX_CANDIDATES, Handlers  # noqa: E402

PKG = "models.autoports.qwen_qwen3_vl_reranker_8b"


@pytest.fixture(scope="module")
def service():
    keys = {"QWEN3VL_RERANKER_FAKE": "1", "QWEN3VL_RERANKER_MESH_SHAPE": "1x2"}
    old = {k: os.environ.get(k) for k in keys}
    os.environ.update(keys)
    from models.autoports.qwen_qwen3_vl_reranker_8b.server.app import build_service

    svc = build_service()
    asyncio.run(svc.warm())
    yield svc
    svc.close()
    for k, v in old.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


@pytest.fixture(scope="module")
def library():
    return Library()


@pytest.fixture(scope="module")
def handlers(library, service):
    return Handlers(library, lambda: service)


def collect(agen) -> list:
    async def run():
        return [x async for x in agen]

    return asyncio.run(run())


def card_scores(page: str) -> list[float]:
    return [float(x) for x in re.findall(r'data-score="([-0-9.]+)"', page)]


def test_library_contents(library):
    counts = library.counts()
    assert counts == {"image": 61, "text": 42, "video": 10, "code": 29}
    assert all(e.poster and e.poster.startswith("data:image/jpeg") for e in library.entries if e.kind == "video")
    assert all(e.license and e.source_url for e in library.entries)
    licenses = (library.root / "LICENSES.md").read_text()
    for e in library.entries:
        assert f"| {e.id} |" in licenses, e.id
    assert library.examples("media") and library.examples("media", kind="image")
    assert library.examples("multilingual") and library.examples("code")


def test_library_search_text(handlers, library, service):
    pages = collect(handlers.library_search("a red double-decker bus", None, ["image", "text"], "", 7))
    assert len(pages) >= 2 and "Ranking 103" in pages[0]
    final = pages[-1]
    scores = card_scores(final)
    assert len(scores) == 7 and scores == sorted(scores, reverse=True)
    assert "103 pairs scored" in final and "pairs/s" in final and "fake engines" in final

    # the top card's score is what the service gives that (query, document) pair directly
    top_id = re.search(r'data-id="([^"]+)"', final).group(1)
    doc = library.by_id[top_id].document(library.root)
    res = asyncio.run(service.score_pairs([("a red double-decker bus", doc)]))
    assert abs(res.scores[0] - scores[0]) < 1e-5


def test_library_search_image_query_and_videos(handlers, library):
    path, _ = library.examples("media", kind="image")[0]
    img = Image.open(path)
    final = collect(handlers.library_search("", img, ["video"], "", 20))[-1]
    assert final.count("<video") == 10 and 'poster="data:image/jpeg' in final
    assert "query: image" in final and "/demo-files/videos/" in final


def test_library_search_errors(handlers, expect_error):
    with expect_error(gr.Error, ""):
        collect(handlers.library_search("  ", None, ["image"], "", 5))
    with expect_error(gr.Error, ""):
        collect(handlers.library_search("cats", None, [], "", 5))


def test_own_candidates(handlers, library, expect_error):
    values = handlers.example_candidates()
    q_text, q_image, instr, n = values[:4]
    texts, images = values[4 : 4 + MAX_CANDIDATES], values[4 + MAX_CANDIDATES :]
    assert len(images) == MAX_CANDIDATES and n == 5
    final = collect(handlers.own_candidates(q_text, q_image, "Find the beach scene.", n, *texts, *images))[-1]
    scores = card_scores(final)
    assert len(scores) == 5 and scores == sorted(scores, reverse=True)
    assert final.count("data:image/jpeg") == 2  # the two image candidates

    # image query + rows past n_rows are ignored
    q = library.by_id["img-dog-labrador"].image
    final = collect(handlers.own_candidates("", q, "", 2, "a dog", "a car", "ignored", *[""] * 7, *[None] * 10))[-1]
    assert len(card_scores(final)) == 2
    with expect_error(gr.Error, ""):
        collect(handlers.own_candidates("q", None, "", 3, *[""] * 10, *[None] * 10))


def test_about(handlers):
    page = handlers.about()
    assert "1x2" in page and "/v1/rerank" in page and "2 of 2 serving" in page


def test_mount_routes(service):
    from models.autoports.qwen_qwen3_vl_reranker_8b.demo.mount import mount_demo

    app = FastAPI()
    app.state.service = service
    mount_demo(app)
    with TestClient(app) as c:
        r = c.get("/", follow_redirects=False)
        assert r.status_code in (302, 307) and r.headers["location"].endswith("/demo/")
        assert c.get("/demo/").status_code == 200
        r = c.get("/demo-files/videos/vid-fireworks.mp4")
        assert r.status_code == 200 and r.headers["content-type"] == "video/mp4"
        assert c.get("/demo-files/LICENSES.md").status_code == 200


def test_default_server_does_not_import_gradio():
    env = {k: v for k, v in os.environ.items() if k != "QWEN3VL_RERANKER_DEMO"}
    code = f"import sys, {PKG}.server.app as a; assert 'gradio' not in sys.modules; print(len(a.app.routes))"
    subprocess.run([sys.executable, "-c", code], check=True, env=env, timeout=300)
