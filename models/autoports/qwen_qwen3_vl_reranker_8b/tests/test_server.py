# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""HTTP server tests against FakeEngines (no device): QWEN3VL_RERANKER_FAKE=1, mesh 1x4 (four fake dies).

The PairFormatter (chat template, tokenizer, image / video processor) is the real one, so the processor must be in
the local HF cache. Run: pytest -q models/autoports/qwen_qwen3_vl_reranker_8b/tests/test_server.py
"""
from __future__ import annotations

import base64
import io
import math
import os
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
from PIL import Image

os.environ.setdefault("HF_HUB_OFFLINE", "1")

from fastapi.testclient import TestClient  # noqa: E402

import models.autoports.qwen_qwen3_vl_reranker_8b.server.app as server_app  # noqa: E402
import models.autoports.qwen_qwen3_vl_reranker_8b.server.media as server_media  # noqa: E402
from models.autoports.qwen_qwen3_vl_reranker_8b.tt.engine_api import FakeEngine  # noqa: E402

MODEL = "Qwen/Qwen3-VL-Reranker-8B"
QUERY = "Which animal is known for purring?"
DOCS = [
    "Cats purr when they are content.",
    "The stock market closed higher today.",
    "Dogs bark at strangers.",
    "A recipe for sourdough bread.",
    "Lions are large cats that roar.",
]


def png_b64(color=(200, 30, 30), size=(48, 40)) -> str:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def sig(x):
    return 1.0 / (1.0 + math.exp(-x))


@pytest.fixture(scope="module")
def env():
    keys = {
        "QWEN3VL_RERANKER_FAKE": "1",
        "QWEN3VL_RERANKER_MESH_SHAPE": "1x4",
        "QWEN3VL_RERANKER_MAX_PAIRS_PER_CALL": "16",
        "QWEN3VL_RERANKER_ALLOW_URLS": "0",
    }
    old = {k: os.environ.get(k) for k in [*keys, "QWEN3VL_RERANKER_FAKE_MAX_TOKENS"]}
    os.environ.update(keys)
    os.environ.pop("QWEN3VL_RERANKER_FAKE_MAX_TOKENS", None)
    yield
    for k, v in old.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


@pytest.fixture(scope="module")
def client(env):
    with TestClient(server_app.app) as c:
        yield c


def expected_raw(client, query, doc, instruction=None):
    """What a fresh FakeEngine scores for this pair (checks the gather maps scores back to the right index)."""
    p = client.app.state.service.formatter(query, doc, instruction)
    return FakeEngine().score([p])[0]


def die_pairs(client):
    return {e["die"]: e["pairs"] for e in client.get("/health").json()["engines"]}


# ---- rerank -------------------------------------------------------------------------------------------------------
def test_rerank_sorted_and_scores_match(client):
    r = client.post("/v1/rerank", json={"model": MODEL, "query": QUERY, "documents": DOCS, "raw_scores": True})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["id"].startswith("rerank-") and body["model"] == MODEL
    res = body["results"]
    assert sorted(x["index"] for x in res) == list(range(len(DOCS)))
    rel = [x["relevance_score"] for x in res]
    assert rel == sorted(rel, reverse=True)
    for x in res:
        assert x["relevance_score"] == pytest.approx(sig(x["score"]), rel=1e-9)
        assert x["score"] == pytest.approx(expected_raw(client, QUERY, DOCS[x["index"]]), rel=1e-6)
        assert "document" not in x
    assert body["usage"]["total_tokens"] == body["usage"]["prompt_tokens"] > 50 * len(DOCS)


def test_rerank_top_n_return_documents_no_raw(client):
    full = client.post("/v1/rerank", json={"query": QUERY, "documents": DOCS}).json()["results"]
    r = client.post(
        "/v1/rerank", json={"query": QUERY, "documents": DOCS, "top_n": 2, "return_documents": True}
    ).json()["results"]
    assert len(r) == 2
    assert [x["index"] for x in r] == [x["index"] for x in full[:2]]
    for x in r:
        assert x["document"] == {"text": DOCS[x["index"]]}
        assert "score" not in x
    big = client.post("/v1/rerank", json={"query": QUERY, "documents": DOCS, "top_n": 99}).json()["results"]
    assert len(big) == len(DOCS)


def test_instruction_changes_scores(client):
    a = client.post("/v1/rerank", json={"query": QUERY, "documents": DOCS[:1], "raw_scores": True}).json()
    b = client.post(
        "/v1/rerank",
        json={"query": QUERY, "documents": DOCS[:1], "raw_scores": True, "instruction": "Find cooking recipes."},
    ).json()
    assert a["results"][0]["score"] != b["results"][0]["score"]
    assert b["results"][0]["score"] == pytest.approx(expected_raw(client, QUERY, DOCS[0], "Find cooking recipes."))


def test_rerank_image_video_and_data_url(client):
    img = png_b64()
    docs = [
        {"image": img},
        {"image": "data:image/png;base64," + png_b64((10, 200, 10)), "text": "a green square"},
        {"video": {"frames": [png_b64((i * 20, 0, 0), (32, 32)) for i in range(4)], "fps": 2}},
        {"video": [png_b64((0, 0, i * 3), (32, 32)) for i in range(100)]},  # sampled down to 64 frames
        "plain text",
        {},  # -> "NULL", as the official reranker formats an empty part
    ]
    r = client.post(
        "/v1/rerank",
        json={"query": {"text": "a red square", "image": img}, "documents": docs, "return_documents": True},
    )
    assert r.status_code == 200, r.text
    res = r.json()["results"]
    assert len(res) == len(docs)
    by_idx = {x["index"]: x for x in res}
    assert by_idx[0]["document"] == docs[0]
    assert by_idx[4]["document"] == {"text": "plain text"}
    assert all(0.0 < x["relevance_score"] < 1.0 for x in res)


def test_video_mp4_clip(client):
    av = pytest.importorskip("av")
    buf = io.BytesIO()
    with av.open(buf, mode="w", format="mp4") as c:
        s = c.add_stream("mpeg4", rate=4)
        s.width, s.height, s.pix_fmt = 32, 32, "yuv420p"
        for i in range(8):
            fr = av.VideoFrame.from_image(Image.new("RGB", (32, 32), (i * 30, 0, 0)))
            for pkt in s.encode(fr):
                c.mux(pkt)
        for pkt in s.encode():
            c.mux(pkt)
    clip = base64.b64encode(buf.getvalue()).decode()
    r = client.post("/v1/rerank", json={"query": "red", "documents": [{"video": {"data": clip, "fps": 2}}]})
    assert r.status_code == 200, r.text
    bad = client.post(
        "/v1/rerank",
        json={"query": "red", "documents": [{"video": {"data": base64.b64encode(b"not a video at all").decode()}}]},
    )
    assert bad.status_code == 400 and "documents[0].video" in bad.json()["error"]["message"]


def test_url_refused(client):
    r = client.post("/v1/rerank", json={"query": "x", "documents": [{"image": "https://example.com/cat.png"}]})
    assert r.status_code == 400
    err = r.json()["error"]
    assert err["type"] == "invalid_request_error" and "URL inputs are disabled" in err["message"]


@pytest.mark.parametrize(
    "body, needle",
    [
        ({"query": "x"}, "documents"),
        ({"documents": ["a"]}, "query"),
        ({"query": "x", "documents": []}, "1..1000"),
        ({"query": "x", "documents": ["a"] * 1001}, "1..1000"),
        ({"query": "x", "documents": "a"}, "documents must be a list"),
        ({"query": "x", "documents": [3]}, "documents[0]"),
        ({"query": "x", "documents": ["a"], "top_n": 0}, "top_n"),
        ({"query": "x", "documents": ["a"], "top_n": "2"}, "top_n"),
        ({"query": "x", "documents": ["a"], "max_tokens_per_doc": 0}, "max_tokens_per_doc"),
        ({"query": "x", "documents": [{"audio": "AAAA"}]}, "unsupported fields"),
        ({"query": "x", "documents": [{"image": "not base64!!"}]}, "not valid base64"),
        ({"query": "x", "documents": [{"image": base64.b64encode(b"hello").decode()}]}, "not a decodable image"),
        ({"query": "x", "documents": [{"image": "data:image/png,abc"}]}, "base64 encoded"),
        ({"query": "x", "documents": [{"video": {"frames": []}}]}, "non-empty"),
        ({"query": "x", "documents": [{"video": {"frames": [png_b64()], "fps": -1}}]}, "fps"),
        (
            {"query": "x", "documents": [{"video": [png_b64(size=(32, 32)), png_b64(size=(40, 32))]}]},
            "same size",
        ),
        ({"query": "x", "documents": [{"text": 5}]}, "text must be a string"),
    ],
)
def test_validation_errors(client, body, needle):
    r = client.post("/v1/rerank", json=body)
    assert r.status_code == 400, r.text
    err = r.json()["error"]
    assert set(err) == {"message", "type", "param", "code"}
    assert err["type"] == "invalid_request_error"
    assert needle in err["message"], err


def test_malformed_json(client):
    r = client.post("/v1/rerank", content=b"{not json", headers={"content-type": "application/json"})
    assert r.status_code == 400 and "error" in r.json()


def test_model_not_found(client):
    r = client.post("/v1/rerank", json={"model": "gpt-4", "query": "x", "documents": ["a"]})
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "model_not_found" and r.json()["error"]["param"] == "model"
    for alias in ("Qwen3-VL-Reranker-8B", "qwen3-vl-reranker-8b"):
        assert client.post("/v1/rerank", json={"model": alias, "query": "x", "documents": ["a"]}).status_code == 200
    assert client.post("/v1/score", json={"model": "nope", "text_1": "a", "text_2": "b"}).status_code == 404


# ---- score --------------------------------------------------------------------------------------------------------
def test_score_one_to_n(client):
    r = client.post("/v1/score", json={"model": MODEL, "text_1": QUERY, "text_2": DOCS})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["object"] == "list" and body["usage"]["total_tokens"] > 0
    assert [d["index"] for d in body["data"]] == list(range(len(DOCS)))  # request order, not sorted
    for d in body["data"]:
        assert d["object"] == "score"
        assert d["score"] == pytest.approx(sig(expected_raw(client, QUERY, DOCS[d["index"]])))
    single = client.post("/v1/score", json={"text_1": [QUERY], "text_2": DOCS[0]}).json()["data"]
    assert len(single) == 1 and single[0]["score"] == pytest.approx(body["data"][0]["score"])


def test_score_n_to_n(client):
    q = ["what purrs?", {"text": "red square", "image": png_b64()}, "bread"]
    r = client.post("/v1/score", json={"text_1": q, "text_2": DOCS[:3], "raw_scores": True})
    assert r.status_code == 200, r.text
    data = r.json()["data"]
    assert len(data) == 3
    assert data[0]["raw_score"] == pytest.approx(expected_raw(client, q[0], DOCS[0]))
    assert data[2]["raw_score"] == pytest.approx(expected_raw(client, q[2], DOCS[2]))
    bad = client.post("/v1/score", json={"text_1": ["a", "b"], "text_2": ["c", "d", "e"]})
    assert bad.status_code == 400 and bad.json()["error"]["param"] == "text_2"
    assert client.post("/v1/score", json={"text_1": [], "text_2": ["c"]}).status_code == 400
    assert client.post("/v1/score", json={"text_1": "a"}).status_code == 400


# ---- dispatch -----------------------------------------------------------------------------------------------------
def test_fan_out_across_dies(client):
    before = die_pairs(client)
    docs = [f"document number {i} about cats and dogs " * (1 + i % 3) for i in range(40)]
    r = client.post("/v1/rerank", json={"query": "cats", "documents": docs, "raw_scores": True})
    assert r.status_code == 200, r.text
    after = die_pairs(client)
    used = [d for d in after if after[d] > before.get(d, 0)]
    assert len(used) > 1, (before, after)
    assert sum(after.values()) - sum(before.values()) == 40
    h = client.get("/health").json()
    assert all(e["max_call_pairs"] <= 16 for e in h["engines"])
    for x in r.json()["results"][:5]:  # sorted-by-length chunking still maps scores back to the right document
        assert x["score"] == pytest.approx(expected_raw(client, "cats", docs[x["index"]]))


def test_concurrent_requests(client):
    def one(k):
        docs = [f"doc {k}-{j}" for j in range(1 + k % 7)]
        r = client.post("/v1/rerank", json={"query": f"query {k}", "documents": docs, "raw_scores": True})
        assert r.status_code == 200, r.text
        res = r.json()["results"]
        assert len(res) == len(docs)
        x = res[0]
        assert x["score"] == pytest.approx(expected_raw(client, f"query {k}", docs[x["index"]]))
        return len(docs)

    with ThreadPoolExecutor(8) as ex:
        n = sum(ex.map(one, range(24)))
    assert n == sum(1 + k % 7 for k in range(24))
    h = client.get("/health").json()
    assert all(e["alive"] and e["outstanding_pairs"] == 0 for e in h["engines"])


# ---- health / models ----------------------------------------------------------------------------------------------
def test_health_and_models(client):
    h = client.get("/health")
    assert h.status_code == 200
    body = h.json()
    assert body["status"] == "ok" and body["ready"] and body["dies"] == 4 and body["mesh_shape"] == "1x4"
    assert body["fake_engines"] is True and body["max_tokens"] == 10240
    assert [e["die"] for e in body["engines"]] == [0, 1, 2, 3]
    assert all(e["warm"] == {"fake": True} for e in body["engines"])
    m = client.get("/v1/models").json()
    assert m["object"] == "list" and m["data"][0]["id"] == MODEL and m["data"][0]["object"] == "model"


def test_not_ready_503(client, monkeypatch):
    monkeypatch.setattr(client.app.state, "service", None)
    for path in ("/health", "/v1/models"):
        r = client.get(path)
        assert r.status_code == 503 and r.json()["error"]["code"] == "not_ready"
    assert client.post("/v1/rerank", json={"query": "x", "documents": ["a"]}).status_code == 503


def test_unknown_route(client):
    r = client.get("/v2/nothing")
    assert r.status_code == 404 and r.json()["error"]["type"] == "not_found_error"


# ---- separate servers ---------------------------------------------------------------------------------------------
def test_context_length_exceeded(env, monkeypatch):
    monkeypatch.setenv("QWEN3VL_RERANKER_FAKE_MAX_TOKENS", "100")
    with TestClient(server_app.app) as c:
        long_doc = " ".join(["word"] * 200)
        r = c.post("/v1/rerank", json={"query": "q", "documents": ["short", long_doc, "also short"]})
        assert r.status_code == 400, r.text
        err = r.json()["error"]
        assert err["code"] == "context_length_exceeded" and err["param"] == "documents"
        assert "documents[1]" in err["message"]
        assert c.post("/v1/rerank", json={"query": "q", "documents": ["short"]}).status_code == 200
        assert c.get("/health").json()["stats"]["rejected_requests"] == 1


class FlakyEngine(FakeEngine):
    """Raises on scoring any pair whose text contains 'boom'."""

    def __init__(self, *a, formatter=None, **k):
        super().__init__(*a, **k)
        self.boom = formatter.processor.tokenizer.encode("boom")[0]

    def score(self, pairs):
        if any(self.boom in p.input_ids.tolist() for p in pairs):
            raise RuntimeError("device exploded")
        return super().score(pairs)


def test_engine_error_is_500_and_worker_survives(env, monkeypatch):
    def build():
        fmt = server_app.build_formatter()
        return server_app.RerankService([FlakyEngine(die=i, formatter=fmt) for i in range(2)], fmt, mesh_shape=(1, 2))

    monkeypatch.setattr(server_app, "build_service", build)
    with TestClient(server_app.app) as c:
        r = c.post("/v1/rerank", json={"query": "q", "documents": ["fine", "boom", "fine too"]})
        assert r.status_code == 500, r.text
        err = r.json()["error"]
        assert err["type"] == "server_error" and err["code"] == "engine_error" and "device exploded" in err["message"]
        ok = c.post("/v1/rerank", json={"query": "q", "documents": ["fine", "fine too"]})
        assert ok.status_code == 200
        h = c.get("/health").json()
        assert all(e["alive"] for e in h["engines"]) and sum(e["errors"] for e in h["engines"]) >= 1
        assert h["stats"]["failed_requests"] == 1
    assert threading.active_count() < 200


def test_long_query_keeps_image_tokens(env):
    """Truncation keeps every special token (the 14 the official reranker keeps), so an image document after a
    query longer than the 10240-token text budget still has all its image tokens."""
    fmt = server_app.build_formatter()
    img = server_media.decode_image(png_b64(size=(96, 64)))
    p = fmt({"text": "word " * 12000}, {"image": img})
    cfg = fmt.config
    want = int(p.image_grid_thw.prod(-1).sum()) // cfg.vision_config.spatial_merge_size**2
    assert int((p.input_ids == cfg.image_token_id).sum()) == want > 0
    with TestClient(server_app.app) as c:
        r = c.post("/v1/rerank", json={"query": "word " * 12000, "documents": [{"image": png_b64()}]})
    assert r.status_code in (200, 400), r.text  # 400 only as context_length_exceeded against the fake max_tokens
    if r.status_code == 400:
        assert r.json()["error"]["code"] == "context_length_exceeded"


class DeadAfterWarmEngine(FakeEngine):
    def __init__(self, *a, dead=False, **kw):
        super().__init__(*a, **kw)
        self.dead = dead

    def alive(self):
        return not self.dead


def test_dead_die_reported_and_skipped(env, monkeypatch):
    engines = []

    def build():
        fmt = server_app.build_formatter()
        engines[:] = [DeadAfterWarmEngine(die=i) for i in range(2)]
        return server_app.RerankService(engines, fmt, mesh_shape=(1, 2))

    monkeypatch.setattr(server_app, "build_service", build)
    with TestClient(server_app.app) as c:
        assert c.get("/health").status_code == 200
        engines[1].dead = True
        assert c.get("/health").status_code == 503
        before = engines[0].pairs
        r = c.post("/v1/rerank", json={"query": "q", "documents": [f"doc {i}" for i in range(12)]})
        assert r.status_code == 200, r.text
        assert engines[0].pairs - before == 12 and engines[1].pairs == 0
        engines[0].dead = True
        r = c.post("/v1/rerank", json={"query": "q", "documents": ["a", "b"]})
        assert r.status_code == 503 and r.json()["error"]["code"] == "worker_dead"
