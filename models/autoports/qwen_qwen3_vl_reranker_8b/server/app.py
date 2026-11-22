# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""OpenAI / Cohere / Jina / vLLM-style HTTP server for Qwen3-VL-Reranker-8B on Tenstorrent.

    uvicorn models.autoports.qwen_qwen3_vl_reranker_8b.server.app:app --host 0.0.0.0 --port 8000

Endpoints: POST /v1/rerank, POST /v1/score, GET /health, GET /v1/models. Startup (the lifespan) builds one scoring
engine per die and warms them all before uvicorn reports "Application startup complete".

Environment:
  QWEN3VL_RERANKER_MESH_SHAPE           1x1 | 1x2 | 1x4 (default 1x1): one engine per die
  QWEN3VL_RERANKER_FAKE=1               device-free FakeEngines (tests)
  QWEN3VL_RERANKER_FAKE_MAX_TOKENS      FakeEngine.max_tokens (default 10240)
  QWEN3VL_RERANKER_FAKE_DELAY_S         FakeEngine per-pair delay (default 0)
  QWEN3VL_RERANKER_MAX_PAIRS_PER_CALL   largest chunk handed to one engine.score call (default 64)
  QWEN3VL_RERANKER_PREPROCESS_THREADS   host preprocessing threads (default 4)
  QWEN3VL_RERANKER_ALLOW_URLS=1         allow http(s) image / video URLs (default: refused)
  QWEN3VL_RERANKER_MAX_MEDIA_BYTES      per media item (default 32 MiB)
  QWEN3VL_RERANKER_PATH                 local checkpoint dir (default: the HF cache snapshot)
  QWEN3VL_RERANKER_DEMO=1               mount the Gradio demo at /demo (/ redirects there; needs gradio)
"""
from __future__ import annotations

import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Optional

from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, ConfigDict, StrictBool, StrictInt, StrictStr
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.responses import JSONResponse

from models.autoports.qwen_qwen3_vl_reranker_8b.server.media import RequestError
from models.autoports.qwen_qwen3_vl_reranker_8b.server.service import (
    EngineError,
    RerankService,
    env_int,
    parse_mesh_shape,
    sigmoid,
)

log = logging.getLogger("qwen3vl_reranker.server")

SERVED_MODEL = "Qwen/Qwen3-VL-Reranker-8B"
MODEL_ALIASES = (SERVED_MODEL, "Qwen3-VL-Reranker-8B", "qwen3-vl-reranker-8b")
MAX_DOCUMENTS = 1000
MAX_SCORE_PAIRS = 1000


# ---- service factory ----------------------------------------------------------------------------------------------
_FORMATTER = None


def build_formatter():
    """One PairFormatter per process (loading the processor takes seconds; it is reused across lifespans)."""
    global _FORMATTER
    if _FORMATTER is None:
        from models.autoports.qwen_qwen3_vl_reranker_8b.tt.inputs import PairFormatter, model_path

        _FORMATTER = PairFormatter(model_path())
    return _FORMATTER


def build_service() -> RerankService:
    mesh_shape = parse_mesh_shape(os.environ.get("QWEN3VL_RERANKER_MESH_SHAPE", "1x1"))
    t0 = time.perf_counter()
    formatter = build_formatter()
    fake = os.environ.get("QWEN3VL_RERANKER_FAKE", "0") == "1"
    on_close = None
    if fake:
        from models.autoports.qwen_qwen3_vl_reranker_8b.tt.engine_api import FakeEngine

        max_tokens = env_int("QWEN3VL_RERANKER_FAKE_MAX_TOKENS", 10240)
        delay = float(os.environ.get("QWEN3VL_RERANKER_FAKE_DELAY_S", "0") or 0)
        engines = [
            FakeEngine(die=i, max_tokens=max_tokens, delay_s=delay) for i in range(mesh_shape[0] * mesh_shape[1])
        ]
    else:
        from models.autoports.qwen_qwen3_vl_reranker_8b.tt import engine as tt_engine

        engines = tt_engine.build_engines(mesh_shape)
        on_close = getattr(tt_engine, "close_engines", None)  # optional: release the shared mesh after the dies
    service = RerankService(
        engines,
        formatter,
        preprocess_threads=env_int("QWEN3VL_RERANKER_PREPROCESS_THREADS", 4),
        max_pairs_per_call=env_int("QWEN3VL_RERANKER_MAX_PAIRS_PER_CALL", 64),
        mesh_shape=mesh_shape,
        on_close=on_close,
        fake=fake,
    )
    service.startup_s["build"] = round(time.perf_counter() - t0, 2)
    return service


@asynccontextmanager
async def lifespan(app):
    service = build_service()
    try:
        await service.warm()
    except BaseException:
        await run_in_threadpool(service.close)
        raise
    app.state.service = service
    log.warning(
        "qwen3-vl-reranker ready: mesh=%s dies=%d max_tokens=%d fake=%s startup=%s",
        os.environ.get("QWEN3VL_RERANKER_MESH_SHAPE", "1x1"),
        len(service.workers),
        service.max_tokens,
        service.fake,
        service.startup_s,
    )
    try:
        yield
    finally:
        app.state.service = None
        await run_in_threadpool(service.close)


app = FastAPI(title="Qwen3-VL-Reranker-8B on Tenstorrent", lifespan=lifespan)


# ---- errors (OpenAI shape) ----------------------------------------------------------------------------------------
def error_response(status, message, param=None, code=None, kind="invalid_request_error"):
    return JSONResponse(
        status_code=status, content={"error": {"message": message, "type": kind, "param": param, "code": code}}
    )


MAX_BODY_BYTES = int(os.environ.get("QWEN3VL_RERANKER_MAX_BODY_BYTES", str(256 * 1024 * 1024)))


@app.middleware("http")
async def limit_body_size(request: Request, call_next):
    """Refuse an oversized body from its Content-Length before reading it (media arrives base64 in JSON)."""
    length = request.headers.get("content-length")
    if length is not None and length.isdigit() and int(length) > MAX_BODY_BYTES:
        return error_response(
            413, f"request body is {int(length)} bytes; the limit is {MAX_BODY_BYTES}", code="body_too_large"
        )
    return await call_next(request)


@app.exception_handler(RequestError)
async def request_error_handler(request: Request, exc: RequestError):
    return error_response(exc.status, exc.message, exc.param, exc.code, exc.kind)


@app.exception_handler(EngineError)
async def engine_error_handler(request: Request, exc: EngineError):
    return error_response(500, str(exc), code="engine_error", kind="server_error")


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    parts, param = [], None
    for e in exc.errors():
        loc = ".".join(str(x) for x in e.get("loc", ()) if x != "body")
        param = param or (loc or None)
        parts.append(f"{loc}: {e.get('msg')}" if loc else str(e.get("msg")))
    return error_response(400, "; ".join(parts) or "invalid request", param=param)


@app.exception_handler(StarletteHTTPException)
async def http_error_handler(request: Request, exc: StarletteHTTPException):
    kind = {404: "not_found_error", 405: "invalid_request_error"}.get(exc.status_code, "error")
    return error_response(exc.status_code, str(exc.detail), kind=kind)


@app.exception_handler(Exception)
async def internal_error_handler(request: Request, exc: Exception):
    log.exception("unhandled error")
    return error_response(500, f"internal error: {type(exc).__name__}: {exc}", kind="server_error")


# ---- helpers ------------------------------------------------------------------------------------------------------
def service_or_503(request: Request) -> RerankService:
    service = getattr(request.app.state, "service", None)
    if service is None or not service.ready:
        raise RequestError("model is not loaded yet", status=503, code="not_ready", kind="server_error")
    if not any(w.alive() for w in service.workers):  # degraded (some dies down) still serves; /health says 503
        raise RequestError("no die is serving", status=503, code="worker_dead", kind="server_error")
    return service


def check_model(model: Optional[str]):
    if model is not None and model not in MODEL_ALIASES:
        raise RequestError(f"model {model!r} is not served; use {SERVED_MODEL!r}", "model", 404, "model_not_found")


def check_input(value, what: str):
    if value is None:
        raise RequestError(f"{what} is required", param=what)
    if not isinstance(value, (str, dict)):
        raise RequestError(f"{what} must be a string or an object with text / image / video fields", param=what)


def usage(tokens: int) -> dict:
    return {"prompt_tokens": tokens, "total_tokens": tokens}


# ---- request bodies -----------------------------------------------------------------------------------------------
class RerankRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    query: Any
    documents: Any
    model: Optional[StrictStr] = None
    top_n: Optional[StrictInt] = None
    return_documents: StrictBool = False
    instruction: Optional[StrictStr] = None
    max_tokens_per_doc: Optional[StrictInt] = None  # validated, then ignored: the official truncation applies
    raw_scores: StrictBool = False


class ScoreRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    text_1: Any
    text_2: Any
    model: Optional[StrictStr] = None
    instruction: Optional[StrictStr] = None
    raw_scores: StrictBool = False


# ---- endpoints ----------------------------------------------------------------------------------------------------
@app.get("/health")
async def health(request: Request):
    service = service_or_503(request)
    body = {"status": "ok", "model": SERVED_MODEL, **service.summary()}
    if not service.alive():  # a die is down: still serving on the others, but report it so the box gets restarted
        body["status"] = "degraded"
        return JSONResponse(status_code=503, content=body)
    return body


@app.get("/v1/models")
async def models(request: Request):
    service = service_or_503(request)
    return {
        "object": "list",
        "data": [
            {
                "id": SERVED_MODEL,
                "object": "model",
                "created": int(service.started),
                "owned_by": "Qwen",
                "root": SERVED_MODEL,
                "aliases": list(MODEL_ALIASES[1:]),
                "max_model_len": service.max_tokens,
                "task": "rerank",
                "modalities": ["text", "image", "video"],
            }
        ],
    }


@app.post("/v1/rerank")
async def rerank(body: RerankRequest, request: Request):
    service = service_or_503(request)
    check_model(body.model)
    check_input(body.query, "query")
    docs = body.documents
    if not isinstance(docs, list):
        raise RequestError("documents must be a list of strings or objects", param="documents")
    if not 1 <= len(docs) <= MAX_DOCUMENTS:
        raise RequestError(f"documents must hold 1..{MAX_DOCUMENTS} items, got {len(docs)}", param="documents")
    for i, d in enumerate(docs):
        check_input(d, f"documents[{i}]")
    if body.top_n is not None and body.top_n < 1:
        raise RequestError("top_n must be a positive integer", param="top_n")
    if body.max_tokens_per_doc is not None and body.max_tokens_per_doc < 1:
        raise RequestError("max_tokens_per_doc must be a positive integer", param="max_tokens_per_doc")

    query = await service.decode(body.query, "query")
    decoded = await service.decode_many(docs, "documents")
    res = await service.score_pairs(
        [(query, d) for d in decoded], body.instruction, labels=[f"documents[{i}]" for i in range(len(docs))]
    )
    rows = []
    for i, s in enumerate(res.scores):
        row = {"index": i, "relevance_score": sigmoid(s)}
        if body.raw_scores:
            row["score"] = s
        if body.return_documents:
            row["document"] = {"text": docs[i]} if isinstance(docs[i], str) else docs[i]
        rows.append(row)
    rows.sort(key=lambda r: (-r["relevance_score"], r["index"]))
    if body.top_n is not None:
        rows = rows[: body.top_n]
    return {"id": f"rerank-{uuid.uuid4().hex}", "model": SERVED_MODEL, "results": rows, "usage": usage(res.tokens)}


def _as_inputs(value, what: str) -> list:
    items = value if isinstance(value, list) else [value]
    if not items:
        raise RequestError(f"{what} must not be an empty list", param=what)
    for i, x in enumerate(items):
        check_input(x, f"{what}[{i}]" if isinstance(value, list) else what)
    return items


@app.post("/v1/score")
async def score(body: ScoreRequest, request: Request):
    """vLLM semantics: text_1 single (or a one-item list) scores 1:N against text_2; equal-length lists score N:N."""
    service = service_or_503(request)
    check_model(body.model)
    q = _as_inputs(body.text_1, "text_1")
    d = _as_inputs(body.text_2, "text_2")
    if len(q) > 1 and len(q) != len(d):
        raise RequestError(
            f"text_1 has {len(q)} items and text_2 has {len(d)}; send one text_1 (1:N) or lists of equal length (N:N)",
            param="text_2",
        )
    if len(d) > MAX_SCORE_PAIRS:
        raise RequestError(f"at most {MAX_SCORE_PAIRS} pairs per request, got {len(d)}", param="text_2")
    qs = await service.decode_many(q, "text_1")
    ds = await service.decode_many(d, "text_2")
    one = len(qs) == 1
    pairs = [(qs[0] if one else qs[i], x) for i, x in enumerate(ds)]
    res = await service.score_pairs(
        pairs,
        body.instruction,
        labels=[f"text_2[{i}]" for i in range(len(ds))],
        group=[0 if one else i for i in range(len(ds))],
    )
    data = []
    for i, s in enumerate(res.scores):
        row = {"index": i, "object": "score", "score": sigmoid(s)}
        if body.raw_scores:
            row["raw_score"] = s
        data.append(row)
    return {
        "id": f"score-{uuid.uuid4().hex}",
        "object": "list",
        "created": int(time.time()),
        "model": SERVED_MODEL,
        "data": data,
        "usage": usage(res.tokens),
    }


# ---- optional demo page (gradio is imported only when enabled) ----------------------------------------------------
if os.environ.get("QWEN3VL_RERANKER_DEMO", "0") == "1":
    from models.autoports.qwen_qwen3_vl_reranker_8b.demo.mount import mount_demo

    mount_demo(app)
