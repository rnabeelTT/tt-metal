# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host side of the reranker server: preprocessing, dispatch across dies, gather.

``RerankService`` owns one ``PairFormatter`` (plus private copies of its processor for the preprocessing threads:
HF fast tokenizers are not safe to share across threads) and one ``EngineWorker`` per die. Every call on an engine
(``max_tokens``, ``warm``, ``score``, ``stats``, ``close``) runs on that engine's own worker thread.

A request is a list of (query, document) pairs. They are decoded and formatted on the host pool (never on the event
loop), checked against ``max_tokens``, sorted by (query, length) so similar pairs share a call, cut into chunks of at
most ``max_pairs_per_call`` (and into one chunk per idle die, so an idle server fans one request out across all
dies), and each chunk goes to the die with the least outstanding work (queued + running tokens). Chunks are
gathered back into request order.
"""
from __future__ import annotations

import asyncio
import copy
import logging
import math
import os
import queue
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

from models.autoports.qwen_qwen3_vl_reranker_8b.server.media import RequestError, decode_input

log = logging.getLogger("qwen3vl_reranker.server")


class EngineError(RuntimeError):
    """A scoring engine failed while serving a request (HTTP 500 for that request only)."""

    def __init__(self, die, exc):
        super().__init__(f"scoring failed on die {die}: {type(exc).__name__}: {exc}")
        self.die = die


def parse_mesh_shape(text: str) -> tuple[int, int]:
    parts = [p.strip() for p in text.lower().replace(",", "x").split("x")]
    if len(parts) != 2 or not all(p.isdigit() and int(p) > 0 for p in parts):
        raise ValueError(f"QWEN3VL_RERANKER_MESH_SHAPE must look like 1x1, 1x2 or 1x4; got {text!r}")
    return int(parts[0]), int(parts[1])


def sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


@dataclass
class _Job:
    fn: Callable[[Any], Any]
    tokens: int = 0
    pairs: int = 0
    future: Future = field(default_factory=Future)


class EngineWorker:
    """One die: a FIFO of jobs run on a dedicated thread. A failing job fails its future, never the thread."""

    def __init__(self, engine, index: int):
        self.engine = engine
        self.die = getattr(engine, "die", index)
        self.q: "queue.Queue[Optional[_Job]]" = queue.Queue()
        self.lock = threading.Lock()
        self.outstanding_tokens = 0
        self.outstanding_pairs = 0
        self.max_tokens: Optional[int] = None
        self.warm_summary: Optional[dict] = None
        self.warm_s: Optional[float] = None
        self.engine_stats: dict = {}
        self.counters = {"calls": 0, "pairs": 0, "tokens": 0, "errors": 0, "busy_s": 0.0, "max_call_pairs": 0}
        self.thread = threading.Thread(target=self._loop, name=f"qwen3vl-reranker-die{self.die}", daemon=True)
        self.thread.start()

    # ---- submission (any thread) ------------------------------------------------------------------------------
    def submit(self, fn, tokens: int = 0, pairs: int = 0) -> Future:
        job = _Job(fn, tokens, pairs)
        with self.lock:
            self.outstanding_tokens += tokens
            self.outstanding_pairs += pairs
        self.q.put(job)
        return job.future

    def score(self, pairs: Sequence) -> Future:
        return self.submit(lambda e: self._score(e, pairs), sum(p.length for p in pairs), len(pairs))

    def warm(self) -> Future:
        def run(e):
            t0 = time.perf_counter()
            self.max_tokens = int(e.max_tokens())
            self.warm_summary = e.warm()
            self.warm_s = round(time.perf_counter() - t0, 2)
            return self.warm_summary

        return self.submit(run)

    def alive(self) -> bool:
        """The worker thread runs and, when the engine can tell (a die process), the die is up."""
        engine_alive = getattr(self.engine, "alive", None)
        return self.thread.is_alive() and (engine_alive is None or bool(engine_alive()))

    def load(self) -> tuple:
        with self.lock:
            return (self.outstanding_tokens, self.outstanding_pairs)

    def close(self, timeout: float = 120.0) -> None:
        if not self.alive():
            return
        done = self.submit(lambda e: e.close())
        self.q.put(None)
        self.thread.join(timeout)
        if done.done() and done.exception() is not None:
            log.warning("die %s: engine.close() failed: %r", self.die, done.exception())

    # ---- worker thread ----------------------------------------------------------------------------------------
    def _score(self, engine, pairs):
        t0 = time.perf_counter()
        scores = [s if hasattr(s, "__len__") else float(s) for s in engine.score(pairs)]  # embeddings are vectors
        if len(scores) != len(pairs):
            raise RuntimeError(f"engine returned {len(scores)} scores for {len(pairs)} pairs")
        c = self.counters
        c["calls"] += 1
        c["pairs"] += len(pairs)
        c["tokens"] += sum(p.length for p in pairs)
        c["busy_s"] += time.perf_counter() - t0
        c["max_call_pairs"] = max(c["max_call_pairs"], len(pairs))
        return scores

    def _loop(self):
        while True:
            job = self.q.get()
            if job is None:
                return
            try:
                if job.future.set_running_or_notify_cancel():
                    try:
                        job.future.set_result(job.fn(self.engine))
                    except BaseException as exc:  # noqa: B036 -- the thread must survive anything the engine does
                        self.counters["errors"] += 1
                        log.exception("die %s: engine job failed", self.die)
                        job.future.set_exception(exc)
                try:
                    self.engine_stats = dict(self.engine.stats())
                except Exception as exc:
                    self.engine_stats = {"error": repr(exc)}
            finally:
                with self.lock:
                    self.outstanding_tokens -= job.tokens
                    self.outstanding_pairs -= job.pairs

    def summary(self) -> dict:
        tokens, pairs = self.load()
        c = dict(self.counters)
        c["busy_s"] = round(c["busy_s"], 3)
        return {
            "die": self.die,
            "alive": self.alive(),
            "max_tokens": self.max_tokens,
            "queue_depth": self.q.qsize(),
            "outstanding_pairs": pairs,
            "outstanding_tokens": tokens,
            "warm_s": self.warm_s,
            "warm": self.warm_summary,
            **c,
            "engine": self.engine_stats,
        }


@dataclass
class ScoreResult:
    scores: list  # raw logit_yes - logit_no, in pair order
    tokens: int


class RerankService:
    def __init__(
        self,
        engines: Sequence,
        formatter,
        preprocess_threads: int = 4,
        max_pairs_per_call: int = 16,
        mesh_shape: Optional[tuple] = None,
        on_close: Optional[Callable[[list], None]] = None,
        fake: bool = False,
    ):
        if not engines:
            raise ValueError("RerankService needs at least one engine")
        self.formatter = formatter
        self.max_pairs_per_call = max(1, int(max_pairs_per_call))
        self.preprocess_threads = max(1, int(preprocess_threads))
        self.mesh_shape = mesh_shape
        self.fake = fake
        self.on_close = on_close
        self.engines = list(engines)
        # one formatter per preprocessing thread: the processor (fast tokenizer) is not thread safe
        self._formatters: "queue.Queue" = queue.Queue()
        self._formatters.put(formatter)
        for _ in range(self.preprocess_threads - 1):
            self._formatters.put(_clone_formatter(formatter))
        self.pool = ThreadPoolExecutor(self.preprocess_threads, thread_name_prefix="qwen3vl-reranker-pre")
        self.workers = [EngineWorker(e, i) for i, e in enumerate(self.engines)]
        self._dispatch_lock = threading.Lock()
        self.ready = False
        self.closed = False
        self.started = time.time()
        self.startup_s: dict = {}
        self.stats = {
            "requests": 0,
            "pairs": 0,
            "tokens": 0,
            "engine_calls": 0,
            "rejected_requests": 0,
            "failed_requests": 0,
            "preprocess_s": 0.0,
        }

    # ---- lifecycle --------------------------------------------------------------------------------------------
    async def warm(self) -> list:
        """Warm every die in parallel (each on its own worker thread); raises if any die fails."""
        t0 = time.perf_counter()
        out = await asyncio.gather(*(asyncio.wrap_future(w.warm()) for w in self.workers))
        self.startup_s["warm"] = round(time.perf_counter() - t0, 2)
        self.ready = True
        return out

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.ready = False
        for w in self.workers:
            w.close()
        if self.on_close is not None:
            try:
                self.on_close(self.engines)
            except Exception:
                log.exception("engine shutdown hook failed")
        self.pool.shutdown(wait=True, cancel_futures=True)

    @property
    def max_tokens(self) -> int:
        known = [w.max_tokens for w in self.workers if w.max_tokens is not None]
        return min(known) if known else 0

    def alive(self) -> bool:
        return all(w.alive() for w in self.workers)

    # ---- preprocessing (host pool) ----------------------------------------------------------------------------
    async def _in_pool(self, fn, *args):
        return await asyncio.get_running_loop().run_in_executor(self.pool, fn, *args)

    async def decode(self, item, what: str):
        """Decode one request input (base64 media, frames, clips) off the event loop."""
        if isinstance(item, str):
            return item
        return await self._in_pool(decode_input, item, what)

    async def decode_many(self, items: Sequence, what: str) -> list:
        return list(await asyncio.gather(*(self.decode(x, f"{what}[{i}]") for i, x in enumerate(items))))

    def _format(self, query, document, instruction, label):
        fmt = self._formatters.get()
        try:
            return fmt(query, document, instruction)
        except RequestError:
            raise
        except (ValueError, TypeError, RuntimeError) as exc:
            raise RequestError(f"{label}: could not prepare the pair: {exc}", param=label.split("[")[0])
        finally:
            self._formatters.put(fmt)

    # ---- scoring ----------------------------------------------------------------------------------------------
    async def score_pairs(
        self,
        pairs: Sequence[tuple],
        instruction: Optional[str] = None,
        labels: Optional[Sequence[str]] = None,
        group: Optional[Sequence[int]] = None,
    ) -> ScoreResult:
        """Raw scores for decoded (query, document) pairs. ``labels`` name each pair in errors (``documents[3]``);
        ``group`` is the query index of each pair (pairs of one group are kept together when chunking)."""
        n = len(pairs)
        labels = list(labels) if labels is not None else [f"pairs[{i}]" for i in range(n)]
        group = list(group) if group is not None else [0] * n
        try:
            t0 = time.perf_counter()
            prepared = await asyncio.gather(
                *(self._in_pool(self._format, q, d, instruction, labels[i]) for i, (q, d) in enumerate(pairs))
            )
            self.stats["preprocess_s"] += time.perf_counter() - t0
            limit = self.max_tokens
            for i, p in enumerate(prepared):
                if limit and p.length > limit:
                    raise RequestError(
                        f"{labels[i]}: the pair is {p.length} tokens (template + query + document, after the "
                        f"official {self.formatter.max_length}-token text truncation); this server's maximum is "
                        f"{limit} tokens",
                        param=labels[i].split("[")[0],
                        code="context_length_exceeded",
                    )
        except RequestError:
            self.stats["rejected_requests"] += 1
            raise
        try:
            scores = await self._dispatch(prepared, group)
        except EngineError:
            self.stats["failed_requests"] += 1
            raise
        tokens = sum(p.length for p in prepared)
        self.stats["requests"] += 1
        self.stats["pairs"] += n
        self.stats["tokens"] += tokens
        return ScoreResult(scores, tokens)

    def _chunks(self, prepared, group) -> list[list[int]]:
        n = len(prepared)
        order = sorted(range(n), key=lambda i: (group[i], prepared[i].length, i))
        idle = sum(1 for w in self.workers if w.load()[1] == 0 and w.alive())
        n_chunks = min(n, max(math.ceil(n / self.max_pairs_per_call), max(1, idle)))
        size, extra = divmod(n, n_chunks)
        chunks, start = [], 0
        for k in range(n_chunks):
            end = start + size + (1 if k < extra else 0)
            chunks.append(order[start:end])
            start = end
        return chunks

    async def _dispatch(self, prepared, group) -> list:
        with self._dispatch_lock:
            chunks = self._chunks(prepared, group)
            # heaviest chunk first, each to the die with the least outstanding work (LPT)
            chunks.sort(key=lambda c: -sum(prepared[i].length for i in c))
            live = [w for w in self.workers if w.alive()]
            if not live:
                raise RequestError("no die is serving (see /health)", status=503, code="engine_unavailable")
            planned = {id(w): list(w.load()) for w in live}
            sent = []
            for c in chunks:
                w = min(live, key=lambda w: (planned[id(w)][0], planned[id(w)][1], w.die))
                batch = [prepared[i] for i in c]
                planned[id(w)][0] += sum(p.length for p in batch)
                planned[id(w)][1] += len(batch)
                sent.append((c, w, w.score(batch)))
        self.stats["engine_calls"] += len(sent)
        results = await asyncio.gather(*(asyncio.wrap_future(f) for _, _, f in sent), return_exceptions=True)
        scores: list = [None] * len(prepared)
        for (c, w, _), r in zip(sent, results):
            if isinstance(r, BaseException):
                raise EngineError(w.die, r) from r
            for i, s in zip(c, r):
                scores[i] = s
        return scores

    # ---- reporting --------------------------------------------------------------------------------------------
    def summary(self) -> dict:
        s = dict(self.stats)
        s["preprocess_s"] = round(s["preprocess_s"], 3)
        return {
            "ready": self.ready,
            "fake_engines": self.fake,
            "mesh_shape": "x".join(map(str, self.mesh_shape)) if self.mesh_shape else None,
            "dies": len(self.workers),
            "max_tokens": self.max_tokens,
            "max_text_tokens": self.formatter.max_length,
            "max_pairs_per_call": self.max_pairs_per_call,
            "preprocess_threads": self.preprocess_threads,
            "startup_s": self.startup_s,
            "uptime_s": round(time.time() - self.started, 1),
            "stats": s,
            "engines": [w.summary() for w in self.workers],
        }


def _clone_formatter(formatter):
    """A formatter with its own processor (tokenizer) copy; config and settings are shared."""
    clone = copy.copy(formatter)
    clone.processor = copy.deepcopy(formatter.processor)
    return clone


def env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        raise ValueError(f"{name} must be an integer, got {raw!r}")
