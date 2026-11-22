# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The contract between the HTTP server and a per-die scoring engine.

The server owns request parsing, host preprocessing (``PairFormatter``: chat template, tokenization, image / video
pixel prep), dispatch across dies and response shaping. Each die runs one ``ScoringEngine``; every call on an engine
happens on that engine's own worker thread (engines are not thread safe).

``FakeEngine`` scores deterministically on host without a device: the server tests use it.
"""
from __future__ import annotations

import hashlib
from typing import Protocol, Sequence

from models.autoports.qwen_qwen3_vl_reranker_8b.tt.inputs import PairInputs


class ScoringEngine(Protocol):
    die: int  # index of the die this engine owns (0 for a one-die server)

    def max_tokens(self) -> int:
        """Longest pair (template + query + document + visual tokens) this engine can score."""

    def warm(self) -> dict:
        """Compile and capture everything; returns a summary for /health (buckets, traces, seconds)."""

    def score(self, pairs: Sequence[PairInputs]) -> list[float]:
        """Raw scores (logit_yes - logit_no) for ``pairs`` in order. The engine batches and buckets internally.
        Pairs in one call usually share a query (one rerank request); pairs longer than ``max_tokens`` never arrive
        here (the server rejects them)."""

    def stats(self) -> dict:
        """Counters for /health (pairs scored, passes, trace replays ...)."""

    def close(self) -> None:
        ...


class FakeEngine:
    """Device-free stand-in: score = deterministic function of the token ids (and a bonus when the document shares
    words with the query), so tests can check ordering, batching and dispatch without hardware."""

    def __init__(self, die: int = 0, max_tokens: int = 10240, delay_s: float = 0.0):
        self.die = die
        self._max = max_tokens
        self.delay_s = delay_s
        self.calls = 0
        self.pairs = 0

    def max_tokens(self) -> int:
        return self._max

    def warm(self) -> dict:
        return {"fake": True}

    def score(self, pairs):
        import time

        if self.delay_s:
            time.sleep(self.delay_s * len(pairs))
        self.calls += 1
        self.pairs += len(pairs)
        out = []
        for p in pairs:
            h = hashlib.sha256(p.input_ids.numpy().tobytes()).digest()
            out.append((int.from_bytes(h[:4], "little") / 2**32) * 8.0 - 6.0)
        return out

    def stats(self) -> dict:
        return {"calls": self.calls, "pairs": self.pairs}

    def close(self) -> None:
        pass
