# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Per-die scoring engines (``engine_api.ScoringEngine``) and the mesh they share.

Multi-die serving is data parallel: the server opens the whole ``(1, N)`` mesh, splits it into N one-die submeshes
and runs one full copy of the model on each, so the dies never talk to each other. Each engine:

- text-only pairs: sorted by length and packed into captured-trace passes of B pairs at bucket length T (the slot
  table ``TEXT_SLOTS``; short pairs ride many to a pass, which is what keeps a weight-bandwidth-bound die busy);
- pairs with images or video: the vision tower and one text pass per pair (MRoPE positions differ per pair).
"""
from __future__ import annotations

import os
import threading
import time
from typing import Sequence

import ttnn
from models.autoports.qwen_qwen3_vl_reranker_8b.tt.inputs import PairInputs, model_path
from models.autoports.qwen_qwen3_vl_reranker_8b.tt.runner import EagerRunner, bucket_for
from models.autoports.qwen_qwen3_vl_reranker_8b.tt.text_model import Precision, TextScorer
from models.autoports.qwen_qwen3_vl_reranker_8b.tt.traced import TracedText
from models.autoports.qwen_qwen3_vl_reranker_8b.tt.vision_tower import VisionTower
from models.autoports.qwen_qwen3_vl_reranker_8b.tt.weights import Checkpoint

# (bucket T, batch sizes captured for it). B * T <= 4096 rows per pass for the short buckets.
TEXT_SLOTS = {
    128: (1, 2, 4, 8, 16, 32),
    256: (1, 2, 4, 8, 16),
    384: (1, 2, 4, 8),
    512: (1, 2, 4, 8),
    768: (1, 2, 4),
    1024: (1, 2, 4),
    1536: (1, 2),
    2048: (1, 2),
    3072: (1,),
    4096: (1,),
    8192: (1,),
    16384: (1,),
    32768: (1,),  # the card's 32K context: long documents and HD videos (a 64-frame 720p clip is ~23K tokens)
}


def _cap_slots(slots: dict) -> dict:
    """Drop the buckets beyond what the served checkpoint accepts (its spec's ``max_tokens``)."""
    from models.autoports.qwen_qwen3_vl_reranker_8b.tt.spec import current

    cap = current().max_tokens
    keep = {T: Bs for T, Bs in slots.items() if T <= cap}
    above = [T for T in sorted(slots) if T > cap]
    if above and max(keep, default=0) < cap:  # the smallest bucket that still covers the cap
        keep[above[0]] = slots[above[0]]
    return keep


TEXT_SLOTS = _cap_slots(TEXT_SLOTS)
TEXT_BUCKETS = tuple(sorted(TEXT_SLOTS))
MAX_TOKENS = TEXT_BUCKETS[-1]
MAX_IMAGE_PATCHES = 7200  # official per-image budget: 1800 merged tokens
L1_SMALL_SIZE = 32768
TRACE_REGION_SIZE = int(os.environ.get("QWEN3VL_RERANKER_TRACE_REGION", str(2_000_000_000)))


# pass cost model for the planner, fitted to traced passes on one die (B=1/4 at T=512, B=1 at T=4096):
# ~15 ms per pass on top of ~1/11000 s per padded row
PASS_FIXED_S = 0.015
ROW_S = 1.0 / 11000


def _pass_cost(B: int, T: int) -> float:
    return PASS_FIXED_S + B * T * ROW_S


def plan_text_passes(lengths: list[int]) -> list[tuple[int, int, list[int]]]:
    """Group pair indices into (B, T, indices) passes over the captured slots, minimising the modelled time.

    Pairs sorted by length; a pass is a contiguous run whose longest pair sets the bucket T, and whose size rounds up
    to the smallest captured B (dummy rows fill the rest). Dynamic program over the sorted order: O(n^2)."""
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    n = len(order)
    best = [0.0] + [float("inf")] * n
    choice = [None] * (n + 1)
    for j in range(1, n + 1):
        T = bucket_for(lengths[order[j - 1]], TEXT_BUCKETS)
        Bs = TEXT_SLOTS[T]
        for i in range(j - 1, -1, -1):
            cnt = j - i
            if cnt > Bs[-1]:
                break
            B = min(b for b in Bs if b >= cnt)
            c = best[i] + _pass_cost(B, T)
            if c < best[j]:
                best[j], choice[j] = c, (i, B, T)
    passes = []
    j = n
    while j > 0:
        i, B, T = choice[j]
        passes.append((B, T, order[i:j]))
        j = i
    return passes[::-1]


class _cache_lock:
    """Exclusive ``flock`` on the weight cache dir (shared by every die process of a server)."""

    def __enter__(self):
        import fcntl

        from models.autoports.qwen_qwen3_vl_reranker_8b.tt.weights import cache_dir

        self.f = open(cache_dir() / ".build.lock", "w")
        fcntl.flock(self.f, fcntl.LOCK_EX)
        return self

    def __exit__(self, *exc):
        import fcntl

        fcntl.flock(self.f, fcntl.LOCK_UN)
        self.f.close()


class DieEngine:
    def __init__(self, mesh, die: int, path: str, formatter_config, precision: Precision | None = None, traced=True):
        self.mesh = mesh
        self.die = die
        self.path = path
        self.cfg = formatter_config
        self.precision = precision
        self.traced_enabled = traced
        self.scorer = None
        self.vision = None
        self.traced = None
        self.counters = {"pairs": 0, "text_passes": 0, "vision_pairs": 0, "busy_s": 0.0}

    def max_tokens(self) -> int:
        return MAX_TOKENS

    def warm(self) -> dict:
        t0 = time.time()
        ckpt = Checkpoint(self.path)
        # One die at a time builds the model: on a cold cache the first converts and writes the weight files, and the
        # others (die processes share the cache dir) then read them instead of writing the same files concurrently.
        with _cache_lock():
            self.scorer = TextScorer(self.mesh, ckpt, self.cfg, precision=self.precision)
            if getattr(self.cfg, "vision_config", None) is not None:
                self.vision = VisionTower(self.mesh, ckpt, self.cfg, self.path)
        self.eager = EagerRunner(self.scorer, self.vision, buckets=TEXT_BUCKETS)
        t1 = time.time()
        shapes = [(B, T) for T, Bs in TEXT_SLOTS.items() for B in Bs]
        if self.traced_enabled:
            self.traced = TracedText(self.scorer, shapes)
        t2 = time.time()
        if self.vision is not None:
            self.warm_vision()
        return {
            "die": self.die,
            "load_s": round(t1 - t0, 1),
            "capture_s": round(t2 - t1, 1),
            "vision_warm_s": round(time.time() - t2, 1),
            "text_slots": len(shapes) if self.traced_enabled else 0,
            "buckets": list(TEXT_BUCKETS),
        }

    def warm_vision(self):
        """Compile the tower for every padded patch bucket an image can land in (images up to the official
        1800-token budget = 7200 patches -> padded 8192)."""
        import torch

        from models.autoports.qwen_qwen3_vl_reranker_8b.tt.vision_tower import padded_buckets

        for padded in padded_buckets(MAX_IMAGE_PATCHES):
            n = padded - 32 * 8
            g = torch.tensor([[1, 32, n // 32]])
            tabs = self.vision.tables(torch.randn(n, 1536), g)
            tabs.deallocate()
        ttnn.synchronize_device(self.mesh)

    def score(self, pairs: Sequence[PairInputs]) -> list[float]:
        t0 = time.time()
        out = [0.0] * len(pairs)
        text = [i for i, p in enumerate(pairs) if not p.has_vision]
        vis = [i for i, p in enumerate(pairs) if p.has_vision]
        for B, T, idx in plan_text_passes([pairs[i].length for i in text]):
            chosen = [pairs[text[j]] for j in idx]
            if self.traced is not None:
                s = self.traced.score(chosen, B, T)
            else:
                s = self.eager.score_text(chosen, T)
            for j, v in zip(idx, s):
                out[text[j]] = v
            self.counters["text_passes"] += 1
        cache = {}
        try:
            for i in vis:
                out[i] = self.eager.score_pair(pairs[i], cache=cache)
                self.counters["vision_pairs"] += 1
        finally:
            if self.vision is not None:
                self.vision.free_cache(cache)
        self.counters["pairs"] += len(pairs)
        self.counters["busy_s"] += time.time() - t0
        return out

    def stats(self) -> dict:
        d = dict(self.counters)
        d["busy_s"] = round(d["busy_s"], 2)
        if self.traced is not None:
            d["trace_replays"] = sum(s.replays for s in self.traced.slots.values())
        return d

    def close(self) -> None:
        if self.traced is not None:
            self.traced.release()
            self.traced = None


_MESH = {"parent": None, "subs": None, "lock": threading.Lock()}


def open_dies(mesh_shape: tuple[int, int]):
    """Open the mesh and return one 1x1 submesh per die (the parent itself for a 1x1 mesh)."""
    rows, cols = mesh_shape
    n = rows * cols
    if n == 1:
        os.environ.setdefault(
            "TT_MESH_GRAPH_DESC_PATH",
            os.path.join(
                os.environ.get("TT_METAL_HOME", "/opt/tt-metal"),
                "tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto",
            ),
        )
    parent = ttnn.open_mesh_device(
        ttnn.MeshShape(rows, cols), l1_small_size=L1_SMALL_SIZE, trace_region_size=TRACE_REGION_SIZE
    )
    subs = [parent] if n == 1 else list(parent.create_submeshes(ttnn.MeshShape(1, 1)))
    _MESH["parent"], _MESH["subs"] = parent, subs
    return subs


def build_engines(mesh_shape: tuple[int, int], traced: bool = True) -> list:
    """One engine per die, each in its own process (``proc_engine``) so the server's HTTP and preprocessing
    threads never share a GIL with a die. ``QWEN3VL_RERANKER_DIE_PROCS=0``: engines in this process (threads over
    submeshes of one mesh, GIL bound; tests and scripts)."""
    n = mesh_shape[0] * mesh_shape[1]
    in_worker = os.environ.get("QWEN3VL_RERANKER_IN_DIE_PROC") == "1"
    if not in_worker and os.environ.get("QWEN3VL_RERANKER_DIE_PROCS", "1") == "1":
        from models.autoports.qwen_qwen3_vl_reranker_8b.tt.proc_engine import build_proc_engines

        return build_proc_engines(n, traced=traced)
    path = model_path()
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(path)
    subs = open_dies(mesh_shape)
    return [DieEngine(m, i, path, cfg, traced=traced) for i, m in enumerate(subs)]


def close_engines(engines) -> None:
    for e in engines:
        try:
            e.close()
        except Exception:
            pass
    parent = _MESH["parent"]
    if parent is not None:
        subs = _MESH["subs"] or []
        if len(subs) > 1:
            for s in subs:
                try:
                    ttnn.close_mesh_device(s)
                except Exception:
                    pass
        ttnn.close_mesh_device(parent)
        _MESH["parent"] = _MESH["subs"] = None
