# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Passes over the chip-sharded scorer, and the combiner that packs both models' requests into them.

A pass has one (B, T) shape for every chip, and per-chip inputs: chip ``i`` scores up to B of its lane's pairs (or
dummy rows when its lane has nothing queued). Text pairs run in captured traces, one per (B, T) slot of
``TEXT_SLOTS`` up to ``QWEN3VL_COHOST_TRACE_MAX_T``; longer buckets run eagerly. Pairs with images run eagerly, one
pair per pass, with the vision tower of the lane's model (``QWEN3VL_COHOST_VISION=1`` only).

``Lane`` is the ``ScoringEngine`` the stock ``RerankService`` drives (one per chip): its ``score`` hands the pairs to
the ``Combiner`` and waits. The combiner gathers what every lane has queued (a request split across a model's two
chips arrives as two lane calls within microseconds), plans each lane's passes, and runs the heads of all lanes'
plans as ONE pass whenever a captured slot covers them all. Device work runs through ``host.step`` (exclusive use of
the shared mesh, late compiles counted against the member).
"""
from __future__ import annotations

import logging
import os
import queue
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import torch

import ttnn
from models.autoports.qwen_qwen3_vl_reranker_8b.tt.inputs import PairInputs
from models.autoports.qwen_qwen3_vl_reranker_8b.tt.runner import HostBatch, bucket_for, visual_index

from .split import SplitScorer

log = logging.getLogger("qwen3vl_cohost")

# (bucket T, batch sizes captured). A subset of the single-chip server's table: fewer slots keep the shared trace
# region small (the LLM's traces live in the same region).
ALL_SLOTS = {
    128: (1, 2, 4, 8, 16, 32),
    256: (1, 2, 4, 8, 16),
    512: (1, 2, 4, 8),
    1024: (1, 2, 4),
    2048: (1, 2),
    4096: (1,),
    8192: (1,),
}


def _slots_from_env() -> Dict[int, tuple]:
    max_tokens = int(os.environ.get("QWEN3VL_COHOST_MAX_TOKENS", "8192"))
    keep = {T: Bs for T, Bs in ALL_SLOTS.items() if T <= max_tokens}
    if max(keep) < max_tokens:
        raise ValueError(f"QWEN3VL_COHOST_MAX_TOKENS={max_tokens} is above the largest bucket {max(ALL_SLOTS)}")
    return keep


TEXT_SLOTS = _slots_from_env()
TEXT_BUCKETS = tuple(sorted(TEXT_SLOTS))
MAX_TOKENS = TEXT_BUCKETS[-1]
TRACE_MAX_T = int(os.environ.get("QWEN3VL_COHOST_TRACE_MAX_T", "4096"))
GATHER_S = float(os.environ.get("QWEN3VL_COHOST_GATHER_MS", "3")) / 1000

PASS_FIXED_S = 0.015
ROW_S = 1.0 / 11000


def plan_text_passes(lengths: List[int]) -> List[tuple]:
    """(B, T, indices) passes over the captured slots, minimising the modelled time (the single-chip planner)."""
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    n = len(order)
    best = [0.0] + [float("inf")] * n
    choice: List[Optional[tuple]] = [None] * (n + 1)
    for j in range(1, n + 1):
        T = bucket_for(lengths[order[j - 1]], TEXT_BUCKETS)
        Bs = TEXT_SLOTS[T]
        for i in range(j - 1, -1, -1):
            cnt = j - i
            if cnt > Bs[-1]:
                break
            B = min(b for b in Bs if b >= cnt)
            c = best[i] + PASS_FIXED_S + B * T * ROW_S
            if c < best[j]:
                best[j], choice[j] = c, (i, B, T)
    passes, j = [], n
    while j > 0:
        i, B, T = choice[j]
        passes.append((B, T, order[i:j]))
        j = i
    return passes[::-1]


def covering_slot(B: int, T: int) -> Optional[tuple]:
    """The smallest captured (B', T) with B' >= B, or None."""
    Bs = [b for b in TEXT_SLOTS.get(T, ()) if b >= B]
    return (min(Bs), T) if Bs else None


def _dummy(n: int) -> PairInputs:
    ids = torch.full((n,), 11, dtype=torch.long)
    return PairInputs(input_ids=ids, position_ids=torch.arange(n)[None].expand(3, n).clone())


# ---- per-chip inputs ------------------------------------------------------------------------------------------------
def _cat(hbs: List[HostBatch], key: str) -> torch.Tensor:
    return torch.cat([getattr(hb, key) for hb in hbs], dim=0)


def build_inputs(scorer: SplitScorer, per_chip: List[List[PairInputs]], T: int, B: Optional[int] = None, device=None,
                 vis_index=None) -> dict:
    """Sharded inputs of shape (B, T): chip i gets ``per_chip[i]``, padded with dummy rows to B (a captured slot's B,
    which can be larger than any chip's pair count). ``device=None``: host tensors (for ``copy_host_to_device_tensor``
    into a slot)."""
    B = B or max(len(p) for p in per_chip)
    assert all(len(p) <= B for p in per_chip), (B, [len(p) for p in per_chip])
    hbs = [HostBatch(scorer, list(p) + [_dummy(1)] * (B - len(p)), T) for p in per_chip]
    mapper = ttnn.ShardTensorToMesh(scorer.device, dim=0)
    kw = dict(mesh_mapper=mapper, device=device)
    out = {
        "ids": ttnn.from_torch(_cat(hbs, "ids"), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, **kw),
        "cos": ttnn.from_torch(_cat(hbs, "cos"), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, **kw),
        "sin": ttnn.from_torch(_cat(hbs, "sin"), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, **kw),
        "select": ttnn.from_torch(_cat(hbs, "select"), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, **kw),
    }
    if vis_index is not None:  # one vision pair, the same on every chip
        idx = vis_index.reshape(1, -1).int().expand(len(per_chip), -1).contiguous()
        out["vis_index"] = ttnn.from_torch(idx, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, **kw)
    return out


@dataclass
class Slot:
    B: int
    T: int
    inputs: dict
    output: object = None
    trace_id: object = None
    replays: int = 0


class SplitEngine:
    """Device side: the scorer, its traced slots, the optional vision towers. Every method runs as device work."""

    def __init__(self, scorer: SplitScorer, towers: Optional[Dict[str, object]] = None):
        self.scorer = scorer
        self.device = scorer.device
        self.lanes = scorer.lanes
        self.towers = towers or {}
        self.slots: Dict[tuple, Slot] = {}
        self.counters = {"passes": 0, "combined_passes": 0, "eager_passes": 0, "vision_pairs": 0}

    def traced_shapes(self):
        return [(B, T) for T, Bs in TEXT_SLOTS.items() if T <= TRACE_MAX_T for B in Bs]

    def eager_shapes(self):
        return [(B, T) for T, Bs in TEXT_SLOTS.items() if T > TRACE_MAX_T for B in Bs]

    def _forward(self, d: dict, B: int, T: int, tables=None):
        if tables is None:
            return self.scorer.forward(d["ids"], d["cos"], d["sin"], d["select"], B, T)
        return self.scorer.forward(d["ids"], d["cos"], d["sin"], d["select"], B, T, vis_index=d["vis_index"],
                                   vis_tables=tables.all())

    @staticmethod
    def _free(*ts):
        for t in ts:
            if isinstance(t, tuple):
                for x in t:
                    ttnn.deallocate(x)
            elif t is not None:
                ttnn.deallocate(t)

    # ---- boot ------------------------------------------------------------------------------------------------------
    def compile_all(self):
        """Round 1 of the co-host boot: every program of every shape, eagerly; no trace, nothing kept."""
        t0 = time.time()
        n = len(self.lanes)
        for B, T in self.traced_shapes() + self.eager_shapes():
            d = build_inputs(self.scorer, [[_dummy(T - 1)] * B] * n, T, device=self.device)
            out = self._forward(d, B, T)
            self._free(out, *d.values())
        for task, tower in self.towers.items():
            self._warm_vision(tower)
        ttnn.synchronize_device(self.device)
        log.info("qwen3vl-cohost: compiled %d shapes in %.1f s", len(self.traced_shapes() + self.eager_shapes()),
                 time.time() - t0)

    def _warm_vision(self, tower):
        from models.autoports.qwen_qwen3_vl_reranker_8b.tt.vision_tower import padded_buckets

        for padded in padded_buckets(7200):
            npatch = padded - 32 * 8
            g = torch.tensor([[1, 32, npatch // 32]])
            tabs = tower.tables(torch.randn(npatch, 1536), g)
            tabs.deallocate()

    def capture_all(self):
        """Round 2 (the capture phase, after the LLM's captures): a trace per traced slot."""
        t0 = time.time()
        n = len(self.lanes)
        for B, T in self.traced_shapes():
            slot = Slot(B=B, T=T, inputs=build_inputs(self.scorer, [[_dummy(T - 1)] * B] * n, T, device=self.device))
            out = self._forward(slot.inputs, B, T)  # program cache hit (compiled in round 1)
            self._free(out)
            ttnn.synchronize_device(self.device)
            slot.trace_id = ttnn.begin_trace_capture(self.device, cq_id=0)
            slot.output = self._forward(slot.inputs, B, T)
            ttnn.end_trace_capture(self.device, slot.trace_id, cq_id=0)
            ttnn.synchronize_device(self.device)
            self.slots[(B, T)] = slot
        try:
            mv = ttnn.get_memory_view(self.device, ttnn.BufferType.TRACE)
            used = f"; trace region now {mv.total_bytes_allocated_per_bank * mv.num_banks / 2**20:.0f} MiB used"
        except Exception:  # noqa: BLE001 - accounting only
            used = ""
        log.info("qwen3vl-cohost: captured %d traces in %.1f s%s", len(self.slots), time.time() - t0, used)

    def release(self):
        for s in self.slots.values():
            if s.trace_id is not None:
                ttnn.release_trace(self.device, s.trace_id)
            self._free(s.output, *s.inputs.values())
        self.slots.clear()

    # ---- one pass --------------------------------------------------------------------------------------------------
    def run_text(self, per_chip: List[List[PairInputs]], B: int, T: int) -> List[list]:
        """One text pass of shape (B, T); returns each chip's results for its real pairs."""
        slot = self.slots.get((B, T))
        if slot is not None:
            host = build_inputs(self.scorer, per_chip, T, B)
            for k, v in host.items():
                ttnn.copy_host_to_device_tensor(v, slot.inputs[k])
            ttnn.execute_trace(self.device, slot.trace_id, cq_id=0, blocking=False)
            slot.replays += 1
            out, d = slot.output, None
        else:
            d = build_inputs(self.scorer, per_chip, T, B, device=self.device)
            out = self._forward(d, B, T)
            self.counters["eager_passes"] += 1
        res = [self.scorer.read_lane(out, i, len(p)) if p else [] for i, p in enumerate(per_chip)]
        if d is not None:
            self._free(out, *d.values())
        self.counters["passes"] += 1
        if sum(1 for p in per_chip if p) > 1:
            self.counters["combined_passes"] += 1
        return res

    def run_vision(self, chip: int, pair: PairInputs) -> object:
        task = self.lanes[chip]
        tower = self.towers.get(task)
        if tower is None:
            raise RuntimeError("image and video inputs are not served by this profile (QWEN3VL_COHOST_VISION=0)")
        T = bucket_for(pair.length, TEXT_BUCKETS)
        tables = tower.tables(pair.pixel_values, pair.image_grid_thw, pair.pixel_values_videos, pair.video_grid_thw)
        try:
            vi = visual_index(self.scorer, pair, tables, T)
            n = len(self.lanes)
            d = build_inputs(self.scorer, [[pair]] * n, T, 1, device=self.device, vis_index=vi)
            out = self._forward(d, 1, T, tables=tables)
            res = self.scorer.read_lane(out, chip, 1)[0]
            self._free(out, *d.values())
        finally:
            tables.deallocate()
        self.counters["vision_pairs"] += 1
        return res


# ---- the combiner ---------------------------------------------------------------------------------------------------
@dataclass
class _Job:
    chip: int
    pairs: Sequence[PairInputs]
    future: Future = field(default_factory=Future)


class Combiner:
    """One thread: gathers every lane's queued job, plans, runs combined passes through ``step(who, fn)``."""

    def __init__(self, engine: SplitEngine, step: Callable[[str, Callable], object], member_of: Dict[str, str]):
        self.engine = engine
        self.step = step                  # host.step(member_name, fn) in the co-host; a direct call in tests
        self.member_of = member_of        # task -> co-host member name (for late-compile accounting)
        self.q: "queue.Queue[Optional[_Job]]" = queue.Queue()
        self.thread = threading.Thread(target=self._loop, name="qwen3vl-cohost-combiner", daemon=True)
        self.thread.start()

    def submit(self, chip: int, pairs) -> Future:
        job = _Job(chip, list(pairs))
        self.q.put(job)
        return job.future

    def close(self):
        self.q.put(None)
        self.thread.join(30)

    def _gather(self, first: _Job) -> List[_Job]:
        jobs = [first]
        deadline = time.monotonic() + GATHER_S
        while True:
            left = deadline - time.monotonic()
            try:
                j = self.q.get(timeout=max(left, 0)) if left > 0 else self.q.get_nowait()
            except queue.Empty:
                return jobs
            if j is None:
                self.q.put(None)
                return jobs
            jobs.append(j)

    def _loop(self):
        while True:
            first = self.q.get()
            if first is None:
                return
            jobs = self._gather(first)
            try:
                self._run(jobs)
            except BaseException as exc:  # noqa: BLE001 - every job of the batch fails, the thread survives
                log.exception("qwen3vl-cohost: pass failed")
                for j in jobs:
                    if not j.future.done():
                        j.future.set_exception(exc)

    def _run(self, jobs: List[_Job]):
        eng = self.engine
        n = len(eng.lanes)
        results = {id(j): [None] * len(j.pairs) for j in jobs}
        # per job: its text passes (B, T, indices) and its vision pairs
        plans = []
        for j in jobs:
            text = [i for i, p in enumerate(j.pairs) if not p.has_vision]
            vis = [i for i, p in enumerate(j.pairs) if p.has_vision]
            passes = [(B, T, [text[k] for k in idx]) for B, T, idx in plan_text_passes([j.pairs[i].length for i in text])]
            plans.append((j, passes, vis))
        # A lane can hold several jobs only if one service sent it two chunks; they run one after the other.
        queues: Dict[int, List[tuple]] = {}
        for j, passes, _ in plans:
            queues.setdefault(j.chip, []).extend((j, B, T, idx) for B, T, idx in passes)
        while any(queues.values()):
            heads = {c: q[0] for c, q in queues.items() if q}
            T = max(h[2] for h in heads.values())
            chosen = {c: h for c, h in heads.items() if h[2] <= T}
            slot = covering_slot(max(h[1] for h in chosen.values()), T)
            if slot is None or T > TRACE_MAX_T:
                # no captured slot covers every head: run the lane(s) at the longest bucket alone
                chosen = {c: h for c, h in heads.items() if h[2] == T}
                B = max(h[1] for h in chosen.values())
                slot = covering_slot(B, T) or (B, T)
            B, T = slot
            per_chip = [[] for _ in range(n)]
            for c, (j, _, _, idx) in chosen.items():
                per_chip[c] = [j.pairs[i] for i in idx]
            owner = self.member_of[eng.lanes[next(iter(chosen))]]
            out = self.step(owner, lambda: eng.run_text(per_chip, B, T))
            for c, (j, _, _, idx) in chosen.items():
                for i, v in zip(idx, out[c]):
                    results[id(j)][i] = v
                queues[c].pop(0)
        for j, _, vis in plans:
            for i in vis:
                owner = self.member_of[eng.lanes[j.chip]]
                results[id(j)][i] = self.step(owner, lambda j=j, i=i: eng.run_vision(j.chip, j.pairs[i]))
        for j in jobs:
            j.future.set_result(results[id(j)])


class Lane:
    """The ``ScoringEngine`` of one chip for ``RerankService`` (its ``EngineWorker`` calls these on its thread)."""

    def __init__(self, combiner: Combiner, chip: int, task: str):
        self.combiner, self.die, self.task = combiner, chip, task
        self.counters = {"pairs": 0, "calls": 0}

    def max_tokens(self) -> int:
        return MAX_TOKENS

    def warm(self) -> dict:  # the device boot ran in the co-host phases already
        return {"die": self.die, "task": self.task, "buckets": list(TEXT_BUCKETS), "traced_max_T": TRACE_MAX_T}

    def score(self, pairs):
        if self.combiner.engine.towers.get(self.task) is None and any(p.has_vision for p in pairs):
            from models.autoports.qwen_qwen3_vl_reranker_8b.server.media import RequestError

            raise RequestError("this server is text-only: image and video inputs are not served "
                               "(the co-host profile runs without the vision towers)", code="unsupported_input")
        out = self.combiner.submit(self.die, pairs).result()
        self.counters["pairs"] += len(pairs)
        self.counters["calls"] += 1
        return out

    def stats(self) -> dict:
        return {**self.counters, **self.combiner.engine.counters}

    def alive(self) -> bool:
        return self.combiner.thread.is_alive()

    def close(self) -> None:
        pass
