# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Captured-trace replay of the text tower, one trace per (batch, bucket) slot.

Each slot owns persistent device inputs (token ids, MRoPE cos/sin, the one-hot row select) and the score tile its
trace writes. A pass copies the host tensors into the slot's inputs and replays the trace; nothing is allocated per
request, and no program is compiled after warm-up.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch

import ttnn
from models.autoports.qwen_qwen3_vl_reranker_8b.tt.inputs import PairInputs
from models.autoports.qwen_qwen3_vl_reranker_8b.tt.runner import HostBatch, read_scores
from models.autoports.qwen_qwen3_vl_reranker_8b.tt.text_model import TextScorer


def _dummy_pair(n: int) -> PairInputs:
    ids = torch.full((n,), 11, dtype=torch.long)
    return PairInputs(input_ids=ids, position_ids=torch.arange(n)[None].expand(3, n).clone())


@dataclass
class Slot:
    B: int
    T: int
    inputs: dict
    output: object = None
    trace_id: object = None
    replays: int = 0
    capture_s: float = 0.0
    host_bufs: dict = field(default_factory=dict)


class TracedText:
    def __init__(self, scorer: TextScorer, shapes):
        """shapes: iterable of (B, T)."""
        self.scorer = scorer
        self.device = scorer.device
        self.slots: dict[tuple[int, int], Slot] = {}
        for B, T in shapes:
            self.slots[(B, T)] = self._capture(B, T)

    def _host(self, hb: HostBatch) -> dict:
        rep = ttnn.ReplicateTensorToMesh(self.device)
        return {
            "ids": ttnn.from_torch(hb.ids, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=rep),
            "cos": ttnn.from_torch(hb.cos, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=rep),
            "sin": ttnn.from_torch(hb.sin, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=rep),
            "select": ttnn.from_torch(hb.select, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=rep),
        }

    def _run(self, slot: Slot):
        d = slot.inputs
        return self.scorer.forward(d["ids"], d["cos"], d["sin"], d["select"], slot.B, slot.T)

    def _capture(self, B, T) -> Slot:
        t0 = time.time()
        hb = HostBatch(self.scorer, [_dummy_pair(T - 1)] * B, T)
        slot = Slot(B=B, T=T, inputs=hb.to_device(self.device))
        out = self._run(slot)  # compile
        ttnn.deallocate(out)
        ttnn.synchronize_device(self.device)
        slot.trace_id = ttnn.begin_trace_capture(self.device, cq_id=0)
        slot.output = self._run(slot)
        ttnn.end_trace_capture(self.device, slot.trace_id, cq_id=0)
        ttnn.synchronize_device(self.device)
        slot.capture_s = time.time() - t0
        return slot

    def score(self, pairs: list[PairInputs], B: int, T: int) -> list[float]:
        slot = self.slots[(B, T)]
        n = len(pairs)
        assert n <= B
        padded = list(pairs) + [_dummy_pair(1)] * (B - n)
        hb = HostBatch(self.scorer, padded, T)
        host = self._host(hb)
        for k, v in host.items():
            ttnn.copy_host_to_device_tensor(v, slot.inputs[k])
        ttnn.execute_trace(self.device, slot.trace_id, cq_id=0, blocking=False)
        slot.replays += 1
        return read_scores(slot.output, B)[:n]

    def release(self):
        for s in self.slots.values():
            if s.trace_id is not None:
                ttnn.release_trace(self.device, s.trace_id)
                s.trace_id = None
