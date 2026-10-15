# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Decoder-layer harness shared by the tests and the perf tools: build a layer, run chunked prefill for one user
slot, run decode steps eagerly or as a captured trace (fixed-address inputs refreshed between replays)."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import torch

import ttnn
from models.autoports.zai_org_glm_5_3_flash.tt.ckpt import CheckpointReader
from models.autoports.zai_org_glm_5_3_flash.tt.functional_decoder import FunctionalDecoder
from models.autoports.zai_org_glm_5_3_flash.tt.synthetic import synthetic_layer_state

HERE = Path(__file__).resolve().parent
STATS = HERE / "weight_stats.json"
HC = 4
HIDDEN = 4096
H4 = HC * HIDDEN


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.flatten().double(), b.flatten().double()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm() + 1e-30))


def open_mesh(trace_region_size: int = 256 << 20):
    """GLM53_WORKER_L1_SIZE: optional smaller worker L1 (bytes); the kernel-config ring buffer grows by the difference
    (watcher-instrumented KDA prefill programs need ~75.6 KB of it; default ring 70,656 B, default worker L1 1,461,248 B).
    """
    kw = {}
    if os.environ.get("GLM53_WORKER_L1_SIZE"):
        kw["worker_l1_size"] = int(os.environ["GLM53_WORKER_L1_SIZE"])
    return ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=24576, trace_region_size=trace_region_size, **kw)


def layer_weights(rd: CheckpointReader, layer: int, weights) -> dict:
    if weights == "real":
        return rd.layer_state(layer)
    return synthetic_layer_state(json.loads(STATS.read_text()), layer, seed=int(weights["synthetic"]))


def build_layer(
    mesh, rd, layer: int, weights="real", *, batch=8, prefill_lengths=(128, 2048), max_context=4096, decoder_cls=None
):
    """decoder_cls: FunctionalDecoder (default) or a subclass with the same contract (e.g. tt/fused_decoder.FusedDecoder)."""
    dec = (decoder_cls or FunctionalDecoder).from_state_dict(
        layer_weights(rd, layer, weights),
        hf_config=rd.text_config,
        layer_idx=layer,
        mesh_device=mesh,
        prefill_lengths=prefill_lengths,
        max_batch=batch,
        dsa_kwargs={"max_context": max_context},
    )
    dec.allocate_decode_state(batch)
    return dec


def prefill_user(dec, mesh, x: torch.Tensor, length: int, slot: int, bucket: int = 2048) -> torch.Tensor:
    """x [S, H4] stream rows of one user (S >= length). Chunked prefill into decode slot `slot` (chunks of `bucket`,
    the last chunk on the smallest constructed length that holds it); returns the output rows [length, H4]."""
    outs, st = [], None
    c0 = 0
    while c0 < length:
        ln = min(bucket, length - c0)
        n = next(b for b in sorted(dec.prefill_lengths) if b >= ln)
        xp = torch.zeros(1, 1, n, H4)
        xp[0, 0, :ln] = x[c0 : c0 + ln]
        X = ttnn.from_torch(xp, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh)
        inputs = dec.prefill_inputs(n, start=c0, length=ln, slot=slot)
        Y, st = dec.prefill_forward(X, inputs=inputs, attn_state=st)
        outs.append(ttnn.to_torch(Y)[0, 0, :ln])
        ttnn.deallocate(X)
        ttnn.deallocate(Y)
        c0 += ln
    dec.commit_prefill(slot, st)
    return torch.cat(outs, dim=0)


@dataclass
class DecodeRunner:
    """Decode steps for all `batch` slots of a layer (inactive slots: position -1, zero rows)."""

    dec: FunctionalDecoder
    mesh: object
    batch: int
    x_buf: object = None
    dsa_bufs: dict = None
    trace_id: object = None
    y_out: object = None
    traced_steps: int = field(default=0)

    def __post_init__(self):
        self.x_buf = ttnn.from_torch(
            torch.zeros(1, 1, self.batch, H4), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=self.mesh
        )

    def _set_inputs(self, rows: dict, positions: list):
        xd = torch.zeros(1, 1, self.batch, H4)
        for s, v in rows.items():
            xd[0, 0, s] = v
        ttnn.copy_host_to_device_tensor(ttnn.from_torch(xd, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT), self.x_buf)
        if self.dsa_bufs is None:
            self.dsa_bufs = self.dec.decode_inputs(positions)
        else:
            self.dec.update_decode_inputs(self.dsa_bufs, positions)

    def _forward(self):
        return self.dec.decode_forward(self.x_buf, inputs=self.dsa_bufs)

    def eager_step(self, rows: dict, positions: list) -> torch.Tensor:
        self._set_inputs(rows, positions)
        y = self._forward()
        out = ttnn.to_torch(y)[0, 0]
        ttnn.deallocate(y)
        return out

    def capture(self, rows: dict, positions: list):
        """Capture the decode step on fixed inputs (the capture records; nothing executes until a replay)."""
        self._set_inputs(rows, positions)
        self.trace_id = ttnn.begin_trace_capture(self.mesh, cq_id=0)
        try:
            self.y_out = self._forward()
        finally:
            ttnn.end_trace_capture(self.mesh, self.trace_id, cq_id=0)

    def traced_step(self, rows: dict, positions: list, sync: bool = True) -> torch.Tensor | None:
        self._set_inputs(rows, positions)
        ttnn.execute_trace(self.mesh, self.trace_id, cq_id=0, blocking=False)
        self.traced_steps += 1
        if not sync:
            return None
        return ttnn.to_torch(self.y_out)[0, 0]

    def release(self):
        if self.trace_id is not None:
            ttnn.release_trace(self.mesh, self.trace_id)
            self.trace_id = None
