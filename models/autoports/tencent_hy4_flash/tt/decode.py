# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Trace-safe decode / verify passes of the Hy4 model (one sequence, R <= 32 consecutive rows).

Everything position-dependent comes from the host as small tensors copied into persistent device buffers before each
replay (StepInputs): token ids, RoPE tables, Engram rows, the staging maps (one-hot A / P matrices: which window rows
carry over, where the new rows land), the 32-row slab offsets of the KV / group caches, the index lists of the dense
regime and of the sliding window, the per-row group visibility and the index tail (sink, own group). The KV / token-key /
group-key staging windows are device state the passes update in place.

Traces are captured per (rows, regime, bucket): the dense regime (first row < 2048: no indexer scoring) and, in the
sparse regime, one per context bucket (the indexer scores the first `bucket` groups; rows mask what they cannot see).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import torch
import ttnn
from loguru import logger

from models.autoports.tencent_hy4_flash.tt.attention import K_CHUNK, SENTINEL
from models.autoports.tencent_hy4_flash.tt.rope import cos_sin

ROWS = 32
BUCKETS_TOKENS = [int(b) for b in os.environ.get("HY4_IDX_BUCKETS", "8192,16384,32768,65536,131072,262144,524288,1048576").split(",")]


def _rm(t, dtype):
    return ttnn.from_torch(t.contiguous(), dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT)


def _tile(t, dtype):
    return ttnn.from_torch(t.contiguous(), dtype=dtype, layout=ttnn.TILE_LAYOUT)


@dataclass
class StepInputs:
    tok: object = None
    cos_sin: dict = field(default_factory=dict)
    engram: dict = field(default_factory=dict)
    A: object = None
    P: object = None
    Ag: object = None
    Pg: object = None
    slot: object = None
    off_sparse: list = field(default_factory=list)
    off_swa: list = field(default_factory=list)
    off_grp: list = field(default_factory=list)
    idx_dense: object = None
    idx_swa: object = None
    idx_tail: object = None
    gcos: object = None
    gsin: object = None
    vis: object = None
    sparse: bool = False
    bucket: int = 0


class Hy4Decoder:
    """Drives model.step() with persistent inputs, eagerly or through captured traces."""

    def __init__(self, model, trace: bool = True):
        self.model, self.cfg, self.mesh = model, model.cfg, model.mesh
        self.trace = trace
        cfg = self.cfg
        a0 = next(l.attn for l in model.layers if l.attn.kind == "sparse")
        s0 = next((l.attn for l in model.layers if l.attn.kind == "swa"), None)
        self.sink_sparse = a0.sink_row
        self.ring = s0.rows if s0 is not None else 0
        self.sink_swa = s0.sink_row if s0 is not None else 0
        self.K_dense = -(-(cfg.idx_candidate_topk + 1) // K_CHUNK) * K_CHUNK
        self.K_swa = -(-(cfg.window + 1) // K_CHUNK) * K_CHUNK
        self.n_sel = (cfg.idx_group_topk - 1) * cfg.idx_ratio
        self.gcap = a0.indexer.gcap if a0.indexer is not None else 0
        self.buckets = [b // cfg.idx_ratio for b in BUCKETS_TOKENS if b // cfg.idx_ratio + 32 <= self.gcap]
        self.si = self._alloc()
        self.ob = self.ogb = None  # the staging windows' bases
        self.pos = None
        self.traces = {}
        self.out = {}
        # HY4_DECODE_L1=1: the step's intermediates in L1 (interleaved) instead of DRAM
        self.mem = ttnn.L1_MEMORY_CONFIG if os.environ.get("HY4_DECODE_L1", "1") == "1" else ttnn.DRAM_MEMORY_CONFIG

    # ------------------------------------------------------------------ persistent inputs
    def _dev(self, t, dtype, layout):
        return ttnn.from_torch(t.contiguous(), dtype=dtype, layout=layout, device=self.mesh,
                               mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh))

    def _alloc(self):
        cfg, R = self.cfg, ROWS
        z = torch.zeros
        si = StepInputs()
        si.tok = self._dev(z(1, R, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        for k in ("sparse", "swa"):
            si.cos_sin[k] = (self._dev(z(1, 1, R, cfg.qk_rope), ttnn.float32, ttnn.TILE_LAYOUT),
                             self._dev(z(1, 1, R, cfg.qk_rope), ttnn.float32, ttnn.TILE_LAYOUT))
        for li in self.model.engram_li():
            si.engram[li] = self._dev(z(1, 1, R, 1536), ttnn.bfloat16, ttnn.TILE_LAYOUT)
        si.A = self._dev(z(1, 1, 64, 64), ttnn.bfloat16, ttnn.TILE_LAYOUT)
        si.P = self._dev(z(1, 1, 64, R), ttnn.bfloat16, ttnn.TILE_LAYOUT)
        si.Ag = self._dev(z(1, 1, 64, 64), ttnn.bfloat16, ttnn.TILE_LAYOUT)
        si.Pg = self._dev(z(1, 1, 64, 32), ttnn.bfloat16, ttnn.TILE_LAYOUT)
        one = lambda: self._dev(z(1, 1, 1, 1, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)  # noqa: E731
        si.slot = one()
        si.off_sparse, si.off_swa, si.off_grp = [one(), one()], [one(), one()], [one(), one()]
        si.idx_dense = self._dev(z(1, 1, R, self.K_dense, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        si.idx_swa = self._dev(z(1, 1, R, self.K_swa, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        si.idx_tail = self._dev(z(1, 1, R, self.K_dense - self.n_sel, dtype=torch.int32), ttnn.uint32,
                                ttnn.ROW_MAJOR_LAYOUT)
        si.gcos = self._dev(z(1, 1, 32, cfg.qk_rope), ttnn.float32, ttnn.TILE_LAYOUT)
        si.gsin = self._dev(z(1, 1, 32, cfg.qk_rope), ttnn.float32, ttnn.TILE_LAYOUT)
        si.vis = self._dev(z(1, 1, R, 1), ttnn.float32, ttnn.TILE_LAYOUT)
        return si

    # ------------------------------------------------------------------ prefill -> decode
    def start(self, pos: int):
        """After a prefill of `pos` tokens: load the staging windows from the caches."""
        nb = (pos // 32) * 32
        gb = ((nb // self.cfg.idx_ratio) // 32) * 32
        for layer in self.model.layers:
            layer.attn.init_stage(nb)
            if layer.attn.indexer is not None:
                layer.attn.indexer.init_stage(nb, gb)
        self.ob, self.ogb, self.pos = nb, gb, pos

    # ------------------------------------------------------------------ host side of a step
    def _host_inputs(self, ids: list[int], p0: int):
        cfg, R = self.cfg, len(ids)
        r = cfg.idx_ratio
        nb = (p0 // 32) * 32
        assert p0 + R <= nb + 64, "rows must fit the 64-row staging window"
        gw0 = nb // r
        gb = (gw0 // 32) * 32
        h = {}
        tok = torch.zeros(1, ROWS, dtype=torch.int32)
        tok[0, :R] = torch.tensor(ids, dtype=torch.int32)
        h["tok"] = _rm(tok, ttnn.uint32)
        pos = torch.full((ROWS,), p0 + R - 1, dtype=torch.long)
        pos[:R] = torch.arange(p0, p0 + R)
        for k in ("sparse", "swa"):
            c, s = cos_sin(pos, cfg.rope_theta[k], cfg.qk_rope)
            h[f"cos_{k}"] = _tile(c.float().reshape(1, 1, ROWS, -1), ttnn.float32)
            h[f"sin_{k}"] = _tile(s.float().reshape(1, 1, ROWS, -1), ttnn.float32)
        A = torch.zeros(64, 64)
        P = torch.zeros(64, ROWS)
        for i in range(64):
            q = nb + i
            if p0 <= q < p0 + R:
                P[i, q - p0] = 1
            elif q < p0 and self.ob <= q < self.ob + 64:
                A[i, q - self.ob] = 1
        h["A"], h["P"] = _tile(A.reshape(1, 1, 64, 64), ttnn.bfloat16), _tile(P.reshape(1, 1, 64, ROWS), ttnn.bfloat16)
        Ag = torch.zeros(64, 64)
        Pg = torch.zeros(64, 32)
        for i in range(64):
            g = gb + i
            if gw0 <= g < gw0 + 16:
                Pg[i, g - gw0] = 1
            elif self.ogb <= g < self.ogb + 64:
                Ag[i, g - self.ogb] = 1
        h["Ag"], h["Pg"] = _tile(Ag.reshape(1, 1, 64, 64), ttnn.bfloat16), _tile(Pg.reshape(1, 1, 64, 32), ttnn.bfloat16)
        one = lambda v: _rm(torch.tensor([v], dtype=torch.int32).reshape(1, 1, 1, 1), ttnn.uint32)  # noqa: E731
        h["off_sparse"] = [one(nb), one(nb + 32)]
        h["off_swa"] = [one(nb % self.ring), one((nb + 32) % self.ring)] if self.ring else [one(0), one(0)]
        h["off_grp"] = [one(gb), one(gb + 32)]
        gpos = torch.zeros(32, dtype=torch.long)
        gpos[:16] = torch.arange(nb, nb + 64, r)
        gc, gs = cos_sin(gpos, cfg.rope_theta["sparse"], cfg.qk_rope)
        h["gcos"] = _tile(gc.float().reshape(1, 1, 32, -1), ttnn.float32)
        h["gsin"] = _tile(gs.float().reshape(1, 1, 32, -1), ttnn.float32)
        vis = torch.tensor([(int(pos[s])) // r for s in range(ROWS)], dtype=torch.float32)
        h["vis"] = _tile(vis.reshape(1, 1, ROWS, 1), ttnn.float32)
        dense = np.full((ROWS, self.K_dense), SENTINEL, dtype=np.uint32)
        swa = np.full((ROWS, self.K_swa), SENTINEL, dtype=np.uint32)
        tail = np.full((ROWS, self.K_dense - self.n_sel), SENTINEL, dtype=np.uint32)
        dense[:, 0], swa[:, 0], tail[:, 0] = self.sink_sparse, self.sink_swa, self.sink_sparse
        for s in range(ROWS):
            p = int(pos[s])
            n = min(p + 1, self.K_dense - 1)
            dense[s, 1 : 1 + n] = np.arange(p + 1 - n, p + 1)
            w = np.arange(max(0, p - cfg.window + 1), p + 1)
            swa[s, 1 : 1 + w.size] = w % self.ring if self.ring else w
            own = np.arange((p // r) * r, p + 1)
            tail[s, 1 : 1 + own.size] = own
        u = lambda a: _rm(torch.from_numpy(a.view(np.int32)).reshape(1, 1, ROWS, -1), ttnn.uint32)  # noqa: E731
        h["idx_dense"], h["idx_swa"], h["idx_tail"] = u(dense), u(swa), u(tail)
        sparse = p0 >= cfg.idx_candidate_topk
        bucket = 0
        if sparse:
            need = int(vis.max())
            bucket = next((b for b in self.buckets if b >= need), None)
            if bucket is None:
                raise RuntimeError(f"context {p0 + R} past the largest indexer bucket")
        for li in self.model.engram_li():
            rows = self.model.engram.rows(li, torch.tensor(ids), p0)
            e = torch.zeros(1, 1, ROWS, rows.shape[-1], dtype=torch.bfloat16)
            e[0, 0, :R] = rows
            h[f"eng_{li}"] = _tile(e, ttnn.bfloat16)
        self.ob, self.ogb = nb, gb
        return h, sparse, bucket

    def _copy_inputs(self, h):
        si, c = self.si, ttnn.copy_host_to_device_tensor
        c(h["tok"], si.tok)
        for k in ("sparse", "swa"):
            c(h[f"cos_{k}"], si.cos_sin[k][0])
            c(h[f"sin_{k}"], si.cos_sin[k][1])
        for name in ("A", "P", "Ag", "Pg", "idx_dense", "idx_swa", "idx_tail", "gcos", "gsin", "vis"):
            c(h[name], getattr(si, name))
        for j in range(2):
            c(h["off_sparse"][j], si.off_sparse[j])
            c(h["off_swa"][j], si.off_swa[j])
            c(h["off_grp"][j], si.off_grp[j])
        for li, t in si.engram.items():
            if f"eng_{li}" in h:
                c(h[f"eng_{li}"], t)

    # ------------------------------------------------------------------ a pass
    def run(self, ids: list[int], logits_rows: str = "last") -> torch.Tensor:
        """Rows `ids` at positions pos .. pos + R - 1 -> logits fp32 [V] (last row) or [R, V]."""
        import time as _t

        R, p0 = len(ids), self.pos
        t0 = _t.perf_counter()
        h, sparse, bucket = self._host_inputs(ids, p0)
        t1 = _t.perf_counter()
        self._copy_inputs(h)
        t2 = _t.perf_counter()
        self.si.sparse, self.si.bucket = sparse, bucket
        key = (R, sparse, bucket)
        if not self.trace:
            lg = self.model.step(self.si, R, mem=self.mem)
            out = self.model.read_logits(lg, R)
            ttnn.deallocate(lg)
        else:
            if key not in self.traces:
                # compile eagerly first (this pass's result is the real one), then capture for the next passes
                lg = self.model.step(self.si, R, mem=self.mem)
                out = self.model.read_logits(lg, R)
                ttnn.deallocate(lg)
                self._capture(key, ids, p0)
            else:
                ttnn.execute_trace(self.mesh, self.traces[key], cq_id=0, blocking=False)
                if os.environ.get("HY4_TIME_STEPS") == "1":
                    ttnn.synchronize_device(self.mesh)
                t3 = _t.perf_counter()
                out = self.model.read_logits(self.out[key], R)
                t4 = _t.perf_counter()
                self.timing = (t1 - t0, t2 - t1, t3 - t2, t4 - t3)
        self.pos = p0 + R
        return out[R - 1] if logits_rows == "last" else out[:R]

    def _capture(self, key, ids, p0):
        """Capture a trace of the pass with the current inputs. The pass is idempotent for fixed inputs (the staging
        update re-places the same rows, the slab writes rewrite the same rows), so whether the capture also executes it
        does not change the state."""
        logger.info(f"capturing decode trace {key} ...")
        tid = ttnn.begin_trace_capture(self.mesh, cq_id=0)
        self.out[key] = self.model.step(self.si, key[0], mem=self.mem)
        ttnn.end_trace_capture(self.mesh, tid, cq_id=0)
        self.traces[key] = tid
        logger.info(f"captured decode trace {key}")

    def release(self):
        for tid in self.traces.values():
            ttnn.release_trace(self.mesh, tid)
        self.traces, self.out = {}, {}
