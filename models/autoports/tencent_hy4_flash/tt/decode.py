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
from models.autoports.tencent_hy4_flash.tt.rope import cos_sin, rpos

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
    base_f: object = None  # [1, 1, 32, 1] fp32: the active slot's sparse cache base (added to the selected rows)
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
        # the last bucket: every group the cache holds (a context past the largest power-of-two bucket that fits, e.g.
        # 8846 tokens in a 12K slot, raised "past the largest indexer bucket" and ended the served request)
        cap = ((self.gcap - 32) // 32) * 32
        if self.gcap and (not self.buckets or cap > self.buckets[-1]):
            self.buckets.append(cap)
        self.min_bucket = 0  # warm-up: force a larger score bucket than the context needs (compile / capture it)
        self.si = self._alloc()
        self._alloc_outputs()
        self.ob = self.ogb = None  # the staging windows' bases
        self.pos = None
        self.tokens = []  # every token fed so far, by position (Engram history after a rewind)
        self.traces = {}
        self.out = {}
        self._prog_n = None  # program-cache entries when the live traces were captured (see _guard_programs)
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
        si.base_f = self._dev(z(1, 1, R, 1), ttnn.float32, ttnn.TILE_LAYOUT)
        return si

    def _alloc_outputs(self):
        """Persistent pass outputs (Hy4Model.step copies into them): logits, final hidden, the indexer selections."""
        m = self.model
        if getattr(m, "step_bufs", None) is not None:
            return
        N, H = m.N, self.cfg.hidden
        z = torch.zeros
        lg = ttnn.from_torch(z(1, 1, ROWS, m.Vp), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=self.mesh,
                             mesh_mapper=ttnn.ShardTensorToMesh(self.mesh, dim=-1))
        hn = self._dev(z(1, 1, ROWS, H), ttnn.bfloat16, ttnn.TILE_LAYOUT)
        lr = ttnn.from_torch(z(1, 1, m.LG_ROWS, m.Vp), dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.mesh,
                             mesh_mapper=ttnn.ShardTensorToMesh(self.mesh, dim=-1))
        m.step_bufs = {"lg": lg, "hn": hn, "lg_rows": lr}
        for l in m.layers:
            if l.attn.indexer is not None:
                l.attn.indexer.sel_buf = self._dev(z(1, 1, ROWS, self.n_sel, dtype=torch.int32), ttnn.uint32,
                                                   ttnn.ROW_MAJOR_LAYOUT)

    # ------------------------------------------------------------------ prefill -> decode
    def _bases(self):
        a0 = next(l.attn for l in self.model.layers if l.attn.kind == "sparse")
        s0 = next((l.attn for l in self.model.layers if l.attn.kind == "swa"), None)
        return a0.base, (s0.base if s0 is not None else 0)

    def start(self, pos: int):
        """After a prefill of `pos` tokens: load the staging windows from the caches."""
        nb = (pos // 32) * 32
        gb = ((nb // self.cfg.idx_ratio) // 32) * 32
        for layer in self.model.layers:
            layer.attn.init_stage(nb)
            if layer.attn.indexer is not None:
                layer.attn.indexer.init_stage(nb, gb)
        self.ob, self.ogb, self.pos = nb, gb, pos

    def set_history(self, tokens):
        """The sequence's tokens so far (the prompt): the Engram n-gram history of a rewound pass."""
        self.tokens = [int(t) for t in tokens]

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
            c, s = cos_sin(rpos(pos), cfg.rope_theta[k], cfg.qk_rope)
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
        bs, bw = self._bases()  # the active slot's cache regions
        h["off_sparse"] = [one(bs + nb), one(bs + nb + 32)]
        h["off_swa"] = [one(bw + nb % self.ring), one(bw + (nb + 32) % self.ring)] if self.ring else [one(0), one(0)]
        h["base_f"] = _tile(torch.full((1, 1, ROWS, 1), float(bs)), ttnn.float32)
        h["off_grp"] = [one(gb), one(gb + 32)]
        gpos = torch.zeros(32, dtype=torch.long)
        gpos[:16] = torch.arange(nb, nb + 64, r)
        gc, gs = cos_sin(rpos(gpos), cfg.rope_theta["sparse"], cfg.qk_rope)
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
            dense[s, 1 : 1 + n] = bs + np.arange(p + 1 - n, p + 1)
            w = np.arange(max(0, p - cfg.window + 1), p + 1)
            swa[s, 1 : 1 + w.size] = bw + (w % self.ring if self.ring else w)
            own = np.arange((p // r) * r, p + 1)
            tail[s, 1 : 1 + own.size] = bs + own
        u = lambda a: _rm(torch.from_numpy(a.view(np.int32)).reshape(1, 1, ROWS, -1), ttnn.uint32)  # noqa: E731
        h["idx_dense"], h["idx_swa"], h["idx_tail"] = u(dense), u(swa), u(tail)
        sparse = p0 >= cfg.idx_candidate_topk
        bucket = 0
        if sparse:
            need = max(int(vis.max()), self.min_bucket)
            bucket = next((b for b in self.buckets if b >= need), None)
            if bucket is None:
                raise RuntimeError(f"context {p0 + R} past the largest indexer bucket")
        del self.tokens[p0:]  # a pass may start before the end of the last one (rejected verify rows)
        self.tokens += [int(t) for t in ids]
        eng = self.model.engram
        for li in self.model.engram_li():
            if eng is not None and hasattr(eng, "pos") and eng.pos.get(li) != p0:
                keep = getattr(eng, "keep", 2)
                eng.hist[li] = np.array(self.tokens[max(0, p0 - keep) : p0], dtype=np.int64)
                eng.pos[li] = p0
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
        for name in ("A", "P", "Ag", "Pg", "idx_dense", "idx_swa", "idx_tail", "gcos", "gsin", "vis", "base_f"):
            c(h[name], getattr(si, name))
        for j in range(2):
            c(h["off_sparse"][j], si.off_sparse[j])
            c(h["off_swa"][j], si.off_swa[j])
            c(h["off_grp"][j], si.off_grp[j])
        for li, t in si.engram.items():
            if f"eng_{li}" in h:
                c(h[f"eng_{li}"], t)

    # ------------------------------------------------------------------ trace safety
    def _programs(self):
        try:
            return int(self.mesh.num_program_cache_entries())
        except Exception:  # noqa: BLE001
            return None

    def _guard_programs(self, where):
        """A program compiled after a trace capture has its kernel binaries allocated in DRAM, possibly inside that
        trace's freed intermediates, which every replay overwrites: the next launch runs corrupted kernels and hangs the
        device (the Inkling streamed-prefill hang; here: a prompt of a new length after the decode / draft captures,
        2026-11-19). Whenever the program cache grew since the live traces were captured, release them all (they are
        re-captured on their next use)."""
        n = self._programs()
        if n is not None and self._prog_n is not None and n != self._prog_n and self._has_traces():
            logger.info(f"{n - self._prog_n} programs compiled since the trace captures ({where}): releasing the traces")
            self.release()
        if not self._has_traces():
            self._prog_n = n

    def _has_traces(self):
        return bool(self.traces)

    # ------------------------------------------------------------------ a pass
    def run(self, ids: list[int], logits_rows: str = "last", p0: int | None = None) -> torch.Tensor:
        """Rows `ids` at positions p0 .. p0 + R - 1 (default: right after the last pass) -> logits fp32 [V] (last row)
        or [R, V]. A verify pass that accepted n of its rows continues with p0 = its start + n."""
        import time as _t

        R = len(ids)
        p0 = self.pos if p0 is None else int(p0)
        t0 = _t.perf_counter()
        h, sparse, bucket = self._host_inputs(ids, p0)
        t1 = _t.perf_counter()
        self._copy_inputs(h)
        t2 = _t.perf_counter()
        self.si.sparse, self.si.bucket = sparse, bucket
        key = (R, sparse, bucket)
        if not self.trace:
            lg = self.model.step(self.si, R, mem=self.mem)
            out = self.model.read_logits(lg, R)  # (lg: the persistent logits buffer)
        else:
            self._guard_programs("before a pass")
            if key not in self.traces:
                # compile eagerly first (this pass's result is the real one), then capture for the next passes
                lg = self.model.step(self.si, R, mem=self.mem)
                out = self.model.read_logits(lg, R)  # (lg: the persistent logits buffer)
                self._guard_programs("the eager pass of a new trace")
                self._capture(key, ids, p0)
                self._prog_n = self._programs()
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
        if self.traces:
            ttnn.synchronize_device(self.mesh)  # (a non-blocking replay may still run)
        for tid in self.traces.values():
            ttnn.release_trace(self.mesh, tid)
        self.traces, self.out = {}, {}


@dataclass
class DraftInputs:
    tok: object = None
    bonus: object = None
    cos: object = None
    sin: object = None
    idx_dense: object = None
    idx_tail: object = None
    sparse: bool = False


class Hy4SpecDecoder(Hy4Decoder):
    """Hy4Decoder + the DFly2 drafter (tt/dfly.py): draft passes traced per (target pass key, accepted row)."""

    def __init__(self, model, trace: bool = True, k: int | None = None):
        super().__init__(model, trace)
        from models.autoports.tencent_hy4_flash.tt.dfly import W, Hy4DFly

        self.W = W
        self.dfly = Hy4DFly(model)
        # drafts per pass (<= W - 1): fewer drafts = fewer sequential correction steps and narrower verify rows
        self.k = self.dfly.k = min(W - 1, int(k if k is not None else os.environ.get("HY4_SPEC_K", min(4, W - 1))))
        self.scratch = self.dfly.layers[0]["attn"].scratch_row
        self.ds = self._alloc_draft()
        self.dfly.out_buf = self._dev(torch.zeros(1, self.k, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        self.dtraces, self.dout = {}, {}
        self.saved = {}  # target pass key -> (last_hn, {layer: sel}) of that pass's trace
        self.last_key = None

    def set_k(self, k: int):
        """Change the drafts per pass (releases the traces; the draft output buffer is re-allocated)."""
        self.release()
        self.k = self.dfly.k = min(self.W - 1, int(k))
        ttnn.deallocate(self.dfly.out_buf)
        self.dfly.out_buf = self._dev(torch.zeros(1, self.k, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)

    def _alloc_draft(self):
        z = torch.zeros
        ds = DraftInputs()
        ds.tok = self._dev(z(1, ROWS, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        ds.bonus = self._dev(z(1, 1, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        ds.cos = self._dev(z(1, 1, ROWS, self.cfg.qk_rope), ttnn.float32, ttnn.TILE_LAYOUT)
        ds.sin = self._dev(z(1, 1, ROWS, self.cfg.qk_rope), ttnn.float32, ttnn.TILE_LAYOUT)
        ds.idx_dense = self._dev(z(1, 1, ROWS, self.K_dense, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        ds.idx_tail = self._dev(z(1, 1, ROWS, self.K_dense - self.n_sel, dtype=torch.int32), ttnn.uint32,
                                ttnn.ROW_MAJOR_LAYOUT)
        return ds

    def _remember(self, key):
        """After a target pass (eager or the capture of its trace): the tensors the drafter reads."""
        sel = {l.li: l.attn.indexer.sel for l in self.model.layers
               if l.attn.indexer is not None and getattr(l.attn.indexer, "sel", None) is not None}
        self.saved[key] = (self.model.last_hn, sel)

    def run(self, ids, logits_rows="last", p0=None):
        out = super().run(ids, logits_rows, p0)
        R = len(ids)
        key = (R, self.si.sparse, self.si.bucket)
        if not self.trace or key not in self.saved:
            self._remember(key)
        self.last_key = key
        return out

    def draft(self, bonus: int, anchor: int, n: int) -> list[int]:
        """The 5 greedy drafts after the last target pass (its last accepted row n; `bonus` at position anchor)."""
        cfg, r = self.cfg, self.cfg.idx_ratio
        tkey = self.last_key
        sparse = tkey[1]
        h = {}
        tok = torch.zeros(1, ROWS, dtype=torch.int32)
        tok[0, 0] = bonus
        tok[0, 1 : self.W] = self.dfly.mask_id
        h["tok"] = _rm(tok, ttnn.uint32)
        h["bonus"] = _rm(torch.tensor([[bonus]], dtype=torch.int32), ttnn.uint32)
        pos = torch.full((ROWS,), anchor + self.W - 1, dtype=torch.long)
        pos[: self.W] = torch.arange(anchor, anchor + self.W)
        c, s = cos_sin(rpos(pos), cfg.rope_theta["sparse"], cfg.qk_rope)
        h["cos"] = _tile(c.float().reshape(1, 1, ROWS, -1), ttnn.float32)
        h["sin"] = _tile(s.float().reshape(1, 1, ROWS, -1), ttnn.float32)
        dense = np.full((ROWS, self.K_dense), SENTINEL, dtype=np.uint32)
        tail = np.full((ROWS, self.K_dense - self.n_sel), SENTINEL, dtype=np.uint32)
        dense[:, 0] = tail[:, 0] = self.sink_sparse
        bs = self._bases()[0]  # the active slot's sparse cache region
        own = bs + np.arange(((anchor - 1) // r) * r, anchor)
        for j in range(ROWS):
            jj = min(j, self.W - 1)
            blk = self.scratch + np.arange(jj + 1)
            if not sparse:
                keys = np.concatenate([bs + np.arange(anchor), blk])
                dense[j, 1 : 1 + keys.size] = keys
            t = np.concatenate([own, blk])
            tail[j, 1 : 1 + t.size] = t
        u = lambda a: _rm(torch.from_numpy(a.view(np.int32)).reshape(1, 1, ROWS, -1), ttnn.uint32)  # noqa: E731
        h["idx_dense"], h["idx_tail"] = u(dense), u(tail)
        c_ = ttnn.copy_host_to_device_tensor
        for k in ("tok", "bonus", "cos", "sin", "idx_dense", "idx_tail"):
            c_(h[k], getattr(self.ds, k))
        self.ds.sparse = sparse
        dkey = (tkey, n)
        hn, sel = self.saved[tkey]
        self.model.last_hn = hn
        for l in self.model.layers:
            if l.li in sel:
                l.attn.indexer.sel = sel[l.li]
        if self.trace and sparse not in self.__dict__.setdefault("_pre_n", set()):
            # the drafter slices row n (the last accepted row) of the target's last hidden rows and selections, and a
            # slice bakes its offset into its program: compile every row's slice now (one trace release, here at the
            # first draft of the regime, typically in warm-up), not one per new accepted count while serving
            self._pre_n.add(sparse)
            H = int(self.model.last_hn.shape[-1])
            for m in range(self.W):
                ttnn.deallocate(ttnn.slice(self.model.last_hn, (0, 0, m, 0), (1, 1, m + 1, H), memory_config=self.mem))
                if sparse:
                    for li, t_ in sel.items():
                        ttnn.deallocate(ttnn.slice(t_, (0, 0, m, 0), (1, 1, m + 1, int(t_.shape[-1]))))
                        break  # (every layer's selection has the same shape)
        if self.trace:
            self._guard_programs("before a draft")
        if not self.trace:
            d = self.dfly.step(self.ds, n, mem=self.mem)
            out = ttnn.to_torch(ttnn.get_device_tensors(d)[0]).reshape(-1)[: self.k]
            pass  # d: the drafter's persistent output buffer
        elif dkey not in self.dtraces:
            d = self.dfly.step(self.ds, n, mem=self.mem)
            out = ttnn.to_torch(ttnn.get_device_tensors(d)[0]).reshape(-1)[: self.k]
            pass  # d: the drafter's persistent output buffer
            self._guard_programs("the eager pass of a new draft trace")
            tid = ttnn.begin_trace_capture(self.mesh, cq_id=0)
            self.dout[dkey] = self.dfly.step(self.ds, n, mem=self.mem)
            ttnn.end_trace_capture(self.mesh, tid, cq_id=0)
            self.dtraces[dkey] = tid
            self._prog_n = self._programs()
            logger.info(f"captured draft trace {dkey}")
        else:
            ttnn.execute_trace(self.mesh, self.dtraces[dkey], cq_id=0, blocking=False)
            out = ttnn.to_torch(ttnn.get_device_tensors(self.dout[dkey])[0]).reshape(-1)[: self.k]
            if os.environ.get("HY4_DFLY_CHECK") == "1":
                d = self.dfly.step(self.ds, n, mem=self.mem)
                e = ttnn.to_torch(ttnn.get_device_tensors(d)[0]).reshape(-1)[: self.k]
                pass  # d: the drafter's persistent output buffer
                logger.info(f"draft {dkey}: trace {out.tolist()} eager {e.tolist()}")
        return [int(v) & 0xFFFFFFFF for v in out.tolist()]

    def _has_traces(self):
        return bool(self.traces) or bool(self.dtraces)

    def release(self):
        super().release()
        for tid in self.dtraces.values():
            ttnn.release_trace(self.mesh, tid)
        self.dtraces, self.dout = {}, {}
        self.saved = {}
