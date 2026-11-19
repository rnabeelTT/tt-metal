# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Batched decode: up to L sequences (model slots) in one trace-safe pass (rows <= 32, user-contiguous).

Each sequence holds a lane: lane l's 64-row KV / token / group windows are rows 64 l .. of the layers' L-lane staging
tensors and its group keys rows l gcap .. of the indexers' lane key tensors; the KV cache stays the slot's region
(index lists and slab offsets carry its base). add() loads a slot's single-path windows (after its prefill and
Hy4Decoder-style init_stage) into a lane, remove() writes them back. One pass updates every lane's windows with
block-diagonal one-hot maps and writes every lane's slabs with one slab_write per cache (idle lanes: trash rows).
Sparse-layer index lists: one score over the lanes' keys, masked per row to its lane's visible groups; rows below the
dense-regime boundary take host lists (mixed per row in fp32, sentinel 2^32 -> 0xFFFFFFFF). Traces per (rows, width).
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import ttnn
from loguru import logger

from models.autoports.tencent_hy4_flash.tt.attention import K_CHUNK, SENTINEL
from models.autoports.tencent_hy4_flash.tt.decode import ROWS, Hy4Decoder, _rm, _tile
from models.autoports.tencent_hy4_flash.tt.rope import cos_sin

SENT_F = float(2**32)


@dataclass
class BatchInputs:
    L: int = 0
    R: int = 0
    width: int = 0
    sel: bool = True  # some row selects (context >= the dense regime): score + top-k; else the dense lists only
    tok: object = None
    cos_sin: dict = field(default_factory=dict)
    engram: dict = field(default_factory=dict)
    A: object = None
    P: object = None
    Ag: object = None
    Pg: object = None
    lane_offs: dict = field(default_factory=dict)
    gcos: object = None
    gsin: object = None
    idx_swa: object = None
    lo: object = None
    hi: object = None
    delta: object = None
    nmask: object = None
    tail_f: object = None
    dense_f: object = None


class Hy4BatchDecoder(Hy4Decoder):
    def __init__(self, model, lanes: int = 8, trace: bool = True):
        super().__init__(model, trace=trace)
        self.L = L = lanes
        self.a0 = next(l.attn for l in model.layers if l.attn.kind == "sparse")
        self.s0 = next((l.attn for l in model.layers if l.attn.kind == "swa"), None)
        self.lanes = [None] * L
        self.bsi = self._alloc_b()
        self.btraces, self.bout = {}, {}
        # lane block one-hots (eager loads / unloads)
        self._M = {}
        self.min_width = 0  # warm-up: force a score width bucket (any width >= the need is exact: rows are masked)

    # ------------------------------------------------------------------ persistent inputs
    def _alloc_b(self):
        cfg, L, R = self.cfg, self.L, ROWS
        z = torch.zeros
        si = BatchInputs(L=L)
        si.tok = self._dev(z(1, R, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        for k in ("sparse", "swa"):
            si.cos_sin[k] = (self._dev(z(1, 1, R, cfg.qk_rope), ttnn.float32, ttnn.TILE_LAYOUT),
                             self._dev(z(1, 1, R, cfg.qk_rope), ttnn.float32, ttnn.TILE_LAYOUT))
        for li in self.model.engram_li():
            si.engram[li] = self._dev(z(1, 1, R, 1536), ttnn.bfloat16, ttnn.TILE_LAYOUT)
        si.A = self._dev(z(1, 1, 64 * L, 64 * L), ttnn.bfloat16, ttnn.TILE_LAYOUT)
        si.P = self._dev(z(1, 1, 64 * L, R), ttnn.bfloat16, ttnn.TILE_LAYOUT)
        si.Ag = self._dev(z(1, 1, 64 * L, 64 * L), ttnn.bfloat16, ttnn.TILE_LAYOUT)
        si.Pg = self._dev(z(1, 1, 64 * L, 32 * L), ttnn.bfloat16, ttnn.TILE_LAYOUT)
        n = -(-2 * L // 8) * 8
        for k in ("sparse", "swa", "grp"):
            si.lane_offs[k] = self._dev(z(1, 1, 1, n, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        si.gcos = self._dev(z(1, 1, 32 * L, cfg.qk_rope), ttnn.float32, ttnn.TILE_LAYOUT)
        si.gsin = self._dev(z(1, 1, 32 * L, cfg.qk_rope), ttnn.float32, ttnn.TILE_LAYOUT)
        si.idx_swa = self._dev(z(1, 1, R, self.K_swa, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        for name in ("lo", "hi", "delta", "nmask"):
            setattr(si, name, self._dev(z(1, 1, R, 1), ttnn.float32, ttnn.TILE_LAYOUT))
        si.tail_f = self._dev(z(1, 1, R, self.K_dense - self.n_sel), ttnn.float32, ttnn.TILE_LAYOUT)
        si.dense_f = self._dev(z(1, 1, R, self.K_dense), ttnn.float32, ttnn.TILE_LAYOUT)
        return si

    def _onehots(self, lane):
        """(Mk [L64, L64] identity without lane's block, Mp [L64, 64] placement, Ms [64, L64] selection) bf16."""
        if lane not in self._M:
            L = self.L
            Mk = torch.eye(64 * L)
            Mk[64 * lane : 64 * lane + 64, 64 * lane : 64 * lane + 64] = 0
            Mp = torch.zeros(64 * L, 64)
            Mp[64 * lane : 64 * lane + 64] = torch.eye(64)
            up = lambda t: self._dev(t.reshape(1, 1, *t.shape), ttnn.bfloat16, ttnn.TILE_LAYOUT)  # noqa: E731
            self._M[lane] = (up(Mk), up(Mp), up(Mp.T.contiguous()))
        return self._M[lane]

    # ------------------------------------------------------------------ lanes
    def add(self, slot: int, pos: int, tokens) -> int:
        """Sequence in `slot` (prefilled to `pos` tokens; `tokens` its ids so far) -> a lane (returned)."""
        lane = self.lanes.index(None)
        m = self.model
        m.set_slot(slot)
        nb = (pos // 32) * 32
        gb = ((nb // self.cfg.idx_ratio) // 32) * 32
        Mk, Mp, _ = self._onehots(lane)
        for layer in m.layers:
            layer.attn.init_stage(nb)
            layer.attn.lane_load(self.L, lane, Mk, Mp)
            if layer.attn.indexer is not None:
                layer.attn.indexer.init_stage(nb, gb)
                layer.attn.indexer.lane_load(self.L, lane, Mk, Mp)
        eng = m.engram
        es = (dict(eng.hist), dict(eng.pos)) if eng is not None and hasattr(eng, "hist") else None
        self.lanes[lane] = {"slot": slot, "pos": pos, "ob": nb, "ogb": gb, "tokens": [int(t) for t in tokens],
                            "eng": es, "base": {"sparse": self.a0.base, "swa": self.s0.base if self.s0 else 0}}
        return lane

    def remove(self, lane: int):
        """Write the lane's windows / group keys back to its slot's single-path state and free the lane."""
        ln = self.lanes[lane]
        m = self.model
        m.set_slot(ln["slot"])
        _, _, Ms = self._onehots(lane)
        for layer in m.layers:
            layer.attn.lane_unload(self.L, lane, Ms)
            if layer.attn.indexer is not None:
                layer.attn.indexer.lane_unload(self.L, lane, Ms)
        m.pos = ln["pos"]
        eng = m.engram
        if ln["eng"] is not None:
            eng.hist, eng.pos = dict(ln["eng"][0]), dict(ln["eng"][1])
        self.lanes[lane] = None

    def drop(self, lane: int):
        """Free a finished sequence's lane (no write-back: its slot is re-prefilled before any reuse)."""
        self.lanes[lane] = None

    # ------------------------------------------------------------------ host side of a pass
    def _host_b(self, rows: dict):
        cfg, L, R0 = self.cfg, self.L, ROWS
        r = cfg.idx_ratio
        gcap, ring = self.gcap, self.ring
        order = [(l, list(rows[l])) for l in sorted(rows)]
        R = sum(len(ids) for _, ids in order)
        assert R <= R0
        h = {}
        tok = torch.zeros(1, R0, dtype=torch.int32)
        pos = torch.zeros(R0, dtype=torch.long)
        lane_of = np.full(R0, -1)
        A = torch.zeros(64 * L, 64 * L)
        P = torch.zeros(64 * L, R0)
        Ag = torch.zeros(64 * L, 64 * L)
        Pg = torch.zeros(64 * L, 32 * L)
        offs = {k: np.zeros(-(-2 * L // 8) * 8, dtype=np.int32) for k in ("sparse", "swa", "grp")}
        gpos = torch.zeros(32 * L, dtype=torch.long)
        s = 0
        active = {l for l, _ in order}
        for l in range(L):
            if l not in active:  # idle lane: windows kept, slabs to the trash rows
                A[64 * l : 64 * l + 64, 64 * l : 64 * l + 64] = torch.eye(64)
                Ag[64 * l : 64 * l + 64, 64 * l : 64 * l + 64] = torch.eye(64)
                offs["sparse"][2 * l : 2 * l + 2] = self.a0.trash_row
                offs["swa"][2 * l : 2 * l + 2] = self.s0.trash_row if self.s0 else 0
                offs["grp"][2 * l : 2 * l + 2] = L * gcap
        for l, ids in order:
            ln = self.lanes[l]
            p0, n = ln["pos"], len(ids)
            nb = (p0 // 32) * 32
            assert p0 + n <= nb + 64, "rows must fit the 64-row staging window"
            gw0 = nb // r
            gb = (gw0 // 32) * 32
            tok[0, s : s + n] = torch.tensor(ids, dtype=torch.int32)
            pos[s : s + n] = torch.arange(p0, p0 + n)
            lane_of[s : s + n] = l
            for i in range(64):
                q = nb + i
                if p0 <= q < p0 + n:
                    P[64 * l + i, s + q - p0] = 1
                elif q < p0 and ln["ob"] <= q < ln["ob"] + 64:
                    A[64 * l + i, 64 * l + q - ln["ob"]] = 1
                g = gb + i
                if gw0 <= g < gw0 + 16:
                    Pg[64 * l + i, 32 * l + g - gw0] = 1
                elif ln["ogb"] <= g < ln["ogb"] + 64:
                    Ag[64 * l + i, 64 * l + g - ln["ogb"]] = 1
            bs, bw = ln["base"]["sparse"], ln["base"]["swa"]
            offs["sparse"][2 * l : 2 * l + 2] = [bs + nb, bs + nb + 32]
            offs["swa"][2 * l : 2 * l + 2] = [bw + nb % ring, bw + (nb + 32) % ring] if ring else [0, 0]
            offs["grp"][2 * l : 2 * l + 2] = [l * gcap + gb, l * gcap + gb + 32]
            gpos[32 * l : 32 * l + 16] = torch.arange(nb, nb + 64, r)
            ln["nb"], ln["gb"] = nb, gb
            s += n
        pos[R:] = pos[R - 1] if R else 0
        h["tok"] = _rm(tok, ttnn.uint32)
        for k in ("sparse", "swa"):
            c, sn = cos_sin(pos, cfg.rope_theta[k], cfg.qk_rope)
            h[f"cos_{k}"] = _tile(c.float().reshape(1, 1, R0, -1), ttnn.float32)
            h[f"sin_{k}"] = _tile(sn.float().reshape(1, 1, R0, -1), ttnn.float32)
        for name, t in (("A", A), ("P", P), ("Ag", Ag), ("Pg", Pg)):
            h[name] = _tile(t.reshape(1, 1, *t.shape), ttnn.bfloat16)
        for k, v in offs.items():
            h[f"offs_{k}"] = _rm(torch.from_numpy(v).reshape(1, 1, 1, -1), ttnn.uint32)
        gc, gs = cos_sin(gpos, cfg.rope_theta["sparse"], cfg.qk_rope)
        h["gcos"] = _tile(gc.float().reshape(1, 1, 32 * L, -1), ttnn.float32)
        h["gsin"] = _tile(gs.float().reshape(1, 1, 32 * L, -1), ttnn.float32)
        # per-row lists
        swa = np.full((R0, self.K_swa), SENTINEL, dtype=np.uint32)
        swa[:, 0] = self.sink_swa
        lo = torch.zeros(R0, 1)
        hi = torch.zeros(R0, 1)
        delta = torch.zeros(R0, 1)
        nmask = torch.zeros(R0, 1)
        tail = torch.full((R0, self.K_dense - self.n_sel), SENT_F)
        tail[:, 0] = float(self.sink_sparse)
        dense = torch.zeros(R0, self.K_dense)
        need = 0
        for i in range(R0):
            if i >= R:  # padding rows: the sink only
                dense[i, :] = SENT_F
                dense[i, 0] = float(self.sink_sparse)
                continue
            l = int(lane_of[i])
            ln = self.lanes[l]
            p = int(pos[i])
            bs, bw = ln["base"]["sparse"], ln["base"]["swa"]
            w = np.arange(max(0, p - cfg.window + 1), p + 1)
            swa[i, 1 : 1 + w.size] = bw + (w % ring if ring else w)
            vis = p // r
            lo[i, 0], hi[i, 0] = l * gcap, l * gcap + vis
            need = max(need, l * gcap + vis)
            delta[i, 0] = bs - r * l * gcap
            if p < cfg.idx_candidate_topk:
                dense[i, :] = SENT_F
                dense[i, 0] = float(self.sink_sparse)
                dense[i, 1 : 2 + p] = (bs + torch.arange(0, p + 1)).float()
            else:
                nmask[i, 0] = 1.0
                own = torch.arange((p // r) * r, p + 1)
                tail[i, 1 : 1 + own.numel()] = (bs + own).float()
        h["idx_swa"] = _rm(torch.from_numpy(swa.view(np.int32)).reshape(1, 1, R0, -1), ttnn.uint32)
        for name, t in (("lo", lo), ("hi", hi), ("delta", delta), ("nmask", nmask)):
            h[name] = _tile(t.reshape(1, 1, R0, 1), ttnn.float32)
        h["tail_f"] = _tile(tail.reshape(1, 1, R0, -1), ttnn.float32)
        h["dense_f"] = _tile(dense.reshape(1, 1, R0, -1), ttnn.float32)
        top = L * gcap + 32
        need = max(need, min(self.min_width, L * gcap))
        width = next((b for b in (2048 * 2**i for i in range(12)) if b >= need and b + 32 <= top), None)
        width = width or (L * gcap)
        # Engram rows per lane (its own n-gram history)
        eng = self.model.engram
        for li in self.model.engram_li():
            e = torch.zeros(1, 1, R0, 1536, dtype=torch.bfloat16)
            h[f"eng_{li}"] = e
        if eng is not None and self.model.engram_li():
            saved = (dict(eng.hist), dict(eng.pos)) if hasattr(eng, "hist") else None
            s = 0
            for l, ids in order:
                ln = self.lanes[l]
                if ln["eng"] is not None:
                    eng.hist, eng.pos = dict(ln["eng"][0]), dict(ln["eng"][1])
                for li in self.model.engram_li():
                    h[f"eng_{li}"][0, 0, s : s + len(ids)] = eng.rows(li, torch.tensor(ids), ln["pos"])
                if ln["eng"] is not None:
                    ln["eng"] = (dict(eng.hist), dict(eng.pos))
                s += len(ids)
            if saved is not None:
                eng.hist, eng.pos = saved
        for li in self.model.engram_li():
            h[f"eng_{li}"] = _tile(h[f"eng_{li}"], ttnn.bfloat16)
        self._sel = bool(nmask.any())
        return h, R, width, order

    def _copy_b(self, h):
        si, c = self.bsi, ttnn.copy_host_to_device_tensor
        c(h["tok"], si.tok)
        for k in ("sparse", "swa"):
            c(h[f"cos_{k}"], si.cos_sin[k][0])
            c(h[f"sin_{k}"], si.cos_sin[k][1])
        for name in ("A", "P", "Ag", "Pg", "gcos", "gsin", "idx_swa", "lo", "hi", "delta", "nmask", "tail_f", "dense_f"):
            c(h[name], getattr(si, name))
        for k in ("sparse", "swa", "grp"):
            c(h[f"offs_{k}"], si.lane_offs[k])
        for li, t in si.engram.items():
            c(h[f"eng_{li}"], t)

    # ------------------------------------------------------------------ a pass
    def run_batch(self, rows: dict) -> dict:
        """rows: {lane: [token ids]} (each lane continues its sequence) -> {lane: logits fp32 [n, V]}."""
        tm = os.environ.get("HY4_TIME_BATCH") == "1"
        t0 = time.perf_counter()
        h, R, width, order = self._host_b(rows)
        t1 = time.perf_counter()
        self._copy_b(h)
        if tm:
            ttnn.synchronize_device(self.mesh)
        t2 = time.perf_counter()
        si = self.bsi
        si.R, si.sel = R, self._sel
        si.width = width if si.sel else 0
        key = ("b", R, si.width)
        if not self.trace:
            lg = self.model.step(si, R, mem=self.mem)
            out = self.model.read_logits(lg, R)
        else:
            self._guard_programs("before a batched pass")
            if key not in self.btraces:
                lg = self.model.step(si, R, mem=self.mem)
                out = self.model.read_logits(lg, R)
                self._guard_programs("the eager pass of a new batched trace")
                logger.info(f"capturing batched decode trace {key} ...")
                tid = ttnn.begin_trace_capture(self.mesh, cq_id=0)
                self.bout[key] = self.model.step(si, R, mem=self.mem)
                ttnn.end_trace_capture(self.mesh, tid, cq_id=0)
                self.btraces[key] = tid
                self._prog_n = self._programs()
            else:
                ttnn.execute_trace(self.mesh, self.btraces[key], cq_id=0, blocking=False)
                out = self.model.read_logits(self.bout[key], R)
        if tm:
            t3 = time.perf_counter()
            self._tm = getattr(self, "_tm", [])
            self._tm.append((t1 - t0, t2 - t1, t3 - t2))
            if len(self._tm) % 64 == 0:
                med = [sorted(x[i] for x in self._tm[-64:])[32] * 1e3 for i in range(3)]
                logger.info(f"batched pass R={R}: host {med[0]:.2f} ms, copy {med[1]:.2f} ms, device+read {med[2]:.2f} ms")
        res, s = {}, 0
        for l, ids in order:
            n = len(ids)
            res[l] = out[s : s + n]
            ln = self.lanes[l]
            ln["pos"] += n
            ln["ob"], ln["ogb"] = ln["nb"], ln["gb"]
            ln["tokens"] += [int(t) for t in ids]
            s += n
        return res

    def _has_traces(self):
        return bool(self.traces) or bool(getattr(self, "btraces", None))

    def release(self):
        super().release()
        if getattr(self, "btraces", None):
            ttnn.synchronize_device(self.mesh)
            for tid in self.btraces.values():
                ttnn.release_trace(self.mesh, tid)
            self.btraces, self.bout = {}, {}
