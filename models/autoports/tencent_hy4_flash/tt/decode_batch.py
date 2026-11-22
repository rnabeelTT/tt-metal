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
from models.autoports.tencent_hy4_flash.tt.decode import ROWS, Hy4Decoder, Hy4SpecDecoder, _rm, _tile
from models.autoports.tencent_hy4_flash.tt.rope import POSMAPS, cos_sin, rpos_rows

SENT_F = float(2**32)
# KV rows written straight to each slot's cache (one slab_write of 1-row slabs at per-row offsets) instead of the
# lanes' 64-row staging windows (block-diagonal one-hot matmuls [64 L, 64 L] x [64 L, 608] per layer, ~105 us at L = 8)
LANE_ROWS = os.environ.get("HY4_LANE_ROWS", "1") == "1"
# Indexer pooling of only the groups a pass can change: per lane gw0 .. the last row's group (<= 8 for one row a lane),
# gathered by a per-pass one-hot Esel [r n, 64 L] (n = max(32, 8 L) pooled rows) instead of every lane's 16 window
# groups (constant Ecat [r 32 L, 64 L], n = 32 L). Same values for every visible group (recomputed from the same rows).
LANE_POOL = os.environ.get("HY4_LANE_POOL", "1") == "1"
PRECOPY = os.environ.get("HY4_PRECOPY", "1") == "1"


def pool_rows(L):
    """(pooled rows n, rows per lane M) of the batched indexer pooling."""
    if LANE_POOL:
        n = max(32, 8 * L)
        return n, n // L
    return 32 * L, 32


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
    row_offs: dict = field(default_factory=dict)  # LANE_ROWS: per row, its KV cache row (padding rows: trash)
    lane_rows: bool = False
    Esel: object = None
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
        npool, _ = pool_rows(L)
        si.Pg = self._dev(z(1, 1, 64 * L, npool), ttnn.bfloat16, ttnn.TILE_LAYOUT)
        if LANE_POOL:
            si.Esel = self._dev(z(1, 1, cfg.idx_ratio * npool, 64 * L), ttnn.bfloat16, ttnn.TILE_LAYOUT)
        n = -(-2 * L // 8) * 8
        for k in ("sparse", "swa", "grp"):
            si.lane_offs[k] = self._dev(z(1, 1, 1, n, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        si.lane_rows = LANE_ROWS
        for k in ("sparse", "swa"):
            si.row_offs[k] = self._dev(z(1, 1, 1, R, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        si.gcos = self._dev(z(1, 1, npool, cfg.qk_rope), ttnn.float32, ttnn.TILE_LAYOUT)
        si.gsin = self._dev(z(1, 1, npool, cfg.qk_rope), ttnn.float32, ttnn.TILE_LAYOUT)
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
    def add(self, slot: int, pos: int, tokens, stage=None) -> int:
        """Sequence in `slot` (prefilled to `pos` tokens; `tokens` its ids so far) -> a lane (returned).
        stage=(ob, ogb): the sequence decoded on the single path since; its staging windows (bases ob / ogb) are the
        device state (the host prefill state lacks the decoded tokens' keys: re-initializing from it zeroed their token
        keys and restored stale group keys over the decoded groups)."""
        lane = self.lanes.index(None)
        m = self.model
        m.set_slot(slot)
        if stage is None:
            nb = (pos // 32) * 32
            gb = ((nb // self.cfg.idx_ratio) // 32) * 32
        else:
            nb, gb = stage
        Mk, Mp, _ = self._onehots(lane)
        for layer in m.layers:
            if not LANE_ROWS:  # (LANE_ROWS: the KV rows go straight to the cache; no attention window in a lane)
                if stage is None:
                    layer.attn.init_stage(nb)
                layer.attn.lane_load(self.L, lane, Mk, Mp)
            if layer.attn.indexer is not None:
                if stage is None:
                    layer.attn.indexer.init_stage(nb, gb)
                layer.attn.indexer.lane_load(self.L, lane, Mk, Mp)
        eng = m.engram
        es = None
        if eng is not None and hasattr(eng, "hist"):
            # the n-gram history of `tokens` (a single-path verify pass may have fed rejected rows past pos)
            keep = int(getattr(eng, "keep", 2))
            tl = [int(t) for t in tokens]
            es = ({li: np.asarray(tl[max(0, pos - keep) : pos], dtype=np.int64) for li in eng.hist},
                  {li: int(pos) for li in eng.pos})
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
            if LANE_ROWS:  # the single path's window from the cache (every row of the lane's passes is written there)
                layer.attn.init_stage(ln["ob"])
            else:
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
    def _host_b(self, rows: dict, with_tok: bool = True):
        """Every per-pass host input (vectorized; ~/m27-work/hy4/hostb_bench.py checks it bit-exact). with_tok=False:
        only the position-dependent ones (no token ids / Engram rows: _host_tok adds them; run_batch prebuilds the next
        pass this way while the device runs)."""
        cfg, L, R0 = self.cfg, self.L, ROWS
        r = cfg.idx_ratio
        gcap, ring = self.gcap, self.ring
        order = [(l, list(rows[l])) for l in sorted(rows)]
        R = sum(len(ids) for _, ids in order)
        assert R <= R0
        h = {}
        tok = np.zeros((1, R0), dtype=np.int32)
        pos = np.zeros(R0, dtype=np.int64)
        # per-row lane data (rows < R): lane, sparse base, swa base
        row_l = np.zeros(R0, dtype=np.int64)
        row_bs = np.zeros(R0, dtype=np.int64)
        row_bw = np.zeros(R0, dtype=np.int64)
        nL = 64 * L
        A = np.zeros((nL, nL), dtype=np.float32)
        P = np.zeros((nL, R0), dtype=np.float32)
        Ag = np.zeros((nL, nL), dtype=np.float32)
        npool, M = pool_rows(L)
        Pg = np.zeros((nL, npool), dtype=np.float32)
        Esel = np.zeros((r * npool, nL), dtype=np.float32) if LANE_POOL else None
        no = -(-2 * L // 8) * 8
        offs = {k: np.zeros(no, dtype=np.int32) for k in ("sparse", "swa", "grp")}
        gpos = np.zeros(npool, dtype=np.int64)
        i64 = np.arange(64)
        active = {l for l, _ in order}
        idle = [l for l in range(L) if l not in active]
        if idle:  # idle lanes: windows kept (identity blocks), slabs to the trash rows
            d = (64 * np.asarray(idle)[:, None] + i64[None, :]).ravel()
            A[d, d] = 1
            Ag[d, d] = 1
            t_s, t_w = self.a0.trash_row, (self.s0.trash_row if self.s0 else 0)
            for l in idle:
                offs["sparse"][2 * l : 2 * l + 2] = t_s
                offs["swa"][2 * l : 2 * l + 2] = t_w
                offs["grp"][2 * l : 2 * l + 2] = L * gcap
        s = 0
        for l, ids in order:
            ln = self.lanes[l]
            p0, n = ln["pos"], len(ids)
            nb = (p0 // 32) * 32
            assert p0 + n <= nb + 64, "rows must fit the 64-row staging window"
            gw0 = nb // r
            gb = (gw0 // 32) * 32
            ob, ogb = ln["ob"], ln["ogb"]
            bs, bw = ln["base"]["sparse"], ln["base"]["swa"]
            if n:
                tok[0, s : s + n] = ids
                pos[s : s + n] = np.arange(p0, p0 + n)
                row_l[s : s + n] = l
                row_bs[s : s + n] = bs
                row_bw[s : s + n] = bw
            q = nb + i64
            mP = (q >= p0) & (q < p0 + n)
            mA = ~mP & (q < p0) & (q >= ob) & (q < ob + 64)
            P[64 * l + i64[mP], s + q[mP] - p0] = 1
            A[64 * l + i64[mA], 64 * l + q[mA] - ob] = 1
            g = gb + i64
            if LANE_POOL:  # the groups gw0 .. of the lane's last row (pooled row M l + i: group gw0 + i)
                ng = (p0 + max(n, 1) - 1) // r - gw0 + 1
                assert ng <= M, f"lane {l}: {ng} groups to pool > {M}"
                mPg = (g >= gw0) & (g < gw0 + ng)
                Pg[64 * l + i64[mPg], M * l + g[mPg] - gw0] = 1
                for j in range(r):  # slot j of pooled row (l, i) = token row r (gw0 + i) + j of the lane's window
                    Esel[j * npool + M * l + np.arange(ng), 64 * l + r * (gw0 + np.arange(ng)) + j - nb] = 1
                gpos[M * l : M * l + ng] = r * (gw0 + np.arange(ng))
            else:
                mPg = (g >= gw0) & (g < gw0 + 16)
                Pg[64 * l + i64[mPg], 32 * l + g[mPg] - gw0] = 1
                gpos[32 * l : 32 * l + 16] = np.arange(nb, nb + 64, r)
            mAg = ~mPg & (g >= ogb) & (g < ogb + 64)
            Ag[64 * l + i64[mAg], 64 * l + g[mAg] - ogb] = 1
            offs["sparse"][2 * l : 2 * l + 2] = [bs + nb, bs + nb + 32]
            offs["swa"][2 * l : 2 * l + 2] = [bw + nb % ring, bw + (nb + 32) % ring] if ring else [0, 0]
            offs["grp"][2 * l : 2 * l + 2] = [l * gcap + gb, l * gcap + gb + 32]
            ln["nb"], ln["gb"] = nb, gb
            s += n
        pos[R:] = pos[R - 1] if R else 0
        t_s, t_w = self.a0.trash_row, (self.s0.trash_row if self.s0 else 0)
        h["roffs_sparse"] = _rm(torch.from_numpy(np.where(np.arange(R0) < R, row_bs + pos, t_s).astype(np.int32))
                                .reshape(1, 1, 1, R0), ttnn.uint32)
        h["roffs_swa"] = _rm(torch.from_numpy(np.where(np.arange(R0) < R, row_bw + (pos % ring if ring else 0), t_w)
                                              .astype(np.int32)).reshape(1, 1, 1, R0), ttnn.uint32)
        pos_t = torch.from_numpy(pos)
        if POSMAPS:  # image prompts: every row's rope planes through its sequence's map
            rl = row_l[np.minimum(np.arange(R0), max(R - 1, 0))]
            pos_t = rpos_rows(pos_t, [self.lanes[int(x)]["slot"] if R and self.lanes[int(x)] else -1 for x in rl])
        for k in ("sparse", "swa"):
            c, sn = cos_sin(pos_t, cfg.rope_theta[k], cfg.qk_rope)
            h[f"cos_{k}"] = _tile(c.float().reshape(1, 1, R0, -1), ttnn.float32)
            h[f"sin_{k}"] = _tile(sn.float().reshape(1, 1, R0, -1), ttnn.float32)
        for name, t in (("A", A), ("P", P), ("Ag", Ag), ("Pg", Pg)):
            h[name] = _tile(torch.from_numpy(t).to(torch.bfloat16).reshape(1, 1, *t.shape), ttnn.bfloat16)
        for k, v in offs.items():
            h[f"offs_{k}"] = _rm(torch.from_numpy(v).reshape(1, 1, 1, -1), ttnn.uint32)
        gpos_t = torch.from_numpy(gpos)
        if POSMAPS:
            gpos_t = rpos_rows(gpos_t, [self.lanes[i // M]["slot"] if self.lanes[i // M] else -1  # (M rows a lane)
                                        for i in range(gpos_t.numel())])
        gc, gs = cos_sin(gpos_t, cfg.rope_theta["sparse"], cfg.qk_rope)
        h["gcos"] = _tile(gc.float().reshape(1, 1, npool, -1), ttnn.float32)
        h["gsin"] = _tile(gs.float().reshape(1, 1, npool, -1), ttnn.float32)
        if LANE_POOL:
            h["Esel"] = _tile(torch.from_numpy(Esel).to(torch.bfloat16).reshape(1, 1, *Esel.shape), ttnn.bfloat16)
        # per-row lists (rows >= R: padding, the sink only)
        K_swa, K_dense, n_sel = self.K_swa, self.K_dense, self.n_sel
        W, topk = cfg.window, cfg.idx_candidate_topk
        swa = np.full((R0, K_swa), SENTINEL, dtype=np.uint32)
        swa[:, 0] = self.sink_swa
        lo = np.zeros((R0, 1), dtype=np.float32)
        hi = np.zeros((R0, 1), dtype=np.float32)
        delta = np.zeros((R0, 1), dtype=np.float32)
        nmask = np.zeros((R0, 1), dtype=np.float32)
        tail = np.full((R0, K_dense - n_sel), SENT_F, dtype=np.float32)
        tail[:, 0] = float(self.sink_sparse)
        dense = np.zeros((R0, K_dense), dtype=np.float32)
        dense[R:, :] = SENT_F
        dense[R:, 0] = float(self.sink_sparse)
        need = 0
        if R:
            p, rl, rbs, rbw = pos[:R], row_l[:R], row_bs[:R], row_bw[:R]
            # sliding window: positions max(0, p - W + 1) .. p
            st = np.maximum(0, p - W + 1)
            j = np.arange(W)
            w = st[:, None] + j[None, :]
            ok = j[None, :] < (p - st + 1)[:, None]
            v = rbw[:, None] + (w % ring if ring else w)
            sub = swa[:R, 1 : 1 + W]
            sub[ok] = v[ok]
            vis = p // r
            lg = rl * gcap
            lo[:R, 0] = lg
            hi[:R, 0] = lg + vis
            need = max(0, int((lg + vis).max()))
            delta[:R, 0] = rbs - r * lg
            dn = p < topk
            if dn.any():  # dense regime rows: sink, then 0 .. p
                pd, bd = p[dn], rbs[dn]
                jd = np.arange(K_dense - 1)
                blk = np.full((pd.size, K_dense), SENT_F, dtype=np.float32)
                blk[:, 0] = float(self.sink_sparse)
                okd = jd[None, :] < (pd + 1)[:, None]
                vd = (bd[:, None] + jd[None, :]).astype(np.float32)
                bsub = blk[:, 1:]
                bsub[okd] = vd[okd]
                dense[:R][dn] = blk
            sl = ~dn
            if sl.any():  # selecting rows: tail = sink, own group's positions
                nmask[:R, 0][sl] = 1.0
                ps, bss = p[sl], rbs[sl]
                o0 = (ps // r) * r
                jt = np.arange(K_dense - n_sel - 1)
                okt = jt[None, :] < (ps - o0 + 1)[:, None]
                vt = (bss[:, None] + o0[:, None] + jt[None, :]).astype(np.float32)
                tsub = tail[:R][sl]
                ts = tsub[:, 1:]
                ts[okt] = vt[okt]
                tail[:R][sl] = tsub
        h["idx_swa"] = _rm(torch.from_numpy(swa.view(np.int32)).reshape(1, 1, R0, -1), ttnn.uint32)
        for name, t in (("lo", lo), ("hi", hi), ("delta", delta), ("nmask", nmask)):
            h[name] = _tile(torch.from_numpy(t).reshape(1, 1, R0, 1), ttnn.float32)
        h["tail_f"] = _tile(torch.from_numpy(tail).reshape(1, 1, R0, -1), ttnn.float32)
        h["dense_f"] = _tile(torch.from_numpy(dense).reshape(1, 1, R0, -1), ttnn.float32)
        top = L * gcap + 32
        need = max(need, min(self.min_width, L * gcap))
        width = next((b for b in (2048 * 2**i for i in range(12)) if b >= need and b + 32 <= top), None)
        width = width or (L * gcap)
        if with_tok:
            self._host_tok(order, h)
        self._sel = bool(nmask.any())
        return h, R, width, order

    def _host_tok(self, order, h):
        """The token-dependent inputs of a pass: token ids and every lane's Engram rows (its own n-gram history)."""
        R0 = ROWS
        tok = np.zeros((1, R0), dtype=np.int32)
        s = 0
        for _, ids in order:
            tok[0, s : s + len(ids)] = ids
            s += len(ids)
        h["tok"] = _rm(torch.from_numpy(tok), ttnn.uint32)
        eng = self.model.engram
        eli = self.model.engram_li()
        for li in eli:
            h[f"eng_{li}"] = torch.zeros(1, 1, R0, 1536, dtype=torch.bfloat16)
        if eng is not None and eli and _eng_batched(eng, eli, order, self.lanes, h):
            pass
        elif eng is not None and eli:
            saved = (dict(eng.hist), dict(eng.pos)) if hasattr(eng, "hist") else None
            s = 0
            for l, ids in order:
                ln = self.lanes[l]
                if ln["eng"] is not None:
                    eng.hist, eng.pos = dict(ln["eng"][0]), dict(ln["eng"][1])
                for li in eli:
                    h[f"eng_{li}"][0, 0, s : s + len(ids)] = eng.rows(li, torch.tensor(ids), ln["pos"])
                if ln["eng"] is not None:
                    ln["eng"] = (dict(eng.hist), dict(eng.pos))
                s += len(ids)
            if saved is not None:
                eng.hist, eng.pos = saved
        for li in eli:
            h[f"eng_{li}"] = _tile(h[f"eng_{li}"], ttnn.bfloat16)

    def _snap(self, rows: dict):
        """What the position-dependent inputs of a pass depend on (a prebuilt pass is used only on an exact match)."""
        return (self.min_width,) + tuple(
            (l, len(rows[l]), self.lanes[l]["pos"], self.lanes[l]["ob"], self.lanes[l]["ogb"], self.lanes[l]["slot"],
             self.lanes[l]["base"]["sparse"], self.lanes[l]["base"]["swa"]) for l in sorted(rows))

    def _copy_b(self, h):
        self._copy_tok(h)
        self._copy_pos(h)

    def _copy_tok(self, h):
        """The token-dependent inputs (token ids, Engram rows)."""
        si, c = self.bsi, ttnn.copy_host_to_device_tensor
        c(h["tok"], si.tok)
        for li, t in si.engram.items():
            c(h[f"eng_{li}"], t)

    def _copy_pos(self, h):
        """The position-dependent inputs (everything _host_b(with_tok=False) builds)."""
        si, c = self.bsi, ttnn.copy_host_to_device_tensor
        for k in ("sparse", "swa"):
            c(h[f"cos_{k}"], si.cos_sin[k][0])
            c(h[f"sin_{k}"], si.cos_sin[k][1])
        for name in ("A", "P", "Ag", "Pg", "gcos", "gsin", "idx_swa", "lo", "hi", "delta", "nmask", "tail_f", "dense_f"):
            c(h[name], getattr(si, name))
        for k in ("sparse", "swa", "grp"):
            c(h[f"offs_{k}"], si.lane_offs[k])
        for k in ("sparse", "swa"):
            c(h[f"roffs_{k}"], si.row_offs[k])
        if si.Esel is not None:
            c(h["Esel"], si.Esel)

    # ------------------------------------------------------------------ a pass
    def run_batch(self, rows: dict) -> dict:
        """rows: {lane: [token ids]} (each lane continues its sequence) -> {lane: logits fp32 [n, V]}."""
        return self.collect(self.launch(rows))

    def collect(self, hd) -> dict:
        """The logits of a launched pass (waits for the device): {lane: logits fp32 [n, V]}."""
        out = hd["out"]
        if out is None:
            out = self.model.read_logits(self.bout[hd["key"]], hd["R"])
        if hd["tm"] is not None:
            t0, t1, t2 = hd["tm"]
            t3 = time.perf_counter()
            self._tm = getattr(self, "_tm", [])
            self._tm.append((t1 - t0, t2 - t1, t3 - t2))
            if len(self._tm) % 64 == 0:
                med = [sorted(x[i] for x in self._tm[-64:])[32] * 1e3 for i in range(3)]
                logger.info(f"batched pass R={hd['R']}: host {med[0]:.2f} ms, copy {med[1]:.2f} ms, device+read {med[2]:.2f} ms; "
                            f"prebuilt passes so far {getattr(self, 'n_prebuilt', 0)}")
        self.last_out = (out, [l for l, _ in hd["order"]])  # (engine_batch: one sampling pass over all rows)
        res, s = {}, 0
        for l, ids in hd["order"]:
            n = len(ids)
            res[l] = out[s : s + n]
            s += n
        return res

    def launch(self, rows: dict) -> dict:
        """The host inputs, copies and the (traced: non-blocking) pass of rows {lane: [token ids]}; the lanes advance.
        -> a handle for collect(). A replayed pass's logits stay in its trace output until collect(): no other pass of
        this decoder may run in between."""
        tm = os.environ.get("HY4_TIME_BATCH", "0") in ("1", "2")  # 2: no device sync (copy = host enqueue time only)
        t0 = time.perf_counter()
        pre, self._pre = getattr(self, "_pre", None), None
        if pre is not None and pre[0] == self._snap(rows):  # prebuilt during the last pass: add the token inputs
            h, R, width, order, sel, nbgb = pre[1]
            order = [(l, list(rows[l])) for l in sorted(rows)]
            for l, (nb, gb) in nbgb.items():
                self.lanes[l]["nb"], self.lanes[l]["gb"] = nb, gb
            self._host_tok(order, h)
            self._sel = sel
            self.n_prebuilt = getattr(self, "n_prebuilt", 0) + 1
            copied = pre[2]
        else:
            h, R, width, order = self._host_b(rows)
            copied = False
        t1 = time.perf_counter()
        if copied:  # the position inputs went in behind the last pass (in-order queue): only the token inputs now
            self._copy_tok(h)
        else:
            self._copy_b(h)
        if tm and os.environ.get("HY4_TIME_BATCH") == "1":
            ttnn.synchronize_device(self.mesh)
        t2 = time.perf_counter()
        si = self.bsi
        si.R, si.sel = R, self._sel
        si.width = width if si.sel else 0
        key = ("b", R, si.width)
        advanced = False
        out = None
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
                self._advance(order)
                # exact (batch gate 0.8650 / 0.2229 unchanged); 8 users 45.80 -> 44.64 ms per step (no Engram), 4: 39.60 ->
                # 38.46, 2: 35.94 -> 35.03 (2026-11-20)
                if os.environ.get("HY4_PREBUILD", "1") == "1":
                    self._prebuild(order)  # the next pass's position inputs while the device runs this one
                    # HY4_PRECOPY=1 (default): and their copies, queued behind this pass (the queue is in order: the
                    # pass reads the current inputs first); the next pass then copies only its token / Engram inputs.
                    # Only run_batch writes these inputs, and a pass whose snapshot differs copies them all again.
                    if self._pre is not None and PRECOPY:
                        self._copy_pos(self._pre[1][0])
                        self._pre = self._pre[:2] + (True,)
                advanced = True
        if not advanced:
            self._advance(order)
        return {"key": key, "R": R, "order": order, "out": out, "tm": (t0, t1, t2) if tm else None}

    def _advance(self, order):
        for l, ids in order:
            ln = self.lanes[l]
            ln["pos"] += len(ids)
            ln["ob"], ln["ogb"] = ln["nb"], ln["gb"]
            ln["tokens"] += [int(t) for t in ids]

    def _prebuild(self, order):
        """Position inputs of the likely next pass (the same lanes, one row each)."""
        nxt = {l: [0] for l, _ in order}
        if any(self.lanes[l]["pos"] + 1 > (self.lanes[l]["pos"] // 32) * 32 + 64 for l in nxt):
            return
        snap = self._snap(nxt)
        sel_now = self._sel
        h, R, width, o = self._host_b(nxt, with_tok=False)
        nbgb = {l: (self.lanes[l]["nb"], self.lanes[l]["gb"]) for l in nxt}
        self._pre = (snap, (h, R, width, o, self._sel, nbgb), False)
        self._sel = sel_now

    def _has_traces(self):
        return super()._has_traces() or bool(getattr(self, "btraces", None))

    def release(self):
        super().release()
        if getattr(self, "btraces", None):
            ttnn.synchronize_device(self.mesh)
            for tid in self.btraces.values():
                ttnn.release_trace(self.mesh, tid)
            self.btraces, self.bout = {}, {}


class Hy4BatchSpecDecoder(Hy4BatchDecoder, Hy4SpecDecoder):
    """Hy4BatchDecoder whose single path (a sequence alone: tt/engine_batch.py solo) also drafts (DFly2) and verifies
    k + 1 rows (Hy4SpecDecoder.run / draft)."""

    def __init__(self, model, lanes: int = 8, trace: bool = True, k: int = 4):
        super().__init__(model, lanes=lanes, trace=trace)
        if int(k) != self.k:
            self.set_k(int(k))


def _eng_batched(eng, eli, order, lanes, h):
    """All lanes' Engram rows of every layer: the n-grams once (every layer sees the same tokens), then per layer one
    table gather + dequant + RMSNorm over all lanes (instead of one EngramRows.rows call per lane and layer).
    Row-wise math, so bit-identical to the per-lane calls; per-lane history / position updated exactly as
    EngramRows.rows does. Only for EngramRows-like objects with every active lane holding its own state (ln["eng"] set,
    as add() does); otherwise returns False and the caller takes the per-lane path."""
    store = getattr(eng, "store", None)
    if store is None or not hasattr(eng, "norm_w") or not hasattr(store, "hashers") or not order or not eli:
        return False
    if any(lanes[l]["eng"] is None for l, _ in order):
        return False
    from models.experimental.hybrid_moe.engram_store import FP32_EPS, dequant

    keep = eng.keep
    st = {l: (dict(lanes[l]["eng"][0]), dict(lanes[l]["eng"][1])) for l, _ in order}
    hs0 = store.hashers[eli[0]]
    ngs, idss = [], []
    for l, ids in order:
        hist, pos = st[l]
        p0 = lanes[l]["pos"]
        a = np.asarray(ids, dtype=np.int64).reshape(-1)
        for li in eli:
            if p0 == 0:
                hist[li], pos[li] = np.zeros(0, np.int64), 0
            if p0 != pos[li]:
                raise ValueError(f"engram rows: layer {li} expected position {pos[li]}, got {p0} (call reset())")
        h0 = hist[eli[0]]
        if any(not np.array_equal(hist[li], h0) for li in eli[1:]):  # (never: every layer is fed the same tokens)
            return _eng_batched_layers(eng, eli, order, lanes, h)
        ngs.append(hs0.ngram_tokens(a, p0, h0 if p0 else None))
        idss.append(a)
    ng = np.concatenate(ngs)
    T = ng.shape[0]
    for li in eli:  # (the n-grams depend on the tokens only; hashing / offsets are per layer)
        hs = store.hashers[li]
        rows = hs.rows(ng).reshape(-1)
        q8 = torch.from_numpy(store.q[li][rows])
        sc = torch.from_numpy(store.s[li][rows].astype(np.uint16)).view(torch.bfloat16)[:, None]
        # (per layer: one [T x 16, 96] block stays under torch's intra-op threading threshold; a stacked 4-layer block
        # spun up the thread pool: 0.3 -> 3.5 ms on a loaded host)
        xf = dequant(q8, sc).view(T, hs.moduli.shape[0], -1).float()
        y = (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + FP32_EPS) * eng.norm_w[li].float()).to(torch.bfloat16)
        h[f"eng_{li}"][0, 0, :T] = y.flatten(1)
    for (l, _), a in zip(order, idss):
        hist, pos = st[l]
        for li in eli:
            hist[li] = np.concatenate([hist[li], a])[-keep:]
            pos[li] = lanes[l]["pos"] + len(a)
    for l, _ in order:
        lanes[l]["eng"] = st[l]
    return True


def _eng_batched_layers(eng, eli, order, lanes, h):
    """(Per layer: the previous _eng_batched.) All lanes' Engram rows of a layer in ONE table gather + dequant + RMSNorm (instead of one EngramRows.rows call
    per lane and layer). Row-wise math, so bit-identical to the per-lane calls; per-lane history / position updated
    exactly as EngramRows.rows does. Only for EngramRows-like objects with every active lane holding its own state
    (ln["eng"] set, as add() does); otherwise returns False and the caller takes the per-lane path."""
    store = getattr(eng, "store", None)
    if store is None or not hasattr(eng, "norm_w") or not hasattr(store, "hashers") or not order:
        return False
    if any(lanes[l]["eng"] is None for l, _ in order):
        return False
    from models.experimental.hybrid_moe.engram_store import FP32_EPS, dequant

    keep = eng.keep
    st = {l: (dict(lanes[l]["eng"][0]), dict(lanes[l]["eng"][1])) for l, _ in order}
    for li in eli:
        hs = store.hashers[li]
        ngs, idss = [], []
        for l, ids in order:
            hist, pos = st[l]
            p0 = lanes[l]["pos"]
            a = np.asarray(ids, dtype=np.int64).reshape(-1)
            if p0 == 0:
                hist[li], pos[li] = np.zeros(0, np.int64), 0
            if p0 != pos[li]:
                raise ValueError(f"engram rows: layer {li} expected position {pos[li]}, got {p0} (call reset())")
            ngs.append(hs.ngram_tokens(a, p0, hist[li] if p0 else None))
            idss.append(a)
        ng = np.concatenate(ngs)
        T = ng.shape[0]
        rows = hs.rows(ng)
        q8 = torch.from_numpy(store.q[li][rows.reshape(-1)])
        sc = torch.from_numpy(store.s[li][rows.reshape(-1)].astype(np.uint16)).view(torch.bfloat16)[:, None]
        x = dequant(q8, sc).view(T, hs.moduli.shape[0], -1)
        xf = x.float()
        y = (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + FP32_EPS) * eng.norm_w[li].float()).to(torch.bfloat16)
        h[f"eng_{li}"][0, 0, :T] = y.flatten(1)
        for (l, _), a in zip(order, idss):
            hist, pos = st[l]
            hist[li] = np.concatenate([hist[li], a])[-keep:]
            pos[li] = lanes[l]["pos"] + len(a)
    for l, _ in order:
        lanes[l]["eng"] = st[l]
    return True
