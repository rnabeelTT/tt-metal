# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Multi-user generation engine for Hy4-flash: up to model.slots sequences, one batched decode pass per step.

Each sequence owns a model slot (its cache regions, tt/model.py set_slot) and, while it decodes, a lane of the batched
decoder (tt/decode_batch.py). Prefill runs per sequence on the single path into its slot; decode steps every live
sequence in one pass (one row each). Sampling (temperature / top-k / top-p / seed) runs on the host per sequence over
its fp32 logits (tt/engine.py HostSampler). Solo (HY4_BATCH_SOLO=1, default): while a step has one sequence, it leaves
its lane (Hy4BatchDecoder.remove: windows / group keys back to its slot) and decodes on the single path (no lane
overhead); the next multi-sequence step or prefill puts it back (add). No speculative decoding here.

Shared context pool (tt/pool.py, model.pooled): each slot's latent rows / group keys sit in an extent of one pool
(_ensure: admission reserves the prefill's padded rows, decode grows it ahead in HY4_POOL_GROW steps; under pressure the
live slots' headroom is trimmed, kept sequences evicted (LRU), the pool compacted; a move relocates every row exactly).

Contract (tt/generator_vllm.py): alloc() -> a free slot; prefill(slot, ids, sampling) -> first token;
step({slot: (tok, pos, authoritative)}) -> {slot: [next]}; step_logits({slot: (tok, pos)}) -> {slot: logits};
finish(slot) frees it.
"""

from __future__ import annotations

import os
import time

import numpy as np
import torch
from loguru import logger

from models.autoports.tencent_hy4_flash.tt.common import bspec_policy
from models.autoports.tencent_hy4_flash.tt.engine import HostSampler, sample_rows

# HY4_RUNAHEAD=1 (default): after a batched step samples its tokens, the next batched pass (those tokens, the same
# slots) is launched before step() returns, so the device runs while vLLM processes the step and calls the next one;
# the next step() takes that pass's logits for every slot it continues with the same token (any other slot: undone on
# the host, recomputed). Same passes, same inputs: identical outputs.
RUNAHEAD = os.environ.get("HY4_RUNAHEAD", "1") == "1"
# HY4_TIME_BSPEC=1: batched spec step breakdown (draft inputs, draft wait, verify host / device, accept) logged per 32 steps
TIME_BSPEC = os.environ.get("HY4_TIME_BSPEC", "0") == "1"
SPEC_DBG = os.environ.get("HY4_SPEC_DBG", "0") == "1"  # (diagnostic: drafts vs target picks, _dbg_spec)


def _accept(smp, lgs, drafts):
    """(accepted drafts, emitted tokens) of a verify pass's logits lgs [k + 1, V] over `drafts` (greedy drafts): greedy:
    the matching prefix + the target's next token; sampled: exact speculative sampling (one-hot draft q)."""
    if smp.greedy:
        picks = lgs.argmax(-1).tolist()
        acc = 0
        while acc < len(drafts) and picks[acc] == drafts[acc]:
            acc += 1
        return acc, drafts[:acc] + [int(picks[acc])]
    acc = 0
    while acc < len(drafts):
        p = smp.probs(lgs[acc])
        dd = drafts[acc]
        if smp.uniform() < float(p[dd]):
            acc += 1
            continue
        p[dd] = 0.0
        return acc, drafts[:acc] + [smp.draw(p / p.sum())]
    return acc, drafts + [smp.draw(smp.probs(lgs[len(drafts)]))]


def _prog_trace(mesh):
    """HY4_PROG_TRACE=1 (diagnostic): after warm-up, log every ttnn call that adds a program-cache entry (a serve-time
    compile: the decoders then release and re-capture their traces) with its input shapes and the port's call site.
    (Wraps the ttnn / ttnn.experimental / ttnn.transformer callables: the fast runtime mode skips the op hooks.)"""
    import traceback

    import ttnn

    st = {"n": int(mesh.num_program_cache_entries())}

    def wrap(fn, name):
        def w(*args, **kwargs):
            out = fn(*args, **kwargs)
            n = int(mesh.num_program_cache_entries())
            if n != st["n"]:
                shp = [tuple(a.shape) for a in list(args) + list(kwargs.values()) if isinstance(a, ttnn.Tensor)][:4]
                site = [f"{f.name}:{f.lineno}" for f in traceback.extract_stack()[:-1]
                        if "tencent_hy4_flash" in f.filename][-4:]
                logger.warning(f"HY4 PROG +{n - st['n']}: {name} {shp} at {site}")
                st["n"] = n
            return out

        return w

    skip = {"from_torch", "to_torch", "get_device_tensors", "deallocate", "Shape", "CoreCoord", "CoreRange",
            "CoreRangeSet", "TensorAccessorArgs"}
    mods = [(ttnn, "ttnn"), (ttnn.experimental, "ttnn.experimental"), (ttnn.transformer, "ttnn.transformer")]
    dp = getattr(ttnn.experimental, "deepseek_prefill", None)
    if dp is not None:
        mods.append((dp, "ttnn.experimental.deepseek_prefill"))
    for mod, mname in mods:
        for a in dir(mod):
            if a.startswith("_") or a in skip:
                continue
            f = getattr(mod, a, None)
            if callable(f) and not isinstance(f, type) and not hasattr(f, "__self__") or (
                    callable(f) and type(f).__name__ in ("FastOperation", "Operation", "builtin_function_or_method")):
                try:
                    setattr(mod, a, wrap(f, f"{mname}.{a}"))
                except (AttributeError, TypeError):
                    pass


class Hy4BatchEngine:
    multi = True

    def __init__(self, model, trace: bool = True, k: int = 0):
        from models.autoports.tencent_hy4_flash.tt.decode_batch import Hy4BatchDecoder, Hy4BatchSpecDecoder

        self.model, self.cfg = model, model.cfg
        self.S = int(model.slots)
        lanes = next(b for b in (2, 4, 8, 16, 32) if b >= self.S)
        # k > 0: DFly2 speculative decoding while one sequence decodes alone (the solo single path)
        self.k = int(k) if os.environ.get("HY4_BATCH_SOLO", "1") == "1" else 0
        if self.k > 0:
            self.dec = Hy4BatchSpecDecoder(model, lanes=lanes, trace=trace, k=self.k)
            self.k = self.dec.k
        else:
            self.dec = Hy4BatchDecoder(model, lanes=lanes, trace=trace)
        self.trace = bool(trace)
        # HY4_BSPEC (k > 0): batched speculative decoding while 2+ sequences decode: "u:K,..." drafts per verify pass
        # for u live sequences (one batched DFly2 draft pass + one verify pass of u (K + 1) rows); u not listed or K 0:
        # plain batched steps. "0": off.
        self.bspec = {u: K for u, K in bspec_policy(self.k).items() if u <= self.S} if self.k > 0 else {}
        self._hn = {}  # slot -> (hrow, orow, opos) of the last batched pass (the batched drafter's target rows)
        self._no_bspec = False
        self.V = int(self.cfg.vocab)
        self.eos = {int(self.cfg.eos)}
        self.chunk = int(model.chunk)
        self.seqs = {}  # slot -> {"lane", "tok", "pos", "smp", "adopt", "toks"}
        self.solo = None  # the slot decoding on the single path (its lane released)
        self.solo_on = os.environ.get("HY4_BATCH_SOLO", "1") == "1"
        self.last_logits = {}
        self._ra = None  # the run-ahead pass in flight: {"hd": decoder handle, "slots": {slot: info}}
        self._ra_done = None  # its collected rows: {slot: (logits [V], info)}
        self._warming = False  # (warm(): no run-ahead; every step compiles / captures its own pass)
        self.stats = {"steps": 0, "rows": 0, "t_step": 0.0, "t_prefill": 0.0, "prefills": 0, "prefill_tokens": 0,
                      "reused_tokens": 0}
        # prefix reuse (HY4_PREFIX_CACHE, default 1): a finished sequence's slot keeps its caches and token history; a new
        # prompt sharing a prefix with it continues from there (tt/model.py rewind) instead of a full prefill
        self.pc_on = os.environ.get("HY4_PREFIX_CACHE", "1") == "1"
        self.pc_min = int(os.environ.get("HY4_PC_MIN", "512"))
        # HY4_PC_EXACT (default 1): reuse only positions the kept sequence's prefill computed in whole chunks at a fresh
        # prefill's chunk boundaries (0, 2048, 2048 + chunk, ...), resuming at such a boundary: every chunk then runs
        # exactly as in a fresh prefill of the new prompt (same rows, same programs): bit-identical output. 0: any
        # 512-aligned prefix (dense regime too, decoded positions too): device-noise-level differences
        self.pc_exact = os.environ.get("HY4_PC_EXACT", "1") == "1"
        # the single path's indexer state in fixed buffers (_solo_in): its traces are keyed (rows, regime, bucket), not
        # by slot, and capture the addresses of the active slot's group keys / staging windows (per-slot tensors): a
        # sequence decoding solo in another slot than the capture's read and wrote the capture slot's group keys
        # (the group keys: the indexers' one work buffer, which the solo slot owns: tt/indexer.py wb_take)
        import ttnn

        self._sb = []
        for layer in model.layers:
            ix = layer.attn.indexer
            if ix is not None:
                z = lambda n: ttnn.zeros((1, 1, n, ix.d), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,  # noqa: E731
                                         device=model.mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                self._sb.append((ix, (z(64), z(64))))
        # the shared context pool (tt/pool.py): extents of model.ext
        from models.autoports.tencent_hy4_flash.tt.pool import PoolOps

        self.pooled = bool(getattr(model, "pooled", False))
        self.po = PoolOps(model, self.dec)
        self.pa = self.po.pa
        self.po.warm()
        self.GROW = int(os.environ.get("HY4_POOL_GROW", "2048"))
        self.SLACK = 384  # extent rows past the position: the staging lookahead (2 x 32 rows) and group window (64 x 4)
        self._pinned = set()  # slots being prefilled together (never evicted meanwhile)
        # slots whose requests the host left out of its last decode step (vLLM finished / preempted them; the adapter
        # frees a slot only when its row is prefilled again): their extents are evictable (tt/generator_vllm.py)
        self.idle = set()
        self.on_evict = None
        self.pool_stats = self.po.stats
        self.pool_stats.update({"evicted": 0, "trims": 0})
        self.kept = {}  # slot -> {"toks": the finished sequence's fed tokens (positions 0 ..), "t": finish time, "img",
        #                          "full": its positions < full were computed by whole prefill chunks (_full)}
        # prompts prefilled in parts across decode steps (prefill_part): slot -> {"done": positions computed, "go": its
        # computation started (reset / rewind), "ids", "parts", "t"}; such a slot is never allocated / evicted meanwhile
        self.cp = {}

    # ------------------------------------------------------------------ slots
    def alloc(self, taken=(), ids=None, part=False) -> int:
        """A free slot: (ids given, prefix reuse on) the one whose kept sequence shares the longest reusable prefix with
        ids (ties: the lowest), else the least recently finished (never used first). part: ids is the first part of a
        longer prompt (prefill_part): the longest prefix the kept sequence covers (_cp_cover), at least one chunk."""
        free = [s for s in range(self.S) if s not in self.seqs and s not in taken and s not in self.cp]
        if not free:
            raise RuntimeError(f"Hy4: all {self.S} slots busy")
        if not self.pc_on:
            return free[0]
        if ids is not None:
            ids = [int(t) for t in ids]
            if part:
                r, s = max((self._grid(self._cp_cover(s, ids)), -s) for s in free)
            else:
                r, s = max((self._reuse(s, ids), -s) for s in free)
            if r > 0:
                return -s
        return min(free, key=lambda s: (self.kept.get(s, {}).get("t", 0.0), s))

    def finish(self, slot: int):
        self._settle()
        if int(slot) in self.cp:
            self._cp_drop(int(slot))
        q = self.seqs.pop(int(slot), None)
        if self.solo == int(slot):
            self.solo = None
            self._solo_out(int(slot))
        elif q is not None:
            if self.pc_on:
                self._park(q["lane"])  # (makes the finished slot the model's active one)
            else:
                self.dec.drop(q["lane"])
            self._solo_active()
        if q is not None and self.pc_on:
            self.kept[int(slot)] = {"toks": q["toks"], "t": time.time(), "img": bool(q.get("img")),
                                    "full": int(q.get("full", 0))}
        elif q is not None and self.pooled:
            self._free(int(slot))
        self.last_logits.pop(int(slot), None)

    # ------------------------------------------------------------------ the shared context pool
    def _ixs(self):
        return [l.attn.indexer for l in self.model.layers if l.attn.indexer is not None]

    def _free(self, slot):
        """Slot's extent back to the pool (its kept sequence, if any, forgotten; the work buffer dropped unwritten)."""
        self.model.ext.pop(slot, None)
        self.kept.pop(slot, None)
        self.model.swa_drop(slot)
        for ix in self._ixs():
            if ix.wb is not None and ix.wb[0] == slot:
                ix.wb, ix.wb_hi = None, ix.gcap

    def _ensure(self, slot, need, soft=True):
        """Slot's extent holds >= need tokens (rounded to 128; soft: + HY4_POOL_GROW headroom when it fits). Grows in
        place, else moves to a fitting gap; under pressure trims the live slots' headroom, evicts kept sequences (LRU),
        compacts the pool. Fixed extents (not pooled): checks only."""
        from models.autoports.tencent_hy4_flash.tt.pool import PoolFull, rup

        m = self.model
        need = min(rup(need), m.ext_rows)
        cur = m.ext.get(slot)
        if cur is not None and cur[1] >= need:
            return
        if not self.pooled:
            raise PoolFull(f"Hy4 slot {slot}: {need} tokens > its fixed extent {m.ext_rows}")
        want = min(rup(need + self.GROW), m.ext_rows) if soft else need
        self._settle()
        stage = 0
        while True:
            for size in dict.fromkeys((want, need)):
                if self._grow(slot, size):
                    m.refresh_bases()
                    return
            if stage == 0:
                stage = 1
                if self._trim(exclude=slot):
                    continue
            victims = sorted((self.kept.get(u, {}).get("t", -1.0), u) for u in m.ext
                             if u != slot and u not in self.seqs and u not in self._pinned and u not in self.cp)
            if victims:
                self._free(victims[0][1])
                self.pool_stats["evicted"] += 1
                continue
            zombies = [u for u in self.seqs if u in self.idle and u != slot and u not in self._pinned
                       and u not in self.cp]
            if zombies:  # (finished in vLLM, its pool tokens already counted free there)
                for u in zombies:
                    self.finish(u)
                    self._free(u)
                    if self.on_evict is not None:  # (the adapter forgets the row -> slot map entry)
                        self.on_evict(u)
                self.pool_stats["evicted"] += len(zombies)
                continue
            if stage == 1:
                stage = 2
                if self._compact(around=slot if slot in m.ext else None):
                    continue
            raise PoolFull(f"Hy4 pool full: slot {slot} needs {need} tokens; extents {dict(m.ext)} of {self.pa.P}")

    def _grow(self, slot, size) -> bool:
        m = self.model
        cur = m.ext.get(slot)
        if cur is None:
            b = self.pa.fit(size)
            if b is None:
                return False
            m.ext[slot] = (b, size)
            return True
        b, n = cur
        if self.pa.room_after(slot) >= size - n:
            m.ext[slot] = (b, size)
            self._zero_tail(slot, n)
            return True
        nb = self.pa.fit(size)
        if nb is None:
            return False
        self._move(slot, nb)
        m.ext[slot] = (nb, size)
        self._zero_tail(slot, n)
        return True

    def _zero_tail(self, slot, n0):
        self.po.zero_tail(slot, n0)

    def _move(self, slot, nb):
        self._settle()
        self.po.move(slot, nb)

    def _compact(self, around=None) -> bool:
        self._settle()
        return self.po.compact(around)

    def _trim(self, exclude=None) -> bool:
        """Live slots' extents cut to what they need now (position + lookahead): their growth headroom freed."""
        from models.autoports.tencent_hy4_flash.tt.pool import rup

        m, cut = self.model, False
        for u, q in self.seqs.items():
            e = m.ext.get(u)
            if u == exclude or e is None:
                continue
            keep = min(rup(q["pos"] + self.k + 3 + self.SLACK), m.ext_rows)
            if keep < e[1]:
                m.ext[u] = (e[0], keep)
                cut = True
        if cut:
            self.pool_stats["trims"] += 1
        return cut

    def _ensure_step(self, slots):
        """Before a decode step of `slots`: each extent holds its next rows (k + 1 verify rows, a run-ahead row)."""
        for s in slots:
            self._ensure(s, self.seqs[s]["pos"] + self.k + 3 + self.SLACK)

    # ------------------------------------------------------------------ prefix reuse
    def _park(self, lane):
        """A finished sequence's lane freed with its indexers' group keys written back to its slot (decode in a lane
        writes them only to the lane's region; the KV rows go to the slot's cache directly)."""
        d = self.dec
        ln = d.lanes[lane]
        m = self.model
        m.set_slot(ln["slot"])
        _, _, Ms = d._onehots(lane)
        for layer in m.layers:
            if layer.attn.indexer is not None:
                layer.attn.indexer.lane_unload(d.L, lane, Ms)
        d.lanes[lane] = None

    def _big(self) -> int:
        """The prefill chunk past the dense regime (tt/model.py prefill)."""
        return min(int(os.environ.get("HY4_PREFILL_CHUNK", self.model.chunk)), self.model.chunk)

    def _full(self, n) -> int:
        """Positions < this of an n-token prompt are computed by whole chunks of a fresh prefill (boundaries 2048,
        2048 + chunk, ...; the last, partial chunk is padded to a bucket and differs with the prompt length)."""
        dense = int(self.cfg.idx_candidate_topk)
        if n < dense or (n == dense and os.environ.get("HY4_PF_TRACE", "0") == "1"):  # (a traced first chunk)
            return 0
        return dense + ((n - dense) // self._big()) * self._big()

    def _reuse(self, slot, ids) -> int:
        """Positions of `ids` (list) the free slot's kept sequence can serve: r <= the common prefix d, r < len(ids) (the
        last row runs for its logits). HY4_PC_EXACT=1: r a fresh prefill's chunk boundary <= the kept prefill's whole
        chunks (bit-identical). Else aligned for the continued prefill (r < 2048, the dense regime: a multiple of
        HY4_PC_ALIGN_DENSE, default 512 >= the SWA window, so the chunk takes the flash SWA path; else of
        HY4_PC_ALIGN, default 512: the group-key slab writes of a chunk at group offset r / 4 then take the slab sizes
        warm() compiles), the SWA ring
        still holding positions [r - window, r), r >= HY4_PC_MIN. 0: no reuse."""
        k = self.kept.get(int(slot))
        m = self.model
        if (not self.pc_on or k is None or k["img"] or int(slot) in self.seqs or int(slot) not in m.ext
                or (m.engram is not None and not hasattr(m.engram, "hist"))):
            return 0
        toks = k["toks"]
        lim = min(len(toks), len(ids) - 1)
        if lim < self.pc_min:
            return 0
        a = np.asarray(toks[:lim], dtype=np.int64)
        b = np.asarray(ids[:lim], dtype=np.int64)
        neq = np.flatnonzero(a != b)
        dpre = int(neq[0]) if neq.size else lim
        dense = int(self.cfg.idx_candidate_topk)
        if self.pc_exact:  # a chunk boundary of a fresh prefill, inside the kept prefill's whole chunks
            c = min(dpre, int(k.get("full", 0)))
            r = dense + ((c - dense) // self._big()) * self._big() if c >= dense else 0
        elif dpre >= dense:
            A = int(os.environ.get("HY4_PC_ALIGN", "512"))
            r = (dpre // A) * A
            if r < dense:
                r = dense
        else:
            A = int(os.environ.get("HY4_PC_ALIGN_DENSE", "512"))
            r = (dpre // A) * A
        if r <= 0 or r % 128 or r < max(self.pc_min, int(self.cfg.window)):
            return 0
        swa = next((l.attn for l in m.layers if l.attn.kind == "swa"), None)
        if getattr(m, "_swa", None):  # split SWA rings: the window before r in the prefill ring or the slot's snapshot
            return r if m.swa_window_ok(int(slot), r) else 0
        if swa is not None:  # ring rows hold the last swa.rows positions written (prefill padding, decode lookahead)
            hiw = max(getattr(m, "_hiw", {}).get(int(slot), 0), len(toks) + 96 + self.k)
            if r - int(self.cfg.window) < hiw - swa.rows:
                return 0
        return r

    @property
    def live(self):
        return sorted(self.seqs)

    # ------------------------------------------------------------------ prefill / decode
    def prefill(self, slot: int, prompt_ids, sampling=None, mm=None) -> int:
        """mm: image inputs of the prompt (Hy4Model.set_images kwargs, tt/vision.py Hy4Vision.prepare)."""
        t0 = time.time()
        slot = int(slot)
        self._settle()
        self._hn = {}
        self.idle.discard(slot)
        self.finish(slot)
        if self.solo is not None:  # the attention staging windows are shared: the solo sequence back to a lane first
            self._exit_solo()
        ids = torch.as_tensor(prompt_ids, dtype=torch.long).reshape(-1)
        n = int(ids.numel())
        if n + 64 > self.model.max_ctx:
            raise ValueError(f"Hy4: prompt of {n} tokens exceeds the slot context {self.model.max_ctx - 64}")
        m = self.model
        idl = ids.tolist()
        r = self._reuse(slot, idl) if mm is None else 0
        if self.pooled:  # its extent: the kept one (reuse), else a fresh one; the prefill's padded rows reserved
            if not r:
                self._free(slot)
            self._ensure(slot, max(m.pf_end(n, r), n + self.SLACK), soft=False)
        m.set_slot(slot)
        m.refresh_bases()
        if r:
            m.rewind(r, idl)  # the slot's kept sequence serves positions < r
        else:
            m.reset()
        self.kept.pop(slot, None)
        if mm is not None:
            m.set_images(**mm)
        lg = m.prefill(ids[r:])
        if self.pc_on and mm is None:  # (split SWA rings: the window before the last whole chunk -> host, for reuse)
            m.swa_snapshot(self._full(n))
        m.prefill_caches_release()  # (before any decode replay: tt/model.py)
        self._fit(slot, n)
        lane = self.dec.add(slot, n, idl)
        smp = HostSampler(sampling)
        tok = smp(lg)
        self.seqs[slot] = {"lane": lane, "tok": tok, "pos": n, "smp": smp, "adopt": False, "toks": idl,
                           "img": mm is not None, "full": self._full(n)}
        self.last_logits[slot] = lg
        dt = time.time() - t0
        self.stats["t_prefill"] += dt
        self.stats["prefills"] += 1
        self.stats["prefill_tokens"] += n - r
        self.stats["reused_tokens"] += r
        logger.info(f"Hy4 prefill {n} tokens ({r} reused) into slot {slot} (lane {lane}) in {dt:.2f} s; live {self.live}")
        return tok

    # ------------------------------------------------------------------ a prompt in parts (vLLM chunked prefill)
    def _grid(self, n) -> int:
        """The last chunk boundary of a fresh prefill <= n (0, 2048, 2048 + chunk, ...)."""
        dense = int(self.cfg.idx_candidate_topk)
        return 0 if n < dense else dense + ((n - dense) // self._big()) * self._big()

    def _cp_cover(self, slot, ids) -> int:
        """Positions < this of the prompt known so far (ids) the slot's kept sequence holds as whole prefill chunks (its
        prefix reuse can reach at least there once the rest of the prompt is known); 0: none."""
        k = self.kept.get(slot)
        m = self.model
        if (not self.pc_on or k is None or k["img"] or slot in self.seqs or slot not in m.ext
                or (m.engram is not None and not hasattr(m.engram, "hist"))):
            return 0
        lim = min(len(k["toks"]), len(ids))
        neq = np.flatnonzero(np.asarray(k["toks"][:lim], dtype=np.int64) != np.asarray(ids[:lim], dtype=np.int64))
        return min(int(neq[0]) if neq.size else lim, int(k.get("full", 0)))

    def prefill_part(self, slot: int, ids, final: bool, sampling=None):
        """A text prompt prefilled in parts with other sequences' decode steps in between (vLLM chunked prefill):
        ids = its tokens known so far (positions 0 .. e), final: e is its end. Computes up to the last fresh-prefill
        chunk boundary <= e (final: to the end): every chunk runs exactly as in prefill() of the whole prompt (same
        positions, rows, programs: bit-identical caches and logits). Between parts the slot is pinned (never allocated,
        trimmed or evicted); other slots' decode passes never address its caches; its model / indexer state is kept by
        set_slot; the prefill ring's window before its next part by _swa_yield / swa_resume; the prefill constants are
        released after every part (before any decode replay). Prefix reuse: a first part the slot's kept sequence
        already holds is not started until the prompt leaves it (the reuse then reaches as far as prefill()'s).
        -> the first token (final), else None."""
        slot = int(slot)
        idl = [int(t) for t in ids]
        n = len(idl)
        m = self.model
        st = self.cp.get(slot)
        if st is None:
            if final:  # (the whole prompt in one part)
                return self.prefill(slot, idl, sampling)
            self._settle()
            self.idle.discard(slot)
            self.finish(slot)
            st = self.cp[slot] = {"done": 0, "go": False, "parts": 0, "t": 0.0, "r": 0, "ids": idl}
        if n + 64 > m.max_ctx:
            raise ValueError(f"Hy4: prompt of {n} tokens exceeds the slot context {m.max_ctx - 64}")
        if idl[: st["done"]] != st["ids"][: st["done"]]:
            raise RuntimeError(f"Hy4 slot {slot}: a prompt part does not continue the computed {st['done']} tokens")
        st["ids"] = idl
        t0 = time.time()
        tgt = n if final else self._grid(n)
        lg = None
        if not st["go"] and tgt > 0 and (final or self._cp_cover(slot, idl) < tgt):
            self._cp_start(slot, idl, st, tgt, final)
        if st["go"] and tgt > st["done"]:
            done = st["done"]
            self._settle()
            if self.solo is not None:  # (the attention staging windows are shared)
                self._exit_solo()
            if self.pooled:
                self._ensure(slot, max(m.pf_end(tgt, done), tgt + self.SLACK), soft=not final)
            m.set_slot(slot)
            m.refresh_bases()
            if m.pos != done:
                raise RuntimeError(f"Hy4 slot {slot}: model at {m.pos}, its prompt computed to {done}")
            if st["parts"]:
                m.swa_resume(done)
            m._swa_hold.add(slot)
            lg = m.prefill(torch.as_tensor(idl[done:tgt], dtype=torch.long), head=final)
            st["done"] = tgt
            st["parts"] += 1
        if not final:
            t1 = time.time()
            m.prefill_caches_release()  # (before any decode replay; its readbacks wait for the part's device work)
            st["t"] += time.time() - t0
            if os.environ.get("HY4_CP_LOG", "0") == "1":
                logger.info(f"Hy4 part slot {slot}: {st['done']} of {n} known, {(t1 - t0) * 1e3:.1f} ms enqueue, "
                            f"{(time.time() - t1) * 1e3:.1f} ms release")
            return None
        st["t"] += time.time() - t0
        self.cp.pop(slot)
        m._swa_hold.discard(slot)
        r = st["r"]
        if self.pc_on:
            m.swa_snapshot(self._full(n))
        m.prefill_caches_release()
        self._hn = {}
        self._fit(slot, n)
        lane = self.dec.add(slot, n, idl)
        smp = HostSampler(sampling)
        tok = smp(lg)
        self.seqs[slot] = {"lane": lane, "tok": tok, "pos": n, "smp": smp, "adopt": False, "toks": idl, "img": False,
                           "full": self._full(n)}
        self.last_logits[slot] = lg
        self.stats["t_prefill"] += st["t"]
        self.stats["prefills"] += 1
        self.stats["prefill_tokens"] += n - r
        self.stats["reused_tokens"] += r
        self.stats["prefill_parts"] = self.stats.get("prefill_parts", 0) + st["parts"]
        logger.info(f"Hy4 prefill {n} tokens ({r} reused) into slot {slot} (lane {lane}) in {st['parts']} parts, "
                    f"{st['t']:.2f} s; live {self.live}")
        return tok

    def _cp_start(self, slot, idl, st, tgt, final):
        """A prompt's first computed part: prefill()'s set-up (the reuse decision on the prompt known so far). (The
        decoding sequences' batched-spec state stays: no lane changes until the last part.)"""
        self._settle()
        if self.solo is not None:
            self._exit_solo()
        m = self.model
        r = self._reuse(slot, idl)
        if self.pooled:
            if not r:
                self._free(slot)
            self._ensure(slot, max(m.pf_end(tgt, r), tgt + self.SLACK), soft=not final)
        m.set_slot(slot)
        m.refresh_bases()
        if r:
            m.rewind(r, idl)
        else:
            m.reset()
        self.kept.pop(slot, None)
        st.update(go=True, done=r, r=r)

    def _cp_drop(self, slot):
        """A prompt in parts abandoned (vLLM preempted / aborted it): its computed positions kept for prefix reuse (the
        resumed prompt continues from them), else its extent freed."""
        st = self.cp.pop(slot)
        m = self.model
        m._swa_hold.discard(slot)
        if not st["go"]:  # (its kept sequence untouched)
            return
        done = st["done"]
        if self.pc_on and slot in m.ext and done >= max(self.pc_min, int(self.cfg.window)):
            m._swa_save(slot, done)  # (the ring's window, unless _swa_yield saved it)
            self.kept[slot] = {"toks": st["ids"][:done], "t": time.time(), "img": False, "full": done}
        elif self.pooled:
            self._free(slot)
        logger.info(f"Hy4 prompt in parts dropped from slot {slot} at {done} of {len(st['ids'])} known tokens")

    def _fit(self, slot, n):
        """After a prefill of n tokens: the extent cut to n + lookahead + growth headroom (the padded rows past n are
        never read)."""
        from models.autoports.tencent_hy4_flash.tt.pool import rup

        m = self.model
        e = m.ext.get(slot)
        if self.pooled and e is not None:
            keep = min(rup(n + self.k + 3 + self.SLACK + self.GROW), m.ext_rows)
            if keep < e[1]:
                m.ext[slot] = (e[0], keep)

    def prefill_many(self, reqs) -> list:
        """reqs [(slot, ids, sampling)] -> first tokens. Simultaneous short prompts (dense regime, ordinary text) run as
        one packed prefill pass (Hy4Model.forward_packed: B prompts on the attention's batch dimension); otherwise one
        prefill each."""
        self._settle()
        self._hn = {}
        m = self.model
        idl = [torch.as_tensor(ids, dtype=torch.long).reshape(-1) for _, ids, _ in reqs]
        if self.pc_on and len(reqs) > 1:  # prompts continuing a kept sequence: their own (continued) prefill
            for slot, _, _ in reqs:
                self.finish(slot)
            hit = [self._reuse(slot, t.tolist()) > 0 for (slot, _, _), t in zip(reqs, idl)]
            if any(hit):
                out = [None] * len(reqs)
                rest = [i for i, h in enumerate(hit) if not h]
                for i, tok in zip(rest, self.prefill_many([reqs[i] for i in rest]) if rest else []):
                    out[i] = tok
                for i in [i for i, h in enumerate(hit) if h]:
                    out[i] = self.prefill(reqs[i][0], idl[i], reqs[i][2])
                return out
        plan = m.pack_plan([int(t.numel()) for t in idl])
        if plan is None or not all(m.engram_keep_all(t) for t in idl):
            return [self.prefill(slot, ids, sp) for (slot, _, sp), ids in zip(reqs, idl)]
        t0 = time.time()
        for slot, _, _ in reqs:
            self.finish(slot)
        if self.solo is not None:
            self._exit_solo()
        for slot, _, _ in reqs:
            self.kept.pop(slot, None)
            if self.pooled:
                self._free(slot)
        self.idle -= {slot for slot, _, _ in reqs}
        if self.pooled:
            self._pinned = {slot for slot, _, _ in reqs}
            try:
                for (slot, _, _), ids in zip(reqs, idl):
                    self._ensure(slot, max(plan[1], int(ids.numel()) + self.SLACK), soft=False)
            finally:
                self._pinned = set()
        lgs = m.forward_packed([(slot, ids) for (slot, _, _), ids in zip(reqs, idl)], plan)
        m.prefill_caches_release()
        out = []
        for (slot, _, sp), ids, lg in zip(reqs, idl, lgs):
            self._fit(slot, int(ids.numel()))
            lane = self.dec.add(slot, int(ids.numel()), ids.tolist())
            smp = HostSampler(sp)
            tok = smp(lg)
            self.seqs[slot] = {"lane": lane, "tok": tok, "pos": int(ids.numel()), "smp": smp, "adopt": False,
                               "toks": ids.tolist()}
            self.last_logits[slot] = lg
            out.append(tok)
        dt = time.time() - t0
        self.stats["t_prefill"] += dt
        self.stats["prefills"] += len(reqs)
        self.stats["prefill_tokens"] += sum(int(t.numel()) for t in idl)
        logger.info(f"Hy4 packed prefill of {len(reqs)} prompts {[int(t.numel()) for t in idl]} (plan {plan}) into slots "
                    f"{[r[0] for r in reqs]} in {dt:.2f} s; live {self.live}")
        return out

    def _enter_solo(self, slot):
        q = self.seqs[slot]
        ln = self.dec.lanes[q["lane"]]
        self.dec.remove(q["lane"])  # model slot = slot, its pos / Engram state; staging windows = the lane's
        d = self.dec
        d.ob, d.ogb, d.pos = ln["ob"], ln["ogb"], ln["pos"]
        d.set_history(q["toks"])
        q["lane"] = None
        self.solo = slot
        self._hn = {}
        self._solo_in(slot)

    _SOLO_ATTRS = ("tk_stage", "g_stage")

    def _solo_in(self, slot):
        """The slot's indexer state (token / group windows) -> the fixed solo buffers, which the indexers then use as the
        slot's (the single-path traces read and write them); its group keys -> the work buffer (wb_take)."""
        import ttnn

        self.model.set_slot(slot)
        for ix, bufs in self._sb:
            ix.wb_take(slot, load=True)
            orig = tuple(getattr(ix, a) for a in self._SOLO_ATTRS)
            for src, dst in zip(orig, bufs):
                ttnn.copy(src, dst)
            ix._solo_orig = orig
            for a, b in zip(self._SOLO_ATTRS, bufs):
                setattr(ix, a, b)

    def _solo_out(self, slot):
        """The solo buffers -> the slot's own tensors (restored)."""
        import ttnn

        self.model.set_slot(slot)
        for ix, bufs in self._sb:
            if ix.wb is not None and ix.wb[0] == slot:
                ix.wb_release()
            orig = ix.__dict__.pop("_solo_orig")
            for src, dst in zip(bufs, orig):
                ttnn.copy(src, dst)
            for a, t in zip(self._SOLO_ATTRS, orig):
                setattr(ix, a, t)

    def _solo_active(self):
        """The solo sequence's slot back as the model's active one. The single path addresses the active slot's cache
        regions (layer bases, staging, Engram / position state); finishing another slot meanwhile (_park: the adapter's
        idle zombies evicted under pool pressure, a vLLM batch remap) made that slot active, and once its extent was
        freed (_free) every base fell back to the trash rows: the solo passes then wrote their KV / group rows at
        trash_row + position, past the end of the caches (other tensors' device memory: spec acceptance 1.00 for good)."""
        if self.solo is not None and self.model.slot != self.solo:
            self.model.set_slot(self.solo)

    def _solo_check(self):
        """Before a single-path pass: the solo slot is the active one and holds its pool extent (a pass based on a
        missing extent would write past the caches' end: an error instead of silent device memory corruption)."""
        m = self.model
        if m.slot != self.solo:
            logger.warning(f"Hy4 solo slot {self.solo} was not the active slot ({m.slot}): re-activated")
            self._solo_active()
        if self.pooled and m.slot not in m.ext:
            raise RuntimeError(f"Hy4 solo slot {m.slot} has no pool extent (extents {dict(m.ext)})")

    def _exit_solo(self):
        slot, self.solo = self.solo, None
        self._solo_out(slot)
        q = self.seqs[slot]
        d = self.dec
        q["lane"] = d.add(slot, q["pos"], q["toks"], stage=(d.ob, d.ogb))  # its current single-path windows
        q["n"] = None

    # ------------------------------------------------------------------ run-ahead
    def _settle(self):
        """Wait for the run-ahead pass in flight (if any) and keep its rows for the next step()."""
        ra, self._ra = self._ra, None
        if ra is None:
            return
        lo = self.dec.collect(ra["hd"])
        done = self._ra_done or {}
        for slot, info in ra["slots"].items():
            done[slot] = (lo[info["lane"]][0], info)
        self._ra_done = done

    def _ra_undo(self, slot, info):
        """A run-ahead row not taken: the slot's host state back to before that pass (its lane's staging windows stay
        based where the pass left them; the recomputing pass at the same position overwrites what it wrote). A finished
        slot needs nothing (its fed token is in its token history, its KV written)."""
        q = self.seqs.get(slot)
        if q is None or q.get("lane") != info["lane"] or self.dec.lanes[info["lane"]] is not info["ln"]:
            return
        ln = info["ln"]
        ln["pos"], ln["eng"] = info["pos"], info["eng"]
        del ln["tokens"][info["ntok_lane"]:]
        del q["toks"][info["ntok"]:]
        self.dec._pre = None
        self.stats["ra_undone"] = self.stats.get("ra_undone", 0) + 1

    def _take_ra(self, rows: dict) -> dict:
        """{slot: logits} of the settled run-ahead rows when `rows` continues exactly the pass's sequences (same slots,
        lanes and tokens); otherwise every run-ahead row is undone (a request finished or joined, a row skipped, a token
        overridden: about once per request) and the caller runs the step itself: the pass a step runs is always the one
        it would run without run-ahead (same rows, so the same programs and numerics: a 2-row pass's row differs from
        the solo path's, and a pass of other rows could need a program warm-up never compiled)."""
        self._settle()
        done, self._ra_done = self._ra_done or {}, None
        # (the same sequence: its slot still holds the lane object the pass advanced; a finished slot needs nothing)
        alive = {s for s, (_, info) in done.items() if self.seqs.get(s) is not None
                 and self.seqs[s].get("lane") == info["lane"] and self.dec.lanes[info["lane"]] is info["ln"]}
        acc = {s: lg for s, (lg, info) in done.items() if s in alive and s in rows and list(rows[s]) == [info["tok"]]}
        if acc and set(acc) == set(rows) == alive == set(done):
            return acc
        for slot, (lg, info) in done.items():
            self._ra_undo(slot, info)
        return {}

    def _launch_ra(self, slots):
        """The next batched pass of `slots` (each feeds its sampled token), launched now; collected by _settle."""
        d = self.dec
        if not (RUNAHEAD and self.trace and d.trace and not self._warming and not d.min_width and self.solo is None
                and len(slots) >= 2) or self.bspec.get(len(slots), 0) > 0:  # (batched spec: its draft reads this pass)
            return
        rows, info = {}, {}
        for slot in slots:
            q = self.seqs[slot]
            lane = q.get("lane")
            if lane is None or q["tok"] is None or self.model.max_ctx - 64 <= q["pos"] + 1:
                return
            ln = d.lanes[lane]
            rows[lane] = [q["tok"]]
            info[slot] = {"lane": lane, "tok": q["tok"], "ln": ln, "pos": ln["pos"], "eng": ln["eng"],
                          "ntok_lane": len(ln["tokens"]), "ntok": len(q["toks"])}
        key = ("b", len(rows))
        if not any(k[:2] == key for k in d.btraces):  # (a pass without a trace yet: compile / capture synchronously)
            return
        t0 = time.time()
        hd = d.launch(rows)
        for slot in slots:
            q = self.seqs[slot]
            q["toks"] += [q["tok"]]
            q["n"] = None
        self._ra = {"hd": hd, "slots": info}
        self.stats["t_step"] += time.time() - t0
        self.stats["ra_launched"] = self.stats.get("ra_launched", 0) + 1

    def _pass(self, rows: dict) -> dict:
        """rows {slot: [ids]} -> {slot: logits [n, V]} (solo: the single path)."""
        t0 = time.time()
        if self.solo_on and len(rows) == 1 and self.S > 1:
            (slot, ids), = rows.items()
            if self.solo not in (None, slot):
                self._exit_solo()
            if self.solo is None:
                self._enter_solo(slot)
            self._solo_check()
            out = {slot: self.dec.run(list(ids), logits_rows="all", p0=self.seqs[slot]["pos"]).reshape(len(ids), -1)}
        else:
            if self.solo is not None:
                self._exit_solo()
            lo = self.dec.run_batch({self.seqs[s]["lane"]: ids for s, ids in rows.items()})
            out = {s: lo[self.seqs[s]["lane"]] for s in rows}
            self._set_hn(rows)
        solo = self.solo is not None
        for s, ids in rows.items():
            self.seqs[s]["toks"] += [int(t) for t in ids]
            self.seqs[s]["n"] = 0 if solo else None  # spec: the drafter continues from a solo 1-row pass's row 0
        self.stats["t_step"] += time.time() - t0
        self.stats["steps"] += 1
        self.stats["rows"] += len(rows)
        return out

    def _check_pos(self, slot, q, pos):
        if self.model.max_ctx - 64 <= q["pos"]:
            raise RuntimeError(f"Hy4 slot {slot}: context full ({q['pos']} tokens)")
        if pos is not None and int(pos) != q["pos"]:
            raise RuntimeError(f"Hy4 slot {slot}: step at pos {pos}, engine at {q['pos']}")

    def step(self, reqs: dict) -> dict:
        """reqs {slot: (tok, pos, host_authoritative)} -> {slot: [next token(s)]} (one batched pass; a sequence alone
        on the solo path with k > 0: one verify pass, 1 .. k + 1 tokens)."""
        if self.pooled:
            self._ensure_step(reqs)
        rows = {}
        for slot, (tok, pos, auth) in reqs.items():
            q = self.seqs[slot]
            adopt, q["adopt"] = q["adopt"], False
            if tok is not None and (q["tok"] is None or ((auth or adopt) and int(tok) != q["tok"])):
                self._check_pos(slot, q, pos)
                q["tok"] = int(tok)
            self._check_pos(slot, q, None)
            rows[slot] = [q["tok"]]
        acc = self._take_ra(rows)
        if not acc and self._bspec_ok(rows):
            return self._bspec_step(rows)
        if not acc and self.k > 0 and len(rows) == 1 and self.solo is not None and self.solo in rows:
            q = self.seqs[self.solo]
            if q.get("n") is not None and q["pos"] + self.k + 1 + 64 <= self.model.max_ctx:
                return {self.solo: self._spec_step(self.solo)}
        if acc:
            out = {s: lg[None] for s, lg in acc.items()}
            self.stats["steps"] += 1
            self.stats["rows"] += len(acc)
            self.stats["ra_taken"] = self.stats.get("ra_taken", 0) + len(acc)
            self._set_hn(acc)
        else:
            out = self._pass(rows)
        picks = self._sample_batch(out)
        res = {}
        for slot, lg in out.items():
            q = self.seqs[slot]
            nxt = picks[slot] if picks is not None else q["smp"](lg[0])
            q["tok"], q["pos"] = nxt, q["pos"] + 1
            res[slot] = [nxt]
        self._launch_ra(sorted(res))
        return res

    def _sample_batch(self, out):
        """HY4_BATCH_SAMPLE=1 (default): a batched pass's next tokens by tt/engine.sample_rows over the pass's logits rows
        (identical tokens to row-by-row sampling), or None (solo pass, rows not one per slot of the last collected pass).
        out {slot: logits [1, V]}: each row must be the collected pass's own row (same storage), checked."""
        lo = getattr(self.dec, "last_out", None)
        if lo is None or self.solo is not None or os.environ.get("HY4_BATCH_SAMPLE", "1") != "1":
            return None
        full, lanes = lo
        if len(lanes) != len(out) or int(full.shape[0]) != len(lanes):
            return None
        at = {l: i for i, l in enumerate(lanes)}
        slots = sorted(out, key=lambda s: at.get(self.seqs[s]["lane"], -1))
        idx = [at.get(self.seqs[s]["lane"]) for s in slots]
        if idx != list(range(len(lanes))):
            return None
        if any(out[s].reshape(-1)[0].data_ptr() != full[i].reshape(-1)[0].data_ptr() for s, i in zip(slots, idx)):
            return None
        return dict(zip(slots, sample_rows(full, [self.seqs[s]["smp"] for s in slots])))

    def _spec_step(self, slot) -> list:
        """One verify pass of the solo sequence: rows [tok, d_1 .. d_k] (DFly2 drafts on its last solo pass); greedy:
        the matching prefix + the target's next token; sampled: exact speculative sampling (tt/engine.py)."""
        t0 = time.time()
        q, d, smp = self.seqs[slot], self.dec, self.seqs[slot]["smp"]
        self._solo_check()
        drafts = d.draft(q["tok"], q["pos"], q["n"])[: self.k]
        rows = [q["tok"]] + drafts
        lgs = d.run(rows, logits_rows="all", p0=q["pos"])  # [k + 1, V]
        acc, emit = _accept(smp, lgs, drafts)
        if SPEC_DBG:
            self._dbg_spec("solo", slot, q, drafts, lgs, acc)
        q["toks"] += rows[: len(emit)]  # the accepted rows' tokens (fed at pos .. pos + acc)
        q["n"] = acc
        q["pos"] += len(emit)
        q["tok"] = emit[-1]
        self.last_logits[slot] = lgs[acc]
        self.stats["t_step"] += time.time() - t0
        self.stats["steps"] += 1
        self.stats["rows"] += 1
        self.stats["spec_emitted"] = self.stats.get("spec_emitted", 0) + len(emit)
        self.stats["spec_passes"] = self.stats.get("spec_passes", 0) + 1
        return emit

    def _dbg_spec(self, kind, slot, q, drafts, lgs, acc):
        """HY4_SPEC_DBG=1: drafts vs the target's greedy picks, the first 4 passes then every 128th per kind."""
        c = self.__dict__.setdefault("_dbgc", {})
        c[kind] = c.get(kind, 0) + 1
        if c[kind] <= 4 or c[kind] % 128 == 0:
            picks = lgs.argmax(-1).tolist()[: len(drafts) + 1]
            logger.info(f"HY4DBG {kind} #{c[kind]} slot {slot} pos {q['pos']} n {q.get('n')} bonus {q['tok']} drafts "
                        f"{drafts} picks {picks} acc {acc} greedy {q['smp'].greedy} solo {self.solo} "
                        f"model.slot {self.model.slot} seqs {sorted(self.seqs)} idle {sorted(self.idle)}")

    # ------------------------------------------------------------------ batched speculative decoding
    def _set_hn(self, slots):
        """After a plain batched pass of `slots` (one row each, rows in lane order): the drafter's target rows."""
        order = sorted(slots, key=lambda s: self.seqs[s]["lane"])
        self._hn = {s: (i, i, self.seqs[s]["pos"]) for i, s in enumerate(order)}

    def _bspec_ok(self, rows) -> bool:
        """This step can be one batched draft + verify pass: K drafts for len(rows) sequences (HY4_BSPEC), every
        sequence in the last batched pass (its target rows in last_hn / the lane index lists) and no other."""
        K = self.bspec.get(len(rows), 0)
        if K <= 0 or self._no_bspec or self.solo is not None or set(self._hn) != set(rows):
            return False
        return all(self.seqs[s].get("lane") is not None and self.seqs[s]["pos"] + K + 1 + 64 <= self.model.max_ctx
                   for s in rows)

    def _bspec_step(self, rows) -> dict:
        """One batched draft pass (tt/decode_batch.py draft_lanes) and one verify pass of every live sequence (rows
        [tok, d_1 .. d_K] each, lane order); per sequence the accepted prefix + the target's next token (greedy) or
        exact speculative sampling; each lane rolled back to its accepted rows (position, token history, Engram
        n-grams; KV / group keys of the rejected rows are rewritten before any row sees them)."""
        t0 = time.time()
        d = self.dec
        u = len(rows)
        K = self.bspec[u]
        order = sorted(rows, key=lambda s: self.seqs[s]["lane"])
        users = []
        for s in order:
            q = self.seqs[s]
            hrow, orow, opos = self._hn[s]
            users.append({"bonus": q["tok"], "anchor": q["pos"], "hrow": hrow, "orow": orow, "opos": opos, "slot": s,
                          "base": d.lanes[q["lane"]]["base"]["sparse"]})
        tb = [time.perf_counter()] if TIME_BSPEC else None
        drafts = d.draft_lanes(users, K, then=lambda: d.prebuild({self.seqs[s]["lane"]: K + 1 for s in order}), tb=tb)
        feed = {s: [self.seqs[s]["tok"]] + dr for s, dr in zip(order, drafts)}
        hd = d.launch({self.seqs[s]["lane"]: feed[s] for s in order})
        if tb is not None:
            tb.append(time.perf_counter())
        lo = d.collect(hd)
        if tb is not None:
            tb.append(time.perf_counter())
        eng = self.model.engram
        keep = int(getattr(eng, "keep", 2))
        res, hn, emitted = {}, {}, 0
        job = lambda s, dr: _accept(self.seqs[s]["smp"], lo[self.seqs[s]["lane"]], dr)  # noqa: E731
        sam = [i for i, s in enumerate(order) if not self.seqs[s]["smp"].greedy]
        accs = {}
        if len(sam) > 1:  # sampled rows' probabilities in threads (each sequence its own generator; 8 users 9.5 -> 4.5 ms)
            if getattr(self, "_pool", None) is None:
                from concurrent.futures import ThreadPoolExecutor

                self._pool = ThreadPoolExecutor(self.S)
            accs = dict(zip(sam, self._pool.map(lambda i: job(order[i], drafts[i]), sam)))
        for i, (s, dr) in enumerate(zip(order, drafts)):
            q = self.seqs[s]
            lgs = lo[q["lane"]]  # [K + 1, V]
            acc, emit = accs[i] if i in accs else job(s, dr)
            if SPEC_DBG:
                self._dbg_spec("bspec", s, q, dr, lgs, acc)
            p0, ln = q["pos"], d.lanes[q["lane"]]
            p1 = p0 + acc + 1  # rows p0 .. p0 + acc accepted
            ln["pos"] = p1
            del ln["tokens"][p1:]
            if ln["eng"] is not None:
                tl = ln["tokens"]
                ln["eng"] = ({li: np.asarray(tl[max(0, p1 - keep) : p1], dtype=np.int64) for li in ln["eng"][0]},
                             {li: p1 for li in ln["eng"][1]})
            q["toks"] += feed[s][: acc + 1]
            q["pos"], q["tok"], q["n"] = p1, emit[-1], None
            self.last_logits[s] = lgs[acc]
            hn[s] = (i * (K + 1) + acc, i * (K + 1), p0)
            res[s] = emit
            emitted += len(emit)
        d._pre = None  # (prebuilt for one row per lane at the unaccepted positions)
        self._hn = hn
        if tb is not None:
            tb.append(time.perf_counter())
            self._tb = getattr(self, "_tb", [])
            self._tb.append(np.diff(tb) * 1e3)
            if len(self._tb) % 32 == 0:
                m = np.median(np.array(self._tb[-32:]), axis=0)
                logger.info(f"bspec u={u} K={K} step {m.sum():.2f} ms: draft inputs+copy {m[0]:.2f}, launch {m[1]:.2f}, "
                            f"verify prebuild {m[2]:.2f}, draft wait+read {m[3]:.2f}, verify host+copy+launch {m[4]:.2f}, "
                            f"verify device+read {m[5]:.2f}, accept {m[6]:.2f}")
        st = self.stats
        st["t_step"] += time.time() - t0
        st["steps"] += 1
        st["rows"] += u
        st["bspec_passes"] = st.get("bspec_passes", 0) + 1
        st["bspec_emitted"] = st.get("bspec_emitted", 0) + emitted
        st["bspec_rows"] = st.get("bspec_rows", 0) + u
        return res

    def step_logits(self, reqs: dict) -> dict:
        """Host-sampled step: reqs {slot: (tok, pos)} (each continues its sequence) -> {slot: next-token logits}."""
        self._take_ra({})  # (no run-ahead here: its rows undone)
        if self.pooled:
            self._ensure_step(reqs)
        rows = {}
        for slot, (tok, pos) in reqs.items():
            q = self.seqs[slot]
            self._check_pos(slot, q, pos)
            rows[slot] = [int(tok)]
        out = self._pass(rows)
        res = {}
        for slot, lg in out.items():
            q = self.seqs[slot]
            q["pos"] += 1
            q["tok"], q["adopt"] = None, True
            res[slot] = lg[0]
        return res

    # ------------------------------------------------------------------ warm-up
    def warm(self, prompt_len=None):
        """Compile every program serving runs before any trace is captured: the prefill buckets (slot 0), then each
        batch size 1..S at every score width bucket (eager), then capture the common traces."""
        self._take_ra({})
        self._warming = True
        self._no_bspec = True  # (plain batched passes; the batched spec ones by _warm_bspec)
        try:
            self._warm(prompt_len)
        finally:
            self._warming = False
            self._no_bspec = False
            self._hn = {}

    def _warm(self, prompt_len=None):
        t0 = time.time()
        g = torch.Generator().manual_seed(0)
        rnd = lambda n: torch.randint(1000, 100000, (n,), generator=g).tolist()  # noqa: E731
        m = self.model
        m._pf_hold = True  # (tt/model.py prefill_caches_release: kept until prefill_caches_mark below)
        if os.environ.get("HY4_PF_BUCKETS", "1") == "1":
            for b in m.PF_BUCKETS:
                if b <= self.chunk and b < m.max_ctx:
                    self.prefill(0, rnd(b - 1))
                    m.warm_head_blocks(b)
            # later chunks at every start the slot context allows (their programs change with the indexer's score width
            # bucket; a 12K prompt after a 6K warm compiled 24 programs at serve time and released every trace)
            # HY4_WARM_LONG=auto (default): every (score bucket, row bucket) pair once (tt/model.py warm_long: one long
            # prefill + rewinds; before: every chunk start x row bucket from position 0, O(n^2) at long slots)
            m_fit = max(0, (m.max_ctx - 64 - self.chunk) // self.chunk)
            wl = os.environ.get("HY4_WARM_LONG", "auto")
            if wl == "auto":
                self.finish(0)
                if self.pooled:  # (the whole slot context: every other extent freed)
                    for u in list(m.ext):
                        self._free(u)
                    self._ensure(0, m.ext_rows, soft=False)
                m.warm_long(rnd, slot=0)
            for k in range(1, (0 if wl == "auto" else int(wl)) + 1):
                for b in m.PF_BUCKETS:
                    L = self.chunk * k + b - 1
                    if b <= self.chunk and L + 64 < m.max_ctx:
                        self.prefill(0, rnd(L))
                        m.warm_head_blocks(b)
        # continued prefills from a reused prefix in the dense regime (r = 512 / 1024 / 1536: a chunk to 2048 through the
        # sparse layers' listed path and the group-key slab writes at that offset): every bucket
        if self.pc_on and not self.pc_exact:
            A = int(os.environ.get("HY4_PC_ALIGN_DENSE", "512"))
            dense = int(self.cfg.idx_candidate_topk)
            for r0 in range(max(A, self.pc_min), dense, A):
                for b in m.PF_BUCKETS:
                    T = min(b - 1, dense - r0)
                    if next(x for x in m.PF_BUCKETS if x >= T) != b or r0 + b + 64 > m.max_ctx:
                        continue
                    base = rnd(r0 + 1)
                    self.prefill(0, base)
                    self.prefill(0, base[:r0] + rnd(T))
                    m.warm_head_blocks(b)
            # past the dense regime: a full chunk at group offset 768 (slabs of 256 groups; the dense ones above compile
            # the 128-group slabs, smaller chunks write one slab of their size)
            r0, T = dense + 1024, self.chunk - 1
            if int(os.environ.get("HY4_PC_ALIGN", "512")) <= 1024 and r0 + T + 64 < m.max_ctx:
                base = rnd(r0 + 1)
                self.prefill(0, base)
                self.prefill(0, base[:r0] + rnd(T))
        # packed prefills: every (B, Tb) plan up to the slot count (their batched attention / segment slices)
        for B in (2, 4, 8, 16):
            for Tb in m.PF_BUCKETS:
                if B <= self.S and m.pack_plan([Tb - 1] * B) == (B, Tb):
                    self.prefill_many([(s, rnd(Tb - 1), None) for s in range(B)])
                    m.warm_head_blocks(B * Tb)
        # split SWA rings (tt/model.py swa_snapshot / rewind): a prefill continued from a slot's host snapshot after
        # another slot's prefill took the prefill ring (the window's slab copies in and out)
        if self.pc_on and getattr(m, "_swa", None) and self.chunk + 1064 < m.max_ctx:
            a = rnd(self.chunk + 1000)
            self.prefill(0, a)
            self.prefill(1, rnd(64))
            self.prefill(0, a + rnd(64))
        logger.info(f"warm: prefill buckets done; {m.dram_report()}")
        # batched programs: slot 0 past the dense regime (its rows select), the others short (dense rows)
        long_n = self.chunk + 1000 if self.chunk + 1064 < m.max_ctx else int(prompt_len or 64)
        for s in range(self.S):
            self.prefill(s, rnd(long_n if s == 0 else int(prompt_len or 64)))
        dec = self.dec
        top = dec.a0.indexer.gtrash
        widths = sorted({w for w in (2048 * 2**i for i in range(12)) if w + 32 <= top + 32} | {top})
        if dec._lscore():  # (lane-aware score: one width serves; the others compile the same programs)
            widths = [2048]
        dec.trace = False
        # R = 1 runs solo (the single path) when HY4_BATCH_SOLO=1: slot 0 selects, the others are dense; every slot once,
        # so every lane is unloaded once (indexer.lane_unload slices at lane * gcap: one program per lane; a lane first
        # unloaded at serve time compiled and released every trace, served run 2026-11-19)
        solo = [{s: (None, None, False)} for s in range(self.S)] if self.solo_on and self.S > 1 else []
        for R in range(1, self.S + 1):
            for w in widths:
                dec.min_width = w
                self.step({s: (None, None, False) for s in range(R)})
        for r in solo:
            self.step(r)
        # the single path's larger score buckets (slot 0 selects): a context past the first bucket compiled at serve time
        for b in dec.buckets[1:] if solo else []:
            dec.min_bucket = b
            self.step(solo[0])
        dec.min_bucket = 0
        self._warm_spec(solo)
        self._warm_bspec(widths)
        dec.min_width = 0
        # dense-only passes (no row selects: no scoring) at every batch size: slot 0 short again
        self.prefill(0, rnd(int(prompt_len or 64)))
        for R in range(2, self.S + 1):
            self.step({s: (None, None, False) for s in range(R)})
        self._warm_bspec([0])
        m.prefill_caches_mark()  # (the warm-up prefills above ran before any capture: their cached constants stay)
        dec.trace = self.trace
        logger.info(f"warm: eager batched passes done; {m.dram_report()}")
        steps = int(os.environ.get("HY4_WARM_STEPS", "3"))
        # captures in both regimes (slot 0 past the dense regime: its rows select, the passes' score width 2048, the
        # drafts sparse; then slot 0 short: dense-only passes): served runs captured the regime the warm-up left out
        # (and every draft row n) at serve time, up to 620 ms per step. The long chunks' prefill matmuls size their CBs
        # to the L1 room (tt/common.py l1_cb_room), which the captures lower (their L1 residents): the first round's
        # long prefill compiles them (the decoders release their traces at the next pass); rounds repeat until one
        # compiles nothing (the room is stable then)
        L = 2 * self.chunk + 1000 if 2 * self.chunk + 1064 < m.max_ctx else self.chunk + 1000
        prompts = [rnd(L), rnd(int(prompt_len or 64))] if self.chunk + 1064 < m.max_ctx else [rnd(int(prompt_len or 64))]
        for it in range(int(os.environ.get("HY4_WARM_ROUNDS", "3"))):
            n0 = int(m.mesh.num_program_cache_entries())
            self._warm_capture(solo, steps, prompts)
            n1 = int(m.mesh.num_program_cache_entries())
            logger.info(f"warm: capture round {it}: {n1 - n0} programs compiled")
            if n1 == n0:
                break
        for s in range(self.S):
            self.finish(s)
            m.set_slot(s)
            m.reset()
        m.set_slot(0)
        self.kept.clear()
        if self.pooled:  # (the warm-up's extents back to the pool)
            for u in list(m.ext):
                self._free(u)
        logger.info(f"warm: traces captured; {m.dram_report()}")
        self.stats = {k: 0 if isinstance(v, int) else 0.0 for k, v in self.stats.items()}
        if os.environ.get("HY4_PROG_TRACE", "0") == "1":
            _prog_trace(self.model.mesh)
        logger.info(f"Hy4 batch engine warm-up {time.time() - t0:.1f} s ({self.S} slots, widths {widths}, "
                    f"{len(dec.btraces)} batched traces, {len(dec.traces)} single)")

    def _warm_capture(self, solo, steps, prompts=(None,)):
        """The traced passes of warm(): per slot-0 prompt (None: as it is) every batch size, the batched spec, the
        single path (both regimes, every score bucket)."""
        dec = self.dec
        for ids in prompts:
            if ids is not None:
                self.prefill(0, ids)
            for _ in range(2):  # spec first: the drafter's row slices compile at its first traced draft (a trace release)
                self._warm_spec(solo)
            for R in range(1, self.S + 1):
                for _ in range(steps):
                    self.step({s: (None, None, False) for s in range(R)})
            self._warm_bspec([0], steps=steps + 1)
            for r in solo[:2]:  # capture the single-path traces of both regimes
                for _ in range(steps):
                    self.step(r)
            for b in dec.buckets[1:] if solo else []:  # and of every larger score bucket
                dec.min_bucket = b
                for _ in range(steps):
                    self.step(solo[0])
            dec.min_bucket = 0

    def _warm_spec(self, solo):
        """k > 0: the solo path's verify passes (k + 1 rows) and drafts, both regimes (slot 0 selects, slot 1 dense)
        and every larger score bucket (slot 0): per bucket a 1-row pass (re-entering solo), then two verify passes."""
        if not (self.k and solo):
            return
        dec = self.dec
        for b in [0] + list(dec.buckets[1:]):
            dec.min_bucket = b
            for r in (solo[:2] if b == 0 else solo[:1]):
                if self.solo in r:  # leave and re-enter solo: the first pass is a 1-row pass
                    self.step(solo[1] if len(solo) > 1 and r is solo[0] else solo[0])
                for _ in range(3):
                    self.step(r)
                self._warm_drafts()
        dec.min_bucket = 0

    def _warm_drafts(self):
        """The solo sequence's drafts after its last verify pass at every accepted-row count n (a draft slices row n of
        the pass's hidden rows / selections: a program and a trace per n; served runs met the n's the warm-up's random
        tokens never accepted). Drafts only: the sequence's state is untouched (its next step drafts again)."""
        d, s = self.dec, self.solo
        if s is None or not self.k or getattr(d, "last_key", (0,))[0] != self.k + 1:
            return
        q = self.seqs[s]
        for n in range(self.k + 1):
            d.draft(q["tok"], q["pos"], n)

    def _warm_bspec(self, widths, steps=3):
        """Batched spec (HY4_BSPEC): per listed batch size and score width bucket a plain pass, then verify passes (the
        first drafts from one row per sequence, the next from k + 1: both draft programs) of the sequences in slots
        0 .. u - 1 (eager: compile; traced: capture)."""
        dec = self.dec
        for u in sorted(self.bspec):
            for w in widths:
                dec.min_width = w
                self._hn = {}
                self._no_bspec = False
                try:
                    for _ in range(steps):
                        self.step({s: (None, None, False) for s in range(u)})
                finally:
                    self._no_bspec = True
        self._hn = {}

    def capture(self):
        """Batched traces are captured on first use per (rows, width) by tt/decode_batch.Hy4BatchDecoder."""
