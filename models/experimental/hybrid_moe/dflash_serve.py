# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""DFlash speculative decoding on the hybrid MiniMax-M2.7, traced: the single-user step shared by the standalone runner
(run_dflash_traced.py) and vLLM serving (the port's generator_vllm loads `build` through QWEN35_EXT_SPEC).

Per pass two trace replays:
    drafter pass : the previous verify's accepted rows into the drafter's context cache, then the next block's drafts
                   (dflash_tt.DFlashDrafterPaged.pass_body; argmax on the device)
    verify       : the target over the R = k + 1 rows [t, d_1..d_k] (the port's multi-row decode, the hybrid MoE per
                   layer with its host steps between trace segments); the tap layers' outputs are all-gathered on the
                   device straight into the drafter's context input; argmax on the device

Prefill is eager: hooks on the tap layers capture their outputs, which are read to the host and filled into the
drafter's paged cache (`end_prefill`). The traces are captured per request after its prefill and released before the
next one: eager device work after trace replays hung the next replay (2026-11-06). Every persistent buffer (drafter,
verify inputs) exists before the first capture: a buffer allocated after a capture can reuse a trace's intermediates.

The drafter sees the last DFLASH_WINDOW (default 1024) context tokens: its acceptance falls with its context
(window_oracle.py on an 8K prompt: 2.03 tokens per pass at 1024, 1.47 at 4096, 1.10 with the whole context). When the
window is full, the drafter cache is refilled from host copies of the last window's features and the traces re-captured.

Sampling: temperature 0 (or top_k 1) is greedy (drafts accepted while they equal the verify's argmax). Otherwise the
step samples exactly from the target: the drafts are the drafter's argmax (a point-mass proposal), so speculative
sampling (Leviathan et al. 2023) accepts draft d_i with probability p_i(d_i) and, on the first rejection, emits a sample
from p_i with d_i removed; p_i is the verify row's distribution after the request's temperature / top-k / top-p (the
port's SpecSampler, the same processing as its MTP step). The emitted stream is distributed as plain sampling from the
target. The verify's logits stay on the device; a second small trace (Verify.cands_body, replayed only for sampled
requests) takes per-chunk top-64 candidates, which hold any top-k <= 64 exactly (cand_probs checks it; otherwise the
row's logits are read). ~2 ms per pass. Penalties are not applied. The first token comes from the prefill's device
sampler, which caps top-k at 32. DFLASH_SAMPLE_DEBUG=N logs the first N sampled passes' distributions.
env: DFLASH_SNAPSHOT, DFLASH_CMAX (drafter cache, default 2048), DFLASH_WINDOW, DFLASH_TP (drafter split over the mesh,
default 1), DFLASH_WARMUP (serving: tokens of the startup warm-up generation, default 64, 0 = off).
"""

from __future__ import annotations

import json
import os
import time

import torch
from loguru import logger

import ttnn
from dflash_tt import DFlashDrafterPaged, argmax_max, read_argmax
from hybrid import hfpath
from hybrid.patch import HYB, _port

DRAM = ttnn.DRAM_MEMORY_CONFIG
WARMUP_PROMPT = "Write a haiku about the sea."
_SAMPLE_DEBUG = int(os.environ.get("DFLASH_SAMPLE_DEBUG", "0") or 0)  # log the first N sampled passes
LP_DIR = os.environ.get("DFLASH_LP_DIR")  # forced runs: per-token verify log-probs saved here (quality A/B)


def load_drafter(mesh, rows):
    """The DFlash drafter on `mesh` with the target's host embedding table (LM head attached by DFlashServing)."""
    import ref_m27

    ref_m27.SNAP = hfpath.snapshot("MiniMaxAI/MiniMax-M2.7")
    ref_m27.WMAP = json.load(open(f"{ref_m27.SNAP}/model.safetensors.index.json"))["weight_map"]
    snap = os.environ.get("DFLASH_SNAPSHOT") or hfpath.snapshot("nvidia/MiniMax-M2.7-DFlash")
    return DFlashDrafterPaged(
        mesh,
        snap,
        ref_m27.get("model.embed_tokens.weight"),
        None,
        None,
        cmax=int(os.environ.get("DFLASH_CMAX", "2048")),
        rows=rows,
        tp=os.environ.get("DFLASH_TP", "1") == "1",
    )


class Verify:
    """The target's R-row verify pass with persistent inputs."""

    def __init__(self, gen, drafter, R):
        self.rows = _port("mtp_rows")
        self.gen, self.model, self.dr, self.R = gen, gen.model, drafter, R
        m = self.model
        self.mesh = m.mesh_device
        self.BW = self.rows.rows_width(R)
        self.W = int(gen.page_table.shape[-1])
        tok, pos, rot, pt = m.prepare_decode_inputs_host(
            torch.zeros(self.BW, dtype=torch.int32),
            torch.zeros(self.BW, dtype=torch.int32),
            torch.zeros(self.BW, self.W, dtype=torch.int32),
        )
        dev = lambda t: ttnn.to_device(t, self.mesh, memory_config=DRAM)  # noqa: E731
        self.tok, self.pos, self.rot, self.pt = dev(tok), dev(pos), dev(rot), dev(pt)
        self.masks = dev(self.rows.rows_pos_masks_host([-1] * self.BW, self.BW))
        self.trace, self.outs, self.cand_trace, self.cand_outs = None, None, None, None

    def stage(self, tokens, positions, page_table_row):
        row = torch.as_tensor(page_table_row).reshape(-1).to(torch.int32)
        if row.numel() < self.W:  # the plugin's row may be narrower than the bound width: the tail is never read
            row = torch.cat([row, torch.zeros(self.W - row.numel(), dtype=torch.int32)])
        n = len(positions)
        if n > self.BW or len(tokens) != n:
            raise ValueError(f"verify stage: {len(tokens)} tokens / {n} positions for {self.BW} rows")
        pad = self.BW - n  # R below the kernel width (rows_width): the last row repeated, its KV write masked off
        tok, pos, rot, pt = self.model.prepare_decode_inputs_host(
            torch.tensor(list(tokens) + [tokens[-1]] * pad, dtype=torch.int32),
            torch.tensor(list(positions) + [positions[-1]] * pad, dtype=torch.int32),
            row[: self.W].repeat(self.BW, 1),
        )
        for h, d in ((tok, self.tok), (pos, self.pos), (rot, self.rot), (pt, self.pt)):
            ttnn.copy_host_to_device_tensor(h, d)
        masks = self.rows.rows_pos_masks_host(list(positions) + [-1] * pad, self.BW)
        ttnn.copy_host_to_device_tensor(masks, self.masks)

    def body(self):
        m, BW = self.model, self.BW
        cos, sin = m._rope_from_idxs(self.rot)
        x = m.embd(m._decode_tokens_view(self.tok, BW))
        x = ttnn.to_memory_config(ttnn.reshape(x, (1, 1, BW, x.shape[-1])), DRAM)
        taps = []
        # the per-row KV-write index tensors, once per pass for every layer (instead of slice + reshape per layer)
        row_idx = [ttnn.reshape(ttnn.slice(self.masks, (b, 0), (b + 1, BW)), (BW,)) for b in range(BW)]
        for layer in m.layers:
            layer.attention.rows_pos_rows = row_idx
        for li, layer in enumerate(m.layers):
            y = self.rows.layer_forward_rows(
                layer,
                x,
                self.R,
                cos=cos,
                sin=sin,
                position_tensor=self.pos,
                page_table=self.pt,
                rows_pos_masks=self.masks,
                gdn_index=0,
            )
            if li in self.dr.taps:
                taps.append(ttnn.all_gather(y, dim=3, topology=ttnn.Topology.Linear, memory_config=DRAM))
            ttnn.deallocate(x)
            x = y
        for layer in m.layers:
            layer.attention.rows_pos_rows = None
        for t in row_idx:
            ttnn.deallocate(t)
        cat = ttnn.concat(taps, dim=-1)  # [1, 1, BW, n_taps * H], replicated
        if self.BW > self.R:  # the padded rows are not context
            taps.append(cat)
            cat = ttnn.slice(cat, (0, 0, 0, 0), (1, 1, self.R, int(cat.shape[-1])), memory_config=DRAM)
        ttnn.copy(cat, self.dr.c_taps[self.R])
        for t in taps + [cat, cos, sin]:
            ttnn.deallocate(t)
        h = m._final_norm_decode(x)
        logits = ttnn.linear(h, m.lm_head_weight, compute_kernel_config=m._lm_head_ckc())
        for t in (h, x):
            ttnn.deallocate(t)
        out = argmax_max(logits)
        self.logits = logits  # a trace output: sampled requests (and DFLASH_LP_DIR captures) read it back
        return out

    def read_logits_raw(self, rows, vocab):
        """Host FP32 logits [rows, vocab] of the last run."""
        lg = ttnn.to_torch(self.logits, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh, dim=-1)).float()
        return lg.reshape(-1, lg.shape[-1])[:rows, :vocab]

    def read_logits(self, rows, vocab):
        """Host FP32 log-probs [rows, vocab] of the last run."""
        return torch.log_softmax(self.read_logits_raw(rows, vocab), dim=-1)

    # sampled requests read per-chunk top-k candidates of the logits, not the logits: each die's vocab columns are padded to
    # a power of two and cut into CAND_CHUNKS chunks (<= 32768 wide, the multi-core top-k bound), top-CAND_K (<= 64) of
    # each. A chunk can hide part of a row's top-k only if all its CAND_K candidates are in it, so top-k <= CAND_K is
    # always exact (common tokens have low ids: with 32 per chunk the top 40 often filled die 0's first chunk). The host
    # still checks it (cand_probs) and reads the row's logits otherwise (top-k > 64, or no top-k).
    CAND_K, CAND_CHUNKS = 64, 2

    def cands_body(self):
        lg = self.logits
        W = int(lg.shape[-1])
        Wp = 1 << (W - 1).bit_length()
        x = ttnn.pad(lg, [(0, 0), (0, 0), (0, 0), (0, Wp - W)], value=-3.0e38) if Wp != W else lg
        chunks = ttnn.split(x, Wp // self.CAND_CHUNKS, dim=3)
        if x is not lg:
            ttnn.deallocate(x)
        vs, ids = [], []
        for c in chunks:
            v, i = ttnn.topk(c, k=self.CAND_K, dim=-1)
            vs.append(v)
            ids.append(i)
            ttnn.deallocate(c)
        self.cand_chunk = Wp // self.CAND_CHUNKS
        # one values and one ids tensor [1, 1, BW, chunks * K] (ids local to their chunk): two read-backs per pass
        outs = [ttnn.concat(vs, dim=-1), ttnn.concat(ids, dim=-1)]
        for t in vs + ids:
            ttnn.deallocate(t)
        return outs

    def run_cands(self):
        """Candidates of the last run's logits: (values [rows, n], vocab ids [rows, n], per-chunk minima [rows, chunks])."""
        if self.cand_trace is not None:
            ttnn.execute_trace(self.mesh, self.cand_trace, cq_id=0, blocking=False)
            outs = self.cand_outs
        else:
            outs = self.cands_body()
        comp = ttnn.ConcatMeshToTensor(self.mesh, dim=0)
        nd = len(self.mesh.get_device_ids())
        per_die = int(ttnn.get_device_tensors(self.model.lm_head_weight)[0].shape[-1])
        C, K = self.CAND_CHUNKS, self.CAND_K
        v = ttnn.to_torch(outs[0], mesh_composer=comp).float().reshape(nd, -1, C, K)  # [nd, rows, chunks, K]
        i = ttnn.to_torch(outs[1], mesh_composer=comp).long().reshape(nd, -1, C, K)
        i = i + torch.arange(C).view(1, 1, C, 1) * self.cand_chunk
        v = torch.where(i < per_die, v, torch.full_like(v, float("-inf")))  # the pad columns
        i = i + torch.arange(nd).view(nd, 1, 1, 1) * per_die
        rows = v.shape[1]
        v = v.permute(1, 0, 2, 3).reshape(rows, nd * C, K)  # [rows, nd * chunks, K]
        i = i.permute(1, 0, 2, 3).reshape(rows, nd * C, K)
        if outs is not self.cand_outs:
            for t in outs:
                ttnn.deallocate(t)
        return v.reshape(rows, -1), i.reshape(rows, -1), v.min(dim=-1).values

    def run(self):
        HYB.fetch.commit(HYB.slot_of)  # async fetch: last pass's CQ1 writes land before this pass reads the slots
        if self.trace is not None:
            ttnn.execute_trace(self.mesh, self.trace, cq_id=0, blocking=False)
            outs = self.outs
        else:
            outs = self.body()
        return read_argmax(self.mesh, outs, int(ttnn.get_device_tensors(self.model.lm_head_weight)[0].shape[-1]))

    def release(self):
        if self.trace is not None:
            ttnn.release_trace(self.mesh, self.trace)
        if self.cand_trace is not None:
            ttnn.release_trace(self.mesh, self.cand_trace)
        self.trace, self.outs, self.cand_trace, self.cand_outs = None, None, None, None

    def capture(self):
        tid = ttnn.begin_trace_capture(self.mesh, cq_id=0)
        self.outs = self.body()
        ttnn.end_trace_capture(self.mesh, tid, cq_id=0)
        self.trace = tid
        # a separate trace on the verify's persistent logits: greedy passes never pay for it
        tid = ttnn.begin_trace_capture(self.mesh, cq_id=0)
        self.cand_outs = self.cands_body()
        ttnn.end_trace_capture(self.mesh, tid, cq_id=0)
        self.cand_trace = tid


def accept_point_mass(smp, P, drafts):
    """Speculative sampling with point-mass drafts (the drafter's argmax): P(i) is the dense processed target distribution
    [V] of row i (after the drafts before d_i). Accept d_i with probability p_i(d_i); on the first rejection emit a sample
    from p_i without d_i; after k accepts, a sample from p_k. Returns (accepted count, next token)."""
    for i, d in enumerate(drafts):
        pi = P(i).clone()
        if float(torch.rand(1, generator=smp.gen)) < float(pi[d]):
            continue
        pi[d] = 0.0
        tot = float(pi.sum())
        if tot <= 0.0:
            raise RuntimeError(f"DFlash sampling: row {i} rejected draft {d} but has no other mass (top_k {smp.top_k})")
        return i, smp.sample(pi / tot)
    return len(drafts), smp.sample(P(len(drafts)))


def cand_probs(smp, vals, ids, chunk_min, V):
    """The request's processed distribution [V] of one verify row from its device candidates (vals / vocab ids: the top
    CAND_K of every vocab chunk; chunk_min: each chunk's smallest kept value), or None when the candidates may not hold
    the row's top-k: a chunk whose smallest kept value reaches the k-th candidate may hide larger values. Same
    processing as SpecSampler.probs: top-k of the logits / temperature, softmax over them, then top-p."""
    k = smp.top_k
    if not 0 < k <= vals.numel():
        return None
    v, j = torch.topk(vals, k)
    if bool((chunk_min >= v[-1]).any()):
        return None
    p = smp._nucleus(torch.softmax(v.float() * (1.0 / smp.temperature), dim=-1))
    out = torch.zeros(V)
    out[ids[j]] = p
    return out


def _rows_sdpa_shared_from():
    """QWEN35_ROWS_SDPA_SHARED_FROM: the context position from which the verify uses the shared-KV rows SDPA (default
    4096; 0 = never)."""
    return int(os.environ.get("QWEN35_ROWS_SDPA_SHARED_FROM", "4096") or 0)


class DFlashServing:
    """Single-user DFlash step with MtpServing's single-user interface (generator_vllm drives it as its `_mtp`)."""

    adaptive = multi = per_slot = in_trace = False

    def __init__(self, gen, k: int = 3, drafter=None):
        self.gen, self.k, self.R = gen, int(k), int(k) + 1
        self.model = gen.model
        self.mesh = self.model.mesh_device
        self.H, self.V = int(gen.args.dim), int(gen.args.vocab_size)
        self.dr = drafter if drafter is not None else load_drafter(self.mesh, self.R)
        if self.dr.rows != self.R:
            raise ValueError(f"drafter built for {self.dr.rows} verify rows, the step verifies {self.R}")
        self.dr.lm_head, self.dr.vocab = self.model.lm_head_weight, self.V
        self.window = int(os.environ.get("DFLASH_WINDOW", "1024"))
        self.ver = None
        self._hooks = None
        self._pos = self._tok = None
        self.stats = None
        self._sampler, self._sampling = None, None

    # ------------------------------------------------------------------ startup (before any capture)
    def allocate_kv(self, kv_shape, dtype):
        pass  # the drafter's cache is allocated with the drafter; the target's KV belongs to vLLM

    def warm(self):
        """Host library resident and pinned, verify inputs allocated, every program compiled (verify + drafter pass)."""
        t0 = time.time()
        for li in range(len(self.model.layers)):  # the host library's RAM copies load in the background since build
            HYB.lib_layer(li)
        logger.info(f"DFlash: host library resident ({time.time() - t0:.1f} s wait)")
        HYB.fetch.finalize(HYB.loader, HYB.stream)  # HYB_PINX: pinned zero-copy expert library (before any capture)
        if getattr(HYB, "wcache", None) is not None:
            print(HYB.wcache.summary(), flush=True)
        import threading

        logger.info(f"DFlash: python threads {[t.name for t in threading.enumerate()]}")
        self.ver = Verify(self.gen, self.dr, self.R)
        if HYB.elastic_demote:  # HYB_ELASTIC_DEMOTE: full layers <-> KV pool per request (hybrid/elastic.py)
            from hybrid.elastic import Elastic

            HYB.elastic = Elastic(HYB, self.gen, self, HYB.elastic_demote)
        # positions far from any real prompt's rows, on the warm-up page table (blocks the warm-up owns)
        self.ver.stage([0] * self.R, list(range(8, 8 + self.R)), self.gen.page_table[0])
        self._warm_verify()
        self.dr.reset()
        self.dr.stage_pass([], 0)
        self.dr.run("pass", self.dr.pass_body)
        ttnn.synchronize_device(self.mesh)

    def map_prefill(self, prompt_ids, page_table_row, start=0):
        """Serving with an elastic pool (hybrid/elastic.py): before begin_prefill, the profile this prompt needs, then
        vLLM's block ids -> pool blocks (a prefix-cache hit whose blocks are gone restarts at 0). Returns the pool row
        and the start; without an elastic pool, both unchanged."""
        if HYB.elastic is None:
            return page_table_row, start
        self.release()
        HYB.elastic.prepare(len(prompt_ids))
        self._vrow = torch.as_tensor(page_table_row).reshape(-1).to(torch.int32).clone()
        return HYB.elastic.map_row(self._vrow, len(prompt_ids), start)

    def warm_after_switch(self):
        """After an elastic profile switch (no trace live): one eager verify pass compiles the programs of the new KV
        pool / layer kinds and allocates the new hybrid layers' host-step buffers (before any capture). Its K / V rows
        go to the scratch block (every page-table column points there), never to a block a request owns."""
        self.ver = Verify(self.gen, self.dr, self.R)  # the page table's width follows the pool
        row = torch.full_like(self.gen.page_table[0], int(self.gen.kv_scratch_block))
        self.ver.stage([0] * self.R, list(range(8, 8 + self.R)), row)
        self._warm_verify()
        ttnn.synchronize_device(self.mesh)

    def _warm_verify(self):
        """Two eager verify passes per SDPA variant a capture can pick (_set_rows_sdpa): every program compiled before
        any trace capture (the shared-KV op's programs depend on shapes only, not on the positions)."""
        for shared in (False, True) if _rows_sdpa_shared_from() else (False,):
            self._set_rows_sdpa(shared)
            self.ver.run()
            self.ver.run()
        self.ver.run_cands()  # compiles the candidates' programs (eager; outputs freed)
        self._set_rows_sdpa(False)

    def _set_rows_sdpa(self, shared):
        for layer in self.model.layers:
            if hasattr(layer.attention, "paged_k"):
                layer.attention.rows_sdpa_shared = shared

    def warm_prefill_path(self):
        pass  # the served prefill is the generator's eager path (warmed by its own _warm_prefill)

    def capture(self):
        """The per-request traces are captured after each prefill; at startup only the warm-up generation runs (fills
        the fetch LRU slots and touches the pinned / locked host pages, so the first request runs at steady speed)."""
        self._elastic_prewarm()
        n = int(os.environ.get("DFLASH_WARMUP", "64"))
        if n:
            r = self.generate(WARMUP_PROMPT, n, self.gen.page_table[:1])
            logger.info(f"DFlash warm-up: {r['tokens']} tokens at {r['tok_s']} tok/s")

    def _elastic_prewarm(self):
        """HYB_ELASTIC_PREWARM (0 = off, the default; auto = once per HYB_WCACHE; 1 = every start): at startup, every
        demotable layer to hybrid and back. A layer's first demotion converts its hybrid device image (~20 s per layer)
        and the long layouts compile their programs; done here, the first long request does not wait for them (100K
        tokens on a fresh cache: a 243 s switch vs 34 s after the prewarm). Off by default: after the full 24-layer
        round trip the short decode runs ~12% slower for the rest of the process (2026-11-14: 49-52 vs 58-59 tok/s on
        the bench; a 4-layer round trip leaves it unchanged), cause not found. Skipped when the free host RAM cannot
        hold the full-context layout."""
        el, mode = HYB.elastic, os.environ.get("HYB_ELASTIC_PREWARM", "0")
        if el is None or mode == "0" or not el.D:
            return
        import hashlib

        from hybrid import hostmem

        root = os.environ.get("HYB_WCACHE", "")
        tag = hashlib.sha1(f"{list(el.D)}|{el.short_tokens}|{el.long_tokens}".encode()).hexdigest()[:12]
        marker = os.path.join(root, f"elastic_prewarm_{tag}.done") if root else ""
        if mode == "auto" and (not marker or os.path.exists(marker)):
            return
        extra = sum(hostmem.layer_bytes(HYB.lib_dir, HYB.fetch.dir).get(L, 0) for L in el.D)
        if extra > hostmem.available():
            logger.warning(f"HYB_ELASTIC_PREWARM skipped: the full-context layout needs {extra / 1e9:.0f} GB more RAM")
            return
        t0 = time.time()
        self.release()
        el._switch(len(el.D), keep=False)
        el._switch(0, keep=False)
        if marker:
            open(marker, "w").close()
        logger.info(f"HYB_ELASTIC_PREWARM: {len(el.D)} layers to hybrid and back in {time.time() - t0:.1f} s")

    # ------------------------------------------------------------------ prefill
    def release(self):
        """Drop the per-request traces (before any eager device work)."""
        if self.ver is not None and self.ver.trace is not None:
            ttnn.synchronize_device(self.mesh)
        if self.ver is not None:
            self.ver.release()
        for name in list(self.dr.traces):
            ttnn.release_trace(self.mesh, self.dr.traces.pop(name))
        self.dr.outs = {}

    # a prefix-cache hit may resume the prefill at `start` (begin_prefill), as long as the feature window's rows before
    # it are still held from the previous request (feature_start)
    prefix_start_ok = True

    def _held(self, prompt_ids, upto):
        """The previous request's context features for prompt rows [P - window, upto), or None if any is not held (the
        rows' ids must match the context they were computed in, from position 0)."""
        lo = max(0, len(prompt_ids) - self.window)
        if upto <= lo:
            return torch.zeros(0, 0)
        ids = getattr(self, "_ctx_ids", None)
        if ids is None:
            return None
        first = len(ids) - self.n_hist
        if first > lo or len(ids) < upto or ids[:upto] != list(prompt_ids[:upto]):
            return None
        return torch.cat(self.hist)[lo - first : upto - first]

    def feature_start(self, prompt_ids, cached: int) -> int:
        """The prefill start for a prompt whose first `cached` rows hold KV: `cached` when the window's features before
        it are held, otherwise the window's first row (or `cached`, if earlier)."""
        lo = max(0, len(prompt_ids) - self.window)
        cached = int(cached)
        return cached if cached <= lo or self._held(prompt_ids, cached) is not None else lo

    def begin_prefill(self, prompt_ids, page_table_row, slot: int = 0, start: int = 0):
        if int(slot) != 0:
            raise ValueError(f"DFlash serving is single-user (--max-num-seqs 1); prefill into slot {slot}")
        self._log_stats("previous request")
        self.release()
        if HYB.elastic is not None:  # before the page-table row is read: a switch rewrites the generator's table
            HYB.elastic.prepare(len(prompt_ids))
        self.dr.reset()
        self._row = torch.as_tensor(page_table_row).reshape(-1).to(torch.int32).clone()
        self._P = len(prompt_ids)
        # per tap: [first row, host rows]; only the rows from P - window on are kept (end_prefill reads no more), and a
        # segment that ends before them is not read back. The mesh concat's first H columns are the layer output (prefill
        # outputs are width-sharded over the dies; a replicated output holds tp copies side by side).
        keep_from = max(0, self._P - self.window)
        self._ids = list(prompt_ids)
        self._reuse = None
        if int(start) > keep_from:
            self._reuse = self._held(prompt_ids, int(start))
            if self._reuse is None:
                raise ValueError(
                    f"prefill start {start} skips features of the last {self.window} rows of a {self._P}-token prompt "
                    f"that the previous request's context does not hold"
                )
        captured = {li: [0, []] for li in self.dr.taps}
        seen = {li: int(start) for li in self.dr.taps}
        orig = {}
        for li in self.dr.taps:
            layer = self.model.layers[li]
            orig[li] = layer.forward

            def hook(*a, _f=orig[li], _li=li, **kw):
                y = _f(*a, **kw)
                off, rows = seen[_li], int(y.shape[-2])
                seen[_li] = off + rows
                if off + rows > keep_from:
                    if not captured[_li][1]:
                        captured[_li][0] = off
                    captured[_li][1].append(
                        ttnn.to_torch(y, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh, dim=-1)).float()
                    )
                return y

            layer.forward = hook
        self._hooks = (orig, captured)

    def _unhook(self):
        if self._hooks is None:
            return None
        orig, captured = self._hooks
        for li, f in orig.items():
            self.model.layers[li].forward = f
        self._hooks = None
        return captured

    def abort_prefill(self):
        self._unhook()

    def end_prefill(self, tok0, P):
        captured = self._unhook()
        if captured is None:
            raise RuntimeError("end_prefill without begin_prefill")
        P = int(P)
        if P != self._P:
            raise ValueError(f"end_prefill P={P} but begin_prefill saw {self._P} prompt tokens")
        W = self.window
        tails = []
        for li in self.dr.taps:
            off, pieces = captured[li]
            rows = torch.cat([c.reshape(-1, c.shape[-1])[:, : self.H] for c in pieces])[: P - off]
            if off + rows.shape[0] != P:
                raise RuntimeError(f"tap {li}: hooks captured rows {off}..{off + rows.shape[0]} for a {P}-token prompt")
            tails.append(rows[-W:])
        feats = torch.cat(tails, dim=-1)
        if self._reuse is not None and self._reuse.shape[0]:
            feats = torch.cat([self._reuse.to(feats.dtype), feats])[-W:]
        self._reuse = None
        self.hist = [feats.clone()]  # host copies of the context features (refills; the next request's prefix)
        self.n_hist = self.hist[0].shape[0]
        self._ctx_ids = self._ids  # the context's ids; the last n_hist have their features in hist
        self.dr.fill_prompt(feats[-W:], start=max(0, P - W))
        self._capture(P)
        self._pos, self._tok, self._prev_acc = P, int(tok0), []
        self._sampler, self._sampling = None, None  # the request's SpecSampler (created by its first sampled step)
        self.stats = {"passes": 0, "tokens": 0, "hist": [0] * self.R, "draft": 0.0, "verify": 0.0, "wall": 0.0}
        self.stats["refills"] = 0
        self.stats["stage"] = 0.0

    def _capture(self, pos):
        """Capture the drafter and verify traces for decoding from context position `pos`. Traces whose context reaches
        QWEN35_ROWS_SDPA_SHARED_FROM run the verify's SDPA through the shared-KV rows op (attn_decode/rows_sdpa.py: each
        K / V chunk read once for all rows; 40K context ~20% faster per token); shorter ones keep the stock decode
        (cheaper below ~2K positions). Both variants are compiled by _warm_verify."""
        ttnn.synchronize_device(self.mesh)
        thr = _rows_sdpa_shared_from()
        self._set_rows_sdpa(bool(thr) and int(pos) >= thr)
        self.dr.capture("pass", self.dr.pass_body)
        self.ver.capture()
        ttnn.synchronize_device(self.mesh)

    # ------------------------------------------------------------------ decode
    def step(self, tok: int, pos: int, page_table_row=None, host_authoritative=False, sampling=None) -> list[int]:
        """Returns the accepted drafts followed by the verified next token (1..R ids). The step owns the continuation
        (as MtpServing.step: on the plugin's steady async path its host token can lag by a step); the plugin's tok / pos
        are checked when it flags the host inputs authoritative (the first decode after a prefill)."""
        if self._pos is None:
            raise RuntimeError("DFlash step before any prefill")
        if host_authoritative and (int(tok), int(pos)) != (self._tok, self._pos):
            raise RuntimeError(
                f"DFlash step input (tok {tok}, pos {pos}) differs from the step's continuation "
                f"(tok {self._tok}, pos {self._pos})"
            )
        smp = self._sampler_for(sampling)
        if page_table_row is not None:
            self._row = torch.as_tensor(page_table_row).reshape(-1).to(torch.int32)
            # every block the verify window [pos, pos + R) writes must be allocated: vLLM pads a row with block 0 (its
            # null block), so a write there is lost and the real block keeps another request's K / V (the plugin shim
            # reserves 2K + 1 lookahead slots for this window; positions past the context are never read)
            bs, limit = int(self.gen.block_size), int(self.gen.max_seq_len)
            blocks = range(self._pos // bs, min(self._pos + self.R - 1, limit - 1) // bs + 1)
            null = [b for b in blocks if b >= self._row.numel() or int(self._row[b]) == 0]
            if null:
                raise RuntimeError(
                    f"DFlash verify window {self._pos}..{self._pos + self.R - 1} reaches unallocated block(s) {null}: "
                    f"row {self._row[: max(blocks) + 2].tolist()}, plugin pos {pos}, block size {bs}, context {limit}"
                )
        a0 = time.perf_counter()
        if HYB.elastic is not None and self._pos + self.R + 1 > HYB.elastic.capacity:
            # the context reached the elastic short pool: long profile now, keeping the pool's blocks (hybrid/elastic.py)
            HYB.elastic.grow_now(self._pos + self.R + 1)
            if page_table_row is None:  # the runner's row is the generator's table (the plugin passes it every step)
                self._row = self.gen.page_table[0].clone()
            self._capture(self._pos)
        if page_table_row is not None and HYB.elastic is not None and HYB.elastic.bmap is not None:
            self._row, _ = HYB.elastic.map_row(self._row, self._pos + self.R + 1)  # serving: vLLM ids -> pool blocks
        dr, ver, K, R, W = self.dr, self.ver, self.k, self.R, self.window
        t, p = self._tok, self._pos
        if dr.need_refill():  # the window is full: refill from the last W features (traces re-captured)
            self.release()
            self.hist = [torch.cat(self.hist)[-W:]]
            self.n_hist = self.hist[0].shape[0]
            dr.fill_prompt(self.hist[0], start=p - self.n_hist)
            self._prev_acc = []
            self._capture(p)
            self.stats["refills"] += 1
        s0 = time.perf_counter()
        dr.stage_pass(self._prev_acc, t)  # the previous verify's accepted rows (c_taps[R] filled by the verify)
        s1 = time.perf_counter()
        drafts = dr.run("pass", None)[1 : 1 + K]
        a1 = time.perf_counter()
        ver.stage([t] + drafts, list(range(p, p + R)), self._row)
        s2 = time.perf_counter()
        top = ver.run()[:R]
        a2 = time.perf_counter()
        if self._force is not None:  # fixed-text benchmark: the recorded continuation replaces the verify's argmax
            i = self._force_i
            top = self._force[i + 1 : i + 1 + R]
            if len(top) < R:
                raise RuntimeError(f"DFlash forced decode ran past the recorded {len(self._force)} tokens")
        if smp is None or self._force is not None:
            a = 0
            while a < K and drafts[a] == top[a]:
                a += 1
        else:
            c0 = time.perf_counter()
            cands = self.ver.run_cands()
            dense = {}

            def P(i):  # row i's processed target distribution, built when the accept loop reaches it
                if i not in dense:
                    cv, ci = cands[0][i], cands[1][i]
                    hit = ci == top[i]
                    if not bool(hit.any()) or float(cv[hit].max()) != float(cv.max()):
                        # the device argmax is every chunk set's maximum: a mismatch means the vocab ids are misread
                        raise RuntimeError(
                            f"DFlash sampling: row {i} device argmax {top[i]} is not the candidates' maximum "
                            f"(candidate argmax {int(ci[cv.argmax()])}, value {float(cv.max())})"
                        )
                    dense[i] = cand_probs(smp, cands[0][i], cands[1][i], cands[2][i], self.V)
                    if dense[i] is None:  # the candidates may miss part of the request's top-k: the exact row
                        self.stats["full_rows"] = self.stats.get("full_rows", 0) + 1
                        dense[i] = smp.probs(self.ver.read_logits_raw(i + 1, self.V)[i])
                return dense[i]

            a, nxt = accept_point_mass(smp, P, drafts)
            self.stats["sample"] = self.stats.get("sample", 0.0) + time.perf_counter() - c0
            if _SAMPLE_DEBUG and self.stats["passes"] < _SAMPLE_DEBUG:
                tk = self.gen.tokenizer
                rows = []
                for i in range(R):
                    pr = P(i)
                    v, ix = pr.topk(4)
                    rows.append(" ".join(f"{tk.decode([int(j)])!r}:{float(x):.2f}" for x, j in zip(v, ix)))
                logger.info(
                    f"DFLASH_SAMPLE pos {p} in {tk.decode([t])!r} drafts {[tk.decode([d]) for d in drafts]} "
                    f"argmax {[tk.decode([x]) for x in top[:R]]} -> accepted {a}, next {tk.decode([nxt])!r} | "
                    + " || ".join(rows)
                )
            top = top[:a] + [nxt]  # rows < a: the accepted drafts; row a: the emitted token
        if self._force is not None:
            if LP_DIR:  # rows 0..a predict the tokens this pass commits (the recorded text)
                lp = ver.read_logits(a + 1, self.V)
                tgt = torch.tensor(top[: a + 1])
                tl, ti = lp.topk(32, dim=-1)
                self._lp.append((lp.gather(1, tgt[:, None])[:, 0], ti.int(), tl.half()))
            self._force_i += a + 1
        rows_h = ttnn.to_torch(ttnn.get_device_tensors(dr.c_taps[R])[0]).float().reshape(R, -1)[: a + 1]
        self.hist.append(rows_h)
        self.n_hist += rows_h.shape[0]
        self._ctx_ids += [t] + drafts[:a]
        if self.n_hist > 2 * W:  # bound the host copy
            self.hist = [torch.cat(self.hist)[-W:]]
            self.n_hist = W
        out = drafts[:a] + [top[a]]
        self._prev_acc = list(range(p, p + a + 1))
        self._tok, self._pos = top[a], p + a + 1
        st = self.stats
        st["passes"] += 1
        st["tokens"] += len(out)
        st["hist"][a] += 1
        st["draft"] += a1 - a0
        st["verify"] += a2 - a1
        st["stage"] += (s1 - s0) + (s2 - a1)
        st["wall"] += time.perf_counter() - a0
        if st["passes"] % 64 == 0:
            self._log_stats("running")
        return out

    def _sampler_for(self, sampling):
        """The request's SpecSampler, or None for greedy decoding. Created on the request's first step (end_prefill
        clears it) so its generator (seeded by the request's seed) runs across the whole request."""
        if sampling is None:
            return None
        if self._sampler is None or self._sampling != tuple(sampling):
            t, k, p, seed = sampling
            self._sampler = _port("mtp_sampling").SpecSampler(t, k, p, seed)
            self._sampling = tuple(sampling)
            logger.info(f"DFlash request sampling: temperature {t} top_k {k} top_p {p} seed {seed}")
        return None if self._sampler.greedy else self._sampler

    def _log_stats(self, why):
        st = self.stats
        if st and st["passes"]:
            n = st["passes"]
            logger.info(
                f"DFLASH_STATS ({why}): {n} passes, {st['tokens'] / n:.2f} tokens per pass, acceptance hist "
                f"{st['hist']}, draft {1e3 * st['draft'] / n:.1f} + verify {1e3 * st['verify'] / n:.1f} ms per pass, "
                f"{st['refills']} refills, sampling {1e3 * st.get('sample', 0.0) / n:.1f} ms per pass "
                f"({st.get('full_rows', 0)} full-row reads), stage {1e3 * st['stage'] / n:.1f} ms -> "
                f"{st['tokens'] / st['wall']:.1f} tok/s (decode steps only)"
            )

    # ------------------------------------------------------------------ standalone generation
    _force = None  # generate(force=...): the recorded token sequence a fixed-text run decodes

    def generate(self, text, n_new, page_table, chat=True, force=None):
        """Eager prefill + traced DFlash decode of one prompt on `page_table` [1, W] (the runner; the serving warm-up).
        Returns the generated ids and timings; the traces are released at the end. ``force`` = a recorded output
        (ids, first token first): every pass accepts the drafts that match it and continues with it, so runs of
        different configurations decode the same text (same verify inputs, comparable cost per pass)."""
        tok = self.gen.tokenizer
        if chat:
            text = tok.apply_chat_template(
                [{"role": "user", "content": text}], add_generation_prompt=True, enable_thinking=False, tokenize=False
            )
        ids = tok.encode(text, add_special_tokens=False)
        P = len(ids)
        if HYB.elastic is not None:  # the profile first: a switch replaces the generator's page table (runner)
            self.release()
            HYB.elastic.prepare(P)
            page_table = self.gen.page_table[: page_table.shape[0]]
        self.gen.reset()
        t_pf = time.time()
        self.begin_prefill(ids, page_table[0])
        try:
            logits0 = self.gen.prefill_forward(torch.tensor(ids).reshape(1, -1), page_table=page_table, prompt_lens=[P])
        except Exception:
            self.abort_prefill()
            raise
        t0 = int(torch.as_tensor(logits0).reshape(-1, logits0.shape[-1])[-1, : self.V].argmax())
        if force is not None:
            t0 = int(force[0])
            n_new = min(n_new, len(force) - R_MAX_PAD)
        self._force, self._force_i = (list(force), 0) if force is not None else (None, 0)
        self._lp = []
        self.end_prefill(t0, P)
        t_pf = time.time() - t_pf
        HYB.reset_stats()
        out, pos = [t0], P
        t1 = time.time()
        while len(out) < n_new:
            new = self.step(out[-1], pos)
            out += new
            pos += len(new)
        dt = time.time() - t1
        if LP_DIR and self._force is not None and self._lp:
            os.makedirs(LP_DIR, exist_ok=True)
            tag = str(len(os.listdir(LP_DIR)))
            torch.save(
                {
                    "prompt": text[:80],
                    "ids": out[:n_new],
                    "prompt_ids": list(ids),  # the templated prompt (a reference model can teacher-force the same text)
                    "tgt_lp": torch.cat([x[0] for x in self._lp]),
                    "top_ids": torch.cat([x[1] for x in self._lp]),
                    "top_lp": torch.cat([x[2] for x in self._lp]),
                },
                os.path.join(LP_DIR, f"lp_{tag}.pt"),
            )
        self._force = None
        self.release()
        st, n = self.stats, max(1, self.stats["passes"])
        return {
            "prompt_tokens": P,
            "ids": out[:n_new],
            "tokens": len(out),
            "passes": st["passes"],
            "tokens_per_pass": round((len(out) - 1) / n, 3),
            "accept_hist": st["hist"],
            "ms_per_token": round(1e3 * dt / (len(out) - 1), 1),
            "tok_s": round((len(out) - 1) / dt, 2),
            "ms_per_pass": {
                "draft": round(1e3 * st["draft"] / n, 1),
                "verify": round(1e3 * st["verify"] / n, 1),
                "stage": round(1e3 * st["stage"] / n, 2),  # host input uploads of both (inside draft + verify)
            },
            "prefill_s": round(t_pf, 2),
            "drafter_refills": st["refills"],
        }


R_MAX_PAD = 8  # a forced run stops this many tokens before the end of its recording (a pass reads R ahead)


def build(gen, k):
    """generator_vllm's QWEN35_EXT_SPEC factory."""
    return DFlashServing(gen, k)
