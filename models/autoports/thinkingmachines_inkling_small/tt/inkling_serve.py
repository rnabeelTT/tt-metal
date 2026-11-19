# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Single-user generation engine for Inkling: chunked prefill, then speculative decoding with the native MTP drafter
(k drafts per verify pass; k = 0 is plain decode). The vLLM adapter (tt/generator_vllm.py) drives it as its
speculative step object; tools can drive it directly (generate()).

One step (k drafts d_1..d_k pending from the previous step, last emitted token x at position p):
  verify   the target on rows [x, d_1 .. d_k] at positions p .. p + k (rows mode, conv states held), one logits row
           each; greedy: n = the leading drafts equal to the argmax rows, next = argmax row n; sampled: exact
           speculative sampling (min(1, p / q) accept, residual resample: M2.7's SpecSampler)
  commit   the target's conv states advance past the n + 1 accepted rows
  draft    MTP depth 0 over the n + 1 accepted rows (the target's normed hidden of row j, the token after it), its last
           row -> d'_1; depths 1 .. k - 1 chain one row each -> d'_2 .. d'_k
Emits d_1 .. d_n, next. The output equals plain greedy decode token for token (tools/spec_bench.py checks it).

Traces (trace=True): every verify width 1 .. k + 1 and its commits (n = 1 .. R), the drafter's depth 0 per (verify
width, accepted rows) and each deeper depth. The hybrid MoE's host step runs between trace segments (hybrid/seg.py).
Rules learned in the dev harness (each was a real bug): run one eager pass at every row width AND the drafter at
every width / depth BEFORE any capture, then capture everything up front -- a device tensor a layer allocates lazily
after a capture can sit in that trace's freed intermediates and is overwritten by its replays; the drafter runs on
its own CCL semaphores (mtp.own_ccl); hybrid needs HYB_DECODE_ROWS >= the verify width; logits are untilized on the
device before the readback (a tile readback moves 32 padded rows).

KV: the 7 global layers keep their own paged caches (identity page table), the sliding layers their rings, the MTP
depths theirs; vLLM's KV allocation is unused (single user, no prefix caching). Every prefill starts a new sequence.
"""

from __future__ import annotations

import os
import time

import numpy as np
import torch
import ttnn
from loguru import logger

from models.autoports.thinkingmachines_inkling_small.tt import mm as mm_rows
from models.autoports.thinkingmachines_inkling_small.tt.inkling_mtp import InklingMtp
from models.autoports.thinkingmachines_inkling_small.tt.lazy_guard import LazyCacheGuard
from models.autoports.thinkingmachines_inkling_small.tt.sampling import InklingSampler


def _hyb():
    """The hybrid overlay's state when it is installed (models/experimental/hybrid_moe), else None."""
    import sys

    return getattr(sys.modules.get("hybrid.patch"), "HYB", None)


def install_program_log(mesh, is_on):
    """Debug (INKLING_PROGLOG=1): every ttnn op called while is_on() that grows the program cache is logged with its
    tensor arguments' shapes (which programs compile after the trace captures)."""
    import functools

    state = {"n": None}

    def desc(v):
        if isinstance(v, ttnn.Tensor):
            return f"{tuple(v.shape)}:{str(v.dtype).split('.')[-1]}"
        if isinstance(v, (list, tuple)) and v and isinstance(v[0], ttnn.Tensor):
            return f"[{len(v)} tensors]"
        return None

    def wrap(name, f):
        @functools.wraps(f)
        def g(*a, **k):
            out = f(*a, **k)
            try:
                n = int(mesh.num_program_cache_entries())
            except Exception:  # noqa: BLE001
                return out
            if state["n"] is not None and n > state["n"] and is_on():
                args = [d for d in (desc(v) for v in list(a) + list(k.values())) if d]
                logger.warning(f"PROGLOG +{n - state['n']} program(s): {name} {args}")
            state["n"] = n
            return out

        return g

    import ttnn.experimental as tx

    for mod, prefix in (
        (ttnn, "ttnn."),
        (tx, "ttnn.experimental."),
        (getattr(ttnn, "transformer", None), "ttnn.transformer."),
    ):
        if mod is None:
            continue
        for name in dir(mod):
            if name.startswith("_") or name in (
                "to_torch",
                "from_torch",
                "synchronize_device",
                "deallocate",
                "execute_trace",
                "begin_trace_capture",
                "end_trace_capture",
            ):
                continue
            f = getattr(mod, name, None)
            if callable(f) and type(f).__name__ in (
                "FastOperation",
                "Operation",
                "builtin_function_or_method",
                "nb_func",
                "function",
            ):
                try:
                    setattr(mod, name, wrap(prefix + name, f))
                except (AttributeError, TypeError):
                    pass


def prefill_buckets(chunk: int):
    """Row counts a (last) prefill chunk is padded to: 128, 256, ... up to the chunk."""
    out, b = [], 128
    while b < chunk:
        out.append(b)
        b *= 2
    return out + [chunk]


class InklingSpecServing:
    """The engine; also the vLLM adapter's speculative step object (`step` / `warm` / `capture` / `k`)."""

    multi = False  # the plugin-facing flags the shared adapter code reads
    adaptive = False
    in_trace = False

    def __init__(self, model, args, mesh, k: int, trace: bool = True):
        self.model, self.args, self.mesh = model, args, mesh
        self.k = int(k)
        self.trace = bool(trace)
        self.rep = ttnn.ReplicateTensorToMesh(mesh)
        self.V = int(args.unpadded_vocab_size)
        self.mup = float(args.logits_mup_width_multiplier)
        att = model.layers[0].attention
        self.max_seq_len = int(args.max_seq_len)
        self.chunk = int(att.prefill_chunk)
        self.block = int(att.block)
        self.buckets = prefill_buckets(self.chunk)
        self.pt_host = torch.arange(att.max_blocks, dtype=torch.int32)
        self.pt = ttnn.from_torch(
            self.pt_host.reshape(1, -1),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh,
            mesh_mapper=self.rep,
        )
        self.mtp = InklingMtp(model, self.k) if self.k else None
        # INKLING_MTP_PROMPT (default 1): drafter depth 0 also runs over the prompt in prefill (its KV / conv state then
        # holds the prompt, as vLLM's MTP does), instead of starting each request from an empty cache
        self.mtp_prompt = self.mtp is not None and os.environ.get("INKLING_MTP_PROMPT", "1") == "1"
        # INKLING_SUFFIX (default 0): prompt-lookup ("suffix") drafts, M2.7's hybrid/suffix_draft.py. The continuation
        # of the longest earlier occurrence of the context's last tokens replaces the MTP drafts (same verify width)
        # when the request's head-to-head tally for that match length favours it; the drafter still runs every pass.
        # Off: at k=2 MTP already takes the copy spans (2.1-2.4 tokens/step on code edits and quotes), the tally never
        # switched (served A/B, 2026-11-17, HANDOFF_INKLING.md); its gain needs wider verify passes than k + 1
        self.sfx_on = self.mtp is not None and os.environ.get("INKLING_SUFFIX", "0") == "1"
        self._sfx = self._choose = None
        self.sconvs = [
            c
            for layer in model.layers
            for c in (layer.attention.k_sconv, layer.attention.v_sconv, layer.attn_sconv, layer.mlp_sconv)
            + ((layer.attn_sconv_r, layer.mlp_sconv_r) if getattr(layer, "rep", None) is not None else ())
        ]
        # Prefix cache (INKLING_PREFIX_CACHE, default 1; vLLM's own prefix caching stays off: the engine keeps every
        # cache itself). The global layers' paged KV is indexed by position, so a new prompt that shares the previous
        # prompt's first b tokens can keep it; the per-sequence state at b is the 4 short-conv states of every layer
        # (device copies taken during the prefill, when a layer's chunk ends at b) and the sliding layers' rings (read to
        # host after the prefill: a ring holds the window + a chunk + a block, so the 512 positions before b, the last
        # chunk boundary before the prompt's end, are still in it). The next prompt with the same first b tokens
        # restores both and prefills from b. INKLING_PREFIX_MIN (default 2048): the shortest prefix snapshotted.
        self.px_on = os.environ.get("INKLING_PREFIX_CACHE", "1") == "1"
        self.px_min = int(os.environ.get("INKLING_PREFIX_MIN", "2048"))
        self._px = None
        # vLLM-paged serving (enable_vllm_paging, set by the vLLM adapter): the global layers' KV through the request's
        # vLLM block table, vLLM's prefix cache for it, and the per-sequence state at a cached prefix's end (conv states,
        # sliding rings) in the plugin's prefix-state slots (_ps: slot -> host snapshot) instead of the engine's own entry
        self.paged = False
        self._ps = {}
        if self.px_on:
            blocks = list(model.layers) + ([self.mtp.depths[0].block] if self.mtp_prompt else [])
            convs = lambda bl: (bl.attention.k_sconv, bl.attention.v_sconv, bl.attn_sconv, bl.mlp_sconv)  # noqa: E731
            # snapshot buffers allocated here, before any trace capture (a buffer allocated after the captures can land
            # in a trace's freed intermediates); copies into / out of them run the program reset() compiles
            self._px_convs = [
                [(c, ttnn.clone(c._zero, memory_config=c.st.memory_config())) for c in convs(bl)] for bl in blocks
            ]
            self._px_rings = [
                t for bl in blocks if bl.attention.sliding for t in (bl.attention.k_cache, bl.attention.v_cache)
            ]
        # global layers scale q / the relative bias by tau(pos) = 1 + alpha log(max(pos / n_floor, 1)) past n_floor
        # positions (128000): with a context beyond it, every verify pass carries a per-row tau tensor (ones below)
        self.n_floor = args.inkling.get("log_scaling_n_floor")
        self.alpha = float(args.inkling.get("log_scaling_alpha") or 0.0)
        self.long = self.n_floor is not None and self.max_seq_len > int(self.n_floor)
        # Inkling's block markers (<|end_message|>, <|message_model|>, <|content_text|>, ...): one step emits either text
        # tokens or one marker, never both -- vLLM's streaming Inkling parser misses a marker that arrives in the same
        # delta as text (it leaked "<|end_message|>" into content). INKLING_SPLIT_MARKERS=0: off.
        self.split_markers = os.environ.get("INKLING_SPLIT_MARKERS", "1") == "1"
        self.markers = self._marker_ids(args)
        self.end_message = self._token_id(args, "<|end_message|>")
        eos = args.inkling.get("eos_token_id", 200006)
        self.eos = set(eos) if isinstance(eos, (list, tuple)) else {int(eos)}
        # verify traces: R -> persistent inputs (ids, positions, kv index), trace id, outputs; commits (R, n) -> id
        self.bufs, self.tid, self.out, self.keep, self.ctid = {}, {}, {}, {}, {}
        self.last_R = None  # the width of the last traced verify (its commit traces read that capture's conv history)
        self.last_traced = False
        # drafter traces
        self.t0, self.tj = {}, {}
        self.tc = {}  # greedy draft chains: (verify width, accepted rows) -> (trace, ids [1, k])
        self.chain = os.environ.get("INKLING_DRAFT_CHAIN", "1") == "1"
        self.d_ids = self.d_pos = self.prev = None
        self._prog_n = None
        self.recaptures = 0
        self.captured = False
        # the prefill's lazily built device caches: snapshot at the end of the captures, purged after every prefill
        # (tt/lazy_guard.py: a cache built after a capture is overwritten by the replays)
        ffs = [getattr(layer, "feed_forward", None) for layer in model.layers]
        if self.mtp is not None:
            ffs += [getattr(getattr(dp, "block", None), "feed_forward", None) for dp in self.mtp.depths]
        self.guard = LazyCacheGuard(ffs, [args])
        # INKLING_WATCHDOG=N (seconds, default 600; 0 = off): a prefill / decode step that makes no progress for N s dumps
        # every Python thread's stack to stderr (host-side vs device-wait hangs), once per stall
        self.watchdog = float(os.environ.get("INKLING_WATCHDOG", "600"))
        # INKLING_MEMDUMP=dir (debug): a native thread copies the large host mappings (command-queue sysmem) into dir
        # when steps stop for INKLING_MEMDUMP_SECS (default 90)
        self._memdump = None
        if os.environ.get("INKLING_MEMDUMP"):
            import ctypes

            so = os.path.join(os.path.dirname(__file__), "../../../experimental/hybrid_moe/hybrid/pin/memdump.so")
            self._memdump = ctypes.CDLL(os.path.abspath(so))
            self._memdump.memdump_start.argtypes = [ctypes.c_double, ctypes.c_char_p, ctypes.c_uint64, ctypes.c_uint64]
            self._memdump.memdump_start(
                float(os.environ.get("INKLING_MEMDUMP_SECS", "90")),
                os.environ["INKLING_MEMDUMP"].encode(),
                1 << 30,
                1 << 29,
            )
        if os.environ.get("INKLING_PROGLOG") == "1":
            install_program_log(mesh, lambda: self.captured)
        # the continuation: the next token to verify, its position, the pending drafts (+ their q when sampling)
        self._tok = self._pos = None
        self._drafts, self._qs = [], None
        self._pending = []
        self._smp = InklingSampler(0.0)
        # the next step takes the plugin's token as authoritative (after a host-sampled step / logits prefill)
        self.adopt_next = False
        self.stats = {"steps": 0, "emitted": 0, "t_verify": 0.0, "t_draft": 0.0, "t_prefill": 0.0, "prefills": 0}

    @staticmethod
    def _token_id(args, text):
        try:
            return int(args.tokenizer.convert_tokens_to_ids(text))
        except Exception:  # noqa: BLE001
            return -1

    @staticmethod
    def _marker_ids(args):
        tok = getattr(args, "tokenizer", None)
        try:
            ids = set(int(i) for i in tok.all_special_ids)
            dec = getattr(tok, "added_tokens_decoder", {}) or {}
            ids |= {int(i) for i, t in dec.items() if getattr(t, "special", False)}
            return ids
        except Exception:  # noqa: BLE001 - no tokenizer: every id at or past the first special token (200000)
            return set(range(200000, 201088))

    # ------------------------------------------------------------------ device inputs
    def _ids_host(self, ids):
        return ttnn.from_torch(
            torch.tensor(ids, dtype=torch.int32).reshape(1, len(ids)),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=self.rep,
        )

    def _pos_host(self, p0, n):
        return ttnn.from_torch(
            torch.arange(p0, p0 + n, dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=self.rep,
        )

    @staticmethod
    def _idx_torch(p0, n):
        """kv_rows_update's [n, n] index: row u -> position p0 + u on the diagonal, -1 elsewhere."""
        m = torch.full((n, n), -1, dtype=torch.int32)
        m[torch.arange(n), torch.arange(n)] = torch.arange(p0, p0 + n, dtype=torch.int32)
        return m

    def _idx_host(self, p0, n):
        return ttnn.from_torch(
            self._idx_torch(p0, n), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=self.rep
        )

    def _tau_host(self, p0, n):
        pos = torch.arange(p0, p0 + n, dtype=torch.float64) + 1.0
        tau = (1.0 + self.alpha * torch.log(torch.clamp(pos / float(self.n_floor), min=1.0))).float()
        return ttnn.from_torch(
            tau.reshape(1, n, 1, 1), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, mesh_mapper=self.rep
        )

    def _dev(self, host_t):
        return ttnn.to_device(host_t, self.mesh)

    @staticmethod
    def _rm_logits(lg):
        """Logits untilized on the device (same values): a tile readback moves all 32 padded rows."""
        if lg.is_sharded():
            lg = ttnn.sharded_to_interleaved(lg, ttnn.DRAM_MEMORY_CONFIG)
        return ttnn.untilize_with_unpadding(lg, [int(d) - 1 for d in lg.shape])

    def _read(self, lg, R):
        return ttnn.to_torch(ttnn.get_device_tensors(lg)[0]).float().reshape(-1, lg.shape[-1])[:R, : self.V]

    # ------------------------------------------------------------------ target: verify rows + commits
    def _body(self, di, dp, dk, R, dt=None):
        model = self.model
        x = model.embd(di)
        x = ttnn.reshape(x, (1, 1, R, x.shape[-1]))
        for layer in model.layers:
            x = layer.forward_rows(x, dp, self.pt, commit_all=False, kv_idx=dk, tau=dt)
        h = model._final_norm_decode(x)  # replicated, normed (the model's 1/16 folded into the norm weight)
        # the drafter wants final_norm(h) without that 1/16: x16
        return self._rm_logits(model._lm_head(h)), ttnn.multiply(h, self.mup)

    def _rows_eager(self, ids, p0):
        """Eager rows pass: -> (logits [R, V] fp32 host, normed hidden x mup [1, 1, R, H] device)."""
        R = len(ids)
        lg, h = self._body(
            self._dev(self._ids_host(ids)),
            self._dev(self._pos_host(p0, R)),
            self._dev(self._idx_host(p0, R)),
            R,
            self._dev(self._tau_host(p0, R)) if self.long else None,
        )
        self.last_traced = False
        return self._read(lg, R), h

    def _commit_eager(self, n):
        for layer in self.model.layers:
            layer.commit_rows(n)

    def _capture_verify(self, R):
        di, dp, dk, dt = self.bufs[R]
        tid = ttnn.begin_trace_capture(self.mesh, cq_id=0)
        out = self._body(di, dp, dk, R, dt)
        ttnn.end_trace_capture(self.mesh, tid, cq_id=0)
        self.tid[R], self.out[R] = tid, out
        self.keep[R] = [c._last_seq for c in self.sconvs]  # trace-owned conv history the commit traces read
        for n in range(1, R + 1):
            t = ttnn.begin_trace_capture(self.mesh, cq_id=0)
            self._commit_eager(n)
            ttnn.end_trace_capture(self.mesh, t, cq_id=0)
            self.ctid[(R, n)] = t

    def _verify(self, ids, p0):
        """Verify rows: traced when this width was captured, else eager. -> (logits [R, V] host, hidden x mup)."""
        R = len(ids)
        self._check_backed(int(p0), R)
        if R not in self.tid:
            return self._rows_eager(ids, p0)
        di, dp, dk, dt = self.bufs[R]
        ttnn.copy_host_to_device_tensor(self._idx_host(p0, R), dk)
        ttnn.copy_host_to_device_tensor(self._ids_host(ids), di)
        ttnn.copy_host_to_device_tensor(self._pos_host(p0, R), dp)
        if dt is not None:
            ttnn.copy_host_to_device_tensor(self._tau_host(p0, R), dt)
        ttnn.execute_trace(self.mesh, self.tid[R], cq_id=0, blocking=False)
        self.last_R, self.last_traced = R, True
        lg, hm = self.out[R]
        return self._read(lg, R), hm

    def _commit(self, n):
        if self.last_traced:
            ttnn.execute_trace(self.mesh, self.ctid[(self.last_R, n)], cq_id=0, blocking=False)
        else:
            self._commit_eager(n)

    # ------------------------------------------------------------------ drafter
    def _pick(self, lg, smp, qs, ctx=()):
        """A draft from a drafter logits row; ctx: the drafts before it in this chain (penalties count them)."""
        d, q = smp.pick(self._read(lg, 1)[0], ctx)
        if q is not None:
            qs.append(q)
        return d

    def _draft_eager(self, hrows, toks, p0, k, smp):
        mtp, model = self.mtp, self.model
        R = len(toks)
        dp = mtp.depths[0]
        e = model.embd(self._dev(self._ids_host(toks)))
        e = mtp._gather(ttnn.reshape(e, (1, 1, R, e.shape[-1])))
        out = dp.block.forward_rows(
            mtp.inproj(dp, hrows, e), self._dev(self._pos_host(p0, R)), self.pt, commit_all=True
        )
        full = mtp._gather(out)
        prev = ttnn.slice(full, (0, 0, R - 1, 0), (1, 1, R, full.shape[-1]))
        drafts, qs = [], []
        d = self._pick(model._lm_head(ttnn.multiply(prev, mtp.inv_mup)), smp, qs, drafts)
        drafts.append(d)
        for j in range(1, k):
            lg, prev = mtp.step(
                j, prev, self._dev(self._ids_host([d])), self._dev(self._pos_host(p0 + R - 1 + j, 1)), self.pt
            )
            d = self._pick(lg, smp, qs, drafts)
            drafts.append(d)
        return drafts, (None if smp.greedy else qs)

    def _d_depth0(self, R_src, w):
        mtp, model = self.mtp, self.model
        dp = mtp.depths[0]
        h = self.out[R_src][1]
        hrows = ttnn.slice(h, (0, 0, 0, 0), (1, 1, w, h.shape[-1]))
        e = model.embd(self.d_ids[w])
        e = mtp._gather(ttnn.reshape(e, (1, 1, w, e.shape[-1])))
        out = dp.block.forward_rows(mtp.inproj(dp, hrows, e), self.d_pos[w], self.pt, commit_all=True)
        full = mtp._gather(out)
        last = ttnn.slice(full, (0, 0, w - 1, 0), (1, 1, w, full.shape[-1]))
        lg = self._rm_logits(model._lm_head(ttnn.multiply(last, mtp.inv_mup)))
        ttnn.copy(ttnn.typecast(last, ttnn.bfloat16) if last.dtype != ttnn.bfloat16 else last, self.prev)
        return lg

    def _d_depth(self, j):
        lg, full = self.mtp.step(j, self.prev, self.d_ids[1], self.d_pos[1], self.pt)
        ttnn.copy(ttnn.typecast(full, ttnn.bfloat16) if full.dtype != ttnn.bfloat16 else full, self.prev)
        return self._rm_logits(lg)

    def _argmax_dev(self, lg_rm):
        """Greedy pick on the device: the real-vocab argmax of a row-major [1, 1, 1, Vp] logits row -> [1, 1] uint32."""
        v = ttnn.slice(lg_rm, (0, 0, 0, 0), (1, 1, 1, self.V))
        return ttnn.reshape(ttnn.argmax(v, dim=-1), (1, 1))

    def _d_chain(self, R_src, w):
        """Greedy drafts without host round trips: depth 0, then each deeper depth fed the previous depth's device
        argmax; -> the k draft ids [1, k] uint32 (one readback per step instead of one per depth)."""
        toks = [self._argmax_dev(self._d_depth0(R_src, w))]
        prev = self.prev
        for j in range(1, self.k):
            lg, full = self.mtp.step(j, prev, toks[-1], self.c_pos[j], self.pt)
            prev = ttnn.typecast(full, ttnn.bfloat16) if full.dtype != ttnn.bfloat16 else full
            toks.append(self._argmax_dev(self._rm_logits(lg)))
        return ttnn.concat(toks, dim=-1) if len(toks) > 1 else toks[0]

    def _d_capture(self, f):
        f()  # eager first: compiles every program (a capture cannot)
        tid = ttnn.begin_trace_capture(self.mesh, cq_id=0)
        out = f()
        ttnn.end_trace_capture(self.mesh, tid, cq_id=0)
        return tid, out

    def _draft(self, R_src, toks, p0, k, smp):
        """k drafts after the accepted rows toks (the tokens following verify rows 0 .. w - 1 of the last pass)."""
        w = len(toks)
        if smp.greedy and self.last_traced and (R_src, w) in self.tc:
            ttnn.copy_host_to_device_tensor(self._ids_host(toks), self.d_ids[w])
            ttnn.copy_host_to_device_tensor(self._pos_host(p0, w), self.d_pos[w])
            for j in range(1, self.k):
                ttnn.copy_host_to_device_tensor(self._pos_host(p0 + w - 1 + j, 1), self.c_pos[j])
            tid, ids = self.tc[(R_src, w)]
            ttnn.execute_trace(self.mesh, tid, cq_id=0, blocking=False)
            got = ttnn.to_torch(ttnn.get_device_tensors(ids)[0]).reshape(-1)[: self.k].tolist()
            return [int(v) for v in got][:k], None
        if not self.t0 or not self.last_traced or (R_src, w) not in self.t0:
            hrows = ttnn.slice(self._last_h, (0, 0, 0, 0), (1, 1, w, self._last_h.shape[-1]))
            return self._draft_eager(hrows, toks, p0, k, smp)
        ttnn.copy_host_to_device_tensor(self._ids_host(toks), self.d_ids[w])
        ttnn.copy_host_to_device_tensor(self._pos_host(p0, w), self.d_pos[w])
        tid, lg = self.t0[(R_src, w)]
        ttnn.execute_trace(self.mesh, tid, cq_id=0, blocking=False)
        drafts, qs = [], []
        d = self._pick(lg, smp, qs, drafts)
        drafts.append(d)
        for j in range(1, k):
            ttnn.copy_host_to_device_tensor(self._ids_host([d]), self.d_ids[1])
            ttnn.copy_host_to_device_tensor(self._pos_host(p0 + w - 1 + j, 1), self.d_pos[1])
            tid, lg = self.tj[j]
            ttnn.execute_trace(self.mesh, tid, cq_id=0, blocking=False)
            d = self._pick(lg, smp, qs, drafts)
            drafts.append(d)
        return drafts, (None if smp.greedy else qs)

    # ------------------------------------------------------------------ warm-up / capture
    def _mtp_prompt_hook(self, ids, start=0, snap=None, mm=None):
        """on_chunk for prefill_tokens: drafter depth 0 over the chunk's rows t (the target's normed hidden of t, the
        token at t + 1), committed past the real rows; ids = every prompt token (the one after the prefilled part too),
        start = the prefill's first position (the chunk offsets are relative to it); snap = the prefix-snapshot position.
        """
        from models.tt_transformers.tt.common import Mode

        mtp, model = self.mtp, self.model
        dp = mtp.depths[0]

        def hook(c0, n, Tp, x):
            c0 = start + c0
            with mtp.own_ccl():
                h = ttnn.multiply(model.norm(x, mode=Mode.PREFILL), self.mup)
                if h.is_sharded():
                    h = ttnn.sharded_to_interleaved(h, ttnn.DRAM_MEMORY_CONFIG)
                nxt = torch.zeros(Tp, dtype=torch.int32)
                tail = torch.as_tensor(ids[c0 + 1 : c0 + 1 + n]).to(torch.int32)
                nxt[: tail.numel()] = tail
                e = model.embd(self._dev(self._ids_host(nxt.tolist())))
                e = ttnn.reshape(e, (1, 1, Tp, e.shape[-1]))
                if mm is not None:  # the drafter's input embedding of an image / audio position is its tower row
                    e = mm_rows.merge(e, mm, c0 + 1, n, Tp, self.mesh, int(model.args.dim))
                e = mtp._gather(e)
                x_in = ttnn.concat([mtp._norm(h, dp.hn), mtp._norm(e, dp.en)], dim=-1)
                for t in (h, e):
                    ttnn.deallocate(t)
                xm = ttnn.linear(x_in, dp.w_in)
                ttnn.deallocate(x_in)
                out = dp.block.forward_prefill(xm, c0, self.pt_host, valid=(n if n < Tp else None))
                for t in (xm, out):
                    ttnn.deallocate(t)
            if snap is not None and c0 + n == snap:
                self._px_save(len(self._px_convs) - 1)

        return hook

    def _prefill(self, ids, start=0, snap=None, mm=None):
        """The target's prefill of ids[start:-1] (+ the drafter's prompt pass); the states must hold ids[:start].
        snap: a position (a chunk boundary past start) at which every layer's conv states are copied (prefix cache)."""
        hyb = _hyb()
        if hyb is not None and hyb.enabled:
            hyb.stream.begin_prefill(
                len(ids) - 1 - start
            )  # the streaming threshold / prefetch banking by prompt length
            if hyb.hostk:
                # prefill: the host kernel's whole pool (compute-bound), or INKLING_PREFILL_HOSTK_THREADS of it
                hyb.hostk_threads = int(os.environ.get("INKLING_PREFILL_HOSTK_THREADS", "0"))
        hook = self._mtp_prompt_hook(ids, start, snap, mm) if self.mtp_prompt else None
        on_layer = None
        if snap is not None:
            on_layer = lambda li, end, part: self._px_save(li, part) if end == snap else None  # noqa: E731
        try:
            return self.model.prefill_tokens(  # the last prompt row's logits [V] (tools/prefill_bench.py PF_DUMP)
                torch.as_tensor(ids[start:-1]),
                self.pt_host,
                start=start,
                buckets=self.buckets,
                on_chunk=hook,
                on_layer=on_layer,
                mm=mm,
            )
        finally:
            if self.captured and self.guard.purge():
                self.stats["cache_purges"] = self.stats.get("cache_purges", 0) + 1
            if hyb is not None and hyb.enabled and hyb.hostk:
                # decode: INKLING_DECODE_HOSTK_THREADS (default 6; bandwidth-bound, the rest leave DRAM bandwidth to
                # the uploads)
                hyb.hostk_threads = int(os.environ.get("INKLING_DECODE_HOSTK_THREADS", "6"))

    def enable_vllm_paging(self, scratch_base, scratch_blocks, context):
        """vLLM-paged serving: logical blocks map to the request's vLLM block ids (set_page_row); a logical block vLLM
        has not allocated (id 0) maps to one of the engine's scratch blocks [scratch_base, + scratch_blocks) (the caches
        hold them: INKLING_KV_EXTRA_BLOCKS). context: vLLM's max_model_len (positions past it are never read)."""
        if not self.px_on:
            raise RuntimeError("vLLM prefix caching needs the prefix snapshot buffers (INKLING_PREFIX_CACHE=1)")
        self.paged = True
        self._scratch_base, self._scratch_n, self._context = int(scratch_base), int(scratch_blocks), int(context)
        self._row_key = None
        self._backed = self.pt_host.numel()  # logical blocks [0, _backed) are vLLM-allocated

    def set_page_row(self, row):
        """The request's vLLM block table row (logical block -> vLLM block id, 0 = not allocated) -> pt_host / pt. A
        change is written into the persistent device table the decode traces read."""
        if not self.paged or row is None:
            return
        r = torch.as_tensor(row).reshape(-1).to(torch.int64)
        key = r.numpy().tobytes()
        if key == self._row_key:
            return
        self._row_key = key
        n = self.pt_host.numel()
        m = torch.zeros(n, dtype=torch.int64)
        k = min(n, r.numel())
        m[:k] = r[:k]
        unalloc = m <= 0
        self._backed = int(unalloc.nonzero()[0]) if bool(unalloc.any()) else n
        m = torch.where(unalloc, self._scratch_base + torch.arange(n) % self._scratch_n, m).to(torch.int32)
        if not torch.equal(m, self.pt_host):
            self.pt_host = m
            ttnn.copy_host_to_device_tensor(
                ttnn.from_torch(m.reshape(1, -1), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=self.rep),
                self.pt,
            )

    def _check_backed(self, pos, n):
        """Positions [pos, pos + n) about to be written: a write past vLLM's allocation below its context lands in a
        scratch block, and the KV it holds is lost (vLLM's lookahead covers the engine's lead: counted, never expected)."""
        if not self.paged:
            return
        last = min(pos + n, self._context) - 1
        if last >= 0 and last // self.block >= self._backed:
            self.stats["kv_unbacked"] = self.stats.get("kv_unbacked", 0) + 1
            if self.stats["kv_unbacked"] == 1:
                logger.warning(
                    f"Inkling: KV write at positions {pos}..{last} past vLLM's allocated blocks ({self._backed}): raise "
                    "INKLING_KV_LOOKAHEAD"
                )

    def _ps_save(self, slot, pos):
        """The prefix-state snapshot of the prefill that just ended (its hooks copied every block's conv states at
        `pos` into the staging buffers; each sliding ring still holds the window before pos, pos being the last chunk
        boundary before the prompt's end) -> host, as prefix-state slot `slot`."""
        self._ps[int(slot)] = {
            "pos": int(pos),
            "convs": [[ttnn.from_device(buf) for _, buf in blk] for blk in self._px_convs],
            "rings": [ttnn.from_device(t) for t in self._px_rings],
        }
        self.stats["prefix_saves"] = self.stats.get("prefix_saves", 0) + 1

    def _ps_restore(self, slot, pos):
        """Prefix-state slot `slot` (a snapshot at `pos`) -> the conv states and sliding rings (after _reset())."""
        rec = self._ps.get(int(slot))
        if rec is None or rec["pos"] != int(pos):
            raise RuntimeError(
                f"Inkling: prefix-state slot {slot} holds {None if rec is None else rec['pos']}, restore asked {pos}"
            )
        for blk, hs in zip(self._px_convs, rec["convs"]):
            for (c, buf), h in zip(blk, hs):
                ttnn.copy_host_to_device_tensor(h, buf)
                ttnn.copy(buf, c.st)
        for h, t in zip(rec["rings"], self._px_rings):
            ttnn.copy_host_to_device_tensor(h, t)

    def _px_save(self, i, part=None):
        """Block i's conv states -> its snapshot buffers (block i = target layer i; the last = drafter depth 0).
        part: "attn" (k, v, attn convs), "mlp" (the MLP conv) or None (all four)."""
        self._px_count = getattr(self, "_px_count", 0) + 1
        convs = self._px_convs[i]
        for c, buf in convs[:3] if part == "attn" else convs[3:] if part == "mlp" else convs:
            ttnn.copy(c.st, buf)

    def _px_restore(self, px):
        for blk in self._px_convs:
            for c, buf in blk:
                ttnn.copy(buf, c.st)
        for h, t in zip(px["rings"], self._px_rings):
            ttnn.copy_host_to_device_tensor(h, t)

    @staticmethod
    def _mm_key(mm):
        """The prompt's media items as (first placeholder position, identifier): a prefix with images / audio matches
        only the same items at the same places (their placeholder token ids are the same for any content)."""
        return [] if mm is None else list(mm["keys"])

    def _px_match(self, prompt_ids, mmkey=()):
        """The prefix-cache entry whose first b tokens (and media items before b) begin this prompt (b < len(prompt)),
        else None."""
        px = self._px
        if px is None or px["b"] > len(prompt_ids) - 1:
            return None
        b = px["b"]
        a = np.asarray(px["ids"][:b], dtype=np.int64)
        if px.get("mm", []) != [k for k in mmkey if k[0] < b]:
            return None
        return px if np.array_equal(a, np.asarray(prompt_ids[:b], dtype=np.int64)) else None

    def _reset(self):
        self.model.reset_sequence_state()
        if self.mtp is not None:
            self.mtp.reset()

    def warm(self, prompt_len=None):
        """Every program a request can run, eagerly, before any capture: each prefill bucket (a warm-up prompt of
        `prompt_len` tokens covers the full chunks too), the verify rows at every width 1..8, the drafter at every
        accepted-row width and depth. Then the sequence state is reset."""
        t0 = time.time()
        self._px = None
        hyb = _hyb()
        if hyb is not None and hyb.enabled:
            for li in list(hyb.lib):  # the hybrid layers' host library RAM copies (loading since the build)
                hyb.lib_layer(li)
            if hyb.fetch.dir:  # HYB_DEV_EXPERTS: the pinned device-format copies for prefill streaming / fetch
                hyb.fetch.finalize(hyb.loader, hyb.stream)
        ids = torch.arange(1000, 1000 + max(self.buckets) + 64, dtype=torch.int64) % 200000
        for b in self.buckets:  # one prefill per bucket size (a b-row prompt pads to bucket b)
            self._reset()
            self._prefill(ids[: b + 1].tolist())
            self._reset()
            self._prefill(ids[: b - 37].tolist())  # a padded last chunk (its valid-row commits)
        self.model.warm_last_row_slices(self.buckets)  # the last real row's logits slice at every offset
        if hyb is not None and hyb.enabled and getattr(hyb.stream, "on", False) and hyb.stream.NS > 0:
            # prefill streaming: its programs (the merged resident + staging expert call, the staging rounds, the
            # prefetch banks) run only for long prompts; compile them now, for every tail bucket, before any capture
            # (a program first compiled after the captures can have its binaries overwritten by the replays: see
            # _new_programs). Spread token ids: many missed experts, so every streaming path runs.
            # Every miss streams during these (no CPU split, a 1-token threshold), so the merged call and the rounds
            # compile whatever the routing of the warm-up text; short prompts use the single staging, long ones the
            # prefetch banks (HYB_PF_PREFETCH_MIN): both layouts.
            st = hyb.stream
            saved = (hyb.pf_split, st.min_short, st.min_long)
            hyb.pf_split, st.min_short, st.min_long = False, 1, 1
            try:
                for n in [b for b in self.buckets] + [2 * self.chunk + b for b in self.buckets]:
                    self._reset()
                    self._prefill((torch.arange(n + 1) * 7919 % 150000 + 1000).tolist())
            finally:
                hyb.pf_split, st.min_short, st.min_long = saved
            # and nothing streamed (every miss on the CPU): a layer step without a staging round runs the resident-only
            # call, whose programs (per bucket and per layer dtype) the all-streamed passes above never compile
            saved = (st.min_short, st.min_long)
            st.min_short = st.min_long = 1 << 30
            try:
                for n in [b for b in self.buckets] + [2 * self.chunk + b for b in self.buckets]:
                    self._reset()
                    self._prefill((torch.arange(n + 1) * 104729 % 150000 + 1000).tolist())
            finally:
                st.min_short, st.min_long = saved
            from hybrid.patch import warm_host_rows_dense

            for dt in (ttnn.bfloat16, ttnn.float32):
                warm_host_rows_dense(self.mesh, self.buckets, int(self.args.dim), dt)
            logger.info(
                f"Inkling warm-up: streamed prefills of {self.buckets} and {2 * self.chunk} + {self.buckets} " "tokens"
            )
        p = self.buckets[-1]
        smp = InklingSampler(0.0)
        # the verify widths only (1 .. k + 1): every prompt token goes through prefill, and wider eager rows would take
        # the unfused path (its FP32-accumulation matmuls do not fit the L1 beside the socket FIFOs)
        for R in range(1, self.k + 2):
            _, h = self._rows_eager([7] * R, p)
            if self.mtp is not None and R <= self.k + 1:
                with self.mtp.own_ccl():
                    self._draft_eager(ttnn.slice(h, (0, 0, 0, 0), (1, 1, R, h.shape[-1])), [7] * R, p, self.k, smp)
            for n in range(1, R + 1):  # the commit programs (state restored by the reset below)
                self._commit_eager(n)
        self._reset()
        self.log_memory("after the warm-up")
        logger.info(
            f"Inkling engine warm-up: {time.time() - t0:.1f}s (prefill buckets {self.buckets}, "
            f"rows 1..{self.k + 1}, drafter k={self.k})"
        )

    def capture(self):
        """Every verify width 1 .. k + 1 with its commits, then the drafter (depth 0 per (width, accepted rows), each
        deeper depth). The eager passes here advance the states: reset after."""
        if not self.trace or self.captured:
            return
        t0 = time.time()
        self._px = None  # the eager passes below write KV rows of the global caches
        p = self.buckets[-1]
        for R in range(1, self.k + 2):
            if R not in self.bufs:  # (kept across a re-capture)
                self.bufs[R] = (
                    self._dev(self._ids_host([0] * R)),
                    self._dev(self._pos_host(0, R)),
                    self._dev(self._idx_host(0, R)),
                    self._dev(self._tau_host(0, R)) if self.long else None,
                )
        if self.mtp is not None and self.prev is None:
            H = int(self.args.dim)
            self.prev = ttnn.from_torch(
                torch.zeros(1, 1, 1, H),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh,
                mesh_mapper=self.rep,
            )
            self.d_ids = {w: self._dev(self._ids_host([0] * w)) for w in range(1, self.k + 2)}
            self.d_pos = {w: self._dev(self._pos_host(0, w)) for w in range(1, self.k + 2)}
            self.c_pos = {j: self._dev(self._pos_host(0, 1)) for j in range(1, self.k)}  # (before any capture)
        for R in range(1, self.k + 2):
            di, dp, dk, dt = self.bufs[R]
            ttnn.copy_host_to_device_tensor(self._idx_host(p, R), dk)
            ttnn.copy_host_to_device_tensor(self._pos_host(p, R), dp)
            self._body(di, dp, dk, R, dt)  # eager once more right before the capture (programs cached, buffers warm)
            self._capture_verify(R)
        if self.mtp is not None and self.chain:
            with self.mtp.own_ccl():
                self.tc[(1, 1)] = self._d_capture(lambda: self._d_chain(1, 1))
                for kk in range(1, self.k + 1):
                    for w in range(1, kk + 2):
                        self.tc[(kk + 1, w)] = self._d_capture(lambda kk=kk, w=w: self._d_chain(kk + 1, w))
        if self.mtp is not None:
            with self.mtp.own_ccl():
                self.t0[(1, 1)] = self._d_capture(lambda: self._d_depth0(1, 1))
                for kk in range(1, self.k + 1):
                    for w in range(1, kk + 2):
                        self.t0[(kk + 1, w)] = self._d_capture(lambda kk=kk, w=w: self._d_depth0(kk + 1, w))
                for j in range(1, self.k):
                    self.tj[j] = self._d_capture(lambda j=j: self._d_depth(j))
        self.captured = True
        self.guard.snapshot()
        self._prog_n = self._programs()
        self._reset()
        self.log_memory("after the captures")
        logger.info(
            f"Inkling engine captures: {time.time() - t0:.1f}s (verify widths 1..{self.k + 1}, "
            f"{len(self.ctid)} commits, {len(self.t0) + len(self.tj)} drafter traces)"
        )

    def _programs(self):
        try:
            return int(self.mesh.num_program_cache_entries())
        except Exception:  # noqa: BLE001 - diagnostics only
            return None

    def _release_traces(self):
        """Every trace released (their outputs dropped); the persistent inputs stay."""
        ttnn.synchronize_device(self.mesh)
        for t in list(self.tid.values()) + list(self.ctid.values()):
            ttnn.release_trace(self.mesh, t)
        for d in (self.t0, self.tj, self.tc):
            for tid, _ in d.values():
                ttnn.release_trace(self.mesh, tid)
        self.tid, self.out, self.keep, self.ctid = {}, {}, {}, {}
        self.t0, self.tj, self.tc = {}, {}, {}
        self.captured = self.last_traced = False
        self.last_R = None

    def _new_programs(self):
        """True when a program was compiled since the captures. Its kernel binaries were then allocated in DRAM while
        the traces existed, possibly inside a trace's freed intermediates, which every replay overwrites: the next
        launch of that program runs corrupted kernels (the streamed-prefill hang, 2026-11-15: the streaming path's
        programs first compiled by a long prompt after the captures). warm() compiles every prefill program; this is
        the safety net."""
        n = self._programs()
        return self.captured and n is not None and self._prog_n is not None and n != self._prog_n

    def log_memory(self, where):
        """DRAM and trace-region use per bank (die 0's view; every die holds the same layout)."""
        try:
            d = ttnn.get_memory_view(self.mesh, ttnn.BufferType.DRAM)
            t = ttnn.get_memory_view(self.mesh, ttnn.BufferType.TRACE)
            logger.info(
                f"Inkling memory {where}: DRAM free {d.total_bytes_free_per_bank * d.num_banks / 2**30:.2f} GiB "
                f"(largest block {d.largest_contiguous_bytes_free_per_bank / 2**20:.0f} MiB per bank), trace "
                f"region used {t.total_bytes_allocated_per_bank * t.num_banks / 2**20:.0f} of "
                f"{t.total_bytes_per_bank * t.num_banks / 2**20:.0f} MiB"
            )
        except Exception as e:  # noqa: BLE001 - diagnostics only
            logger.warning(f"memory view unavailable: {e!r}")

    # ------------------------------------------------------------------ serving
    def allocate_kv(self, kv_shape, dtype):
        """The adapter hook: the engine's caches are its own (see the module doc)."""

    def warm_prefill_path(self):
        """The adapter hook: warm() covers the prefill programs."""

    def _first(self, last_tok, p, smp):
        """The first verify pass of a sequence (the last prompt token, no drafts) and the drafts after it."""
        L, h = self._verify([last_tok], p)
        self._last_h = h
        self.last_logits = L[0]
        nxt, _ = smp.pick(L[0])
        smp.observe([nxt])
        self._commit(1)
        self._drafts, self._qs = [], None
        if self.mtp is not None and nxt not in self.eos:
            with self.mtp.own_ccl():
                self._drafts, self._qs = self._draft(1, [nxt], p, self.k, smp)
        self._tok, self._pos = nxt, p + 1
        return nxt

    def _arm(self):
        if self._memdump is not None:
            self._memdump.memdump_tick()
        if self.watchdog > 0:
            import faulthandler
            import sys

            faulthandler.dump_traceback_later(self.watchdog, repeat=False, file=sys.stderr, exit=False)

    def _disarm(self):
        if self.watchdog > 0:
            import faulthandler

            faulthandler.cancel_dump_traceback_later()

    def prefill(self, prompt_ids, sampling=None, mm=None, start=0, page_row=None, restore=None, saves=()) -> int:
        """A new sequence: prefill the prompt (all but its last token; that one is the first verify row) and return
        the first generated token. sampling = (temperature, top_k, top_p, seed) or None (greedy). mm: the prompt's
        image / audio rows (tt/mm.py build(); their placeholder positions take them instead of token embeddings)."""
        self._arm()
        try:
            hyb = _hyb()
            if hyb is not None and hyb.enabled:  # the expert placement as at startup: output independent of earlier requests
                hyb.reset_placement()
            return self._prefill_request(prompt_ids, sampling, mm, start, page_row, restore, saves)
        finally:
            self._disarm()

    def _prefill_request(self, prompt_ids, sampling, mm=None, start=0, page_row=None, restore=None, saves=()):
        """vLLM-paged (enable_vllm_paging): start = vLLM's cached prefix length, restore = (slot, start) the
        prefix-state snapshot to resume from, saves = [(pos, slot)] the snapshots to take (the plugin's prefix_state
        directives; Inkling's policy asks one per prompt, at its last chunk boundary before the last token)."""
        t0 = time.time()
        P = len(prompt_ids)
        if P < 1:
            raise ValueError("empty prompt")
        if P + self.k + 1 > self.max_seq_len:
            raise ValueError(f"prompt of {P} tokens + the verify window exceeds the context ({self.max_seq_len})")
        self._log_request()
        self._smp = self._sampler(sampling, prompt_ids, self.V)
        self.adopt_next = False
        self._pending = []
        self._reset()
        if mm is not None and int(mm["pos"].max()) >= P - 1:
            raise ValueError("an image / audio placeholder as the prompt's last token (the first verify row)")
        mmkey = self._mm_key(mm)
        if self.paged:
            self.set_page_row(page_row)
            px, self._px = None, None
            saves = [(int(q), int(sl)) for q, sl in (saves or ())]
            if len(saves) > 1:
                raise RuntimeError(f"Inkling: {len(saves)} prefix-state saves in one prefill (the policy asks one)")
            snap = saves[0][0] if saves else 0
            b0 = int(start or 0)
            if snap and (snap % self.chunk or snap <= b0 or snap > P - 1 or (snap - b0) % self.chunk):
                raise RuntimeError(f"Inkling: prefix-state save at {snap} is not a chunk end of this prefill ({b0}, {P})")
            if restore is not None:
                if int(restore[1]) != b0:
                    raise RuntimeError(f"Inkling: restore at {restore[1]} for a prefill from {b0}")
                if b0:
                    self._ps_restore(restore[0], b0)
                    self.stats["prefix_hits"] = self.stats.get("prefix_hits", 0) + 1
                    self.stats["prefix_tokens"] = self.stats.get("prefix_tokens", 0) + b0
            elif b0:
                raise RuntimeError(f"Inkling: prefill from {b0} without a prefix-state restore")
        else:
            px, self._px = self._px_match(prompt_ids, mmkey) if P > 1 else None, None  # (invalid until this ends)
            b0 = px["b"] if px is not None else 0
            snap = ((P - 2) // self.chunk) * self.chunk if self.px_on and P > 1 else 0
            snap = snap if snap >= self.px_min else 0
            if b0:
                self._px_restore(px)
                self.stats["prefix_hits"] = self.stats.get("prefix_hits", 0) + 1
                self.stats["prefix_tokens"] = self.stats.get("prefix_tokens", 0) + b0
        self._px_count = 0
        if P - 1 > b0:
            self._prefill(list(prompt_ids), start=b0, snap=(snap if snap > b0 else None), mm=mm)
            if self._new_programs() and self.recaptures < int(os.environ.get("INKLING_MAX_RECAPTURES", "4")):
                # programs compiled after the captures (see _new_programs): fresh traces whose intermediates avoid the
                # new binaries, then the prompt again (the captures' eager passes reset the sequence state)
                self.recaptures += 1
                logger.warning(
                    f"Inkling: {self._programs() - self._prog_n} programs compiled after the trace captures "
                    f"(prompt of {P} tokens): re-capturing the traces and re-running the prefill "
                    f"({self.recaptures})"
                )
                self._release_traces()
                self.capture()
                self._reset()
                b0 = 0
                self._px_count = 0
                self._prefill(list(prompt_ids), snap=(snap or None), mm=mm)
        if self.paged:
            if snap:
                want = 2 * len(self.model.layers) + (1 if self.mtp_prompt else 0)
                if self._px_count != want:
                    raise RuntimeError(f"Inkling: prefix-state snapshot at {snap} took {self._px_count} of {want} copies")
                self._ps_save(saves[0][1], snap)
        elif snap and snap > b0:
            self._px = {
                "ids": list(prompt_ids[:snap]),
                "mm": [k for k in mmkey if k[0] < snap],
                "b": snap,
                "rings": [ttnn.from_device(t) for t in self._px_rings],
            }
        elif snap and px is not None:
            self._px = px  # the same boundary: the restored snapshot still holds
        if b0 or self._px is not None or (self.paged and snap):
            at = snap if self.paged else (self._px["b"] if self._px else None)
            logger.info(f"Inkling prefix cache: prompt of {P} tokens, reused {b0}, snapshot at {at or '-'}")
        nxt = self._first(int(prompt_ids[-1]), P - 1, self._smp)
        if self.sfx_on:
            from hybrid.suffix_draft import Chooser, SuffixIndex

            self._sfx, self._choose = SuffixIndex(list(prompt_ids) + [nxt]), Chooser()
        self.stats["t_prefill"] += time.time() - t0
        self.stats["prefills"] += 1
        return nxt

    def _log_request(self):
        """The previous request's speculative-decode summary (logged when the next one starts)."""
        st, prev = self.stats, getattr(self, "_rq0", None)
        keys = ("steps", "emitted", "sfx_steps", "sfx_acc", "t_verify", "t_draft")
        now = {k: st.get(k, 0) for k in keys}
        self._rq0 = now
        if prev is None or now["steps"] == prev["steps"]:
            return
        d = {k: now[k] - prev[k] for k in keys}
        logger.info(
            f"Inkling request: {d['emitted'] / d['steps']:.3f} tokens/step over {d['steps']} steps, suffix drafts "
            f"{d['sfx_steps']} steps ({d['sfx_acc'] / max(1, d['sfx_steps']):.2f} accepted), verify "
            f"{1e3 * d['t_verify'] / d['steps']:.1f} + draft {1e3 * d['t_draft'] / d['steps']:.1f} ms per step"
        )

    @staticmethod
    def _sampler(sampling, prompt_ids, V):
        """sampling = (temperature, top_k, top_p, seed[, presence, frequency, repetition]) or None (greedy)."""
        if sampling is None:
            return InklingSampler(0.0)
        t, k, p, seed, *pen = sampling
        pres, freq, rep = (list(pen) + [0.0, 0.0, 1.0][len(pen) :])[:3]
        return InklingSampler(
            float(t),
            int(k or 0),
            float(p if p is not None else 1.0),
            seed=(int(seed) if seed is not None else None),
            presence=pres,
            frequency=freq,
            repetition=rep,
            prompt_ids=prompt_ids,
            vocab=V,
        )

    def step(
        self, tok: int = None, pos: int = None, page_table_row=None, host_authoritative=False, sampling=None
    ) -> list[int]:
        """-> the next tokens for vLLM (1 .. k + 1 ids: the accepted drafts and the verified next token of a verify
        pass, staged by _emit). The engine owns the continuation (on the plugin's async path its host token can lag a
        step); the plugin's tok / pos are checked when it flags them authoritative."""
        if self._pos is None:
            raise RuntimeError("Inkling step before any prefill")
        self.set_page_row(page_table_row)
        self._arm()
        try:
            return self._step(tok, pos, host_authoritative)
        finally:
            self._disarm()

    def _step(self, tok, pos, host_authoritative):
        buf, self._pending = self._pending, []
        adopt, self.adopt_next = self.adopt_next, False
        if (
            not buf
            and (host_authoritative or adopt)
            and tok is not None
            and (int(tok), int(pos)) != (self._tok, self._pos)
        ):
            if int(pos) != self._pos:
                raise RuntimeError(
                    f"Inkling step input (tok {tok}, pos {pos}) differs from the engine's continuation "
                    f"(tok {self._tok}, pos {self._pos})"
                )
            # the plugin sampled the token itself (host sampling after a logits prefill): adopt it, drop the drafts
            self._tok, self._drafts, self._qs = int(tok), [], None
            self.stats["reanchors"] = self.stats.get("reanchors", 0) + 1
        while True:
            if buf:
                out = self._emit(buf)
                if out is not None:
                    return out
            buf = buf + self._device_step()

    def step_logits(self, tok: int, pos: int, page_table_row=None) -> torch.Tensor:
        """Host-sampled decode (the plugin samples: logprobs, min_p, logit_bias, bad words, structured outputs ...):
        one verify row [tok] at pos, no drafts; -> the logits row [V] of the token after it. The plugin's token is
        authoritative (its sampler, not the engine's, picked it), so the continuation is re-anchored on it."""
        if self._pos is None:
            raise RuntimeError("Inkling step before any prefill")
        self.set_page_row(page_table_row)
        if int(pos) != self._pos:
            raise RuntimeError(
                f"Inkling host-sampled step at position {pos}; the engine's continuation is at {self._pos}"
            )
        self._pending, self._drafts, self._qs = [], [], None
        self._sfx_at(int(pos), int(tok))
        a = time.time()
        L, h = self._verify([int(tok)], int(pos))
        self._last_h = h
        self._commit(1)
        self._tok, self._pos = (
            int(L[0].argmax()),
            int(pos) + 1,
        )  # (placeholder token: the next call brings the real one)
        self.adopt_next = True
        st = self.stats
        st["host_steps"] = st.get("host_steps", 0) + 1
        st["t_verify"] += time.time() - a
        return L[0]

    def _sfx_at(self, pos, tok):
        """The suffix index holds the context through `tok` at `pos` (the token the next verify pass starts from)."""
        if self._sfx is None:
            return
        if self._sfx.n < pos:  # (a gap: positions the engine never saw; not expected) no more lookups this request
            self._sfx = None
            return
        self._sfx.n = pos
        self._sfx.extend([tok])

    def _device_step(self) -> list[int]:
        """One verify pass (+ commit, + the next drafts): -> the new tokens (ends at the end of turn)."""
        x, p, drafts, qs, smp = self._tok, self._pos, self._drafts, self._qs, self._smp
        if p + len(drafts) + 1 > self.max_seq_len:  # the context's last positions: no drafts past it
            drafts = drafts[: max(0, self.max_seq_len - p - 1)]
            qs = qs[: len(drafts)] if qs is not None else None
        # prompt-lookup drafts: picked from the match length and the request's tally before either draft is compared
        self._sfx_at(p, x)
        sfx_n, sfx, dfl, use_sfx = 0, None, drafts, False
        if self._sfx is not None and drafts:
            sfx_n, cont = self._sfx.lookup(len(drafts))
            if sfx_n:
                sfx = cont + drafts[len(cont) :]  # a continuation shorter than the drafts: the drafter's after it
                use_sfx = self._choose.pick(sfx_n)
                if use_sfx:
                    drafts = sfx
                    if qs is not None:  # the suffix rows are point-mass proposals
                        qs = [torch.nn.functional.one_hot(torch.tensor(d), qs[0].numel()).float() for d in cont] + list(
                            qs[len(cont) :]
                        )
        rows = [x] + drafts
        a = time.time()
        L, h = self._verify(rows, p)
        self._last_h = h
        n, nxt = smp.verify(L, drafts, qs)
        self._commit(n + 1)
        b = time.time()
        new = drafts[:n] + [nxt]
        hit = next((i for i, t in enumerate(new) if t in self.eos), None)
        if hit is not None:
            new = new[: hit + 1]
        smp.observe(new)
        if sfx is not None:
            self._choose.record(sfx_n, sfx, dfl, use_sfx, n, nxt)
            if use_sfx:
                self.stats["sfx_steps"] = self.stats.get("sfx_steps", 0) + 1
                self.stats["sfx_acc"] = self.stats.get("sfx_acc", 0) + n
        self._drafts, self._qs = [], None
        if self.mtp is not None and hit is None and p + n + 1 < self.max_seq_len - 1:
            with self.mtp.own_ccl():
                self._drafts, self._qs = self._draft(len(rows), rows[1 : n + 1] + [nxt], p, self.k, smp)
        self._tok, self._pos = nxt, p + n + 1
        st = self.stats
        st["steps"] += 1
        st["emitted"] += len(new)
        st["t_verify"] += b - a
        st["t_draft"] += time.time() - b
        if st["steps"] % 512 == 0:
            logger.info(
                f"Inkling spec: {st['emitted'] / st['steps']:.2f} tokens/step over {st['steps']} steps, "
                f"verify {1e3 * st['t_verify'] / st['steps']:.1f} ms, draft "
                f"{1e3 * st['t_draft'] / st['steps']:.1f} ms per step, suffix drafts {st.get('sfx_steps', 0)} steps "
                f"({st.get('sfx_acc', 0) / max(1, st.get('sfx_steps', 0)):.2f} accepted)"
            )
        return new

    def _emit(self, toks):
        """The part of `toks` handed to vLLM now (the rest waits in self._pending), or None when nothing can be
        handed over yet. With INKLING_SPLIT_MARKERS (default): a text run up to the next block marker, or one marker
        alone; and an <|end_message|> directly before the end of turn is dropped (it closes the last block and
        carries nothing: vLLM's streamed content path, which no longer runs the parser, leaked it), so a trailing
        <|end_message|> waits for its successor. Only the hand-over is staged: the device state already covers every
        token, so greedy and sampled output are unchanged."""
        if not self.split_markers:
            self._pending = []
            return list(toks)
        mk = self.markers
        if toks[0] == self.end_message:
            if len(toks) == 1:
                self._pending = list(toks)
                return None
            if toks[1] in self.eos:
                toks = toks[1:]
        cut = 1 if toks[0] in mk else next((i for i, t in enumerate(toks) if t in mk), len(toks))
        self._pending = list(toks[cut:])
        return list(toks[:cut])

    def generate(self, prompt_ids, max_new: int, sampling=None):
        """Standalone use: -> generated ids (stops at the end of turn)."""
        out = [self.prefill(prompt_ids, sampling)]
        while len(out) < max_new and out[-1] not in self.eos:
            out += self.step()
        return out[:max_new]
