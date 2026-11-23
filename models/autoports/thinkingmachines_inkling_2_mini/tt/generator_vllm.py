# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""vLLM adapter for thinkingmachines/Inkling-2-Mini (text) on a (1, N) Blackhole mesh (N = 1, 2, 4: tensor parallel).

The engine owns every cache (tt/attention.AttnCache with one slot per decode row: the global layers' paged KV, the
sliding layers' rings, the conv states), so the KV allocation vLLM asks for is a token budget only and vLLM's page
table is not used. Requests are prefilled into a physical slot in chunks (one sequence at a time on the lane) and then
decode together in one traced batched step over every slot (tt/batched.py).

vLLM rows vs physical slots: the plugin condenses its persistent batch (a finished request's row is refilled with the
last live one and the move is reported as ``slot_remap``). Device state never moves: the adapter keeps a row -> slot
map, updated by each remap, and permutes the per-step inputs / outputs.

Sampling: on the host (the step returns the next-token logits); prefix caching and chunked prefill are off (v0).

Config: vLLM 0.26 / transformers 5.12 do not know ``inkling2_loop``; serve with ``--hf-config-path`` pointing at this
port's ``hf_config/`` (a flat config with an auto_map, ``--trust-remote-code``). The tokenizer and the chat template come
from the checkpoint; vLLM's ``inkling`` reasoning / tool parsers apply (same message tokens as Inkling v1).
"""

from __future__ import annotations

import os
import time

import torch
import ttnn
from loguru import logger

from models.autoports.thinkingmachines_inkling_2_mini.tt.batched import BatchedDecoder, TracedDecoder
from models.autoports.thinkingmachines_inkling_2_mini.tt.model import Inkling2Model
from models.autoports.thinkingmachines_inkling_2_mini.tt.sampling import _param

HF_MODEL = "thinkingmachines/Inkling-2-Mini"
DT = {"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b, "bfp4": ttnn.bfloat4_b}
EXPERT_DT = os.environ.get("I2M_EXPERT_DT", "bfp8")
KV_DT = os.environ.get("I2M_KV_DT", "bf16")
PREFILL_CHUNK = int(os.environ.get("I2M_PREFILL_CHUNK", "256"))
# a prompt's last chunk is padded to a multiple of this (64 | 128 | 256 ...): fewer prefill programs to warm up
PREFILL_PAD = int(os.environ.get("I2M_PREFILL_PAD", str(min(256, PREFILL_CHUNK))))
TRACE = os.environ.get("I2M_TRACE", "1") == "1"
TIMING = os.environ.get("I2M_TIMING", "0") == "1"
# decode width buckets: one decoder + trace per width (sharing the full width's state); a step runs the narrowest
# width covering the highest live slot (new requests take the lowest free slot)
WIDTHS = [int(w) for w in os.environ.get("I2M_DECODE_WIDTHS", "1,2,4,8,16,32").split(",")]  # per-100-step decode timing breakdown in the server log
# I2M_PREFIX (default 1): vLLM's prefix cache. vLLM pages the global layers' KV (tt/attention.GlobalPages) and keeps the
# cached prefixes; the rest of a sequence's state at a cached prefix's end (conv states, sliding-window KV) lives in the
# plugin's prefix-state slots: one snapshot per prompt at its last 64-token boundary (tt/prefix.py, device pools).
# I2M_PREFIX_SLOTS: snapshots kept (LRU; default 1 on one die, 16 on 2 / 4 dies: ~168 MB / dies each),
# I2M_PREFIX_MIN: the shortest prompt snapshotted.
PREFIX = os.environ.get("I2M_PREFIX", "1") == "1"
# I2M_SPEC d (default 0: off): speculative decoding of a request decoding alone (slot 0, prompt prefilled from 0, on-device
# sampling): the DSpark drafter (tt/drafter.py) proposes d tokens, one verify pass of d + 1 rows scores them, the
# accepted ones (tt/sampling.HostSampler.spec: the target's distribution exactly) go to vLLM in one decode step through
# the plugin's variable-width output shim (the M2.7 port's mtp_plugin_shim: 1..d + 1 tokens per row, -1 padded; the
# scheduler reserves the verify window's KV slots). d + 1 must be 2 or 4. I2M_SPEC_MAX_LEN: the longest sequence it
# serves (the drafter's caches).
SPEC = int(os.environ.get("I2M_SPEC", "0"))
if SPEC:
    from models.autoports.minimaxai_minimax_m2_7.tt import mtp_plugin_shim

    mtp_plugin_shim.apply(SPEC + 1)
SPEC_MAX_LEN = int(os.environ.get("I2M_SPEC_MAX_LEN", "32768"))
SPEC_DEBUG = os.environ.get("I2M_SPEC_DEBUG", "0") == "1"
# I2M_SPEC_MIN (2.0): a request whose tokens per pass (running mean, over >= 8 passes) fall below this goes back to plain
# decode for the rest of its generation (break-even ~ a pass's time / a one-row step's: ~1.9 on one die)
SPEC_MIN = float(os.environ.get("I2M_SPEC_MIN", "2.0"))  # (log every pass: position, anchor, drafts, emitted)
PREFIX_MIN = int(os.environ.get("I2M_PREFIX_MIN", "512"))


def _prefix_slots():
    n = os.environ.get("I2M_PREFIX_SLOTS")
    if n:
        return int(n)
    tv = os.environ.get("TT_VISIBLE_DEVICES", "")
    md = os.environ.get("MESH_DEVICE", "")
    one_die = md in ("P150", "N150") or (tv != "" and "," not in tv)
    return 1 if one_die else 16


PREFIX_SLOTS = _prefix_slots()


def _host_tensor(t):
    if t is None:
        return None
    return t if isinstance(t, torch.Tensor) else torch.as_tensor(t)


def _row(t, r):
    t = _host_tensor(t)
    return None if t is None else t[r]



# The global layers' KV pool (tokens), shared by the slots through vLLM's block manager: slots x max_model_len, capped
# at what fits beside the weights (bf16 KV: 8 global applications x 2 x 8 kv heads x 128 x 2 B = 32 KiB per token,
# / dies). tools/mem_probe.py with POOL=65536 at the card's slots x context, after the warm-ups and trace captures,
# free DRAM per die: 4 dies 16.55 GiB (32 x 1M), 2 dies 10.16 GiB (16 x 512K), 1 die 0.29 GiB (8 x 64K); the caps keep
# ~1.25 GiB free for prefill transients (the Ornith port's headroom). I2M_KV_POOL_TOKENS overrides the cap. 2048-row
# prefill chunks (the sliding rings hold the window + one chunk): +1.29 GiB per die at 4 dies x 32 slots. One die:
# 7 x 8K tokens, so the fused decode glue (MixPlan / pre / post programs) fits: 0.29 GiB free after the trace captures.
KV_POOL_CAP = {4: 2_064_384, 2: 647_168, 1: 57_344}
KV_POOL_CAP_2048 = {4: 1_895_296}


def kv_pool_tokens(num_devices, max_model_len, max_num_seqs):
    want = max_model_len * max_num_seqs
    env = os.environ.get("I2M_KV_POOL_TOKENS")
    caps = KV_POOL_CAP_2048 if PREFILL_CHUNK == 2048 else KV_POOL_CAP
    cap = int(env) if env else caps.get(num_devices, want)
    return (max(max_model_len, min(want, cap)) + 63) // 64 * 64

class Inkling2ForCausalLM:
    """vLLM bridge (HF arch Inkling2LoopForConditionalGeneration, text only)."""

    supports_multimodal = False
    model_capabilities = {
        "supports_prefix_caching": PREFIX,
        **({"prefix_state_cache": {"slots": PREFIX_SLOTS, "segment": 0, "min_tokens": PREFIX_MIN, "prompt_end": True,
                                   "prompt_end_before_last": True}} if PREFIX else {}),
        "supports_async_decode": False,
        "supports_sample_on_device": True,  # decode: device top-32 + host draw (tt/sampling.py)
        "supports_chunked_prefill": False,
        **({"max_output_tokens_per_step": SPEC + 1} if SPEC else {}),
    }

    def __init__(self, model: Inkling2Model = None, mesh=None, B: int = 1, max_seq_len: int = 0, vllm_config=None,
                 prefix: str = ""):
        # (vllm_config / prefix: vLLM's registry inspects the class for its model protocol; the TT plugin builds the
        # adapter through initialize_vllm_model)
        self.m, self.mesh, self.B, self.max_seq_len = model, mesh, B, max_seq_len
        self.dec = BatchedDecoder(model, B)
        self.widths = sorted({w for w in WIDTHS if w < B} | {B})
        self.decs = {w: (self.dec if w == B else BatchedDecoder(model, w, shared=self.dec)) for w in self.widths}
        self.trs = {w: TracedDecoder(d) for w, d in self.decs.items()} if TRACE else {}
        self.tr = self.trs.get(B)
        self.row2slot = {}  # live vLLM row -> physical slot (slots stay put; new requests take the lowest free one)
        from models.autoports.thinkingmachines_inkling_2_mini.tt.sampling import DeviceTopK, HostSampler

        self.topk = DeviceTopK(mesh, model.Vs, model.V, sharded=model.head_shard)
        self.hs = HostSampler(model.V)
        self.V = model.V
        self.vocab = int(model.cfg.vocab_size)
        self.N = mesh.get_num_devices()
        self.store = None
        if PREFIX and model is not None:
            from models.autoports.thinkingmachines_inkling_2_mini.tt.prefix import PrefixStore

            self.store = PrefixStore(model, PREFIX_SLOTS)  # (device pools: before any trace capture)
            logger.info(f"prefix cache: {PREFIX_SLOTS} state snapshots, prompts >= {PREFIX_MIN} tokens")
        self.px_stats = {"hits": 0, "hit_tokens": 0, "saves": 0, "fallbacks": 0}
        self.dr = self.spec = None
        if SPEC and TRACE and model is not None and B >= SPEC + 1:
            from models.autoports.thinkingmachines_inkling_2_mini.tt.drafter import DSparkDrafter

            self.dr = DSparkDrafter(mesh, model, max_len=min(max_seq_len, SPEC_MAX_LEN) + 64, d_max=SPEC)
            self.ver = BatchedDecoder(model, SPEC + 1, shared=self.dec, verify=SPEC + 1)
            self.vtr = TracedDecoder(self.ver)
            self.sp_stats = [0, 0]  # passes, drafts kept
            logger.info(f"speculative decoding: {SPEC} drafts per pass (sequences up to {SPEC_MAX_LEN} tokens)")
        self._captured = False
        self.already_warmed_up_prefill = False
        self._decode_steps = 0

    # vLLM's text-generation protocol (registry inspection only; the TT plugin drives prefill_forward / decode_forward)
    def embed_input_ids(self, input_ids):
        raise NotImplementedError("TT model: driven through prefill_forward / decode_forward")

    def forward(self, input_ids, positions, *args, **kwargs):
        raise NotImplementedError("TT model: driven through prefill_forward / decode_forward")

    def compute_logits(self, hidden_states):
        raise NotImplementedError("TT model: driven through prefill_forward / decode_forward")

    # ------------------------------------------------------------------ construction
    @classmethod
    def initialize_vllm_model(cls, hf_config, mesh_device, max_batch_size, max_seq_len=None, tt_data_parallel=1,
                              optimizations=None, **kwargs):
        if int(tt_data_parallel) != 1:
            raise ValueError(f"{cls.__name__}: tensor parallel only (tt_data_parallel={tt_data_parallel})")
        max_seq_len = int(max_seq_len or 32768)
        B = int(max_batch_size)
        t0 = time.time()
        pool = kv_pool_tokens(mesh_device.get_num_devices(), max_seq_len, B)
        model = Inkling2Model(mesh_device, max_seq_len=max_seq_len + max(64, PREFILL_PAD), expert_dtype=DT[EXPERT_DT], kv_dtype=DT[KV_DT],
                              prefill_chunk=PREFILL_CHUNK, slots=B, kv_pool_tokens=pool)
        logger.info(f"{cls.__name__}: global-layer KV pool {pool} tokens (shared by the {B} slots)")
        logger.info(f"{cls.__name__}: {B} slots x {max_seq_len} tokens, experts {EXPERT_DT}, KV {KV_DT}, "
                    f"{mesh_device.get_num_devices()} device(s), up in {time.time() - t0:.0f}s (engine-owned caches)")
        return cls(model, mesh_device, B, max_seq_len)

    # ------------------------------------------------------------------ KV: a token budget only
    @classmethod
    def get_max_tokens_all_users(cls, model_name="", num_devices=1, tt_data_parallel=1, max_model_len=None,
                                 max_num_seqs=None, **kwargs):
        return kv_pool_tokens(int(num_devices or 1), int(max_model_len or 32768), int(max_num_seqs or 1))

    def allocate_kv_cache(self, kv_cache_shape, dtype, num_layers):
        nb = int(kv_cache_shape[0])
        if PREFIX:
            self.m.pages.enable_vllm(nb)
        logger.info(f"KV: vLLM asked for {num_layers} x {tuple(kv_cache_shape)}; the engine keeps its own caches"
                    + (f" (the global layers' KV through vLLM's {nb} block ids)" if PREFIX else ""))
        return [(None, None)] * int(num_layers)

    def allocate_kv_cache_per_layer(self, per_layer_specs):
        """(the plugin's per-layer form: [(shape, dtype, layer name)], one homogeneous full-attention spec here)"""
        shape, dtype = per_layer_specs[0][0], per_layer_specs[0][1]
        self.allocate_kv_cache(shape, dtype, 1)
        return [(None, None)] * len(per_layer_specs)

    # ------------------------------------------------------------------ warm-up
    def warmup_model_prefill(self, kv_cache=None, enable_trace=False, *args, **kwargs):
        """Compile every prefill chunk size (multiples of 64 up to the chunk) on slot 0 before any trace capture."""
        if not getattr(self, "_pf_warmed", False):  # (the plugin may call this again after the decode capture)
            self._pf_warmed = True
            t0 = time.time()
            for n in range(PREFILL_PAD, PREFILL_CHUNK + 1, PREFILL_PAD):
                self.m.reset()
                self._host_logits(self.m.forward_packed(torch.zeros(n, dtype=torch.long), [(0, n, 0, n, 0, 0)]), 1)
            # several sequences per chunk (every lane once), at every chunk size
            L = self.m.lanes
            for n in range(PREFILL_PAD, PREFILL_CHUNK + 1, PREFILL_PAD):
                k = max(2, min(L, n // 64))
                sizes = [n // k // 64 * 64] * k
                sizes[-1] += n - sum(sizes)
                for first in range(0, L, k):
                    segs, r0 = [], 0
                    for j, R in enumerate(sizes):
                        segs.append((r0, R, 0, R, j % self.B, (first + j) % L))
                        r0 += R
                    self.m.reset()
                    self._host_logits(self.m.forward_packed(torch.zeros(n, dtype=torch.long), segs), len(segs))
            # long prompts: the global layers' log-scaling (tau) ops past position 8192, at every chunk size
            n_floor = int(self.m.cfg.log_scaling_n_floor or 0)
            if n_floor and self.max_seq_len > n_floor + PREFILL_CHUNK:
                p_hi = (n_floor + 63) // 64 * 64
                for n in range(PREFILL_PAD, PREFILL_CHUNK + 1, PREFILL_PAD):
                    self.m.reset()
                    self.m.forward_packed(torch.zeros(n, dtype=torch.long), [(0, n, p_hi, n, 0, 0)])
            lg = self.m.forward_packed(torch.zeros(PREFILL_PAD, dtype=torch.long), [(0, PREFILL_PAD, 0, PREFILL_PAD, 0, 0)])
            for u in range(1, L + 1):  # the logits readback of u packed rows
                self._host_logits(lg, u)
            self.m.reset()
            if self.store is not None:  # the snapshot copy program and the conv-state slices of every chunk size
                self.store.warm()
                for n in range(PREFILL_PAD, PREFILL_CHUNK + 1, PREFILL_PAD):
                    for r in (64, 0):
                        self.m.reset()
                        self.store.arm(r, 0)
                        self.m.forward_packed(torch.zeros(n, dtype=torch.long), [(0, n, 0, n, 0, 0)])
                        self.store.save(0, 0, max(r, 64), 0)
                self.store.restore(0, 0, 64, 0)
                self.m.reset()
            logger.info(f"prefill warm-up ({PREFILL_CHUNK // PREFILL_PAD} chunk sizes) in {time.time() - t0:.0f}s")
        self.already_warmed_up_prefill = True

    def warmup_model_decode(self, *args, **kwargs):
        if not self.already_warmed_up_prefill:
            self.warmup_model_prefill()
        if self.trs and not self._captured:
            t0 = time.time()
            for w in self.trs:  # every width's programs first (eager), then the captures
                self.decs[w].set_inputs(torch.zeros(w, dtype=torch.long), torch.full((w,), -1))
                out = self.decs[w].step()
                self._host_logits(out, w)  # (the readback untilize is a program too)
                self.topk(out, w)  # the device-sampling programs of this width
            if self.dr is not None:
                self._spec_warm()
            ttnn.synchronize_device(self.mesh)
            if self.dr is not None:
                R = SPEC + 1
                self.ver.set_inputs(torch.zeros(R, dtype=torch.long), torch.full((R,), -1))
                self.vtr.capture(compile=False, post=lambda lg: self.topk.device(lg, R))
                # (its taps buffer: the verify trace's taps output; sampled drafts: zero noise = greedy)
                self.dr.capture(self.ver.taps, SPEC, sample=True)
            for w, tr in self.trs.items():
                self.decs[w].set_inputs(torch.zeros(w, dtype=torch.long), torch.full((w,), -1))
                tr.capture(compile=False, post=lambda lg, w=w: self.topk.device(lg, w))
            self._captured = True
            logger.info(f"decode traces captured (widths {self.widths}) in {time.time() - t0:.1f}s")

    # ------------------------------------------------------------------ helpers
    def _host_logits(self, t, rows):
        """[1, 1, rows(+pad), Vs] fp32 TILE logits -> host [rows, V] (tt/model.Inkling2Model.logits_host)."""
        return self.m.logits_host(t, rows)

    def _apply_remap(self, slot_remap):
        """The plugin condensed its rows: new row i holds what old row remap[i] held (its slot moves with it)."""
        if slot_remap is None:
            return
        remap = [int(v) for v in torch.as_tensor(slot_remap).reshape(-1).tolist()]
        if remap == list(range(len(remap))):
            return
        old, new, used = dict(self.row2slot), {}, set()
        for i, r in enumerate(remap):
            if r in old and old[r] not in used:
                new[i] = old[r]
                used.add(old[r])
        self.row2slot = new

    def _slot_for(self, row):
        """A new request in vLLM row `row`: the lowest slot no other live row holds."""
        self.row2slot.pop(row, None)
        self.hs.forget(row)
        used = set(self.row2slot.values())
        slot = next(s for s in range(self.B) if s not in used)
        self.row2slot[row] = slot
        return slot

    # ------------------------------------------------------------------ prefill
    def prefill_forward(self, tokens, page_table, kv_cache, prompt_lens, **kwargs):
        """Prefill N requests into their rows' slots; returns the last prompt token's logits [N, 1, vocab] (host).

        Packed: up to LANES requests at a time share each prefill chunk (their rows stacked, 64-row aligned; the MoE
        reads each expert once per chunk however many sequences it holds). Prefix cache: a request vLLM resumes at
        start_pos > 0 restores the snapshot its directive names (the plugin's prefix_state) and prefills from there; a
        directive's saves snapshot the state at their positions."""
        N = int(tokens.shape[0])
        if self.spec is not None:  # (another request: the speculative one goes back to plain decode first)
            self._spec_exit()
        start_pos = kwargs.get("start_pos")
        starts = [int(v) for v in torch.as_tensor(start_pos).reshape(-1).tolist()] if start_pos is not None else [0] * N
        if any(starts) and not PREFIX:
            raise NotImplementedError("prefix caching is off (start_pos must be 0)")
        pt = None
        if PREFIX and page_table is not None:
            pt = page_table if isinstance(page_table, torch.Tensor) else torch.as_tensor(page_table)
        directives = kwargs.get("prefix_state") or [None] * N
        rows = kwargs.get("empty_slots")
        rows = list(range(N)) if rows is None else [int(r) for r in rows]
        plens = [int(prompt_lens[u]) for u in range(N)] if prompt_lens is not None else [int(tokens.shape[1])] * N
        out = torch.full((N, 1, self.vocab), float("-inf"))
        t0 = time.time()
        reqs = []
        for u in range(N):
            slot = self._slot_for(rows[u])
            if pt is not None:  # (its prompt's blocks only: see GlobalPages.set_row)
                self.m.pages.set_row(slot, pt[u], (plens[u] + 63) // 64)
            d = directives[u] or {}
            reqs.append(dict(u=u, slot=slot, ids=torch.as_tensor(tokens[u, : plens[u]]).to(torch.long), P=plens[u],
                             b0=starts[u], restore=d.get("restore"),
                             saves=sorted((int(q), int(k)) for q, k in (d.get("saves") or ()))))
        L = self.m.lanes
        for g in range(0, N, L):
            self._prefill_group(reqs[g : g + L], out)
        if N > 1 or TIMING:
            logger.info(f"prefill of {N} requests ({sum(r['P'] - r['b0'] for r in reqs)} new tokens) in "
                        f"{time.time() - t0:.2f}s")
        return out

    def _prefill_group(self, reqs, out):
        """Up to LANES requests, one prefill lane each, packed into shared chunks until every prompt is done."""
        spec = (self.dr is not None and len(reqs) == 1 and reqs[0]["slot"] == 0 and not reqs[0]["b0"]
                and reqs[0]["P"] + 64 <= SPEC_MAX_LEN)
        if spec:
            self.dr.reset()
        for lane, r in enumerate(reqs):
            r["lane"] = lane
            self.m.reset(lane)
            b0 = r["b0"]
            if b0:
                rs = r["restore"]
                if rs is not None and int(rs[1]) == b0 and self.store is not None:
                    self.store.restore(int(rs[0]), r["slot"], b0, lane)
                    self.px_stats["hits"] += 1
                    self.px_stats["hit_tokens"] += b0
                else:  # no snapshot at b0: the cached KV alone cannot resume the prompt; prefill it whole (exact)
                    logger.warning(f"prefix cache: no snapshot at {b0} (restore {rs}): prefilling {r['P']} tokens")
                    self.px_stats["fallbacks"] += 1
                    r["b0"] = b0 = 0
            r["p"] = b0
            r["saves"] = [(q, k) for q, k in r["saves"] if q > b0]
        live = list(reqs)
        while live:
            segs, ids, armed, used = [], [], [], 0
            for r in live:
                rem = r["P"] - r["p"]
                space = PREFILL_CHUNK - used
                if rem <= space and (rem + 63) // 64 * 64 <= space:
                    n, R = rem, (rem + 63) // 64 * 64
                else:
                    n = R = min(space, rem) // 64 * 64
                if n <= 0:
                    break
                p0 = r["p"]
                c = r["ids"][p0 : p0 + n]
                if R > n:
                    c = torch.cat([c, torch.zeros(R - n, dtype=c.dtype)])
                sv = [(q, k) for q, k in r["saves"] if p0 < q <= p0 + n]
                assert len(sv) <= 1, (r["saves"], p0, n)
                if sv:
                    self.store.arm(sv[0][0] - p0, r["lane"])
                    armed.append((r, sv[0]))
                segs.append([used, R, p0, n, r["slot"], r["lane"]])
                ids.append(c)
                used += R
            if used % PREFILL_PAD:  # whole chunk to a multiple of PREFILL_PAD rows: the last segment takes the padding
                extra = (-used) % PREFILL_PAD
                segs[-1][1] += extra
                ids.append(torch.zeros(extra, dtype=torch.long))
                used += extra
            taps = [] if spec else None
            lg = self.m.forward_packed(torch.cat(ids), [tuple(sg) for sg in segs], taps=taps)
            if spec:  # (one segment: the drafter's context of its rows)
                _, R0, p0, n0, _, _ = segs[0]
                self._dr_commit(ttnn.slice(taps[0], [0, 0, 0, 0], [1, 4, R0, self.m.H]), p0, n0)
            for r, (q, k) in armed:
                self.store.save(k, r["slot"], q, r["lane"])
                self.px_stats["saves"] += 1
            done = []
            for i, (_, R, p0, n, slot, lane) in enumerate(segs):
                r = next(x for x in live if x["lane"] == lane)
                r["p"] = p0 + n
                if r["p"] >= r["P"]:
                    done.append((i, r))
            if done:
                lgh = self._host_logits(lg, len(segs))
                for i, r in done:
                    self.dec.load_slot(r["slot"], r["lane"])
                    if spec:
                        self.ver.set_accept([0])
                        self.spec = dict(row=next(k for k, v in self.row2slot.items() if v == 0), n=None)
                    out[r["u"], 0, : self.V] = lgh[i]
                    live.remove(r)
                    logger.info(f"prefill {r['P']} tokens (from {r['b0']}) -> slot {r['slot']}"
                                + (f", snapshot at {r['saves'][0][0]}" if r["saves"] else ""))

    # ------------------------------------------------------------------ decode
    def decode_forward(self, *args, **kwargs):
        tokens = kwargs.get("tokens", args[0] if args else None)
        start_pos = kwargs.get("start_pos", args[1] if len(args) > 1 else None)
        self._apply_remap(kwargs.get("slot_remap"))
        tok = torch.as_tensor(tokens).reshape(-1).to(torch.long)
        pos = torch.as_tensor(start_pos).reshape(-1).to(torch.long)
        R = int(pos.numel())
        # rows the plugin no longer decodes have finished (pos -1, or past the batch it sends): their slots are free
        for r in list(self.row2slot):
            if r >= R or int(pos[r]) < 0:
                self.m.pages.free(self.row2slot.pop(r))  # (its blocks may go to another request: scratch rows)
        live = [r for r in range(R) if int(pos[r]) >= 0 and r in self.row2slot]
        if self.spec is not None:
            s = self.spec
            if s["row"] not in live:  # (finished)
                self.spec = None
            elif live == [s["row"]] and self.row2slot[s["row"]] == 0 and kwargs.get("sampling_params") is not None:
                return self._spec_decode(s["row"], int(tok[s["row"]]), int(pos[s["row"]]), R, kwargs)
            else:
                self._spec_exit(int(pos[s["row"]]))
        top = max((self.row2slot[r] for r in live), default=0)
        W = next(w for w in self.widths if w > top)
        if TIMING and W != getattr(self, "_last_w", None):
            logger.info(f"decode width {W} ({len(live)} live rows, top slot {top})")
            self._last_w = W
        tok_s = torch.zeros(W, dtype=torch.long)
        pos_s = torch.full((W,), -1, dtype=torch.long)
        pt = kwargs.get("page_table", args[2] if len(args) > 2 else None)
        if pt is not None and self.m.pages.paged:
            pt = pt if isinstance(pt, torch.Tensor) else torch.as_tensor(pt)
        for r in live:
            s = self.row2slot[r]
            tok_s[s], pos_s[s] = tok[r], pos[r]
            if pt is not None and self.m.pages.paged:  # (the blocks up to the one written this step)
                self.m.pages.set_row(s, pt[r], int(pos[r]) // 64 + 1)
        self.m.pages.sync(W)
        d = self.trs[W] if (self.trs and self._captured) else self.decs[W]
        t0 = time.perf_counter()
        d.set_inputs(tok_s, pos_s)
        t1 = time.perf_counter()
        dev = d.step()
        if TIMING:
            ttnn.synchronize_device(self.mesh)
        t2 = time.perf_counter()
        sp = kwargs.get("sampling_params")
        if sp is not None:  # [W, 32] candidates by physical slot (the traced step computed them too)
            extra = getattr(d, "extra", None)
            idx, vals = self.topk.host(extra) if extra is not None else self.topk(dev, W)
        else:
            lg = self._host_logits(dev, W)  # [W, V] by physical slot
        t3 = time.perf_counter()
        if sp is not None:  # device sampling: the tokens [R, 1]
            toks = self.hs.sample(idx, vals, [(r, self.row2slot[r]) for r in live], sp,
                                  output_tokens=_host_tensor(kwargs.get("output_tokens")),
                                  prompt_tokens=_host_tensor(kwargs.get("prompt_tokens")))
            out = torch.zeros((R, 1), dtype=torch.int32)
            for r, t in toks.items():
                out[r, 0] = t
        else:
            out = torch.full((R, 1, self.vocab), float("-inf"))
            for r in live:
                out[r, 0, : self.V] = lg[self.row2slot[r]]
        t4 = time.perf_counter()
        self._decode_steps += 1
        if TIMING:
            acc = self._tacc = getattr(self, "_tacc", [0.0] * 5)
            for i, v in enumerate((t1 - t0, t2 - t1, t3 - t2, t4 - t3, t0 - getattr(self, "_tlast", t0))):
                acc[i] += v
            self._tlast = t4
            if self._decode_steps % 100 == 0:
                n = 100
                logger.info(f"decode timing (ms/step over {n}): inputs {acc[0] / n * 1e3:.1f}  device {acc[1] / n * 1e3:.1f}  "
                            f"readback {acc[2] / n * 1e3:.1f}  pack {acc[3] / n * 1e3:.1f}  outside (plugin + sampling) "
                            f"{acc[4] / n * 1e3:.1f}")
                self._tacc = [0.0] * 5
        return out

    # ------------------------------------------------------------------ speculative decode
    def _spec_warm(self):
        """Every eager program of speculative decode, compiled before the trace captures (the drafter warmed its own when
        built)."""
        R = SPEC + 1
        self.ver.set_inputs(torch.zeros(R, dtype=torch.long), torch.full((R,), -1))
        self.topk(self.ver.step(), R)
        for j in range(1, R):  # leaving speculative decode from row j
            self.ver.move_row(j, 0)

    def _dr_commit(self, taps, p0, valid):
        """A prefill chunk's taps [1, 4, T, H] (rows past `valid`: padding) into the drafter, <= 512 rows per call."""
        for c0 in range(0, valid, 512):
            c1 = min(valid, c0 + 512)
            T = (c1 - c0 + 63) // 64 * 64
            self.dr.commit(ttnn.slice(taps, [0, 0, c0, 0], [1, 4, c0 + T, self.m.H]), p0 + c0, valid=c1 - c0)

    def _spec_decode(self, r, t, p, nrows, kwargs):
        """The speculative request's step (token t at position p): one pass -> its accepted tokens, all in this step
        ([nrows, d + 1] int32, -1 padded). A pass: the drafter's trace (commit the last pass's accepted rows, draft d),
        then the verify trace."""
        s = self.spec
        commit = None  # (rows of the last pass's taps the drafter commits: (first position, count))
        if s["n"] is not None:
            i = p - 1 - s["n"]  # (vLLM holds tokens up to position p - 1: rows 0 .. i of the last pass)
            if not (i == s["k"] and t == s["em"][s["k"]]):  # (vLLM went another way, e.g. a stop it trimmed)
                assert 0 <= i <= s["k"], (p, s["n"], s["k"])
                logger.warning(f"speculative decode: resync at position {p}")
                self.ver.set_accept([i])
            commit = (s["n"], i + 1)
        d, R = SPEC, SPEC + 1
        pt = kwargs.get("page_table")
        if pt is not None and self.m.pages.paged:  # (the verify window's blocks: the shim's lookahead reserved them)
            pt = pt if isinstance(pt, torch.Tensor) else torch.as_tensor(pt)
            self.m.pages.set_row(0, pt[r], (p + R - 1) // 64 + 1)
        self.m.pages.sync(self.ver.table_key)
        sp = kwargs["sampling_params"]
        self.dr.set_gumbel(self.hs.gumbel(r, sp, d, 32))  # (the drafter samples from its q at the request's T)
        if commit is None:  # (right after the prefill, whose commit is done)
            _, drafts = self.dr.draft(t, p, d=d, sample=True)
        else:
            self.dr.set_inputs(t, *commit)
            self.dr.replay()
            drafts = [int(x) for x in torch.as_tensor(self.dr.tokens_host()).reshape(-1)[:d]]
        q = None
        if float(_param(sp, "temperature", r, 0.0)) > 0.0:
            ci, cl = self.dr.cand_host(d)
            q = [(ci[j], cl[j]) for j in range(d)]
        self.vtr.set_inputs(torch.tensor([t] + drafts), torch.arange(p, p + R))
        self.vtr.step()
        idx, vals = self.topk.host(self.vtr.extra)
        em, k = self.hs.spec(idx, vals, r, drafts, sp, outs=_row(kwargs.get("output_tokens"), r),
                             prompt=_row(kwargs.get("prompt_tokens"), r), q=q)
        self.ver.set_accept([k])
        if SPEC_DEBUG:
            logger.info(f"spec pass p={p} t={t} commit={commit} drafts={drafts} emitted={em}")
        s.update(n=p, k=k, em=em)
        s["np"], s["nt"] = s.get("np", 0) + 1, s.get("nt", 0) + len(em)
        if s["np"] >= 8 and s["nt"] / s["np"] < SPEC_MIN:  # (not paying off for this request: plain decode from here)
            if TIMING:
                logger.info(f"speculative decode off for this request: {s['nt'] / s['np']:.2f} tokens per pass")
            self._spec_exit(p + k + 1)
        st = self.sp_stats
        st[0], st[1] = st[0] + 1, st[1] + k
        if TIMING and st[0] % 100 == 0:
            logger.info(f"speculative decode: {st[1] / st[0]:.2f} drafts kept per pass over {st[0]} passes")
        out = torch.full((nrows, R), -1, dtype=torch.int32)
        out[:, 0] = 0  # (rows vLLM is not decoding: one token, as the plain path returns)
        out[r, : len(em)] = torch.tensor(em, dtype=torch.int32)
        return out

    def _spec_exit(self, p=None):
        """Back to plain decode: the speculative request's state to its slot's row (rewound to what vLLM holds: the
        position p it decodes next, or every token of the last pass)."""
        s, self.spec = self.spec, None
        if s["n"] is None:
            return
        i = (p - 1 - s["n"]) if p is not None else s["k"]  # (vLLM holds every token a pass emitted)
        if i > 0:
            self.ver.move_row(i, 0)
        self.ver.set_accept([0])

    def read_decode_output(self, tt_out, async_read=False, **kwargs):
        return (tt_out, []) if async_read else tt_out

    def process_decode_output_host(self, tt_out, is_tokens=False, **kwargs):
        return tt_out


__all__ = ["Inkling2ForCausalLM", "HF_MODEL"]
