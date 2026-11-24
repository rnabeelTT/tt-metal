# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""vLLM adapter for tencent/Hy4-flash (text) on a (1, 4) Blackhole mesh (vllm-tt-plugin, dflash branch).

One user (``--max-num-seqs 1``: tt/engine.py, optional speculative decoding HY4_SPEC_K) or up to 8 (tt/engine_batch.py:
every live request decodes in one batched pass; vLLM rows map to model slots, a ``slot_remap`` only permutes that map).
The engine owns every cache, so
the KV allocation vLLM asks for is a token budget only and vLLM's prefix caching is off. Prefill returns the first token
(on-device sampling mode) or, when the plugin samples on the host (logprobs, structured outputs, logit bias ...), the
first-token logits; decode returns a host int32 [B, 1] token row (or [B, 1, V] logits for host-sampled requests).

Registration: vllm_bundle/hy4_flash/vllm_metadata.json (EXTRA_MODELS_DIR), the HF config / `hy_v4` parsers through
serving/sitecustomize.py.
"""

from __future__ import annotations

import math
import os
import time

import torch
from loguru import logger

HF_MODEL = "tencent/Hy4-flash"
BLOCK_SIZE = 64
SPEC_K = int(os.environ.get("HY4_SPEC_K", "0"))

if SPEC_K > 0:
    # variable-width block output (1..k+1 tokens per step, -1 padded) through the plugin (the M2.7 / Inkling shim)
    from models.autoports.minimaxai_minimax_m2_7.tt import mtp_plugin_shim

    mtp_plugin_shim.apply(SPEC_K + 1)


def _mtp_sampling(sp, row):
    """(temperature, top_k, top_p, seed) of request `row` from the plugin's sampling params; None when absent."""
    if sp is None:
        return None

    def pick(v, default):
        if v is None:
            return default
        if isinstance(v, (int, float)):
            return v
        try:
            x = v[row]
        except (IndexError, TypeError):
            return default
        if isinstance(x, torch.Tensor):
            x = x.item()
        return default if x is None else x

    t = float(pick(getattr(sp, "temperature", None), 0.0))
    k = int(pick(getattr(sp, "top_k", None), 0) or 0)
    p = float(pick(getattr(sp, "top_p", None), 1.0))
    seed = pick(getattr(sp, "seed", None), None)
    seed = None if seed is None or int(seed) < 0 else int(seed)
    return (t, k, p, seed)


def _resample(smp, sampling):
    """Keep a request's sampler (its RNG stream) unless its sampling params changed."""
    from models.autoports.tencent_hy4_flash.tt.engine import HostSampler

    if sampling is None:
        return smp
    t, k, p, _ = sampling
    if (float(t), int(k or 0), float(p)) == (smp.t, smp.k, smp.p):
        return smp
    return HostSampler(sampling)


try:  # image inputs (HY4_VISION=1, default): the multimodal processor (serving/hy4_mm.py)
    try:
        import hy4_mm as _hy4_mm
    except ImportError:
        from models.autoports.tencent_hy4_flash.serving import hy4_mm as _hy4_mm
except ImportError as _e:  # dev env without vllm (engine smokes): no multimodal processor, text only
    if not (_e.name or "").startswith("vllm"):
        raise
    _hy4_mm = None
_VISION = _hy4_mm is not None and _hy4_mm.enabled()


def cache_gib(users: int, ctx: int, chunk: int = 2048) -> float:
    """Per-die DRAM of the slots' caches (tt/attention.py, tt/indexer.py, tt/decode_batch.py): the 12 sparse layers'
    latent rows (608 bf16 + lookahead), the 30 SWA layers' rings (chunk + window + 32 rows), the indexer group keys per
    slot (+ the batched decoder's lane copy)."""
    S, m = int(users), int(ctx) + 64
    rows = -(-m // 32) * 32 + 64
    gcap = -(-(m // 4 + 96) // 2048) * 2048
    ring = -(-(chunk + 512 + 32) // 32) * 32
    b = S * rows * 1216 * 12 + S * ring * 1216 * 30 + 12 * gcap * 256 * S
    if S > 1:
        b += 12 * (S * gcap + 64) * 256
    return b / 2**30


def cache_budget_gib(users: int) -> float:
    """The slots' cache budget per die (HY4_CACHE_GIB overrides): the DRAM the caches + free space take after warm-up
    and a long prefill, measured 2026-11-21 (tools/dram_budget.py, probe_compile.py: 1 slot, spec 0, no ViT, 400 MiB
    trace region: 1 x 262144 cached 3.85 GiB with 1.00 GiB free), less the DFly2 drafter (HY4_SPEC_K > 0), the ViT
    (HY4_VISION), the batched engine's lanes / traces (users > 1), a long prefill's scratch peak (~0.5 GiB at 128K) and a
    margin; plus whatever the trace region (HY4_TRACE_REGION) leaves of 400 MiB."""
    env = os.environ.get("HY4_CACHE_GIB")
    if env:
        return float(env)
    g = BUDGET_BASE_GIB - BUDGET_RESERVE_GIB
    if int(os.environ.get("HY4_SPEC_K", "0") or 0) > 0:
        g -= BUDGET_SPEC_GIB
    if os.environ.get("HY4_VISION", "1") == "1":
        g -= BUDGET_VISION_GIB
    if int(users) > 1:
        g -= BUDGET_MULTI_GIB
    g += (400 * 2**20 - int(os.environ.get("HY4_TRACE_REGION", str(400 * 2**20)))) / 2**30
    return g


BUDGET_BASE_GIB = 4.85  # 1 x 262144 (3.85 GiB of caches) + 1.00 GiB free after a 262000-token prefill
BUDGET_RESERVE_GIB = 0.75  # a long prefill's scratch peak (+0.51 GiB at 131K) + margin
BUDGET_SPEC_GIB = 0.56  # DFly2 drafter layers + its traces (1 x 128K: 2.85 -> 2.29 GiB free)
BUDGET_VISION_GIB = 0.25  # the ViT's weights
BUDGET_MULTI_GIB = 0.0  # (calibrated below from the 8-slot run)


# HY4_POOL=1 (default): 2..8 users share one context pool (tt/pool.py): one user can take nearly all of it
POOL = os.environ.get("HY4_POOL", "1") == "1"
# HY4_CHUNKED_PREFILL=1 (default): vLLM chunked prefill. A long prompt arrives in parts of HY4_CP_TOKENS (default 2048)
# tokens, one prefill step per part with the other users' decode steps in between (serving/sitecustomize.py: the
# scheduler alternates; the runner passes which rows are intermediate parts). The engine computes each part exactly as
# its whole-prompt prefill would (tt/engine_batch.py prefill_part): same outputs, no multi-second decode stalls.
CHUNKED = os.environ.get("HY4_CHUNKED_PREFILL", "1") == "1"
CP_LOG = os.environ.get("HY4_CP_LOG", "0") == "1"
STATS_EVERY = max(1, int(os.environ.get("HY4_STATS_EVERY", "1024")))  # decode steps between engine stats log lines
# HY4_SPEC_DBG=1 (diagnostic): per decode step what vLLM hands the adapter (auth flags, host vs device sampling, its
# token / position vs the engine's), summarized every 256 steps and logged at once on a change of that state
SPEC_DBG = os.environ.get("HY4_SPEC_DBG", "0") == "1"


def _dbg_step(self, eng, live, tokens, pos, sp, kwargs, auth):
    st = self.__dict__.setdefault("_dbg", {"n": 0, "auth": 0, "host": 0, "tokdiff": 0, "posdiff": 0, "last": None})
    st["n"] += 1
    st["auth"] += int(bool(auth))
    st["host"] += int(sp is None)
    diffs = []
    for i in live:
        q = eng.seqs.get(self._row_slot.get(i))
        if q is None:
            continue
        if q["tok"] is not None and int(tokens[i]) != q["tok"]:
            st["tokdiff"] += 1
            diffs.append(("tok", i, int(tokens[i]), q["tok"]))
        if int(pos[i]) != q["pos"]:
            st["posdiff"] += 1
            diffs.append(("pos", i, int(pos[i]), q["pos"]))
    state = (len(live), bool(auth), sp is None, bool(kwargs.get("reload_inputs", False)), bool(kwargs.get("reset_batch", False)))
    if state != st["last"] or (diffs and st["tokdiff"] + st["posdiff"] <= 50):
        logger.info(f"HY4DBG step {st['n']}: live {len(live)} rows {live} slots {[self._row_slot.get(i) for i in live]} "
                    f"auth {auth} reload {state[3]} reset {state[4]} host_sampling {sp is None} diffs {diffs[:4]} "
                    f"idle {sorted(eng.idle)} seqs {sorted(eng.seqs)} solo {eng.solo} model.slot {eng.model.slot}")
        st["last"] = state
    if st["n"] % 256 == 0:
        logger.info(f"HY4DBG summary {st}; spec {eng.stats.get('spec_emitted', 0)}/{eng.stats.get('spec_passes', 0)} "
                    f"bspec {eng.stats.get('bspec_emitted', 0)}/{eng.stats.get('bspec_rows', 0)} ra_taken "
                    f"{eng.stats.get('ra_taken', 0)} ra_undone {eng.stats.get('ra_undone', 0)} pool {eng.pool_stats}")
POOL_TOKEN_BYTES = 12 * 1216 + 12 * 256 // 4  # per token per die: the sparse layers' latent row + its group-key share


def pool_plan(users: int, max_model_len: int | None = None):
    """(P pool tokens, vLLM token budget, the largest max_model_len) of the shared pool for `users` (2..8) slots: the
    cache budget (cache_budget_gib) less the SWA rings, the indexers' work buffer / score scratch (sized for max_model_len;
    default: the largest that fits) and the sink rows, over 15,360 B/token. The vLLM budget leaves each slot its
    extent's lookahead / rounding (640 tokens) and one prefill's bucket padding (2048) + the plugin's output reservation
    (512): vLLM preempts before the pool runs out."""
    S = int(users)
    budget = cache_budget_gib(S) * 2**30
    ring = -(-(2048 + 512 + 32) // 32) * 32
    if os.environ.get("HY4_SWA_SPLIT", "1") == "1":  # (tt/attention.py: per slot a 576-row decode ring + one prefill ring)
        rings = S * (-(-(512 + 64) // 32) * 32) + ring
    else:
        rings = S * ring
    fixed = rings * 1216 * 30 + 12 * 96 * 1216 + 12 * 96 * 256

    def plan(mml):
        gcap = -(-((mml + 64) // 4 + 96) // 2048) * 2048
        wb = (1 if os.environ.get("HY4_POOL_DIRECT", "1") == "1" else 12) * gcap * 256  # (direct: one score scratch)
        P = int((budget - fixed - wb) // POOL_TOKEN_BYTES) // 2048 * 2048
        V = (P - S * 640 - 2048 - 512) // BLOCK_SIZE * BLOCK_SIZE
        return P, V

    if max_model_len is None:
        mml = 262144
        for _ in range(4):  # (the work buffer depends on max_model_len: a fixed point)
            P, V = plan(mml)
            mml = V
        P, V = plan(mml)
        return P, V, min(mml, V) // 2048 * 2048
    P, V = plan(int(max_model_len))
    return P, V, V


def max_ctx_for(users: int) -> int:
    """The largest max_model_len (a multiple of 2048) the budget allows for `users` slots."""
    c = 2048
    while cache_gib(users, c + 2048) <= cache_budget_gib(users):
        c += 2048
    return c


class HYV4FlashForConditionalGeneration:
    """vLLM bridge (HF arch HYV4FlashForConditionalGeneration; text + images with HY4_VISION=1)."""

    supports_multimodal = False
    model_capabilities = {
        "supports_prefix_caching": False,
        "supports_async_decode": False,
        "supports_sample_on_device": True,
        "supports_chunked_prefill": CHUNKED,
        # (vllm-tt-plugin with the chunked-prefill interleave patch: the part size, and the intermediate-part mask passed
        # to prefill_forward; older plugins: serving/sitecustomize.py does both)
        "chunked_prefill_tokens": int(os.environ.get("HY4_CP_TOKENS", "2048")),
        "prefill_intermediate_mask": True,
    }
    decode_input_update_contract = 1

    def __init__(self, model, engine, mesh_device, max_seq_len):
        self.model, self._engine, self.mesh_device = model, engine, mesh_device
        self._context = max_seq_len
        self._warmed = False
        self.already_warmed_up_prefill = False
        self._decode_steps = 0
        self.multi = bool(getattr(engine, "multi", False))
        self._row_slot = {}  # multi: vLLM batch row -> model slot
        if self.multi:  # (a finished request's slot the engine evicted from the shared pool: no row maps to it)
            engine.on_evict = lambda s: [self._row_slot.pop(r) for r in [r for r, v in self._row_slot.items() if v == s]]

    @classmethod
    def initialize_vllm_model(cls, hf_config, mesh_device, max_batch_size, max_seq_len=None, tt_data_parallel=1,
                              optimizations=None, **kwargs):
        B = int(max_batch_size)
        if int(tt_data_parallel) != 1 or not 1 <= B <= 8:
            raise ValueError(f"{cls.__name__} serves 1..8 users on one (1,4) mesh (--max-num-seqs <= 8, data parallel 1)")
        max_seq_len = int(max_seq_len or 131072)
        pool = None
        if POOL and B > 1:  # one shared context pool: any user up to max_model_len, all users together up to the pool
            pool, vtok, _ = pool_plan(B, max_seq_len)
            if max_seq_len > vtok:
                raise ValueError(f"{B} users: max_model_len {max_seq_len} > the shared pool's {vtok} tokens "
                                 f"({pool} pool tokens, {cache_budget_gib(B):.2f} GiB budget, HY4_CACHE_GIB): use "
                                 f"--max-model-len <= {pool_plan(B)[2]}")
        else:
            # every slot holds max_model_len tokens: the slots' caches (cache_gib) must leave the DRAM a long prefill's
            # scratch needs after warm-up (tools/dram_budget.py; HY4_SLOT_TOKENS_MAX: the old users x ctx token cap)
            need, cap = cache_gib(B, max_seq_len), cache_budget_gib(B)
            legacy = os.environ.get("HY4_SLOT_TOKENS_MAX")
            if (legacy and B > 1 and B * max_seq_len > int(legacy)) or (not legacy and need > cap):
                raise ValueError(f"{B} users x max_model_len {max_seq_len}: caches {need:.2f} GiB per die exceed the "
                                 f"{cap:.2f} GiB budget (HY4_CACHE_GIB): use --max-model-len <= {max_ctx_for(B)}")
        from models.autoports.tencent_hy4_flash.tt.config import Hy4Config, checkpoint_dir
        from models.autoports.tencent_hy4_flash.tt.engine import Hy4Engine
        from models.autoports.tencent_hy4_flash.tt.engine_batch import Hy4BatchEngine
        from models.autoports.tencent_hy4_flash.tt.model import Hy4Model
        from models.experimental.hybrid_moe.engram_store import EngramRows

        cfg = Hy4Config.load()
        from models.experimental.hybrid_moe import engram_store

        edir = os.environ.get("HY4_ENGRAM_DIR", os.path.expanduser("~/.cache/hybrid-moe/hy4_engram"))
        from models.autoports.tencent_hy4_flash.tt import hostmem

        hostmem.check_startup(edir)  # HY4_RAM_CHECK: refuse to start when the host cannot hold the server
        engram_store.OUT = edir
        engram_store._SNAP = checkpoint_dir()
        engram_store.convert()  # first boot: the FP8 tables (~65 GB, idempotent: converted layers are skipped)
        engram = EngramRows(edir, checkpoint_dir())
        cache_dir = os.environ.get("HY4_TENSOR_CACHE", os.path.expanduser("~/.cache/hybrid-moe/hy4_tt"))
        model = Hy4Model(mesh_device, cfg, max_ctx=max_seq_len + 64, chunk=int(os.environ.get("HY4_PREFILL_CHUNK", "2048")),
                         engram=engram, cache_dir=cache_dir, slots=B, pool_tokens=pool)
        # B > 1 with HY4_SPEC_K > 0: speculative decoding while one request decodes alone (tt/engine_batch.py solo)
        engine = Hy4BatchEngine(model, k=SPEC_K) if B > 1 else Hy4Engine(model, SPEC_K)
        logger.info(f"{cls.__name__}: max_model_len={max_seq_len} users={B} layers={len(model.layers)} spec k={SPEC_K}"
                    + (f" shared pool {pool} tokens" if pool else ""))
        self = cls(model, engine, mesh_device, max_seq_len)
        if _VISION:  # the vision tower (tt/vision.py: tensor parallel, ~0.22 GB per die in bf16)
            from models.autoports.tencent_hy4_flash.tt.vision import Hy4Vision

            self.vit = Hy4Vision(mesh_device)
        return self

    # ------------------------------------------------------------------ KV: a token budget only
    @classmethod
    def get_max_tokens_all_users(cls, model_name="", num_devices=1, tt_data_parallel=1, max_model_len=None,
                                 max_num_seqs=None, **kwargs):
        B = max(1, int(max_num_seqs or 1))
        if POOL and B > 1:  # the shared pool's token budget (pool_plan): vLLM preempts before the pool runs out
            return pool_plan(B, int(max_model_len or 131072))[1]
        n = int(max_model_len or 131072) + BLOCK_SIZE
        return math.ceil(n / BLOCK_SIZE) * BLOCK_SIZE * B

    def allocate_kv_cache(self, kv_cache_shape, dtype, num_layers):
        logger.info(f"KV: vLLM asked for {num_layers} layers of {tuple(kv_cache_shape)}; the engine keeps its own caches")
        return [(None, None)] * int(num_layers)

    def allocate_kv_cache_per_layer(self, per_layer_specs):
        return [(None, None)] * len(per_layer_specs)

    # ------------------------------------------------------------------ warm-up
    def warmup_model_prefill(self, kv_cache, enable_trace, *args, **kwargs):
        if getattr(self, "vit", None) is not None and not self._warmed:
            self.vit.warm(self.model)  # every ViT bucket + the image-row splice before any trace capture
        if not self._warmed:
            self._engine.warm()
            self._warmed = True
        self.already_warmed_up_prefill = True

    def warmup_model_decode(self, *args, **kwargs):
        if not self._warmed:
            self._engine.warm()
            self._warmed = True
        if kwargs.get("enable_trace", False):
            self._engine.capture()

    # ------------------------------------------------------------------ prefill
    def _bind(self, row, slot):
        """vLLM row -> model slot; another row still mapped to that slot (its request finished, the slot reallocated)
        forgets it."""
        for r in [r for r, s in self._row_slot.items() if s == slot and r != row]:
            del self._row_slot[r]
        self._row_slot[row] = slot

    def _parts(self, N, kwargs):
        """(start positions, intermediate flags) of a prefill step's rows. A row is an intermediate part of its prompt
        when vLLM chunked prefill scheduled only part of it (the plugin's intermediate_prefill_mask: the kwarg
        `intermediate_prefill`, or set on the model by serving/sitecustomize.py's runner shim)."""
        sp0 = kwargs.get("start_pos")
        start = [int(v) for v in torch.as_tensor(sp0).reshape(-1).tolist()] if sp0 is not None else [0] * N
        inter = kwargs.get("intermediate_prefill")
        if inter is None:
            inter, self._hy4_inter = getattr(self, "_hy4_inter", None), None
        inter = [bool(v) for v in torch.as_tensor(inter).reshape(-1).tolist()] if inter is not None else None
        if inter is None:
            if any(start):
                raise NotImplementedError("Hy4: a prompt part (start_pos > 0) without the runner's intermediate-part "
                                          "mask (serving/sitecustomize.py not loaded?); set HY4_CHUNKED_PREFILL=0")
            inter = [False] * N
        return start, inter

    def prefill_forward(self, tokens, page_table, kv_cache, prompt_lens, **kwargs):
        t0 = time.perf_counter()
        last = getattr(self, "_t_ret", None)
        out = self._prefill_forward(tokens, page_table, kv_cache, prompt_lens, **kwargs)
        t1 = time.perf_counter()
        if CP_LOG:  # (HY4_CP_LOG=1: each prefill step's time in the model and since the last model call returned)
            logger.info(f"Hy4 prefill step: {int(tokens.shape[0])} rows, ends {[int(v) for v in prompt_lens]}, "
                        f"{(t1 - t0) * 1e3:.1f} ms in the model, {(t0 - last) * 1e3 if last else 0:.1f} ms before")
        self._t_ret = t1
        return out

    def _prefill_forward(self, tokens, page_table, kv_cache, prompt_lens, **kwargs):
        N = int(tokens.shape[0])
        start, inter = self._parts(N, kwargs)
        plens = [int(prompt_lens[u]) for u in range(N)] if prompt_lens is not None else [int(tokens.shape[1])] * N
        sp = kwargs.get("sampling_params")
        if sp is not None and any(inter):  # (the plugin samples on the host when a step has an intermediate part)
            raise RuntimeError("Hy4: device sampling asked for a step with intermediate prompt parts")
        mms = self._images(tokens, plens, kwargs, skip=inter)
        if self.multi:
            out = self._prefill_multi(tokens, plens, sp, kwargs.get("empty_slots"), mms, start, inter)
            self._check_orphans("prefill")
            return out
        V = int(self._engine.V)
        out, logits = [], torch.zeros((N, 1, V), dtype=torch.float32)
        for u in range(N):  # (one user: no one decodes meanwhile; the whole prompt at its last part)
            if inter[u]:
                out.append(0)
                continue
            out.append(self._engine.prefill(tokens[u, : plens[u]].tolist(), _mtp_sampling(sp, u), mm=mms[u]))
            logits[u, 0, :] = self._engine.last_logits[:V]
        if sp is not None:
            return torch.tensor(out, dtype=torch.int64).reshape(N, 1)
        self._engine.adopt_next = True
        return logits

    def _images(self, tokens, plens, kwargs, skip=None):
        """Per request: Hy4Model.set_images kwargs of its images (the ViT runs here), or None (text, or skip[u]: an
        intermediate part, prefilled whole at its last part)."""
        N = int(tokens.shape[0])
        if getattr(self, "vit", None) is None or not (kwargs.get("pixel_values") or kwargs.get("pixel_values_videos")):
            return [None] * N
        from models.autoports.tencent_hy4_flash.tt.vision import request_images, request_videos

        out = []
        for u in range(N):
            imgs, vids = request_images(kwargs, u), request_videos(kwargs, u)  # videos: serving/sitecustomize.py shim
            if (imgs or vids) and skip is not None and skip[u]:
                out.append("defer")
                continue
            out.append(self.vit.prepare(tokens[u, : plens[u]].tolist(), imgs, vids) if imgs or vids else None)
        return out

    def _prefill_parts(self, tokens, plens, sp, rows, mms, start, inter):
        """A prefill step with prompt parts (vLLM chunked prefill): every part row through Hy4BatchEngine.prefill_part
        (its slot: the row's prompt in parts, else a new one), the whole short prompts packed as before. A prompt in
        parts that this step does not continue was preempted / aborted: dropped (its computed positions kept for
        prefix reuse). Image prompts: nothing until their last part, then one whole prefill."""
        N = int(tokens.shape[0])
        eng = self._engine
        V = int(eng.V)
        part = [u for u in range(N) if (inter[u] or start[u] > 0) and mms[u] is None]
        cont = {}
        for u in part:  # continuations: the slot holding the prompt's earlier parts (the plugin may move its row)
            if start[u] > 0:
                n = plens[u]
                own = [c for c, st in eng.cp.items() if c not in cont.values() and len(st["ids"]) <= n
                       and torch.equal(tokens[u, : len(st["ids"])].to(torch.long), torch.tensor(st["ids"]))]
                s = own[0] if own else self._row_slot.get(rows[u])
                if s is not None and s not in eng.seqs and (own or s not in eng.cp):
                    cont[u] = s
                    # the plugin re-picks a continuing prompt's state slot (its row here) every step: the row's previous
                    # sequence (a finished / preempted request: the plugin gives a prefill only a slot no live request
                    # holds) is gone, as on every other path that rebinds a row; unbound it would hold its engine slot
                    # forever ("all 8 slots busy", soak 2026-11-24)
                    prev = self._row_slot.get(rows[u])
                    if prev is not None and prev != s and prev in eng.seqs:
                        eng.finish(prev)
                    self._bind(rows[u], s)
        for s in [s for s in list(eng.cp) if s not in cont.values()]:
            eng.finish(s)
        whole = [u for u in range(N) if u not in part and mms[u] != "defer"]
        out, logits = [0] * N, torch.zeros((N, 1, V), dtype=torch.float32)
        if whole:
            sub = self._prefill_multi(tokens[whole], [plens[u] for u in whole], sp, [rows[u] for u in whole],
                                      [mms[u] for u in whole], taken=list(cont.values()))
            for i, u in enumerate(whole):
                if sp is not None:
                    out[u] = int(sub[i, 0])
                else:
                    logits[u] = sub[i]
        for u in part:
            row = rows[u]
            slot = cont.get(u)
            if slot is None:  # a new prompt (or one whose parts were dropped: prefix reuse continues it)
                old = self._row_slot.pop(row, None)
                if old is not None:
                    eng.finish(old)
                slot = eng.alloc(taken=[self._row_slot[rows[w]] for w in whole] + list(cont.values()),
                                 ids=tokens[u, : plens[u]].tolist(), part=bool(inter[u]))
                self._bind(row, slot)
            tok = eng.prefill_part(slot, tokens[u, : plens[u]].tolist(), final=not inter[u],
                                   sampling=_mtp_sampling(sp, u))
            if not inter[u]:
                out[u] = tok
                if sp is None:
                    eng.seqs[slot]["adopt"] = True
                    logits[u, 0, :] = eng.last_logits[slot][:V]
        if sp is not None:
            return torch.tensor(out, dtype=torch.int64).reshape(N, 1)
        return logits

    def _prefill_multi(self, tokens, plens, sp, rows, mms=None, start=None, inter=None, taken=()):
        N = int(tokens.shape[0])
        rows = list(range(N)) if rows is None else [int(r) for r in rows]
        eng = self._engine
        if inter is not None and (any(inter) or any(start) or eng.cp):
            return self._prefill_parts(tokens, plens, sp, rows, mms or [None] * N, start, inter)
        slots, reqs = [], []
        for u, row in enumerate(rows):
            old = self._row_slot.pop(row, None)  # the row's previous request is gone
            if old is not None:
                eng.finish(old)
            # (text prompts: the slot whose kept sequence shares the longest prefix, tt/engine_batch.py prefix reuse)
            text = mms is None or mms[u] is None
            slot = eng.alloc(taken=slots + list(taken), ids=tokens[u, : plens[u]].tolist() if text else None)
            self._bind(row, slot)
            slots.append(slot)
            reqs.append((slot, tokens[u, : plens[u]].tolist(), _mtp_sampling(sp, u)))
            if os.environ.get("HY4_LOG_PROMPT", "0") == "1":  # (debug: what the engine is given)
                ids = reqs[-1][1]
                logger.info(f"Hy4 prompt row {row} slot {slot}: {len(ids)} tok, hash {hash(tuple(ids)) & 0xFFFFFFFF:08x}, "
                            f"head {ids[:12]}, tail {ids[-16:]}, sampling {reqs[-1][2]}")
        if mms is not None and any(m is not None for m in mms):  # image prompts one by one, the text ones packed
            txt = [i for i, m in enumerate(mms) if m is None]
            out = [None] * len(reqs)
            for i, t in zip(txt, eng.prefill_many([reqs[i] for i in txt]) if txt else []):
                out[i] = t
            for i, m in enumerate(mms):
                if m is not None:
                    slot, ids, smp = reqs[i]
                    out[i] = eng.prefill(slot, ids, smp, mm=m)
        else:
            out = eng.prefill_many(reqs)  # simultaneous short prompts: one packed prefill pass
        if sp is not None:
            return torch.tensor(out, dtype=torch.int64).reshape(N, 1)
        for slot in slots:
            eng.seqs[slot]["adopt"] = True
        V = int(eng.V)
        logits = torch.full((N, 1, V), float("-inf"), dtype=torch.float32)
        for u, slot in enumerate(slots):
            logits[u, 0, :] = eng.last_logits[slot][:V]
        return logits

    def _check_orphans(self, where):
        """Every engine sequence is reachable from a row (decode reads slots only through _row_slot; an unreachable one
        is a leaked engine slot that ends in "all N slots busy"). Logged as an error, not repaired: a new leak path
        must show up, not be hidden."""
        orphans = set(self._engine.seqs) - set(self._row_slot.values())
        if orphans and orphans != getattr(self, "_orphans_seen", None):
            logger.error(f"Hy4 slot leak after {where}: engine slots {sorted(orphans)} have no row (rows "
                         f"{self._row_slot}, seqs {sorted(self._engine.seqs)}, cp {sorted(self._engine.cp)})")
        self._orphans_seen = orphans

    # ------------------------------------------------------------------ decode
    def _remap(self, mapping):
        """vLLM batch condense: mapping[new row] = old row (None / -1: empty)."""
        old = dict(self._row_slot)
        self._row_slot = {}
        for new, o in enumerate(torch.as_tensor(mapping).reshape(-1).tolist() if not isinstance(mapping, (list, tuple))
                                else list(mapping)):
            if o is not None and int(o) >= 0 and int(o) in old:
                self._row_slot[new] = old[int(o)]
        for slot in set(old.values()) - set(self._row_slot.values()):
            self._engine.finish(slot)

    def _decode_multi(self, tokens, pos, sp, kwargs):
        eng = self._engine
        live = [i for i in range(pos.numel()) if int(pos[i]) >= 0]
        # (a row off this step keeps its slot: freed only when the row is prefilled again; rows <= slots, so no leak)
        # slots of rows not in this step (finished / preempted in vLLM): evictable when the shared pool runs short
        eng.idle = {s for r, s in self._row_slot.items() if r not in live}
        missing = [i for i in live if i not in self._row_slot]
        if missing:
            raise RuntimeError(f"Hy4 decode rows {missing} have no prefilled slot (rows {self._row_slot})")
        if sp is None:  # host sampling: next-token logits
            if SPEC_DBG:
                _dbg_step(self, eng, live, tokens, pos, sp, kwargs, None)
            V = int(eng.V)
            logits = torch.full((tokens.numel(), 1, V), float("-inf"), dtype=torch.float32)
            res = eng.step_logits({self._row_slot[i]: (int(tokens[i]), int(pos[i])) for i in live}) if live else {}
            for i in live:
                logits[i, 0, :] = res[self._row_slot[i]][:V]
            return logits
        out = torch.full((tokens.numel(), SPEC_K + 1 if SPEC_K > 0 else 1), -1, dtype=torch.int32)
        if not live:
            return out
        auth = bool(kwargs.get("reload_inputs", False) or kwargs.get("reset_batch", False))
        if SPEC_DBG:
            _dbg_step(self, eng, live, tokens, pos, sp, kwargs, auth)
        reqs = {}
        for i in live:
            slot = self._row_slot[i]
            q = eng.seqs[slot]
            q["smp"] = q["smp"] if not auth else _resample(q["smp"], _mtp_sampling(sp, i))
            reqs[slot] = (int(tokens[i]), int(pos[i]), auth)
        res = eng.step(reqs)
        for i in live:
            ids = res[self._row_slot[i]]
            out[i, : len(ids)] = torch.tensor(ids, dtype=torch.int32)
        if self._decode_steps % STATS_EVERY == 0:
            logger.info(f"Hy4 batch engine stats: {eng.stats}")
        return out

    def decode_forward(self, *args, **kwargs):
        """(timed: the median time inside the model per decode step vs between steps, i.e. vLLM's own share, logged
        every 512 steps)"""
        t0 = time.perf_counter()
        last = getattr(self, "_t_ret", None)
        out = self._decode_forward(*args, **kwargs)
        t1 = time.perf_counter()
        self._t_ret = t1
        tl = self.__dict__.setdefault("_t_log", [])
        if last is not None and t0 - last < 1.0:  # back-to-back steps (not after an idle gap / a prefill)
            tl.append((t1 - t0, t0 - last))
            try:  # per live-row count: mean / p90 / max / spikes (HY4_STEP_STATS=1, default)
                pos_ = torch.as_tensor(kwargs.get("start_pos", args[1] if len(args) > 1 else None)).reshape(-1)
                nl = int((pos_ >= 0).sum())
                ps = self.__dict__.setdefault("_t_live", {}).setdefault(nl, [])
                ps.append(t1 - t0)
                if len(ps) >= 256:
                    v = sorted(ps)
                    md = v[len(v) // 2]
                    logger.info(f"Hy4 decode steps, {nl} live: n {len(v)} median {md * 1e3:.2f} mean "
                                f"{sum(v) / len(v) * 1e3:.2f} p90 {v[int(0.9 * len(v))] * 1e3:.2f} max {v[-1] * 1e3:.2f} ms, "
                                f"{sum(1 for x in v if x > 1.5 * md)} > 1.5x median (sum excess "
                                f"{sum(x - md for x in v if x > 1.5 * md) * 1e3:.0f} ms)")
                    ps.clear()
            except Exception:  # noqa: BLE001
                pass
        if CP_LOG and getattr(self._engine, "cp", None):
            logger.info(f"Hy4 decode step (a prompt in parts): {(t1 - t0) * 1e3:.1f} ms in the model, "
                        f"{(t0 - last) * 1e3 if last else 0:.1f} ms before")
        if len(tl) >= 512:
            med = [sorted(x[i] for x in tl)[len(tl) // 2] * 1e3 for i in (0, 1)]
            logger.info(f"Hy4 decode steps (median of {len(tl)}): {med[0]:.2f} ms in the model, {med[1]:.2f} ms between "
                        f"steps (vLLM)")
            tl.clear()
        return out

    def _decode_forward(self, *args, **kwargs):
        tokens = torch.as_tensor(kwargs.get("tokens", args[0] if args else None)).reshape(-1)
        pos = torch.as_tensor(kwargs.get("start_pos", args[1] if len(args) > 1 else None)).reshape(-1)
        sp = kwargs.get("sampling_params", args[6] if len(args) > 6 else None)
        if self.multi:
            remap = kwargs.get("slot_remap", args[10] if len(args) > 10 else None)
            if remap is not None:
                self._remap(remap)
                self._check_orphans("remap")
            self._decode_steps += 1
            return self._decode_multi(tokens, pos, sp, kwargs)
        live = [i for i in range(pos.numel()) if int(pos[i]) >= 0]
        assert len(live) <= 1, "single-user Hy4 step got several live rows"
        self._decode_steps += 1
        if sp is None:  # host sampling: next-token logits
            V = int(self._engine.V)
            logits = torch.full((tokens.numel(), 1, V), float("-inf"), dtype=torch.float32)
            for i in live:
                logits[i, 0, :] = self._engine.step_logits(int(tokens[i]), int(pos[i]))[:V]
            return logits
        out = torch.full((tokens.numel(), max(1, SPEC_K + 1) if SPEC_K > 0 else 1), -1, dtype=torch.int32)
        authoritative = bool(kwargs.get("reload_inputs", False) or kwargs.get("reset_batch", False))
        for i in live:
            ids = self._engine.step(int(tokens[i]), int(pos[i]), host_authoritative=authoritative,
                                    sampling=_mtp_sampling(sp, i))
            ids = ids if SPEC_K > 0 else ids[:1]
            out[i, : len(ids)] = torch.tensor(ids, dtype=torch.int32)
        if self._decode_steps % 1024 == 0:
            logger.info(f"Hy4 engine stats: {self._engine.stats}")
        return out

    def read_decode_output(self, tt_out, async_read=False):
        return (tt_out, []) if async_read else tt_out

    def process_decode_output_host(self, tt_out, is_tokens=True):
        return tt_out


if _VISION:
    HYV4FlashForConditionalGeneration = _hy4_mm.register(HYV4FlashForConditionalGeneration)

__all__ = ["HYV4FlashForConditionalGeneration", "HF_MODEL"]
