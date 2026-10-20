# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Generator of the full GLM-5.3-Flash model (stage 6): the readiness-check / serving contract
(readiness_check.contract.Generator) over tt/model.Glm53Model.

Decode is split into two traces, replayed back to back without host work in between:
  1. the model trace: per-step inputs derived on device from the position tensor (tt/decode_inputs.py), 45 layers,
     head -> logits [1,1,S,V/tp]; positions advanced at its end;
  2. the sampler trace (models.common.sampling.SamplingGenerator, semantically greedy params): token written into the
     persistent token buffer `tok_buf` that the model trace reads (tt_out_tok), so step N's sample is step N+1's input.
The host writes tokens / positions only when it is authoritative: right after a prefill, when teacher forcing overrides
the sampled token, or in the host-sampling compatibility mode (on_device_sampling=False: logits to the host, host
argmax).

Cache ownership: the model owns its caches (DSA latent pool per layer, KDA state per slot). `kv_cache` arguments of the
low-level API are accepted for the contract and must be None or `self.kv_cache`; `page_table` [S, blocks] maps the
DSA latent pool blocks of every slot.
"""

from __future__ import annotations

import os

import torch
from loguru import logger

import ttnn
from models.autoports.zai_org_glm_5_3_flash.tt.model import Glm53Model, model_from_env

try:
    from readiness_check.contract import Generator
except ImportError:  # the readiness package is optional at import time (unit tests)

    class Generator:  # noqa: D101
        tokenizer = None


L1_SMALL_SIZE = 24576
# ------------------------------------------------------------------------------------------------ DRAM budget
# Per-die DRAM budget of the full 45-layer model on p300x2 (TP4). The data source is serving_profiles.json
# (tt/serving_profiles.py: the single sizing rule, written from measured capacity probes) when its precision id is the
# build's (tt/precision.PrecisionOverrides), else the built-in measurement of the shipped bf16-latent policy below.
# DSA bytes per block follow the latent dtype (bf16 / fp8_e4m3). Used to choose the DSA cache layout and the default
# context, and to refuse a (slots, context) build that cannot fit, before any weight is loaded; the standalone
# generator re-checks against the free DRAM it measures after loading the weights (Glm53Generator._allocate_pool).
DSA_LAYERS, BLOCK_TOKENS, LATENT, POOL_KEY = 11, 128, 512, 128


def _latent_dtype() -> str:
    try:
        from .precision import PrecisionOverrides

        return str(PrecisionOverrides.from_env().kv_latent_dtype or "bf16")
    except Exception:  # noqa: BLE001
        return "bf16"


def _profiles_budget():
    try:
        from . import serving_profiles as sp
        from .precision import PrecisionOverrides

        data = sp.load()
        if (data.get("precision") or {}).get("id") == PrecisionOverrides.from_env().id:
            return data
    except Exception:  # noqa: BLE001
        pass
    return None


def _budget() -> dict:
    lat = _latent_dtype()
    lat_bytes = 1 if lat == "fp8_e4m3" else 2
    blk_rep = DSA_LAYERS * BLOCK_TOKENS * (LATENT * lat_bytes + POOL_KEY * 2 // 4)
    blk_shard = DSA_LAYERS * (BLOCK_TOKENS * 544 * 2 // 4 + 32 * POOL_KEY * 2)  # tt/dsa_shard.py (bf16 latent only)
    data = _profiles_budget()
    if data is not None:
        b = data["budget"]
        free, kda, reserve = int(b["free_after_load_bytes"]), int(b["kda_bytes_per_slot"]), int(b["reserve_bytes"])
        src = f"serving_profiles.json ({data['precision']['id']})"
    else:
        # shipped policy, stage-7 code: s4r capacity probe 2026-10-17 (64 MiB trace region, 2 KDA slots)
        free, kda, reserve, src = 1_984_045_568, 35_651_584, 734_003_200, "built-in (shipped bf16 latent policy)"
    # reserve: the larger of the data source's and GLM53_DRAM_RESERVE_MIB (default 1200 MiB: headroom above the measured
    # 1,024 MiB prefill workspace; lower it to trade safety margin for context, doc/full_model/README.md "Defaults")
    reserve = max(reserve, int(float(os.environ.get("GLM53_DRAM_RESERVE_MIB", 1200)) * (1 << 20)))
    return {
        "free_after_load_bytes": int(os.environ.get("GLM53_BUDGET_FREE_BYTES", free)),
        "kda_bytes_per_slot": kda,
        "reserve_bytes": reserve,
        "block_bytes": {"replicated": blk_rep, **({"sharded": blk_shard} if lat_bytes == 2 else {})},
        "latent_dtype": lat,
        "source": src,
    }


BUDGET = _budget()


def _trace_region_default() -> int:
    data = _profiles_budget()
    return int(data.get("trace_region_bytes", 64 << 20)) if data is not None else 64 << 20


TRACE_REGION_SIZE = int(os.environ.get("GLM53_TRACE_REGION", _trace_region_default()))  # decode needs ~46 MiB
LAYOUTS = ("replicated", "sharded")  # fastest first (sharded: +~4.6 ms per decode token, tt/dsa_shard.py)


def _kda_slots(S: int) -> int:
    return S + S % 2  # MultichipKDA pads the decode batch to fill tile rows (16 local heads): an odd S gets one more


def pool_blocks(S: int, layout: str, free_bytes: int | None = None) -> int:
    """DSA pool blocks (incl. the dummy block) that fit per die next to S slots' KDA state and the reserve."""
    b = BUDGET
    free = b["free_after_load_bytes"] if free_bytes is None else free_bytes
    avail = free - (_kda_slots(S) - 2) * b["kda_bytes_per_slot"] - b["reserve_bytes"]
    if layout not in b["block_bytes"]:  # the block-sharded layout supports the bf16 latent only
        return 0
    return max(0, avail // b["block_bytes"][layout])


def blocks_needed(S: int, T: int, serving: bool = False) -> int:
    """Standalone: every slot owns T/128 blocks + the dummy block. Serving (vLLM pool): one request of T tokens, one
    block for each other slot, vLLM's null block and the dummy block."""
    return (T // 128 + (S - 1) + 2) if serving else (S * (T // 128) + 1)


def context_granularity(T: int) -> int:
    """max_seq_len must be a multiple of the largest prefill length that fits it (functional_decoder asserts it)."""
    return 2048 if T >= 2048 else (512 if T >= 512 else 128)


def max_context(S: int, layout: str, serving: bool = False) -> int:
    """Largest max_seq_len of S slots in `layout` (a multiple of context_granularity)."""
    nb = pool_blocks(S, layout)
    t = ((nb - 2 - (S - 1)) if serving else (nb - 1) // S) * 128
    return 0 if t < 128 else t // context_granularity(t) * context_granularity(t)


def plan_context(S: int, T: int | None = None, layout: str | None = None, serving: bool = False) -> dict:
    """(slots, max_seq_len) -> {"layout", "max_seq_len", "dsa_shard", ...}. T None: the default context of S slots (the
    replicated layout's largest if >= 8192 tokens, else the larger of the two). layout None: the fastest layout that
    fits (GLM53_DSA_SHARD=0/1 forces one). Raises ValueError when the request cannot fit."""
    env = os.environ.get("GLM53_DSA_SHARD", "").strip().lower()
    if layout is None and env:
        layout = "sharded" if env in ("1", "true", "on", "yes") else "replicated"
    caps = {L: max_context(S, L, serving) for L in LAYOUTS}
    if T is None:
        if layout is not None:
            T = caps[layout]
        else:
            T = caps["replicated"] if caps["replicated"] >= 8192 else max(caps.values())
        if T <= 0:
            raise ValueError(f"{S} slots do not fit in device DRAM in any DSA layout (budget {BUDGET})")
    T = int(T)
    g = context_granularity(T)
    if T % g:
        raise ValueError(f"max_seq_len {T} must be a multiple of {g} (the largest prefill chunk length that fits it)")
    cands = [layout] if layout is not None else list(LAYOUTS)
    for L in cands:
        if blocks_needed(S, T, serving) <= pool_blocks(S, L):
            return {
                "layout": L,
                "dsa_shard": L == "sharded",
                "max_seq_len": T,
                "slots": S,
                "blocks_needed": blocks_needed(S, T, serving),
                "pool_blocks_budget": pool_blocks(S, L),
                "max_seq_len_per_layout": caps,
            }
    raise ValueError(
        f"max_batch_size {S} x max_seq_len {T} does not fit in device DRAM ({'/'.join(cands)} DSA layout"
        f"{'' if layout else 's'}): needs {blocks_needed(S, T, serving)} DSA blocks, the budget holds "
        + ", ".join(f"{L} {pool_blocks(S, L)} (max_seq_len <= {caps[L]})" for L in cands)
        + " per die (tt/generator.BUDGET; doc/full_model/memory_budget.json)"
    )


SHARED_FREES: dict = {}  # GLM53_DEALLOC_CHECK=1: call site -> count of force-frees of a buffer another tensor shares


def install_dealloc_check():
    """Debug mode (GLM53_DEALLOC_CHECK=1, installed at import): ttnn.deallocate runs with force=False and records every
    call that leaves the tensor allocated, i.e. one whose default force=True would have freed storage that another
    tensor (a view: reshape, in-tile pad, full slice) still uses. That bug class corrupted decode (stage-6/7 rework:
    the decode logits pad). tests/test_model_probe.py fails on any such site in the autoport's own tt/ modules."""
    import traceback

    real = ttnn.deallocate
    if getattr(real, "_glm53_check", False):
        return

    def dealloc(t, *a, **k):
        k["force"] = False
        r = real(t, *a, **k)
        if isinstance(t, ttnn.Tensor) and t.is_allocated():
            st = [f for f in traceback.extract_stack()[:-1] if "/models/" in f.filename]
            key = " <- ".join(
                f"{os.path.relpath(f.filename, os.path.dirname(__file__))}:{f.lineno}" for f in st[-3:][::-1]
            )
            SHARED_FREES[key] = SHARED_FREES.get(key, 0) + 1
        return r

    dealloc._glm53_check = True
    ttnn.deallocate = dealloc


if os.environ.get("GLM53_DEALLOC_CHECK") == "1":
    install_dealloc_check()


def poison_free_dram(mesh, value=float("nan"), chunk_mb: int = 256, leave_mb: int = 32) -> int:
    """Diagnostic (GLM53_POISON_DRAM=1): fill the free DRAM of every die with `value` (allocate chunks of ttnn.full,
    then free them; the memory keeps the pattern), so a later read of memory that was never written shows up as NaN
    instead of the zeros a fresh device happens to hold. Returns the bytes poisoned per die."""
    ts, done = [], 0
    rows = chunk_mb * (1 << 20) // (1024 * 2) // 32 * 32
    while True:
        v = ttnn.get_memory_view(mesh, ttnn.BufferType.DRAM)
        largest = int(v.largest_contiguous_bytes_free_per_bank) * int(v.num_banks)
        n = min(rows, (largest - leave_mb * (1 << 20)) // 2048 // 32 * 32)
        if n < 32:
            break
        try:
            ts.append(ttnn.full((1, 1, n, 1024), value, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh))
        except Exception:  # noqa: BLE001
            break
        done += n * 2048
    ttnn.synchronize_device(mesh)
    for t in ts:
        ttnn.deallocate(t)
    logger.info(f"poisoned {done / 2**20:.0f} MB of free DRAM per die with {value}")
    return done


class Glm53Generator(Generator):
    def __init__(
        self,
        mesh,
        *,
        max_batch_size: int = 1,
        max_seq_len: int | None = None,
        defer_caches: bool = False,
        dsa_layout: str | None = None,
        budget_check: bool = True,
        **model_kw,
    ):
        """defer_caches=True (vLLM serving, tt/generator_vllm.py): load the weights and the KDA slot state only. The DSA
        pools come from bind_kv_cache(num_blocks) (vLLM's block count); warm-up and trace capture from
        warm_programs() / capture_traces() (the plugin's two warm-up phases). False = the standalone generator.
        max_seq_len None = the default context of max_batch_size slots (plan_context); dsa_layout None = the fastest DSA
        cache layout that holds the request (replicated, else block-sharded), "replicated" / "sharded" forces one. A
        request that fits in neither raises ValueError before any weight is loaded (budget_check=False skips the
        budget: capacity probes that size the pool from the measured free DRAM themselves)."""
        self.mesh = mesh
        self.max_batch_size = max_batch_size
        self.defer_caches = defer_caches
        full = "layer_indices" not in model_kw and not os.environ.get("GLM53_LAYERS")
        if full and tuple(mesh.shape) == (1, 4) and budget_check:  # the budget is the full model's on p300x2
            self.plan = plan_context(max_batch_size, max_seq_len, dsa_layout, serving=defer_caches)
        else:  # reduced / other meshes: no budget; layout from dsa_layout / GLM53_DSA_SHARD
            T = max_seq_len or 4096
            env = os.environ.get("GLM53_DSA_SHARD", "").strip().lower() in ("1", "true", "on", "yes")
            self.plan = {"layout": dsa_layout or ("sharded" if env else "replicated"), "max_seq_len": T}
            self.plan["dsa_shard"] = self.plan["layout"] == "sharded"
        max_seq_len = self.plan["max_seq_len"]
        logger.info(f"generator plan: {self.plan}")
        model_kw.setdefault("dsa_shard", self.plan["dsa_shard"])
        dp = os.environ.get("GLM53_DEFER_POOL", "0")  # 0 = HEAD (pool per layer at construction); 1 = after the
        defer_pool = defer_caches or dp == "1"  # weights (_allocate_pool, then the sampler: Glm53Model._init_sampler)
        model_kw.setdefault("allocate_dsa", "early" if (dp == "2" and not defer_caches) else not defer_pool)
        model_kw.setdefault("lockstep_sampling", os.environ.get("GLM53_LOCKSTEP", "1") == "1")
        if "prefill_lengths" not in model_kw and not os.environ.get("GLM53_PREFILL_LENGTHS"):
            g = context_granularity(max_seq_len)
            model_kw["prefill_lengths"] = tuple(n for n in (128, 512, 2048) if n <= g)
        if os.environ.get("GLM53_POISON_DRAM") == "1":
            poison_free_dram(mesh)
        self.model: Glm53Model = model_from_env(mesh, max_batch=max_batch_size, max_context=max_seq_len, **model_kw)
        self.max_seq_len = max_seq_len
        self.kv_cache = None  # model-owned (see module docstring)
        self.tokenizer = self._load_tokenizer()
        S = max_batch_size
        self.page_table = None  # defer_caches: bind_kv_cache installs an all-null table
        if not defer_caches:
            if defer_pool:
                self._allocate_pool()
            dsa = next((d.attn for d in self.model.layers if d.attn_kind == "dsa"), None)
            self.page_table = dsa.default_page_table()[:, : max_seq_len // 128] if dsa is not None else None
        if self.page_table is not None:
            self.model.set_page_table(self.page_table)
        self.page_table_copies = 0  # device page-table installs after construction (set_page_table on change only)
        self.tok_buf = ttnn.from_torch(
            torch.zeros(1, 1, 1, self.model.SR, dtype=torch.int32),  # sampler width (>= S): the model reads [:S]
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        self.trace_id = None
        self.logits_out = None
        self.host_trace_id = None
        self._sampler_ready = False
        self.positions = [-1] * S
        # serving state (defer_caches mode)
        self.num_blocks = None
        self._fresh = set()  # slots prefilled since the last serving decode (their host token is authoritative)
        self._serving_decoded = False
        self._sstate = None
        self._warmed_programs = False
        self._program_cache_after_capture = None
        if self.model.sampling is not None:
            from models.autoports.zai_org_glm_5_3_flash.tt.serving_state import SlotSamplingState

            self._sstate = SlotSamplingState(self.model.sampling, self.model.SR)
        if os.environ.get("GLM53_POISON_DRAM") == "1":  # again after the build (weight staging freed, pool allocated)
            poison_free_dram(mesh)
        self._ballast = None
        mb = os.environ.get("GLM53_BALLAST_MB")
        if os.environ.get("GLM53_BALLAST_LEAVE_MB"):  # ballast that leaves this much DRAM free per die
            v = ttnn.get_memory_view(mesh, ttnn.BufferType.DRAM)
            free = int(v.total_bytes_free_per_bank) * int(v.num_banks)
            mb = max(0.0, (free - float(os.environ["GLM53_BALLAST_LEAVE_MB"]) * (1 << 20)) / (1 << 20))
            logger.info(f"ballast {mb:.0f} MB (free {free / 2**20:.0f} MB)")
        if mb and os.environ.get("GLM53_BALLAST_AFTER") != "1":  # diagnostic: DRAM pressure without a larger pool
            n = int(float(mb) * (1 << 20)) // (2 * 32 * 1024) * 32
            self._ballast = ttnn.zeros(
                (1, 1, n, 1024),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        if not defer_caches:
            self._warm()
        if mb and os.environ.get("GLM53_BALLAST_AFTER") == "1":  # diagnostic: the ballast after warm-up / capture
            n = int(float(mb) * (1 << 20)) // (2 * 32 * 1024) * 32
            self._ballast = ttnn.zeros(
                (1, 1, n, 1024),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

    # ------------------------------------------------------------------------------------------------ setup
    def _allocate_pool(self):
        """Standalone generator: the DSA pool of every slot at max_seq_len (S * T/128 blocks + the dummy block),
        checked against the DRAM actually free after the weights (a clear error instead of an OOM later)."""
        S, T, m = self.max_batch_size, self.max_seq_len, self.model
        nb = S * (T // 128) + 1
        dsa = [d.attn for d in m.layers if d.attn_kind == "dsa"]
        if dsa:
            from .dsa_shard import dsa_bytes_per_token

            need = int(round(128 * dsa_bytes_per_token(dsa))) * nb
            try:
                v = ttnn.get_memory_view(self.mesh, ttnn.BufferType.DRAM)
                free = int(v.total_bytes_free_per_bank) * int(v.num_banks)
            except Exception as e:  # noqa: BLE001
                logger.warning(f"memory view unavailable ({e}): DSA pool not checked against free DRAM")
                free = None
            self.plan["free_after_load_bytes"] = free
            self.plan["pool_bytes"] = need
            if free is not None and need + BUDGET["reserve_bytes"] > free:
                raise ValueError(
                    f"max_batch_size {S} x max_seq_len {T}: the DSA pool ({nb} blocks, {need / 2**30:.2f} GiB per die, "
                    f"{self.plan['layout']} layout) plus the {BUDGET['reserve_bytes'] / 2**20:.0f} MiB reserve exceeds "
                    f"the {free / 2**30:.2f} GiB free per die after loading; use a shorter max_seq_len or fewer slots "
                    f"(tt/generator.plan_context)"
                )
        m.allocate_dsa_caches(nb)

    def _load_tokenizer(self):
        try:
            from transformers import AutoTokenizer

            return AutoTokenizer.from_pretrained(str(self.model.rd.path), trust_remote_code=True)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"tokenizer unavailable ({e})")
            return None

    def _tok_view(self):
        t = ttnn.reshape(self.tok_buf, (1, self.model.SR))
        return t if self.model.SR == self.max_batch_size else ttnn.slice(t, (0, 0), (1, self.max_batch_size))

    def _warm(self):
        """Compile everything before any trace capture, with every slot inactive (no state / cache changes): one eager
        decode step, the head, the sampler programs; then capture the model trace."""
        # one prefill per constructed length first: prefill builds persistent constants lazily (DSA per-length tables,
        # ...). Built after the capture, they land in freed trace-intermediate memory and the next decode replay
        # overwrites them (a prefill after replays then hung, 2026-10-16). Slot 0's state is zeroed again below.
        self.model.choose_moe_split()
        for L, sizes in self._warm_plans():
            self.model.prefill(torch.zeros(L, dtype=torch.int64), 0, sizes=sizes)
        self._warm_extra()
        self.model.reset_state()
        self.model.prepare_prefill_traces()  # persistent inputs before any capture
        self.model.inputs.set_positions([-1] * self.max_batch_size)
        logits = self.model.decode_step(self._tok_view())
        if self.model.sampling is not None:
            self._warm_sampler(logits)
        ttnn.deallocate(logits)
        ttnn.synchronize_device(self.mesh)
        if os.environ.get("GLM53_POISON_PRECAPTURE") == "1":  # diagnostic: freed-but-referenced buffers read NaN
            poison_free_dram(self.mesh)
        self.trace_id = ttnn.begin_trace_capture(self.mesh, cq_id=0)
        try:
            self.logits_out = self.model.decode_step(self._tok_view())
        finally:
            ttnn.end_trace_capture(self.mesh, self.trace_id, cq_id=0)
        # capture the sampler traces on the model trace's logits buffer now: a first capture inside decode_forward,
        # right after a non-blocking model replay, hung the probe test (2026-10-16). Writes a junk token into
        # tok_buf; the first decode step after a prefill always reloads the tokens.
        self._capture_sampler()
        ttnn.synchronize_device(self.mesh)
        self.model.capture_prefill_traces()  # last: their outputs are consumed by each prefill before any decode replay
        self.model.reset_state()
        self.model.inputs.set_positions([-1] * self.max_batch_size)
        ttnn.synchronize_device(self.mesh)
        self._program_cache_after_capture = self.program_cache_entries()
        self._check_trace_buffers()
        self._trace_addr = self.trace_addresses()
        logger.info(f"decode traces captured (model + sampler variants); prefill traces {sorted(self.model._ptraces)}")

    def _trace_tensors(self) -> dict:
        """Every persistent device tensor a decode replay reads or writes: token / logits / position / page-table /
        per-step input buffers, each layer's KDA state and DSA caches + open-pool accumulators."""
        m, inp = self.model, self.model.inputs
        out = {"tok_buf": self.tok_buf, "logits_out": self.logits_out, "pos": inp.pos}
        if inp.dsa_bufs is not None:
            out["page_table"] = inp.pt
            out.update({f"dsa_in.{k}": v for k, v in inp.dsa_bufs.items()})
        if isinstance(inp.kda_bufs, dict):
            out.update({f"kda_in.{k}": v for k, v in inp.kda_bufs.items()})
        for i, d in zip(m.layer_indices, m.layers):
            a = d.attn
            if d.attn_kind == "kda" and d.decode_state is not None:
                out[f"l{i}.recurrent"] = d.decode_state.recurrent
                out.update({f"l{i}.conv{j}": c for j, c in enumerate(d.decode_state.conv)})
            elif d.attn_kind == "dsa":
                for k in ("latent", "pooled", "page_table", "acc_m", "acc_den", "acc_num", "lpt"):
                    if getattr(a, k, None) is not None:
                        out[f"l{i}.{k}"] = getattr(a, k)
                tail = a._shared_tail() if hasattr(a, "_shared_tail") else None  # stage-5 dsa_tail_shared step buffer
                if tail is not None:
                    out["dsa_shared_tail"] = tail
        return {k: v for k, v in out.items() if isinstance(v, ttnn.Tensor)}

    def trace_addresses(self) -> dict:
        return {k: (t.buffer_address() if t.is_allocated() else None) for k, t in self._trace_tensors().items()}

    def check_trace_addresses(self, where: str = ""):
        """Address-stability guard: every buffer the decode trace reads / writes must keep the address it had at the
        capture (a persistent tensor replaced after the capture leaves the trace on freed memory)."""
        ref = getattr(self, "_trace_addr", None)
        if ref is None or os.environ.get("GLM53_TRACE_ADDR_CHECK", "1") != "1":
            return
        now = self.trace_addresses()
        bad = {k: (ref[k], now.get(k)) for k in ref if now.get(k) != ref[k]}
        if bad:
            raise RuntimeError(f"decode trace buffers moved or were freed {where}: {dict(list(bad.items())[:6])}")

    def _check_trace_buffers(self):
        """Every persistent buffer a decode replay reads or writes must still be allocated after warm-up / capture
        (a shared-code helper that deallocates a caller's tensor would leave the trace on recycled memory). Note: a
        buffer freed through a view can still report allocated; the penalty-count clone (Glm53Model._init_sampler)
        is the guard for the known case."""
        inp = self.model.inputs
        bufs = {"tok_buf": self.tok_buf, "logits_out": self.logits_out, "pos": inp.pos}
        if inp.dsa_bufs is not None:
            bufs["page_table"] = inp.pt
            bufs.update({f"dsa.{k}": v for k, v in inp.dsa_bufs.items()})
        if inp.kda_bufs is not None:
            bufs.update({f"kda.{k}": v for k, v in (inp.kda_bufs.items() if isinstance(inp.kda_bufs, dict) else [])})
        bad = [k for k, t in bufs.items() if isinstance(t, ttnn.Tensor) and not t.is_allocated()]
        assert not bad, f"decode trace buffers freed during warm-up / capture: {bad}"

    def _warm_sampler(self, logits):
        """Compile every sampler program a captured variant records, outside any capture: greedy params, every
        (penalties, log-probs, force-argmax) combination (SamplingGenerator.precompile(all_configs=True)), plus the
        penalty TOKEN-COUNT update. precompile runs penalties with count_tokens=False, but the capture records
        count_tokens=True (TTPenalties.update_output_tokens: scatter_add / tilize / ...), so capturing the penalties
        variant compiled those programs inside the capture: "Cannot load new binaries during trace capture" (reduced
        probe, 2026-10-17). The scratch pass counts one junk token; the counters are zeroed after it."""
        from models.common.sampling.generator import format_sampling_params
        from models.common.sampling.sampling_params import SamplingParams

        sm = self.model.sampling
        sm.reset_sampling_params(
            format_sampling_params(SamplingParams(temperature=0.0, top_k=1, top_p=1.0), self.model.SR)
        )
        if os.environ.get("GLM53_SAMPLER_WARM", "all") != "all":
            sm.precompile(logits, tt_out_tok=self.tok_buf, all_configs=False)
            self._sampler_ready = True
            return
        sm.precompile(logits, tt_out_tok=self.tok_buf, all_configs=True)
        saved = sm._penalties_active
        scratch = ttnn.clone(logits)  # TTPenalties.apply rewrites its input in place
        try:
            sm._penalties_active = True
            sm._run_sampling(scratch, penalties_on=True, tt_out_tok=self.tok_buf, count_tokens=True)
        finally:
            sm._penalties_active = saved
            ttnn.deallocate(scratch)
        sm.tt_penalties.reset_output_tokens()
        self._sampler_ready = True

    def _capture_sampler(self):
        """One sampler trace per variant (penalties off / on x force-argmax off / on where allowed; device log-probs
        are never requested on this mesh) on the model trace's logits, writing into tok_buf."""
        sm = self.model.sampling
        if sm is None:
            return
        ts = sm.tt_sampling
        saved = (sm._penalties_active, ts._force_argmax_sampling)
        variants = os.environ.get("GLM53_SAMPLER_VARIANTS", "all")
        try:
            for penalties in (False, True) if variants == "all" else (False,):
                for fa in (False, True):
                    if fa and not getattr(ts, "_allow_force_argmax_sampling", False):
                        continue
                    sm._penalties_active, ts._force_argmax_sampling = penalties, fa
                    _, slot = sm._trace_slot(penalties, False, fa)
                    if slot["id"] is None:
                        sm.capture_trace(self.logits_out, tt_out_tok=self.tok_buf, skip_precompile=True)
        finally:
            sm._penalties_active, ts._force_argmax_sampling = saved
        if sm._penalties_active:
            sm.tt_penalties.reset_output_tokens()

    # ------------------------------------------------------------------------------------------------ serving (vLLM)
    def program_cache_entries(self):
        """Compiled programs on the mesh (None if the binding is missing): must not grow after capture_traces()."""
        fn = getattr(self.mesh, "num_program_cache_entries", None)
        try:
            return int(fn()) if fn is not None else None
        except Exception:  # noqa: BLE001
            return None

    def bind_kv_cache(self, num_blocks: int):
        """defer_caches mode: allocate every DSA layer's pools for vLLM's `num_blocks` blocks + the model's dummy block
        (id num_blocks), install an all-null page table (block 0 = vLLM's null block). Returns the cache object that
        vLLM passes back as kv_cache: [(latent, pooled)] per DSA layer."""
        assert self.defer_caches and self.kv_cache is None, "bind_kv_cache: defer_caches mode, once"
        self.num_blocks = int(num_blocks)
        self.kv_cache = self.model.allocate_dsa_caches(self.num_blocks + 1)  # also builds the sampler (Glm53Model)
        if self._sstate is None and self.model.sampling is not None:
            from models.autoports.zai_org_glm_5_3_flash.tt.serving_state import SlotSamplingState

            self._sstate = SlotSamplingState(self.model.sampling, self.model.SR)
        if any(d.attn_kind == "dsa" for d in self.model.layers):
            self.page_table = torch.zeros(self.max_batch_size, self.max_seq_len // 128, dtype=torch.int32)
            self.model.set_page_table(self.page_table)
        self.model.inputs.set_positions([-1] * self.max_batch_size)
        return self.kv_cache

    def _warm_plans(self):
        """Prefill warm-up prompts [(length, chunk sizes)]: every (chunk length, DSA path) pair a prefill chunk plan
        (Glm53Model.chunk_plan) can produce, before any capture (lazily built prefill constants must exist first):
        flash = each length from position 0; dense = a chunk at start n (n < top) and the smallest length after the
        top one ending at position <= 2050; sparse = each length after the top one."""
        lens = list(self.model.prefill_lengths)
        top, T = max(lens), self.max_seq_len
        plans = [(min(n, T), [n]) for n in lens]
        plans += [(2 * n, [n, n]) for n in lens if n < top and 2 * n <= T]
        plans += [(top + 2, [top, lens[0]])] if len(lens) > 1 and top + 2 <= T else []
        plans += [(top + n, [top, n]) for n in lens if top + n <= T]
        return [(L, sz) for L, sz in plans if self.num_blocks is None or -(-L // 128) < self.num_blocks]

    def _warm_extra(self):
        """Programs of the public prefill paths the length warm-up does not reach, before any capture (a program first
        run after the decode capture allocates its kernel binaries / buffers then, which under DRAM pressure land in
        the trace's freed intermediates; rework 2026-10-18): a non-aligned prompt (open pool at length % 4 != 0, the
        last row inside a tile) and the all-logits head (prefill check / prefill_logits)."""
        if os.environ.get("GLM53_WARM_EXTRA", "1") != "1":
            return
        m = self.model
        lens = m.prefill_lengths
        # a prompt ending inside a pool for every chunk length (open-pool hand-off, GlmDSA._set_open_pool), a last
        # row inside a tile, and the all-logits path
        for L in sorted({min(n - 1, self.max_seq_len) for n in lens} | {min(lens[0] + 33, self.max_seq_len)}):
            if L > 0 and (self.num_blocks is None or -(-L // 128) < self.num_blocks):
                m.prefill(torch.zeros(L, dtype=torch.int64), 0)
                if L < lens[0]:
                    m.prefill(torch.zeros(L, dtype=torch.int64), 0, all_logits=True)
        # programs whose slice offsets depend on the request (prompt length, slot): every value, before any capture
        # (a program first built after the capture allocates program-cache buffers while the trace is live)
        kda = next((d for d in m.layers if d.attn_kind == "kda"), None)
        if kda is not None:  # KDA commit: slice / concat at the slot index (one program set per slot)
            st = kda.attn.new_prefill_state()
            for s in range(self.max_batch_size):
                kda.commit_prefill(s, st)
        for n in lens:  # last-row tile of a chunk (row offset r0) and the row inside the head's 32-row tile
            X = m.embed(m.tokens_dev(torch.zeros(n, dtype=torch.int64)))
            W = int(X.shape[-1])
            for r0 in range(0, n, 32):
                ttnn.deallocate(ttnn.slice(X, (0, 0, r0, 0), (1, 1, r0 + 32, W), memory_config=ttnn.DRAM_MEMORY_CONFIG))
            if n <= 512:  # all-logits head of a chunk (prefill check / prefill_logits, test-only API); a 2048-row head
                ttnn.deallocate(m.head(X))  # holds 158 MB of logits per die: kept out of the warm-up peak
            if n == lens[0]:
                Xl = ttnn.slice(X, (0, 0, 0, 0), (1, 1, 32, W), memory_config=ttnn.DRAM_MEMORY_CONFIG)
                lg = m.head(Xl)
                ttnn.deallocate(Xl)
                for row in range(32):
                    m._row_to_host(ttnn.clone(lg), row)
                    ttnn.deallocate(m._rows_of_last(ttnn.clone(lg), row))
                ttnn.deallocate(lg)
            ttnn.deallocate(X)

    def _warm_lengths(self):
        """Lengths of the warm-up prompts (_warm_plans)."""
        return [L for L, _ in self._warm_plans()]

    def warm_programs(self):
        """Plugin warm-up phase 1 (no capture, idempotent): every prefill path, the prefill sampler, one eager decode
        step, the head and every sampler variant, with every slot left inactive afterwards."""
        if self._warmed_programs:
            return
        assert self.kv_cache is not None or not self.defer_caches, "bind_kv_cache before warm_programs"
        S = self.max_batch_size
        if self.page_table is not None:
            warm_pt = self.page_table.clone()
            nb = warm_pt.shape[1]
            if self.num_blocks is not None:  # free blocks 1.. (no request is live during the warm-up)
                n = min(nb, self.num_blocks - 1)
                warm_pt[0, :n] = torch.arange(1, n + 1, dtype=torch.int32)
            self.model.set_page_table(warm_pt)
        self.model.choose_moe_split()
        for L, sizes in self._warm_plans():
            self.model.prefill(torch.zeros(L, dtype=torch.int64), 0, sizes=sizes)
        self._warm_extra()
        logits = None
        if self.model.sampling is not None:
            from models.common.sampling.generator import format_sampling_params
            from models.common.sampling.sampling_params import SamplingParams

            greedy = format_sampling_params(SamplingParams(temperature=0.0, top_k=1, top_p=1.0), self.model.SR)
            self.model.sampling.reset_sampling_params(greedy)
            # the prefill sampler: last-row device logits [1,1,S,V/tp] -> eager sample into a fresh tensor
            lg = self.model.prefill(torch.zeros(min(100, self.max_seq_len), dtype=torch.int64), 0, device_logits=True)
            tt_tok, _ = self.model.sampling.sample(lg, enable_trace=False, count_tokens=False)
            ttnn.deallocate(lg)
        self.model.reset_state()
        if self.page_table is not None:
            self.model.set_page_table(self.page_table)
        self.model.prepare_prefill_traces()  # persistent inputs before any capture
        self.model.inputs.set_positions([-1] * S)
        logits = self.model.decode_step(self._tok_view())
        self.model.logits_to_host(logits)  # host-logits compatibility mode
        if self.model.sampling is not None:
            self._warm_sampler(logits)
        ttnn.deallocate(logits)
        self.model.inputs.set_positions([-1] * S)
        ttnn.synchronize_device(self.mesh)
        self._warmed_programs = True

    def capture_traces(self):
        """Plugin warm-up phase 2 (idempotent): the model decode trace, then one sampler trace per variant (penalties
        off / on x force-argmax off / on where allowed; device log-probs are never requested on this mesh). Nothing
        may be captured after this (a later capture after a non-blocking replay hung, 2026-10-16)."""
        if self.trace_id is not None:
            return
        self.warm_programs()
        ttnn.synchronize_device(self.mesh)
        self.trace_id = ttnn.begin_trace_capture(self.mesh, cq_id=0)
        try:
            self.logits_out = self.model.decode_step(self._tok_view())
        finally:
            ttnn.end_trace_capture(self.mesh, self.trace_id, cq_id=0)
        self._capture_sampler()
        ttnn.synchronize_device(self.mesh)
        self.model.capture_prefill_traces()  # last (see _warm)
        self.model.reset_state()
        self.model.inputs.set_positions([-1] * self.max_batch_size)
        ttnn.synchronize_device(self.mesh)
        self._program_cache_after_capture = self.program_cache_entries()
        self._check_trace_buffers()
        self._trace_addr = self.trace_addresses()
        logger.info(f"serving traces captured (model + sampler variants); programs {self._program_cache_after_capture}")

    def _read_positions(self) -> torch.Tensor:
        t = ttnn.to_torch(ttnn.get_device_tensors(self.model.inputs.pos)[0])
        return t.reshape(-1)[: self.max_batch_size].round().to(torch.int64)

    def _decode_serving(
        self,
        tokens,
        start_pos,
        *,
        page_table=None,
        enable_trace=True,
        slots=None,
        sampling_params=None,
        reload_inputs=True,
        reload_sampling_params=False,
        reset_sampling_state=False,
        fresh_slots=(),
        prompt_tokens=None,
        output_tokens=None,
        read_from_device=True,
        **_,
    ):
        """vLLM decode step (decode_forward(..., serving=True)). Rows -> model slots via `slots`; sampling_params,
        prompt_tokens, output_tokens already in SLOT order. Returns host logits [B, V] (no sampling_params), host
        tokens [B] in row order (read_from_device) or the device token buffer (read_decode_output reads it)."""
        from models.autoports.zai_org_glm_5_3_flash.tt.serving_state import merge_async_ahead

        S = self.max_batch_size
        tokens = torch.as_tensor(tokens).reshape(-1).to(torch.int64)
        start_pos = torch.as_tensor(start_pos).reshape(-1).to(torch.int64)
        slots = list(slots) if slots is not None else list(range(tokens.shape[0]))
        if page_table is not None:
            self._set_page_rows(page_table, slots)  # host compare; device copy only on change
        device = sampling_params is not None and self.model.sampling is not None
        fresh = set(int(s) for s in fresh_slots) | self._fresh
        if reload_inputs:
            pos = torch.full((S,), -1, dtype=torch.int64)
            tok = torch.zeros(S, dtype=torch.int64)
            for r, s in enumerate(slots):
                pos[s] = int(start_pos[r])
                tok[s] = int(tokens[r])
            if device and enable_trace and self._serving_decoded:
                tok, pos = merge_async_ahead(tok, pos, self._read_tokens(), self._read_positions(), fresh)
            pos_l = [int(p) for p in pos.tolist()]
            self._check_positions(pos_l)
            if pos_l != self.model.inputs.positions:
                self.model.inputs.set_positions(pos_l)
            self._write_tokens([int(t) for t in tok.tolist()])
        if device and self._sstate is not None:
            self._sstate.apply_decode(
                sampling_params,
                list(self.model.inputs.positions) + [-1] * (self.model.SR - S),
                reload_params=reload_sampling_params,
                reset_state=reset_sampling_state,
                host_authoritative=reload_inputs,
                prompt_tokens_slots=prompt_tokens,
                output_tokens_slots=output_tokens,
            )
        if enable_trace:
            ttnn.execute_trace(self.mesh, self.trace_id, cq_id=0, blocking=False)
            logits = self.logits_out
        else:
            logits = self.model.decode_step(self._tok_view())
        self.model.inputs.host_advance()
        self._fresh.clear()
        self._serving_decoded = True
        if device:
            self.model.sampling.sample(logits, tt_out_tok=self.tok_buf, enable_trace=enable_trace)
            if not enable_trace:
                ttnn.deallocate(logits)
            if not read_from_device:
                return self.tok_buf
            out = self._read_tokens()
            return torch.stack([out[s] for s in slots])
        host = self.model.logits_to_host(logits)[0, 0, :, : self.model.vocab]
        if not enable_trace:
            ttnn.deallocate(logits)
        return torch.stack([host[s] for s in slots])

    def read_decode_output(self, handle, async_read=True):
        """Device token buffer -> host copy (non-blocking + event when async_read)."""
        if not async_read:
            return handle.cpu()
        host = handle.cpu(blocking=False)
        return host, [ttnn.record_event(self.mesh, 0)]

    def process_decode_output_host(self, host, slots):
        """Host copy of the token buffer -> int64 tokens in row order (die 0's row: every die holds the same token
        with lockstep sampling)."""
        t = ttnn.to_torch(ttnn.get_device_tensors(host)[0]).reshape(-1)[: self.max_batch_size].to(torch.int64)
        return torch.stack([t[int(s)] for s in slots])

    # ------------------------------------------------------------------------------------------------ state
    def reset(self):
        self.check_trace_addresses("before reset")
        self.model.reset_state()
        self.positions = [-1] * self.max_batch_size
        self.model.inputs.set_positions(self.positions)
        if self.page_table is not None:
            self.model.set_page_table(self.page_table)

    def _check_positions(self, positions):
        """Decode-overrun guard: a position at or past max_seq_len has no cache row of its own (the page-table column
        T/128 is the shared dummy block, later columns resolve to block 0 = another slot's cache). The device refresh
        also deactivates such a slot (tt/decode_inputs.py), so a replay never writes there; the host refuses it."""
        bad = [(s, p) for s, p in enumerate(positions) if p >= self.max_seq_len]
        if bad:
            raise ValueError(
                f"decode position(s) {bad[:4]} (slot, position) >= max_seq_len {self.max_seq_len}: the slot's cache is "
                f"full; build the generator with a larger max_seq_len"
            )

    def _write_tokens(self, toks: list[int]):
        t = torch.zeros(1, 1, 1, self.model.SR, dtype=torch.int32)
        t[0, 0, 0, : len(toks)] = torch.tensor(toks, dtype=torch.int32)
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(
                t, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh)
            ),
            self.tok_buf,
        )

    def _read_tokens(self) -> torch.Tensor:
        t = ttnn.to_torch(ttnn.get_device_tensors(self.tok_buf)[0])
        return t.reshape(-1)[: self.max_batch_size].to(torch.int64)

    # ------------------------------------------------------------------------------------------------ low level
    def prefill_forward(
        self,
        tokens,
        *,
        page_table=None,
        kv_cache=None,
        prompt_lens=None,
        return_all_logits=False,
        slots=None,
        sampling_params=None,
        **kw,
    ):
        """tokens [B, L_pad]; prompt_lens: real lengths; slots: decode slot per row (default 0..B-1). Returns logits
        [B, 1, V] (last position) or [B, L, V] (return_all_logits, B = 1). sampling_params (serving): one
        models.common SamplingParams per row -> on-device sampled int64 tokens [B] instead of logits."""
        assert kv_cache is None or kv_cache is self.kv_cache
        tokens = torch.as_tensor(tokens)
        B = tokens.shape[0]
        prompt_lens = list(prompt_lens) if prompt_lens is not None else [tokens.shape[1]] * B
        slots = list(slots) if slots is not None else list(range(B))
        if page_table is not None:
            self._set_page_rows(page_table, slots)
        if sampling_params is not None and self.model.sampling is not None and not return_all_logits:
            return self._prefill_sampled(tokens, prompt_lens, slots, sampling_params)
        outs = []
        for b in range(B):
            ids = tokens[b, : prompt_lens[b]].to(torch.int64)
            self._fresh.add(int(slots[b]))
            out = self.model.prefill(ids, slots[b], all_logits=return_all_logits)
            outs.append(out if return_all_logits else out.reshape(1, -1))
            self.positions[slots[b]] = prompt_lens[b]
        if (
            os.environ.get("GLM53_PREFILL_SYNC") == "1"
        ):  # diagnostic: all dies drained at the prefill -> decode boundary
            ttnn.synchronize_device(self.mesh)
            self._prefill_syncs = getattr(self, "_prefill_syncs", 0) + 1
            logger.info(f"prefill boundary sync #{self._prefill_syncs}")
        self.check_trace_addresses("after prefill")
        return torch.stack(outs, dim=0)

    def _prefill_sampled(self, tokens, prompt_lens, slots, sampling_params):
        """Serving prefill with on-device sampling: per request, the last row's device logits (copied to every sampler
        row) are sampled eagerly with that request's params set on its slot; the token is row `slot`. The output is a
        fresh tensor: the decode token buffer (live slots' fed-back tokens) is not touched."""
        out = []
        for b, s in enumerate(slots):
            ids = tokens[b, : prompt_lens[b]].to(torch.int64)
            lg = self.model.prefill(ids, int(s), device_logits=True)
            self._sstate.apply_prefill(sampling_params[b], int(s), ids)
            tt_tok, _ = self.model.sampling.sample(lg, enable_trace=False)
            ttnn.deallocate(lg)
            row = ttnn.to_torch(ttnn.get_device_tensors(tt_tok)[0]).reshape(-1)
            out.append(int(row[int(s)]))
            self.positions[int(s)] = int(prompt_lens[b])
            self._fresh.add(int(s))
        return torch.tensor(out, dtype=torch.int64)

    def _set_page_rows(self, page_table, slots):
        pt = torch.as_tensor(page_table).to(torch.int32)
        if self.page_table is None:
            return
        new = self.page_table.clone()
        for r, s in enumerate(slots):
            if r >= pt.shape[0]:  # padding rows beyond the caller's table (inactive): keep the slot's row
                break
            n = min(pt.shape[1], new.shape[1])
            new[s, :n] = pt[r, :n]
        if not torch.equal(new, self.page_table):
            self.page_table = new
            self.model.set_page_table(new)
            self.page_table_copies += 1

    def decode_forward(
        self,
        tokens,
        start_pos,
        *,
        page_table=None,
        kv_cache=None,
        enable_trace=True,
        on_device_sampling=True,
        reload_inputs=True,
        slots=None,
        **kw,
    ):
        """One decode step. tokens [B, 1], start_pos [B] (slot rows; -1 = inactive). Returns sampled tokens [B]
        (on-device greedy) or logits [B, V] (host compatibility mode). reload_inputs=False: the device already holds
        the tokens (fed back by the sampler) and positions (advanced by the trace). serving=True: the vLLM step
        (_decode_serving: per-slot sampling params, async-ahead reload, read_from_device=False)."""
        assert kv_cache is None or kv_cache is self.kv_cache
        if kw.pop("serving", False):
            return self._decode_serving(
                tokens,
                start_pos,
                page_table=page_table,
                enable_trace=enable_trace,
                reload_inputs=reload_inputs,
                slots=slots,
                **kw,
            )
        S = self.max_batch_size
        if os.environ.get("GLM53_FORCE_EAGER_DECODE") == "1":  # diagnostic: every decode step untraced
            enable_trace = False
        tokens = torch.as_tensor(tokens).reshape(-1).to(torch.int64)
        start_pos = torch.as_tensor(start_pos).reshape(-1).to(torch.int64)
        B = tokens.shape[0]
        slots = list(slots) if slots is not None else list(range(B))
        if page_table is not None:
            self._set_page_rows(page_table, slots)
        if reload_inputs:
            pos = [-1] * S
            tok = [0] * S
            for r, s in enumerate(slots):
                pos[s] = int(start_pos[r])
                tok[s] = int(tokens[r])
            self._check_positions(pos)
            if pos != self.model.inputs.positions:
                self.model.inputs.set_positions(pos)
            self._write_tokens(tok)
        else:  # the device advanced the positions itself; the host mirror (host_advance) must still be in range
            self._check_positions(self.model.inputs.positions)
        if enable_trace:
            ttnn.execute_trace(self.mesh, self.trace_id, cq_id=0, blocking=False)
            logits = self.logits_out
        else:
            logits = self.model.decode_step(self._tok_view())
        self.model.inputs.host_advance()
        if on_device_sampling and self.model.sampling is not None:
            self.model.sampling.sample(logits, tt_out_tok=self.tok_buf, enable_trace=enable_trace)
            out = self._read_tokens()
            return torch.stack([out[s] for s in slots])
        host = self.model.logits_to_host(logits)[0, 0, :, : self.model.vocab]
        if not enable_trace:
            ttnn.deallocate(logits)
        return torch.stack([host[s] for s in slots])

    # ------------------------------------------------------------------------------------------------ high level
    def prefill_logits(self, prompt_token_ids):
        self.reset()
        ids = torch.tensor(list(prompt_token_ids), dtype=torch.int64).reshape(1, -1)
        return self.prefill_forward(ids, prompt_lens=[ids.shape[1]], return_all_logits=True)

    def generate(
        self, prompt_token_ids, max_new_tokens, *, next_input=None, enable_trace=True, on_device_sampling=True, **kw
    ):
        """Greedy generation (slot 0). next_input(i, pred_i) may override the token fed at step i (teacher forcing).
        Returns the model's own predictions."""
        on_device_sampling = on_device_sampling and self.model.sampling is not None
        ids = torch.tensor(list(prompt_token_ids), dtype=torch.int64).reshape(1, -1)
        P = ids.shape[1]
        # decode feeds positions P .. P + max_new_tokens - 2 (the last prediction needs no cache row)
        if P + int(max_new_tokens) - 1 > self.max_seq_len:
            raise ValueError(
                f"prompt {P} + max_new_tokens {max_new_tokens} - 1 exceeds max_seq_len {self.max_seq_len} of this "
                f"generator"
            )
        self.reset()
        logits = self.prefill_forward(ids, prompt_lens=[P])
        pred = int(torch.argmax(logits[0, -1, : self.model.vocab]).item())  # first token: the prefill logits row
        preds = []
        pos = P
        reload = True
        for i in range(int(max_new_tokens)):
            preds.append(pred)
            if i == int(max_new_tokens) - 1:
                break
            forced = next_input(i, pred) if next_input is not None else None
            tok = pred if forced is None else int(forced)
            reload = reload or tok != pred or not on_device_sampling
            out = self.decode_forward(
                torch.tensor([[tok]]),
                torch.tensor([pos]),
                enable_trace=enable_trace,
                on_device_sampling=on_device_sampling,
                reload_inputs=reload,
            )
            pred = int(out[0]) if on_device_sampling else int(torch.argmax(out[0]).item())
            pos += 1
            reload = False
        if next_input is not None and len(preds) == int(max_new_tokens):
            next_input(len(preds) - 1, preds[-1])
        return preds

    def teardown(self):
        self.model.release_prefill_traces()
        if self.trace_id is not None:
            ttnn.release_trace(self.mesh, self.trace_id)
            self.trace_id = None


def _check_mesh(mesh_device):
    try:
        trace = ttnn.get_memory_view(mesh_device, ttnn.BufferType.TRACE).total_bytes_per_bank
        small = ttnn.get_memory_view(mesh_device, ttnn.BufferType.L1_SMALL).total_bytes_per_bank
    except Exception as e:  # noqa: BLE001
        logger.warning(f"memory view unavailable ({e}); assuming the mesh has a trace region")
        return
    if trace <= 0 or small < L1_SMALL_SIZE:
        raise RuntimeError(
            f"mesh opened with trace_region_size={trace} l1_small_size={small}; this model needs a trace region and "
            f"l1_small_size>={L1_SMALL_SIZE}: open it with models/autoports/zai_org_glm_5_3_flash/tools/readiness.py"
        )


def build_generator(model_dir, mesh_device, **kwargs):
    """Readiness / standard Metal generator entry point. max_batch_size (GLM53_MAX_BATCH, default 1) and max_seq_len
    (GLM53_MAX_SEQ_LEN, default: the largest context of that slot count in the fastest layout, tt/generator
    .plan_context) choose the build; dsa_layout ("replicated" / "sharded") forces the DSA cache layout; other keyword
    arguments (layer_indices, prefill_lengths, ...) go to Glm53Model. A combination that does not fit in device DRAM
    raises ValueError naming the largest context that does."""
    _check_mesh(mesh_device)
    kw = dict(kwargs)
    S = int(kw.pop("max_batch_size", os.environ.get("GLM53_MAX_BATCH", 1)))
    T = kw.pop("max_seq_len", os.environ.get("GLM53_MAX_SEQ_LEN"))
    return Glm53Generator(mesh_device, max_batch_size=S, max_seq_len=int(T) if T else None, **kw)
