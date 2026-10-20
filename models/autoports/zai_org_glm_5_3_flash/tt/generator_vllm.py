# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""vLLM adapter of the GLM-5.3-Flash (E192) autoport on a (1, 4) Blackhole mesh (text only), stage 9.

Design: doc/vllm/DESIGN.md. This file only translates the vllm-tt-plugin interface onto the serving API of
tt/generator.py (Glm53Generator(defer_caches=True), bind_kv_cache, warm_programs, capture_traces, prefill_forward,
decode_forward(serving=True), read_decode_output, process_decode_output_host, program_cache_entries). Every serving
number (DSA pool per slot count, max_model_len limits, trace region, DSA layout, precision policy) comes from
serving_profiles.json (tt/serving_profiles.py).

Contract summary
  * Architecture "Glm5NextForConditionalGeneration" (HF config.architectures), registered by the plugin as
    "TTGlm5NextForConditionalGeneration" from EXTRA_MODELS_DIR (vllm_bundle/glm_5_3_flash/vllm_metadata.json).
  * Cache ownership: vLLM owns the BLOCK ALLOCATION (block ids, block tables, num_blocks; block size 128). The model
    allocates the DSA latent / pooled-key pools of every DSA layer for exactly vLLM's num_blocks (+ 1 model dummy block)
    in allocate_kv_cache and returns them; the adapter passes that exact object back on every call. KDA recurrent /
    conv state and the DSA open-pool accumulators are constant-size per-slot model state (no vLLM KV spec).
  * Slots: the plugin assigns a device state slot per request (empty_slots at prefill, slot_remap at decode). The
    adapter keeps a virtual -> physical slot map instead of moving device state: a remap is a host permutation.
  * Sampling: on-device split sampling of the generator (model trace + sampler trace, token fed back into the
    persistent token buffer). Host-logits compatibility mode when the plugin sends no sampling params.
  * Async decode: decode_forward(read_from_device=False) -> device token buffer; read_decode_output(async_read=True)
    -> non-blocking copy + event; process_decode_output_host -> tokens in row order.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
from loguru import logger

import ttnn
from models.autoports.zai_org_glm_5_3_flash.tt import serving_profiles

MODEL_DIR = Path(__file__).resolve().parents[1]
HF_MODEL = "zai-org/GLM-5.3-Flash"
ARCH = "Glm5NextForConditionalGeneration"
BLOCK_SIZE = 128  # DSA page block: its 32 pool keys are one tile row of the pooled-key cache (tt/dsa.py BLOCK)
TP = 4


# ------------------------------------------------------------------------------------------------ budget / context
# Every serving number (DSA pool per slot count, max_model_len limits, trace region, DSA layout, precision policy) comes
# from serving_profiles.json through tt/serving_profiles.py. No constants here.
def _profiles() -> dict:
    return serving_profiles.load()


def _extra_reserve_bytes() -> int:
    """GLM53_VLLM_EXTRA_RESERVE_MIB: DRAM per die kept free besides the measured reserve (debug / safety margin)."""
    return int(float(os.environ.get("GLM53_VLLM_EXTRA_RESERVE_MIB", "0")) * 2**20)


def max_model_len_for(max_num_seqs: int) -> int:
    """Largest max_model_len the DSA pool allows at this max_num_seqs (serving_profiles.json)."""
    return serving_profiles.max_model_len_for(_profiles(), int(max_num_seqs), _extra_reserve_bytes())


def _contract() -> dict:
    p = MODEL_DIR / "doc" / "context_contract.json"
    return json.loads(p.read_text()) if p.exists() else {}


def _check_context(max_model_len: int, data: dict):
    """max_model_len <= the advertised context. Warn when the context contract's supported context is not the 1-slot
    limit of the data file (the stage gate reads the contract)."""
    c = _contract()
    adv = int(c.get("advertised_context", 1 << 20))
    if max_model_len > adv:
        raise ValueError(f"max_model_len {max_model_len} exceeds the advertised context {adv}")
    sup = c.get("supported_context")
    one = serving_profiles.supported_context(data)
    if sup is not None and int(sup) != one:
        logger.warning(
            f"doc/context_contract.json supported_context {sup} != the 1-slot limit {one} of {serving_profiles.path()}: "
            f"update the contract after the capacity probe (the stage-9 gate compares served max_model_len with it)"
        )


def _precision_id() -> str:
    """The resolved policy id the model builds with (logged at init)."""
    from .precision import PrecisionOverrides

    return PrecisionOverrides.from_env().id


def apply_precision_default(data: dict | None = None):
    """GLM53_PRECISION unset -> the selected policy file named in serving_profiles.json (absolute path). A relative
    GLM53_PRECISION that does not exist from the cwd is resolved against the model dir (the container's cwd is not the
    source tree). The stage-7 policy (GLM53_PRECISION=none / shipped / off / 0) is refused: the serving budget in
    serving_profiles.json is measured for the selected policy only (the stage-7 bf16 latent needs about 1.9x the DSA
    bytes per block and has less free DRAM)."""
    data = data if data is not None else _profiles()
    cfg = (data.get("precision") or {}).get("config")
    cur = os.environ.get("GLM53_PRECISION", "").strip()
    if cur.lower() in ("none", "shipped", "off", "0"):
        raise ValueError(
            f"GLM53_PRECISION={cur!r} selects the stage-7 precision policy, which the vLLM path does not serve: "
            f"serving_profiles.json budgets the selected policy ({(data.get('precision') or {}).get('id')}). "
            "Use the standalone generator (tools/readiness.py) for stage-7 A/B runs."
        )
    if cur:
        if not Path(cur).exists() and (MODEL_DIR / cur).exists():
            os.environ["GLM53_PRECISION"] = str(MODEL_DIR / cur)
        return
    if not cfg:
        return
    if not (MODEL_DIR / cfg).exists():
        raise RuntimeError(f"selected precision policy {MODEL_DIR / cfg} is missing (serving_profiles.json precision)")
    os.environ["GLM53_PRECISION"] = str(MODEL_DIR / cfg)


def _check_weight_cache():
    base = os.environ.get("GLM53_CACHE_DIR", "").strip().lower()
    if base in ("", "0", "off", "none"):
        logger.warning(
            "GLM53_CACHE_DIR is not set: the weights are converted from the FP8 checkpoint on every start (~25 min "
            "instead of ~2 min); set GLM53_CACHE_DIR to a persistent directory (the bundle uses /weight-cache)"
        )


# ------------------------------------------------------------------------------------------------ checkpoint
def _overlay_root() -> Path:
    env = os.environ.get("GLM53_CKPT_OVERLAY_DIR")
    if env:
        return Path(env)
    return Path(os.environ.get("TT_CACHE_PATH", Path.home() / ".cache" / "ttnn")) / "glm53_ckpt_overlay"


def resolve_checkpoint(hf_config) -> str:
    """The checkpoint directory the model reads (sets GLM53_CKPT). Order:
    1. GLM53_CKPT if set (the E192 build of the bring-up host);
    2. the served model path (hf_config._name_or_path) when it is a local directory (an E192 build, or a parent
       snapshot that carries e192_mask.json);
    3. the local HF snapshot of that repo id at GLM53_WEIGHTS_REVISION (the bundle's pinned parent revision; unset =
       the cached default ref), overlaid with the autoport's tt/e192_mask.json: a directory of symlinks to the snapshot
       files plus the mask, under GLM53_CKPT_OVERLAY_DIR (the parent snapshot has 288 experts and no mask;
       CheckpointReader reads the 192 kept experts through the mask)."""
    env = os.environ.get("GLM53_CKPT")
    if env:
        return env
    name = str(getattr(hf_config, "_name_or_path", "") or HF_MODEL)
    if Path(name).is_dir():
        os.environ["GLM53_CKPT"] = name
        return name
    from huggingface_hub import snapshot_download

    rev = os.environ.get("GLM53_WEIGHTS_REVISION") or None
    snap = Path(snapshot_download(name, revision=rev, local_files_only=True))
    if (snap / "e192_mask.json").exists():
        os.environ["GLM53_CKPT"] = str(snap)
        return str(snap)
    mask = MODEL_DIR / "tt" / "e192_mask.json"
    if not mask.exists():
        raise RuntimeError(f"{snap} has no e192_mask.json and the autoport ships none ({mask})")
    parent = json.loads(mask.read_text()).get("parent")
    if parent and parent != name:
        raise RuntimeError(f"tt/e192_mask.json was made for {parent}, the served model is {name}")
    root = _overlay_root() / snap.name
    root.mkdir(parents=True, exist_ok=True)
    for f in snap.iterdir():
        link = root / f.name
        if link.is_symlink() or link.exists():
            continue
        link.symlink_to(f.resolve() if f.is_file() else f)
    tgt = root / "e192_mask.json"
    if not tgt.exists() or tgt.read_bytes() != mask.read_bytes():
        tmp = root / ".e192_mask.json.tmp"
        tmp.write_bytes(mask.read_bytes())
        tmp.replace(tgt)
    os.environ["GLM53_CKPT"] = str(root)
    logger.info(f"checkpoint: {snap} (revision {rev or 'default ref'}) + tt/e192_mask.json overlay at {root}")
    return str(root)


# ------------------------------------------------------------------------------------------------ slots
class SlotMap:
    """vLLM state slot (virtual) -> model slot (physical). The plugin's slot_remap says "row i now reads slot
    remap[i]" and expects the state to move; we move the map instead (zero device work). Always a permutation."""

    def __init__(self, n: int):
        self.phys = list(range(n))

    def remap(self, remap) -> bool:
        r = [int(x) for x in torch.as_tensor(remap).reshape(-1).tolist()]
        assert sorted(r) == list(range(len(self.phys))), f"slot_remap is not a permutation: {r}"
        if r == list(range(len(r))):
            return False
        self.phys = [self.phys[j] for j in r]
        return True

    def __getitem__(self, v: int) -> int:
        return self.phys[int(v)]

    def rows(self, n_rows: int) -> list[int]:
        """Decode row r carries the request that owns virtual slot r (after the step's remap)."""
        return self.phys[:n_rows]


def pad_page_table(pt, width: int) -> torch.Tensor:
    """vLLM block table [rows, n] -> [rows, width] int32. Missing columns are 0 = vLLM's null block (never allocated
    to a request), so every padded / not-yet-allocated column points at a block no request owns."""
    pt = torch.as_tensor(pt if isinstance(pt, torch.Tensor) else ttnn.to_torch(pt)).to(torch.int32)
    if pt.shape[1] >= width:
        return pt[:, :width].contiguous()
    out = torch.zeros(pt.shape[0], width, dtype=torch.int32)
    out[:, : pt.shape[1]] = pt
    return out


@dataclass
class _DecodeOut:
    """Device decode output on its way to the host: the generator's handle plus the row -> slot map of the step."""

    handle: object
    slots: list
    n_rows: int
    host: object = None
    events: list = field(default_factory=list)


def _tensor_kw(kwargs, args, name, idx):
    if name in kwargs:
        return kwargs[name]
    return args[idx] if len(args) > idx else None


# ------------------------------------------------------------------------------------------------ adapter
class Glm5NextForConditionalGeneration:
    """vLLM bridge for zai-org/GLM-5.3-Flash (E192 expert subset), text only."""

    supports_multimodal = False
    already_warmed_up_prefill = False

    # supports_async_decode: the decode trace keeps tokens (sampler -> persistent token buffer) and positions
    # (DeviceDecodeInputs.advance inside the trace) on device; the split submit / read / host-format path is below.
    # MUST stay False until the stale-input tests of tests/test_generator_vllm.py pass on device (DESIGN.md 7).
    model_capabilities = {
        "supports_prefix_caching": False,
        "supports_async_decode": os.environ.get("GLM53_VLLM_ASYNC_DECODE", "0") == "1",
        "supports_sample_on_device": True,
        "supports_chunked_prefill": False,
    }
    decode_input_update_contract = 1

    # ---------------------------------------------------------------------------------------------- vLLM model protocol
    # vLLM 0.26 has no upstream class for this architecture, so the server is started with
    # --hf-overrides '{"architectures": ["TTGlm5NextForConditionalGeneration"]}' and ModelConfig inspects THIS class to
    # decide whether it can `--runner generate` (VllmModelForTextGeneration protocol: __init__(vllm_config, prefix),
    # embed_input_ids, forward(input_ids, positions), compute_logits). The TT plugin never calls these; it builds the
    # model with initialize_vllm_model and drives prefill_forward / decode_forward.
    def __init__(self, vllm_config=None, prefix: str = ""):
        pass

    def embed_input_ids(self, input_ids):
        raise NotImplementedError("TT model: the vllm-tt-plugin drives prefill_forward / decode_forward")

    def forward(self, input_ids, positions, *args, **kwargs):
        raise NotImplementedError("TT model: the vllm-tt-plugin drives prefill_forward / decode_forward")

    def compute_logits(self, hidden_states):
        raise NotImplementedError("TT model: the vllm-tt-plugin drives prefill_forward / decode_forward")

    # ---------------------------------------------------------------------------------------------- vLLM sizing
    @classmethod
    def get_kv_cache_spec(cls, vllm_config):
        """One full-attention group: the 11 DSA layers share ONE block table (identical paging in every layer).
        The KDA layers hold constant-size per-slot state inside the model and have no vLLM spec. The spec's page
        size only feeds the plugin's synthetic byte budget (get_max_tokens_all_users sets the real block count)."""
        from vllm.v1.kv_cache_interface import FullAttentionSpec

        return {
            "model.language_model.layers.3.self_attn": FullAttentionSpec(
                block_size=int(vllm_config.cache_config.block_size),
                num_kv_heads=1,
                head_size=512,  # kv_lora_rank: the cached latent row
                dtype=torch.bfloat16,
            )
        }

    @classmethod
    def get_max_tokens_all_users(
        cls, model_name="", num_devices=1, tt_data_parallel=1, max_model_len=None, max_num_seqs=None, **kwargs
    ):
        """The shared DSA token pool at this max_num_seqs (serving_profiles.json: measured pool, else the budget rule).
        The plugin adds one block per user and rounds up, so vLLM's num_blocks = pool_blocks - 1; the model adds its
        dummy block in bind_kv_cache. Raises when max_model_len does not fit. GLM53_MAX_TOKENS_ALL_USERS overrides
        (debug only)."""
        B = int(max_num_seqs or 1)
        override = os.environ.get("GLM53_MAX_TOKENS_ALL_USERS")
        if override:
            logger.warning(f"GLM53_MAX_TOKENS_ALL_USERS={override} overrides serving_profiles.json")
            return int(override)
        if int(num_devices) != TP or int(tt_data_parallel) != 1:
            raise ValueError(
                f"GLM-5.3-Flash serves one (1,{TP}) mesh; got {num_devices} devices, DP {tt_data_parallel}"
            )
        data = _profiles()
        extra = _extra_reserve_bytes()
        if max_model_len is not None:
            serving_profiles.check_fit(data, B, int(max_model_len), extra)
        nb = serving_profiles.pool_blocks(data, B, extra)
        tokens = serving_profiles.max_tokens_all_users(data, B, extra)
        b = data["budget"]
        logger.info(
            f"GLM53 KV pool: {B} slots -> {nb} blocks = {nb * BLOCK_SIZE} tokens per die incl. the dummy block "
            f"({serving_profiles.pool_source(data, B)}; {b['dsa_bytes_per_block']} B per block, "
            f"{b['kda_bytes_per_slot'] / 2**20:.1f} MiB KDA state per slot); max_tokens_all_users {tokens}"
        )
        return max(tokens, BLOCK_SIZE)

    # ---------------------------------------------------------------------------------------------- construction
    @classmethod
    def initialize_vllm_model(
        cls, hf_config, mesh_device, max_batch_size, max_seq_len=None, tt_data_parallel=1, optimizations=None, **kw
    ):
        from models.autoports.zai_org_glm_5_3_flash.tt.generator import Glm53Generator, _check_mesh

        if int(tt_data_parallel) != 1:
            raise ValueError(f"{cls.__name__} serves one (1,{TP}) mesh; got tt_data_parallel={tt_data_parallel}")
        if tuple(mesh_device.shape) != (1, TP):
            raise ValueError(f"{cls.__name__} needs a (1,{TP}) mesh (P300x2 / P150x4); got {tuple(mesh_device.shape)}")
        data = _profiles()
        B = int(max_batch_size)
        max_seq_len = int(max_seq_len or max_model_len_for(B))
        _check_context(max_seq_len, data)
        reduced = bool(os.environ.get("GLM53_LAYERS"))  # inner-loop target: the full-model budget does not apply
        if not reduced:
            serving_profiles.check_fit(data, B, max_seq_len, _extra_reserve_bytes())
        tr = os.environ.get("GLM53_TRACE_REGION")
        if tr and int(tr) != int(data["trace_region_bytes"]):
            logger.warning(
                f"GLM53_TRACE_REGION {tr} != serving_profiles.json trace_region_bytes {data['trace_region_bytes']}"
            )
        _check_mesh(mesh_device)  # trace region and l1_small_size of the opened mesh
        apply_precision_default(data)
        _check_weight_cache()
        ckpt = resolve_checkpoint(hf_config)
        os.environ.setdefault("HF_MODEL", HF_MODEL)
        self = cls()
        self.mesh = mesh_device
        self.B = B
        self.max_model_len = max_seq_len
        # the model's max_context: a multiple of 128 and of the largest prefill chunk that fits it (functional_decoder)
        self.T = serving_profiles.model_context(max_seq_len)
        self.layout = data["dsa_layout"]
        t0 = time.time()
        # defer_caches: weights + KDA slot state now; DSA pools in allocate_kv_cache (vLLM's num_blocks), warm-up and
        # trace capture in the plugin's warm-up phases. The DRAM check is serving_profiles.check_fit above (measured
        # data); budget_check=False skips the generator's constant budget (tt/generator.BUDGET), which does not know
        # the selected precision (fp8 latent, bfp4 projections). The layout is the one the profiles were measured with.
        self._gen = Glm53Generator(
            mesh_device,
            max_batch_size=B,
            max_seq_len=self.T,
            defer_caches=True,
            dsa_layout=self.layout,
            budget_check=False,
        )
        self.slots = SlotMap(B)
        self._fresh = set()  # physical slots prefilled since the last decode submit (their host token is authoritative)
        self._kv = None
        self._phase = {"programs": False, "traces": False}
        self._program_cache_after_capture = None
        self._decode_steps = 0
        self._guard = os.environ.get("GLM53_PROGRAM_GUARD", "1") == "1"
        if os.environ.get("GLM53_FAULTHANDLER", "1") == "1":
            import faulthandler
            import signal

            faulthandler.register(signal.SIGUSR1, all_threads=True, chain=False)  # kill -USR1 <EngineCore pid>
        logger.info(
            f"{cls.__name__}: B={B} max_model_len={max_seq_len} (model context {self.T}, DSA layout {self.layout}, "
            f"precision {_precision_id()}) layers={len(self._gen.model.layers)} "
            f"checkpoint={ckpt} built in {time.time() - t0:.0f}s (vLLM owns the blocks)"
        )
        return self

    # ---------------------------------------------------------------------------------------------- cache
    def allocate_kv_cache(self, kv_cache_shape, dtype, num_layers):
        """kv_cache_shape = (num_blocks, 1, block_size, 512) from get_kv_cache_spec. The model allocates every DSA
        layer's latent + pooled-key pool for num_blocks (+ its dummy block) and returns the handles; vLLM keeps them
        and passes them back as kv_cache."""
        num_blocks, _, block, _ = (int(d) for d in kv_cache_shape)
        assert block == BLOCK_SIZE, f"--block-size must be {BLOCK_SIZE} (DSA page block); got {block}"
        if not os.environ.get("GLM53_LAYERS"):
            nb = serving_profiles.pool_blocks(_profiles(), self.B, _extra_reserve_bytes())
            if num_blocks + 1 > nb:
                logger.warning(
                    f"vLLM asks for {num_blocks} + 1 blocks, serving_profiles.json allows {nb} at {self.B} slots"
                )
        self._kv = self._gen.bind_kv_cache(num_blocks)
        self._pt_width = self.T // BLOCK_SIZE
        logger.info(
            f"DSA pools: {num_blocks} vLLM blocks + 1 dummy x {block} tokens in {len(self._kv)} DSA layers "
            f"(vLLM spec layers {num_layers}); page-table width {self._pt_width}"
        )
        return self._kv

    def _check_kv(self, kv_cache):
        if kv_cache is not None and self._kv is not None and kv_cache is not self._kv:
            raise RuntimeError("kv_cache is not the object allocate_kv_cache returned (hidden standalone cache?)")

    # ---------------------------------------------------------------------------------------------- warm-up
    def _validate_device_sampling(self, requested):
        if requested and self._gen.model.sampling is None:
            raise RuntimeError("on-device sampling requested but the model has no sampler (needs tp > 1)")

    def warmup_model_prefill(self, kv_cache=None, enable_trace=False, *args, **kwargs):
        """Phase 1 (enable_trace False): compile every prefill program (all DSA paths x prefill lengths), the eager
        decode step, the head, every sampler variant and the prefill sampler - before any capture. Phase 2: prefill
        stays eager in stage 9 (no prefill traces yet); nothing to capture."""
        self._check_kv(kv_cache)
        self._validate_device_sampling(kwargs.get("can_sample_on_device", False))
        if not enable_trace:
            if not self._phase["programs"]:
                t0 = time.time()
                self._gen.warm_programs()
                self._phase["programs"] = True
                logger.info(f"warm-up phase 1 (programs) done in {time.time() - t0:.0f}s")
            return
        self.already_warmed_up_prefill = True

    def warmup_model_decode(self, *args, **kwargs):
        self._check_kv(kwargs.get("kv_cache"))
        self._validate_device_sampling(kwargs.get("can_sample_on_device", False))
        if not kwargs.get("enable_trace", False):
            if not self._phase["programs"]:
                self._gen.warm_programs()
                self._phase["programs"] = True
            return
        if not self._phase["traces"]:
            t0 = time.time()
            self._gen.capture_traces()
            self._phase["traces"] = True
            self._program_cache_after_capture = self._program_cache()
            logger.info(
                f"warm-up phase 2 (model + sampler traces) done in {time.time() - t0:.0f}s; "
                f"PROGRAM_GUARD {'active' if self._guard else 'OFF'}"
            )

    def _program_cache(self):
        fn = getattr(self._gen, "program_cache_entries", None)
        return fn() if fn is not None else None

    def _guard_programs(self, where):
        if not self._guard or self._program_cache_after_capture is None:
            return
        now = self._program_cache()
        if now is not None and now != self._program_cache_after_capture:
            raise RuntimeError(
                f"PROGRAM_GUARD: {now - self._program_cache_after_capture} program(s) compiled after the trace capture "
                f"({where}); a warm-up phase misses this shape"
            )

    # ---------------------------------------------------------------------------------------------- prefill
    def prefill_forward(self, tokens, page_table, kv_cache, prompt_lens, **kwargs):
        """N whole prompts (no chunked prefill, no prefix caching) into their slots. Any prompt length <= max_model_len
        (the model pads the last chunk internally). Returns tokens [N] (device sampling: the plugin sent sampling
        params) or logits [N, 1, V] (host sampling)."""
        self._check_kv(kv_cache)
        tokens = torch.as_tensor(tokens)
        N = tokens.shape[0]
        starts = kwargs.get("start_pos")
        if starts is not None and any(int(s) for s in torch.as_tensor(starts).reshape(-1).tolist()[:N]):
            raise NotImplementedError("chunked prefill / prefix caching is not supported (model_capabilities)")
        plens = [int(prompt_lens[u]) for u in range(N)] if prompt_lens is not None else [tokens.shape[1]] * N
        if max(plens) > self.max_model_len:
            raise ValueError(f"prompt of {max(plens)} tokens exceeds max_model_len {self.max_model_len}")
        empty = kwargs.get("empty_slots")
        virt = list(range(N)) if empty is None else [int(s) for s in empty]
        phys = [self.slots[v] for v in virt]
        pt = pad_page_table(page_table, self._pt_width)
        sp = kwargs.get("sampling_params")
        sp_rows = None
        if sp is not None:  # one SamplingParams per request (scalars), in request order
            sp_rows = [_row_params(sp, u) for u in range(N)]
        logger.info(f"prefill {N} request(s): lengths {plens}, vLLM slots {virt} -> model slots {phys}")
        out = self._gen.prefill_forward(
            tokens,
            page_table=pt,
            kv_cache=self._kv,
            prompt_lens=plens,
            slots=phys,
            sampling_params=sp_rows,
        )
        self._fresh |= set(phys)
        self._guard_programs(f"prefill of lengths {plens}")
        if sp_rows is not None:
            return torch.as_tensor(out).reshape(N, 1).to(torch.int64)
        return torch.as_tensor(out).reshape(N, 1, -1)

    # ---------------------------------------------------------------------------------------------- decode
    def decode_forward(self, *args, **kwargs):
        """One traced decode step of every live request. Plugin kwargs: tokens [B,1], start_pos [B] (-1 = padding
        row), page_table [B, n], sampling_params (absent = host-logits mode), prompt_tokens / output_tokens, slot_remap,
        contract-v1 commands reload_inputs / reload_page_table / reload_sampling_params / reset_sampling_state,
        enable_trace, read_from_device."""
        from models.autoports.zai_org_glm_5_3_flash.tt.serving_state import rows_to_slots, rows_to_slots_tokens

        tokens = torch.as_tensor(_tensor_kw(kwargs, args, "tokens", 0)).reshape(-1)
        start_pos = torch.as_tensor(_tensor_kw(kwargs, args, "start_pos", 1)).reshape(-1)
        page_table = _tensor_kw(kwargs, args, "page_table", 2)
        self._check_kv(kwargs.get("kv_cache"))
        remap = kwargs.get("slot_remap")
        moved = self.slots.remap(remap) if remap is not None else False
        n_rows = int(tokens.numel())
        slots = self.slots.rows(n_rows)
        sp = kwargs.get("sampling_params")
        reload_inputs = bool(kwargs.get("reload_inputs", True)) or moved
        reload_pt = bool(kwargs.get("reload_page_table", False)) or reload_inputs
        sampling_reset = bool(kwargs.get("reset_sampling_state", False))
        sampling_reload = bool(kwargs.get("reload_sampling_params", False)) or sampling_reset or moved
        S = self.B
        out = self._gen.decode_forward(
            tokens,
            start_pos,
            # always handed over: the generator compares it on the host (a [B, T/128] int32 compare) and copies it to
            # the device only when a row changed. reload_page_table alone is not trusted (page growth, DESIGN.md 8.3)
            page_table=pad_page_table(page_table, self._pt_width) if page_table is not None else None,
            page_table_changed_hint=reload_pt,
            kv_cache=self._kv,
            enable_trace=bool(kwargs.get("enable_trace", True)),
            slots=slots,
            sampling_params=rows_to_slots(sp, slots, S) if sp is not None else None,
            reload_inputs=reload_inputs,
            reload_sampling_params=sampling_reload,
            reset_sampling_state=sampling_reset,
            fresh_slots=sorted(self._fresh),
            prompt_tokens=rows_to_slots_tokens(kwargs.get("prompt_tokens"), slots, S) if sp is not None else None,
            output_tokens=rows_to_slots_tokens(kwargs.get("output_tokens"), slots, S) if sp is not None else None,
            read_from_device=bool(kwargs.get("read_from_device", True)) or sp is None,
            serving=True,
        )
        self._fresh.clear()
        self._decode_steps += 1
        if self._decode_steps == 1 or self._decode_steps % 2000 == 0:
            self._guard_programs(f"decode step {self._decode_steps}")
            logger.info(f"decode step {self._decode_steps}: traced={kwargs.get('enable_trace', True)} rows={n_rows}")
        if sp is None:  # host-logits compatibility mode: [B, V] host logits in row order
            return torch.as_tensor(out).reshape(n_rows, 1, -1)
        if isinstance(out, torch.Tensor):  # read_from_device=True: host tokens in row order
            return out.reshape(n_rows, 1).to(torch.int64)
        return _DecodeOut(handle=out, slots=slots, n_rows=n_rows)

    def read_decode_output(self, tt_out, async_read=False):
        if isinstance(tt_out, torch.Tensor):
            return (tt_out, []) if async_read else tt_out
        host, events = self._gen.read_decode_output(tt_out.handle, async_read=True)
        tt_out.host, tt_out.events = host, events
        if async_read:
            return tt_out, events
        for e in events:
            ttnn.event_synchronize(e)
        return tt_out

    def process_decode_output_host(self, tt_out, is_tokens=True):
        if isinstance(tt_out, torch.Tensor):
            return tt_out
        assert is_tokens, "device decode output is always tokens"
        if tt_out.host is None:  # synchronous plugin path: submit without async_read, then host formatting
            self.read_decode_output(tt_out, async_read=False)
        toks = self._gen.process_decode_output_host(tt_out.host, tt_out.slots)  # [n_rows] in row order
        return torch.as_tensor(toks).reshape(tt_out.n_rows, 1).to(torch.int64)


def _row_params(sp, u: int):
    """Request u's sampling params (scalars) from the plugin's per-row TTSamplingParams."""
    from dataclasses import fields as dc_fields

    from models.common.sampling.sampling_params import SamplingParams

    vals = {}
    for f in dc_fields(SamplingParams):
        v = getattr(sp, f.name, None)
        if isinstance(v, torch.Tensor):
            v = v.tolist()
        if isinstance(v, (list, tuple)):
            v = v[u] if u < len(v) else (v[0] if v else None)
        vals[f.name] = v
    defaults = {
        "temperature": 0.0,
        "top_k": 1,
        "top_p": 1.0,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "repetition_penalty": 1.0,
        "enable_log_probs": False,
        "num_logprobs": 0,
    }
    for k, d in defaults.items():
        if vals.get(k) is None:
            vals[k] = d
    if vals.get("seed") is not None and int(vals["seed"]) < 0:
        vals["seed"] = None
    return SamplingParams(**vals)
