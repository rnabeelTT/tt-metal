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


class HYV4FlashForConditionalGeneration:
    """vLLM bridge (HF arch HYV4FlashForConditionalGeneration; text only)."""

    supports_multimodal = False
    model_capabilities = {
        "supports_prefix_caching": False,
        "supports_async_decode": False,
        "supports_sample_on_device": True,
        "supports_chunked_prefill": False,
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

    @classmethod
    def initialize_vllm_model(cls, hf_config, mesh_device, max_batch_size, max_seq_len=None, tt_data_parallel=1,
                              optimizations=None, **kwargs):
        B = int(max_batch_size)
        if int(tt_data_parallel) != 1 or not 1 <= B <= 8:
            raise ValueError(f"{cls.__name__} serves 1..8 users on one (1,4) mesh (--max-num-seqs <= 8, data parallel 1)")
        if B > 1 and SPEC_K > 0:
            raise ValueError("HY4_SPEC_K (speculative decoding) serves one user: use --max-num-seqs 1")
        max_seq_len = int(max_seq_len or 131072)
        # multi-user: every slot holds max_model_len tokens; the DRAM left after warm-up must fit a prefill chunk's
        # MoE buffers (8 x 12K: 0.9 GiB free, a 12K prompt with 7 live users ok; 8 x 16K: out of memory, 2026-11-19)
        budget = int(os.environ.get("HY4_SLOT_TOKENS_MAX", "98304"))
        if B > 1 and B * max_seq_len > budget:
            raise ValueError(f"{B} users x max_model_len {max_seq_len} = {B * max_seq_len} tokens exceed the "
                             f"{budget}-token cache budget (HY4_SLOT_TOKENS_MAX): use --max-model-len <= {budget // B}")
        from models.autoports.tencent_hy4_flash.tt.config import Hy4Config, checkpoint_dir
        from models.autoports.tencent_hy4_flash.tt.engine import Hy4Engine
        from models.autoports.tencent_hy4_flash.tt.engine_batch import Hy4BatchEngine
        from models.autoports.tencent_hy4_flash.tt.model import Hy4Model
        from models.experimental.hybrid_moe.engram_store import EngramRows

        cfg = Hy4Config.load()
        from models.experimental.hybrid_moe import engram_store

        edir = os.environ.get("HY4_ENGRAM_DIR", os.path.expanduser("~/.cache/hybrid-moe/hy4_engram"))
        engram_store.OUT = edir
        engram_store._SNAP = checkpoint_dir()
        engram_store.convert()  # first boot: the FP8 tables (~65 GB, idempotent: converted layers are skipped)
        engram = EngramRows(edir, checkpoint_dir())
        cache_dir = os.environ.get("HY4_TENSOR_CACHE", os.path.expanduser("~/.cache/hybrid-moe/hy4_tt"))
        model = Hy4Model(mesh_device, cfg, max_ctx=max_seq_len + 64, chunk=int(os.environ.get("HY4_PREFILL_CHUNK", "2048")),
                         engram=engram, cache_dir=cache_dir, slots=B)
        engine = Hy4BatchEngine(model) if B > 1 else Hy4Engine(model, SPEC_K)
        logger.info(f"{cls.__name__}: max_model_len={max_seq_len} users={B} layers={len(model.layers)} spec k={SPEC_K}")
        return cls(model, engine, mesh_device, max_seq_len)

    # ------------------------------------------------------------------ KV: a token budget only
    @classmethod
    def get_max_tokens_all_users(cls, model_name="", num_devices=1, tt_data_parallel=1, max_model_len=None,
                                 max_num_seqs=None, **kwargs):
        n = int(max_model_len or 131072) + BLOCK_SIZE
        return math.ceil(n / BLOCK_SIZE) * BLOCK_SIZE * max(1, int(max_num_seqs or 1))

    def allocate_kv_cache(self, kv_cache_shape, dtype, num_layers):
        logger.info(f"KV: vLLM asked for {num_layers} layers of {tuple(kv_cache_shape)}; the engine keeps its own caches")
        return [(None, None)] * int(num_layers)

    def allocate_kv_cache_per_layer(self, per_layer_specs):
        return [(None, None)] * len(per_layer_specs)

    # ------------------------------------------------------------------ warm-up
    def warmup_model_prefill(self, kv_cache, enable_trace, *args, **kwargs):
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
    def prefill_forward(self, tokens, page_table, kv_cache, prompt_lens, **kwargs):
        N = int(tokens.shape[0])
        start_pos = kwargs.get("start_pos")
        if start_pos is not None and any(int(v) for v in torch.as_tensor(start_pos).reshape(-1).tolist()):
            raise NotImplementedError("prefix caching / chunked prefill are off for Hy4 (start_pos must be 0)")
        plens = [int(prompt_lens[u]) for u in range(N)] if prompt_lens is not None else [int(tokens.shape[1])] * N
        sp = kwargs.get("sampling_params")
        if self.multi:
            return self._prefill_multi(tokens, plens, sp, kwargs.get("empty_slots"))
        out = []
        for u in range(N):
            out.append(self._engine.prefill(tokens[u, : plens[u]].tolist(), _mtp_sampling(sp, u)))
        if sp is not None:
            return torch.tensor(out, dtype=torch.int64).reshape(N, 1)
        self._engine.adopt_next = True
        V = int(self._engine.V)
        logits = torch.full((N, 1, V), float("-inf"), dtype=torch.float32)
        logits[:, 0, :] = self._engine.last_logits[:V]
        return logits

    def _prefill_multi(self, tokens, plens, sp, rows):
        N = int(tokens.shape[0])
        rows = list(range(N)) if rows is None else [int(r) for r in rows]
        eng = self._engine
        out, slots = [], []
        for u, row in enumerate(rows):
            old = self._row_slot.pop(row, None)  # the row's previous request is gone
            if old is not None:
                eng.finish(old)
            slot = eng.alloc()
            self._row_slot[row] = slot
            slots.append(slot)
            out.append(eng.prefill(slot, tokens[u, : plens[u]].tolist(), _mtp_sampling(sp, u)))
        if sp is not None:
            return torch.tensor(out, dtype=torch.int64).reshape(N, 1)
        for slot in slots:
            eng.seqs[slot]["adopt"] = True
        V = int(eng.V)
        logits = torch.full((N, 1, V), float("-inf"), dtype=torch.float32)
        for u, slot in enumerate(slots):
            logits[u, 0, :] = eng.last_logits[slot][:V]
        return logits

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
        missing = [i for i in live if i not in self._row_slot]
        if missing:
            raise RuntimeError(f"Hy4 decode rows {missing} have no prefilled slot (rows {self._row_slot})")
        if sp is None:  # host sampling: next-token logits
            V = int(eng.V)
            logits = torch.full((tokens.numel(), 1, V), float("-inf"), dtype=torch.float32)
            res = eng.step_logits({self._row_slot[i]: (int(tokens[i]), int(pos[i])) for i in live}) if live else {}
            for i in live:
                logits[i, 0, :] = res[self._row_slot[i]][:V]
            return logits
        out = torch.full((tokens.numel(), 1), -1, dtype=torch.int32)
        if not live:
            return out
        auth = bool(kwargs.get("reload_inputs", False) or kwargs.get("reset_batch", False))
        reqs = {}
        for i in live:
            slot = self._row_slot[i]
            q = eng.seqs[slot]
            q["smp"] = q["smp"] if not auth else _resample(q["smp"], _mtp_sampling(sp, i))
            reqs[slot] = (int(tokens[i]), int(pos[i]), auth)
        res = eng.step(reqs)
        for i in live:
            out[i, 0] = int(res[self._row_slot[i]][0])
        if self._decode_steps % 1024 == 0:
            logger.info(f"Hy4 batch engine stats: {eng.stats}")
        return out

    def decode_forward(self, *args, **kwargs):
        tokens = torch.as_tensor(kwargs.get("tokens", args[0] if args else None)).reshape(-1)
        pos = torch.as_tensor(kwargs.get("start_pos", args[1] if len(args) > 1 else None)).reshape(-1)
        sp = kwargs.get("sampling_params", args[6] if len(args) > 6 else None)
        if self.multi:
            remap = kwargs.get("slot_remap", args[10] if len(args) > 10 else None)
            if remap is not None:
                self._remap(remap)
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


__all__ = ["HYV4FlashForConditionalGeneration", "HF_MODEL"]
