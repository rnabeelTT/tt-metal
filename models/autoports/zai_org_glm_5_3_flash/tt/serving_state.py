# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Serving-side decode state of the GLM-5.3-Flash generator (stage 9 draft, host logic only).

These helpers belong to the generator side of the vLLM contract (doc/vllm/DESIGN.md, "Required generator / model API
changes"). tt/generator.py calls them; tt/generator_vllm.py does not touch the sampler directly.

  install_lockstep_sampling   every die feeds back the token the host reads (multi-die PRNG drift, port of the
                              Qwen3-Coder-Next fix, qwen_qwen3_coder_next/tt/model.py)
  SlotSamplingState           per-request sampling params, seeds and penalty state in SLOT order (prefill + decode)
  merge_async_ahead           host-authoritative reload that keeps the device's one-step-ahead token / position of the
                              continuing slots (async scheduling: the host view lags the device by one token)

Every index here is a model slot (0..S-1), never a vLLM batch row: the adapter translates rows to slots first.
"""

from __future__ import annotations

import os
from dataclasses import fields

import torch
from loguru import logger

import ttnn
from models.common.sampling.generator import SeedManager, format_sampling_params
from models.common.sampling.sampling_params import SamplingParams

DRAM = ttnn.DRAM_MEMORY_CONFIG


# ------------------------------------------------------------------------------------------------ sampler lockstep
class _LockstepSeedManager(SeedManager):
    """SeedManager whose UNSEEDED path pushes fresh identical seeds before EVERY draw (never the SKIP sentinel).

    ttnn.sampling runs on every die; each die feeds ITS OWN token back through tt_out_tok while the host reads die 0.
    Free-running device PRNGs drift apart with context length on multi-die meshes (122B: 162/600 disagreeing steps at
    95K context; Coder-Next: 196/400 at 20K), so the user sees die 0's token while the model continues from another
    die's token. Fresh identical seeds per draw keep the dies in lockstep. Cost: one 128-byte host write per step.
    """

    def get_new_values(self, empty_slots=None, replicate_seeds=False):
        if self._seed_active:
            return super().get_new_values(empty_slots, replicate_seeds=replicate_seeds)
        self._active_request_seed = False
        new_seeds = [self._next_unseeded_device_seed() for _ in range(self.max_batch_size)]
        self.write_device_seed_values(new_seeds)
        self._reseted = False
        self._needs_skip = False
        return tuple(new_seeds)


def install_lockstep_sampling(sampling, mesh_device):
    """(1) identical fresh seeds before every draw; (2) after ttnn.sampling, all-gather the token row and copy die 0's
    row into the token buffer of every die. GLM53_SAMPLER_TOKEN_BROADCAST=0 disables (2) (A/B only). Call once, right
    after the SamplingGenerator is built and BEFORE any precompile / trace capture (the broadcast is part of the
    captured sampler trace)."""
    if mesh_device.get_num_devices() <= 1:
        return
    old = sampling.seed_manager
    sampling.seed_manager = _LockstepSeedManager(
        sampling.tt_sampling, max_batch_size=old.max_batch_size, salt_duplicate_seeds=old.salt_duplicate_seeds
    )
    ts = sampling.tt_sampling
    orig_forward = ts.forward
    broadcast = os.environ.get("GLM53_SAMPLER_TOKEN_BROADCAST", "1") == "1"

    def forward(x, tt_out_tok=None):
        tokens, log_probs = orig_forward(x, tt_out_tok=tt_out_tok)
        if not broadcast or ts.force_argmax_sampling:
            return tokens, log_probs
        gathered = ttnn.all_gather(tokens, dim=2, memory_config=DRAM)  # [1,1,N,S]
        first = ttnn.slice(gathered, [0, 0, 0, 0], [1, 1, 1, tokens.shape[-1]])
        ttnn.deallocate(gathered)
        ttnn.copy(first, tokens)
        ttnn.deallocate(first)
        return tokens, log_probs

    ts.forward = forward
    logger.info(
        f"sampler lockstep installed on {mesh_device.get_num_devices()} dies (fresh identical seeds every draw"
        f"{', die-0 token broadcast' if broadcast else ''})"
    )


# ------------------------------------------------------------------------------------------------ sampling state
def _field_list(sp, name, n, default):
    v = getattr(sp, name, None)
    if v is None:
        return [default] * n
    if isinstance(v, torch.Tensor):
        v = v.tolist()
    if not isinstance(v, (list, tuple)):
        return [v] * n
    v = list(v)
    return (v + [default] * n)[:n]


_DEFAULTS = {
    "temperature": 0.0,
    "top_k": 1,
    "top_p": 1.0,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "repetition_penalty": 1.0,
    "seed": None,
    "enable_log_probs": False,
    "num_logprobs": 0,
}


def rows_to_slots(sp, slots, S: int) -> SamplingParams:
    """Per-row sampling params (vLLM TTSamplingParams or models.common SamplingParams, lists in ROW order) -> a
    SamplingParams with S entries in SLOT order: slot slots[r] gets row r's values, unused slots get the defaults
    (greedy, no penalties, unseeded). Rows past len(slots) are padding and are dropped."""
    n = len(slots)
    out = {}
    for f in fields(SamplingParams):
        rows = _field_list(sp, f.name, n, _DEFAULTS[f.name])
        col = [_DEFAULTS[f.name]] * S
        for r, s in enumerate(slots):
            col[int(s)] = rows[r]
        out[f.name] = col
    out["seed"] = [None if (s is None or int(s) < 0) else int(s) for s in out["seed"]]
    return SamplingParams(**out)


def rows_to_slots_tokens(t, slots, S: int, fill: int = -1):
    """[B_rows, L] host token history (prompt / output tokens, -1 padded) -> [S, L] in slot order."""
    if t is None:
        return None
    t = torch.as_tensor(t)
    out = torch.full((S, t.shape[-1]), fill, dtype=t.dtype)
    for r, s in enumerate(slots):
        if r < t.shape[0]:
            out[int(s)] = t[r]
    return out


class SlotSamplingState:
    """Drives models.common.sampling.SamplingGenerator for a slot-indexed batch: the tt_transformers decode-sampling
    sequence (apply params, seed registration / position alignment, penalty history) without the tt_transformers
    Generator. One instance per Glm53Generator."""

    def __init__(self, sampling, S: int):
        self.sampling = sampling
        self.S = S
        self.live = set()

    # ---- prefill (one request, eager sampler on the prefill's last-row logits)
    def apply_prefill(self, sp_row: SamplingParams, slot: int, prompt_tokens: torch.Tensor | None):
        """sp_row: ONE request's params (scalars or length-1 lists). The params are broadcast to every sampler row so
        any row of the replicated logits samples with them; the caller reads row `slot`."""
        sm = self.sampling
        # drop seed state of slots that are neither running nor being admitted (ghost seeds change salts)
        sm.seed_manager.deactivate_slots_except(sorted(self.live - {slot}))
        one = {f.name: _field_list(sp_row, f.name, 1, _DEFAULTS[f.name])[0] for f in fields(SamplingParams)}
        bcast = SamplingParams(**{k: [v] * self.S for k, v in one.items()})
        formatted = format_sampling_params(bcast, self.S)
        rep_prompt = None
        if prompt_tokens is not None:
            rep_prompt = torch.as_tensor(prompt_tokens).reshape(1, -1).to(torch.long).repeat(self.S, 1)
        sm.apply_prefill_state(sampling_params=formatted, prompt_tokens=rep_prompt, empty_slots=[int(slot)])
        self.live.add(int(slot))

    # ---- decode
    def apply_decode(
        self,
        sp_slots: SamplingParams,
        positions_slots: list[int],
        *,
        reload_params: bool,
        reset_state: bool,
        host_authoritative: bool,
        prompt_tokens_slots=None,
        output_tokens_slots=None,
    ):
        """sp_slots / positions_slots: S entries in slot order (position -1 = inactive slot). reload_params: the batch
        params changed (contract v1 reload_sampling_params); reset_state: reset penalty history + seeds (contract v1
        reset_sampling_state); host_authoritative: positions are exact (reload_inputs) and may re-anchor seed
        counters. Call once per decode step, before the sampler replay."""
        sm = self.sampling
        if reload_params or reset_state:
            sm.apply_decode_state(
                [sp_slots],
                reset_batch=reset_state,
                prompt_tokens=prompt_tokens_slots,
                output_tokens=output_tokens_slots,
            )
        active = [s for s, p in enumerate(positions_slots) if int(p) >= 0]
        self.live = set(active)
        sm.seed_manager.deactivate_slots_except(active)
        if active:
            seeds = format_sampling_params(sp_slots, self.S).seed
            if reset_state:
                sm.seed_manager.reset_seed_from_slots(seeds, active)
                reseeded = list(active)
            else:
                reseeded = sm.seed_manager.reset_seed_from_slots_if_needed(seeds, active)
            align = active if host_authoritative else reseeded
            if align:
                sm.seed_manager.align_seed_counters_to_positions(seeds, align, list(positions_slots))
        sm.seed_manager.get_new_values(active or None)


# ------------------------------------------------------------------------------------------------ async-ahead merge
def merge_async_ahead(host_tok, host_pos, dev_tok, dev_pos, fresh_slots=()):
    """Slot-ordered [S] tensors. Under async scheduling the host's last token / position of a CONTINUING slot may lag
    the device by one step (the sampler already fed token N back on device). On a host-authoritative reload keep the
    device pair for such slots (device position == host or host + 1); slots prefilled since the last decode and slots
    whose positions disagree otherwise take the host pair. Returns (tokens, positions) in slot order."""
    host_tok = torch.as_tensor(host_tok).reshape(-1).to(torch.int64)
    host_pos = torch.as_tensor(host_pos).reshape(-1).to(torch.int64)
    dev_tok = torch.as_tensor(dev_tok).reshape(-1).to(torch.int64)[: host_tok.numel()]
    dev_pos = torch.as_tensor(dev_pos).reshape(-1).to(torch.int64)[: host_pos.numel()]
    use_dev = (host_pos >= 0) & ((dev_pos == host_pos) | (dev_pos == host_pos + 1))
    for s in fresh_slots:
        if 0 <= int(s) < use_dev.numel():
            use_dev[int(s)] = False
    return torch.where(use_dev, dev_tok, host_tok), torch.where(use_dev, dev_pos, host_pos)
