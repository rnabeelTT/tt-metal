# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Serving profiles of GLM-5.3-Flash (E192) on a (1, 4) Blackhole mesh: ONE data file, ONE sizing rule.

Data: `serving_profiles.json` next to `tt-model.yaml` (GLM53_SERVING_PROFILES overrides the path). The capacity probe
(tools/memory_probe.py capacity) measures the numbers; tools/serving_profiles.py records them in the file and renders
the `serve` / `serve_profiles` sections of tt-model.yaml from it. Readers:

  tt/generator_vllm.py        get_max_tokens_all_users (vLLM block count), max_model_len check, trace region, layout
  tools/serving_profiles.py   ingest / refresh / render-yaml / check
  ~/glm53-flash-bringup/scripts/09-serve.sh, 09-g9-probe.sh   (profile -> max_num_seqs, max_model_len, trace region)
  tests/test_generator_vllm_host.py

Sizing rule (per die; the DSA pool is replicated on every die, the KDA state is per slot):
  pool_blocks(S)    = measured[S].pool_blocks if the capacity probe measured S slots, else
                      floor((free_after_load - (kda_slots(S) - 2) * kda_bytes_per_slot - reserve) / dsa_bytes_per_block)
                      (free_after_load is measured with 2 KDA slots: a 1-slot build pads KDA to 2 slots)
  vLLM num_blocks   = pool_blocks - 1  (the model adds its dummy block: Glm53Generator.bind_kv_cache(num_blocks))
  max_model_len(S)  = largest L, a multiple of context_granularity(L), with
                      ceil(L / 128) + (S - 1) + 2 <= pool_blocks(S)
                      (one request of L tokens, one block for every other slot, vLLM's null block, the dummy block:
                      tt/generator.blocks_needed(serving=True))

This module has no ttnn / torch import: the bundle's verify step, the scripts and the host tests import it alone.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parents[1]
DEFAULT_PATH = MODEL_DIR / "serving_profiles.json"
BLOCK = 128  # DSA page block (tt/dsa.py BLOCK); vLLM --block-size


def path() -> Path:
    p = os.environ.get("GLM53_SERVING_PROFILES")
    return Path(p) if p else DEFAULT_PATH


def load(p: str | os.PathLike | None = None) -> dict:
    p = Path(p) if p is not None else path()
    if not p.exists():
        raise FileNotFoundError(
            f"serving profiles {p} not found: the vLLM adapter sizes the DSA pool from this file "
            f"(tools/serving_profiles.py writes it)"
        )
    return json.loads(p.read_text())


# ------------------------------------------------------------------------------------------------ sizing rule
def kda_slots(S: int) -> int:
    """MultichipKDA pads the decode batch to fill tile rows (16 local heads): an odd S gets one more slot."""
    return S + S % 2


def context_granularity(T: int) -> int:
    """max_context must be a multiple of the largest prefill length that fits it (tt/functional_decoder asserts it).
    Same rule as tt/generator.context_granularity (the host test checks that they agree)."""
    return 2048 if T >= 2048 else (512 if T >= 512 else 128)


def floor_to_granularity(T: int) -> int:
    if T < BLOCK:
        return 0
    g = context_granularity(T)
    return T // g * g


def model_context(max_model_len: int) -> int:
    """The model's max_context for a vLLM max_model_len: rounded up to 128, then to the granularity of that value."""
    t = math.ceil(int(max_model_len) / BLOCK) * BLOCK
    g = context_granularity(t)
    return math.ceil(t / g) * g


def blocks_needed(S: int, max_model_len: int) -> int:
    return math.ceil(int(max_model_len) / BLOCK) + (int(S) - 1) + 2


def _measured(data: dict, S: int) -> dict | None:
    m = (data.get("measured") or {}).get(str(int(S)))
    return m if m and m.get("pool_blocks") else None


def pool_blocks(data: dict, S: int, extra_reserve_bytes: int = 0) -> int:
    """DSA pool blocks per die (incl. the model's dummy block) at S slots."""
    m = _measured(data, S)
    if m is not None:
        per_block = int(data["budget"]["dsa_bytes_per_block"])
        return max(0, int(m["pool_blocks"]) - math.ceil(int(extra_reserve_bytes) / per_block))
    b = data["budget"]
    avail = (
        int(b["free_after_load_bytes"])
        - (kda_slots(S) - 2) * int(b["kda_bytes_per_slot"])
        - int(b["reserve_bytes"])
        - int(extra_reserve_bytes)
    )
    return max(0, avail // int(b["dsa_bytes_per_block"]))


def pool_source(data: dict, S: int) -> str:
    m = _measured(data, S)
    return (
        f"measured ({m.get('evidence', '?')})"
        if m is not None
        else f"budget rule ({data['budget'].get('status', '?')})"
    )


def max_model_len_for(data: dict, S: int, extra_reserve_bytes: int = 0) -> int:
    """Largest max_model_len that fits the DSA pool at S slots (0 if none)."""
    nb = pool_blocks(data, S, extra_reserve_bytes)
    return floor_to_granularity((nb - 2 - (int(S) - 1)) * BLOCK)


def max_tokens_all_users(data: dict, S: int, extra_reserve_bytes: int = 0) -> int:
    """The value for the plugin's get_max_tokens_all_users. The plugin adds one block per user and rounds up to whole
    blocks, so its num_blocks = pool_blocks - 1; the model adds the dummy block."""
    return (pool_blocks(data, S, extra_reserve_bytes) - 1) * BLOCK - BLOCK * int(S)


def check_fit(data: dict, S: int, max_model_len: int, extra_reserve_bytes: int = 0) -> None:
    nb = pool_blocks(data, S, extra_reserve_bytes)
    need = blocks_needed(S, max_model_len)
    if need > nb:
        raise ValueError(
            f"max_model_len {max_model_len} does not fit the DSA pool at max_num_seqs {S}: needs {need} blocks of "
            f"{BLOCK} tokens, the pool holds {nb} per die ({pool_source(data, S)}); the largest max_model_len here is "
            f"{max_model_len_for(data, S, extra_reserve_bytes)} ({path()})"
        )


# ------------------------------------------------------------------------------------------------ profiles
def profiles(data: dict) -> list[dict]:
    return list(data.get("profiles") or [])


def profile(data: dict, name: str | None = None) -> dict:
    """A profile by name (None = default_profile)."""
    name = name or data.get("default_profile")
    for p in profiles(data):
        if p["name"] == name:
            return p
    raise KeyError(f"no serving profile {name!r} in {path()} (have {[p['name'] for p in profiles(data)]})")


def supported_context(data: dict) -> int:
    """The context-contract value: the 1-slot limit (the largest context the DRAM allows)."""
    return max_model_len_for(data, 1)


def validate(data: dict) -> list[str]:
    """Consistency problems of the data file (empty = consistent)."""
    errs = []
    if int(data.get("block_size", BLOCK)) != BLOCK:
        errs.append(f"block_size must be {BLOCK}")
    for k in ("free_after_load_bytes", "kda_bytes_per_slot", "dsa_bytes_per_block", "reserve_bytes"):
        if not isinstance((data.get("budget") or {}).get(k), int):
            errs.append(f"budget.{k} missing or not an int")
    if errs:
        return errs
    names = [p["name"] for p in profiles(data)]
    if data.get("default_profile") not in names:
        errs.append(f"default_profile {data.get('default_profile')!r} is not a profile")
    for p in profiles(data):
        S, L = int(p["max_num_seqs"]), int(p["max_model_len"])
        if blocks_needed(S, L) > pool_blocks(data, S):
            errs.append(
                f"profile {p['name']}: max_model_len {L} does not fit at {S} slots "
                f"(largest {max_model_len_for(data, S)})"
            )
    return errs
