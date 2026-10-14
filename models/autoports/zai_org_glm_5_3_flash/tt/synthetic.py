# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Weight statistics of the real checkpoint and deterministic synthetic weights generated from them.

Pure torch (imported by the HF reference venv and by the TTNN tests). Synthetic tensors use checkpoint names and real
shapes; large tensors are N(mean, std), small tensors (<= SMALL elements: norms, gate biases, A_log, dt_bias, mHC
base/scale) are stored verbatim, because their exact values set the operating point (decay rates, mixing weights).
"""

from __future__ import annotations

import base64
import hashlib

import torch

SMALL = 16384
# representative layers per kind: KDA + dense MLP, DSA + MoE, KDA + MoE
STATS_LAYERS = (0, 3, 4)


def layer_kind(layer: int) -> str:
    attn = "dsa" if layer % 4 == 3 else "kda"
    mlp = "dense" if layer < 3 else "moe"
    return f"{attn}_{mlp}"


def stats_layer_for(layer: int) -> int:
    return {"kda_dense": 0, "dsa_moe": 3, "kda_moe": 4}[layer_kind(layer)]


def tensor_stats(t: torch.Tensor) -> dict:
    f = t.float()
    rec = {"shape": list(t.shape), "mean": float(f.mean()), "std": float(f.std()) if f.numel() > 1 else 0.0}
    if t.numel() <= SMALL:  # exact values, native bytes (bf16 / fp32) base64-encoded
        rec["b64"] = base64.b64encode(t.contiguous().flatten().view(torch.uint8).numpy().tobytes()).decode()
    return rec


def collect_stats(rd, layers=STATS_LAYERS, expert_sample: int = 8) -> dict:
    """{str(layer): {suffix: stats}}; routed experts summarised as 'mlp.experts.*.<proj>.weight' from a sample."""
    out = {"num_experts": rd.num_experts, "layers": {}}
    for layer in layers:
        rec = {}
        for suffix in rd.layer_names(layer):
            name = f"model.language_model.layers.{layer}.{suffix}"
            fp32 = rd.raw(name).dtype == torch.float32
            t = rd.get(name, dtype=torch.float32 if fp32 else torch.bfloat16)  # router rows masked to the E192 set
            rec[suffix] = tensor_stats(t)
            rec[suffix]["dtype"] = "float32" if fp32 else "bfloat16"
        if layer_kind(layer).endswith("moe"):
            for proj in ("gate_proj", "up_proj", "down_proj"):
                ts = [rd.expert(layer, e, proj) for e in range(0, rd.num_experts, rd.num_experts // expert_sample)]
                st = tensor_stats(torch.stack(ts))
                st.pop("values", None)
                st["shape"] = list(ts[0].shape)
                st["dtype"] = "bfloat16"
                rec[f"mlp.experts.*.{proj}.weight"] = st
        out["layers"][str(layer)] = rec
    return out


def _generator(seed: int, layer: int, name: str) -> torch.Generator:
    h = int.from_bytes(hashlib.sha256(f"{seed}/{layer}/{name}".encode()).digest()[:8], "little") & ((1 << 63) - 1)
    return torch.Generator().manual_seed(h)


def _make(rec: dict, seed: int, layer: int, name: str) -> torch.Tensor:
    dtype = torch.float32 if rec.get("dtype") == "float32" else torch.bfloat16
    if "b64" in rec:
        raw = bytearray(base64.b64decode(rec["b64"]))
        return torch.frombuffer(raw, dtype=dtype).clone().reshape(rec["shape"])
    g = _generator(seed, layer, name)
    t = torch.randn(rec["shape"], generator=g, dtype=torch.float32) * rec["std"] + rec["mean"]
    return t.to(dtype)


def synthetic_layer_state(stats: dict, layer: int, seed: int = 0) -> dict[str, torch.Tensor]:
    """Checkpoint-named synthetic tensors for decoder layer `layer` (shapes and kind of the real layer)."""
    rec = stats["layers"][str(stats_layer_for(layer))]
    num_experts = int(stats["num_experts"])
    sd = {}
    for suffix, st in rec.items():
        if suffix.startswith("mlp.experts.*."):
            proj = suffix.split(".")[3]
            for e in range(num_experts):
                sd[f"mlp.experts.{e}.{proj}.weight"] = _make(st, seed, layer, f"mlp.experts.{e}.{proj}")
            continue
        sd[suffix] = _make(st, seed, layer, suffix)
    return sd
