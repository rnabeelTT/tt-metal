# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Size a qwen4_exp (Qwen3.8-Flash-Next-style) checkpoint against Blackhole profiles.

Answers, for a config that may not exist yet: how many parameters each part holds, how many bytes each part takes
on device at the port's precisions, whether it fits a p150 / p300 / p300x2 (= p150x4) mesh, and where the PLE n-gram
table has to live (device DRAM, host page cache, or NVMe read per token).

    python -m models.demos.blackhole.qwen4_prep.sizing --config CONFIG.json [--set hidden_size=5120 ...]
    python -m models.demos.blackhole.qwen4_prep.sizing --scenarios        # the built-in Qwen4 guesses

Calibration: Qwen3.8-Flash-Next's own config gives 180.0B parameters = 360.0 GB in bf16, the checkpoint index's
total_size (359,999,963,128 bytes) to 0.01 %.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
GB = 1e9

# Bytes per value on device. BFP formats share one exponent byte per 16 values.
BYTES = {"bf16": 2.0, "bf8": 17 / 16, "bf4": 9 / 16, "fp32": 4.0}

# 32 GB of DRAM per Blackhole chip; keep a reserve per chip for activations, traces, CCL buffers and the program
# cache (the Ornith/Qwen3.5 bundles run with 2-4 GB free per chip after capture).
CHIP_DRAM_GB = 32.0
CHIP_RESERVE_GB = 3.0
PROFILES = {"p150": 1, "p300": 2, "p300x2/p150x4": 4, "galaxy": 32}


@dataclass
class Arch:
    hidden: int
    layers: int
    layer_types: list[str]
    vocab: int
    tie_embeddings: bool
    # gated residual (hyper-connections); hc=1 and rank=0 means a plain residual stream
    hc: int
    hc_rank: int
    # GDN
    gdn_k_heads: int
    gdn_v_heads: int
    gdn_k_dim: int
    gdn_v_dim: int
    conv_k: int
    # attention (QSA when the indexer fields are set)
    q_heads: int
    kv_heads: int
    head_dim: int
    gated_attn: bool
    idx_q_heads: int
    idx_kv_heads: int
    idx_dim: int
    idx_budget: int
    idx_ratio: int
    # FFN: num_experts == 0 means a dense MLP of width `intermediate`
    num_experts: int
    top_k: int
    expert_inter: int
    shared_inter: int
    intermediate: int
    # PLE n-gram embedding
    ple_layers: int
    ple_dim: int
    ngram_size: int
    heads_per_ngram: int
    ngram_base: int
    ple_conv_k: int
    mtp_layers: int
    vision: dict = field(default_factory=dict)

    @classmethod
    def from_config(cls, document: dict, overrides: dict | None = None) -> "Arch":
        t = dict(document.get("text_config", document))
        t.update(overrides or {})
        layers = int(t["num_hidden_layers"])
        types = t.get("layer_types")
        if not types or len(types) != layers:
            interval = int(t.get("full_attention_interval", 4))
            types = ["full_attention" if (i + 1) % interval == 0 else "linear_attention" for i in range(layers)]
        hidden = int(t["hidden_size"])
        num_experts = int(t.get("num_experts") or 0)
        return cls(
            hidden=hidden,
            layers=layers,
            layer_types=list(types),
            vocab=int(t["vocab_size"]),
            tie_embeddings=bool(t.get("tie_word_embeddings", False)),
            hc=int(t.get("hc_count") or 1),
            hc_rank=int(t.get("hc_lowrank") or 0),
            gdn_k_heads=int(t.get("linear_num_key_heads", 0)),
            gdn_v_heads=int(t.get("linear_num_value_heads", 0)),
            gdn_k_dim=int(t.get("linear_key_head_dim", 128)),
            gdn_v_dim=int(t.get("linear_value_head_dim", 128)),
            conv_k=int(t.get("linear_conv_kernel_dim", 4)),
            q_heads=int(t["num_attention_heads"]),
            kv_heads=int(t["num_key_value_heads"]),
            head_dim=int(t.get("head_dim") or hidden // int(t["num_attention_heads"])),
            gated_attn=bool(t.get("attn_output_gate", t.get("output_gate_type") is not None)),
            idx_q_heads=int(t.get("indexer_n_heads") or 0),
            idx_kv_heads=int(t.get("indexer_kv_heads") or 0),
            idx_dim=int(t.get("indexer_head_dim") or 0),
            idx_budget=int(t.get("indexer_budget") or 0),
            idx_ratio=int(t.get("indexer_compress_ratio") or 1),
            num_experts=num_experts,
            top_k=int(t.get("num_experts_per_tok") or 0),
            expert_inter=int(t.get("moe_intermediate_size") or 0),
            shared_inter=int(t.get("shared_expert_intermediate_size") or 0),
            intermediate=int(t.get("intermediate_size") or 0),
            ple_layers=len(t.get("ple_layer_ids") or []),
            ple_dim=int(t.get("ple_embed_dim") or hidden),
            ngram_size=int(t.get("ngram_size") or 0),
            heads_per_ngram=int(t.get("heads_per_ngram") or 0),
            ngram_base=int(t.get("ngram_vocab_size_base") or 0),
            ple_conv_k=int(t.get("ple_conv_kernel_size") or 4),
            mtp_layers=int(t.get("mtp_num_hidden_layers") or 0),
            vision=dict(document.get("vision_config") or {}),
        )

    @property
    def n_gdn(self) -> int:
        return sum(1 for x in self.layer_types if x == "linear_attention")

    @property
    def n_attn(self) -> int:
        return self.layers - self.n_gdn

    @property
    def stream(self) -> int:
        return self.hc * self.hidden


def _gr(a: Arch) -> int:
    """One gated-residual read/write pair (hc_norm, low-rank read gate, per-branch write)."""
    if a.hc <= 1:
        return a.hidden  # a plain RMSNorm
    w = a.stream
    return w + 2 * a.hc_rank * w + a.hc * w


def _mixer(a: Arch) -> int:
    return a.hidden if a.hc <= 1 else a.stream + 2 * a.hc_rank * a.stream


def _gdn(a: Arch) -> int:
    qkv = 2 * a.gdn_k_heads * a.gdn_k_dim + a.gdn_v_heads * a.gdn_v_dim
    v = a.gdn_v_heads * a.gdn_v_dim
    h = a.hidden
    return qkv * h + v * h + 2 * a.gdn_v_heads * h + qkv * a.conv_k + 2 * a.gdn_v_heads + a.gdn_v_dim + h * v


def _attn(a: Arch) -> int:
    h, d = a.hidden, a.head_dim
    q = a.q_heads * d * (2 if a.gated_attn else 1)
    p = q * h + 2 * a.kv_heads * d * h + h * a.q_heads * d + 2 * d
    if a.idx_q_heads:
        p += (a.idx_q_heads + a.idx_kv_heads) * a.idx_dim * h + 2 * a.idx_dim
    return p


def _ffn_dense_part(a: Arch) -> int:
    """Everything in the FFN except the routed experts."""
    h = a.hidden
    if not a.num_experts:
        return 3 * a.intermediate * h
    return a.num_experts * h + 3 * a.shared_inter * h + h


def _experts(a: Arch) -> int:
    return a.num_experts * 3 * a.expert_inter * a.hidden


def ple_table_params(a: Arch) -> int:
    """Sum of the per-head prime vocabularies (~ngram_base each) times the per-head row width."""
    if not a.ple_layers:
        return 0
    heads = (a.ngram_size - 1) * a.heads_per_ngram
    return a.ple_layers * heads * a.ngram_base * (a.ple_dim // heads)


def _ple_proj(a: Arch) -> int:
    if not a.ple_layers:
        return 0
    w = a.stream
    return a.ple_layers * (w * a.ple_dim + a.hidden * a.ple_dim + w * a.ple_conv_k + 3 * w)


def _vision(a: Arch) -> int:
    v = a.vision
    if not v:
        return 0
    d, f = v["hidden_size"], v["intermediate_size"]
    block = 4 * d * d + 4 * d + 2 * d * f + d + f + 4 * d
    patch = v.get("in_channels", 3) * v.get("temporal_patch_size", 2) * v["patch_size"] ** 2 * d
    m = v.get("spatial_merge_size", 2) ** 2 * d
    merger = m * m + m * v["out_hidden_size"] + 2 * m
    return v["depth"] * block + patch + v.get("num_position_embeddings", 0) * d + merger


def breakdown(a: Arch) -> dict[str, int]:
    layer_common = a.layers * (2 * _gr(a) + _ffn_dense_part(a))
    parts = {
        "embed+lm_head": a.vocab * a.hidden * (1 if a.tie_embeddings else 2),
        "gdn layers": a.n_gdn * _gdn(a),
        "attention layers": a.n_attn * _attn(a),
        "gated residual + router + shared/dense FFN": layer_common + _mixer(a),
        "routed experts": a.layers * _experts(a),
        "ple projections": _ple_proj(a),
        "ple n-gram table": ple_table_params(a),
        "mtp": a.mtp_layers * (_attn(a) + 2 * _gr(a) + _ffn_dense_part(a) + _experts(a))
        + (2 * a.hidden * a.hidden + a.hidden + a.stream + _mixer(a) if a.mtp_layers else 0),
        "vision": _vision(a),
    }
    return parts


@dataclass
class Precision:
    experts: str = "bf4"
    dense: str = "bf8"
    embed: str = "bf8"
    ple: str = "bf16"  # a device gather (ttnn.embedding) reads row-major bf16 rows
    kv: str = "bf8"


def device_bytes(a: Arch, parts: dict[str, int], prec: Precision, *, mtp: bool, vision: bool) -> dict[str, float]:
    b = {
        "embed+lm_head": parts["embed+lm_head"] * BYTES[prec.embed],
        "dense": (
            parts["gdn layers"]
            + parts["attention layers"]
            + parts["gated residual + router + shared/dense FFN"]
            + parts["ple projections"]
        )
        * BYTES[prec.dense],
        "routed experts": parts["routed experts"] * BYTES[prec.experts],
    }
    if mtp:
        b["mtp"] = parts["mtp"] * BYTES[prec.experts if a.num_experts else prec.dense]
    if vision:
        b["vision"] = parts["vision"] * BYTES["bf16"]
    return b


def context_bytes(a: Arch, ctx: int, users: int, prec: Precision) -> float:
    """KV + indexer key cache for the attention layers, fp32 recurrent state for GDN, per user."""
    kv = a.n_attn * 2 * a.kv_heads * a.head_dim * ctx * BYTES[prec.kv]
    idx = a.n_attn * a.idx_kv_heads * a.idx_dim * math.ceil(ctx / max(a.idx_ratio, 1)) * BYTES["bf16"]
    gdn = a.n_gdn * (
        a.gdn_v_heads * a.gdn_k_dim * a.gdn_v_dim * BYTES["fp32"]
        + (2 * a.gdn_k_heads * a.gdn_k_dim + a.gdn_v_heads * a.gdn_v_dim) * (a.conv_k - 1) * BYTES["bf16"]
    )
    return users * (kv + idx + gdn)


def ple_placement(table_gb: float, free_device_gb: float, host_ram_gb: float) -> str:
    if table_gb == 0:
        return "none"
    if table_gb <= free_device_gb:
        return "device"
    if table_gb <= 0.8 * host_ram_gb:
        return "host page cache"
    return "NVMe per-token reads"


def report(name: str, a: Arch, *, prec: Precision, ctx: int, users: int, host_ram_gb: float, mtp: bool, vision: bool):
    parts = breakdown(a)
    total = sum(parts.values())
    print(f"\n== {name}")
    print(
        f"   hidden {a.hidden}, {a.layers} layers ({a.n_gdn} GDN / {a.n_attn} attn), "
        f"{'%d experts top-%d inter %d' % (a.num_experts, a.top_k, a.expert_inter) if a.num_experts else 'dense FFN %d' % a.intermediate}, "
        f"GR {a.hc}x{a.hc_rank}, PLE base {a.ngram_base / 1e6:g}M x dim {a.ple_dim}"
    )
    for k, v in parts.items():
        if v:
            print(f"   {k:<44s} {v / 1e9:8.2f} B")
    print(f"   {'total (bf16 checkpoint ' + format(2 * total / GB, '.1f') + ' GB)':<44s} {total / 1e9:8.2f} B")
    active = total - parts["routed experts"] - parts["ple n-gram table"] - parts["mtp"] - parts["vision"]
    if a.num_experts:
        active += a.layers * a.top_k * 3 * a.expert_inter * a.hidden
    print(f"   active per token (no PLE/MTP/vision)          {active / 1e9:8.2f} B")

    dev = device_bytes(a, parts, prec, mtp=mtp, vision=vision)
    state = context_bytes(a, ctx, users, prec)
    weights_gb = sum(dev.values()) / GB
    table_gb = parts["ple n-gram table"] * BYTES[prec.ple] / GB
    print(
        f"   device: weights {weights_gb:.1f} GB ({', '.join(f'{k} {v / GB:.1f}' for k, v in dev.items())}), "
        f"state {state / GB:.1f} GB @ {ctx // 1024}k x {users} user(s); PLE table {table_gb:.1f} GB ({prec.ple})"
    )
    print(f"   {'profile':<15s} {'usable GB':>9s} {'free after weights+state':>25s}  PLE table goes to")
    for prof, chips in PROFILES.items():
        usable = chips * (CHIP_DRAM_GB - CHIP_RESERVE_GB)
        free = usable - weights_gb - state / GB
        where = ple_placement(table_gb, free, host_ram_gb) if free >= 0 else "-- model does not fit --"
        print(f"   {prof:<15s} {usable:9.0f} {free:25.1f}  {where}")


def scenarios() -> list[tuple[str, dict, dict]]:
    """Guesses for Qwen4 variants smaller than Flash-Next, built on released Qwen3.5/3.8 shapes plus the
    qwen4_exp blocks (GR, QSA, PLE) at Flash-Next's ratios: rank = hidden / 8, PLE dim = hidden."""
    fn = json.loads((HERE / "flash_next_config.json").read_text())
    dense27 = json.loads((HERE / "qwen38_27b_config.json").read_text())
    q4_blocks = {
        "hc_count": 4,
        "indexer_n_heads": 4,
        "indexer_kv_heads": 1,
        "indexer_head_dim": 128,
        "indexer_budget": 2048,
        "indexer_compress_ratio": 4,
        "ple_layer_ids": [2],
        "ngram_size": 3,
        "heads_per_ngram": 8,
        "ple_conv_kernel_size": 4,
        "mtp_num_hidden_layers": 1,
    }
    out = [("Qwen3.8-Flash-Next (released)", fn, {})]
    for base in (20_000_000, 5_000_000, 1_000_000):
        out.append(
            (
                f"guess: Qwen4-27B dense (Qwen3.8-27B shape + qwen4 blocks), PLE base {base // 10**6}M",
                {"text_config": dense27["text_config"], "vision_config": fn["vision_config"]},
                {**q4_blocks, "hc_lowrank": 640, "ple_embed_dim": 5120, "ngram_vocab_size_base": base},
            )
        )
    for base in (20_000_000, 5_000_000):
        out.append(
            (
                f"guess: Qwen4-35B-A3B (Qwen3.5-35B-A3B shape + qwen4 blocks), PLE base {base // 10**6}M",
                {"text_config": fn["text_config"], "vision_config": fn["vision_config"]},
                {
                    **q4_blocks,
                    "hidden_size": 2048,
                    "num_hidden_layers": 40,
                    "layer_types": None,
                    "linear_num_value_heads": 32,
                    "num_attention_heads": 16,
                    "num_experts": 256,
                    "num_experts_per_tok": 8,
                    "moe_intermediate_size": 512,
                    "shared_expert_intermediate_size": 512,
                    "hc_lowrank": 256,
                    "ple_embed_dim": 2048,
                    "ngram_vocab_size_base": base,
                },
            )
        )
    return out


def _parse_set(items: list[str]) -> dict:
    out = {}
    for item in items:
        key, _, value = item.partition("=")
        out[key] = json.loads(value)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, help="config.json of a qwen4_exp (or qwen3_5) checkpoint")
    p.add_argument("--set", nargs="*", default=[], help="text_config overrides, key=json_value")
    p.add_argument("--scenarios", action="store_true", help="run the built-in Qwen4 size guesses")
    p.add_argument("--context", type=int, default=32768)
    p.add_argument("--users", type=int, default=1)
    p.add_argument("--host-ram-gb", type=float, default=249.0)
    p.add_argument("--experts", default="bf4", choices=BYTES)
    p.add_argument("--dense", default="bf8", choices=BYTES)
    p.add_argument("--ple", default="bf16", choices=BYTES)
    p.add_argument("--no-mtp", action="store_true")
    p.add_argument("--no-vision", action="store_true")
    args = p.parse_args()
    prec = Precision(experts=args.experts, dense=args.dense, ple=args.ple)
    kw = dict(
        prec=prec,
        ctx=args.context,
        users=args.users,
        host_ram_gb=args.host_ram_gb,
        mtp=not args.no_mtp,
        vision=not args.no_vision,
    )
    runs = []
    if args.config:
        runs.append((str(args.config), json.loads(args.config.read_text()), _parse_set(args.set)))
    if args.scenarios or not runs:
        runs += scenarios()
    for name, doc, overrides in runs:
        report(name, Arch.from_config(doc, overrides), **kw)


if __name__ == "__main__":
    main()
