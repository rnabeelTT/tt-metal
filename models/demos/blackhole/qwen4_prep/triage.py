# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""First command to run on a new Qwen4 checkpoint: what changed against Qwen3.8-Flash-Next, and where to start.

Reads only config.json, model.safetensors.index.json and the safetensors headers (HTTP range requests; no weights are
downloaded), then prints:

1. the architecture class and whether the installed Transformers knows it;
2. every text_config field that differs from Flash-Next, is new, or is gone;
3. every tensor-name pattern that is new / gone / reshaped against the stored Flash-Next snapshot;
4. the sizing table (``sizing.py``) for p150 / p300 / p300x2 and where the PLE table goes;
5. a recommended starting point (README.md, "Playbook").

    python3 models/demos/blackhole/qwen4_prep/triage.py Qwen/Qwen4-XXB [--revision main] [--json out.json]
    python3 models/demos/blackhole/qwen4_prep/triage.py --snapshot Qwen/Qwen3.8-Flash-Next   # refresh the baseline

Needs only the Python standard library (HF_TOKEN is used when set, for gated repos).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import struct
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sizing  # noqa: E402

BASELINE_CONFIG = HERE / "flash_next_config.json"
BASELINE_TENSORS = HERE / "flash_next_tensors.json"


def _get(url: str, byte_range: str | None = None) -> bytes:
    request = urllib.request.Request(url)
    if byte_range:
        request.add_header("Range", f"bytes={byte_range}")
    if os.environ.get("HF_TOKEN"):
        request.add_header("Authorization", f"Bearer {os.environ['HF_TOKEN']}")
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def pattern(name: str) -> str:
    """Layer indices and shard numbers collapsed, so one pattern stands for every layer's copy."""
    return re.sub(r"\.\d+\.", ".N.", re.sub(r"shard_\d+", "shard_K", name))


def fetch(repo: str, revision: str) -> tuple[dict, dict[str, dict]]:
    """config.json and {pattern: {"shape", "dtype", "count", "example"}} for the repo."""
    base = f"https://huggingface.co/{repo}/resolve/{revision}/"
    config = json.loads(_get(base + "config.json"))
    try:
        weight_map = json.loads(_get(base + "model.safetensors.index.json"))["weight_map"]
    except Exception:  # single-file checkpoint
        weight_map = None
    shards = sorted(set(weight_map.values())) if weight_map else ["model.safetensors"]

    def header(shard: str) -> dict:
        size = struct.unpack("<Q", _get(base + shard, "0-7"))[0]
        return json.loads(_get(base + shard, f"8-{7 + size}"))

    with ThreadPoolExecutor(16) as pool:
        headers = list(pool.map(header, shards))
    out: dict[str, dict] = {}
    for h in headers:
        for name, meta in h.items():
            if name == "__metadata__":
                continue
            entry = out.setdefault(
                pattern(name), {"shape": meta["shape"], "dtype": meta["dtype"], "count": 0, "example": name}
            )
            entry["count"] += 1
    return config, dict(sorted(out.items()))


def text_fields(config: dict) -> dict:
    return dict(config.get("text_config", config))


def diff_config(new: dict, old: dict) -> list[str]:
    a, b = text_fields(new), text_fields(old)
    lines = []
    for key in sorted(set(a) | set(b)):
        if key == "layer_types":
            continue
        if key not in b:
            lines.append(f"  NEW      {key} = {a[key]!r}")
        elif key not in a:
            lines.append(f"  GONE     {key} (Flash-Next: {b[key]!r})")
        elif a[key] != b[key]:
            lines.append(f"  CHANGED  {key}: {b[key]!r} -> {a[key]!r}")
    lt_a, lt_b = a.get("layer_types"), b.get("layer_types")
    if lt_a != lt_b:
        kinds = sorted(set(lt_a or []))
        lines.append(
            f"  CHANGED  layer_types: {len(lt_a or [])} layers, kinds {kinds}"
            + ("" if set(lt_a or []) <= {"linear_attention", "full_attention"} else "  <-- new layer kind")
        )
    return lines


def diff_tensors(new: dict[str, dict], old: dict[str, dict]) -> list[str]:
    lines = []
    skip = re.compile(r"\.visual\.|ngram_embedding\.shard_K")
    for key in sorted(set(new) | set(old)):
        if skip.search(key):
            continue
        if key not in old:
            lines.append(f"  NEW      {key} {new[key]['shape']} {new[key]['dtype']}")
        elif key not in new:
            lines.append(f"  GONE     {key}")
        elif new[key]["shape"] != old[key]["shape"] or new[key]["dtype"] != old[key]["dtype"]:
            lines.append(
                f"  RESHAPED {key}: {old[key]['shape']} {old[key]['dtype']} -> {new[key]['shape']} {new[key]['dtype']}"
            )
    vis_new = sum(v["count"] for k, v in new.items() if ".visual." in k)
    vis_old = sum(v["count"] for k, v in old.items() if ".visual." in k)
    if vis_new != vis_old:
        lines.append(f"  vision tensors: {vis_old} -> {vis_new}")
    return lines


def transformers_support(model_type: str) -> str:
    try:
        import importlib.util

        import transformers

        known = importlib.util.find_spec(f"transformers.models.{model_type}") is not None
        return f"transformers {transformers.__version__}: {'knows' if known else 'does NOT know'} {model_type!r}"
    except ImportError:
        return "transformers not installed here (mini_reference.py needs it; use a CPU venv, see README)"


def recommend(config: dict, config_diff: list[str], tensor_diff: list[str]) -> list[str]:
    t = text_fields(config)
    model_type = config.get("model_type", t.get("model_type"))
    dims_only = all(line.split()[0] == "CHANGED" for line in config_diff)
    new_tensors = [line for line in tensor_diff if line.split()[0] in ("NEW", "GONE")]
    out = []
    if model_type in ("qwen3_5", "qwen3_5_moe", "qwen3_5_text", "qwen3_5_moe_text"):
        out.append("Qwen3.5-family architecture: serve it with our qwen36 stack (no qwen4_exp blocks). Done before.")
        return out
    if model_type != "qwen4_exp":
        out.append(f"model_type {model_type!r} is new: read the HF modeling file first (README, Playbook step 2).")
    if not config_diff and not tensor_diff:
        out.append("Identical architecture to Flash-Next: the upstream port runs it after its pins are relaxed.")
    elif not new_tensors and dims_only:
        out.append(
            "Same blocks, different sizes. 4-chip and same TP-divisibility: start from the upstream port and relax "
            "its pins (Playbook step 4a). Smaller mesh or new widths: port the blocks into qwen36 (step 4b)."
        )
    else:
        out.append("New or removed tensors: new math. Diff the HF modeling file against qwen4_exp (step 2) first.")
    hidden = int(t.get("hidden_size", 0))
    for tp in (1, 2, 4):
        bad = [
            name
            for name, value in (
                ("hidden_size", hidden),
                ("num_attention_heads", t.get("num_attention_heads")),
                ("linear_num_value_heads", t.get("linear_num_value_heads")),
                ("linear_num_key_heads", t.get("linear_num_key_heads")),
            )
            if value and int(value) % tp
        ]
        if bad:
            out.append(f"TP={tp}: {', '.join(bad)} not divisible")
    if hidden and hidden % 32:
        out.append(f"hidden_size {hidden} is not a multiple of 32 (tile)")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("repo", nargs="?", help="Hugging Face repo id of the new checkpoint")
    p.add_argument("--revision", default="main")
    p.add_argument("--snapshot", metavar="REPO", help="write the baseline snapshot from REPO instead")
    p.add_argument("--json", type=Path, help="also write the full triage as JSON")
    p.add_argument("--context", type=int, default=32768)
    p.add_argument("--users", type=int, default=1)
    args = p.parse_args()

    if args.snapshot:
        config, tensors = fetch(args.snapshot, args.revision)
        BASELINE_CONFIG.write_text(json.dumps(config, indent=4) + "\n")
        BASELINE_TENSORS.write_text(json.dumps(tensors, indent=1) + "\n")
        print(f"baseline: {len(tensors)} tensor patterns from {args.snapshot}@{args.revision}")
        return
    if not args.repo:
        p.error("give a repo id (or --snapshot)")

    config, tensors = fetch(args.repo, args.revision)
    base_config = json.loads(BASELINE_CONFIG.read_text())
    base_tensors = json.loads(BASELINE_TENSORS.read_text())
    model_type = config.get("model_type", text_fields(config).get("model_type"))
    print(f"== {args.repo}@{args.revision}: architectures {config.get('architectures')}, model_type {model_type!r}")
    print(f"   {transformers_support(model_type)}")
    config_diff = diff_config(config, base_config)
    tensor_diff = diff_tensors(tensors, base_tensors)
    print(f"\n-- text_config vs Qwen3.8-Flash-Next ({len(config_diff)} differences)")
    print("\n".join(config_diff) or "  identical")
    print(f"\n-- tensor patterns vs Qwen3.8-Flash-Next ({len(tensors)} patterns, {len(tensor_diff)} differences)")
    print("\n".join(tensor_diff) or "  identical")
    try:
        arch = sizing.Arch.from_config(config)
        sizing.report(
            args.repo,
            arch,
            prec=sizing.Precision(),
            ctx=args.context,
            users=args.users,
            host_ram_gb=249.0,
            mtp=arch.mtp_layers > 0,
            vision=bool(arch.vision),
        )
    except Exception as error:  # a config sizing.py does not understand yet
        print(f"\n-- sizing failed ({type(error).__name__}: {error}); extend sizing.Arch for the new fields")
    print("\n-- recommendation")
    for line in recommend(config, config_diff, tensor_diff):
        print(f"  * {line}")
    if args.json:
        args.json.write_text(
            json.dumps(
                {
                    "repo": args.repo,
                    "config": config,
                    "tensors": tensors,
                    "config_diff": config_diff,
                    "tensor_diff": tensor_diff,
                },
                indent=1,
            )
        )


if __name__ == "__main__":
    main()
