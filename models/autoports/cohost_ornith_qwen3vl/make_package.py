#!/usr/bin/env python3
"""Stage a co-host package directory (manifest, card, OCI image) for ``tt-model push``.

    ~/fork-tt-model-manager/.venv/bin/python make_package.py 122b <image-tag> <out-dir> [--no-image]

The serve settings of each Ornith member are the published single-model package's (the same flags and env
``tt-model serve raahemnabeel/ornith-1.5-<size>`` composes), changed only where the shared chips require it; every
change is listed in the card's Limitations.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import subprocess
import sys
from pathlib import Path

GiB = 2**30
HF = "/hf/hub"

QWEN3VL = {
    "embed": ("Qwen/Qwen3-VL-Embedding-8B", "2c4565515e0f265c6511776e7193b22c0968ddc7"),
    "rerank": ("Qwen/Qwen3-VL-Reranker-8B", "b212dc8c91a8164aef1ea2de9c1a867611e75c04"),
}

# The Ornith member: the single-model package's p300x2 profile (flags + env), from its published manifest.
ORNITH = {
    "122b": dict(
        hf="ornith-ai/Ornith-1.5-122B", rev="7681ef608624e7532d8164128bcffdbdbaf1b1af",
        src="raahemnabeel/ornith-1.5-122b", base_profile="p300x2",
        env={"QWEN35_MOE_FUSED_DECODE": "1", "QWEN35_DECODE_REPLICATED": "1", "QWEN35_GDN_DECODE_OPS": "1",
             "QWEN35_ATTN_DECODE_OPS": "1", "QWEN35_MTP": "0", "QWEN35_FABRIC_BOUNDARY": "1", "QWEN35_FB_MAX_ROWS": "4",
             "QWEN35_FB_CONSUMER": "1", "QWEN35_FB_ROUTER": "1", "QWEN35_ROWS_TOPK": "1", "QWEN35_VISION": "1",
             "QWEN35_VISION_MAX_IMAGES": "4", "QWEN35_VISION_MAX_PIXELS": "4194304", "QWEN35_BATCHED_PREFILL": "0",
             "QWEN35_PROGRAM_GUARD": "1"},
        # DRAM per chip (GiB, of ~30.1 allocatable with the 1.75 GiB trace region): 17.66 non-KV (context contract,
        # B=1) + 7 x 0.036 extra decode slots + 8 x 0.047 prefix-cache snapshots + 1.25 transient headroom
        # + 8.0 Qwen3-VL (one 8B copy per chip, bfp8; measured 1.16 table + 0.19 per layer) + KV
        max_num_seqs=8, kv_tokens=262144, kv_bytes_per_token=6528, nonkv_gib=17.66 + 7 * 0.0356 + 8 * 0.047,
    ),
    "35b": dict(
        hf="ornith-ai/Ornith-1.5-35B-A3B", rev="10fbf86fed7ecee4a061f8b499a618f46001cac1",
        src="raahemnabeel/ornith-1.5-35b-a3b", base_profile="p300x2",
        env={"QWEN35_MOE_FUSED_DECODE": "1", "QWEN35_DECODE_REPLICATED": "1", "QWEN35_GDN_DECODE_OPS": "1",
             "QWEN35_ATTN_DECODE_OPS": "1", "QWEN35_MTP": "0", "QWEN35_ROWS_TOPK": "1", "QWEN35_VISION": "1",
             "QWEN35_VISION_MAX_IMAGES": "4", "QWEN35_VISION_MAX_PIXELS": "4194304", "QWEN35_BATCHED_PREFILL": "0",
             "QWEN35_PROGRAM_GUARD": "1", "QWEN35_FABRIC_BOUNDARY": "1", "QWEN35_FB_MAX_ROWS": "4",
             "QWEN35_FB_CONSUMER": "1", "QWEN35_FB_ROUTER": "1", "QWEN35_MOE_SPLIT_PAIR": "1"},
        # 12.0 GiB non-KV on four dies (context contract serving_tp4) + 31 x 0.036 + 32 prefix snapshots: room to spare
        max_num_seqs=32, kv_tokens=1048576, kv_bytes_per_token=5440, nonkv_gib=12.0 + 31 * 0.0356 + 32 * 0.047,
    ),
}

TRACE_REGION = 1792 * 2**20          # Ornith's 768 MiB + 1 GiB for the Qwen3-VL text traces (T <= 4096)
QWEN3VL_GIB = 8.0
HEADROOM_GIB = 1.25
DRAM_GIB = 31.331 - (TRACE_REGION / 2**20 - 512) / 1024   # the contract's total was measured with a 512 MiB region
PORTS = {"ornith": 0, "rerank": 4, "embed": 5}   # base 8000 -> 8000 / 8004 / 8005: the URLs tt-deep-research uses
BASE_PORT = 8000


def budget(o: dict, kv_tokens: int, qwen_gib: float) -> dict:
    kv = kv_tokens * o["kv_bytes_per_token"] / GiB
    used = o["nonkv_gib"] + kv + qwen_gib + HEADROOM_GIB
    return {"ornith_nonkv": round(o["nonkv_gib"], 2), "kv": round(kv, 2), "qwen3vl": qwen_gib,
            "headroom": HEADROOM_GIB, "total": round(used, 2), "allocatable": round(DRAM_GIB, 2),
            "margin": round(DRAM_GIB - used, 2)}


def members(size: str) -> list:
    o = ORNITH[size]
    snap = f"{HF}/models--{o['hf'].replace('/', '--')}/snapshots/{o['rev']}"
    tt = {"sample_on_device_mode": "all", "trace_region_size": TRACE_REGION, "l1_small_size": 24576,
          "fabric_config": "FABRIC_1D"}
    ornith = {
        "name": "ornith", "kind": "vllm", "model": snap, "port_offset": PORTS["ornith"],
        "args": ["--served-model-name", o["hf"], "--max-model-len", "262144", "--max-num-seqs", str(o["max_num_seqs"]),
                 "--block-size", "64", "--additional-config", json.dumps({"tt": tt}),
                 "--enable-auto-tool-choice", "--tool-call-parser", "qwen3_xml", "--reasoning_parser", "qwen3",
                 "--enable-prefix-caching", "--default-chat-template-kwargs", json.dumps({"enable_thinking": True}),
                 "--limit-mm-per-prompt", json.dumps({"image": 4, "video": 0}),
                 "--chat-template", f"/opt/tt-cohost/templates/ornith-1.5-{size}.jinja",
                 "--max-num-batched-tokens", "262144"],
        "env": {**o["env"], "ARCH_NAME": "blackhole", "HF_MODEL": o["hf"], "QWEN35_TRACE_REGION_MIB": "768",
                "VLLM_ENGINE_READY_TIMEOUT_S": "7200"},
        "kv_tokens": o["kv_tokens"],
    }
    if size == "122b":
        ornith["env"]["QWEN35_PREFIX_STATE_SLOTS"] = "8"
    apps = [{"name": t, "kind": "app", "app": f"cohost_qwen3vl.member:{t}", "model": QWEN3VL[t][0],
             "port_offset": PORTS[t]} for t in ("embed", "rerank")]
    return [ornith, *apps]


def profiles(size: str) -> tuple:
    o = ORNITH[size]
    common = {"HF_MODEL": o["hf"], "QWEN35_MAX_TOKENS_ALL_USERS": str(o["kv_tokens"]), "MESH_DEVICE": "P300x2"}
    lean_kv = o["kv_tokens"] * 2
    sp = [
        {"name": f"ornith{size}", "hardware": "p300x2", "mesh_device": "P300x2", "env": dict(common),
         "description": f"Ornith-1.5-{size.upper()} + Qwen3-VL-Embedding-8B + Qwen3-VL-Reranker-8B on all four chips "
                        "(text-only embedder and reranker)."},
        {"name": f"ornith{size}-lean", "hardware": "p300x2", "mesh_device": "P300x2",
         "env": {**common, "QWEN35_MAX_TOKENS_ALL_USERS": str(lean_kv), "QWEN3VL_COHOST_PRECISION": "p3_mlp_bfp4_lofi"},
         "description": "The default with the 8B models' MLP weights in bfp4 (about 2.6 GiB per chip less), and that "
                        "DRAM given to Ornith's KV pool. Use if the default runs out of DRAM; embeddings and rerank "
                        "scores change slightly."},
        {"name": f"ornith{size}-vision", "hardware": "p300x2", "mesh_device": "P300x2",
         "env": {**common, "QWEN3VL_COHOST_VISION": "1"},
         "description": "EXPERIMENTAL: the default plus both 8B vision towers (image inputs to the embedder and "
                        "reranker), tensor-parallel over the four chips."},
        {"name": f"ornith{size}-solo", "hardware": "p300x2", "mesh_device": "P300x2", "env": dict(common),
         "description": "Validation baseline: Ornith alone in the co-host runtime, with the group's settings."},
        {"name": "qwen3vl-solo", "hardware": "p300x2", "mesh_device": "P300x2", "env": dict(common),
         "description": "Validation baseline: the embedder and reranker alone in the co-host runtime."},
    ]
    groups = {f"ornith{size}": ["ornith", "embed", "rerank"], f"ornith{size}-lean": ["ornith", "embed", "rerank"],
              f"ornith{size}-vision": ["ornith", "embed", "rerank"], f"ornith{size}-solo": ["ornith"],
              "qwen3vl-solo": ["embed", "rerank"]}
    return sp, groups


def card(size: str, repo: str, image_tag: str) -> dict:
    o = ORNITH[size]
    b = budget(o, o["kv_tokens"], QWEN3VL_GIB)
    bl = budget(o, o["kv_tokens"] * 2, QWEN3VL_GIB - 2.6)
    S = size.upper()
    return {
        "attribution": f"Serves [{o['hf']}](https://huggingface.co/{o['hf']}) by ornith-ai, "
                       "[Qwen/Qwen3-VL-Embedding-8B](https://huggingface.co/Qwen/Qwen3-VL-Embedding-8B) and "
                       "[Qwen/Qwen3-VL-Reranker-8B](https://huggingface.co/Qwen/Qwen3-VL-Reranker-8B) by the Qwen team, "
                       "with unmodified checkpoints (converted to lower precision on load). See the original model "
                       "cards for training and evaluation details.",
        "description": f"A deep-research serving stack on one QuietBox 2: Ornith-1.5-{S} (chat, reasoning, tool calls, "
                       "images) with Qwen3-VL-Embedding-8B (retrieval embeddings) and Qwen3-VL-Reranker-8B (relevance "
                       "scores), served at the same time from one process on all four Blackhole chips, each on its own "
                       "endpoint. Ornith runs exactly as in its own package "
                       f"([{o['src']}](https://huggingface.co/{o['src']})). The two 8B models share one Qwen3-VL text "
                       "tower whose weights are sharded by chip: chips 0-1 hold the embedder, chips 2-3 the reranker, "
                       "so each chip carries one 8B copy and one pass can serve both models' requests. The models take "
                       "turns on the chips.",
        "intended_use": "Retrieval-augmented and deep-research agents that need a large reasoning model, an embedder "
                        "and a reranker at once on one box, when the LLM needs all four chips.",
        "prerequisites": "- **A tt-model with mixed co-host groups** (vLLM and app members in one group). Until it is "
                         "in a release, install it from the branch that adds it:\n"
                         "  `uv tool install --force git+https://github.com/rnabeelTT/tt-kernel-package-manager@raahem/cohost-mixed-groups`\n"
                         "  (or from a local checkout: `uv tool install --force ~/fork-tt-model-manager`).\n"
                         "- Docker.\n- A QuietBox 2 (two p300 boards) with all four chips free.\n"
                         f"- Disk for the weights: {o['hf']} plus the two 8B checkpoints (about 34 GB), plus about 40 "
                         "GB for the chip-sharded 8B weight cache written on the first start.",
        "quickstart": f"```bash\ntt-model pull {repo} --with-weights\ntt-model serve {repo} --port {BASE_PORT}\n"
                      f"curl -s localhost:{BASE_PORT}/v1/chat/completions -H 'Content-Type: application/json' \\\n"
                      f"  -d '{{\"model\": \"{o['hf']}\", \"messages\": [{{\"role\": \"user\", \"content\": \"Hello\"}}]}}'\n"
                      f"curl -s localhost:{BASE_PORT + PORTS['embed']}/v1/embeddings -H 'Content-Type: application/json' \\\n"
                      "  -d '{\"model\": \"Qwen/Qwen3-VL-Embedding-8B\", \"input\": [\"hello\"]}'\n"
                      f"curl -s localhost:{BASE_PORT + PORTS['rerank']}/v1/rerank -H 'Content-Type: application/json' \\\n"
                      "  -d '{\"model\": \"Qwen/Qwen3-VL-Reranker-8B\", \"query\": \"chip\", \"documents\": [\"Blackhole\", \"a cat\"]}'\n"
                      f"curl -s localhost:{BASE_PORT + PORTS['embed'] + 1}/   # co-host status: readiness, DRAM and L1 per member\n```",
        "capabilities": f"Ports are the serve port plus an offset: Ornith on +0 (OpenAI chat / completions, tool calls, "
                        f"reasoning in `reasoning`, up to 4 images per request), the reranker on +4 (`/v1/rerank`, "
                        f"`/v1/score`), the embedder on +5 (`/v1/embeddings`, 4096-d, Matryoshka `dimensions` from 64), "
                        f"the co-host status on +6. Pairs up to 8,192 tokens; up to 4,096 run in captured traces, "
                        f"longer ones eagerly.",
        "performance": "| Profile | Members | Ornith KV pool | DRAM per chip (budget) | Measured |\n"
                       "| --- | --- | ---: | ---: | --- |\n"
                       f"| `ornith{size}` | Ornith-1.5-{S} (max {o['max_num_seqs']} seqs) + embedder + reranker | "
                       f"{o['kv_tokens']:,} tokens | {b['total']} of {b['allocatable']} GiB | not yet on hardware |\n"
                       f"| `ornith{size}-lean` | the same, 8B MLPs in bfp4 | {o['kv_tokens'] * 2:,} tokens | "
                       f"{bl['total']} of {bl['allocatable']} GiB | not yet on hardware |\n\n"
                       "Budget = Ornith's non-KV footprint at this batch (its context contract) + its KV pool + one "
                       "Qwen3-VL-8B copy per chip (measured on a mock mesh: 1.16 GiB embedding table + 0.19 GiB per "
                       "layer) + Ornith's 1.25 GiB prefill headroom. Throughput and latency will be measured against "
                       "each model served alone (tt-model-cohost-package validation protocol).",
        "limitations": "- **Not yet validated on hardware.** Built and checked on a mock 4-chip mesh (weight layout per "
                       "chip, every program shape compiles, both HTTP APIs, combined passes); trace capture, numerics "
                       "against the single-model packages, DRAM and L1 fit are measured on the first hardware run.\n"
                       "- The models take turns on the chips: a long Ornith prefill delays embedding and rerank "
                       "requests, and the other way round.\n"
                       f"- Ornith serves at most {o['max_num_seqs']} sequences at once with a {o['kv_tokens']:,}-token "
                       "shared KV pool (one user still reaches 262,144 tokens)"
                       + (", and keeps 8 prefix-cache resume points instead of 32" if size == "122b" else "")
                       + ", to leave DRAM for the 8B models.\n"
                       "- The embedder and reranker run one program on every chip, so both use the embedder's "
                       "precision policy (`p0_bfp8_hifi2`); the single-model reranker package defaults to LoFi MLPs "
                       "(`p1_mlp_lofi`), so its scores can differ slightly. Pairs are capped at 8,192 tokens (the "
                       "single-model reranker serves 32,768).\n"
                       "- Text only for the embedder and reranker by default (an image input gets HTTP 400); the "
                       f"`ornith{size}-vision` profile is experimental. Ornith itself accepts images.\n"
                       "- Each model keeps its own license: Ornith-1.5 MIT, Qwen3-VL-Embedding/Reranker Apache-2.0.",
        "license": {"id": "other", "name": "MIT (Ornith-1.5) and Apache-2.0 (Qwen3-VL-Embedding-8B, Qwen3-VL-Reranker-8B)"},
    }


def manifest(size: str, repo: str, image_tag: str, digest: str) -> dict:
    o = ORNITH[size]
    sp, groups = profiles(size)
    name = repo.split("/")[1]
    return {
        "schema_version": "5.1",
        "name": name,
        "tt_metal_version": "0.0.0.dev0",
        "arch": "blackhole",
        "device_count": 4,
        "producer": {"tt_kernel_version": "0.1.0", "created_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                     "hostname": "local-build"},
        "weights": {"repo_id": o["hf"], "revision": o["rev"]},
        "auxiliary_weights": [{"repo_id": h, "revision": r} for h, r in QWEN3VL.values()],
        "container": {
            "image": {"registry": "hf", "repository": name, "tag": image_tag, "digest": digest},
            "kind": "cohost",
            "runtime": {
                "members": members(size),
                "profiles": groups,
                "tt_config": {"fabric_config": "FABRIC_1D", "trace_region_size": TRACE_REGION, "l1_small_size": 24576},
            },
            "serve": {"port": BASE_PORT},
            "serve_profiles": sp,
            "default_profile": f"ornith{size}",
            "verify": ["import tt_kernel.cohost.launch, cohost_qwen3vl.member, qwen_vl_utils"],
            "built": {"image": image_tag, "repo": repo,
                      "created_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                      "base_image": subprocess.run(["docker", "inspect", image_tag, "--format",
                                                    '{{index .Config.Labels "org.tenstorrent.tt-model.base"}}'],
                                                   capture_output=True, text=True, check=True).stdout.strip()},
            "card": card(size, repo, image_tag),
        },
    }


def readme(m: dict) -> str:
    c = m["container"]["card"]
    sp = m["container"]["serve_profiles"]
    rows = "\n".join(f"| `{p['name']}`{' *(default)*' if p['name'] == m['container']['default_profile'] else ''} | "
                     f"{', '.join(m['container']['runtime']['profiles'][p['name']])} | {p['description']} |" for p in sp)
    return f"""---
tags:
- blackhole
- p300x2
- tt-model-cache
- tt-model-container
- cohost
license: other
license_name: mit-and-apache-2.0
base_model:
- {m['weights']['repo_id']}
- Qwen/Qwen3-VL-Embedding-8B
- Qwen/Qwen3-VL-Reranker-8B
---

# {m['name']}

> {c['attribution']}

{c['description']}

Packaged with [tt-model-manager](https://github.com/tenstorrent/tt-model-manager) (manifest schema 5.1, kind `cohost`,
mixed vLLM + app group).

## Intended use

{c['intended_use']}

## Prerequisites

{c['prerequisites']}

## Quickstart

{c['quickstart']}

## Serving profiles

Select one with `--profile`.

| profile | members | description |
| --- | --- | --- |
{rows}

## Capabilities

{c['capabilities']}

## Expected performance

{c['performance']}

## Limitations

{c['limitations']}

## License

{c['license']['name']}. Each model's own license applies to its weights.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("size", choices=sorted(ORNITH))
    ap.add_argument("image")
    ap.add_argument("out")
    ap.add_argument("--repo", default=None)
    ap.add_argument("--no-image", action="store_true", help="manifest and card only (no OCI export)")
    a = ap.parse_args()
    repo = a.repo or f"raahemnabeel/cohost-ornith-{a.size}-qwen3vl"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if not a.no_image and not (out / "image" / "oci-layout").exists():
        from tt_kernel import oci
        oci.save(a.image, out / "image")
    digest = "sha256:" + "0" * 64
    if (out / "image" / "index.json").exists():
        idx = json.loads((out / "image" / "index.json").read_text())
        digest = idx["manifests"][0]["digest"]
    m = manifest(a.size, repo, a.image, digest)
    (out / "tt_kernel_manifest.json").write_text(json.dumps(m, indent=1) + "\n")
    (out / "README.md").write_text(readme(m))
    sums = []
    for p in sorted([out / "tt_kernel_manifest.json", out / "README.md"]):
        sums.append(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}")
    (out / "SHA256SUMS").write_text("\n".join(sums) + "\n")
    o = ORNITH[a.size]
    print(json.dumps({"staged": str(out), "repo": repo, "digest": digest,
                      "budget": budget(o, o["kv_tokens"], QWEN3VL_GIB)}, indent=1))


if __name__ == "__main__":
    sys.exit(main())
