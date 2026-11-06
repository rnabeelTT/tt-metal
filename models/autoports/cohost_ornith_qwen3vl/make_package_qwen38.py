#!/usr/bin/env python3
"""Stage the Qwen3.8-27B co-host package (manifest, card, OCI image) for ``tt-model push``.

    ~/fork-tt-model-manager/.venv/bin/python make_package_qwen38.py <image-tag> <out-dir> [--no-image]

The LLM member runs exactly what the thin bundle's run.sh runs (mando2222/qwen3.8-27b-dflash2-vision-p300x2-q4kv-v6-thin-cp4,
revision below): the same vLLM flags and the same environment, re-expressed for the co-host runtime. The only changes
are the ones the shared chips need, each listed in the card's Limitations.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import make_package as mp

REPO = "raahemnabeel/cohost-qwen38-27b-qwen3vl"
BUNDLE = "mando2222/qwen3.8-27b-dflash2-vision-p300x2-q4kv-v6-thin-cp4"
BUNDLE_REV = "67af7db5a391635a798c493907ec1515291b4016"
HF_ID, HF_REV = "Qwen/Qwen3.8-27B", "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
DRAFTER = "incoai/Qwen3.8-27B-DFlash2@dedf8df68adfb1afeaf7b7480c0a0243108177b4"
SNAP = f"{mp.HF}/models--Qwen--Qwen3.8-27B/snapshots/{HF_REV}"
TRACE_REGION = 2 * 2**30   # the bundle's 1 GiB + 1 GiB for the Qwen3-VL text traces
KV_DEFAULT, KV_FULL = 262144, 524288   # QWEN36_MAX_TOKENS_ALL_USERS: co-host default / the bundle's own
PORTS = {"llm": 0, "rerank": 4, "embed": 5}

# run.sh's exports (the bundle manifest's env plus the lines run.sh adds), unchanged
ENV = {
    "DFLASH_WEIGHTS": DRAFTER, "TT_AUXILIARY_WEIGHTS": DRAFTER,
    "QWEN36_DFLASH_BLOCK": "8", "QWEN36_DFLASH_BUCKETS": "1x8,4x8,8x4", "QWEN36_DFLASH_FOLD_SEED": "1",
    "QWEN36_DFLASH_MAX_PROMPT": "2048", "QWEN36_DFLASH_RAGGED": "1", "QWEN36_DFLASH_SERVE_BLOCK": "32",
    "QWEN36_DRAFTER": "dflash2", "QWEN36_ENABLE_VISION": "1", "QWEN36_GDN_SPEC_FUSED": "1", "QWEN36_KDA_TILE_IN": "1",
    "QWEN36_MAX_IMAGES": "4", "QWEN36_SPEC_EXACT_SDPA": "1", "QWEN_SDPA_KV_DTYPE": "bf4", "TORCHDYNAMO_DISABLE": "1",
    "TT_QWEN35_TEXT_VER": "qwen36_blackhole", "VLLM_CONFIGURE_LOGGING": "1", "VLLM_RPC_TIMEOUT": "900000",
    "QWEN36_DECODE_REPLICATED": "1", "QWEN36_ATTN_DECODE_OPS": "1", "QWEN36_BOUNDARY_AR_LINKS": "2",
    "QWEN36_SAMPLER_LOCKSTEP": "1", "QWEN36_ROWS_TOPK": "1", "QWEN36_FORCE_ARGMAX": "0",
    "TT_PLAIN_DEVICE_RANDOM_SAMPLING": "1", "TT_DFLASH_SAMPLED": "1", "QWEN36_VERIFY_TOPK": "1", "QWEN36_GDN_VSPLIT": "4",
    "QWEN36_DFLASH_DRAFT_SAMPLE": "1.0", "QWEN36_MLP_STREAM": "1",
    "QWEN36_PROJ_STREAM": "gdn_out,gdn_in,attn_qkv,attn_wo,mlp_down", "QWEN36_DFLASH_DRAFT_WDTYPE": "bfp8",
    "QWEN36_MAMBA_STATE": "1",
    # run.sh resolves the weights to the snapshot before vLLM starts and exports both
    "HF_MODEL": SNAP, "MODEL_WEIGHTS_DIR": SNAP, "TT_MODEL_WEIGHTS_REVISION": HF_REV,
    "TT_METAL_VISIBLE_DEVICES": "0,1,2,3", "VLLM_ENGINE_READY_TIMEOUT_S": "7200",
}

TT = {"l1_small_size": 24576, "sample_on_device_mode": "decode_only", "fabric_config": "FABRIC_1D",
      "trace_region_size": TRACE_REGION}


def members() -> list:
    llm = {
        "name": "llm", "kind": "vllm", "model": SNAP, "port_offset": PORTS["llm"],
        # run.sh's flags; --model <hub id> --revision/--tokenizer-revision become the snapshot path + served name
        "args": ["--served-model-name", HF_ID, "--max_num_seqs", "8", "--block_size", "4096", "--max_model_len", "262144",
                 "--enable-auto-tool-choice", "--tool-call-parser", "qwen3_coder", "--reasoning_parser", "qwen3",
                 "--max-num-batched-tokens", "262144", "--no-async-scheduling",
                 "--override-generation-config", json.dumps({"temperature": 0.6, "top_k": 20, "top_p": 0.95}),
                 "--default-chat-template-kwargs", json.dumps({"reasoning_effort": "medium"}),
                 "--middleware", "qwen38.think_budget.ThinkBudgetMiddleware",
                 "--mm-processor-kwargs", json.dumps({"size": {"longest_edge": 1003520, "shortest_edge": 65536}}),
                 "--enable-prefix-caching", "--additional-config", json.dumps({"tt": TT}),
                 "--chat-template", "/opt/tt-cohost/templates/qwen3.8-27b.jinja"],
        "env": dict(ENV),
    }
    apps = [{"name": t, "kind": "app", "app": f"cohost_qwen3vl.member:{t}", "model": mp.QWEN3VL[t][0],
             "port_offset": PORTS[t]} for t in ("embed", "rerank")]
    return [llm, *apps]


def profiles() -> tuple:
    common = {"MESH_DEVICE": "P150x4", "QWEN36_MAX_TOKENS_ALL_USERS": str(KV_DEFAULT), "HF_MODEL": SNAP,
              "MODEL_WEIGHTS_DIR": SNAP}
    sp = [
        {"name": "qwen38", "hardware": "p300x2", "mesh_device": "P150x4", "env": dict(common),
         "description": "Qwen3.8-27B DFlash2 + vision + Qwen3-VL-Embedding-8B + Qwen3-VL-Reranker-8B on all four chips "
                        f"(text-only embedder and reranker); {KV_DEFAULT:,}-token shared KV pool."},
        {"name": "qwen38-lean", "hardware": "p300x2", "mesh_device": "P150x4",
         "env": {**common, "QWEN3VL_COHOST_PRECISION": "p3_mlp_bfp4_lofi"},
         "description": "The default with the 8B models' MLP weights in bfp4 (about 2.6 GiB per chip less). Use if the "
                        "default runs out of DRAM; embeddings and rerank scores change slightly."},
        {"name": "qwen38-vision", "hardware": "p300x2", "mesh_device": "P150x4",
         "env": {**common, "QWEN3VL_COHOST_VISION": "1"},
         "description": "EXPERIMENTAL: the default plus both 8B vision towers (image inputs to the embedder and "
                        "reranker, one image per input), tensor-parallel over the four chips."},
        {"name": "qwen38-solo", "hardware": "p300x2", "mesh_device": "P150x4", "env": dict(common),
         "description": "Validation baseline: Qwen3.8 alone in the co-host runtime, with the group's settings."},
        {"name": "qwen3vl-solo", "hardware": "p300x2", "mesh_device": "P150x4", "env": dict(common),
         "description": "Validation baseline: the embedder and reranker alone in the co-host runtime."},
    ]
    groups = {"qwen38": ["llm", "embed", "rerank"], "qwen38-vision": ["llm", "embed", "rerank"],
              "qwen38-lean": ["llm", "embed", "rerank"], "qwen38-solo": ["llm"], "qwen3vl-solo": ["embed", "rerank"]}
    return sp, groups


def card() -> dict:
    return {
        "attribution": f"Serves [{HF_ID}](https://huggingface.co/{HF_ID}) with the "
                       "[incoai/Qwen3.8-27B-DFlash2](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2) drafter, "
                       "[Qwen/Qwen3-VL-Embedding-8B](https://huggingface.co/Qwen/Qwen3-VL-Embedding-8B) and "
                       "[Qwen/Qwen3-VL-Reranker-8B](https://huggingface.co/Qwen/Qwen3-VL-Reranker-8B), with unmodified "
                       "checkpoints (converted to lower precision on load). The Qwen3.8 serving code is "
                       f"[{BUNDLE}](https://huggingface.co/{BUNDLE}) at `{BUNDLE_REV[:12]}`.",
        "description": "Qwen3.8-27B (DFlash2 speculative decoding, vision, tool calls, reasoning) with "
                       "Qwen3-VL-Embedding-8B and Qwen3-VL-Reranker-8B, served at the same time from one process on all "
                       "four Blackhole chips of a QuietBox 2, each on its own endpoint. The LLM runs the v6 thin bundle "
                       "exactly as its run.sh does (vLLM 0.30, PyPI ttnn 0.79.0, its own plugin and ops wheels), baked "
                       "into the image. The two 8B models share one Qwen3-VL text tower whose weights are sharded by "
                       "chip (chips 0-1 embedder, 2-3 reranker), one 8B copy per chip. The models take turns.",
        "intended_use": "Retrieval-augmented and deep-research agents that need a strong mid-size reasoning / coding "
                        "model, an embedder and a reranker at once on one box.",
        "prerequisites": "- **A tt-model with mixed co-host groups** (branch `raahem/cohost-mixed-groups` of "
                         "rnabeelTT/tt-kernel-package-manager, commit 1008ca2 or later).\n- Docker.\n"
                         "- A QuietBox 2 (two p300 boards) with all four chips free.\n"
                         f"- The weights: {HF_ID} (about 56 GB), the DFlash2 drafter and the two 8B checkpoints "
                         "(about 34 GB), plus about 40 GB for the chip-sharded 8B weight cache written on the first "
                         "start.",
        "quickstart": f"```bash\ntt-model pull {REPO} --with-weights\ntt-model serve {REPO} --port 8000\n"
                      "curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \\\n"
                      f"  -d '{{\"model\": \"{HF_ID}\", \"messages\": [{{\"role\": \"user\", \"content\": \"Hello\"}}]}}'\n"
                      "curl -s localhost:8005/v1/embeddings -H 'Content-Type: application/json' \\\n"
                      "  -d '{\"model\": \"Qwen/Qwen3-VL-Embedding-8B\", \"input\": [\"hello\"]}'\n"
                      "curl -s localhost:8004/v1/rerank -H 'Content-Type: application/json' \\\n"
                      "  -d '{\"model\": \"Qwen/Qwen3-VL-Reranker-8B\", \"query\": \"chip\", \"documents\": [\"Blackhole\", \"a cat\"]}'\n"
                      "curl -s localhost:8006/   # co-host status\n```",
        "capabilities": "Ports are the serve port plus an offset: the LLM on +0 (OpenAI chat / completions, tool calls "
                        "(`qwen3_coder`), reasoning, up to 4 images per request, the bundle's opt-in "
                        "`TT_THINK_BUDGET`), the reranker on +4, the embedder on +5, the co-host status on +6.",
        "performance": ("Not yet measured on hardware. Device-free boot (mock 4-chip mesh, untraced, 2026-11-04), DRAM "
                        "per chip as the mock allocator counts it:\n\n"
                        "| | GiB per chip |\n| --- | ---: |\n"
                        "| Qwen3.8-27B weights + DFlash2 drafter + vision | 10.91 |\n"
                        f"| KV pool ({KV_DEFAULT:,} tokens, 8 slots) incl. 5.10 recurrent-state store (120 x 43.5 MiB) | 7.95 |\n"
                        "| DFlash serving buffers (warmup) | 2.23 |\n"
                        "| Qwen3-VL embedder + reranker (one 8B copy per chip) | 8.03 |\n"
                        "| **total, before request transients** | **29.13** (of about 29.8 allocatable next to a 2 GiB trace region) |\n\n"
                        "L1 held per core by Qwen3.8: 89.6 KiB (every Qwen3-VL program compiles next to it). Expected speed: "
                        "the LLM at its single-model speed while the 8B models are idle, as measured for the "
                        "Ornith-1.5-122B co-host package (147 tok/s alone and in the group)."),
        "limitations": "- **Not yet validated on hardware.** Checked device-free: the full group boots on a mock "
                       "4-chip mesh (all members ready; embedder and reranker answer), the 8B engine passes its mock "
                       "suite on this image's ttnn. Not checked: traced boot, DFlash serving, speed, numerics. The "
                       "runtime and 8B adapter are the ones validated in raahemnabeel/cohost-ornith-122b-qwen3vl, plus "
                       "two changes this model needs (the LLM worker opens the mesh with its model's fabric config; the "
                       "phased warmup passes the plugin's device-grammar flag).\n"
                       "- **DRAM is tight** (29.13 of about 29.8 GiB per chip before request transients): if the boot "
                       "or a long prefill runs out of DRAM, use `qwen38-lean`.\n"
                       f"- Shared KV pool {KV_DEFAULT:,} tokens (the bundle: {KV_FULL:,}, which does not fit next to "
                       "the 8B models).\n"
                       "- The image patches the plugin's executor allow-list for block-output (DFlash) models to admit "
                       "the co-host RemoteExecutor (a UniProcExecutor whose worker runs in the device host).\n"
                       "- Trace region 2 GiB (the bundle: 1 GiB): the 8B text traces share it.\n"
                       "- The chat template is the checkpoint's own with one change: a conversation whose last "
                       "message is not a user turn (agent loops ending on a tool result) no longer raises \"No user "
                       "query found in messages\".\n"
                       "- The models take turns on the chips. The embedder and reranker are text only (image inputs: "
                       "HTTP 400) and use the embedder's precision policy; pairs up to 8,192 tokens.\n"
                       "- Licenses: Qwen3.8-27B and the Qwen3-VL models Apache-2.0; see the drafter's card.",
        "license": {"id": "apache-2.0", "name": "Apache-2.0 (Qwen3.8-27B, Qwen3-VL-Embedding-8B, Qwen3-VL-Reranker-8B)"},
    }


def manifest(image_tag: str, digest: str) -> dict:
    sp, groups = profiles()
    name = REPO.split("/")[1]
    now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    drafter, drev = DRAFTER.split("@")
    return {
        "schema_version": "5.1", "name": name, "tt_metal_version": "0.79.0", "arch": "blackhole", "device_count": 4,
        "producer": {"tt_kernel_version": "0.1.0", "created_at": now, "hostname": "local-build"},
        "weights": {"repo_id": HF_ID, "revision": HF_REV},
        "auxiliary_weights": [{"repo_id": drafter, "revision": drev},
                              *({"repo_id": h, "revision": r} for h, r in mp.QWEN3VL.values())],
        "container": {
            "image": {"registry": "hf", "repository": name, "tag": image_tag, "digest": digest},
            "kind": "cohost",
            "runtime": {"members": members(), "profiles": groups,
                        "tt_config": {"fabric_config": "FABRIC_1D", "trace_region_size": TRACE_REGION,
                                      "l1_small_size": 24576}},
            "serve": {"port": 8000},
            "serve_profiles": sp,
            "default_profile": "qwen38",
            "verify": ["import tt_kernel.cohost.launch, cohost_qwen3vl.member, qwen38, vllm_tt_plugin"],
            "built": {"image": image_tag, "repo": REPO, "created_at": now,
                      "base_bundle": f"{BUNDLE}@{BUNDLE_REV}"},
            "card": card(),
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("out")
    ap.add_argument("--no-image", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if not a.no_image and not (out / "image" / "oci-layout").exists():
        from tt_kernel import oci
        oci.save(a.image, out / "image")
    if (out / "image" / "index.json").exists():
        digest = json.loads((out / "image" / "index.json").read_text())["manifests"][0]["digest"]
    else:
        digest = subprocess.run(["docker", "image", "inspect", a.image, "--format", "{{.Id}}"], capture_output=True,
                                text=True, check=True).stdout.strip()
    m = manifest(a.image, digest)
    (out / "tt_kernel_manifest.json").write_text(json.dumps(m, indent=1) + "\n")
    readme = mp.readme(m).replace("license: other\nlicense_name: mit-and-apache-2.0", "license: apache-2.0")
    readme = readme.replace(f"- {HF_ID}\n", f"- {HF_ID}\n- {DRAFTER.split('@')[0]}\n", 1)
    (out / "README.md").write_text(readme)
    sums = [f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}"
            for p in sorted([out / "tt_kernel_manifest.json", out / "README.md"])]
    (out / "SHA256SUMS").write_text("\n".join(sums) + "\n")
    print(json.dumps({"staged": str(out), "repo": REPO, "digest": digest}, indent=1))


if __name__ == "__main__":
    sys.exit(main())
