#!/usr/bin/env python3
"""Emit ``readiness_vllm/vllm_serving_capability.json`` for this autoport.

Modelled on the artifact the ``tt-hous/gpt-oss-120b-p150x4`` bundle ships: a machine-readable
declaration of what the package actually serves, so a consumer does not have to infer it from
prose. Three things it is meant to pin down:

* **the precision policy, by content hash** - which file was in force and whether the constructed
  tensors matched it, rather than "see the README";
* **device-sampling capabilities, including the top-k ceiling** - the device sampler supports
  ``top_k`` up to 32, and anything above it is silently clamped rather than moved to host sampling,
  so a request using this checkpoint's recommended ``top_k 40`` is NOT served with the sampler the
  model card specifies. That is a disclosure, not a detail;
* **serving counters from a real run** - how many decodes were sampled on device vs host, how many
  decode traces were recaptured, how often the page table was refreshed. Counters are what turn
  "we ran the server" into evidence about which path was exercised.

Usage:
    python -m models.autoports.ornith_ai_ornith_1_5_35b_a3b.tools.serving_capability \
        [--mesh P300x2] [--server-log path] [--out path]

The static fields come from the checked-in config; ``--server-log`` merges the counters this port
already logs (decode bucket histogram, forced resets, program-cache snapshots).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = MODEL_DIR.parents[2]

# The device sampler walks a per-user candidate row of two 16-wide tile faces, so k is capped at 32
# (models/common/sampling/tt_sampling.py: max_top_k). format_sampling_params clamps anything larger.
MAX_DEVICE_SAMPLING_TOP_K = 32


def _sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _precision_block(mesh: str) -> dict:
    tp2 = mesh in ("P300", "P150x2")
    name = "selected_precision_config_tp2.json" if tp2 else "selected_precision_config.json"
    path = MODEL_DIR / "doc" / "datatype_sweep" / name
    policy = json.loads(path.read_text())
    wg = policy["weight_groups"]
    cf = policy["compute_fidelities"]
    return {
        "schema_version": 2,
        "config_id": policy["config_id"],
        "config_path": str(path.relative_to(REPO_ROOT)),
        "config_sha256": _sha256(path),
        "terminal": {
            "embedding_weight": wg["embedding"]["dtype"],
            "normalization_weight": wg["norms"]["dtype"],
            "lm_head_weight": wg["lm_head"]["dtype"],
            "lm_head_math_fidelity": cf["lm_head"],
            "logits_dtype": policy["logits_dtype"],
            "sampling_dtype": policy["sampling_dtype"],
            "full_logits_gather": {
                # stochastic sampling consumes the sharded LM-head output (per-device top-k then
                # all-gather); the greedy force-argmax branch instead gathers the full vocab
                "mode": "not_materialized_for_sampling",
                "runtime_path": "sharded TTSampling consumes LM-head output directly",
                "force_full_vocab_argmax_allowed": True,
                "force_full_vocab_argmax_selected": "per-request (temperature 0)",
            },
        },
        "layer_runtime_groups": [
            {
                "attention_weight": wg["attention"]["dtype"],
                "linear_attention_weight": wg["linear_attention"]["dtype"],
                "router_weight": wg["router"]["dtype"],
                "shared_expert_weight": wg["shared_moe"]["dtype"],
                "routed_expert_gate_up": wg["routed_moe"]["gate_up"],
                "routed_expert_down": wg["routed_moe"]["down"],
                "kv_cache": policy["kv_cache_dtype"],
                "linear_state": policy["linear_state_dtype"],
                "ccl": policy["ccl_dtype"],
                "residual": policy["residual_dtype"],
                "activation": policy["activation_dtype"],
                "routed_fidelity": cf["routed_moe"],
                "attention_fidelity": cf["attention"],
                "router_fidelity": cf["router"],
            }
        ],
        "accuracy_gate": policy.get("accuracy_gate"),
    }


def _counters_from_log(log: Path) -> dict:
    text = log.read_text(errors="ignore")
    out: dict = {}
    m = None
    for m in re.finditer(r"DECODE_BUCKETS steps per width (\{[^}]*\})", text):
        pass
    if m:
        try:
            out["decode_steps_per_width"] = json.loads(m.group(1).replace("'", '"'))
            out["decode_calls"] = sum(out["decode_steps_per_width"].values())
        except Exception:  # noqa: BLE001
            pass
    out["forced_reset_logged"] = "forcing reset_batch" in text
    out["invalid_token_ids_logged"] = len(re.findall(r"Encountered invalid token id", text))
    out["detokenizer_prefix_resets"] = len(re.findall(r"invalid prefix detokenization", text))
    return out


def build(mesh: str, server_log: Path | None) -> dict:
    tp2 = mesh in ("P300", "P150x2")
    doc = {
        "model": "ornith-ai/Ornith-1.5-35B-A3B",
        "architecture": "Qwen3NextForCausalLM",
        "mesh_device": mesh,
        "mesh_shape": [1, 2] if tp2 else [1, 4],
        "resident_layers": 48,
        "max_model_len": 262144,
        "max_num_seqs": 32,
        "page_size": 64,
        "decode_trace_enabled": True,
        "decode_trace_buckets": [1, 2, 4, 8, 16, 32],
        "chunked_prefill": True,
        "max_num_batched_tokens": 2048,
        "kv_cache_owner": "vllm",
        "precision": _precision_block(mesh),
        "model_capabilities": {
            "supports_async_decode": True,
            "supports_sample_on_device": True,
            "max_device_sampling_top_k": MAX_DEVICE_SAMPLING_TOP_K,
            "device_sampling_top_k_above_max": "clamped (NOT moved to host sampling)",
            "supports_batched_prefill": False,
            "supports_prefix_caching": False,
            "supports_mtp": False,
            "supports_vision": False,
            "device_logprobs": "unsupported on a 4-die mesh; any logprobs request falls back to host sampling",
        },
        "known_defects": [
            {
                "id": "long-context-device-sampling-word-corruption",
                "status": "fixed",
                "summary": "Hub revisions up to ef463288: with on-device stochastic sampling, output showed a wrong "
                "word about once in thirty (first token of a word replaced by a sibling candidate).",
                "root_cause": "ttnn.sampling runs on every die with a free-running hardware PRNG; the dies' "
                "draws drifted apart, each die fed its own token back into the model while the host read die 0's.",
                "fix": "tt/model.py _install_lockstep_sampling: identical fresh seeds pushed before every draw and "
                "die 0's token broadcast into every die's token buffer (0 of 400 sampled steps disagree, was 196).",
                "evidence": "models/autoports/ornith_ai_ornith_1_5_35b_a3b/doc/text_quality/README.md",
            }
        ],
        "validation": {
            "teacher_forcing": "greedy top-1 agreement vs HF bf16, 100 tokens, prompts up to 2048 tokens",
            "served_benchmark_evals": None,
            "longest_context_measured": 2048,
            "longest_generation_reviewed": 200,
            "note": "Teacher forcing measures greedy agreement and does not bound sampled serving quality.",
        },
    }
    if server_log and server_log.is_file():
        doc["serving_counters"] = _counters_from_log(server_log)
    return doc


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", default="P300x2", choices=["P300x2", "P150x4", "P300", "P150x2"])
    ap.add_argument("--server-log", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    doc = build(a.mesh, a.server_log)
    out = a.out or (MODEL_DIR / "readiness_vllm" / "vllm_serving_capability.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1) + "\n")
    print(f"wrote {out}")
    print(f"  precision {doc['precision']['config_id']} sha256 {doc['precision']['config_sha256'][:16]}...")
    print(f"  max_device_sampling_top_k {doc['model_capabilities']['max_device_sampling_top_k']}")
    print(f"  known defects: {len(doc['known_defects'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
