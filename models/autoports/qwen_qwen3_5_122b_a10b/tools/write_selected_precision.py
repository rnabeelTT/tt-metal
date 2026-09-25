# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Write doc/datatype_sweep/selected_precision_config.json = the selected candidate's policy (from sweep_results.json) plus
the selection metadata (gate, rule, measurements on every reference, token-out figure, fallback), so the runtime default and
the evidence live in one file.

    python -m ...tools.write_selected_precision [--config-id <id>]   (default: sweep_results.json "selected")
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parents[1]
D = MODEL_DIR / "doc" / "datatype_sweep"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-id", default=None)
    a = ap.parse_args(argv)
    res = json.loads((D / "sweep_results.json").read_text())
    cid = a.config_id or res["selected"]
    row = next(r for r in res["results"] if r["config_id"] == cid)
    pol = json.loads((D / "configs" / f"{cid}.json").read_text())
    base = next(r for r in res["results"] if r["config_id"] == "baseline_s7")
    # the post-selection number the goal wants = the DEFAULT-path run (no env override); the candidate's env-override run
    # is the fallback until 08-post.sh has run
    tok = D / "artifacts" / "tokenout_selected_48layers.json"
    if not tok.is_file():
        tok = D / "artifacts" / f"tokenout_48layers_{cid}.json"
    tok_base = D / "artifacts" / "tokenout_48layers_baseline_s7.json"

    def tok_summary(p):
        if not p.is_file():
            return None
        t = json.loads(p.read_text())
        return {
            "ms_per_step": t.get("ms_per_step"),
            "tokens_per_s_per_user": t.get("tokens_per_s_per_user"),
            "steps": t.get("steps"),
            "regime": "48 layers, B=1, 64 traced decode steps, no readback on the serving path (tools/tokenout_bench.py)",
            "source": str(p.relative_to(MODEL_DIR.parents[2])),
            "construction_path": "default (no QWEN35_PRECISION_CONFIG override)"
            if p.name == "tokenout_selected_48layers.json"
            else "candidate run (QWEN35_PRECISION_CONFIG=configs/<id>.json)",
        }

    refs = {k: row[k] for k in row if k.startswith(("teacher_forcing_book", "prefill_check"))}
    out = dict(pol)
    out["selected_from_config_id"] = cid
    out["selection_metric"] = res["selection_metric"]
    out["selection_rule"] = res["selection_rule"]
    out["accuracy_gate"] = res["accuracy_gate"]
    out["selected_measurement"] = {
        "aime24_chat_teacher_forcing": {
            "top1": row["teacher_forcing_top1"],
            "top5": row["teacher_forcing_top5"],
            "top100": row["teacher_forcing_top100"],
            "ttft_ms": row["ttft_ms"],
            "decode_t_s_u": row["trace_verified_teacher_forcing_decode_t_s_u"],
            "trace_verified": row["trace_verified"],
            "policy_consumed": row["policy_consumed"],
        },
        "other_references": refs,
        "quality_gate": row.get("quality_gate"),
        "post_selection_token_out": tok_summary(tok),
        "baseline_s7_for_comparison": {
            "aime24_top1": base["teacher_forcing_top1"],
            "decode_t_s_u": base["trace_verified_teacher_forcing_decode_t_s_u"],
            "ttft_ms": base["ttft_ms"],
            "token_out": tok_summary(tok_base),
            "other_references": {k: base[k] for k in base if k.startswith(("teacher_forcing_book", "prefill_check"))},
        },
        "measurement_regime": row["measurement_regime"],
        "hardware": row["hardware"],
        "mesh": row["mesh"],
    }
    out["fallback"] = {
        "safe_baseline": "configs/baseline_s7.json (the stage-7 policy): QWEN35_PRECISION_CONFIG=models/autoports/qwen_qwen3_5_122b_a10b/doc/datatype_sweep/configs/baseline_s7.json or =none",
        "intermediate": "configs/lm_head_bfp4.json (bfp4 LM head only): no measured accuracy loss on the 8192-token book prompt, +0.7 % decode",
    }
    out[
        "consumed_by"
    ] = "tt/model.py::create_model via tt/precision.py (load_policy -> apply_policy -> MultichipConfig); the built model is checked against this file (effective_policy / check_effective) on every build"
    (D / "selected_precision_config.json").write_text(json.dumps(out, indent=2) + "\n")
    print("wrote", D / "selected_precision_config.json", "for", cid)


if __name__ == "__main__":
    main()
