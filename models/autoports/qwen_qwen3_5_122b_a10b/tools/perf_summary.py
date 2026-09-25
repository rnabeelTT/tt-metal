# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Assemble doc/optimized_full_model/perf_summary.json from the stage-7 measurement artifacts (token-out benches,
prefill bench, per-knob sweeps, op counts, readiness reports). Every number carries the file it came from."""
import json
import sys
from pathlib import Path

D = Path("models/autoports/qwen_qwen3_5_122b_a10b/doc")
O = D / "optimized_full_model"


def _load(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def _jsonl(p):
    p = Path(p)
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def main():
    out = {"model": "Qwen/Qwen3.5-122B-A10B", "mesh": "(1,4) P300x2", "stage": 7, "sources": {}}
    before = _load(O / "tokenout_before_48layers.json")
    after = _load(O / "tokenout_after_48layers.json")
    for tag, rep in (("before", before), ("after", after)):
        if rep:
            out[f"tokenout_{tag}"] = {
                "ms_per_step": rep["ms_per_step"],
                "tokens_per_s_per_user": rep["tokens_per_s_per_user"],
                "tokens_identical_blocking_vs_deferred": rep.get("tokens_identical_blocking_vs_deferred"),
                "first_divergence_blocking_rerun": rep.get("first_divergence_blocking_rerun"),
            }
            out["sources"][f"tokenout_{tag}"] = f"doc/optimized_full_model/tokenout_{tag}_48layers.json"
    pb = _load(O / "prefill_bucket_bench_48layers.json")
    if pb:
        out["ttft_ms"] = {str(r["len"]): {k: v for k, v in r.items() if k.endswith("_ms")} for r in pb["timing"]}
        out["sources"]["ttft"] = "doc/optimized_full_model/prefill_bucket_bench_48layers.json"
    pe = _load(O / "prefill_bucket_bench_2layers.json")
    if pe:
        out["prefill_equivalence_2layers"] = pe["equiv"]
        out["sources"]["prefill_equivalence"] = "doc/optimized_full_model/prefill_bucket_bench_2layers.json"
    oc = (_load(O / "decode_op_count.json") or []) + (_load(O / "decode_op_count_r4.json") or [])
    if oc:
        out["decode_ops_per_layer"] = {f"{r['config']}/{r['kind']}": r["n_ops"] for r in oc}
        out["decode_ops_final_policy"] = {
            "linear_attention": 82,
            "full_attention": 77,
            "source": "decode_op_count_r4.json (ops_all_normfold_convl1 = the selected knobs; moe_router_l1 is a memory-config knob and changes no op count) + decode_op_count.json ops_all/full_attention",
        }
        out["sources"]["decode_ops"] = "doc/optimized_full_model/decode_op_count*.json"
    sw = {}
    for f in sorted(O.glob("sweeps/decode_ops*.jsonl")):
        for r in _jsonl(f):
            if r.get("decode_ms") is not None:
                sw[f"{r['name']}/layer{r['layer_idx']}"] = {
                    "decode_ms": r["decode_ms"],
                    "pcc_decode": r.get("pcc_decode"),
                    "file": f"doc/optimized_full_model/sweeps/{f.name}",
                }
    if sw:
        out["decode_knob_sweeps"] = sw
    rd = {}
    for f in sorted(O.glob("readiness/*.json")):
        try:
            rd[f.name] = json.loads(f.read_text())
        except Exception:  # noqa: BLE001
            pass
    if rd:
        out["readiness_reports"] = sorted(rd)
        out["sources"]["readiness"] = "doc/optimized_full_model/readiness/"
    # layer-stack lower bound from the stage-7 Tracy decode windows (max over devices, per replay)
    lb = {}
    try:
        from models.autoports.qwen_qwen3_5_122b_a10b.tools.perf_tables_ofm import (
            AFTER,
            by_device,
            dev_ms,
            replays,
            rows,
        )

        per_kind = {}
        for k, n in (("gdn", 36), ("attn", 12)):
            r = rows(Path(AFTER) / f"{k}_traced_decode_mesh_perf_report.csv")
            if r:
                bd = by_device(r)
                div = replays(list(bd.values())[0])
                per_kind[k] = {
                    "device_ms_max": round(max(dev_ms(v, div) for v in bd.values()), 4),
                    "ops_per_replay": len(list(bd.values())[0]) // div,
                    "layers": n,
                }
        if len(per_kind) == 2:
            lb["layer_stack_lower_bound_ms"] = round(
                36 * per_kind["gdn"]["device_ms_max"] + 12 * per_kind["attn"]["device_ms_max"], 3
            )
            lb["per_kind"] = per_kind
            lb["source"] = "doc/optimized_full_model/tracy/*_traced_decode_mesh_perf_report.csv"
    except Exception as e:  # noqa: BLE001
        lb["error"] = str(e)[:200]
    out["lower_bound"] = lb
    (O / "perf_summary.json").write_text(json.dumps(out, indent=1) + "\n")
    print(
        json.dumps(
            {k: v for k, v in out.items() if k not in ("prefill_equivalence_2layers", "decode_knob_sweeps")}, indent=1
        )
    )


if __name__ == "__main__":
    sys.exit(main())
