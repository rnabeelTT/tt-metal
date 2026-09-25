# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Datatype-sweep bookkeeping (stage 8): collect the per-candidate readiness runs into sweep_results.json / .csv, rank
them, and draw the Pareto charts.

    python -m ...tools.sweep_datatypes collect --logs ~/qwen35-122b-bringup/logs/08 [--top1 0.90 --top5 0.98]
    python -m ...tools.sweep_datatypes plot

Inputs (written by scripts/08-sweep.sh): doc/datatype_sweep/configs/<id>.json (policy), artifacts/runs/teacher-forcing_aime24_<id>.json
(readiness report), the run log (PRECISION_EFFECTIVE / MEMORY_REPORT / command lines). Ranking metric: trace-verified
teacher-forcing decode t/s/u (the readiness runner always calls generate(enable_trace=True); the generator's decode is the
captured trace, `trace_verified` = the log shows the decode trace captured and the program-cache guard held).
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parents[1]
D = MODEL_DIR / "doc" / "datatype_sweep"
HARDWARE = "4 x Blackhole p300c dies (2 x p300, PCIe x4 each), 31.9 GiB GDDR per die, fw 19.13.1.0"
MESH = "(1,4) P300x2, FABRIC_1D"
REGIME = "AIME24 chat-template prompt (161 tokens), 100 reference tokens, B=1: prefill TTFT + traced teacher-forcing decode (readiness_check.run_teacher_forcing)"


def _grep(log: Path, tag: str):
    if not log.is_file():
        return None
    out = None
    for line in log.read_text(errors="replace").splitlines():
        i = line.find(tag + " ")
        if i >= 0:
            try:  # raw_decode: the print() line can be interleaved with a loguru line on the same row
                out = json.JSONDecoder().raw_decode(line[i + len(tag) + 1 :])[0]
            except json.JSONDecodeError:
                pass
    return out  # the last one (MEMORY_REPORT: after all captures)


def _policy_summary(pol):
    wg = pol["weight_groups"]
    dt = {
        "attention": wg["attention"]["dtype"],
        "linear_attention": wg["linear_attention"]["dtype"],
        "router": wg["router"]["dtype"],
        "shared_moe": wg["shared_moe"]["dtype"],
        "routed_moe_gate_up": wg["routed_moe"]["gate_up"],
        "routed_moe_down": wg["routed_moe"]["down"],
        "lm_head": wg["lm_head"]["dtype"],
        "embedding": wg["embedding"]["dtype"],
        "norms": wg["norms"]["dtype"],
        "activation": pol["activation_dtype"],
        "residual": pol["residual_dtype"],
        "ccl": pol["ccl_dtype"],
        "kv_cache": pol["kv_cache_dtype"],
        "linear_state": pol["linear_state_dtype"],
        "logits": pol["logits_dtype"],
        "sampling": pol["sampling_dtype"],
        "layer_exceptions": pol.get("layer_exceptions", {}),
    }
    return dt, dict(pol["compute_fidelities"])


def collect(a):
    rows = []
    for cfg in sorted((D / "configs").glob("*.json")):
        cid = cfg.stem
        pol = json.loads(cfg.read_text())
        rep = D / "artifacts" / "runs" / f"teacher-forcing_aime24_{cid}.json"
        log = Path(a.logs) / f"teacher-forcing_aime24_{cid}.log"
        dt, cf = _policy_summary(pol)
        row = {
            "config_id": cid,
            "description": pol.get("description", ""),
            "precision_config_path": str(cfg.relative_to(MODEL_DIR.parents[2])),
            "dtype_policy": dt,
            "compute_fidelity_policy": cf,
            "measurement_regime": REGIME,
            "hardware": HARDWARE,
            "mesh": MESH,
            "command": f"QWEN35_PRECISION_CONFIG={cfg.relative_to(MODEL_DIR.parents[2])} python -m models.autoports.qwen_qwen3_5_122b_a10b.tools.readiness teacher-forcing --reference bringup/references/qwen35_122b_aime24_chat_g100.refpt --report {rep.relative_to(MODEL_DIR.parents[2])}",
            "status": "not_run",
        }
        if rep.is_file():
            r = json.loads(rep.read_text())
            r = r[0] if isinstance(r, list) else r
            row.update(
                {
                    "teacher_forcing_top1": r.get("top1"),
                    "teacher_forcing_top5": r.get("top5"),
                    "teacher_forcing_top100": r.get("top100"),
                    "tokens": r.get("total"),
                    "ttft_ms": r.get("ttft_ms"),
                    "trace_verified_teacher_forcing_decode_t_s_u": r.get("decode_t/s/u"),
                    "teacher_forcing_e2e_t_s_u": r.get("e2e_t/s/u"),
                    "result_artifact": str(rep.relative_to(MODEL_DIR.parents[2])),
                }
            )
            eff = _grep(log, "PRECISION_EFFECTIVE")
            mem = _grep(log, "MEMORY_REPORT")
            txt = log.read_text(errors="replace") if log.is_file() else ""
            row["runtime_policy_effective"] = eff
            row["policy_consumed"] = bool(eff) and eff.get("mismatches") == [] and eff.get("config_id") == cid
            row["trace_verified"] = ("decode trace (model + on-device sampling) captured" in txt) and (
                "program cache entries after warm-up" in txt
            )
            row["memory_report"] = mem
            if mem:
                d = mem.get("dram", {})
                row["dram_allocated_gib_per_device"] = round(
                    d.get("allocated_bytes_per_bank", 0) * d.get("banks", 0) / 2**30, 3
                )
            gate = (
                row["teacher_forcing_top1"] is not None
                and row["teacher_forcing_top1"] >= a.top1
                and row["teacher_forcing_top5"] >= a.top5
                and row["teacher_forcing_top100"] >= a.top100
            )
            row["accuracy_gate_pass"] = bool(gate)
            row["pass"] = bool(gate and row["policy_consumed"] and row["trace_verified"])
            row["status"] = "pass" if row["pass"] else ("fail_accuracy" if not gate else "fail_evidence")
        elif log.is_file():
            txt = log.read_text(errors="replace")
            m = re.findall(r"(TT_FATAL[^\n]{0,200}|Traceback[^\n]{0,200}|Error[^\n]{0,200})", txt)
            row["status"] = "fail_run" if m else "incomplete"  # incomplete = the run has not written its report (yet)
            row["error"] = m[-1] if m else "no report"
        # extra references (prefill check; book 128 / 2048 / 8192-token prompts): every measured reference must pass the
        # same accuracy bar for the candidate to pass
        for extra in (
            "prefill-check_aime24",
            "teacher-forcing_book",
            "teacher-forcing_book_p2048",
            "teacher-forcing_book_p8192",
        ):
            f = D / "artifacts" / "runs" / f"{extra}_{cid}.json"
            if f.is_file():
                r = json.loads(f.read_text())
                r = r[0] if isinstance(r, list) else r
                row[extra.replace("-", "_")] = {
                    k: r.get(k) for k in ("top1", "top5", "top100", "ttft_ms", "decode_t/s/u") if k in r
                }
                ok = r.get("top1", 0) >= a.top1 and r.get("top5", 0) >= a.top5 and r.get("top100", 0) >= a.top100
                if not ok and row.get("pass"):
                    row["pass"] = False
                    row["status"] = f"fail_accuracy_{extra.split('_', 1)[1]}"
                row.setdefault("references_passed", []).append(f"{extra}:{'pass' if ok else 'FAIL'}")
        # complexity = policy fields changed vs the baseline (tie rule: simpler wins inside the noise band)
        base = json.loads((D / "configs" / "baseline_s7.json").read_text())

        def flat(d, pre=""):
            out = {}
            for k, v in d.items():
                if k in ("config_id", "description", "notes"):
                    continue
                if isinstance(v, dict):
                    out.update(flat(v, pre + k + "."))
                else:
                    out[pre + k] = v
            return out

        fb, fp = flat(base), flat(pol)
        row["changes_vs_baseline"] = sorted(k for k in set(fb) | set(fp) if fb.get(k) != fp.get(k))
        rows.append(row)
    # quality-gate overrides (qualitative suite verdicts recorded by hand in quality_gate_overrides.json)
    qg = D / "quality_gate_overrides.json"
    over = json.loads(qg.read_text()) if qg.is_file() else {}
    for row in rows:
        q = over.get(row["config_id"])
        row["quality_gate"] = q or "not_run"
        if q and q.startswith("fail") and row.get("pass"):
            row["pass"] = False
            row["status"] = "fail_quality"
    passing = [r for r in rows if r.get("pass")]
    sel = None
    if passing:
        fastest = max(passing, key=lambda r: r["trace_verified_teacher_forcing_decode_t_s_u"])
        # tie rule (skill: "within measurement noise, prefer the simpler and safer one"): among the passing candidates within
        # --noise t/s/u of the fastest, take the one with the fewest policy changes vs the baseline; ties -> the faster one
        band = [
            r
            for r in passing
            if fastest["trace_verified_teacher_forcing_decode_t_s_u"] - r["trace_verified_teacher_forcing_decode_t_s_u"]
            <= a.noise
        ]
        for r in band:
            r["within_noise_of_fastest"] = True
            t = D / "artifacts" / f"tokenout_48layers_{r['config_id']}.json"
            if t.is_file():  # the 64-step token-out bench (model-only replay stable to ~0.001 ms/step) breaks TF ties
                r["tokenout_model_only_ms"] = json.loads(t.read_text())["ms_per_step"]["replay_model_only"]
        measured = [r for r in band if r.get("tokenout_model_only_ms") is not None]
        if len(measured) >= 2 and fastest in measured:
            # tie-break by the precise bench among the benchmarked band members: fastest model-only step wins. Band members
            # without a bench row are slower than the TF-fastest by construction (which is benchmarked here) and cannot be
            # promoted over it; if the TF-fastest itself had no bench row the rule falls through to the skill's tie rule below
            sel = min(measured, key=lambda r: r["tokenout_model_only_ms"])
        else:
            sel = min(
                band, key=lambda r: (len(r["changes_vs_baseline"]), -r["trace_verified_teacher_forcing_decode_t_s_u"])
            )
    for r in rows:
        r["selected"] = bool(sel and r["config_id"] == sel["config_id"])
    rows.sort(key=lambda r: -(r.get("trace_verified_teacher_forcing_decode_t_s_u") or 0))
    out = {
        "accuracy_gate": {
            "teacher_forcing_top1_min": a.top1,
            "teacher_forcing_top5_min": a.top5,
            "teacher_forcing_top100_min": a.top100,
        },
        "selection_metric": "trace_verified_teacher_forcing_decode_t_s_u",
        "selection_rule": f"fastest passing candidate on the traced teacher-forcing metric; candidates within {a.noise} t/s/u of it are a single-run TF tie, resolved by the 64-step token-out bench (model-only replay ms, repeatable to ~0.001 ms) when the tied candidates were benchmarked, else by the fewest policy changes vs baseline_s7 (skill tie rule); every measured reference must meet the bar; quality_gate_overrides.json can fail a candidate on the qualitative suite",
        "noise_band_t_s_u": a.noise,
        "selected": sel["config_id"] if sel else None,
        "results": rows,
    }
    (D / "sweep_results.json").write_text(json.dumps(out, indent=2) + "\n")
    cols = [
        "config_id",
        "status",
        "selected",
        "teacher_forcing_top1",
        "teacher_forcing_top5",
        "teacher_forcing_top100",
        "ttft_ms",
        "trace_verified_teacher_forcing_decode_t_s_u",
        "teacher_forcing_e2e_t_s_u",
        "trace_verified",
        "policy_consumed",
        "accuracy_gate_pass",
        "quality_gate",
        "references_passed",
        "changes_vs_baseline",
        "within_noise_of_fastest",
        "dram_allocated_gib_per_device",
        "dtype_policy",
        "compute_fidelity_policy",
        "measurement_regime",
        "hardware",
        "mesh",
        "precision_config_path",
        "result_artifact",
        "command",
    ]
    with (D / "sweep_results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(v) if isinstance(v, (dict, list)) else v) for k, v in r.items() if k in cols})
    for r in rows:
        print(
            f"{r['config_id']:26s} {r['status']:14s} top1={r.get('teacher_forcing_top1')} top5={r.get('teacher_forcing_top5')} "
            f"top100={r.get('teacher_forcing_top100')} ttft={r.get('ttft_ms') and round(r['ttft_ms'], 1)} "
            f"dec={r.get('trace_verified_teacher_forcing_decode_t_s_u') and round(r['trace_verified_teacher_forcing_decode_t_s_u'], 2)} "
            f"consumed={r.get('policy_consumed')} traced={r.get('trace_verified')} {'SELECTED' if r['selected'] else ''}"
        )
    print("selected:", out["selected"])


def _pareto(points):
    """Non-dominated set for (accuracy higher-better, perf higher-better)."""
    pts = sorted(points, key=lambda p: (-p[0], -p[1]))
    front, best = [], -1.0
    for acc, perf, cid in pts:
        if perf > best:
            front.append((acc, perf, cid))
            best = perf
    return sorted(front)


def plot(a):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    res = json.loads((D / "sweep_results.json").read_text())
    rows = [r for r in res["results"] if r.get("trace_verified_teacher_forcing_decode_t_s_u")]
    rows.sort(key=lambda r: -r["trace_verified_teacher_forcing_decode_t_s_u"])  # rank 1 = fastest
    gate = res["accuracy_gate"]
    for key, gmin, fname, label in (
        ("teacher_forcing_top1", gate["teacher_forcing_top1_min"], "top1_perf_pareto.png", "teacher-forcing top-1"),
        ("teacher_forcing_top5", gate["teacher_forcing_top5_min"], "top5_perf_pareto.png", "teacher-forcing top-5"),
    ):
        fig, (ax, lax) = plt.subplots(1, 2, figsize=(13, 6.5), dpi=150, gridspec_kw={"width_ratios": [2.2, 1]})
        pts = [(r[key] * 100, r["trace_verified_teacher_forcing_decode_t_s_u"], r["config_id"]) for r in rows]
        front = _pareto(pts)
        fx, fy = [p[0] for p in front], [p[1] for p in front]
        if len(front) > 1:
            ax.plot(fx, fy, color="#7f8c8d", lw=1.2, zorder=1, label="Pareto frontier")
        else:  # degenerate frontier (every candidate at the same accuracy): mark the single non-dominated point
            ax.scatter(
                fx,
                fy,
                s=420,
                facecolors="none",
                edgecolors="#7f8c8d",
                lw=1.5,
                zorder=1,
                label="Pareto frontier (single point)",
            )
        # numbered markers (rank by throughput) + a legend table: no label overlap when many points share one x column
        legend_lines = []
        for n, r in enumerate(rows, 1):
            x, y = r[key] * 100, r["trace_verified_teacher_forcing_decode_t_s_u"]
            sel, ok = r.get("selected"), r.get("pass")
            color = "#e74c3c" if sel else ("#2980b9" if ok else "#95a5a6")
            marker = "*" if sel else ("o" if ok else "x")
            ax.scatter(
                [x], [y], s=330 if sel else 90, c=color, marker=marker, zorder=3, edgecolors="black" if sel else "none"
            )
            # spread the number labels of one x column horizontally by rank parity so they never overprint
            side = 1 if n % 2 else -1
            ax.annotate(
                str(n),
                (x, y),
                textcoords="offset points",
                xytext=(9 * side, -3),
                fontsize=7.5,
                ha="left" if side > 0 else "right",
                color="black",
                zorder=4,
            )
            legend_lines.append(
                f"{n:>2}. {r['config_id']:<34s} {r[key] * 100:5.1f} %  {y:6.2f} t/s/u{'  <- selected' if sel else ''}{'' if ok else '  (rejected)'}"
            )
        ax.axvline(gmin * 100, color="#c0392b", ls=":", lw=1.4, label=f"minimum allowed {label} ({gmin * 100:.0f} %)")
        ax.set_xlabel(f"{label} vs HF reference (%)  [AIME24 chat, 100 tokens]")
        ax.set_ylabel("trace-verified teacher-forcing decode (tokens / s / user)")
        ax.set_title("Qwen3.5-122B-A10B on (1,4) Blackhole: precision candidates")
        ax.grid(alpha=0.3)
        xs = [p[0] for p in pts]
        ax.set_xlim(min(xs + [gmin * 100]) - 2, 101.5)
        ax.scatter([], [], c="#e74c3c", marker="*", s=160, label="selected")
        ax.scatter([], [], c="#2980b9", marker="o", label="passes the gates")
        ax.scatter([], [], c="#95a5a6", marker="x", label="rejected")
        ax.legend(loc="lower left", fontsize=8)
        lax.axis("off")
        lax.text(
            0.0,
            1.0,
            "rank. config                              acc     decode\n" + "\n".join(legend_lines),
            family="monospace",
            fontsize=7.3,
            va="top",
            ha="left",
            transform=lax.transAxes,
        )
        fig.tight_layout()
        fig.savefig(D / fname)
        plt.close(fig)
        print("wrote", D / fname)


def table(a):
    """Markdown results table for the README (from sweep_results.json)."""
    res = json.loads((D / "sweep_results.json").read_text())
    print(
        "| config | status | TF top-1 | TF top-5 | TF top-100 | TTFT ms | TF decode t/s/u | policy consumed | traced | quality gate | DRAM GiB/dev | selected |"
    )
    print("|---|---|---:|---:|---:|---:|---:|---|---|---|---:|---|")
    for r in res["results"]:
        f = lambda k, nd=2: ("-" if r.get(k) is None else (f"{r[k]:.{nd}f}" if isinstance(r[k], float) else str(r[k])))
        print(
            f"| `{r['config_id']}` | {r['status']} | {f('teacher_forcing_top1')} | {f('teacher_forcing_top5')} | {f('teacher_forcing_top100')} | "
            f"{f('ttft_ms', 1)} | {f('trace_verified_teacher_forcing_decode_t_s_u')} | {r.get('policy_consumed', '-')} | {r.get('trace_verified', '-')} | "
            f"{str(r.get('quality_gate', '-')).split(' (')[0]} | {f('dram_allocated_gib_per_device', 3)} | {'**yes**' if r.get('selected') else 'no'} |"
        )


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("table")
    c = sub.add_parser("collect")
    c.add_argument("--logs", default=str(Path.home() / "qwen35-122b-bringup" / "logs" / "08"))
    c.add_argument("--top1", type=float, default=0.90)
    c.add_argument("--top5", type=float, default=0.98)
    c.add_argument("--top100", type=float, default=1.0)
    c.add_argument(
        "--noise",
        type=float,
        default=0.3,
        help="t/s/u band treated as a measurement tie (single-run TF spread of the neutral knobs was +-0.15)",
    )
    sub.add_parser("plot")
    a = ap.parse_args(argv)
    {"collect": collect, "plot": plot, "table": table}[a.cmd](a)


if __name__ == "__main__":
    main()
