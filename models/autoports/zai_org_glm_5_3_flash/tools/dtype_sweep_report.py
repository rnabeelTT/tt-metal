# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage 8 (datatype sweep) report: doc/datatype_sweep/runs/<run>/record.json -> sweep_results.{json,csv} and the two
Pareto charts (top-1 / top-5 vs trace-verified teacher-forcing decode t/s/u).

Context cost: the 1-slot capacity of the shipped policy is 104,448 tokens (doc/full_model/memory_budget.json, 818 blocks
of 128 tokens, 11,968 B per token per die). A candidate that allocates D more bytes per die after build loses
ceil(D / (128 * 11968)) blocks; max_model_len is floored to the 2048-row prefill bucket.

usage: python -m models.autoports.zai_org_glm_5_3_flash.tools.dtype_sweep_report [--selected shipped]
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

DOC = Path(__file__).resolve().parents[1] / "doc" / "datatype_sweep"
TOP1_MIN, TOP5_MIN, TOP100_MIN = 0.90, 0.98, 1.0
SHIPPED_BLOCKS, BLOCK, DSA_BPT, BANKS = 818, 128, 11968, 8


def load_runs(runs_dir: Path):
    out = []
    for p in sorted(runs_dir.glob("*/record.json")):
        r = json.loads(p.read_text())
        r["_run"] = p.parent.name
        out.append(r)
    return out


def dsa_bpt(r) -> int:
    """DSA cache bytes per token per die of the run: 11 layers x (latent 512 x 2 B + pool keys 64 B); fp8 latent: 1 B."""
    lat = 1 if (r.get("precision") or {}).get("kv_latent_dtype") == "fp8_e4m3" else 2
    return 11 * (512 * lat + 64)


def row(r, bases, cfg_dir: Path):
    base_alloc = bases.get(r.get("base_commit") or "s1")
    tf = (r.get("teacher_forcing") or [{}])[0]
    pc = (r.get("prefill_check") or [{}])[0]
    dec = (r.get("decode") or {}).get("1") or (r.get("decode") or {}).get(1) or {}
    ttft = (r.get("ttft") or {}).get("eager", {})
    t161 = ttft.get("161") or ttft.get(161) or {}
    t128 = ttft.get("128") or ttft.get(128) or {}
    t2048 = ttft.get("2048") or ttft.get(2048) or {}
    cid = r["id"]
    cap_id = "combo_a_kv" if cid == "selected_default" else cid  # the post-selection runs are the selected policy
    # decode programs (decode_specs) do not change the weights: the combo_a_kv capacity probes cover these variants
    cap_src = "combo_a_kv" if cid.startswith("combo_a_kv_g4") else cap_id
    gen = r.get("base_commit") or "ac03689d2bb (s8_snapshot)"
    cfg = cfg_dir / f"{cap_id}.json"
    desc = json.loads(cfg.read_text()).get("description", "") if cfg.exists() else ""
    bpt = dsa_bpt(r)
    seq = (r.get("args") or {}).get("seq_len", 16384)
    pool_build = (seq // BLOCK + 1) * BLOCK  # tokens of the DSA pool allocated by the measured build
    extra = 0
    if base_alloc:  # weights / constants beyond the shipped build, the measured builds' DSA pools taken out
        extra = (r.get("dram_alloc_per_bank", 0) - base_alloc) * BANKS - pool_build * (bpt - DSA_BPT)
    blocks = int((SHIPPED_BLOCKS * BLOCK * DSA_BPT - extra) // (BLOCK * bpt))
    ctx = blocks * BLOCK // 2048 * 2048
    ctx_status = "estimate (upper bound: excludes the max_context-dependent DSA constants and workspace)"
    caps = sorted((DOC / "capacity").glob(f"{cap_src}_s1*/capacity.json"))
    if abs(extra) < (16 << 20) and bpt == DSA_BPT:
        ctx, ctx_status = 104448, "measured (stage-6 capacity probe)"
    for cap in caps:  # the largest completed probe with copy accuracy 1.0
        c = json.loads(cap.read_text())
        if "after_prefill" in c and float(c.get("copy_accuracy", 0)) >= 1.0:
            v = (int(c["num_blocks"]) * BLOCK) // 2048 * 2048
            if "measured" not in ctx_status or v > ctx:
                ctx, ctx_status = v, f"measured ({cap.relative_to(DOC)})"
    if r.get("env_dsa_shard") or (r.get("_run", "").endswith("dsashard")):
        ctx, ctx_status = None, "block-sharded bf16 pool (not sized here)"
    replays = r.get("readiness_trace_replays")
    traced = bool(replays) and replays == 2 * int(tf.get("decode_tokens", 99))  # model + sampler trace per step
    top1, top5, top100 = tf.get("top1"), tf.get("top5"), tf.get("top100")
    ok = (
        top1 is not None
        and top1 >= TOP1_MIN
        and top5 >= TOP5_MIN
        and top100 >= TOP100_MIN
        and (pc.get("top1", 0) >= TOP1_MIN)
        and (pc.get("top5", 0) >= TOP5_MIN)
        and (pc.get("top100", 0) >= TOP100_MIN)
    )
    refs = {"p0": (tf, pc)}
    for k, v in (r.get("extra_refs") or {}).items():
        refs["p" + k.rsplit("_p", 1)[-1]] = (v["teacher_forcing"][0], v["prefill_check"][0])
    per_ref = {
        k: {
            "tf": [t.get("top1"), t.get("top5"), t.get("top100")],
            "pc": [c.get("top1"), c.get("top5"), c.get("top100")],
        }
        for k, (t, c) in refs.items()
    }
    min1 = min(min(t["top1"], c["top1"]) for t, c in refs.values()) if top1 is not None else None
    min5 = min(min(t["top5"], c["top5"]) for t, c in refs.values()) if top1 is not None else None
    min100 = min(min(t["top100"], c["top100"]) for t, c in refs.values()) if top1 is not None else None
    return {
        "run": r["_run"],
        "references": sorted(refs),
        "per_reference": per_ref,
        "min_top1_all_refs": min1,
        "min_top5_all_refs": min5,
        "min_top100_all_refs": min100,
        "tf_top1_sum": sum(t.get("matches_top1", 0) for t, _ in refs.values()),
        "pass_selection_rule": bool(
            len(refs) == 3 and min1 is not None and min1 >= 0.91 and min5 >= 0.98 and min100 >= 1.0
        ),
        "config_id": cid,
        "config_path": f"doc/datatype_sweep/configs/{cap_id}.json",
        "description": desc,
        "code": gen,
        "code_md5_map_digest": (r.get("code") or {}).get("md5_map_digest"),
        "requant": json.dumps((r.get("precision") or {}).get("requant") or {}, sort_keys=True),
        "fidelity": json.dumps((r.get("precision") or {}).get("fidelity") or {}, sort_keys=True),
        "kv_latent_dtype": (r.get("precision") or {}).get("kv_latent_dtype", "bf16"),
        "ccl_h_dtype": (r.get("precision") or {}).get("ccl_h_dtype", "bf16"),
        "precision_env": (
            "none (stage-7 policy)" if cid == "shipped" else (r.get("precision_env") or "unset (selected default)")
        ),
        "applied_overrides": len((r.get("precision") or {}).get("applied") or []),
        "tf_top1": top1,
        "tf_top5": top5,
        "tf_top100": top100,
        "tf_tokens": tf.get("total"),
        "pc_top1": pc.get("top1"),
        "pc_top5": pc.get("top5"),
        "pc_top100": pc.get("top100"),
        "tf_decode_tsu": tf.get("decode_t/s/u"),
        "tf_ttft_ms": tf.get("ttft_ms"),
        "readiness_trace_replays": replays,
        "trace_verified": traced,
        "model_trace_ms": dec.get("model_ms"),
        "tokenout_ms": dec.get("tokenout_ms"),
        "tokenout_tsu": 1e3 / dec["tokenout_ms"] if dec.get("tokenout_ms") else None,
        "generate_ms": dec.get("generate_ms"),
        "ttft_128_ms": (t128.get("warm") or {}).get("total_ms"),
        "ttft_161_ms": (t161.get("warm") or {}).get("total_ms"),
        "ttft_2048_ms": (t2048.get("warm") or {}).get("total_ms"),
        "dram_alloc_per_die_bytes": r.get("dram_alloc_per_bank", 0) * BANKS,
        "extra_dram_per_die_bytes": extra,
        "dsa_bytes_per_token": bpt,
        "max_context_1slot": ctx,
        "max_context_status": ctx_status,
        "max_context_note": "estimates: floor((818 x 128 x 11,968 - DRAM delta vs the shipped build of the same code) / "
        "(128 x bytes per token)), an upper bound; 'measured' = a capacity probe of this policy",
        "ptcheck_tokens_equal": (
            all(v.get("tokens_equal") for v in (r.get("ptcheck") or {}).values()) if r.get("ptcheck") else None
        ),
        "pass": ok and traced,
        "regime": "1 slot, max_seq_len 16384, warm; TF = readiness runner traced decode (host wall incl. 1 token readback per step)",
        "hardware": "p300x2 (4 Blackhole dies), mesh (1, 4), FABRIC_1D, TP4",
        "commit": r.get("commit"),
        "command": (
            " ".join(r["argv"])
            if r.get("argv")
            else (
                f"[reconstructed] GLM53_PRECISION={'none' if cid == 'shipped' else 'doc/datatype_sweep/configs/' + cid + '.json'} "
                "python -m ...tools.dtype_sweep (scripts/s8_sweep.sh / s8_batch_inner.sh)"
            )
        ),
        "script": r.get("script") or "[reconstructed] ~/glm53-flash-bringup/scripts/s8_sweep.sh or s8_batch_inner.sh",
    }


def pareto(points):
    """Non-dominated set (higher x and higher y better), sorted by x."""
    keep = [p for p in points if not any(q[0] >= p[0] and q[1] >= p[1] and q != p for q in points)]
    return sorted(keep)


def plot(rows, key, xmin, fname, selected, label=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gen_of = {r["run"]: r.get("code") or "?" for r in rows}
    gens = sorted({gen_of[r["run"]] for r in rows}, key=lambda g: (not g.startswith("ac03"), g))
    styles = [
        ("o", "#4c72b0", "#55a868"),
        ("s", "#dd8452", "#c49a00"),
        ("D", "#8172b3", "#937860"),
        ("^", "#64b5cd", "#8c8c8c"),
    ]
    sty = {g: styles[i % len(styles)] for i, g in enumerate(gens)}
    allpts = [(r[key] * 100, r["tf_decode_tsu"], r["run"]) for r in rows if r[key] is not None and r["tf_decode_tsu"]]
    lo = min(xmin * 100 - 2, min(p[0] for p in allpts if p[0] >= xmin * 100 - 10))
    pts = [p for p in allpts if p[0] >= lo]
    off = [p for p in allpts if p[0] < lo]
    fig, ax = plt.subplots(figsize=(11, 7), dpi=150)
    ax.set_facecolor("#fbfbfd")
    for g in gens:  # one marker / frontier per code generation: speeds compare only within a generation
        gp = [p for p in pts if gen_of[p[2]] == g]
        if not gp:
            continue
        mk, col, fcol = sty[g]
        for x, y, n in gp:
            sel = n == selected
            ax.scatter(
                x,
                y,
                s=130 if sel else 55,
                marker=mk,
                color="#d62728" if sel else col,
                edgecolor="white",
                lw=1.2,
                zorder=3,
            )
        ax.scatter([], [], marker=mk, color=col, label=f"code {g}")
        front = pareto([(x, y) for x, y, _ in gp])
        if len(front) > 1:
            ax.plot(
                [p[0] for p in front],
                [p[1] for p in front],
                "-",
                color=fcol,
                lw=1.8,
                alpha=0.85,
                label=f"Pareto frontier ({g})",
            )
    ax.axvline(xmin * 100, ls=":", color="#c44e52", lw=1.6, label=f"minimum {key.split('_')[1]} = {xmin * 100:.0f} %")
    ys = [p[1] for p in pts]
    y0, y1 = min(ys), max(ys)
    pad = (y1 - y0) * 0.08 or 0.05
    ax.set_ylim(y0 - pad, y1 + pad)
    ax.set_xlim(lo - 0.5, 100.6 + (3.5 if len({round(p[0]) for p in pts}) == 1 else 0.6))
    # labels: points sharing an x column get evenly spaced label rows (leader lines), so none overlap
    cols = {}
    for x, y, n in pts:
        cols.setdefault(round(x, 1), []).append((y, n))
    for x, items in cols.items():
        items.sort()
        k = len(items)
        gap = (y1 - y0 + 2 * pad) / 26
        lys, prev = [], -1e9
        for y, _ in items:  # labels near their points, pushed up just enough not to overlap
            prev = max(y, prev + gap)
            lys.append(prev)
        shift = max(0.0, lys[-1] - (y1 + pad * 0.9))
        lys = [v - shift for v in lys]
        for j, (y, n) in enumerate(items):
            ly = lys[j]
            sel = n == selected
            ax.annotate(
                n,
                (x, y),
                xytext=(x + 0.35, ly),
                textcoords="data",
                fontsize=8,
                va="center",
                color="#d62728" if sel else "#333",
                weight="bold" if sel else "normal",
                arrowprops=dict(arrowstyle="-", color="#bbb", lw=0.6) if abs(ly - y) > gap * 0.3 else None,
            )
    for i, (x, y, n) in enumerate(off):
        ax.text(
            lo - 0.3,
            y0 - pad * 0.6 + i * pad * 0.4,
            f"off-chart: {n} ({x:.0f} %, {y:.2f} t/s/u)",
            fontsize=8,
            color="#c44e52",
        )
    ax.set_xlabel(label or f"teacher-forcing {key.split('_')[1]} accuracy (%), AIME24 chat reference p0, 100 positions")
    ax.set_ylabel("trace-verified teacher-forcing decode (t/s/u)")
    ax.set_title(f"GLM-5.3-Flash (E192) datatype sweep: {key.split('_')[1]} vs decode speed (p300x2, TP4, 1 slot)")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_xticks([t for t in ax.get_xticks() if lo - 0.5 <= t <= 100])
    fig.tight_layout()
    fig.savefig(DOC / fname)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selected", default="combo_a_kv_r3")
    a = ap.parse_args()
    # the post-selection warm rebuild (decode only) and the GLM53_DSA_SHARD fallback run are not sweep candidates
    runs = [
        r
        for r in load_runs(DOC / "runs")
        if r["_run"]
        not in (
            "selected_default_warm",
            "selected_default_r1_warm",
            "selected_default_r2_warm",
            "selected_default_dsashard",
            "smoke_g4_reduced",
        )
    ]
    bases = {}
    for name, key in (("shipped", "s1"), ("shipped_r1", "b8198430147"), ("shipped_r2", "914af5b9594")):
        b = next((r for r in runs if r["_run"] == name), None)
        if b is not None:
            bases[key] = b.get("dram_alloc_per_bank")
    rows = [row(r, bases, DOC / "configs") for r in runs]
    (DOC / "sweep_results.json").write_text(
        json.dumps(
            {
                "thresholds": {"top1": TOP1_MIN, "top5": TOP5_MIN, "top100": TOP100_MIN},
                "selected": a.selected,
                "results": rows,
            },
            indent=1,
        )
        + "\n"
    )
    with open(DOC / "sweep_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    plot(rows, "tf_top1", TOP1_MIN, "top1_perf_pareto.png", a.selected)
    plot(rows, "tf_top5", TOP5_MIN, "top5_perf_pareto.png", a.selected)
    r3 = [dict(r, tf_top1=r["min_top1_all_refs"]) for r in rows if len(r["references"]) == 3]
    if r3:
        plot(
            r3,
            "tf_top1",
            0.91,
            "min_top1_perf.png",
            a.selected,
            label="minimum top-1 over p0 / p1 / p2 (TF and prefill check)",
        )
    for r in rows:
        print(
            f"{r['run']:<22} top1 {r['tf_top1']} top5 {r['tf_top5']} top100 {r['tf_top100']} | pc {r['pc_top1']}/{r['pc_top5']}"
            f" | tf {r['tf_decode_tsu'] and round(r['tf_decode_tsu'], 2)} t/s/u tokenout {r['tokenout_ms'] and round(r['tokenout_ms'], 2)} ms"
            f" | ttft161 {r['ttft_161_ms'] and round(r['ttft_161_ms'])} | ctx {r['max_context_1slot']} | pass {r['pass']}"
        )


if __name__ == "__main__":
    main()
