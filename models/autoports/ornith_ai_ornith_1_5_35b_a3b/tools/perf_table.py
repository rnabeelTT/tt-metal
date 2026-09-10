#!/usr/bin/env python3
"""Collect bench_model.py results into one markdown table.

Kept separate from the benchmark so the table can be rebuilt from artifacts at any time
without re-running the model.
"""
from __future__ import annotations

import argparse
import json
import pathlib


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", required=True, type=pathlib.Path)
    ap.add_argument("--out", required=True, type=pathlib.Path)
    a = ap.parse_args()

    rows = []
    for f in sorted(a.artifacts.glob("*perf-*.json")):
        try:
            d = json.loads(f.read_text())
        except json.JSONDecodeError:
            continue
        kernel = "batched" if f.stem.endswith("batched") else ("loop" if f.stem.endswith("loop") else "-")
        for r in d.get("runs", []):
            rows.append(
                (
                    d["mesh"],
                    d["devices"],
                    d["routed_expert_dtype"],
                    d["experts_per_device"],
                    kernel,
                    r["isl"],
                    r["ttft_ms"],
                    r["prefill_tok_s"],
                    r.get("decode_tok_s"),
                )
            )
    if not rows:
        print("no perf artifacts yet")
        return 0

    lines = [
        "# Measured performance",
        "",
        "| mesh | chips | expert dtype | experts/dev | MoE kernel | ISL | TTFT (ms) | prefill tok/s | decode tok/s |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for m, nd, dt, epd, k, isl, ttft, ptps, dtps in sorted(rows):
        lines.append(
            f"| {m} | {nd} | `{dt}` | {epd} | {k} | {isl} | {ttft:.0f} | "
            f"{ptps:.1f} | {dtps if dtps is not None else 'n/a'} |"
        )
    lines += [
        "",
        "Decode is dispatch-bound in the `loop` kernel: it issues ~3 matmuls plus a",
        "mask, a reduce and two multiplies per LOCAL expert, so ~30,700 op dispatches",
        "per forward on a single chip. `batched` collapses that to ~12 ops per layer.",
    ]
    a.out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
