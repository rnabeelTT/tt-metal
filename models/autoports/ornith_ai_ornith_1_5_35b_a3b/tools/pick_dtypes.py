#!/usr/bin/env python3
"""Turn the stage-09 sweep results into a per-mesh expert-precision decision.

Reads every ``09-<mesh>-<recipe>.json`` the sweep produced and picks, for each mesh,
the CHEAPEST recipe still inside the accuracy budget — cheapest because a smaller
expert footprint buys context and batch, which are real user-visible capability, and
there is no reward for spending DRAM on precision the outputs cannot distinguish.

The budget is expressed against the bf16 4-chip run rather than against the HF model,
so it isolates quantization error from porting error: a layer that is 0.3% off in bf16
is a bring-up bug and belongs to an earlier stage, not to the dtype decision.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re

# Top-1 agreement with the bf16 reference under teacher forcing. Chosen over
# perplexity because it maps directly onto what a user sees: a token that differs
# is a visible divergence, whereas a small logit shift usually is not.
MIN_TOP1_AGREEMENT = 0.98
MIN_PCC = 0.99

# Cheapest first. The sweep only runs recipes a mesh could plausibly hold, so a
# missing result means "did not fit or did not converge", which is itself a decision.
PREFERENCE = ["bfloat4_b", "bfloat4_bfloat8", "bfloat8_b", "bfloat16"]

FOOTPRINT_GB = {  # total model bytes; per-chip is this over the device count
    "bfloat4_b": 23.3,
    "bfloat4_bfloat8": 28.7,
    "bfloat8_b": 39.4,
    "bfloat16": 69.6,
}
DEVICES = {"P150": 1, "P300": 2, "P150x4": 4}


def load(art: pathlib.Path):
    runs = {}
    for f in sorted(art.glob("09-*-*.json")):
        m = re.fullmatch(r"09-(P150x4|P300|P150)-(.+)\.json", f.name)
        if not m:
            continue
        try:
            runs.setdefault(m.group(1), {})[m.group(2)] = json.loads(f.read_text())
        except json.JSONDecodeError:
            print(f"  ! {f.name} is not valid JSON — treating as a failed run")
    return runs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", required=True, type=pathlib.Path)
    ap.add_argument("--out", required=True, type=pathlib.Path)
    a = ap.parse_args()

    runs = load(a.artifacts)
    if not runs:
        print("no sweep results found — stage 09 produced nothing to decide on")
        return 1

    lines = [
        "# Expert-precision decision (stage 09)",
        "",
        "Cheapest recipe per mesh that stays inside the accuracy budget",
        f"(top-1 agreement >= {MIN_TOP1_AGREEMENT}, PCC >= {MIN_PCC} vs the bf16 reference).",
        "",
        "| mesh | chosen | top-1 | PCC | total GB | GB/chip | rejected |",
        "|---|---|---|---|---|---|---|",
    ]
    ok = True
    for mesh in ("P150", "P300", "P150x4"):
        results = runs.get(mesh, {})
        chosen, rejected = None, []
        for r in PREFERENCE:
            d = results.get(r)
            if d is None:
                continue
            t1 = d.get("top1_agreement", d.get("top1", 0.0))
            pcc = d.get("pcc", 0.0)
            if t1 >= MIN_TOP1_AGREEMENT and pcc >= MIN_PCC:
                chosen = (r, t1, pcc)
                break
            rejected.append(f"{r} (top1={t1:.4f}, pcc={pcc:.4f})")
        if chosen is None:
            lines.append(f"| {mesh} | **none passed** | - | - | - | - | {'; '.join(rejected) or 'no runs'} |")
            ok = False
            continue
        r, t1, pcc = chosen
        gb = FOOTPRINT_GB[r]
        lines.append(
            f"| {mesh} | `{r}` | {t1:.4f} | {pcc:.4f} | {gb:.1f} | "
            f"{gb / DEVICES[mesh]:.1f} | {'; '.join(rejected) or '—'} |"
        )
    lines += [
        "",
        "Update `MESH_PRECISION` in the autoport's `tt/model_config.py` to match this table.",
        "",
        "Router and shared expert stay bf16 on every mesh: together 0.5% of weights,",
        "and the router decides which experts run at all.",
    ]
    a.out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
