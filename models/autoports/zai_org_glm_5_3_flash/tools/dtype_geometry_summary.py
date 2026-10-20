# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage 8 review #2 (P2-B): summary of tools/mc_matmul_sweep.py --wdtype bfp4 decode records.

Per role: the current decode program (tt/multichip_decoder.DEFAULT_MC role spec, "" = ttnn default) at the base
fidelity, the fastest candidate that make_role_mm can build (1D: the core count divides the N tiles, as IL1DMatmul
picks it; DS: any), its gain, whether its output is bit-identical to the current program (out_sha), and its error
against fp64 next to the current program's (adoption rule (a): the candidate's error is within the current program's
noise: rel_err <= 1.05 x current and |scale - 1| <= |current scale - 1| + 1e-4).

usage: python -m models.autoports.zai_org_glm_5_3_flash.tools.dtype_geometry_summary <records.jsonl> [--out summary.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# DEFAULT_MC role specs (commit 914af5b9594), as mc_matmul_sweep candidate tags
CURRENT = {
    "kda_in": "default",
    "kda_out": "1d_c16_ib4",
    "dsa_o": "1d_c16_ib4",
    "dsa_qb": "1d_c16_ib4",
    "pack_a": "1d_c32_ib8",
    "shared_gu": "1d_c32_ib16",
    "shared_dn": "1d_c16_ib4",
    "mlp_gu": "1d_c16_ib4",
    "mlp_dn": "1d_c16_ib4",
    "lm_head": "default",
}
FIELD = {"pack_a": "pack_a_spec"}  # MCOpts field of a role (others: same name)


def to_spec(tag: str) -> str:
    if tag == "default":
        return ""
    kind, *parts = tag.split("_")
    return ":".join([kind] + parts)


def buildable(r) -> bool:
    if r["cand"] == "default" or r["family"] == "ds":
        return True
    c = int(r["cand"].split("_")[1][1:])
    return (-(-r["N"] // 32)) % c == 0  # N tiles incl. the padded last tile (kda_in: 8336 -> 261)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("records")
    ap.add_argument("--fidelity", default="MathFidelity.LoFi")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    recs = [json.loads(x) for x in Path(a.records).read_text().splitlines() if x.strip()]
    out = {}
    for role, cur_tag in CURRENT.items():
        rr = [r for r in recs if r["role"] == role and r.get("mode") == "decode" and "us" in r]
        base = [r for r in rr if r["fidelity"] == a.fidelity]
        cur = next((r for r in base if r["cand"] == cur_tag), None)
        if cur is None:
            out[role] = {
                "error": f"current program {cur_tag} not measured",
                "refused": [r["cand"] for r in recs if r["role"] == role and "error" in r],
            }
            continue
        ok = [r for r in base if buildable(r)]
        best = min(ok, key=lambda r: r["us"])
        tol = (
            lambda r: r["rel_err"] <= 1.05 * cur["rel_err"] + 1e-6
            and abs(r["scale"] - 1) <= abs(cur["scale"] - 1) + 1e-4
        )
        best_ok = min([r for r in ok if tol(r)], key=lambda r: r["us"])
        out[role] = {
            "field": FIELD.get(role, role),
            "current": {"tag": cur_tag, "us": round(cur["us"], 2), "rel_err": cur["rel_err"], "scale": cur["scale"]},
            "fastest": {"tag": best["tag"] if "tag" in best else best["cand"], "us": round(best["us"], 2)},
            "fastest_within_noise": {
                "tag": best_ok["cand"],
                "spec": to_spec(best_ok["cand"]),
                "us": round(best_ok["us"], 2),
                "gain_us": round(cur["us"] - best_ok["us"], 2),
                "rel_err": best_ok["rel_err"],
                "scale": best_ok["scale"],
                "bit_identical_to_current": best_ok.get("out_sha") == cur.get("out_sha"),
            },
            "hifi2_trials": {r["cand"]: round(r["us"], 2) for r in rr if r["fidelity"] != a.fidelity},
            "refused": sorted({r["cand"] for r in recs if r["role"] == role and "error" in r}),
            "candidates": len(base),
        }
        f = out[role]["fastest_within_noise"]
        print(
            f"{role:10s} current {cur_tag:14s} {cur['us']:8.2f} us -> {f['tag']:16s} {f['us']:8.2f} us "
            f"(gain {f['gain_us']:6.2f}, bit-identical {f['bit_identical_to_current']})"
        )
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
