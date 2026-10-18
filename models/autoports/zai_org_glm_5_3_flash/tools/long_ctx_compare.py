# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Compare two tools/long_ctx_check.py records (a = baseline, b = candidate) on the same document positions:
top-1 agreement, mean top-5 overlap (|top5_a & top5_b| / 5), and each side's true-next-token top-1 / top-5 rate.

usage: python -m models.autoports.zai_org_glm_5_3_flash.tools.long_ctx_compare a.json b.json [--out cmp.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def compare(a: dict, b: dict) -> dict:
    out = {}
    for L, ra in a["runs"].items():
        rb = b["runs"].get(L)
        if rb is None:
            continue
        assert ra["true_ids"] == rb["true_ids"], f"L={L}: different document positions"
        n = len(ra["true_ids"])
        t1 = sum(x[0] == y[0] for x, y in zip(ra["top5_ids"], rb["top5_ids"])) / n
        ov = sum(len(set(x) & set(y)) / 5 for x, y in zip(ra["top5_ids"], rb["top5_ids"])) / n
        out[L] = {
            "positions": n,
            "prefill_tokens": ra["prefill_tokens"],
            "top1_agreement": round(t1, 4),
            "top5_overlap": round(ov, 4),
            "a_true_top1": ra["true_top1"],
            "a_true_top5": ra["true_top5"],
            "b_true_top1": rb["true_top1"],
            "b_true_top5": rb["true_top5"],
            "a_prefill_s": round(ra["prefill_s"], 1),
            "b_prefill_s": round(rb["prefill_s"], 1),
            "finite": ra["finite"] and rb["finite"],
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--out", default=None)
    x = ap.parse_args()
    res = compare(json.loads(Path(x.a).read_text()), json.loads(Path(x.b).read_text()))
    print(json.dumps(res, indent=1))
    if x.out:
        Path(x.out).write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
