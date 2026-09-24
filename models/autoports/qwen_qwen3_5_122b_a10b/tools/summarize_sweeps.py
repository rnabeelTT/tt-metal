# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Render the stage-3 sweep rows (doc/optimized_decoder/sweeps/*.jsonl) as markdown tables.

    python models/autoports/qwen_qwen3_5_122b_a10b/tools/summarize_sweeps.py > models/autoports/qwen_qwen3_5_122b_a10b/doc/optimized_decoder/sweeps/SUMMARY.md
"""
import glob
import json
import os
import sys

ROOT = os.path.join(os.path.dirname(__file__), "..", "doc", "optimized_decoder", "sweeps")
KIND = {0: "GDN (L0)", 3: "attention (L3)"}


def main():
    files = sorted(glob.glob(os.path.join(ROOT, "*.jsonl")))
    print("# Stage-3 sweep summary (real weights, one Blackhole die)\n")
    print(
        "PCC = prefill T=128 / decode vs the HF layer; prefill ms = warmed wall T=128; decode ms = traced, 3 replays (wall/3).\n"
    )
    for f in files:
        fam = os.path.basename(f)[:-6]
        rows = [json.loads(l) for l in open(f) if l.strip()]
        print(f"## family `{fam}`\n")
        print(
            "| candidate | layer | PCC prefill | PCC decode | prefill ms | decode ms | materialized (routed gu/down, shared, mixer, kv) | note |"
        )
        print("|---|---|---|---|---|---|---|---|")
        for r in rows:
            m = r.get("materialized", {})
            mat = " / ".join(
                str(x).replace("DataType.", "").replace("BFLOAT", "bf").replace("_B", "").lower()
                for x in (
                    m.get("up_gate"),
                    m.get("down"),
                    m.get("shared"),
                    m.get("attn_proj") or m.get("gdn_qkv"),
                    m.get("kv_cache"),
                )
                if x is not None
            )
            if "error" in r:
                err = r["error"].split("\n")[0][:110]
                print(f"| {r['name']} | {KIND.get(r['layer_idx'], r['layer_idx'])} | - | - | - | - | - | FAIL: {err} |")
            else:
                print(
                    f"| {r['name']} | {KIND.get(r['layer_idx'], r['layer_idx'])} | {r.get('pcc_prefill', float('nan')):.5f} | {r.get('pcc_decode', float('nan')):.5f} | "
                    f"{r.get('prefill_ms')} | {r.get('decode_ms')} | {mat} | {r.get('note', '')} |"
                )
        print()


if __name__ == "__main__":
    sys.exit(main())
