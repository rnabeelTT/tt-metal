# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Render the stage-4 mesh sweep rows (doc/multichip_decoder/sweeps/*.jsonl) as markdown tables, with the delta vs the
`mc_selected` baseline row of the same layer.

    python models/autoports/qwen_qwen3_5_122b_a10b/tools/summarize_sweeps_mc.py > models/autoports/qwen_qwen3_5_122b_a10b/doc/multichip_decoder/sweeps/SUMMARY.md
"""
import glob
import json
import os

ROOT = os.path.join(os.path.dirname(__file__), "..", "doc", "multichip_decoder", "sweeps")
KIND = {0: "GDN (L0)", 3: "attention (L3)"}


def main():
    files = sorted(glob.glob(os.path.join(ROOT, "*.jsonl")))
    base = {}
    for f in files:
        for l in open(f):
            r = json.loads(l)
            if r.get("name") == "mc_selected" and "decode_ms" in r:
                base[r["layer_idx"]] = r
    print("# Stage-4 mesh sweep summary (real weights, (1,4) P300x2, `tools/sweep_multichip.py`)\n")
    print(
        "PCC = prefill T=128 / decode vs the HF layer; prefill ms = warmed wall T=128 (host bound, see README section 4); "
        "decode ms = traced, 3 replays (wall/3); delta vs the `mc_selected` row of the same layer (`baseline.jsonl`).\n"
    )
    for f in files:
        fam = os.path.basename(f)[:-6]
        rows = [json.loads(l) for l in open(f) if l.strip()]
        print(f"## family `{fam}`\n")
        print("| candidate | layer | PCC prefill | PCC decode | prefill ms | decode ms | decode delta | note |")
        print("|---|---|---|---|---|---|---|---|")
        for r in rows:
            k = KIND.get(r["layer_idx"], r["layer_idx"])
            if "error" in r:
                print(f"| `{r['name']}` | {k} | – | – | – | – | – | FAILED: {r['error'][:140].replace(chr(10), ' ')} |")
                continue
            b = base.get(r["layer_idx"])
            d = f"{1e3 * (r['decode_ms'] - b['decode_ms']):+.0f} us" if b else "–"
            print(
                f"| `{r['name']}` | {k} | {r['pcc_prefill']:.5f} | {r['pcc_decode']:.5f} | {r['prefill_ms']:.2f} | "
                f"{r['decode_ms']:.3f} | {d} | |"
            )
        print()


if __name__ == "__main__":
    main()
