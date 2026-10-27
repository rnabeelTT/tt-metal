"""Writes the `card:` block of ../tt-model-ornith122.yaml from this directory's *.md and the card_bench.sh runs
(absolute numbers: there is no earlier Ornith-1.5-122B bundle; the Qwen3.5-122B-A10B bundle on the same code is shown
as a reference column).

    python assemble.py            (RUNS: card_o122_p300x2, card_o122_longctx, card_q122fast_p300x2)
"""

import json
import os
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
YAML = HERE.parent / "tt-model-ornith122.yaml"
RUNS = Path(os.environ.get("RUNS", Path.home() / "ornith-fusion" / "runs"))

DESCRIPTION = (
    "ornith-ai/Ornith-1.5-122B (122B total / 10B active MoE: 256 experts, top-8 + shared expert, 48 layers of Gated "
    "DeltaNet + gated attention, vision tower) on four Blackhole dies through vLLM and the Tenstorrent plugin, served by "
    "the fused-decode code of the Qwen3.5-122B-A10B Blackhole bundle (same architecture): about 132 tok/s for one user "
    "on chat text, 21 tok/s each for 32 concurrent users, 262144-token context, text + images, tool calls, thinking on."
)

RELATED = (
    "Same code, Qwen checkpoint: [raahemnabeel/qwen3-5-122b-a10b](https://huggingface.co/raahemnabeel/qwen3-5-122b-a10b). "
    "Smaller sibling: [raahemnabeel/ornith-1.5-35b-a3b](https://huggingface.co/raahemnabeel/ornith-1.5-35b-a3b). "
    "Engineering log: `models/autoports/ornith_ai_ornith_1_5_35b_a3b/fusion/HANDOFF.md` on the `ornith122_fusion` "
    "branch of github.com/rnabeelTT/tt-metal."
)


def load(label):
    out = {}
    for f in (RUNS / label).glob("bench_*.json"):
        isl, osl, u = map(int, re.match(r"bench_(\d+)_(\d+)_u(\d+)\.json", f.name).groups())
        d = json.loads(f.read_text())
        out[(isl, osl, u)] = (d["median_ttft_ms"], d["median_tpot_ms"], d["output_throughput"])
    rt = RUNS / label / "realtext.txt"
    m = re.search(r"median ([\d.]+)", rt.read_text()) if rt.exists() else None
    return out, (float(m.group(1)) if m else None)


def tables():
    new, real = load("card_o122_p300x2")
    ref, ref_real = load("card_q122fast_p300x2")
    lc, lc_real = load("card_o122_longctx")
    s = ["#### p300x2 (default profile)", ""]
    s.append(f"Real text, single user (12 chat prompts x 384 output tokens, temperature 0, median per prompt): "
             f"**{real:.1f} tok/s** (Qwen3.5-122B-A10B on the same code: {ref_real:.1f}).")
    s += ["", "Latency, batch 1 (single user)", "", "| input tokens | TTFT ms | TPOT ms | decode tok/s | Qwen3.5-122B TPOT ms |",
          "| ---: | ---: | ---: | ---: | ---: |"]
    for k in sorted(k for k in new if k[2] == 1):
        ttft, tpot, _ = new[k]
        r = ref.get(k)
        s.append(f"| {k[0]} | {ttft:.0f} | {tpot:.2f} | {1e3 / tpot:.1f} | {r[1]:.2f} |" if r else
                 f"| {k[0]} | {ttft:.0f} | {tpot:.2f} | {1e3 / tpot:.1f} | - |")
    for isl, osl in ((128, 128), (1024, 256)):
        s += ["", f"Concurrency, {isl} input / {osl} output tokens", "",
              "| users | TPOT ms | tok/s per user | total output tok/s | TTFT ms | Qwen3.5-122B TPOT ms |",
              "| ---: | ---: | ---: | ---: | ---: | ---: |"]
        for k in sorted(k for k in new if k[:2] == (isl, osl)):
            ttft, tpot, out = new[k]
            r = ref.get(k)
            s.append(f"| {k[2]} | {tpot:.2f} | {1e3 / tpot:.1f} | {out:.1f} | {ttft:.0f} | {r[1]:.2f} |" if r else
                     f"| {k[2]} | {tpot:.2f} | {1e3 / tpot:.1f} | {out:.1f} | {ttft:.0f} | - |")
    s += ["", "#### p300x2-longctx (chunked prefill at 4096 tokens)", "",
          f"Real text, single user: **{lc_real:.1f} tok/s**.", "", "| input tokens | TTFT ms | TPOT ms |", "| ---: | ---: | ---: |"]
    for k in sorted(k for k in lc if k[2] == 1):
        s.append(f"| {k[0]} | {lc[k][0]:.0f} | {lc[k][1]:.2f} |")
    return "\n".join(s) + "\n"


def block(key, text):
    body = "\n".join(("    " + line) if line.strip() else "" for line in text.strip("\n").splitlines())
    return f"  {key}: |\n{body}\n"


def main():
    perf = (HERE / "performance_head.md").read_text() + "\n" + tables()
    src = YAML.read_text()
    head = src[: src.index("\ncard:\n") + 1]
    card = "card:\n" + f"  description: >\n    {DESCRIPTION}\n"
    card += block("quickstart", (HERE / "quickstart.md").read_text())
    card += block("performance", perf)
    card += block("limitations", (HERE / "limitations.md").read_text())
    card += f"  related: >\n    {RELATED}\n"
    YAML.write_text(head + card)
    print(f"wrote {YAML} card ({len(card)} chars)")


if __name__ == "__main__":
    main()
