"""Writes the `card:` block of ../tt-model-fast-v2.yaml from this directory's *.md and the card_bench.sh results
(runs/card_ov2_<profile> vs the release-time measurement of blackhole-fast, runs/card_new_<profile>).

    python assemble.py p300x2 p300
"""

import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
YAML = HERE.parent / "tt-model-fast-v2.yaml"
TOOLS = HERE.parent / "fusion" / "tools"

DESCRIPTION = (
    "Ornith-1.5-35B-A3B (35B total / 3B active MoE: 256 experts, top-8 + shared expert, 40 layers of Gated DeltaNet + "
    "gated attention, vision tower, MTP head) on Blackhole through vLLM and the Tenstorrent plugin, with a fused decode "
    "step: raahemnabeel/ornith-1.5-35b-a3b-blackhole-fast (fused MoE decode for 1-32 concurrent users, replicated-"
    "residual layer boundary) plus a fused Gated DeltaNet decode mixer (2 device programs instead of ~46 ops per layer) "
    "and attention decode ops (2 device programs instead of ~34 ops), with MTP off on the multi-die profiles. Same "
    "weights, profiles and serving recipe; prefill unchanged."
)

RELATED = (
    "The release this builds on: [raahemnabeel/ornith-1.5-35b-a3b-blackhole-fast]"
    "(https://huggingface.co/raahemnabeel/ornith-1.5-35b-a3b-blackhole-fast); the v2 bundle: "
    "[raahemnabeel/ornith-1.5-35b-a3b](https://huggingface.co/raahemnabeel/ornith-1.5-35b-a3b). Engineering log with "
    "every measurement: `models/autoports/ornith_ai_ornith_1_5_35b_a3b/fusion/HANDOFF.md` on the `ornith_fusion` branch "
    "of github.com/rnabeelTT/tt-metal."
)


def block(key, text):
    body = "\n".join(("    " + l) if l.strip() else "" for l in text.strip("\n").splitlines())
    return f"  {key}: |\n{body}\n"


def main():
    profiles = sys.argv[1:] or ["p300x2", "p300"]
    tables = subprocess.run(
        [sys.executable, str(TOOLS / "card_tables.py"), "card_ov2", "card_new", *profiles],
        check=True, capture_output=True, text=True,
        env={**os.environ, "REALTEXT_DESC": "12 chat prompts x 384 output tokens, temperature 0",
             "PUB_LABEL": "blackhole-fast", "NEW_LABEL": "blackhole-fast-v2"},
    ).stdout
    perf = (HERE / "performance_head.md").read_text() + "\n" + tables
    src = YAML.read_text()
    head = src[: src.index("\ncard:\n") + 1]
    card = "card:\n" + f"  description: >\n    {DESCRIPTION}\n"
    card += block("quickstart", (HERE / "quickstart.md").read_text())
    card += block("performance", perf)
    card += block("limitations", (HERE / "limitations.md").read_text())
    card += f"  related: >\n    {RELATED}\n"
    YAML.write_text(head + card)
    print(f"wrote {YAML} card ({len(card)} chars, profiles {profiles})")


if __name__ == "__main__":
    main()
