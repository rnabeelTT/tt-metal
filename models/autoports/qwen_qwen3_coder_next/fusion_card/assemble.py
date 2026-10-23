"""Writes the `card:` block of ../tt-model-fast.yaml from this directory's *.md and the card_bench.sh results.

    python assemble.py p300x2 p300
"""

import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
YAML = HERE.parent / "tt-model-fast.yaml"
TOOLS = HERE.parents[1] / "ornith_ai_ornith_1_5_35b_a3b" / "fusion" / "tools"

DESCRIPTION = (
    "Qwen3-Coder-Next (80B total / 3B active MoE: 512 experts, top-10 + shared expert, 48 layers of Gated DeltaNet + "
    "gated attention) on Blackhole through vLLM and the Tenstorrent plugin, with a fused decode step: the published "
    "bundle (raahemnabeel/qwen3-coder-next) plus a fused MoE decode for 1-32 concurrent users and a replicated-residual "
    "layer boundary, ported from the Ornith-1.5-35B-A3B fusion. Same weights, profiles and serving recipe; prefill "
    "unchanged."
)

RELATED = (
    "The bundle this builds on: [raahemnabeel/qwen3-coder-next](https://huggingface.co/raahemnabeel/qwen3-coder-next). "
    "The fusion it ports: [raahemnabeel/ornith-1.5-35b-a3b-blackhole-fast]"
    "(https://huggingface.co/raahemnabeel/ornith-1.5-35b-a3b-blackhole-fast); engineering log "
    "`models/autoports/ornith_ai_ornith_1_5_35b_a3b/fusion/HANDOFF.md` on the `coder_next_fusion` branch of "
    "github.com/rnabeelTT/tt-metal."
)


def block(key, text):
    body = "\n".join(("    " + l) if l.strip() else "" for l in text.strip("\n").splitlines())
    return f"  {key}: |\n{body}\n"


def main():
    profiles = sys.argv[1:] or ["p300x2", "p300"]
    tables = subprocess.run(
        [sys.executable, str(TOOLS / "card_tables.py"), "card_cnnew", "card_cnpub", *profiles],
        check=True, capture_output=True, text=True,
        env={**os.environ, "REALTEXT_DESC": "12 chat prompts x 384 output tokens, temperature 0"},
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
