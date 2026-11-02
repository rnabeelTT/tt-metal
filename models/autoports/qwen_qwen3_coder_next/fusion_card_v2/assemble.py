"""Writes the `card:` block of ../tt-model-fast-v2.yaml from this directory's *.md and the card_bench.sh results.

    python assemble.py p300x2 p300
"""

import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
YAML = HERE.parent / "tt-model-fast-v2.yaml"
TOOLS = HERE.parents[1] / "ornith_ai_ornith_1_5_35b_a3b" / "fusion" / "tools"

DESCRIPTION = (
    "Qwen3-Coder-Next (80B total / 3B active MoE: 512 experts, top-10 + shared expert, 48 layers of Gated DeltaNet + "
    "gated attention) on Blackhole through vLLM and the Tenstorrent plugin, with a fused decode step: "
    "raahemnabeel/qwen3-coder-next-blackhole-fast (fused MoE decode for 1-32 concurrent users, replicated-residual "
    "layer boundary) plus a fused Gated DeltaNet decode mixer (conv + gates + norms + recurrent state update + gated "
    "norm in 2 device programs instead of ~46 ops per layer) and attention decode ops (q / k norm + RoPE, gate and the "
    "gated output in 2 device programs instead of ~34 ops), ported from the Ornith-1.5-35B-A3B fusion. Same weights, "
    "profiles and serving recipe; prefill unchanged."
)

RELATED = (
    "The release this builds on: [raahemnabeel/qwen3-coder-next-blackhole-fast]"
    "(https://huggingface.co/raahemnabeel/qwen3-coder-next-blackhole-fast); the original bundle: "
    "[raahemnabeel/qwen3-coder-next](https://huggingface.co/raahemnabeel/qwen3-coder-next). "
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
        [sys.executable, str(TOOLS / "card_tables.py"), "card_cnv2", "card_cnnew", *profiles],
        check=True, capture_output=True, text=True,
        env={**os.environ, "REALTEXT_DESC": "12 chat prompts x 384 output tokens, temperature 0", "PUB_LABEL": "blackhole-fast", "NEW_LABEL": "blackhole-fast-v2"},
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
