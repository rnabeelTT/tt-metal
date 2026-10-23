"""Writes the `card:` block of ../tt-model-fast.yaml from this directory's *.md and the card_bench.sh results.

    python assemble.py p300x2
"""

import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
YAML = HERE.parent / "tt-model-fast.yaml"
TOOLS = HERE.parents[1] / "ornith_ai_ornith_1_5_35b_a3b" / "fusion" / "tools"

DESCRIPTION = (
    "Qwen3.5-122B-A10B (122B total / 10B active MoE: 256 experts, top-8 + shared expert, 48 layers of Gated DeltaNet + "
    "gated attention, vision tower, MTP head) on four Blackhole dies through vLLM and the Tenstorrent plugin, with a "
    "fused decode step: the published bundle (raahemnabeel/qwen3-5-122b-a10b) plus a fused MoE decode for 1-32 "
    "concurrent users and a replicated-residual layer boundary, ported from the Ornith-1.5-35B-A3B fusion. Same weights, "
    "profiles and serving recipe; prefill unchanged."
)

RELATED = (
    "The bundle this builds on: [raahemnabeel/qwen3-5-122b-a10b](https://huggingface.co/raahemnabeel/qwen3-5-122b-a10b). "
    "The fusion it ports: [raahemnabeel/ornith-1.5-35b-a3b-blackhole-fast]"
    "(https://huggingface.co/raahemnabeel/ornith-1.5-35b-a3b-blackhole-fast); engineering log "
    "`models/autoports/ornith_ai_ornith_1_5_35b_a3b/fusion/HANDOFF.md` on the `q122_fusion` branch of "
    "github.com/rnabeelTT/tt-metal."
)


def block(key, text):
    body = "\n".join(("    " + l) if l.strip() else "" for l in text.strip("\n").splitlines())
    return f"  {key}: |\n{body}\n"


def main():
    profiles = sys.argv[1:] or ["p300x2"]
    tables = subprocess.run(
        [sys.executable, str(TOOLS / "card_tables.py"), "card_q122new", "card_q122pub", *profiles],
        check=True, capture_output=True, text=True, env={**os.environ, "PUB_LABEL": "published"},
    ).stdout
    head, sep, lc = tables.partition("#### longctx")
    lc = lc.replace("temperature 0, MTP on;", "temperature 0, MTP off;")
    tables = head + sep.replace("#### longctx", "#### p300x2-longctx (MTP off, chunked prefill at 4096 tokens)") + lc
    tables = tables.replace("| longctx |", "| p300x2-longctx |")
    perf = (HERE / "performance_head.md").read_text() + "\n" + tables
    assert "{" not in perf.split("### Performance")[0] and "{SPEED" not in (HERE / "quickstart.md").read_text(), "fill the placeholders"
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
