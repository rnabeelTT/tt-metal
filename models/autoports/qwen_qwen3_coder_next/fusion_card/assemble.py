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
    "gated attention) on Blackhole through vLLM and the Tenstorrent plugin, with a fused decode step: the v1 "
    "bundle (raahemnabeel/qwen3-coder-next-v1) plus fused MoE, Gated DeltaNet, attention, layer-boundary and sampler decode "
    "ops for 1-32 concurrent users, ported from the Ornith-1.5-35B-A3B fusion. Same weights, profiles and serving "
    "recipe; prefill unchanged."
)

RELATED = (
    "The v1 bundle this builds on: [raahemnabeel/qwen3-coder-next-v1](https://huggingface.co/raahemnabeel/qwen3-coder-next-v1). "
    "The fusion it ports: [raahemnabeel/ornith-1.5-35b-a3b]"
    "(https://huggingface.co/raahemnabeel/ornith-1.5-35b-a3b); engineering log "
    "`models/autoports/ornith_ai_ornith_1_5_35b_a3b/fusion/HANDOFF.md` on the `coder_next_fusion` branch of "
    "github.com/rnabeelTT/tt-metal."
)


def block(key, text):
    body = "\n".join(("    " + l) if l.strip() else "" for l in text.strip("\n").splitlines())
    return f"  {key}: |\n{body}\n"

P300X2_NOTE = (
    "Four-die rows measured on the previous revision, which differs only in the sampler top-k's index width (16 -> 32 "
    "bits, for two dies). This revision, same setup: 1 user 5.60 ms TPOT, real text 177.7 tok/s, 8 users 12.66 ms, "
    "32 users 1024/256 52.7 ms."
)


def main():
    profiles = sys.argv[1:] or ["p300x2", "p300"]
    tables = subprocess.run(
        [sys.executable, str(TOOLS / "card_tables.py"), os.environ.get("NEW", "card_cnfast4"), os.environ.get("PUB", "card_cnpub"), *profiles],
        check=True, capture_output=True, text=True,
        env={**os.environ, "REALTEXT_DESC": "12 chat prompts x 384 output tokens, temperature 0", "PUB_LABEL": "v1", "NEW_LABEL": "current"},
    ).stdout
    tables = tables.replace("#### p300x2\n", "#### p300x2\n\n" + P300X2_NOTE + "\n", 1)
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
