"""Writes the `card:` block of tt-model-fast.yaml from fusion/card/*.md and the card_bench.sh results.

    python assemble.py p300x2 p300 p150
"""

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
YAML = HERE.parents[1] / "tt-model-fast.yaml"
TOOLS = HERE.parent / "tools"

DESCRIPTION = (
    "Ornith-1.5-35B-A3B (35B total / 3B active MoE: 256 experts, top-8 + shared expert, 40 layers of Gated DeltaNet + "
    "gated attention, vision tower, MTP head) on Blackhole through vLLM and the Tenstorrent plugin, with a fused decode "
    "step: the v2 bundle (raahemnabeel/ornith-1.5-35b-a3b) plus a fused MoE decode for 1-32 concurrent users and a "
    "replicated-residual layer boundary. Same weights, profiles and serving recipe; 34-83 % more concurrent output and "
    "up to 44 % faster single-user decode on four dies, prefill unchanged."
)

TEACHER = """
Teacher forcing against the Hugging Face bf16 model (v2 numbers; 100 greedy tokens, top-1 / top-5 / top-100):

| profile | code chat prompt | 2048-token book prompt |
| --- | --- | --- |
| 4 dies (bfp8 experts) | 0.97 / 1.00 / 1.00 | 0.89 / 0.98 / 1.00 |
| 2 dies (bfp8 experts) | 0.96 / 1.00 / 1.00 | 0.91 / 0.97 / 1.00 |
| 1 die (bfp4 experts) | 0.96 / 1.00 / 1.00 | 0.84 / 0.94 / 0.99 |
"""

RELATED = (
    "The v2 bundle this builds on: [raahemnabeel/ornith-1.5-35b-a3b](https://huggingface.co/raahemnabeel/ornith-1.5-35b-a3b). "
    "Engineering log with every measurement: `models/autoports/ornith_ai_ornith_1_5_35b_a3b/fusion/HANDOFF.md` on the "
    "`ornith_fusion` branch of github.com/rnabeelTT/tt-metal."
)


def block(key, text):
    body = "\n".join(("    " + l) if l.strip() else "" for l in text.strip("\n").splitlines())
    return f"  {key}: |\n{body}\n"


def main():
    profiles = sys.argv[1:] or ["p300x2", "p300", "p150"]
    tables = subprocess.run(
        [sys.executable, str(TOOLS / "card_tables.py"), "card_new", "card_pub", *profiles],
        check=True, capture_output=True, text=True,
    ).stdout
    perf = (HERE / "performance_head.md").read_text().replace("### Performance", TEACHER.strip() + "\n\n### Performance")
    perf += "\n" + tables
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
