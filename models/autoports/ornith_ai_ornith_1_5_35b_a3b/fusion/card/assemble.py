"""Writes the `card:` block of tt-model-fast.yaml from fusion/card/*.md and the card_bench.sh results.

    python assemble.py p300x2 p300 p150
"""

import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
YAML = HERE.parents[1] / "tt-model-fast.yaml"
TOOLS = HERE.parent / "tools"

DESCRIPTION = (
    "Ornith-1.5-35B-A3B (35B total / 3B active MoE: 256 experts, top-8 + shared expert, 40 layers of Gated DeltaNet + "
    "gated attention, vision tower, MTP head) on Blackhole through vLLM and the Tenstorrent plugin, with a fused decode "
    "step: the original bundle (raahemnabeel/ornith-1.5-35b-a3b) plus fused MoE, Gated DeltaNet, attention, layer-boundary "
    "and sampler decode ops for 1-32 concurrent users. Same weights, profiles and serving recipe; on four dies single-user "
    "decode 45 -> 220 tok/s and 32-user output 346 -> 591 tok/s, prefill unchanged or faster."
)

TEACHER = """
Teacher forcing against the Hugging Face bf16 model (original bundle; 100 greedy tokens, top-1 / top-5 / top-100):

| profile | code chat prompt | 2048-token book prompt |
| --- | --- | --- |
| 4 dies (bfp8 experts) | 0.97 / 1.00 / 1.00 | 0.89 / 0.98 / 1.00 |
| 2 dies (bfp8 experts) | 0.96 / 1.00 / 1.00 | 0.91 / 0.97 / 1.00 |
| 1 die (bfp4 experts) | 0.96 / 1.00 / 1.00 | 0.84 / 0.94 / 0.99 |
"""

RELATED = (
    "The original bundle this builds on: [raahemnabeel/ornith-1.5-35b-a3b](https://huggingface.co/raahemnabeel/ornith-1.5-35b-a3b). "
    "Engineering log with every measurement: `models/autoports/ornith_ai_ornith_1_5_35b_a3b/fusion/HANDOFF.md` on the "
    "`ornith_fusion` branch of github.com/rnabeelTT/tt-metal."
)

P150_NOTE = (
    "One die keeps the original decode path (the fused MoE kernel needs at most two up|gate columns per DRAM bank, i.e. two "
    "or more dies, and the replicated boundary needs a mesh): p150 decode is the original MoE plus the fused router top-k "
    "on 1-row steps, MTP on, and the tables below show parity. The single-user random-token latency table is left out: "
    "with adaptive MTP on random tokens it does not repeat on one die (15.9, 52.4 and 27.0 ms TPOT at 128/128 in three "
    "runs of the original); the original card's p150 latency rows still describe this bundle."
)

P300X2_NOTE = (
    "Four-die rows measured on the previous revision, which differs only in the sampler top-k's index width (16 -> 32 "
    "bits, for two dies). This revision, same setup: 1 user 4.57 ms TPOT, real text 217.4 tok/s, 8 users 9.71 ms, "
    "32 users 1024/256 24.2 ms."
)


def block(key, text):
    body = "\n".join(("    " + l) if l.strip() else "" for l in text.strip("\n").splitlines())
    return f"  {key}: |\n{body}\n"


def main():
    profiles = sys.argv[1:] or ["p300x2", "p300", "p150"]
    tables = subprocess.run(
        [sys.executable, str(TOOLS / "card_tables.py"), os.environ.get("NEW", "card_fast"), os.environ.get("PUB", "card_pub"), *profiles],
        check=True, capture_output=True, text=True, env={"PUB_LABEL": "original", "REALTEXT_DESC": "12 chat prompts x 384 output tokens, temperature 0", **os.environ,
             "SAME_DECODE": "p150", "SAME_DECODE_NOTE": P150_NOTE},
    ).stdout
    perf = (HERE / "performance_head.md").read_text().replace("### Performance", TEACHER.strip() + "\n\n### Performance")
    tables = tables.replace("#### p300x2\n", "#### p300x2\n\n" + P300X2_NOTE + "\n", 1)
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
