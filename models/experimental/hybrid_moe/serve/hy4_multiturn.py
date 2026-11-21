# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Multi-turn TTFT (Hy4 prefix reuse, tt/engine_batch.py): per conversation, turn 1 = a ~N-token document + question,
turn 2 = turn 1 + the reply + a follow-up (the server can continue the kept slot), turn 3 likewise; plus a repeated
system prompt case (same long system prompt, different user question). Prints TTFT per turn.

    python serve/hy4_multiturn.py [port] [--tokens 4000] [--convs 3] [--max-tokens 128]
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hy4_client import stream_chat  # noqa: E402

FOLLOW = ["Summarize the second paragraph in one sentence.", "Now list three key numbers mentioned above."]
QUESTIONS = ["What is the main topic?", "Which item number is mentioned first?", "Give a one-line title for it."]


def doc(n_tokens, seed):
    # ~10 tokens per sentence
    return " ".join(f"Item {seed}-{i}: the measured value was {(i * 37 + seed) % 1000} units in trial {i}."
                    for i in range(n_tokens // 16))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("port", nargs="?", default="8000")
    ap.add_argument("--tokens", type=int, default=4000)
    ap.add_argument("--convs", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--effort", default="no_think")
    a = ap.parse_args()
    rows = []
    for c in range(a.convs):
        msgs = [{"role": "user", "content": doc(a.tokens, c) + "\n\nWhat is the value in trial 7?"}]
        for t in range(3):
            r = stream_chat(a.port, msgs, a.max_tokens, 0.0, a.effort)
            print(f"TURN conv {c} turn {t + 1}: prompt {r['prompt_tokens']} tokens, TTFT {r['ttft'] * 1000:.0f} ms, "
                  f"{r['out_tokens']} out, {r['tok_s']:.1f} tok/s", flush=True)
            rows.append((t + 1, r["ttft"]))
            msgs = msgs + [{"role": "assistant", "content": r["text"]}]
            if t < 2:
                msgs.append({"role": "user", "content": FOLLOW[t]})
    sysp = doc(a.tokens, 99)
    for i, q in enumerate(QUESTIONS):
        r = stream_chat(a.port, [{"role": "system", "content": sysp}, {"role": "user", "content": q}], a.max_tokens, 0.0,
                        a.effort)
        print(f"SYS shared system prompt, question {i}: prompt {r['prompt_tokens']} tokens, TTFT {r['ttft'] * 1000:.0f} ms",
              flush=True)
        rows.append((10 + (i > 0), r["ttft"]))
    for k, name in ((1, "turn 1 (cold)"), (2, "turn 2"), (3, "turn 3"), (10, "system prompt first"),
                    (11, "system prompt repeat")):
        v = [x for kk, x in rows if kk == k]
        if v:
            print(f"SUMMARY {name}: mean TTFT {sum(v) / len(v) * 1000:.0f} ms over {len(v)}", flush=True)


if __name__ == "__main__":
    main()
