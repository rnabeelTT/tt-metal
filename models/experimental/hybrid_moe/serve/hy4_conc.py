# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Concurrent-user benchmark: for each N in --users, N streaming chats at once (greedy, ignore_eos-free prompts that
answer at length); per request TTFT and decode tok/s, total tok/s over the window where all N decode.

    python serve/hy4_conc.py [port] --users 1,2,4,8 --max-tokens 256
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hy4_client import PROMPTS, stream_chat  # noqa: E402

LONG_ANSWER = [
    "Write a detailed, multi-paragraph explanation of how a CPU cache hierarchy works.",
    "Describe the history of the printing press in several paragraphs.",
    "Write a long story about a lighthouse keeper who finds a message in a bottle.",
    "Explain step by step how to implement a hash map in C, with code.",
    "详细介绍中国古代四大发明及其影响，写几段话。",
    "Give a thorough comparison of TCP and UDP with examples of where each is used.",
    "Write a detailed guide to growing tomatoes at home, from seed to harvest.",
    "Explain the theory of evolution by natural selection in detail for a high-school student.",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("port", nargs="?", default="8000")
    ap.add_argument("--users", default="1,2,4,8")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--effort", default="no_think")
    args = ap.parse_args()
    stream_chat(args.port, [{"role": "user", "content": PROMPTS[0]}], 8, args.temp, args.effort)  # warm the path
    for n in [int(x) for x in args.users.split(",")]:
        res = [None] * n

        def run(i):
            t0 = time.time()
            r = stream_chat(args.port, [{"role": "user", "content": LONG_ANSWER[i % len(LONG_ANSWER)]}],
                            args.max_tokens, args.temp, args.effort)
            r["t0"], r["t1"] = t0, time.time()
            r["t_first"] = t0 + r["ttft"]
            res[i] = r

        th = [threading.Thread(target=run, args=(i,)) for i in range(n)]
        w0 = time.time()
        for t in th:
            t.start()
        for t in th:
            t.join()
        wall = time.time() - w0
        toks = sum(r["out_tokens"] for r in res)
        # the window where every request decodes: after the last first token, before the first finish
        a, b = max(r["t_first"] for r in res), min(r["t1"] for r in res)
        per = [r["tok_s"] for r in res]
        ttft = sorted(r["ttft"] for r in res)
        print(f"USERS {n}: total {toks} tokens in {wall:.1f} s -> {toks / wall:.1f} tok/s overall; per-user decode "
              f"{min(per):.1f}-{max(per):.1f} (mean {sum(per) / n:.1f}) -> {sum(per):.1f} tok/s aggregate; "
              f"TTFT min/median/max {ttft[0]:.2f}/{ttft[n // 2]:.2f}/{ttft[-1]:.2f} s; all-decoding window "
              f"{max(0.0, b - a):.1f} s", flush=True)
        for i, r in enumerate(res):
            print(f"  user {i}: {r['out_tokens']} tokens, TTFT {r['ttft']:.2f} s, {r['tok_s']:.1f} tok/s: "
                  f"{(r['text'] or r['reasoning'])[:80]!r}", flush=True)


if __name__ == "__main__":
    main()
