# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Multi-turn chat through the prefix cache where every answer crosses a 2048-token boundary while decoding (an agent's
conversation: each turn resends the whole history plus the previous answer). Regression check for the prefix-state
cache: a snapshot planned on a decode row was never written, and the next turn's restore of it killed the engine.

usage: python serve/prefix_turns.py [port=8000] [first prompt tokens=13000] [turns=5]
Prints one line per turn (prompt tokens, TTFT, completion tokens) and ALL PASS / FAIL (a non-200 reply, a dead server,
or a turn whose answer did not cross the boundary it was sized for)."""
import glob
import json
import os
import sys
import time
import urllib.request

port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
first = int(sys.argv[2]) if len(sys.argv) > 2 else 13000
turns = int(sys.argv[3]) if len(sys.argv) > 3 else 5
url = f"http://127.0.0.1:{port}/v1"
SEG = 2048

with urllib.request.urlopen(f"{url}/models", timeout=60) as r:
    model = json.load(r)["data"][0]["id"]

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src = ""
for f in sorted(glob.glob(os.path.join(root, "hybrid", "*.py"))):
    src += open(f).read()
    if len(src) > first * 6:
        break


def chat(messages, max_tokens):
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "min_tokens": max_tokens,
            "temperature": 0, "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"reasoning_effort": "none"}}
    req = urllib.request.Request(f"{url}/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0, ttft, text, usage = time.time(), None, "", None
    with urllib.request.urlopen(req, timeout=1800) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            d = json.loads(line[5:])
            for c in d.get("choices", []):
                delta = c.get("delta", {})
                piece = (delta.get("content") or "") + (delta.get("reasoning_content") or "")
                if piece and ttft is None:
                    ttft = time.time() - t0
                text += delta.get("content") or ""
            usage = d.get("usage") or usage
    return text, ttft, usage


# size the first prompt: ~4 characters per token of source
messages = [{"role": "user", "content": "Summarise what this code does, file by file, in detail.\n\n" + src[: first * 4]}]
ok = True
for t in range(turns):
    # probe the prompt length with a 1-token request (also warms the prefix cache), then decode across the next boundary
    _, _, u = chat(messages, 1)
    p = u["prompt_tokens"]
    nxt = (p // SEG + 1) * SEG
    want = nxt - p + 48
    try:
        text, ttft, u = chat(messages, want)
    except Exception as e:  # a 500 or a dead server
        print(f"turn {t + 1}: prompt {p} tokens: FAIL {e!r}")
        ok = False
        break
    n = u["completion_tokens"]
    crossed = p + n > nxt
    ok &= crossed
    print(f"turn {t + 1}: prompt {p} tokens, TTFT {ttft:.2f} s, {n} tokens -> position {p + n} "
          f"({'crossed' if crossed else 'DID NOT cross'} {nxt})", flush=True)
    messages += [{"role": "assistant", "content": text}, {"role": "user", "content": "Go on with the next file."}]
print("ALL PASS" if ok else "FAIL")
