# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Serving benchmark: the held-out prompts (prompts_v2.HELDOUT) through the chat API, greedy, N tokens each; prints
per-prompt TTFT / decode tok/s and the mean. usage: python serve/bench.py [max_tokens] [port] [label]"""
import json
import os
import statistics
import sys
import time
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from prompts_v2 import HELDOUT  # noqa: E402

max_tokens = int(sys.argv[1]) if len(sys.argv) > 1 else 128
port = sys.argv[2] if len(sys.argv) > 2 else "8000"
label = sys.argv[3] if len(sys.argv) > 3 else ""
with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=60) as r:
    model = json.load(r)["data"][0]["id"]  # the served model name
rows = []
for p in HELDOUT:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": p}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0, first, usage = time.time(), None, None
    with urllib.request.urlopen(req, timeout=3600) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            ev = json.loads(line[5:])
            usage = ev.get("usage") or usage
            d = [c.get("delta") or {} for c in ev.get("choices", [])]
            # with a reasoning parser the first tokens arrive as reasoning_content (or reasoning), not content
            if first is None and any(x.get("content") or x.get("reasoning_content") or x.get("reasoning") for x in d):
                first = time.time()
    t1 = time.time()
    n = (usage or {}).get("completion_tokens", 0)
    rows.append(
        {
            "prompt": p[:40],
            "ttft_s": round(first - t0, 2),
            "tok_s": round((n - 1) / max(1e-6, t1 - first), 2),
            "tokens": n,
        }
    )
    print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
print(
    f"BENCH {label} mean decode {statistics.mean(r['tok_s'] for r in rows):.2f} tok/s, "
    f"mean TTFT {statistics.mean(r['ttft_s'] for r in rows):.2f} s",
    flush=True,
)
