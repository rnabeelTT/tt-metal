# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Minimal streaming chat client for the hybrid-MoE server: prints the answer and TTFT / decode tok/s.
usage: python serve/chat.py "your prompt" [max_tokens] [port]"""

import json
import os
import sys
import time
import urllib.request

prompt = sys.argv[1] if len(sys.argv) > 1 else "Explain in three sentences what a mixture-of-experts model is."
max_tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 128
port = sys.argv[3] if len(sys.argv) > 3 else "8000"
body = {
    "model": os.environ.get("MODEL", "minimax-m2.7-hybrid"),
    "messages": [{"role": "user", "content": prompt}],
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
t0 = time.time()
first = None
n = 0
usage = None
with urllib.request.urlopen(req, timeout=3600) as r:
    for line in r:
        line = line.decode().strip()
        if not line.startswith("data:") or line == "data: [DONE]":
            continue
        ev = json.loads(line[5:])
        if ev.get("usage"):
            usage = ev["usage"]
        for ch in ev.get("choices", []):
            piece = ch.get("delta", {}).get("content") or ""
            if piece:
                if first is None:
                    first = time.time()
                n += 1
                print(piece, end="", flush=True)
t1 = time.time()
toks = (usage or {}).get("completion_tokens", n)
print(
    f"\n\nTTFT {first - t0:.1f}s, decode {(toks - 1) / max(1e-6, t1 - first):.2f} tok/s, {toks} tokens, total {t1 - t0:.1f}s"
)
