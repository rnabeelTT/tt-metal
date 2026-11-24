# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Serving benchmark against a running Inkling-2-Mini vLLM server (stdlib only, run on the host): for each concurrency
C, C streamed chat requests with a real-text prompt, max_tokens OUT (ignore_eos), temperature T: TTFT (mean / max),
per-user decode tok/s (after the first token) and total output tok/s over the wall time.

usage: python3 tools/bench_serve.py [port=8000] [concurrencies=1,2,4,8] [out=256] [temperature=0]
"""

import concurrent.futures as cf
import json
import sys
import time
import urllib.request

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
CONCS = [int(c) for c in (sys.argv[2] if len(sys.argv) > 2 else "1,2,4,8").split(",")]
OUT = int(sys.argv[3]) if len(sys.argv) > 3 else 256
TEMP = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
PROMPT = ("Explain, step by step and in plain language, how a refrigerator moves heat from its inside to the room, "
          "covering the compressor, condenser, expansion valve and evaporator, and why the refrigerant's boiling point "
          "matters. Then give two practical tips for keeping one efficient. Request {i}.")


def one(i):
    body = {"model": "inkling-2-mini", "messages": [{"role": "user", "content": PROMPT.format(i=i)}],
            "max_tokens": OUT, "temperature": TEMP, "ignore_eos": True, "stream": True,
            "stream_options": {"include_usage": True}, "chat_template_kwargs": {"reasoning_effort": "none"}}
    req = urllib.request.Request(f"http://localhost:{PORT}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    first, n = None, 0
    with urllib.request.urlopen(req, timeout=1800) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            d = json.loads(line[6:])
            if d.get("usage"):
                n = d["usage"]["completion_tokens"]
            if first is None and d.get("choices") and (d["choices"][0].get("delta") or {}).get("content") is not None:
                first = time.time()
    t1 = time.time()
    first = first or t1
    return first - t0, t1 - first, n, t0, t1


def main():
    for C in CONCS:
        t0 = time.time()
        with cf.ThreadPoolExecutor(C) as ex:
            res = list(ex.map(one, range(C)))
        wall = time.time() - t0
        ttft = [r[0] for r in res]
        dec = [(r[2] - 1) / r[1] for r in res if r[1] > 0 and r[2] > 1]
        tot = sum(r[2] for r in res)
        print(f"C={C}: TTFT mean {sum(ttft) / C:.2f}s max {max(ttft):.2f}s | decode per user {sum(dec) / len(dec):.1f} "
              f"tok/s | total {tot / wall:.1f} tok/s ({tot} tokens, {wall:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
