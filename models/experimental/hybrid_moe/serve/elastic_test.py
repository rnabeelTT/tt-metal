# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Elastic-profile check of the served model (serve_m27.sh ELASTIC=1): short request (short profile), a long prompt
past the short pool (switch to the long profile), the same long prompt again (prefix-cache hit in the long profile), a
short request (switch back), and another short one. Greedy; prints TTFT, decode tok/s and the start of each answer,
and writes them to OUT (json). The server log's HYB_ELASTIC lines show the switches.
usage: python3 serve/elastic_test.py [--port 8000] [--chars 160000] [--out runs/m27/elastic_serve.json]"""

import argparse
import json
import pathlib
import time
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=8000)
ap.add_argument("--chars", type=int, default=160000)  # ~40K tokens of source code
ap.add_argument("--max-tokens", type=int, default=128)
ap.add_argument("--out", default="runs/m27/elastic_serve.json")
a = ap.parse_args()
url = f"http://localhost:{a.port}/v1/chat/completions"
model = json.load(urllib.request.urlopen(f"http://localhost:{a.port}/v1/models"))["data"][0]["id"]
here = pathlib.Path(__file__).resolve().parent.parent
files = sorted((here / "hybrid").glob("*.py")) + sorted(here.glob("*.py"))
doc = "".join(p.read_text() for p in files)[: a.chars]


def ask(content):
    body = dict(model=model, messages=[{"role": "user", "content": content}], max_tokens=a.max_tokens, temperature=0)
    body.update(stream=True, stream_options={"include_usage": True})
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0, ttft, text, usage = time.perf_counter(), None, "", None
    with urllib.request.urlopen(req, timeout=3600) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            ev = json.loads(line[6:])
            usage = ev.get("usage") or usage
            for ch in ev.get("choices", []):
                d = ch.get("delta", {})
                piece = (d.get("content") or "") + (d.get("reasoning_content") or d.get("reasoning") or "")
                if piece and ttft is None:
                    ttft = time.perf_counter() - t0
                text += piece
    total = time.perf_counter() - t0
    n = (usage or {}).get("completion_tokens", 0)
    tps = (n - 1) / (total - ttft) if ttft is not None and n > 1 and total > ttft else 0.0
    return dict(
        ttft_s=round(ttft or 0.0, 2),
        decode_tok_s=round(tps, 1),
        prompt_tokens=(usage or {}).get("prompt_tokens"),
        completion_tokens=n,
        text=text,
    )


steps = [
    ("short", "Write a Rust function that parses a CSV line respecting quoted fields."),
    ("long (switch to long)", "Here is some Python code:\n\n" + doc + "\n\nWhat does this code do? Answer briefly."),
    ("long again (prefix hit)", "Here is some Python code:\n\n" + doc + "\n\nWhat does this code do? Answer briefly."),
    ("short (switch back)", "Explain the difference between a mutex and a semaphore in two sentences."),
    ("short again", "Write a haiku about the sea."),
]
res = []
for name, content in steps:
    r = ask(content)
    r["step"] = name
    res.append(r)
    print(
        f"{name:26s} prompt {r['prompt_tokens']:>6} ttft {r['ttft_s']:6.2f} s  decode {r['decode_tok_s']:5.1f} tok/s  "
        f"{r['text'][:90]!r}",
        flush=True,
    )
json.dump(res, open(a.out, "w"), indent=1)
