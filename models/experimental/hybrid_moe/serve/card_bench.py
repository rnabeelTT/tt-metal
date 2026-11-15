# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Model-card speed rows through the OpenAI endpoint (greedy, OSL 128): the 8 short held-out prompts (prompts_v2) RUNS
times, then for each long-prompt size (characters of this repo's source, as serve/elastic_test.py) CYCLES times: the
long prompt, the same prompt resent (prefix-cache hit) and one short request (an elastic profile switches back). Prints
one JSON line per request and a CARD row per kind: mean TTFT, prefill tok/s (prompt tokens / TTFT), decode tok/s
(tokens after the first / time after the first) and E2EL.
With --sampled the requests use MiniMax's recommended sampling (temperature 1.0, top-p 0.95, top-k 40; seed = request
index) instead of greedy.
usage: python3 serve/card_bench.py --port 20000 [--runs 4] [--long 160000] [--long 300000] [--cycles 2] [--sampled]"""

import argparse
import json
import pathlib
import statistics
import sys
import time
import urllib.request

here = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(here))
from prompts_v2 import HELDOUT  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=20000)
ap.add_argument("--runs", type=int, default=2)
ap.add_argument("--long", type=int, action="append", default=[])
ap.add_argument("--cycles", type=int, default=1)
ap.add_argument("--max-tokens", type=int, default=128)
ap.add_argument("--sampled", action="store_true")
a = ap.parse_args()
url = f"http://localhost:{a.port}/v1/chat/completions"
model = json.load(urllib.request.urlopen(f"http://localhost:{a.port}/v1/models"))["data"][0]["id"]
files = sorted((here / "hybrid").glob("*.py")) + sorted(here.glob("*.py"))
doc = "".join(p.read_text() for p in files)


N_ASKED = [0]


def ask(content):
    body = dict(model=model, messages=[{"role": "user", "content": content}], max_tokens=a.max_tokens, temperature=0)
    if a.sampled:
        body.update(temperature=1.0, top_p=0.95, top_k=40, seed=N_ASKED[0])
    N_ASKED[0] += 1
    body.update(stream=True, stream_options={"include_usage": True})
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0, ttft, usage = time.perf_counter(), None, None
    with urllib.request.urlopen(req, timeout=3600) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            ev = json.loads(line[6:])
            usage = ev.get("usage") or usage
            for ch in ev.get("choices", []):
                d = ch.get("delta", {})
                if ttft is None and (d.get("content") or d.get("reasoning_content") or d.get("reasoning")):
                    ttft = time.perf_counter() - t0
    e2el = time.perf_counter() - t0
    n = (usage or {}).get("completion_tokens", 0)
    return dict(
        isl=(usage or {}).get("prompt_tokens"),
        osl=n,
        ttft_s=round(ttft, 3),
        decode_tok_s=round((n - 1) / (e2el - ttft), 2) if n > 1 else 0.0,
        e2el_s=round(e2el, 3),
    )


rows = {}


def run(kind, content):
    r = ask(content)
    rows.setdefault(kind, []).append(r)
    print(json.dumps({"kind": kind, **r}), flush=True)


for _ in range(a.runs):
    for p in HELDOUT:
        run("short", p)
for chars in a.long:
    content = "Here is some Python code:\n\n" + doc[:chars] + "\n\nWhat does this code do? Answer briefly."
    for _ in range(a.cycles):
        run(f"long {chars}", content)
        run(f"long {chars} resent", content)
        run(f"short after long {chars}", HELDOUT[0])
for kind, rs in rows.items():
    m = lambda k: statistics.mean(r[k] for r in rs)  # noqa: E731
    isl = round(m("isl"))
    print(
        f"CARD {kind}: ISL {isl} N {len(rs)} TTFT {m('ttft_s'):.2f} s prefill {isl / m('ttft_s'):.0f} tok/s "
        f"decode {m('decode_tok_s'):.1f} tok/s E2EL {m('e2el_s'):.1f} s",
        flush=True,
    )
