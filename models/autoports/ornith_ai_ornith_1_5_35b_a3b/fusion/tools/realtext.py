"""Single-user decode speed on real text: the greedy prompts (plus long-form ones), one at a time, streamed at
temperature 0 with a fixed max_tokens; decode tok/s = (tokens - 1) / (last - first token time). MTP's speed-up depends
on how predictable the text is, which random-token benchmarks understate.

    URL=http://localhost:20000 python realtext.py [max_tokens]
SAMPLE=temperature,top_k,top_p (e.g. 1.0,20,0.95): sampled instead of greedy.
CONC=U: U streams at once (rounds of U prompts, the list cycled, at least 2 rounds covering it): per-user decode tok/s
(median over streams) and aggregate decode tok/s (a round's tokens over its first-to-last token span, mean).
"""

import json
import os
import statistics
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(__file__))
from greedy import PROMPTS  # noqa: E402

EXTRA = [
    "Write a Python class implementing an LRU cache with get and put in O(1), with docstrings and a short test.",
    "Explain step by step how to solve the equation 3x^2 - 12x + 9 = 0, then verify the roots.",
    "Write a detailed product description for a stainless steel water bottle.",
    "Implement quicksort in C and explain the partition scheme.",
]
URL = os.environ.get("URL", "http://localhost:20000")


def served_model():
    """MODEL_ID, else the one model the server lists."""
    if os.environ.get("MODEL_ID"):
        return os.environ["MODEL_ID"]
    with urllib.request.urlopen(URL + "/v1/models", timeout=30) as r:
        return json.load(r)["data"][0]["id"]


def run(prompt, max_tokens):
    body = {
        "model": served_model(),
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "ignore_eos": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if os.environ.get("SAMPLE"):
        t, k, p = os.environ["SAMPLE"].split(",")
        body.update(temperature=float(t), top_k=int(k), top_p=float(p))
    req = urllib.request.Request(URL + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    times, t0, n = [], time.perf_counter(), 0
    with urllib.request.urlopen(req, timeout=600) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:") or line.endswith("[DONE]"):
                continue
            d = json.loads(line[5:])
            if d.get("usage"):
                n = d["usage"]["completion_tokens"]
            if d["choices"] and d["choices"][0]["delta"].get("content"):
                times.append(time.perf_counter())
    return times[0] - t0, times, n


def concurrent(mt, u):
    prompts = PROMPTS + EXTRA
    rounds = max(2, -(-len(prompts) // u))
    per, agg = [], []
    with ThreadPoolExecutor(u) as ex:
        for r in range(rounds):
            batch = [prompts[(r * u + i) % len(prompts)] for i in range(u)]
            res = list(ex.map(lambda p: run(p, mt), batch))
            first = min(ts[0] for _, ts, _ in res)
            last = max(ts[-1] for _, ts, _ in res)
            agg.append(sum(n - 1 for _, _, n in res) / (last - first))
            per += [(n - 1) / (ts[-1] - ts[0]) for _, ts, n in res]
    print(f"REALTEXT CONC={u} {rounds} rounds x {mt} tokens: per user median {statistics.median(per):.1f} tok/s "
          f"({min(per):.1f}..{max(per):.1f}), aggregate {statistics.mean(agg):.1f} tok/s")


def main():
    mt = int(sys.argv[1]) if len(sys.argv) > 1 else 384
    if int(os.environ.get("CONC", "1")) > 1:
        return concurrent(mt, int(os.environ["CONC"]))
    toks = secs = 0.0
    per = []
    for p in PROMPTS + EXTRA:
        ttft, ts, n = run(p, mt)
        toks += n - 1
        secs += ts[-1] - ts[0]
        per.append((n - 1) / (ts[-1] - ts[0]))
    tag = f" SAMPLE={os.environ['SAMPLE']}" if os.environ.get("SAMPLE") else ""
    print(f"REALTEXT{tag} {len(per)} prompts x {mt} tokens: decode {toks / secs:.1f} tok/s ({1e3 * secs / toks:.2f} ms/token), "
          f"per prompt {min(per):.1f}..{max(per):.1f} tok/s, median {statistics.median(per):.1f}")


if __name__ == "__main__":
    main()
