"""Greedy (temperature 0) outputs of 8 fixed prompts against the server on :20000, sent one at a time (single-user path:
MTP) and all at once (8-wide plain decode). Writes runs/<label>/greedy.json; `compare A B` reports, per prompt, the
first token position where two runs' outputs differ.

    python greedy.py run <label>
    python greedy.py compare <labelA> <labelB>
"""

import json
import os
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

RUNS = Path(os.environ.get("RUNS", Path.home() / "ornith-fusion" / "runs"))
PROMPTS = [
    "Explain how a hash map handles collisions, with a short Python example.",
    "Write a Python function that returns the n-th Fibonacci number using memoization, then explain its complexity.",
    "Summarize the causes of the French Revolution in five bullet points.",
    "A train leaves at 9:40 and arrives at 13:15. How long is the trip? Show your reasoning step by step.",
    "Translate into French: 'The weather is lovely today, so we will walk to the market after lunch.'",
    "Write a haiku about distributed systems, then explain each line.",
    "What is the difference between TCP and UDP? Give two use cases for each.",
    "Write a SQL query that finds the top 3 customers by total order value, given tables customers(id, name) and orders(id, customer_id, amount).",
]


def ask(p):
    body = {
        "model": "ornith-ai/Ornith-1.5-35B-A3B",
        "messages": [{"role": "user", "content": p}],
        "max_tokens": 256,
        "temperature": 0,
        "ignore_eos": True,
    }
    req = urllib.request.Request(
        "http://localhost:20000/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=1800) as r:
        return json.load(r)["choices"][0]["message"]["content"]


def run(label):
    seq = [ask(p) for p in PROMPTS]
    with ThreadPoolExecutor(len(PROMPTS)) as ex:
        par = list(ex.map(ask, PROMPTS))
    out = RUNS / label
    out.mkdir(parents=True, exist_ok=True)
    (out / "greedy.json").write_text(json.dumps({"sequential": seq, "concurrent": par}, indent=1))
    print(f"wrote {out / 'greedy.json'}")


def first_diff(a, b):
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return None if len(a) == len(b) else n


def compare(la, lb):
    A = json.loads((RUNS / la / "greedy.json").read_text())
    B = json.loads((RUNS / lb / "greedy.json").read_text())
    for mode in ("sequential", "concurrent"):
        same = 0
        for i, (x, y) in enumerate(zip(A[mode], B[mode])):
            d = first_diff(x, y)
            same += d is None
            print(f"{mode:10s} prompt {i}: {'identical' if d is None else f'first difference at char {d} of {len(x)}'}")
        print(f"{mode}: {same}/{len(PROMPTS)} identical")


if __name__ == "__main__":
    {"run": lambda: run(sys.argv[2]), "compare": lambda: compare(sys.argv[2], sys.argv[3])}[sys.argv[1]]()
