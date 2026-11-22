# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Model-card benchmark against a running server (stdlib only, run on the host), the published ports' methodology:
random-token prompts of exactly the input length (token ids through /v1/completions), output pinned with ignore_eos,
greedy, streamed; `users` concurrent requests in a closed loop (each user runs `per_user` requests back to back);
TTFT = time to the first streamed token; TPOT = (E2EL - TTFT) / (output - 1); one unrecorded warm-up per input length.

usage: python3 tools/bench_card.py [port] [cases]     cases: "in:out:users,..." (default 128:128:1,1024:256:1,...)
prints markdown table rows: | in | out | users | N | TTFT ms | TPOT ms | decode tok/s/user | total out tok/s | E2EL s |
"""

import concurrent.futures as cf
import json
import random
import sys
import time
import urllib.request

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
DEFAULT = ("128:128:1,1024:256:1,4096:256:1,16384:256:1,128:128:2,128:128:4,128:128:8,128:128:16,128:128:32,"
           "1024:256:2,1024:256:4,1024:256:8,1024:256:16,1024:256:32")
CASES = [tuple(int(v) for v in c.split(":")) for c in (sys.argv[2] if len(sys.argv) > 2 else DEFAULT).split(",")]
PER_USER = 2
SALT = time.time_ns()
MODEL = "inkling-2-mini"


def one(n_in, n_out, seed):
    r = random.Random(seed)
    body = {"model": MODEL, "prompt": [r.randint(1000, 150000) for _ in range(n_in)], "max_tokens": n_out,
            "temperature": 0, "ignore_eos": True, "stream": True}
    req = urllib.request.Request(f"http://localhost:{PORT}/v1/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    first = None
    n = 0
    with urllib.request.urlopen(req, timeout=7200) as resp:
        for line in resp:
            line = line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            d = json.loads(line[6:])
            if d.get("choices") and d["choices"][0].get("text") is not None:
                if first is None:
                    first = time.time()
                n += 1
    t1 = time.time()
    return (first or t1) - t0, t1 - t0, n


def main():
    warmed = set()
    print("| Input tokens | Output tokens | Users | N | TTFT (ms) | TPOT (ms) | Decode tok/s/user | Total out tok/s | "
          "E2EL (s) |", flush=True)
    print("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |", flush=True)
    for n_in, n_out, users in CASES:
        if n_in not in warmed:
            one(n_in, 8, hash((SALT, n_in, "warm")))
            warmed.add(n_in)
        t0 = time.time()

        def user(u):  # every request its own prompt (no prefix-cache hits between requests, cases or runs)
            return [one(n_in, n_out, hash((SALT, n_in, n_out, users, u, i))) for i in range(PER_USER)]

        with cf.ThreadPoolExecutor(users) as ex:
            res = [x for xs in ex.map(user, range(users)) for x in xs]
        wall = time.time() - t0
        ttft = sum(r[0] for r in res) / len(res)
        e2el = sum(r[1] for r in res) / len(res)
        tpot = sum((r[1] - r[0]) / max(r[2] - 1, 1) for r in res) / len(res)
        tot = sum(r[2] for r in res) / wall
        print(f"| {n_in:,} | {n_out} | {users} | {len(res)} | {ttft * 1e3:,.0f} | {tpot * 1e3:.1f} | {1 / tpot:.1f} | "
              f"{tot:.1f} | {e2el:.2f} |", flush=True)


if __name__ == "__main__":
    main()
