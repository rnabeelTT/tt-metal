# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Long-context sanity check against a running server: a passphrase hidden at depth D (0.1 / 0.5 / 0.9) of ~N tokens of
filler prose, then asked for. Prints prompt tokens, TTFT-ish wall time and whether the answer contains it.

usage: python3 tools/needle_serve.py [port=8000] [lengths=16000,32000,60000]
"""

import json
import random
import sys
import time
import urllib.request

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
LENS = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "16000,32000,60000").split(",")]
WORDS = ("river mountain lantern harbor meadow quiet morning archive copper violet garden station window paper "
         "orchard signal winter coastal market thread canvas ember silver forest bridge timber").split()


def filler(n_tokens, seed):
    r = random.Random(seed)
    out, n = [], 0
    while n < n_tokens:
        s = " ".join(r.choice(WORDS) for _ in range(r.randint(8, 16)))
        out.append(s.capitalize() + ".")
        n += len(s.split()) * 1.15 + 1
    return out


def ask(prompt):
    body = {"model": "inkling-2-mini", "messages": [{"role": "user", "content": prompt}], "max_tokens": 400,
            "temperature": 0, "chat_template_kwargs": {"reasoning_effort": "none"}}
    req = urllib.request.Request(f"http://localhost:{PORT}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    o = json.loads(urllib.request.urlopen(req, timeout=3600).read())
    return o, time.time() - t0


def main():
    for N in LENS:
        for depth in (0.1, 0.5, 0.9):
            key = f"{random.Random(N + int(depth * 10)).randint(1000, 9999)}-{random.Random(N).choice(WORDS)}"
            sents = filler(N, N)
            sents.insert(int(len(sents) * depth), f"Remember this: the secret passphrase is {key}.")
            prompt = " ".join(sents) + "\n\nWhat is the secret passphrase mentioned in the text above? Reply with it only."
            o, dt = ask(prompt)
            msg = o["choices"][0]["message"]
            ans = (msg.get("content") or "") + " | reasoning: " + (msg.get("reasoning_content") or msg.get("reasoning") or "")[-200:]
            ok = key in ans
            print(f"N~{N} depth {depth}: prompt {o['usage']['prompt_tokens']} tokens, {dt:.1f}s, "
                  f"{'OK' if ok else 'MISS'}: {ans.strip()[:300]!r}", flush=True)


if __name__ == "__main__":
    main()
