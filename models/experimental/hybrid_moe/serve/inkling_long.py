# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Long-context check against the running server: a needle (a passphrase) placed in the middle of a long document of
varied filler text, asked for at the end; prints prompt tokens, TTFT, decode tok/s and whether the answer has the
needle. usage: python serve/inkling_long.py [port=8000] [target prompt tokens, comma list; default 2000,8000,30000]"""
import json
import random
import sys
import time
import urllib.request

port = sys.argv[1] if len(sys.argv) > 1 else "8000"
targets = [int(v) for v in (sys.argv[2] if len(sys.argv) > 2 else "2000,8000,30000").split(",")]
url = f"http://127.0.0.1:{port}/v1/chat/completions"
with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=60) as r:
    model = json.load(r)["data"][0]["id"]
rng = random.Random(0)
TOPICS = ["the history of bridges", "how bees communicate", "the chemistry of bread", "tidal energy", "medieval maps",
          "the economics of shipping", "volcanic soils", "early computing", "glacier formation", "urban gardens"]


def filler(n_words):
    out = []
    while len(out) < n_words:
        t = rng.choice(TOPICS)
        s = (f"In a study of {t}, researchers noted that observation {rng.randint(1, 9999)} differed from the "
             f"expected value by {rng.randint(1, 99)} percent, which they attributed to seasonal variation and "
             f"measurement noise. ")
        out += s.split()
    return " ".join(out[:n_words])


ok_all = True
for tgt in targets:
    words = int(tgt * 0.72)
    needle = f"The secret passphrase is {rng.choice(['amber', 'cobalt', 'violet'])}-{rng.randint(1000, 9999)}."
    doc = filler(words // 2) + " " + needle + " " + filler(words - words // 2)
    q = doc + "\n\nWhat is the secret passphrase mentioned in the document above? Reply with the passphrase only."
    body = {"model": model, "messages": [{"role": "user", "content": q}], "max_tokens": 64, "temperature": 0,
            "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"reasoning_effort": "none"}}
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time()
    ttft, text, usage = None, "", None
    with urllib.request.urlopen(req, timeout=7200) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            ev = json.loads(line[5:])
            if ev.get("usage"):
                usage = ev["usage"]
            for ch in ev.get("choices", []):
                piece = ch.get("delta", {}).get("content") or ""
                if piece and ttft is None:
                    ttft = time.time() - t0
                text += piece
    dt = time.time() - t0
    n = usage["completion_tokens"] if usage else 0
    pt = usage["prompt_tokens"] if usage else 0
    dec = (n - 1) / (dt - ttft) if ttft and n > 1 and dt > ttft else 0.0
    key = needle.split("is ")[1].rstrip(".")
    ok = key in text
    ok_all &= ok
    print(f"prompt {pt} tokens: TTFT {ttft or 0:.2f}s ({pt / max(ttft or 1e-9, 1e-9):.0f} tok/s prefill), decode "
          f"{dec:.1f} tok/s, needle {'FOUND' if ok else 'MISSED'} ({text.strip()[:60]!r})", flush=True)
print("ALL FOUND" if ok_all else "SOME MISSED")
