# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Prefix-cache check for the served model: a cold long prompt, the same prompt resent, and a second chat turn on it.
Greedy (temperature 0); prints TTFT and the generated text per request and writes them to OUT (json).
usage: python3 serve/prefix_test.py [--port 8000] [--chars 30000] [--out runs/m27/prefix_test.json]"""

import argparse
import json
import pathlib
import time
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=8000)
ap.add_argument("--chars", type=int, default=30000)
ap.add_argument("--max-tokens", type=int, default=96)
ap.add_argument("--out", default="runs/m27/prefix_test.json")
a = ap.parse_args()
url = f"http://localhost:{a.port}/v1/chat/completions"
model = json.load(urllib.request.urlopen(f"http://localhost:{a.port}/v1/models"))["data"][0]["id"]

here = pathlib.Path(__file__).resolve().parent.parent
doc = "".join(p.read_text() for p in sorted((here / "hybrid").glob("*.py")))[: a.chars]


def ask(messages):
    body = dict(model=model, messages=messages, max_tokens=a.max_tokens, temperature=0, stream=True)
    body["stream_options"] = {"include_usage": True}
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0, ttft, text, reason, usage = time.perf_counter(), None, "", "", None
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
                text += d.get("content") or ""
                reason += d.get("reasoning_content") or d.get("reasoning") or ""
    return dict(ttft=ttft, total=time.perf_counter() - t0, usage=usage, content=text, reasoning=reason)


turn1 = [
    {"role": "user", "content": f"Here is some source code:\n\n{doc}\n\nSummarize what this code does in 3 bullets."}
]
res = {}
res["cold"] = ask(turn1)
res["resend"] = ask(turn1)
reply = res["cold"]["content"] or res["cold"]["reasoning"]
turn2 = turn1 + [
    {"role": "assistant", "content": reply},
    {"role": "user", "content": "Which function handles the expert streaming? Answer in one sentence."},
]
res["turn2"] = ask(turn2)
res["turn2_again"] = ask(turn2)
for k, v in res.items():
    u = v["usage"] or {}
    print(
        f"{k:12s} TTFT {v['ttft']:.2f}s total {v['total']:.2f}s prompt {u.get('prompt_tokens')} "
        f"cached {((u.get('prompt_tokens_details') or {}).get('cached_tokens'))}"
    )
print(
    "resend identical to cold:",
    (res["resend"]["content"], res["resend"]["reasoning"]) == (res["cold"]["content"], res["cold"]["reasoning"]),
)
print("turn2 identical to turn2_again:", res["turn2"]["content"] == res["turn2_again"]["content"])
pathlib.Path(a.out).write_text(json.dumps(res, indent=1))
print("cold reasoning/content:", repr((res["cold"]["reasoning"] + "|" + res["cold"]["content"])[:300]))
