# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Inkling server smoke test: the dev harness's prompts (tools/spec_bench.py PROMPTS) through the chat API, greedy,
reasoning effort none, streamed: prints each reply, TTFT and decode tok/s; with a spec_bench SPEC_OUT json (k1 texts)
also whether each reply equals the harness's text over the shared length.

usage: python serve/inkling_smoke.py [max_tokens=96] [port=8000] [spec_out.json]"""
import json
import sys
import time
import urllib.request

PROMPTS = [
    "Write a Python function that returns the n-th Fibonacci number iteratively, with a docstring and two doctests.",
    "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did "
    "Natalia sell altogether in April and May? Explain step by step.",
    "Explain in three short paragraphs how a hash map handles collisions.",
]
max_tokens = int(sys.argv[1]) if len(sys.argv) > 1 else 96
port = sys.argv[2] if len(sys.argv) > 2 else "8000"
ref = json.load(open(sys.argv[3]))["texts"] if len(sys.argv) > 3 else None
url = f"http://127.0.0.1:{port}/v1/chat/completions"
with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=60) as r:
    model = json.load(r)["data"][0]["id"]
tot_tok, tot_dec = 0, 0.0
for i, p in enumerate(PROMPTS):
    body = {"model": model, "messages": [{"role": "user", "content": p}], "max_tokens": max_tokens, "temperature": 0,
            "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"reasoning_effort": "none"}}
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time()
    ttft, text, reason, usage = None, "", "", None
    with urllib.request.urlopen(req, timeout=3600) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            ev = json.loads(line[5:])
            if ev.get("usage"):
                usage = ev["usage"]
            for ch in ev.get("choices", []):
                d = ch.get("delta", {})
                piece = (d.get("content") or "") + (d.get("reasoning_content") or d.get("reasoning") or "")
                if piece and ttft is None:
                    ttft = time.time() - t0
                text += d.get("content") or ""
                reason += d.get("reasoning_content") or d.get("reasoning") or ""
    dt = time.time() - t0
    n = usage["completion_tokens"] if usage else 0
    dec = (n - 1) / (dt - ttft) if ttft and n > 1 and dt > ttft else 0.0
    tot_tok += n
    tot_dec += dec
    print(f"--- prompt {i}: {usage['prompt_tokens'] if usage else '?'} prompt tokens, {n} generated, TTFT {(ttft or 0):.2f}s, "
          f"decode {dec:.2f} tok/s")
    if reason:
        print(f"[reasoning] {reason[:300]!r}")
    print(repr(text[:400]))
    if ref is not None:
        r0 = ref.get("k1", ref.get("k0"))[i]
        m = min(len(r0), len(text))
        print(f"    matches the harness text over {m} chars: {r0[:m] == text[:m]}")
print(f"mean decode {tot_dec / len(PROMPTS):.2f} tok/s")
