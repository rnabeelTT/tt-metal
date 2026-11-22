# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Hy4-flash server checks: streaming chat (TTFT, decode tok/s, text), a tool call, a long prompt.

    python serve/hy4_client.py [port] [--effort no_think|high] [--max-tokens 256] [--long N] [--temp 0]
"""

from __future__ import annotations

import os
import argparse
import json
import time
import urllib.request

PROMPTS = [
    "A train travels 120 km in 1.5 hours. What is its average speed? Answer briefly.",
    "Write a Python function that returns the n-th Fibonacci number iteratively.",
    "Explain in three sentences why the sky is blue.",
    "用三句话介绍长城。",
    "List five practical tips for writing clear technical documentation.",
    "Translate into French: 'The meeting has been moved to Thursday afternoon.'",
]


def stream_chat(port, messages, max_tokens, temp, effort, tools=None):
    body = {"model": "hy4-flash", "messages": messages, "max_tokens": max_tokens, "temperature": temp,
            "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"reasoning_effort": effort}}
    if os.environ.get("HY4_CLIENT_TOP_P"):  # (bench: nucleus sampling requests)
        body["top_p"] = float(os.environ["HY4_CLIENT_TOP_P"])
    if tools:
        body["tools"] = tools
    req = urllib.request.Request(f"http://localhost:{port}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    t_first = None
    text, reasoning, tool_calls, usage, n_chunks = "", "", [], None, 0
    with urllib.request.urlopen(req, timeout=3600) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            d = json.loads(line[5:])
            if d.get("usage"):
                usage = d["usage"]
            for ch in d.get("choices", []):
                delta = ch.get("delta", {})
                piece = (delta.get("content") or "") + (delta.get("reasoning_content") or delta.get("reasoning") or "")
                if piece and t_first is None:
                    t_first = time.time()
                text += delta.get("content") or ""
                reasoning += delta.get("reasoning_content") or delta.get("reasoning") or ""
                if delta.get("tool_calls"):
                    tool_calls += delta["tool_calls"]
                    if t_first is None:
                        t_first = time.time()
                n_chunks += 1
    t1 = time.time()
    out_tok = (usage or {}).get("completion_tokens", 0)
    ttft = (t_first or t1) - t0
    rate = (out_tok - 1) / max(t1 - (t_first or t0), 1e-9) if out_tok > 1 else 0.0
    return {"ttft": ttft, "tok_s": rate, "out_tokens": out_tok, "prompt_tokens": (usage or {}).get("prompt_tokens"),
            "text": text, "reasoning": reasoning, "tool_calls": tool_calls}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("port", nargs="?", default="8000")
    ap.add_argument("--effort", default="no_think")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--long", type=int, default=0, help="also a prompt of ~N tokens (needle at the start)")
    ap.add_argument("--n", type=int, default=len(PROMPTS))
    args = ap.parse_args()
    rates, ttfts = [], []
    for p in PROMPTS[: args.n]:
        r = stream_chat(args.port, [{"role": "user", "content": p}], args.max_tokens, args.temp, args.effort)
        rates.append(r["tok_s"])
        ttfts.append(r["ttft"])
        print(f"--- {p[:60]!r}: prompt {r['prompt_tokens']} tok, TTFT {r['ttft']:.2f} s, {r['out_tokens']} tok at "
              f"{r['tok_s']:.1f} tok/s\n{(r['reasoning'][:200] + ' || ') if r['reasoning'] else ''}{r['text'][:400]}")
    tools = [{"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city",
                                               "parameters": {"type": "object", "properties": {
                                                   "city": {"type": "string"}}, "required": ["city"]}}}]
    r = stream_chat(args.port, [{"role": "user", "content": "What's the weather in Paris right now?"}], 128, 0.0,
                    args.effort, tools=tools)
    print(f"--- tool call: {json.dumps(r['tool_calls'])[:300]} text {r['text'][:200]!r}")
    if args.long:
        filler = " ".join(f"Sentence {i} is filler text about nothing in particular." for i in range(args.long // 10))
        msg = f"The secret code is PELICAN-42. Remember it.\n{filler}\nWhat is the secret code?"
        r = stream_chat(args.port, [{"role": "user", "content": msg}], 32, 0.0, args.effort)
        print(f"--- long: prompt {r['prompt_tokens']} tok, TTFT {r['ttft']:.2f} s: {r['text'][:200]!r}")
    print(f"SUMMARY decode tok/s mean {sum(rates) / len(rates):.1f} (min {min(rates):.1f} max {max(rates):.1f}); "
          f"TTFT mean {sum(ttfts) / len(ttfts):.2f} s")


if __name__ == "__main__":
    main()
