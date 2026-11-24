# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Smoke test of a running Inkling-2-Mini vLLM server (stdlib only; run on the host):
plain chat (reasoning effort none), default effort (the inkling reasoning parser splits the thinking), a tool call (the
inkling tool parser), and N concurrent requests (batched decode, slot handling). Prints each answer and timings.

usage: python3 tools/smoke_serve.py [port=8000] [concurrency=4]
"""

import concurrent.futures as cf
import json
import sys
import time
import urllib.request

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
CONC = int(sys.argv[2]) if len(sys.argv) > 2 else 4
URL = f"http://localhost:{PORT}/v1/chat/completions"
MODEL = __import__("os").environ.get("MODEL", "inkling-2-mini")


def chat(messages, max_tokens=128, effort="none", tools=None, timeout=600):
    body = {"model": MODEL, "messages": messages, "max_tokens": max_tokens, "temperature": 0,
            "chat_template_kwargs": {"reasoning_effort": effort}}
    if tools:
        body["tools"] = tools
        body["tool_choice"] = "auto"
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.loads(r.read())
    return out, time.time() - t0


def show(name, out, dt):
    m = out["choices"][0]["message"]
    u = out.get("usage", {})
    n = u.get("completion_tokens", 0)
    print(f"--- {name}: {n} tokens in {dt:.1f}s ({n / max(dt, 1e-9):.1f} tok/s), finish {out['choices'][0]['finish_reason']}")
    if m.get("reasoning_content") or m.get("reasoning"):
        print("  reasoning:", repr((m.get("reasoning_content") or m.get("reasoning"))[:300]))
    print("  content:", repr((m.get("content") or "")[:400]))
    if m.get("tool_calls"):
        print("  tool_calls:", json.dumps(m["tool_calls"])[:400])


def main():
    show("plain", *chat([{"role": "user", "content": "What is 84 * 3 / 2? Answer briefly."}]))
    show("effort default", *chat([{"role": "user", "content": "Name three primary colors."}], max_tokens=256,
                                 effort="high"))
    tools = [{"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city",
                                               "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                                                              "required": ["city"]}}}]
    show("tool call", *chat([{"role": "user", "content": "What's the weather in Toronto right now?"}], tools=tools))
    prompts = [f"Write one sentence about the number {i}." for i in range(CONC)]
    t0 = time.time()
    with cf.ThreadPoolExecutor(CONC) as ex:
        res = list(ex.map(lambda p: chat([{"role": "user", "content": p}], max_tokens=64), prompts))
    dt = time.time() - t0
    tot = sum(o.get("usage", {}).get("completion_tokens", 0) for o, _ in res)
    print(f"--- {CONC} concurrent: {tot} tokens in {dt:.1f}s ({tot / dt:.1f} tok/s total)")
    for (o, d), p in zip(res, prompts):
        msg = o["choices"][0]["message"]
        text = msg.get("content") or "(reasoning) " + (msg.get("reasoning_content") or msg.get("reasoning") or "")
        print(f"  [{d:.1f}s] {p!r} -> {text[:150]!r}")
    print("== done")


if __name__ == "__main__":
    main()
