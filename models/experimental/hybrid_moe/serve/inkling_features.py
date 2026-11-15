# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Inkling server feature check: reasoning split at the default effort, a tool call, streamed output free of block
markers, and a sampled request. usage: python serve/inkling_features.py [port=8000]"""
import json
import sys
import urllib.request

port = sys.argv[1] if len(sys.argv) > 1 else "8000"
url = f"http://127.0.0.1:{port}/v1/chat/completions"
with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=60) as r:
    model = json.load(r)["data"][0]["id"]
ok = True


def post(body):
    body = {"model": model, **body}
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        return r.read().decode()


def check(name, cond, detail):
    global ok
    ok &= bool(cond)
    print(f"[{'PASS' if cond else 'FAIL'}] {name}: {detail}")


# 1. default reasoning effort (0.9): reasoning separated from content
d = json.loads(post({"messages": [{"role": "user", "content": "What is 17 * 23? Answer with the number."}],
                     "max_tokens": 1024, "temperature": 0}))
m = d["choices"][0]["message"]
rc = m.get("reasoning_content") or m.get("reasoning") or ""
check("reasoning split", rc and "391" in (m.get("content") or "") and "<|" not in (m.get("content") or ""),
      f"reasoning {len(rc)} chars, content {m.get('content')!r}")

# 2. tool call
tools = [{"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city",
                                           "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                                                          "required": ["city"]}}}]
d = json.loads(post({"messages": [{"role": "user", "content": "What's the weather in Paris right now?"}],
                     "tools": tools, "max_tokens": 1024, "temperature": 0,
                     "chat_template_kwargs": {"reasoning_effort": "low"}}))
m = d["choices"][0]["message"]
tc = m.get("tool_calls") or []
args = json.loads(tc[0]["function"]["arguments"]) if tc else {}
check("tool call", tc and tc[0]["function"]["name"] == "get_weather" and "paris" in str(args).lower(),
      f"{[(t['function']['name'], t['function']['arguments']) for t in tc]} finish {d['choices'][0]['finish_reason']}")

# 3. streamed reply: no markers leak into content
raw = post({"messages": [{"role": "user", "content": "Name three primary colors."}], "max_tokens": 200,
            "temperature": 0, "stream": True, "chat_template_kwargs": {"reasoning_effort": "none"}})
text = "".join((c["delta"].get("content") or "") for line in raw.splitlines() if line.startswith("data: {")
               for c in json.loads(line[6:])["choices"])
check("stream clean", text and "<|" not in text, repr(text[-120:]))

# 3b. streamed with reasoning (default effort): reasoning and content both clean
raw = post({"messages": [{"role": "user", "content": "Is 91 prime? One sentence."}], "max_tokens": 1024,
            "temperature": 0, "stream": True})
ev = [c["delta"] for line in raw.splitlines() if line.startswith("data: {") for c in json.loads(line[6:])["choices"]]
rs = "".join((d.get("reasoning_content") or d.get("reasoning") or "") for d in ev)
ct = "".join((d.get("content") or "") for d in ev)
check("stream reasoning clean", rs and ct and "<|" not in rs + ct, f"reasoning {rs[-60:]!r} | content {ct!r}")

# 4. sampled request
d = json.loads(post({"messages": [{"role": "user", "content": "Write a haiku about the sea."}], "max_tokens": 200,
                     "temperature": 0.7, "top_p": 0.95, "seed": 7, "chat_template_kwargs": {"reasoning_effort": "none"}}))
c = d["choices"][0]["message"].get("content") or ""
check("sampled", len(c.split()) >= 5 and "<|" not in c, repr(c))
print("ALL PASS" if ok else "SOME FAILED")
