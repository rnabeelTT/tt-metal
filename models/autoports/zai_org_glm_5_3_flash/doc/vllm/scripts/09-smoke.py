"""Stage-9 served smokes for GLM-5.3-Flash (E192). Usage: 09-smoke.py URL [--long N]
Chat (short reasoning) + tool call (glm47 parser), host-sampled paths (logprobs / repetition penalty / min_p),
non-aligned prompt lengths (not multiples of 4 / 128 / 2048, incl. one crossing the DSA dense limit 2050), greedy
re-run determinism, concurrent burst with per-stream solo controls, page growth across several 128-token blocks."""

import bz2
import concurrent.futures as cf
import os
import sys
import time
from pathlib import Path

import requests
from transformers import AutoTokenizer

URL = sys.argv[1]
MODEL = "zai-org/GLM-5.3-Flash"
LONG = int(sys.argv[sys.argv.index("--long") + 1]) if "--long" in sys.argv else 0
tok = AutoTokenizer.from_pretrained(os.environ.get("GLM53_CKPT", MODEL))
LOW = {"chat_template_kwargs": {"reasoning_effort": "low"}}
# reduced serving target (GLM53_LAYERS set): no semantics; gate only the mechanical checks
SEMANTIC = not os.environ.get("GLM53_LAYERS")


def chat(msgs, **kw):
    r = requests.post(f"{URL}/v1/chat/completions", json={"model": MODEL, "messages": msgs, **kw}, timeout=3600)
    r.raise_for_status()
    return r.json()


def comp(p, n, **kw):
    body = {"model": MODEL, "prompt": p, "max_tokens": n, "temperature": 0.0, **kw}
    r = requests.post(f"{URL}/v1/completions", json=body, timeout=3600)
    r.raise_for_status()
    return r.json()


ok = True
j = chat(
    [{"role": "user", "content": "Write a Python one-liner that reverses a string s."}],
    max_tokens=1024,
    temperature=0,
    **LOW,
)
m = j["choices"][0]["message"]
print(
    "SMOKE_CHAT",
    j["choices"][0]["finish_reason"],
    "reasoning",
    len(m.get("reasoning") or m.get("reasoning_content") or ""),
    "content",
    repr((m["content"] or "")[:160]),
)
ok &= bool(m["content"]) or not SEMANTIC
tools = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file from the repository",
            "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
        },
    }
]
j = chat(
    [{"role": "user", "content": "Open the file src/main.py so we can look at it."}],
    tools=tools,
    tool_choice="auto",
    max_tokens=1024,
    temperature=0,
    **LOW,
)
tc = j["choices"][0]["message"].get("tool_calls")
print("SMOKE_TOOL", j["choices"][0]["finish_reason"], str(tc)[:300])
ok &= bool(tc) or not SEMANTIC
j = comp("def fibonacci(n):", 16, logprobs=3)
print("SMOKE_LOGPROBS", repr(j["choices"][0]["text"][:60]), (j["choices"][0]["logprobs"]["top_logprobs"] or [None])[0])
print("SMOKE_REPPEN", repr(comp("def fibonacci(n):", 32, repetition_penalty=1.2)["choices"][0]["text"][:80]))
print("SMOKE_MINP", repr(comp("def fibonacci(n):", 32, min_p=0.1, temperature=0.7, seed=3)["choices"][0]["text"][:80]))
print(
    "SMOKE_SEEDED",
    repr(comp("def fibonacci(n):", 32, temperature=0.8, top_k=20, top_p=0.95, seed=7)["choices"][0]["text"][:80]),
)
a = comp("import numpy as np\n\ndef softmax(x):", 48)["choices"][0]["text"]
b = comp("import numpy as np\n\ndef softmax(x):", 48)["choices"][0]["text"]
print("SMOKE_GREEDY_REPEAT", a == b, repr(a[:80]))
ok &= a == b
text = bz2.open(
    os.path.join(os.environ.get("TT_METAL_HOME", "."), "models/tt_transformers/tests/tale-of-two-cities.txt.bz2"),
    "rt",
    errors="ignore",
).read()
ids = tok.encode(text)
for n in (1, 3, 127, 129, 2049, 2051, 2381, 4223):  # pool / block / prefill-chunk / DSA dense-limit boundaries
    j = comp(ids[:n], 8)
    print("SMOKE_NONALIGNED", n, j["usage"]["prompt_tokens"], repr(j["choices"][0]["text"][:50]))
# page growth: one request decodes across >= 3 new 128-token blocks while others run
j = comp(ids[5000:5100], 400, ignore_eos=True)
print("SMOKE_PAGE_GROWTH", j["usage"]["completion_tokens"], repr(j["choices"][0]["text"][-60:]))
N = int(os.environ.get("BURST", "8"))
prompts = [ids[3000 * i : 3000 * i + 100 + (i % 7)] for i in range(N)]
solo = [comp(p, 64)["choices"][0]["text"] for p in prompts]  # per-stream solo controls (greedy)
t0 = time.perf_counter()
with cf.ThreadPoolExecutor(N) as ex:
    outs = list(ex.map(lambda p: comp(p, 64), prompts))
w = time.perf_counter() - t0
same = sum(o["choices"][0]["text"] == s for o, s in zip(outs, solo))
n = sum(o["usage"]["completion_tokens"] for o in outs)
ok &= same == N
print(f"SMOKE_BURST {len(outs)} ok, {n} tokens in {w:.1f}s = {n / w:.1f} tok/s; {same}/{N} identical to solo greedy")
# shipped-mode control: the bundle serves max reasoning effort (template default); the readiness runs serve with low
# effort. Two chat prompts at an explicit max effort with room for the reasoning to finish: content must arrive.
if SEMANTIC and os.environ.get("SMOKE_MAXEFFORT", "1") == "1":
    import json

    rec = []
    for q in (
        "Write a haiku about machine learning.",
        "A train leaves at 3:40 pm and arrives at 6:15 pm. How long is the trip? Answer briefly.",
    ):
        j = chat(
            [{"role": "user", "content": q}],
            max_tokens=4096,
            temperature=0,
            chat_template_kwargs={"reasoning_effort": "max"},
        )
        c = j["choices"][0]
        m = c["message"]
        r = m.get("reasoning") or m.get("reasoning_content") or ""
        rec.append(
            {
                "prompt": q,
                "finish_reason": c["finish_reason"],
                "completion_tokens": j["usage"]["completion_tokens"],
                "reasoning": r,
                "content": m["content"],
            }
        )
        print(
            "SMOKE_MAXEFFORT",
            c["finish_reason"],
            j["usage"]["completion_tokens"],
            "tokens; reasoning",
            len(r),
            "chars:",
            repr(r[:120]),
            "...",
            repr(r[-120:]),
            "| content",
            repr((m["content"] or "")[:200]),
        )
        ok &= bool(m["content"]) and c["finish_reason"] == "stop"
    out = os.environ.get("SMOKE_JSON")
    if out:
        Path(out).write_text(json.dumps(rec, indent=1, ensure_ascii=False))
if LONG:
    t0 = time.perf_counter()
    j = comp((ids * (LONG // len(ids) + 1))[:LONG], 16)
    print(
        f"SMOKE_LONG {j['usage']['prompt_tokens']} wall {time.perf_counter() - t0:.1f}s",
        repr(j["choices"][0]["text"][:60]),
    )
print("SMOKE_RESULT", "PASS" if ok else "FAIL", "" if SEMANTIC else "(reduced target: mechanical gates only)")
