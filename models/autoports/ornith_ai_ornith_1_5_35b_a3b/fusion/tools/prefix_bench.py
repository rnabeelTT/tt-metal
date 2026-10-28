"""Prefix-caching benchmark on a served bundle: what a cached prefix saves in time to first token (TTFT), and whether
the output is unchanged.

    URL=http://localhost:20000 python prefix_bench.py repeat [tokens ...]   # each prompt sent twice: cold, then warm
    URL=http://localhost:20000 python prefix_bench.py chat [doc_tokens] [turns]  # multi-turn chat over one document

repeat: one real-text prompt per length (distinct from the first token, so lengths never share a prefix), sent twice;
TTFT cold / warm and whether the greedy continuations are identical.
chat: a conversation that opens with a long document and asks follow-up questions; each turn resends the whole
conversation (as every chat client does) and records TTFT, prompt tokens and a hash of the greedy reply. Run it against
a server with prefix caching on and one with it off: the replies must match and the TTFT difference is the benefit.
THINK=1 turns thinking on (its reasoning is dropped from the history by the chat template). TAG labels the output.
Streams at temperature 0; TTFT = first streamed token (content or reasoning) after the request is sent.
"""

import glob
import hashlib
import json
import os
import sys
import time
import urllib.request

URL = os.environ.get("URL", "http://localhost:20000")
THINK = os.environ.get("THINK", "0") == "1"
TAG = os.environ.get("TAG", "")
METAL = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../.."))


def _model():
    with urllib.request.urlopen(URL + "/v1/models") as r:
        return json.load(r)["data"][0]["id"]


MODEL = None


def corpus(skip_chars, n_chars):
    """Real text: the repo's Python sources concatenated in a fixed order, from ``skip_chars`` on."""
    out, total = [], 0
    for path in sorted(glob.glob(os.path.join(METAL, "models/demos/**/*.py"), recursive=True)):
        with open(path, errors="ignore") as f:
            t = f"# file: {os.path.relpath(path, METAL)}\n" + f.read()
        if skip_chars >= len(t):
            skip_chars -= len(t)
            continue
        t = t[skip_chars:]
        skip_chars = 0
        out.append(t)
        total += len(t)
        if total >= n_chars:
            break
    return "".join(out)[:n_chars]


def chat(messages, max_tokens):
    body = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": THINK},
    }
    req = urllib.request.Request(
        URL + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    t0 = time.perf_counter()
    ttft, text, reasoning, usage = None, [], [], {}
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            ev = json.loads(line[5:])
            if ev.get("usage"):
                usage = ev["usage"]
            for ch in ev.get("choices", []):
                d = ch.get("delta", {})
                piece = d.get("content") or ""
                rpiece = d.get("reasoning_content") or d.get("reasoning") or ""
                if (piece or rpiece) and ttft is None:
                    ttft = time.perf_counter() - t0
                text.append(piece)
                reasoning.append(rpiece)
    return ttft, "".join(text), "".join(reasoning), usage


def h(s):
    return hashlib.sha1(s.encode()).hexdigest()[:10]


def repeat(lengths):
    rows = []
    for i, n in enumerate(lengths):
        doc = corpus(skip_chars=2_000_003 * (i + 1) % 9_000_000, n_chars=int(n * 3.3))
        msgs = [{"role": "user", "content": f"Here is some code:\n\n{doc}\n\nIn two sentences, what does the last file do?"}]
        t1, o1, _, u1 = chat(msgs, 48)
        t2, o2, _, u2 = chat(msgs, 48)
        row = dict(prompt_tokens=u1.get("prompt_tokens"), ttft_cold=round(t1, 3), ttft_warm=round(t2, 3),
                   speedup=round(t1 / t2, 2), identical=o1 == o2, out=o1[:80])
        print(json.dumps(row), flush=True)
        rows.append(dict(row, text=o1))
    return rows


QUESTIONS = [
    "In two sentences, what does the last file in that dump do?",
    "Which function in it would you refactor first, and why? Be brief.",
    "Write a one-paragraph docstring for that function.",
    "List three edge cases a unit test for it should cover.",
    "Write the test for the first edge case.",
    "Summarize our conversation so far in three bullet points.",
    "What is one performance risk in that code?",
    "Give a final one-line recommendation.",
]


def conversation(doc_tokens, turns):
    doc = corpus(skip_chars=777_777, n_chars=int(doc_tokens * 3.3))
    msgs = [
        {"role": "system", "content": "You are a concise senior code reviewer."},
        {"role": "user", "content": f"Here is some code:\n\n{doc}\n\n{QUESTIONS[0]}"},
    ]
    rows = []
    for t in range(turns):
        ttft, out, reasoning, usage = chat(msgs, 160)
        row = dict(turn=t + 1, prompt_tokens=usage.get("prompt_tokens"), ttft=round(ttft, 3), reply=h(out),
                   reasoning=h(reasoning) if reasoning else "", head=out[:60])
        print(json.dumps(row), flush=True)
        rows.append(dict(row, text=out))
        msgs.append({"role": "assistant", "content": out})
        msgs.append({"role": "user", "content": QUESTIONS[(t + 1) % len(QUESTIONS)]})
    return rows


if __name__ == "__main__":
    MODEL = _model()
    mode = sys.argv[1]
    if mode == "repeat":
        res = repeat([int(a) for a in sys.argv[2:]] or [2048, 8192, 32768])
    elif mode == "chat":
        res = conversation(int(sys.argv[2]) if len(sys.argv) > 2 else 8192, int(sys.argv[3]) if len(sys.argv) > 3 else 6)
    else:
        raise SystemExit(__doc__)
    print("RESULT " + json.dumps({"tag": TAG, "mode": mode, "think": THINK, "rows": res}))
