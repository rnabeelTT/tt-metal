"""Prefix caching with several users interleaved (vLLM's prefix cache + the model's prefix-state slots): users A and B
each send a long first turn, then their second turns alternate (A, B, A's turn again, B's again), each turn the
previous conversation plus a new question. Every cached turn's first token is compared with the same request sent
fresh (a vLLM cache_salt of its own: no prefix to reuse); TTFT of each.

usage: python3 prefix_multi.py <port> [chars A] [chars B]   (the two prompts: this repository's sources, two slices)
"""

import json
import pathlib
import sys
import time
import urllib.request
import uuid

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
CA = int(sys.argv[2]) if len(sys.argv) > 2 else 56000
CB = int(sys.argv[3]) if len(sys.argv) > 3 else 112000
url = f"http://localhost:{PORT}/v1/chat/completions"
model = json.load(urllib.request.urlopen(f"http://localhost:{PORT}/v1/models"))["data"][0]["id"]
here = pathlib.Path(__file__).resolve().parent.parent  # models/experimental/hybrid_moe
src = "".join(p.read_text() for p in sorted((here / "hybrid").glob("*.py")) + sorted(here.glob("*.py")))


def post(body):
    body = dict(model=model, **body)
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=3600) as r:
        out = json.load(r)
    return out, time.perf_counter() - t0


def first(msgs, salt=None):
    """-> (top-5 {token: logprob} of the first token, seconds, prompt tokens)."""
    body = dict(messages=msgs, max_tokens=1, temperature=0, logprobs=True, top_logprobs=5)
    if salt:
        body["cache_salt"] = salt
    r, dt = post(body)
    c = r["choices"][0]["logprobs"]["content"][0]
    return {t["token"]: t["logprob"] for t in c["top_logprobs"]}, dt, r["usage"]["prompt_tokens"]


def reply(msgs):
    r, _ = post(dict(messages=msgs, max_tokens=48, temperature=0, chat_template_kwargs={"reasoning_effort": "none"}))
    return r["choices"][0]["message"]["content"] or ""


def check(name, fresh, cached):
    a, b = fresh[0], cached[0]
    ta, tb = max(a, key=a.get), max(b, key=b.get)
    common = set(a) & set(b)
    d = max((abs(a[k] - b[k]) for k in common), default=float("nan"))
    ok = ta == tb
    print(
        f"CHECK {name}: {cached[2]} prompt tokens, TTFT fresh {fresh[1]:.2f} s cached {cached[1]:.2f} s | top-1 "
        f"{'same' if ok else 'DIFFERENT'} ({ta!r} / {tb!r}), top-5 overlap {len(common)}/5, max |dlogprob| {d:.3f}",
        flush=True,
    )
    return ok


A = [{"role": "user", "content": "Here is some Python code:\n\n" + src[:CA] + "\n\nWhat does it do? One sentence."}]
B = [{"role": "user", "content": "Here is more Python code:\n\n" + src[-CB:] + "\n\nName its main class. One line."}]
ok = True
# first turns (B's prompt shares nothing with A's)
for name, msgs in (("A turn 1", A), ("B turn 1", B)):
    f = first(msgs)
    print(f"{name}: {f[2]} prompt tokens, TTFT {f[1]:.2f} s (fresh)", flush=True)
A = A + [{"role": "assistant", "content": reply(A)}, {"role": "user", "content": "And what is the first function?"}]
B = B + [{"role": "assistant", "content": reply(B)}, {"role": "user", "content": "Which module defines it?"}]
# second turns, interleaved: each resumes from its own first turn's snapshot although the other user ran in between
for name, msgs in (("A turn 2", A), ("B turn 2", B), ("A turn 2 again", A), ("B turn 2 again", B)):
    cached = first(msgs)
    fresh = first(msgs, salt=uuid.uuid4().hex)
    ok &= check(name, fresh, cached)
print("ALL PASS" if ok else "SOME FAILED", flush=True)
