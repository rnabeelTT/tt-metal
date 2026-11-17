"""Prefix-cache check against a running server: first-token top-5 logprobs, fresh prefill vs prefix-restored, for a
resent prompt, a prompt with a different tail, and a second chat turn; then streamed TTFTs (fresh / resent / 2nd turn).
A short request between cases clears the engine's prefix entry (fresh baseline)."""

import json
import pathlib
import sys
import time
import urllib.request

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
CHARS = [int(v) for v in (sys.argv[2] if len(sys.argv) > 2 else "56000").split(",")]
OUT = sys.argv[3] if len(sys.argv) > 3 else None  # json of the fresh first-token top-5 per case
LP_ONLY = len(sys.argv) > 4 and sys.argv[4] == "lp"
saved = {}
url = f"http://localhost:{PORT}/v1/chat/completions"
model = json.load(urllib.request.urlopen(f"http://localhost:{PORT}/v1/models"))["data"][0]["id"]
here = pathlib.Path(__file__).resolve().parent.parent  # models/experimental/hybrid_moe
doc = "".join(p.read_text() for p in sorted((here / "hybrid").glob("*.py")) + sorted(here.glob("*.py")))


def post(body, stream=False):
    body = dict(model=model, **body)
    if stream:
        body.update(stream=True, stream_options={"include_usage": True})
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=3600) as r:
        if not stream:
            return json.load(r), time.perf_counter() - t0
        ttft, usage, text = None, None, ""
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            ev = json.loads(line[6:])
            usage = ev.get("usage") or usage
            for ch in ev.get("choices", []):
                d = ch.get("delta", {})
                piece = d.get("content") or d.get("reasoning_content") or d.get("reasoning") or ""
                if ttft is None and piece:
                    ttft = time.perf_counter() - t0
                text += piece
        return dict(ttft=round(ttft, 3), isl=usage["prompt_tokens"], e2el=round(time.perf_counter() - t0, 2), text=text)


def top5(msgs):
    r, _ = post(dict(messages=msgs, max_tokens=1, temperature=0, logprobs=True, top_logprobs=5))
    c = r["choices"][0]["logprobs"]["content"][0]
    return {t["token"]: t["logprob"] for t in c["top_logprobs"]}


def clear():
    post(dict(messages=[{"role": "user", "content": "Say hi."}], max_tokens=2, temperature=0))


def cmp(name, a, b):
    ta, tb = max(a, key=a.get), max(b, key=b.get)
    common = set(a) & set(b)
    d = max((abs(a[k] - b[k]) for k in common), default=float("nan"))
    print(
        f"CHECK {name}: top-1 fresh {ta!r} cached {tb!r} same={ta == tb} | top-5 overlap {len(common)}/5, "
        f"max |dlogprob| {d:.3f}",
        flush=True,
    )


for chars in CHARS:
    q = "Here is some Python code:\n\n" + doc[:chars] + "\n\n"
    A = [{"role": "user", "content": q + "What does this code do? Answer briefly."}]
    B = [{"role": "user", "content": q + "List three functions defined in this code."}]
    # logprob checks: fresh vs cached
    clear()
    fa = top5(A)
    ca = top5(A)
    cmp(f"{chars} resent", fa, ca)
    clear()
    fa2 = top5(A)
    cmp(f"{chars} fresh vs fresh", fa, fa2)
    saved[f"{chars} A"] = fa
    # B fresh, then A (restored from B's snapshot), then B restored from A's snapshot
    clear()
    fb = top5(B)
    top5(A)
    cb = top5(B)
    cmp(f"{chars} other tail", fb, cb)
    saved[f"{chars} B"] = fb
    if OUT:
        json.dump(saved, open(OUT, "w"), indent=1)
    if LP_ONLY:
        continue
    # TTFT: fresh, resent, second turn
    clear()
    r1 = post(dict(messages=A, max_tokens=128, temperature=0), stream=True)
    r2 = post(dict(messages=A, max_tokens=128, temperature=0), stream=True)
    T2 = A + [
        {"role": "assistant", "content": r1["text"]},
        {"role": "user", "content": "Now explain it in one sentence."},
    ]
    r3 = post(dict(messages=T2, max_tokens=128, temperature=0), stream=True)
    clear()
    r4 = post(dict(messages=T2, max_tokens=128, temperature=0), stream=True)
    print(
        f"TTFT {chars}: ISL {r1['isl']} fresh {r1['ttft']} s | resent {r2['ttft']} s (same text: {r1['text'] == r2['text']}) "
        f"| 2nd turn ISL {r3['isl']} cached {r3['ttft']} s, fresh {r4['ttft']} s (same text: {r3['text'] == r4['text']})",
        flush=True,
    )
print("== done", flush=True)
