"""Prefill vs decode self-consistency on the live server (no external reference).
Greedy decode with top-20 logprobs (decode path) for a few chat prompts; then, for each generated position k, a
prefill of prompt + gen[:k] with max_tokens=1 and top-20 logprobs (prefill path). Same weights, same position:
a correct implementation gives the same distribution up to low-precision noise."""
import json, math, sys, urllib.request, concurrent.futures as cf
B = "http://localhost:20000"; M = "ornith-ai/Ornith-1.5-122B"
def post(p, body):
    r = urllib.request.Request(B + p, data=json.dumps(body).encode(), headers={"content-type": "application/json"})
    with urllib.request.urlopen(r, timeout=1800) as f: return json.load(f)
PROMPTS = {
 "math": "Find all real x such that sqrt(x+3) + sqrt(x-2) = 5. Show your reasoning.",
 "code": "Write a Python function that returns the longest palindromic substring of a string, and explain its complexity.",
 "science": "Why does ice float on water? Explain in terms of hydrogen bonding and density.",
}
G = int(sys.argv[1]) if len(sys.argv) > 1 else 160
def lpd(top): return {int(k.split(":")[1]): v for k, v in top.items()}
res = {}
for name, q in PROMPTS.items():
    ids = post("/tokenize", {"model": M, "messages": [{"role": "user", "content": q}], "add_generation_prompt": True})["tokens"]
    d = post("/v1/completions", {"model": M, "prompt": ids, "max_tokens": G, "temperature": 0, "logprobs": 20,
                                 "return_tokens_as_token_ids": True})["choices"][0]["logprobs"]
    gen = [int(t.split(":")[1]) for t in d["tokens"]]
    dec = [lpd(t) for t in d["top_logprobs"]]
    def pre(k):
        x = post("/v1/completions", {"model": M, "prompt": ids + gen[:k], "max_tokens": 1, "temperature": 0,
                                     "logprobs": 20, "return_tokens_as_token_ids": True})["choices"][0]["logprobs"]
        return lpd(x["top_logprobs"][0])
    with cf.ThreadPoolExecutor(8) as ex: prf = list(ex.map(pre, range(len(gen))))
    res[name] = {"gen": gen, "dec": [{str(k): v for k, v in x.items()} for x in dec], "pre": [{str(k): v for k, v in x.items()} for x in prf]}
    # metrics (position 0 is prefill in both: control)
    agree, kls, d1 = 0, [], []
    for k in range(1, len(gen)):
        a, b = prf[k], dec[k]
        ta, tb = max(a, key=a.get), max(b, key=b.get); agree += ta == tb
        floor = min(min(a.values()), min(b.values())) - 1.0
        kl = sum(math.exp(a[t]) * (a[t] - b.get(t, floor)) for t in a)  # KL(prefill || decode) over prefill top-20
        kls.append(kl); d1.append(abs(a[ta] - b.get(ta, floor)))
    kls.sort(); n = len(kls)
    c0 = max(abs(prf[0][t] - dec[0].get(t, -99)) for t in sorted(prf[0], key=prf[0].get)[-5:])
    print(f"{name}: positions {n}  top-1 agree {agree}/{n} = {agree/n:.3f}  KL mean {sum(kls)/n:.4f} median {kls[n//2]:.4f} "
          f"p95 {kls[int(.95*n)]:.4f} max {kls[-1]:.3f}  |dlogp top1| mean {sum(d1)/n:.3f}  (control pos0 max|d| {c0:.4f})", flush=True)
json.dump(res, open("pd_consistency.json", "w"))
