"""Does the served sampler draw from the distribution it should (temperature / top_k / top_p, device path)?
Expected distribution: the server's own raw top-20 logprobs (logprobs request, host path), with T / top_k / top_p
applied here. Samples: N plain requests without logprobs (device sampling, the path the evals take).
Step 'prefill': the first generated token. Step 'decode': the second token, among samples whose first token is the
forced-likely one (max_tokens=2), against logprobs of prompt+first token."""
import json, math, random, sys, urllib.request, concurrent.futures as cf, collections
B = "http://localhost:20000"; M = "ornith-ai/Ornith-1.5-122B"
def post(p, body):
    r = urllib.request.Request(B + p, data=json.dumps(body).encode(), headers={"content-type": "application/json"})
    with urllib.request.urlopen(r, timeout=900) as f: return json.load(f)
def dist(prompt):
    d = post("/v1/completions", {"model": M, "prompt": prompt, "max_tokens": 1, "temperature": 0, "logprobs": 20})
    return d["choices"][0]["logprobs"]["top_logprobs"][0]  # {text: logprob}
def expected(top, T, k, p):
    lp = sorted(top.items(), key=lambda z: -z[1])[:k]
    w = [math.exp((l - lp[0][1]) / T) for _, l in lp]; z = sum(w)
    keep, c = [], 0.0
    for (t, _), x in zip(lp, w):
        keep.append((t, x / z)); c += x / z
        if c >= p: break
    z = sum(q for _, q in keep); return {t: q / z for t, q in keep}
def tvd(a, b): return 0.5 * sum(abs(a.get(t, 0) - b.get(t, 0)) for t in set(a) | set(b))
def noise_tvd(exp, n, reps=200):
    ts, ps = list(exp), list(exp.values()); r = random.Random(0); out = []
    for _ in range(reps):
        c = collections.Counter(r.choices(ts, ps, k=n)); out.append(tvd(exp, {t: c[t] / n for t in ts}))
    out.sort(); return out[len(out) // 2], out[int(0.95 * len(out))]
def report(label, exp, samples):
    n = len(samples); c = collections.Counter(samples); emp = {t: c[t] / n for t in c}
    med, p95 = noise_tvd(exp, n)
    top = max(exp, key=exp.get)
    print(f"{label}: n={n} TVD={tvd(exp, emp):.3f} (sampling noise median {med:.3f}, p95 {p95:.3f}); "
          f"argmax share expected {exp[top]:.2f} observed {emp.get(top, 0):.2f}; "
          f"outside expected support {sum(v for t, v in emp.items() if t not in exp):.3f}")
    for t in sorted(exp, key=lambda t: -exp[t])[:6]: print(f"    {t!r:>14} exp {exp[t]:.3f} obs {emp.get(t, 0):.3f}")
def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    prompt = "Q: Name one fruit.\nA: One fruit is"
    for T, k, p in [(0.6, 20, 0.95), (1.0, 20, 0.95)]:
        d1 = dist(prompt); exp1 = expected(d1, T, k, p); first = " a"
        exp2 = expected(dist(prompt + first), T, k, p)
        def one(_):
            d = post("/v1/completions", {"model": M, "prompt": prompt, "max_tokens": 2, "temperature": T,
                                         "top_k": k, "top_p": p, "return_tokens_as_token_ids": False})
            return d["choices"][0]["text"]
        with cf.ThreadPoolExecutor(32) as ex: outs = list(ex.map(one, range(N)))
        s1 = [o[:len(first)] if o.startswith(first) else None for o in outs]
        # first-token text: match against the expected support strings by prefix
        firsts = []
        for o in outs:
            m = [t for t in exp1 if o.startswith(t)]; firsts.append(max(m, key=len) if m else o[:6] + "...")
        report(f"T={T} top_k={k} top_p={p} prefill token", exp1, firsts)
        def toks(o): return post("/tokenize", {"model": M, "prompt": o, "add_special_tokens": False})["tokens"]
        fid = toks(first)
        with cf.ThreadPoolExecutor(16) as ex: tk = list(ex.map(toks, outs))
        seconds = [o[len(first):] for o, t in zip(outs, tk) if t[:len(fid)] == fid and len(t) == len(fid) + 1]
        sec = []
        for o in seconds:
            m = [t for t in exp2 if o == t]; sec.append(m[0] if m else o)
        report(f"T={T} top_k={k} top_p={p} decode token (given {first!r})", exp2, sec)
main()
