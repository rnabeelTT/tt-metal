"""Host path (logprobs requested): sample 2 tokens at T=1.0 top_k=20 top_p=0.95; for samples whose first token is ' a',
read the decode-step top logprobs the server reports and the sampled second token."""
import json, urllib.request, concurrent.futures as cf, collections, math
B = "http://localhost:20000"; M = "ornith-ai/Ornith-1.5-122B"
def post(p, body):
    r = urllib.request.Request(B + p, data=json.dumps(body).encode(), headers={"content-type": "application/json"})
    with urllib.request.urlopen(r, timeout=900) as f: return json.load(f)
P = "Q: Name one fruit.\nA: One fruit is"
pre = post("/v1/completions", {"model": M, "prompt": P + " a", "max_tokens": 1, "temperature": 0, "logprobs": 5})
print("prefill-of-(P+' a') top:", {k: round(math.exp(v), 3) for k, v in pre["choices"][0]["logprobs"]["top_logprobs"][0].items()})
def one(_):
    d = post("/v1/completions", {"model": M, "prompt": P, "max_tokens": 2, "temperature": 1.0, "top_k": 20, "top_p": 0.95, "logprobs": 5})
    lp = d["choices"][0]["logprobs"]; return lp["tokens"], lp["top_logprobs"]
with cf.ThreadPoolExecutor(32) as ex: res = list(ex.map(one, range(300)))
sel = [(t, tl) for t, tl in res if t[0] == " a" and len(t) > 1]
print("host-path samples with first ' a':", len(sel), "of", len(res))
if sel:
    print("decode-step reported top (first sample):", {k: round(math.exp(v), 3) for k, v in sel[0][1][1].items()})
    c = collections.Counter(t[1] for t, _ in sel); print("host-path sampled 2nd tokens:", {k: round(v / len(sel), 3) for k, v in c.most_common(6)})
