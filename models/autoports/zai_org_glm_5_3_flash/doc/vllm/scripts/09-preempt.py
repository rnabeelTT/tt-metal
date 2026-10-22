"""Stage-9 review #2 P3 (b): preemption smoke on a live server. Usage: 09-preempt.py URL OUT_JSON PROMPT_TOKENS MAX_TOKENS
Two different prompts of PROMPT_TOKENS tokens (tale-of-two-cities, different offsets), greedy, ignore_eos, MAX_TOKENS out.
1. solo A, then solo B (one request at a time: the controls);
2. A and B concurrently: their KV need (2 x (PROMPT + MAX)) exceeds the DSA pool, so vLLM must preempt one of them
   (recompute) while both decode.
PASS: both concurrent requests finish with MAX_TOKENS tokens. Each is compared with its solo control (first divergence,
token count). A preempted request is re-prefilled over its prompt + generated tokens, so its cache rows come from the
prefill path instead of the decode path; a late near-tie divergence is numerics, an early or systematic one is a bug.
The server log tells which request was preempted (vLLM scheduler)."""

import bz2
import concurrent.futures as cf
import json
import os
import sys
import time

import requests
from transformers import AutoTokenizer

URL, OUT, PLEN, MAXT = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
MODEL = "zai-org/GLM-5.3-Flash"
tok = AutoTokenizer.from_pretrained(os.environ.get("GLM53_CKPT", MODEL))
text = bz2.open(
    os.path.join(os.environ.get("TT_METAL_HOME", "."), "models/tt_transformers/tests/tale-of-two-cities.txt.bz2"),
    "rt",
    errors="ignore",
).read()
ids = tok.encode(text)
rep = ids * (2 * PLEN // len(ids) + 2)
prompts = {"A": rep[:PLEN], "B": rep[PLEN // 2 + 777 : PLEN // 2 + 777 + PLEN]}


def run(p):
    t0 = time.perf_counter()
    r = requests.post(
        f"{URL}/v1/completions",
        timeout=14400,
        json={
            "model": MODEL,
            "prompt": p,
            "max_tokens": MAXT,
            "temperature": 0.0,
            "ignore_eos": True,
            "return_token_ids": True,
        },
    )
    r.raise_for_status()
    c = r.json()["choices"][0]
    return {
        "token_ids": c.get("token_ids"),
        "text_tail": c["text"][-120:],
        "finish": c["finish_reason"],
        "n": len(c.get("token_ids") or []),
        "wall_s": round(time.perf_counter() - t0, 1),
    }


def preemptions():
    try:
        m = requests.get(f"{URL}/metrics", timeout=30).text
        return sum(float(l.split()[-1]) for l in m.splitlines() if l.startswith("vllm:num_preemptions"))
    except Exception:  # noqa: BLE001
        return None


res = {"prompt_tokens": PLEN, "max_tokens": MAXT}
for k in ("A", "B"):
    res[f"solo_{k}"] = run(prompts[k])
    print(f"PREEMPT solo {k}: {res[f'solo_{k}']['n']} tokens, {res[f'solo_{k}']['wall_s']} s", flush=True)
res["preemptions_before"] = preemptions()
t0 = time.perf_counter()
with cf.ThreadPoolExecutor(2) as ex:
    fa, fb = ex.submit(run, prompts["A"]), ex.submit(run, prompts["B"])
    res["conc_A"], res["conc_B"] = fa.result(), fb.result()
res["conc_wall_s"] = round(time.perf_counter() - t0, 1)
res["preemptions_after"] = preemptions()
print(f"PREEMPT vllm:num_preemptions {res['preemptions_before']} -> {res['preemptions_after']}", flush=True)
ok = True
for k in ("A", "B"):
    s, c = res[f"solo_{k}"]["token_ids"], res[f"conc_{k}"]["token_ids"]
    div = next((i for i, (x, y) in enumerate(zip(s, c)) if x != y), None)
    res[f"{k}_first_divergence"] = div
    fin = res[f"conc_{k}"]["n"] == MAXT
    ok &= fin
    print(
        f"PREEMPT concurrent {k}: {res[f'conc_{k}']['n']} tokens (finished {fin}), equal to solo {div is None and len(s) == len(c)}, "
        f"first divergence {div}",
        flush=True,
    )
res["result"] = "PASS" if ok else "FAIL"
json.dump(res, open(OUT, "w"), indent=1, ensure_ascii=False)
print("PREEMPT_RESULT", res["result"], f"concurrent wall {res['conc_wall_s']} s", OUT, flush=True)
