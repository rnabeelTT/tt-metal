"""Stage-9 serving defect: device state-slot leak (TTScheduler swallows finished_req_ids when a 0-token prefill attempt
falls back to decode-only). Usage: 09-slotleak.py URL OUT_JSON MODE   (MODE: t1 | t2)

t1 (deterministic leak trigger): 15 short requests (2,000-token prompts, staggered max_tokens 200..760) run; a blocker whose
   prompt needs almost the whole DSA pool waits behind them, so every short request that finishes while the blocker
   waits goes through the decode-only fallback (unpatched plugin: its finished id is lost and its state slot leaks).
   After the blocker finishes, 16 more short requests run together; with leaked slots the prefill raises
   "no free device state slot" and the engine dies. PASS: every request returns its full output.
t2 (load test): 1 solo greedy control (the last prompt), then 16 concurrent 20,000-token prompts x 512 greedy tokens (more than the
   pool: vLLM must preempt). PASS: no error, every request 512 tokens, num_preemptions grows; the control is compared token by token.
"""

import bz2
import concurrent.futures as cf
import json
import os
import sys
import time

import requests
from transformers import AutoTokenizer

URL, OUT, MODE = sys.argv[1], sys.argv[2], sys.argv[3]
MODEL = "zai-org/GLM-5.3-Flash"
tok = AutoTokenizer.from_pretrained(os.environ.get("GLM53_CKPT", MODEL))
text = bz2.open(
    os.path.join(os.environ.get("TT_METAL_HOME", "."), "models/tt_transformers/tests/tale-of-two-cities.txt.bz2"),
    "rt",
    errors="ignore",
).read()
ids = tok.encode(text)


def prompt(n, off):
    rep = ids * (2 * (n + off) // len(ids) + 2)
    return rep[off : off + n]


def run(p, maxt, tag):
    t0 = time.perf_counter()
    try:
        r = requests.post(
            f"{URL}/v1/completions",
            timeout=7200,
            json={
                "model": MODEL,
                "prompt": p,
                "max_tokens": maxt,
                "temperature": 0.0,
                "ignore_eos": True,
                "return_token_ids": True,
            },
        )
        if r.status_code != 200:
            return {
                "tag": tag,
                "ok": False,
                "error": f"HTTP {r.status_code}: {r.text[:300]}",
                "wall_s": round(time.perf_counter() - t0, 1),
            }
        c = r.json()["choices"][0]
        tids = c.get("token_ids") or []
        return {
            "tag": tag,
            "ok": len(tids) == maxt,
            "n": len(tids),
            "max_tokens": maxt,
            "token_ids": tids,
            "finish": c["finish_reason"],
            "wall_s": round(time.perf_counter() - t0, 1),
        }
    except Exception as e:  # noqa: BLE001
        return {"tag": tag, "ok": False, "error": repr(e)[:300], "wall_s": round(time.perf_counter() - t0, 1)}


def metric(name):
    try:
        m = requests.get(f"{URL}/metrics", timeout=30).text
        return sum(float(l.split()[-1]) for l in m.splitlines() if l.startswith(name + "{") or l.startswith(name + " "))
    except Exception:  # noqa: BLE001
        return None


def log(*a):
    print("SLOTLEAK", *a, flush=True)


res = {"mode": MODE}
t_start = time.perf_counter()
if MODE == "t1":
    pool = int(os.environ.get("POOL_TOKENS", "265216"))
    mlen = int(os.environ.get("MAX_MODEL_LEN", "262144"))
    blk_max = 32
    blk_len = min(mlen - blk_max, pool - 3 * 128 - blk_max)
    res["preemptions_before"] = metric("vllm:num_preemptions_total")
    with cf.ThreadPoolExecutor(32) as ex:
        shorts = [ex.submit(run, prompt(2000, 1000 + 3001 * i), 200 + 40 * i, f"short{i}") for i in range(15)]
        time.sleep(30)  # let the 15 prefills finish first (FCFS: the blocker must queue behind running decodes)
        blocker = ex.submit(run, prompt(blk_len, 5), blk_max, "blocker")
        log(f"t1 phase 1: 15 short requests (2,000 tokens, max_tokens 200..760) + blocker {blk_len} tokens")
        p1 = [f.result() for f in shorts] + [blocker.result()]
        log(
            f"t1 phase 1 done: {sum(r['ok'] for r in p1)}/16 ok, errors {[r.get('error') for r in p1 if not r['ok']][:2]}"
        )
        p2 = [f.result() for f in [ex.submit(run, prompt(2000, 7 + 2003 * i), 64, f"after{i}") for i in range(16)]]
    log(
        f"t1 phase 2 (16 concurrent after the blocker): {sum(r['ok'] for r in p2)}/16 ok, "
        f"errors {[r.get('error') for r in p2 if not r['ok']][:2]}"
    )
    allr = p1 + p2
    res["requests"] = [{k: v for k, v in r.items() if k != "token_ids"} for r in allr]
    res["preemptions_after"] = metric("vllm:num_preemptions_total")
    res["result"] = "PASS" if all(r["ok"] for r in allr) else "FAIL"
elif MODE == "t2":
    n, plen, maxt = 16, int(os.environ.get("T2_PLEN", "20000")), 512
    prompts = [prompt(plen, 11 + 9973 * i) for i in range(n)]
    ctl = [
        n - 1
    ]  # one spot-check; vLLM preempts the latest-admitted running request first: the last prompt is the likeliest
    solo = {i: run(prompts[i], maxt, f"solo{i}") for i in ctl}
    for i in ctl:
        log(f"t2 solo {i}: {solo[i].get('n')} tokens, {solo[i]['wall_s']} s, ok {solo[i]['ok']}")
    res["preemptions_before"] = metric("vllm:num_preemptions_total")
    with cf.ThreadPoolExecutor(n) as ex:
        conc = [f.result() for f in [ex.submit(run, prompts[i], maxt, f"conc{i}") for i in range(n)]]
    res["preemptions_after"] = metric("vllm:num_preemptions_total")
    ok = all(r["ok"] for r in conc)
    log(
        f"t2 concurrent: {sum(r['ok'] for r in conc)}/{n} finished with {maxt} tokens; num_preemptions "
        f"{res['preemptions_before']} -> {res['preemptions_after']}; errors {[r.get('error') for r in conc if not r['ok']][:2]}"
    )
    eq = {}
    for i in ctl:
        s, c = solo[i].get("token_ids") or [], conc[i].get("token_ids") or []
        div = next((k for k, (x, y) in enumerate(zip(s, c)) if x != y), None)
        eq[i] = {"equal": div is None and len(s) == len(c) and len(s) > 0, "first_divergence": div}
        log(
            f"t2 prompt {i}: concurrent equal to solo {eq[i]['equal']}, first divergence {div}, wall {conc[i]['wall_s']} s"
        )
    res["solo"] = solo
    res["concurrent"] = conc
    res["equality"] = eq
    res["result"] = (
        "PASS"
        if ok and solo[n - 1]["ok"] and (res["preemptions_after"] or 0) > (res["preemptions_before"] or 0)
        else "FAIL"
    )
res["wall_s"] = round(time.perf_counter() - t_start, 1)
json.dump(res, open(OUT, "w"), indent=1)
log(f"RESULT {MODE} {res['result']} wall {res['wall_s']} s -> {OUT}")
