"""Stage 10 review #1 P2-2 served batch-invariance check (greedy, temperature 0, no logprobs: the device-sampled path).
For each prompt kind (natural-language text, random token ids): a ~4K-token request alone (solo 1), the same request
while a ~70K-token request decodes (concurrent), then alone again (solo 2, run-to-run determinism). The outputs are
compared as text and as re-tokenized ids (stage-10 tickets t23 / t24).
Usage: python tools/vllm_served_invariance.py URL OUT.json [KINDS, default text,random]"""

import bz2
import json
import os
import random
import sys
import threading

import requests
from transformers import AutoTokenizer

URL, OUT = sys.argv[1], sys.argv[2]
KINDS = (sys.argv[3] if len(sys.argv) > 3 else "text,random").split(",")
MODEL = "zai-org/GLM-5.3-Flash"
SHORT, LONG, N_SHORT, N_LONG = 4100, 70000, 64, 800
tok = AutoTokenizer.from_pretrained(os.environ.get("GLM53_CKPT", MODEL))


def complete(prompt, n, stream=False, started=None):
    body = {"model": MODEL, "prompt": prompt, "max_tokens": n, "temperature": 0.0, "ignore_eos": True}
    if not stream:
        r = requests.post(f"{URL}/v1/completions", json=body, timeout=7200)
        r.raise_for_status()
        return r.json()["choices"][0]["text"]
    body["stream"] = True
    text = []
    with requests.post(f"{URL}/v1/completions", json=body, timeout=7200, stream=True) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line or not line.startswith(b"data: ") or line == b"data: [DONE]":
                continue
            text.append(json.loads(line[6:])["choices"][0]["text"])
            if started is not None and not started.is_set():
                started.set()
    return "".join(text)


def first_diff(a, b):
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None if len(a) == len(b) else min(len(a), len(b)))


def prompts(kind):
    if kind == "random":
        r1, r2 = random.Random(1), random.Random(2)
        return [r1.randint(1000, 100000) for _ in range(SHORT)], [r2.randint(1000, 100000) for _ in range(LONG)]
    p = os.path.join(os.environ.get("TT_METAL_HOME", "."), "models/tt_transformers/tests/tale-of-two-cities.txt.bz2")
    ids = tok.encode(bz2.open(p, "rt", errors="ignore").read(), add_special_tokens=False)
    long_ = (ids * (LONG // len(ids) + 1))[:LONG]
    return ids[20000 : 20000 + SHORT], long_


rec, ok = {}, True
for kind in KINDS:
    short, long_ = prompts(kind)
    solo1 = complete(short, N_SHORT)
    res, started = {}, threading.Event()
    th = threading.Thread(target=lambda: res.__setitem__("long", complete(long_, N_LONG, stream=True, started=started)))
    th.start()
    if not started.wait(timeout=3600):
        print(f"SERVED_INV {kind}: the long request never started decoding", flush=True)
        sys.exit(2)
    conc = complete(short, N_SHORT)
    alive = th.is_alive()
    th.join()
    solo2 = complete(short, N_SHORT)
    i1, ic, i2 = (tok.encode(t, add_special_tokens=False) for t in (solo1, conc, solo2))
    r = {
        "solo1": solo1,
        "concurrent": conc,
        "solo2": solo2,
        "ids_solo1": i1,
        "ids_concurrent": ic,
        "ids_solo2": i2,
        "solo_repeat_equal": solo1 == solo2,
        "concurrent_equal": conc == solo1,
        "long_still_decoding": alive,
        "first_diff_concurrent": first_diff(i1, ic),
        "first_diff_solo_repeat": first_diff(i1, i2),
    }
    rec[kind] = r
    ok &= r["solo_repeat_equal"] and r["concurrent_equal"] and alive
    print(
        f"SERVED_INV {kind}: solo repeat equal {r['solo_repeat_equal']} (first diff {r['first_diff_solo_repeat']}); "
        f"concurrent equal {r['concurrent_equal']} (first diff {r['first_diff_concurrent']}); long still decoding "
        f"{alive}; solo {solo1[:40]!r} concurrent {conc[:40]!r}",
        flush=True,
    )
json.dump(rec, open(OUT, "w"), indent=1)
print("SERVED_INV RESULT", "PASS" if ok else "FAIL", flush=True)
sys.exit(0 if ok else 1)
