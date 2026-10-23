"""Stage 11 review #2 (P2 non-English): sampled-output probe on a running server for the ifeval prompts that came out
garbled or looped in the release run: docs 18 (Kannada), 31 (Punjabi), 50 (Marathi), 87 (Persian), 106 (Vietnamese).
Usage: 11r2-langprobe.py URL OUT_JSON [N_SEEDS] [MAX_TOKENS]
Settings = the eval: temperature 1.0, top_p 0.95, reasoning_effort low; seeds 1..N (default 8), max_tokens 2048.
Per draw: finish reason, the loop rule of 11-loopscan.py (8-gram >= 20, word run >= 50, 16-char substring >= 1,000;
reasoning + content), and the script mix of the content letters: share in the requested script (Latin for
Vietnamese) and the foreign scripts seen. A draw is "mixed" when < 90 % of its content letters are in the requested
script (any CJK / Cyrillic / other-Indic letter in the content also counts)."""
import json
import sys
import time
import unicodedata
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

URL, OUT = sys.argv[1], sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 8
MAXT = int(sys.argv[4]) if len(sys.argv) > 4 else 2048
W = Path.home() / "glm53-flash-bringup"
F = W / (
    "tti/workflow_logs/reports_output/release/zai-org__GLM-5.3-Flash_p300x2_release/eval_id_glm53-flash-autoport_"
    "GLM-5.3-Flash_p300x2/zai-org__GLM-5.3-Flash/samples_ifeval_2026-10-21T19-44-35.100275.jsonl"
)
DOCS = {18: "KANNADA", 31: "GURMUKHI", 50: "DEVANAGARI", 87: "ARABIC", 106: "LATIN"}
PROMPTS = {}
for line in open(F):
    r = json.loads(line)
    if r["doc_id"] in DOCS:
        PROMPTS[r["doc_id"]] = r["doc"]["prompt"]
BASE = {
    "model": "zai-org/GLM-5.3-Flash",
    "temperature": 1.0,
    "top_p": 0.95,
    "reasoning_effort": "low",
    "stop": ["<|endoftext|>"],
    "max_tokens": MAXT,
}


def loopstats(text):
    w = text.split()
    run = best = 1 if w else 0
    for i in range(1, len(w)):
        run = run + 1 if w[i] == w[i - 1] else 1
        best = max(best, run)
    g = Counter(tuple(w[i : i + 8]) for i in range(len(w) - 7))
    c = Counter(text[i : i + 16] for i in range(len(text) - 15))
    top8 = g.most_common(1)[0][1] if g else 0
    top16 = c.most_common(1)[0][1] if c else 0
    return {
        "max_word_run": best,
        "max_8gram": top8,
        "max_16char": top16,
        "loop": best >= 50 or top8 >= 20 or top16 >= 1000,
    }


def script(ch):
    n = unicodedata.name(ch, "")
    if "CJK" in n or "HIRAGANA" in n or "KATAKANA" in n or "HANGUL" in n:
        return "CJK"
    return n.split(" ")[0] if n else "?"


def scriptmix(text, want):
    s = Counter(script(ch) for ch in text if ch.isalpha())
    tot = sum(s.values())
    share = s[want] / tot if tot else 0.0
    foreign = {k: v for k, v in s.items() if k != want and not (want != "LATIN" and k == "LATIN")}
    hard = {
        k: v
        for k, v in foreign.items()
        if k in ("CJK", "CYRILLIC", "THAI", "HEBREW", "GREEK") or (k in DOCS.values() and k != want)
    }
    return {
        "letters": tot,
        "share": round(share, 3),
        "scripts": dict(s.most_common(6)),
        "mixed": tot > 0 and (share < 0.9 or bool(hard)),
    }


def ask(doc, seed):
    t0 = time.time()
    r = requests.post(
        f"{URL}/v1/chat/completions",
        json={**BASE, "seed": seed, "messages": [{"role": "user", "content": PROMPTS[doc]}]},
        timeout=7200,
    )
    r.raise_for_status()
    j = r.json()
    ch = j["choices"][0]
    m = ch["message"]
    reasoning = m.get("reasoning") or m.get("reasoning_content") or ""
    content = m.get("content") or ""
    rec = {
        "doc": doc,
        "seed": seed,
        "finish": ch["finish_reason"],
        "completion_tokens": j["usage"]["completion_tokens"],
        "secs": round(time.time() - t0, 1),
        **loopstats(reasoning + "\n" + content),
        **scriptmix(content, DOCS[doc]),
        "content": content[:1500],
        "reasoning_head": reasoning[:300],
    }
    print(
        json.dumps({k: rec[k] for k in ("doc", "seed", "finish", "completion_tokens", "loop", "share", "mixed")}),
        flush=True,
    )
    return rec


jobs = [(d, s) for d in DOCS for s in range(1, N + 1)]
with ThreadPoolExecutor(16) as ex:
    res = list(ex.map(lambda a: ask(*a), jobs))
summ = {}
for r in res:
    k = f"doc {r['doc']} ({DOCS[r['doc']].lower()})"
    d = summ.setdefault(k, {"draws": 0, "loop": 0, "mixed": 0, "length_cut": 0, "clean": 0})
    d["draws"] += 1
    d["loop"] += r["loop"]
    d["mixed"] += r["mixed"]
    d["length_cut"] += r["finish"] == "length"
    d["clean"] += not (r["loop"] or r["mixed"] or r["finish"] == "length")
out = {"settings": BASE, "seeds": N, "url": URL, "summary": summ, "results": res}
Path(OUT).write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
print("LANGPROBE_SUMMARY " + json.dumps(summ))
