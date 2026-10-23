"""Stage 11 (review P2-2): n-gram repetition scan over every saved eval generation of stage 11.
Usage: 11-loopscan.py OUT_JSON
A generation is a loop when one whitespace 8-gram occurs >= 20 times, or one word repeats >= 50 times in a row, in the
reasoning + content text. The 8-gram count catches phrase loops that a word-run or 4-gram scan on the content alone
misses (a looping think block leaves the content empty). Character rule (stage-11 review #2): one 16-character
substring occurs >= 1,000 times (overlapping count). It catches loops without whitespace, e.g. ifeval doc 87, where
the Persian pair ZWNJ + "ye" repeats 32,766 times and forms one whitespace "word"."""

import json
import sys
from collections import Counter
from pathlib import Path

W = Path.home() / "glm53-flash-bringup"
R = "workflow_logs/reports_output"
M = "eval_id_glm53-flash-autoport_GLM-5.3-Flash_p300x2/zai-org__GLM-5.3-Flash"
SETS = {
    "smoke ifeval (3 docs, p300x2-1m)": f"tti/{R}/evals/zai-org__GLM-5.3-Flash_p300x2_evals/{M}/samples_ifeval_2026-10-21T15-16-17.207329.jsonl",
    "smoke gpqa (p300x2-1m)": f"tti/{R}/evals/zai-org__GLM-5.3-Flash_p300x2_evals/{M}/samples_gpqa_diamond_cot_zeroshot_2026-10-21T15-51-34.615296.jsonl",
    "release ifeval (109 docs, p300x2-1m)": f"tti/{R}/release/zai-org__GLM-5.3-Flash_p300x2_release/{M}/samples_ifeval_2026-10-21T19-44-35.100275.jsonl",
    "release gpqa CI subset (40, p300x2-1m)": f"tti/{R}/release/zai-org__GLM-5.3-Flash_p300x2_release/{M}/samples_gpqa_diamond_cot_zeroshot_2026-10-21T20-17-50.492694.jsonl",
    "gpqa full (198, p300x2)": "artifacts/s11/gpqa_full_p300x2/samples_gpqa_diamond_cot_zeroshot_2026-10-22T07-54-53.999404.jsonl",
    "H1 attempt 1 (stage-7 policy, invalid run)": f"tti-p300x2/{R}/evals/zai-org__GLM-5.3-Flash_p300x2_evals/{M}/samples_gpqa_diamond_cot_zeroshot_2026-10-22T16-44-28.617498.jsonl",
    "H1 chemistry (93, stage-7 policy)": f"tti-p300x2/{R}/evals/zai-org__GLM-5.3-Flash_p300x2_evals/{M}/samples_gpqa_diamond_cot_zeroshot_2026-10-22T23-10-29.508867.jsonl",
}


def first(x):
    while isinstance(x, list):
        x = x[0] if x else ""
    return x or ""


def stats(text):
    w = text.split()
    run = best = 1 if w else 0
    for i in range(1, len(w)):
        run = run + 1 if w[i] == w[i - 1] else 1
        best = max(best, run)
    g = Counter(tuple(w[i : i + 8]) for i in range(len(w) - 7))
    top = g.most_common(1)[0] if g else ((), 0)
    c = Counter(text[i : i + 16] for i in range(len(text) - 15))
    ctop = c.most_common(1)[0] if c else ("", 0)
    return {
        "words": len(w),
        "max_word_run": best,
        "max_8gram": top[1],
        "top_8gram": " ".join(top[0])[:80],
        "max_16char": ctop[1],
        "top_16char": ctop[0],
        "loop": best >= 50 or top[1] >= 20 or ctop[1] >= 1000,
        "loop_char_only": not (best >= 50 or top[1] >= 20) and ctop[1] >= 1000,
    }


# not counted in the totals: the engine crashed in this run (121/198 requests failed, RUN_NOTES H1 attempt 1)
EXCLUDE = {"H1 attempt 1 (stage-7 policy, invalid run)"}
STAGE7 = {"H1 chemistry (93, stage-7 policy)"}
out = {
    "rule": "loop = an 8-gram of whitespace words repeated >= 20 times, or one word repeated >= 50 times in a row, "
    "or one 16-character substring repeated >= 1,000 times (overlapping), in reasoning + content",
    "sets": {},
}
tot = {"all valid": [0, 0], "shipped policy": [0, 0]}
for name, rel in SETS.items():
    seen = {}
    for line in open(W / rel):
        r = json.loads(line)
        if r["doc_id"] in seen:
            continue
        text = first(r.get("reasoning_content")) + "\n" + first(r.get("resps"))
        s = stats(text)
        s["content_empty"] = not first(r.get("resps")).strip()
        seen[r["doc_id"]] = s
    loops = {k: v for k, v in seen.items() if v["loop"]}
    out["sets"][name] = {
        "file": rel,
        "generations": len(seen),
        "loops": len(loops),
        "looped_docs": loops,
        "empty_content": sum(v["content_empty"] for v in seen.values()),
    }
    for k in tot:
        if name in EXCLUDE or (k == "shipped policy" and name in STAGE7):
            continue
        tot[k][0] += len(seen)
        tot[k][1] += len(loops)
    print(
        f"{name}: {len(loops)}/{len(seen)} loops "
        + ", ".join(
            f"doc {k} (8-gram x{v['max_8gram']}, run {v['max_word_run']}, 16-char x{v['max_16char']}"
            f"{', char rule only' if v['loop_char_only'] else ''})"
            for k, v in loops.items()
        )
    )
# H2 (card protocol) kept only the last 300 content characters: scan those, and count budget cuts
h2 = json.load(open(W / "artifacts/s11/h2/h2_cardprompt.json"))
h2l = [r["i"] for r in h2 if stats(r.get("content_tail") or "")["loop"]]
h2cut = [r["i"] for r in h2 if r.get("finish") == "length"]
out["sets"]["H2 card protocol (198, content tail only)"] = {
    "generations": len(h2),
    "loops_in_tail": h2l,
    "budget_cuts": h2cut,
}
print(
    f"H2 card protocol: content tails only (300 chars): {len(h2l)} loops in the tails; {len(h2cut)} budget cuts "
    f"(not scannable: no reasoning text saved)"
)
out["total_full_text"] = {k: {"generations": g, "loops": n, "rate": round(n / g, 4)} for k, (g, n) in tot.items()}
for k, (g, n) in tot.items():
    print(
        f"TOTAL {k} (full reasoning + content saved; invalid H1 attempt 1 excluded): {n}/{g} loops = {100 * n / g:.1f} %"
    )
Path(sys.argv[1]).write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
