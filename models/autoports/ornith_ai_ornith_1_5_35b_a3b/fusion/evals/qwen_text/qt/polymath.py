"""PolyMATH (Qwen/PolyMath): 18 languages x 4 levels x 125 = 9,000. Prompt and
scorer are the official repo's (QwenLM/PolyMath): question + "\\n" + the
per-language boxed-answer note from instruction.py; first \\boxed{} of the answer,
math_equal; per language DW-ACC = (1*low + 2*medium + 4*high + 8*top)/15;
headline = mean over the 18 languages."""
import ast
import glob
import json
import os
import random
import subprocess

import pandas as pd

from .common import HERE, raw, stratified_sample

REPO = os.path.expanduser("~/evals/src/PolyMath")
LEVELS = ["low", "medium", "high", "top"]


def _query_dic():
    tree = ast.parse(open(os.path.join(REPO, "instruction.py"), encoding="utf-8").read())
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) == "query_dic":
            return ast.literal_eval(node.value)
    raise RuntimeError("query_dic not found in PolyMath/instruction.py")


def build(bench):
    q = _query_dic()
    items = []
    for f in sorted(glob.glob(raw("Qwen__PolyMath", "*", "*.parquet"))):
        lang = os.path.basename(os.path.dirname(f))
        level = os.path.basename(f)[:-8]
        for r in pd.read_parquet(f).itertuples():
            items.append({"id": f"pm-{lang}-{level}-{r.id}", "group": lang, "gold": str(r.answer),
                          "messages": [{"role": "user", "content": r.question + "\n" + q[lang]}],
                          "meta": {"lang": lang, "level": level}})
    full = len(items)
    items = stratified_sample(items, key=lambda x: (x["group"], x["meta"]["level"]))
    return items, {"full_size": full, "sample": "group(lang x level)", "avg": "dw-acc, mean over languages"}


def score_batch(bench, rows, out, log):
    py = os.path.join(HERE, ".venv-polymath", "bin", "python")
    inp = "".join(json.dumps({"id": r["item"]["id"], "content": r["content"], "answer": r["item"]["gold"]},
                             ensure_ascii=False) + "\n" for r in rows)
    p = subprocess.run([py, "-W", "ignore", os.path.join(HERE, "qt", "polymath_score.py"), os.path.join(REPO, "eval")],
                       input=inp, capture_output=True, text=True, cwd=os.path.join(REPO, "eval"))
    if p.returncode != 0:
        raise RuntimeError("polymath scorer failed: " + p.stderr[-2000:])
    res = {d["id"]: d for d in map(json.loads, p.stdout.splitlines())}
    return [{"pred": res[r["item"]["id"]]["extracted"], "correct": float(res[r["item"]["id"]]["correct"]),
             "correct_official_raw": float(res[r["item"]["id"]]["correct_official_raw"])} for r in rows]


def aggregate(bench, meta, rows):
    cell = {}
    for r in rows:
        k = (r["item"]["group"], r["item"]["meta"]["level"])
        cell.setdefault(k, []).append(r["score"]["correct"])
    langs = sorted({k[0] for k in cell})
    per_lang, plain = {}, []
    for lg in langs:
        accs = [100 * sum(cell[(lg, lv)]) / len(cell[(lg, lv)]) if (lg, lv) in cell else None for lv in LEVELS]
        if None in accs:   # LIMIT runs may miss a level: weight over the levels present
            w = [(2 ** i, a) for i, a in enumerate(accs) if a is not None]
            per_lang[lg] = sum(x * a for x, a in w) / sum(x for x, _ in w)
        else:
            per_lang[lg] = sum((2 ** i) * a for i, a in enumerate(accs)) / 15
    lvl = {lv: 100 * sum(sum(v) for (g, l), v in cell.items() if l == lv) /
           max(1, sum(len(v) for (g, l), v in cell.items() if l == lv)) for lv in LEVELS}
    acc = 100 * sum(r["score"]["correct"] for r in rows) / max(1, len(rows))
    raw_acc = 100 * sum(r["score"]["correct_official_raw"] for r in rows) / max(1, len(rows))
    return {"metric": "dw_acc", "value": sum(per_lang.values()) / max(1, len(per_lang)), "plain_acc": acc, "plain_acc_official_raw": raw_acc,
            "acc_by_level": {k: round(v, 2) for k, v in lvl.items()},
            "per_group": {k: round(v, 2) for k, v in per_lang.items()}, "result_extra": ["plain_acc", "plain_acc_official_raw"]}


def gold_reply(bench, it):
    g = it["gold"].strip().strip("$").strip()   # gold as a model would box it (no $ delimiters)
    return f"The final answer is $\\boxed{{{g}}}$."


def random_reply(bench, it, rng):
    return f"So the answer is $\\boxed{{{rng.randint(0, 10**6)}}}$."
