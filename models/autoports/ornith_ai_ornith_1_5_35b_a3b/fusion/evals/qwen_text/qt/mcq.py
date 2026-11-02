"""Multiple-choice benchmarks. Every item is a single zero-shot user turn:
question, lettered options, then the card's MCQ format sentence (or the
benchmark's official generative template where one exists). Scored on the
letter extracted from `content` only."""
import ast
import glob
import json
import math
import os
import random

import pandas as pd

from .common import (LETTERS, MCQ_JSON_INSTR, extract_choice, mcq_block, raw,
                     simple_sample, stratified_sample)


def _clean(x):
    if x is None:
        return None
    if isinstance(x, float) and math.isnan(x):
        return None
    s = str(x)
    return None if s.strip() in ("", "nan", "None", "N/A") else s


def mcq_prompt(question, options, instr=MCQ_JSON_INSTR, letters=LETTERS):
    return f"{question.strip()}\n\n{mcq_block(options, letters)}\n\n{instr}"


def item(id_, group, question, options, gold_idx, meta=None, prompt=None):
    return {"id": id_, "group": group, "n_options": len(options), "gold": LETTERS[gold_idx],
            "messages": [{"role": "user", "content": prompt or mcq_prompt(question, options)}],
            "meta": meta or {}}


# ---------------- per-benchmark loaders ----------------

def load_mmlu_pro():
    df = pd.read_parquet(raw("TIGER-Lab__MMLU-Pro", "data", "test-00000-of-00001.parquet"))
    out = []
    for r in df.itertuples():
        opts = [str(o) for o in r.options]
        out.append(item(f"mmlupro-{int(r.question_id):05d}", r.category, r.question, opts,
                        int(r.answer_index), {"category": r.category, "src": r.src}))
    return out


def load_mmlu_redux():
    """MMLU-Redux 2.0 (57 subjects x 100 = 5,700), relabelled exactly as evalscope's
    mmlu_redux adapter (v1.1) does: no_correct_answer -> the annotated correct
    answer replaces the gold option's text; wrong_groundtruth -> re-target to the
    annotated answer; multiple_correct_answers -> any listed option counts; other
    error types keep the original label. All 5,700 items are kept."""
    import datasets
    out = []
    for d in sorted(glob.glob(raw("edinburgh-dawg__mmlu-redux-2.0", "*", "dataset_info.json"))):
        subj = os.path.basename(os.path.dirname(d))
        ds = datasets.load_from_disk(os.path.dirname(d))
        for i, r in enumerate(ds):
            et, choices, ca = r["error_type"], list(r["choices"]), r["correct_answer"]
            if isinstance(ca, float) and not math.isnan(ca):   # some subjects store it as a number
                ca = str(int(ca))
            elif isinstance(ca, int):
                ca = str(ca)
            elif isinstance(ca, float):
                ca = None
            tgt = [int(r["answer"])]
            if et == "no_correct_answer" and ca:
                choices[tgt[0]] = ca
            elif et == "wrong_groundtruth" and ca:
                try:
                    tgt = [int(ca)]
                except ValueError:
                    tgt = [ord(ca) - ord("A")]
            elif et == "multiple_correct_answers" and ca:
                c = ca.strip("()")
                try:
                    c = c.replace(" and ", ",").replace(" or ", ",")
                    tgt = list(map(int, c.replace(",", " ").split()))
                except ValueError:
                    try:
                        tgt = [ord(x) - ord("A") for x in c.split(",")]
                    except TypeError:
                        tgt = [choices.index(x) for x in c.split(",") if x in choices]
            gold = "".join("ABCD"[k] for k in tgt if 0 <= k < 4) or "ABCD"
            it = item(f"redux-{subj}-{i:03d}", subj, r["question"], choices, 0, {"subject": subj, "error_type": et})
            it["gold"] = gold
            out.append(it)
    return out


def load_ceval():
    """C-Eval validation split (1,346 items, 52 subjects): the test split's labels
    are withheld (scored only via the cevalbenchmark.com leaderboard)."""
    out = []
    for f in sorted(glob.glob(raw("ceval__ceval-exam", "*", "val-*.parquet"))):
        subj = os.path.basename(os.path.dirname(f))
        for r in pd.read_parquet(f).itertuples():
            opts = [r.A, r.B, r.C, r.D]
            out.append(item(f"ceval-{subj}-{int(r.id):03d}", subj, r.question, [str(o) for o in opts],
                            "ABCD".index(r.answer), {"subject": subj}))
    return out


def load_supergpqa():
    out = []
    with open(raw("m-a-p__SuperGPQA", "SuperGPQA-all.jsonl")) as f:
        for l in f:
            r = json.loads(l)
            out.append(item(f"sgpqa-{r['uuid']}", r["discipline"], r["question"], r["options"],
                            LETTERS.index(r["answer_letter"]),
                            {"discipline": r["discipline"], "field": r["field"], "difficulty": r["difficulty"]}))
    return out


def load_mmmlu():
    """MMMLU as in the Qwen3 report: 14 languages = English (original MMLU test) +
    the 13 OpenAI translations, Yoruba excluded."""
    out = []
    for f in sorted(glob.glob(raw("openai__MMMLU", "test", "mmlu_*.csv"))):
        lang = os.path.basename(f)[5:-4]
        if lang == "YO-NG":
            continue
        df = pd.read_csv(f, keep_default_na=False, dtype=str)
        for idx, r in enumerate(df.itertuples(index=False)):   # row order = original MMLU test order
            opts = [r.A, r.B, r.C, r.D]
            out.append(item(f"mmmlu-{lang}-{idx:05d}", lang, r.Question, opts, "ABCD".index(r.Answer.strip()),
                            {"lang": lang, "subject": r.Subject}))
    en = pd.read_parquet(raw("cais__mmlu", "all", "test-00000-of-00001.parquet"))
    for idx, r in enumerate(en.itertuples(index=False)):
        out.append(item(f"mmmlu-EN-US-{idx:05d}", "EN-US", r.question, [str(c) for c in r.choices], int(r.answer),
                        {"lang": "EN-US", "subject": r.subject}))
    return out


def load_mmlu_prox():
    out = []
    for f in sorted(glob.glob(raw("li-lab__MMLU-ProX", "*", "test-*.parquet"))):
        lang = os.path.basename(os.path.dirname(f))
        for r in pd.read_parquet(f).itertuples():
            opts = [_clean(getattr(r, f"option_{k}")) for k in range(10)]
            opts = [o for o in opts if o is not None]
            out.append(item(f"prox-{lang}-{int(r.question_id):05d}", lang, r.question, opts, int(r.answer_index),
                            {"lang": lang, "category": r.category}))
    return out


def load_include():
    out = []
    for f in sorted(glob.glob(raw("CohereLabs__include-base-44", "*", "test-*.parquet"))):
        lang = os.path.basename(os.path.dirname(f))
        for i, r in enumerate(pd.read_parquet(f).itertuples()):
            opts = [r.option_a, r.option_b, r.option_c, r.option_d]
            out.append(item(f"include-{lang}-{i:05d}", lang, r.question, [str(o) for o in opts], int(r.answer),
                            {"lang": lang, "country": r.country, "domain": r.domain, "subject": r.subject}))
    return out


# Global PIQA's official "generation" format for instruction-tuned models
# (dataset card -> lm-evaluation-harness tasks/global_piqa/prompted/_template).
GPIQA_TMPL = ("Given the following situation, which option is more likely to be correct?\n\nSituation:\n{p} ...\n\n"
              "Option A: {a}\n\nOption B: {b}\n\nYour response should end with \"The best answer is: [answer_letter]\" "
              "where [answer_letter] is one of A or B.")


def load_global_piqa():
    out = []
    for f in sorted(glob.glob(raw("mrlbenchmarks__global-piqa-nonparallel", "data", "nonparallel_*.tsv"))):
        lang = os.path.basename(f)[len("nonparallel_"):-4]
        if lang.startswith("dev"):
            continue
        df = pd.read_csv(f, sep="\t", keep_default_na=False, dtype=str, quoting=3)
        for r in df.itertuples(index=False):
            p = GPIQA_TMPL.format(p=r.prompt, a=r.solution0, b=r.solution1)
            out.append(item(f"gpiqa-{r.example_id}", lang, r.prompt, [r.solution0, r.solution1], int(r.label),
                            {"lang": lang}, prompt=p))
    return out


# ---------------- registry ----------------
# sample: None = full set; "flat" = 1,000 uniformly; "group" = 1,000 spread evenly over item["group"]
BENCHES = {
    "mmlu_pro":    dict(load=load_mmlu_pro, sample="flat", avg="micro"),
    "mmlu_redux":  dict(load=load_mmlu_redux, sample="group", avg="micro"),
    "ceval":       dict(load=load_ceval, sample=None, avg="micro"),
    "supergpqa":   dict(load=load_supergpqa, sample="flat", avg="micro"),
    "mmmlu":       dict(load=load_mmmlu, sample="group", avg="macro"),
    "mmlu_prox":   dict(load=load_mmlu_prox, sample="group", avg="macro"),
    "include":     dict(load=load_include, sample="group", avg="macro"),
    "global_piqa": dict(load=load_global_piqa, sample="group", avg="macro"),
}


def build(name):
    spec = BENCHES[name]
    items = spec["load"]()
    full = len(items)
    if spec["sample"] == "flat":
        items = simple_sample(items)
    elif spec["sample"] == "group":
        items = stratified_sample(items, key=lambda x: x["group"])
    return items, {"full_size": full, "sample": spec["sample"], "avg": spec["avg"]}


_GPIQA_RX = None


def score(name, it, content):
    pred, how = extract_choice(content, it["n_options"])
    if name == "global_piqa" and how not in ("json", "boxed"):
        # the official template asks for "The best answer is: X"; use the harness regex
        import re
        global _GPIQA_RX
        _GPIQA_RX = _GPIQA_RX or re.compile(
            r"[Tt]he (?:[Bb]est [Aa]nswer|[Ff]inal [Aa]nswer|[Aa]nswer)[^A-B]*([A-B])|[Aa]nswer\s*:[^A-B]*([A-B])|\\boxed\{([A-B])\}")
        ms = [next(g for g in m.groups() if g) for m in _GPIQA_RX.finditer(content or "")]
        pred, how = (ms[-1], "official_regex") if ms else (None, "none")
    return {"pred": pred, "how": how, "correct": float(pred is not None and pred in it["gold"])}


def aggregate(name, meta, rows):
    """rows: [{'item':..., 'score':...}]. micro = mean over items; macro = mean of
    per-group means (per-language average, as the multilingual cards report)."""
    per = {}
    for r in rows:
        per.setdefault(r["item"]["group"], []).append(r["score"]["correct"])
    micro = sum(sum(v) for v in per.values()) / max(1, sum(len(v) for v in per.values()))
    macro = sum(sum(v) / len(v) for v in per.values()) / max(1, len(per))
    head = macro if meta["avg"] == "macro" else micro
    return {"metric": "acc", "value": 100 * head, "micro_acc": 100 * micro, "macro_acc": 100 * macro,
            "per_group": {g: round(100 * sum(v) / len(v), 2) for g, v in sorted(per.items())}}


def gold_reply(name, it):
    if name == "global_piqa":
        return f"Option {it['gold'][0]} fits the situation.\n\nThe best answer is: {it['gold'][0]}"
    return f'{{"answer": "{it["gold"][0]}"}}'


def random_reply(name, it, rng):
    L = rng.choice(LETTERS[:it["n_options"]])
    if name == "global_piqa":
        return f"The best answer is: {L}"
    return f'{{"answer": "{L}"}}'


def chance(name, items):
    return 100 * sum(len(it["gold"]) / it["n_options"] for it in items) / len(items)
