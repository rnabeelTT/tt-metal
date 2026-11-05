# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Accuracy benchmarks against the running server (OpenAI chat API, greedy): IFEval (scored with lm-eval's ifeval
checkers), GSM8K (boxed / last number) and GPQA-Diamond (letter). Resumable: responses append to <out>.jsonl.

usage: python serve/evalq.py <out prefix> <task:n[:think]> ...   e.g.  runs/eval/s15 ifeval:541 gsm8k:400 gpqa:198:think
env: PORT (8000), HF_HUB (~/.cache/huggingface/hub), MAX_TOKENS (4096; thinking 32768)
"""

import csv
import glob
import json
import os
import random
import re
import sys
import time
import urllib.request

HUB = os.environ.get("HF_HUB", os.path.expanduser("~/.cache/huggingface/hub"))
URL = f"http://localhost:{os.environ.get('PORT', '8000')}/v1/chat/completions"


def snap(repo, pat):
    return sorted(glob.glob(f"{HUB}/datasets--{repo}/snapshots/*/{pat}"))[0]


def chat(prompt, think):
    body = {
        "model": "qwen3.5-397b-hybrid",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": int(os.environ.get("MAX_TOKENS", 32768 if think else 4096)),
        "chat_template_kwargs": {"enable_thinking": think},
    }
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=7200))
    m = r["choices"][0]
    return m["message"].get("content") or "", r["usage"]["completion_tokens"], m["finish_reason"]


def ifeval_docs():
    return [json.loads(line) for line in open(snap("google--IFEval", "ifeval_input_data.jsonl"))]


def gsm_docs():
    import pandas as pd

    d = pd.read_parquet(snap("gsm8k", "main/test-*.parquet"))
    return [{"q": q, "a": a.split("####")[-1].strip().replace(",", "")} for q, a in zip(d.question, d.answer)]


def gpqa_docs():
    rng = random.Random(0)
    out = []
    for row in csv.DictReader(open(snap("Idavidrein--gpqa", "gpqa_diamond.csv"))):
        opts = [row["Correct Answer"]] + [row[f"Incorrect Answer {i}"] for i in (1, 2, 3)]
        opts = [o.strip() for o in opts]
        order = list(range(4))
        rng.shuffle(order)
        out.append({"q": row["Question"].strip(), "opts": [opts[i] for i in order], "gold": "ABCD"[order.index(0)]})
    return out


def prompt_of(task, d):
    if task == "ifeval":
        return d["prompt"]
    if task == "gsm8k":
        return d["q"] + "\nPlease reason step by step, and put your final answer within \\boxed{}."
    opts = "\n".join(f"{L}) {o}" for L, o in zip("ABCD", d["opts"]))
    return (
        f"{d['q']}\n\n{opts}\n\nThink step by step, then give your final answer on the last line in the format "
        "'Answer: X' where X is A, B, C or D."
    )


def score(task, d, resp):
    if task == "ifeval":
        from lm_eval.tasks.ifeval import utils

        doc = {
            "key": d["key"],
            "prompt": d["prompt"],
            "instruction_id_list": d["instruction_id_list"],
            "kwargs": [{k: v for k, v in kw.items() if v is not None} for kw in d["kwargs"]],
        }
        r = utils.process_results(doc, [resp])
        return {
            "strict": float(r["prompt_level_strict_acc"]),
            "loose": float(r["prompt_level_loose_acc"]),
            "inst_ok": sum(r["inst_level_strict_acc"]),
            "inst_n": len(r["inst_level_strict_acc"]),
        }
    if task == "gsm8k":
        b = re.findall(r"\\boxed\{([^{}]*)\}", resp)
        cand = b[-1] if b else (re.findall(r"-?\d[\d,]*\.?\d*", resp) or [""])[-1]
        num = re.sub(r"[^\d.\-]", "", cand.replace(",", "")).rstrip(".")
        try:
            return {"acc": float(abs(float(num) - float(d["a"])) < 1e-6)}
        except ValueError:
            return {"acc": 0.0}
    m = re.findall(r"Answer:\s*\(?([ABCD])\)?", resp)
    return {"acc": float(bool(m) and m[-1] == d["gold"])}


def main():
    prefix = sys.argv[1]
    os.makedirs(os.path.dirname(prefix) or ".", exist_ok=True)
    summary = {}
    for spec in sys.argv[2:]:
        parts = spec.split(":")
        task, n, think = parts[0], int(parts[1]), len(parts) > 2 and parts[2] == "think"
        docs = {"ifeval": ifeval_docs, "gsm8k": gsm_docs, "gpqa": gpqa_docs}[task]()[:n]
        tag = f"{task}{'_think' if think else ''}"
        path = f"{prefix}_{tag}.jsonl"
        done = [json.loads(line) for line in open(path)] if os.path.exists(path) else []
        with open(path, "a") as f:
            for i in range(len(done), len(docs)):
                t0 = time.time()
                resp, ntok, fin = chat(prompt_of(task, docs[i]), think)
                rec = {
                    "i": i,
                    "resp": resp,
                    "tokens": ntok,
                    "finish": fin,
                    "s": round(time.time() - t0, 1),
                    **score(task, docs[i], resp),
                }
                f.write(json.dumps(rec) + "\n")
                f.flush()
                done.append(rec)
                if i % 20 == 0:
                    print(
                        f"{tag} {i + 1}/{len(docs)} {json.dumps({k: v for k, v in rec.items() if k != 'resp'})}",
                        flush=True,
                    )
        keys = [k for k in done[0] if k not in ("i", "resp", "tokens", "finish", "s", "inst_ok", "inst_n")]
        extra = (
            {"inst_strict": round(sum(r["inst_ok"] for r in done) / sum(r["inst_n"] for r in done), 4)}
            if task == "ifeval"
            else {}
        )
        summary[tag] = {
            "n": len(done),
            **{k: round(sum(r[k] for r in done) / len(done), 4) for k in keys},
            **extra,
            "mean_tokens": round(sum(r["tokens"] for r in done) / len(done)),
            "truncated": sum(r["finish"] == "length" for r in done),
        }
        print(f"RESULT {tag} {json.dumps(summary[tag])}", flush=True)
    json.dump(summary, open(f"{prefix}_summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
