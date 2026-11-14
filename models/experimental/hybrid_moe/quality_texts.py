# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Build the fixed token sequences for the quality comparison (hybrid vs CPU reference): ~1K tokens each, real text
(not model output) from cached datasets and repo files, chat-formatted where the source is a Q/A pair.
usage: python quality_texts.py <out.json> [train]  (host: HF_HUB=~/.cache/huggingface/hub TT_SRC=image_src)
TOK_REPO: the tokenizer's HF repo (default Qwen/Qwen3.5-397B-A17B-FP8). "train": a disjoint, wider set (other dataset
rows, more languages, other repo files) for building a routing plan; the default set is the held-out one."""

import glob
import json
import os
import sys

import pandas as pd
from transformers import AutoTokenizer

HF = os.environ.get("HF_HUB", "/hf/hub")
TT = os.environ.get("TT_SRC", "/opt/tt-metal/models/autoports")
POC = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get("TOK_REPO", "Qwen/Qwen3.5-397B-A17B-FP8")
tok = AutoTokenizer.from_pretrained(glob.glob(f"{HF}/models--{REPO.replace('/', '--')}/snapshots/*")[0])
TRAIN = len(sys.argv) > 2 and sys.argv[2] == "train"
N = 1000


def snap(repo, pat):
    for r in (repo, f"openai--{repo}"):  # gsm8k is cached as datasets--openai--gsm8k by newer hub clients
        found = sorted(glob.glob(f"{HF}/datasets--{r}/snapshots/*/{pat}"))
        if found:
            return found[0]
    raise FileNotFoundError(f"dataset {repo} ({pat}) not in {HF}: hf download --repo-type dataset ...")


def chat(pairs):
    msgs, ids = [], []
    for q, a in pairs:
        msgs += [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
        ids = tok.encode(tok.apply_chat_template(msgs, tokenize=False, enable_thinking=False), add_special_tokens=False)
        if len(ids) >= N:
            return list(ids)[:N]
    return list(ids)


def raw(text):
    return tok.encode(text, add_special_tokens=False)[:N]


gsm = pd.read_parquet(snap("gsm8k", "main/test-*.parquet"))
mbpp = pd.read_parquet(snap("google-research-datasets--mbpp", "full/test-*.parquet"))
aya = pd.read_parquet(snap("CohereForAI--aya_dataset", "data/train-*.parquet"))
lcb = [json.loads(line) for line in open(snap("livecodebench--code_generation_lite", "test6.jsonl"))]


def aya_pairs(lang, lo=0):
    d = aya[aya.language == lang]
    d = d[d.targets.str.len() > 200]
    return list(zip(d.inputs[lo : lo + 40], d.targets[lo : lo + 40]))


def files(pattern, n):
    paths = sorted(glob.glob(pattern, recursive=True))[:n]
    return raw("\n\n".join(open(f).read() for f in paths))


if TRAIN:
    heval = pd.read_parquet(snap("openai--openai_humaneval", "openai_humaneval/test-*.parquet"))
    ifev = [json.loads(line) for line in open(snap("google--IFEval", "ifeval_input_data.jsonl"))]
    gpqa = pd.read_csv(snap("Idavidrein--gpqa", "gpqa_diamond.csv"))
    seqs = {
        "math_gsm8k_b": chat(list(zip(gsm.question[100:140], gsm.answer[100:140]))),
        "math_gsm8k_c": chat(list(zip(gsm.question[200:240], gsm.answer[200:240]))),
        "code_mbpp_b": chat([(q, c) for q, c in zip(mbpp.text[100:140], mbpp.code[100:140])]),
        "code_humaneval": chat([(p, c) for p, c in zip(heval.prompt[:40], heval.canonical_solution[:40])]),
        "science_gpqa": chat([(q, e) for q, e in zip(gpqa.Question[:40], gpqa.Explanation[:40])]),
        "instructions_ifeval": raw("\n\n".join(d["prompt"] for d in ifev[100:160])),
        "problems_livecodebench_b": raw("\n\n".join(p["question_content"] for p in lcb[20:26])),
        "code_tt_python": files(f"{TT}/qwen_qwen3_5_122b_a10b/tt/m*.py", 3),
        "code_poc_python": files(f"{POC}/hybrid/patch.py", 1),
        "code_c": files(f"{POC}/hybrid/hostk/hostk.c", 1),
        "doc_plan": files(f"{POC}/docs/*.md", 2),
    }
    for lang in ("English", "Simplified Chinese", "French", "Spanish", "German", "Japanese", "Russian", "Arabic",
                 "Hindi", "Portuguese", "Korean", "Italian"):
        seqs["chat_" + lang.split()[-1].lower() + "_b"] = chat(aya_pairs(lang, 100))
    seqs = {k: v for k, v in seqs.items() if len(v) == N}  # drop sources too short for a full sequence
else:
    seqs = {
        "math_gsm8k": chat(list(zip(gsm.question[:40], gsm.answer[:40]))),
        "code_mbpp": chat([(q, c) for q, c in zip(mbpp.text[:40], mbpp.code[:40])]),
        "chat_english": chat(aya_pairs("English")),
        "chat_chinese": chat(aya_pairs("Simplified Chinese")),
        "chat_french": chat(aya_pairs("French")),
        "code_python_file": raw(open(f"{TT}/qwen_qwen3_5_122b_a10b/tt/generator.py").read()),
        "doc_markdown": raw(open(f"{POC}/HANDOFF.md").read()),
        "problems_livecodebench": raw("\n\n".join(p["question_content"] for p in lcb[:6])),
    }
for k, v in seqs.items():
    print(k, len(v), repr(tok.decode(v[:40]))[:120])
json.dump({k: [int(t) for t in v] for k, v in seqs.items()}, open(sys.argv[1], "w"))
