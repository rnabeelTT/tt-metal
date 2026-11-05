# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Build the fixed token sequences for the quality comparison (hybrid vs CPU reference): ~1K tokens each, real text
(not model output) from cached datasets and repo files, chat-formatted where the source is a Q/A pair.
usage: python quality_texts.py <out.json>  (host: HF_HUB=~/.cache/huggingface/hub TT_SRC=image_src)"""

import glob
import json
import os
import sys

import pandas as pd
from transformers import AutoTokenizer

HF = os.environ.get("HF_HUB", "/hf/hub")
TT = os.environ.get("TT_SRC", "/opt/tt-metal/models/autoports")
POC = os.path.dirname(os.path.abspath(__file__))
tok = AutoTokenizer.from_pretrained(glob.glob(f"{HF}/models--Qwen--Qwen3.5-397B-A17B-FP8/snapshots/*")[0])
N = 1000


def snap(repo, pat):
    return sorted(glob.glob(f"{HF}/datasets--{repo}/snapshots/*/{pat}"))[0]


def chat(pairs):
    msgs = []
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


def aya_pairs(lang):
    d = aya[aya.language == lang]
    d = d[d.targets.str.len() > 200]
    return list(zip(d.inputs[:40], d.targets[:40]))


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
