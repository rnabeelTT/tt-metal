# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Chat prompts for the reference-continuation agreement test (8 prompts incl. AIME 2024 I P1, one with thinking).

    python gen_prompts_hy4.py OUT.json        then: REF_MAX_NEW=100 python ref_hy4.py gen OUT.json <dir>
"""
import json
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('tencent/Hy4-flash', revision='b2498eba6a1a3f9bbcab4190bc1313a3c7010c2f')
AIME = ("Every morning Aya goes for a $9$-kilometer-long walk and stops at a coffee shop afterwards. When she walks at a "
        "constant speed of $s$ kilometers per hour, the walk takes her 4 hours, including $t$ minutes spent in the coffee "
        "shop. When she walks $s+2$ kilometers per hour, the walk takes her 2 hours and 24 minutes, including $t$ minutes "
        "spent in the coffee shop. Suppose Aya walks at $s+\\frac{1}{2}$ kilometers per hour. Find the number of minutes "
        "the walk takes her, including the $t$ minutes spent in the coffee shop.")
P = {
    "aime24": (AIME + "\nPlease reason step by step, and put your final answer within \\boxed{}.", "no_think"),
    "aime24_think": (AIME + "\nPlease reason step by step, and put your final answer within \\boxed{}.", "high"),
    "intro": ("Hello! Can you introduce yourself and describe what you can help with?", "no_think"),
    "cache": ("Explain how a CPU cache hierarchy works, in a few paragraphs.", "no_think"),
    "code": ("Write a Python function that returns the n-th Fibonacci number using memoization, and explain it.", "no_think"),
    "zh": ("请详细介绍一下长城的历史。", "no_think"),
    "fr": ("Écris un court essai sur l'histoire de Paris.", "no_think"),
    "tcp": ("Compare TCP and UDP, with examples of where each is used.", "no_think"),
}
out = {}
for k, (msg, eff) in P.items():
    ids = tok.apply_chat_template([{"role": "user", "content": msg}], add_generation_prompt=True, reasoning_effort=eff, tokenize=True)
    ids = ids if isinstance(ids, list) else ids["input_ids"]
    out[k] = ids
    print(k, len(ids), repr(tok.decode(ids[-12:])))
import sys
json.dump(out, open(sys.argv[1], "w"))
