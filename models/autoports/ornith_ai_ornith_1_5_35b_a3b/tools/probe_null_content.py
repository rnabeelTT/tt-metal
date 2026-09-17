# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Is `content: null` an Ornith defect, or vLLM 0.26.0's qwen3 parser doing what it says?

Stage 11's spec tests failed, and after the reasoning-parser fix EVERY remaining failure
was the same thing: a short-max_tokens request returns message.content = null. RUN_NOTES
called that "a genuine serving-contract bug" of this model. This runs vLLM's own parser,
no device and no Ornith code, to find out whose behaviour it actually is.
"""
import os

from transformers import AutoTokenizer
from vllm.reasoning import ReasoningParserManager

W = os.environ["ORNITH_WEIGHTS"]
tok = AutoTokenizer.from_pretrained(W, trust_remote_code=True)
cls = ReasoningParserManager.get_reasoning_parser("qwen3")
print(f"vLLM resolves --reasoning-parser qwen3 to: {cls.__module__}.{cls.__name__}")

CASES = {
    "truncated INSIDE the think block (max_tokens exhausted)": "<think>Let me work through this step by",
    "think block closed, budget gone right after": "<think>Reasoning here.</think>",
    "think block closed, then real answer": "<think>Reasoning here.</think>Paris.",
    "no think block at all": "Paris.",
}

for thinking in (True, False):
    p = cls(tok, None, chat_template_kwargs={"enable_thinking": thinking})
    print(f"\n=== enable_thinking={thinking} ===")
    for label, out in CASES.items():
        r, c = p.extract_reasoning(out, None)
        flag = "  <-- content is NULL" if c is None else ""
        print(f"  {label}\n      reasoning={r!r}\n      content  ={c!r}{flag}")
