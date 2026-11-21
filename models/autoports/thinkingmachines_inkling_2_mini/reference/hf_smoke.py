# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU golden smoke test: load Inkling-2-Mini (text) with transformers >= 5.24 in bf16 and greedy-generate."""
import sys, time
import torch
from transformers import AutoTokenizer, Inkling2LoopForConditionalGeneration

REPO = "thinkingmachines/Inkling-2-Mini"
torch.set_num_threads(8)
t = time.time()
tok = AutoTokenizer.from_pretrained(REPO)
model = Inkling2LoopForConditionalGeneration.from_pretrained(REPO, dtype=torch.bfloat16)
model.eval()
print(type(model).__name__, f"load {time.time()-t:.0f}s", flush=True)
msgs = [{"role": "user", "content": sys.argv[1] if len(sys.argv) > 1 else "What is 84 * 3 / 2? Answer briefly."}]
ids = tok.apply_chat_template(msgs, add_generation_prompt=True, reasoning_effort="none", return_tensors="pt", return_dict=True)
print(repr(tok.decode(ids["input_ids"][0])))
t = time.time()
with torch.no_grad():
    out = model.generate(**ids, max_new_tokens=40, do_sample=False)
n = out.shape[1] - ids["input_ids"].shape[1]
print(f"{n} tokens {time.time()-t:.1f}s")
print(repr(tok.decode(out[0, ids["input_ids"].shape[1]:])))
