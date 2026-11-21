# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Golden fixtures from the transformers reference (CPU, bf16 weights as transformers loads them).

Runs one teacher-forced forward over a chat-formatted prompt + continuation and saves:
  ids            [T] int64
  hidden         [49, T, H] bf16: embeddings (index 0) and the output of each of the 48 layer applications
  logits         [T, V] fp32 (after the final softcap)
  internals      {name: [calls x tensor]} first-call outputs of selected submodules (layer 0, 5, 6, 11), for debugging

usage (host venv with transformers >= 5.24):
  ~/inkling2-work/venv/bin/python reference/golden.py OUT.pt [T]
"""
import sys
import time

import torch
from transformers import AutoTokenizer, Inkling2LoopForConditionalGeneration

REPO = "thinkingmachines/Inkling-2-Mini"
TEXT = (
    "The history of the printing press begins in the fifteenth century, when Johannes Gutenberg combined movable "
    "metal type, oil-based inks and a wooden screw press adapted from wine making. Within fifty years presses had "
    "spread to more than two hundred cities across Europe, and the price of books fell sharply. Historians argue "
    "about how much of the Reformation, the Scientific Revolution and the rise of vernacular literature can be "
    "credited to cheap print, but almost all agree that the speed at which ideas could travel changed for good. "
    "Below we look at three consequences in turn: standardisation of texts, the growth of literacy, and the new "
    "economics of authorship. Each of these unfolded differently in different regions, and the evidence for each "
    "comes from very different kinds of sources, from printers' account books to marginal notes in surviving copies."
)


def main():
    out = sys.argv[1]
    T = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    torch.set_num_threads(8)
    tok = AutoTokenizer.from_pretrained(REPO)
    model = Inkling2LoopForConditionalGeneration.from_pretrained(REPO, dtype=torch.bfloat16).eval()
    msgs = [{"role": "user", "content": "Summarise the following passage in two sentences.\n\n" + TEXT}]
    prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, reasoning_effort="none", tokenize=False)
    ids = tok(prompt + "<|message_model|><|content_text|>" + TEXT, return_tensors="pt", add_special_tokens=False)[
        "input_ids"
    ][0]
    if T:
        ids = ids[:T]
    T = ids.shape[0]
    lm = model.model.language_model
    internals = {}

    def hook(name):
        def f(mod, args, kwargs, output):
            o = output[0] if isinstance(output, tuple) else output
            if torch.is_tensor(o):
                internals.setdefault(name, []).append(o.detach().float().clone())

        return f

    for li in (0, 5, 6, 11):
        layer = lm.layers[li]
        for n, m in layer.named_modules():
            if n and n.count(".") <= 2:
                m.register_forward_hook(hook(f"layers.{li}.{n}"), with_kwargs=True)
        layer.register_forward_hook(hook(f"layers.{li}"), with_kwargs=True)
    lm.message_input.register_forward_hook(hook("message_input"), with_kwargs=True)
    t0 = time.time()
    with torch.no_grad():
        o = model(input_ids=ids[None], output_hidden_states=True, use_cache=False)
    print(f"T={T} forward {time.time() - t0:.1f}s, hidden states {len(o.hidden_states)}")
    hidden = torch.stack([h[0].to(torch.bfloat16) for h in o.hidden_states])
    torch.save(
        {"ids": ids, "hidden": hidden, "logits": o.logits[0].float(), "internals": internals, "prompt_len": None},
        out,
    )
    print("saved", out, {k: len(v) for k, v in list(internals.items())[:8]})


if __name__ == "__main__":
    main()
