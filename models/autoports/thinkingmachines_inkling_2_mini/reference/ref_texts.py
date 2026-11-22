# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU reference for the teacher-forced quality gate: for each held-out text (models/experimental/hybrid_moe/
quality_texts.py, Inkling-2 tokenizer) one forward, saving per position the reference's top-32 token ids and
log-probs (full-vocab log_softmax) and the next-token NLL.

usage (host venv): ~/inkling2-work/venv/bin/python reference/ref_texts.py texts.json out.pt [dtype bf16|fp32]
"""
import json
import sys
import time

import torch
from transformers import Inkling2LoopForConditionalGeneration


def main():
    texts = json.load(open(sys.argv[1]))
    dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[sys.argv[3] if len(sys.argv) > 3 else "bf16"]
    torch.set_num_threads(8)
    m = Inkling2LoopForConditionalGeneration.from_pretrained("thinkingmachines/Inkling-2-Mini", dtype=dtype).eval()
    out = {}
    for name, ids in texts.items():
        t0 = time.time()
        ids = torch.tensor(ids)
        with torch.no_grad():
            lg = m(input_ids=ids[None], use_cache=False).logits[0].float()
        lp = torch.log_softmax(lg, -1)
        top_lp, top_ids = lp.topk(32, -1)
        nll = -lp[:-1].gather(-1, ids[1:, None])[:, 0]
        out[name] = {"ids": ids, "top_ids": top_ids, "top_lp": top_lp, "nll": nll}
        print(f"{name}: nll {nll.mean():.3f} [{time.time() - t0:.0f}s]", flush=True)
    torch.save(out, sys.argv[2])


if __name__ == "__main__":
    main()
