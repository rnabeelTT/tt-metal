# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""DFlash acceptance vs drafter context window, offline on a saved context (run_dflash_traced.py DFLASH_SAVE_CTX=1:
the target's tap features of every position and the greedy tokens). At each decode position p: block [token p, mask..]
with the context = the features of the last w positions (their true positions); drafts accepted while they match the
target's own next tokens. Prints the acceptance length (1 + accepted, k = 3) per window.
usage: python window_oracle.py <ctx.pt> [windows, default 128,512,2048,full] [positions to sample, default 40]"""

import glob
import os
import sys

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import Qwen3Config

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ref_m27 as R  # noqa: E402
from dflash_ref import DFlashDraftModel  # noqa: E402


@torch.no_grad()
def main():
    d = torch.load(sys.argv[1])
    wins = [
        int(w) if w != "full" else 0 for w in (sys.argv[2] if len(sys.argv) > 2 else "128,512,2048,full").split(",")
    ]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 40
    torch.set_num_threads(int(os.environ.get("REF_THREADS", "12")))
    snap = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--nvidia--MiniMax-M2.7-DFlash/snapshots/*"))[0]
    cfg = Qwen3Config.from_pretrained(snap)
    cfg._attn_implementation = "eager"
    m = DFlashDraftModel(cfg).to(torch.bfloat16).eval()
    m.load_state_dict(load_file(os.path.join(snap, "model.safetensors")), strict=False)
    embed, head = R.get("model.embed_tokens.weight"), R.get("lm_head.weight").float()
    feats, toks, P = d["feats"], d["tokens"], d["P"]
    T = min(feats.shape[0], len(toks) - 8)
    pos = np.linspace(P, T - 1, n).astype(int)
    for w in wins:
        A = []
        for p in pos:
            lo = 0 if w == 0 else max(0, p - w)
            blk = torch.tensor([toks[p]] + [m.mask_token_id] * (m.block_size - 1))
            h = m(
                position_ids=torch.arange(lo, p + m.block_size)[None],
                noise_embedding=torch.nn.functional.embedding(blk, embed)[None],
                target_hidden=feats[lo:p].to(torch.bfloat16)[None],
            )
            dr = (h[0, 1:4].float() @ head.T).argmax(-1).tolist()
            a = 0
            while a < 3 and dr[a] == toks[p + 1 + a]:
                a += 1
            A.append(1 + a)
        print(f"window {w or 'full':>5}: AL(k=3) {np.mean(A):.2f} over {len(A)} positions", flush=True)


if __name__ == "__main__":
    main()
