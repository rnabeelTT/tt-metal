# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""CPU acceptance check for a DFlash drafter (dflash_ref.DFlashDraftModel, the z-lab reference) on MiniMax-M2.7:
the target's own per-layer features (ref_m27.py REF_TAPS taps.pt) over prompt + generated text, then at every generated
position p: block [token p, mask x (bs-1)] at positions p.., context = the taps of tokens 0..p-1; the drafts are the
target LM head's argmax over the drafter's rows 1..bs-1. A draft is accepted while it equals the reference's own greedy
next token (ref_<name>.npz top_ids). Acceptance length (AL) = 1 + accepted drafts, per draft length k.

usage: python dflash_oracle.py <drafter snapshot> <seqs.json> <prompt_lens.json> <ref dir (taps.pt, ref_*.npz)>
"""

import json
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
    snap, seqs_p, plen_p, ref_dir = sys.argv[1:5]
    torch.set_num_threads(int(os.environ.get("REF_THREADS", "12")))
    cfg = Qwen3Config.from_pretrained(snap)
    cfg._attn_implementation = "eager"
    model = DFlashDraftModel(cfg).to(torch.bfloat16).eval()
    missing, unexpected = model.load_state_dict(load_file(os.path.join(snap, "model.safetensors")), strict=False)
    missing = [m for m in missing if "rotary" not in m]
    if missing or unexpected:
        raise RuntimeError(f"drafter weights: missing {missing} unexpected {unexpected}")
    bs, mask_id = model.block_size, model.mask_token_id
    embed = R.get("model.embed_tokens.weight")
    head = R.get("lm_head.weight")
    tp = torch.load(os.path.join(ref_dir, "taps.pt"))
    seqs, plens = json.load(open(seqs_p)), json.load(open(plen_p))
    ks = (1, 3, 5, 7)
    allA = {k: [] for k in ks}
    for b, name in enumerate(tp["names"]):
        seq = seqs[name]
        T = len(seq)
        truth = np.load(os.path.join(ref_dir, f"ref_{name}.npz"))["top_ids"][:, 0]  # truth[i]: greedy token after i
        feats = tp["taps"][b].reshape(T, -1)  # [T, 5 H]
        A = {k: [] for k in ks}
        for p in range(plens[name], T - bs):
            block = torch.tensor([seq[p]] + [mask_id] * (bs - 1))
            noise = torch.nn.functional.embedding(block, embed)[None]
            pos = torch.arange(p + bs)[None]
            h = model(position_ids=pos, noise_embedding=noise, target_hidden=feats[:p][None])
            drafts = (h[0, 1:].float() @ head.float().T).argmax(-1).numpy()  # tokens for positions p+1 .. p+bs-1
            # accepted while the draft equals the target's greedy token, given every earlier draft was accepted
            a = 0
            while a < bs - 1 and drafts[a] == truth[p + a] and (a == 0 or seq[p + a] == truth[p + a - 1]):
                a += 1
            for k in ks:
                A[k].append(1 + min(a, k))
        for k in ks:
            allA[k] += A[k]
        print(f"{name}: positions {len(A[1])} " + " ".join(f"AL(k={k}) {np.mean(A[k]):.2f}" for k in ks), flush=True)
    print("ALL " + " ".join(f"AL(k={k}) {np.mean(allA[k]):.2f}" for k in ks))


if __name__ == "__main__":
    main()
