# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU reference of the LM side of an image prompt on a layer subset (HF HYV4FlashTextModel with its first N layers):
the image features scattered into the prompt embeddings, the HF 3-D M-RoPE position ids, Engram off (engram_mask all
False: the Engram output is zero everywhere, as the device's --no-engram). Compared by tools/vision_check.py --dump.

    ~/m27-work/venv-hy4/bin/python models/autoports/tencent_hy4_flash/tools/vision_lm_ref.py --ref DIR/ref_512x512.pt
        --layers 4 [--plain-pos]   (--plain-pos: 1-D positions = slot indices, the sensitivity control)
-> DIR/lmref_<WxH>_L<N>[_plain].pt {"logits" fp16 [T, V], "ids"}. A video reference (--ref DIR/vref_8f_640x480.pt)
-> DIR/lmref_8f_640x480_L<N>[_plain].pt (its video pads take the per-frame features).
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))
sys.path.insert(0, os.path.join(REPO, "models/experimental/hybrid_moe"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--plain-pos", action="store_true")
    args = ap.parse_args()
    import ref_hy4 as R

    n = args.layers
    keep = {k: v for k, v in R.WMAP.items()
            if not k.startswith("model.layers.") or int(k.split(".")[2]) < n}
    R.WMAP.clear()
    R.WMAP.update(keep)
    cfg = R.config()
    cfg.num_hidden_layers = n
    cfg.layer_types = cfg.layer_types[:n]
    t0 = time.time()
    with torch.device("meta"):
        tm = R.M.HYV4FlashTextModel(cfg)
    tm = R.materialize(tm, "model.", cfg)
    lm_head = R.ckpt_tensor("lm_head.weight").float()
    print(f"{n}-layer reference built in {time.time() - t0:.0f} s", flush=True)
    ref = torch.load(args.ref, weights_only=False)
    ids = ref["input_ids"].reshape(1, -1)
    T = ids.shape[1]
    emb = tm.embed_tokens(ids)
    mask = ((ids == 120120) | (ids == 120124)).unsqueeze(-1)  # image / video pads (tools/video_ref.py vref_*.pt)
    emb = emb.masked_scatter(mask, ref["feat"].to(emb.dtype))
    pos = torch.arange(T).view(1, 1, -1).expand(3, 1, -1) if args.plain_pos else ref["pos"].view(3, 1, T)
    ng = torch.zeros(1, T, cfg.engram_max_ngram_size, dtype=torch.long)
    em = torch.zeros(1, T, dtype=torch.bool)
    with torch.no_grad():
        h = tm(inputs_embeds=emb, position_ids=pos, ngram_tokens=ng, engram_mask=em, use_cache=False).last_hidden_state
    lg = (h[0].float() @ lm_head.T).to(torch.float16)
    tag = os.path.basename(args.ref).split("_", 1)[1][:-3]
    out = os.path.join(os.path.dirname(args.ref), f"lmref_{tag}_L{n}{'_plain' if args.plain_pos else ''}.pt")
    torch.save({"logits": lg, "ids": ids[0]}, out)
    print("saved", out, "in", f"{time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
