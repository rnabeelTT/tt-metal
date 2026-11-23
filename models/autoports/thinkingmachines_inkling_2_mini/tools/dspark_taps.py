# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""DSpark drafter fixture for tests/test_drafter_device.py: the HF target's taps (one teacher-forced CPU forward per
text) and the CPU reference drafter's outputs (reference/dspark_ref.py) at evenly spaced anchors.

Saved per text: ids [T], taps [T, 4, H] bf16, anchors (a: context = positions 0..a, anchor token ids[a + 1] at
position p = a + 1), and per anchor the reference block hidden [8, H] bf16, the Markov drafts [7], the base-only drafts
[7] and the top-16 (values, ids) of each Markov step's logits (to tell near-ties from real mismatches).

usage (host venv with transformers >= 5.24):
  ~/inkling2-work/venv/bin/python tools/dspark_taps.py [OUT] [--texts a,b,c] [--anchors 16]
"""
import argparse
import glob
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from reference.dspark_ref import DEFAULT_TEXTS, DSparkDrafter, MTPConfig, TargetRunner  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?", default=os.path.expanduser("~/.cache/inkling2/fixtures/dspark_taps.pt"))
    ap.add_argument("--texts", default="math_gsm8k,code_mbpp,chat_english")
    ap.add_argument("--anchors", type=int, default=16)
    ap.add_argument("--min-anchor", type=int, default=64)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    os.environ.setdefault("HF_HOME", "/opt/tenstorrent/models")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    torch.set_num_threads(args.threads)
    from transformers import Inkling2LoopForConditionalGeneration

    snap = glob.glob(f"{os.environ['HF_HOME']}/hub/models--thinkingmachines--Inkling-2-Mini/snapshots/*/mtp.safetensors")
    assert len(snap) == 1, snap
    t0 = time.time()
    # (the snapshot dir, not the repo id: the hub's newest snapshot may hold only the tokenizer files)
    model = Inkling2LoopForConditionalGeneration.from_pretrained(os.path.dirname(snap[0]), dtype=torch.bfloat16).eval()
    print(f"target loaded in {time.time() - t0:.0f}s", flush=True)
    cfg = MTPConfig.from_hf(model.config)
    lm = model.model.language_model
    dr = DSparkDrafter(cfg, snap[0], embed_weight=lm.embed_tokens.weight.data,
                       embed_norm_weight=lm.embed_tokens.embed_norm.weight.data,
                       lm_head_weight=model.lm_head.weight.data, target_eps=model.config.text_config.rms_norm_eps)
    runner = TargetRunner(model, cfg.target_layer_ids)
    texts = json.load(open(DEFAULT_TEXTS))
    out = {"cfg": vars(cfg), "texts": {}}
    B = cfg.block_size
    for name in args.texts.split(","):
        ids = torch.tensor(texts[name], dtype=torch.long)
        T = ids.shape[0]
        t1 = time.time()
        with torch.no_grad():
            taps, _ = runner.run(ids)
        anchors = torch.linspace(args.min_anchor, T - B - 2, args.anchors).round().long().tolist()
        ctx_kv = dr.context_kv(taps, torch.arange(T))
        rec = {"ids": ids, "taps": taps.contiguous(), "anchors": anchors, "hidden": [], "markov": [], "base": [],
               "top_v": [], "top_i": []}
        for a in anchors:
            p = a + 1
            anchor = int(ids[p])
            h = dr.block_hidden(ctx_kv, p, anchor)
            f_rnn = dr.f_rnn(taps[p - 1])
            d_m, lg = dr.draft_from_hidden(h, anchor, f_rnn, use_markov=True)
            d_b, _ = dr.draft_from_hidden(h, anchor, f_rnn, use_markov=False)
            tv, ti = lg.topk(16, dim=-1)
            rec["hidden"].append(h)
            rec["markov"].append(d_m)
            rec["base"].append(d_b)
            rec["top_v"].append(tv)
            rec["top_i"].append(ti)
        for k in ("hidden", "markov", "base", "top_v", "top_i"):
            rec[k] = torch.stack(rec[k])
        out["texts"][name] = rec
        print(f"[{name}] T={T} {len(anchors)} anchors in {time.time() - t1:.0f}s", flush=True)
    torch.save(out, args.out)
    print(f"saved {args.out} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
