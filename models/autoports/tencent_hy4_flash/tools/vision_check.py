# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Engine-level image check: ViT on the device, image embeddings + M-RoPE positions in the LM prefill, greedy answer.

    ~/m27-work/hy4/run.sh python -m models.autoports.tencent_hy4_flash.tools.vision_check [--size 512] [--layers 0,1,2,3]
        [--no-engram] [--max-new 96] [--batch] [--text-users 0]

Full model (default): the answer to "What text is in the image? What shapes and colors do you see?" about the
tools/vision_ref.py test image (text "TT 4821", a red circle, a blue square). --batch: through Hy4BatchEngine (slot 1),
with --text-users short text prompts decoding alongside in other slots. A layer subset only exercises the plumbing.
"""

from __future__ import annotations

import argparse
import os
import time
from types import SimpleNamespace

import torch
import ttnn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="512", help="W or WxH; several: comma-separated (one model build)")
    ap.add_argument("--image", default="", help="an image file instead of the test image")
    ap.add_argument("--question", default="")
    ap.add_argument("--layers", default="")
    ap.add_argument("--ctx", type=int, default=12352)
    ap.add_argument("--no-engram", action="store_true")
    ap.add_argument("--max-new", type=int, default=96)
    ap.add_argument("--batch", action="store_true")
    ap.add_argument("--text-users", type=int, default=0)
    ap.add_argument("--lmcmp", action="store_true", help="teacher-forced logits vs tools/vision_lm_ref.py (layer subset, "
                    "--no-engram): HF image features and the device ViT's, M-RoPE vs the plain-position control")
    args = ap.parse_args()
    if args.batch:
        os.environ.setdefault("HY4_SLOTS", "8")
    from PIL import Image
    from transformers import AutoTokenizer

    from models.autoports.tencent_hy4_flash.tools.generate import build, open_mesh
    from models.autoports.tencent_hy4_flash.tools.vision_ref import QUESTION, make_test_image
    from models.autoports.tencent_hy4_flash.tt.config import checkpoint_dir
    from models.autoports.tencent_hy4_flash.tt.vision import Hy4Vision
    from models.autoports.tencent_hy4_flash.tt.vision_host import expand_image_tokens, preprocess

    tk = AutoTokenizer.from_pretrained(checkpoint_dir())
    def make(size):
        if args.image:
            img = Image.open(args.image)
        else:
            w, h = (int(v) for v in (size.split("x") if "x" in size else (size, size)))
            img = make_test_image(w, h)
        q = args.question or QUESTION
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q}]}]
        text = tk.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, reasoning_effort="no_think")
        pv, grid = preprocess(img)
        ids = expand_image_tokens(tk(text, add_special_tokens=False)["input_ids"], [grid])
        print(f"image {img.size} -> grid {grid}; prompt {len(ids)} tokens", flush=True)
        return ids, pv, grid

    def enc(m):
        t = tk.apply_chat_template([{"role": "user", "content": m}], tokenize=False, add_generation_prompt=True,
                                   reasoning_effort="no_think")
        return tk(t, add_special_tokens=False)["input_ids"]

    mesh = open_mesh()
    try:
        model = build(mesh, SimpleNamespace(layers=args.layers, ctx=args.ctx, chunk=2048, no_engram=args.no_engram))
        if model.engram is not None and hasattr(model.engram.store, "wait_ram"):
            model.engram.store.wait_ram()
        vit = Hy4Vision(mesh)
        print(model.dram_report(), flush=True)
        eng = None
        if args.lmcmp:
            lm_compare(args, model, vit)
            return
        for size in args.size.split(","):
            ids, pv, grid = make(size)
            for it in range(2):  # the first: compiles
                t0 = time.time()
                mm = vit.prepare(ids, [(pv, grid)])
                t_vit = time.time() - t0
            print(f"ViT + merger (warm) {t_vit * 1e3:.0f} ms; features {tuple(mm['embeds'].shape)}, delta {mm['delta']}",
                  flush=True)
            if args.batch:
                from models.autoports.tencent_hy4_flash.tt.engine_batch import Hy4BatchEngine

                eng = eng or Hy4BatchEngine(model)
                prompts = ["Explain why the sky is blue.", "Write a haiku about rain.", "What is 17 * 23?",
                           "Name three rivers in Europe.", "Describe a cat.", "List two prime numbers.", "Say hello."]
                others = {s: enc(prompts[i]) for i, s in enumerate([0, 2, 3, 4, 5, 6, 7][: args.text_users])}
                outs = {s: [eng.prefill(s, p)] for s, p in others.items()}
                t0 = time.time()
                outs[1] = [eng.prefill(1, ids, mm=mm)]
                t_pf = time.time() - t0
                ts = []
                while len(outs[1]) < args.max_new and outs[1][-1] != model.cfg.eos:
                    t1 = time.time()
                    res = eng.step({s: (None, None, False) for s in outs})
                    ts.append(time.time() - t1)
                    for s, v in res.items():
                        outs[s] += v
                for s in sorted(others):
                    print(f"TEXT slot {s}: {tk.decode(outs[s][:48])!r}", flush=True)
                out = outs[1]
            else:
                from models.autoports.tencent_hy4_flash.tt.engine import Hy4Engine

                eng = eng or Hy4Engine(model, 0)
                t0 = time.time()
                out = [eng.prefill(ids, mm=mm)]
                t_pf = time.time() - t0
                ts = []
                while len(out) < args.max_new and out[-1] != model.cfg.eos:
                    t1 = time.time()
                    out += eng.step()
                    ts.append(time.time() - t1)
            print("=" * 80)
            print("ANSWER:", tk.decode(out))
            print("=" * 80)
            warm = ts[2:] or ts
            print(f"prefill {len(ids)} tok in {t_pf:.2f} s (ViT {t_vit:.2f} s on top); decode {len(out)} tok, "
                  f"warm step {1e3 * sum(warm) / max(1, len(warm)):.1f} ms", flush=True)
    finally:
        ttnn.close_mesh_device(mesh)


def lm_compare(args, model, vit):
    from models.autoports.tencent_hy4_flash.tt.vision_host import prompt_positions

    d = os.path.expanduser("~/m27-work/hy4/vision")
    nl = len(args.layers.split(","))
    for size in args.size.split(","):
        tag = size if "x" in size else f"{size}x{size}"
        ref = torch.load(f"{d}/ref_{tag}.pt", weights_only=False)
        ids = ref["input_ids"].reshape(-1)
        grid = tuple(int(v) for v in ref["grid"][0].tolist())
        pos3, delta = prompt_positions(ids.tolist(), [grid])
        idx = torch.nonzero(ids == 120120).reshape(-1)
        dev_feat = torch.cat(vit.encode([(ref["pixel_values"].float().numpy(), grid)]))
        hf = torch.load(f"{d}/lmref_{tag}_L{nl}.pt")["logits"].float()
        plain = f"{d}/lmref_{tag}_L{nl}_plain.pt"
        plain = torch.load(plain)["logits"].float() if os.path.exists(plain) else None
        for name, feat in (("HF features", ref["feat"]), ("device ViT", dev_feat)):
            model.set_slot(0)
            model.reset()
            model.set_images(pos3, delta, feat, idx)
            lg = model.forward(ids, logits_rows="all").float()
            model.reset()
            msg = []
            for rn, r in (("HF M-RoPE", hf), ("HF plain-pos control", plain)):
                if r is None:
                    continue
                a, b = lg.flatten().double(), r.flatten().double()
                pcc = float(torch.corrcoef(torch.stack([a, b]))[0, 1])
                top1 = float((lg.argmax(-1) == r.argmax(-1)).float().mean())
                tail = slice(int(idx[-1]) + 1, None)  # the text after the image
                t_tail = float((lg[tail].argmax(-1) == r[tail].argmax(-1)).float().mean())
                msg.append(f"vs {rn}: PCC {pcc:.5f} top1 {top1:.4f} (after image {t_tail:.4f})")
            print(f"LMCMP {tag} {nl} layers, {name}: " + "; ".join(msg), flush=True)


if __name__ == "__main__":
    main()
