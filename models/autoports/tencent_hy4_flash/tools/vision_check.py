# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Engine-level image check: ViT on the device, image embeddings + M-RoPE positions in the LM prefill, greedy answer.

    ~/m27-work/hy4/run.sh python -m models.autoports.tencent_hy4_flash.tools.vision_check [--size 512] [--layers 0,1,2,3]
        [--no-engram] [--max-new 96] [--batch] [--text-users 0] [--video 8f_640x480 [--fps 2]]

Full model (default): the answer to "What text is in the image? What shapes and colors do you see?" about the
tools/vision_ref.py test image (text "TT 4821", a red circle, a blue square). --batch: through Hy4BatchEngine (slot 1),
with --text-users short text prompts decoding alongside in other slots. A layer subset only exercises the plumbing.
--video TAG: the tools/video_ref.py clip DIR/video_TAG.npy instead (a red ball moving left -> right under "TT 4821"), in
Tencent's vLLM PR 68296 layout (per frame "<t seconds>" + an image block of video pads); the device features are compared
with DIR/vref_TAG.pt when it exists; with --lmcmp: teacher-forced logits vs DIR/lmref_TAG_L<N>.pt.
"""

from __future__ import annotations

import argparse
import os
import time
from types import SimpleNamespace

import numpy as np
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
    ap.add_argument("--video", default="", help="a tools/video_ref.py clip tag (e.g. 8f_640x480; comma-separated)")
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--sample", action="store_true", help="resample the clip as the PR / the server do for a fully "
                    "decoded video (vision_host.sample_frames: 2 fps); per clip: --video TAG:s")
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

    def make_video(tag):
        tag, smp = (tag[:-2], True) if tag.endswith(":s") else (tag, args.sample)
        from models.autoports.tencent_hy4_flash.tools.video_ref import QUESTION as VQ
        from models.autoports.tencent_hy4_flash.tt.vision_host import expand_video_tokens, video_preprocess

        frames = np.load(os.path.expanduser(f"~/m27-work/hy4/vision/video_{tag}.npy"))
        msgs = [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": args.question or VQ}]}]
        text = tk.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, reasoning_effort="no_think")
        sel = list(range(len(frames)))
        if smp:
            from models.autoports.tencent_hy4_flash.tt.vision_host import sample_frames

            sel = sample_frames(len(frames), args.fps)
        pv, grid, ts = video_preprocess(frames[sel], fps=args.fps, frames_indices=sel)
        ids = expand_video_tokens(tk(text, add_special_tokens=False)["input_ids"], [(grid, ts)], tk)
        print(f"video {frames.shape} -> grid {grid}, timestamps {ts}; prompt {len(ids)} tokens", flush=True)
        ref = os.path.expanduser(f"~/m27-work/hy4/vision/vproc_{tag}.pt")
        if os.path.exists(ref) and not smp:
            r = torch.load(ref, weights_only=False)
            print("prompt ids == PR processor:", r["input_ids"].tolist() == ids, "; pixel values == PR:",
                  torch.equal(torch.from_numpy(pv), r["pixel_values"].float()), flush=True)
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
        for size in (args.video or args.size).split(","):
            if args.video:
                ids, pv, grid = make_video(size)
            else:
                ids, pv, grid = make(size)
            for it in range(2):  # the first: compiles
                t0 = time.time()
                mm = vit.prepare(ids, [], videos=[(pv, grid)]) if args.video else vit.prepare(ids, [(pv, grid)])
                t_vit = time.time() - t0
            print(f"ViT + merger (warm) {t_vit * 1e3:.0f} ms; features {tuple(mm['embeds'].shape)}, delta {mm['delta']}",
                  flush=True)
            vref = os.path.expanduser(f"~/m27-work/hy4/vision/vref_{size}.pt")
            if args.video and os.path.exists(vref) and not args.sample and not size.endswith(":s"):
                r = torch.load(vref, weights_only=False)
                pc = lambda a, b: float(torch.corrcoef(torch.stack([a.flatten().double(), b.flatten().double()]))[0, 1])  # noqa
                cs = torch.nn.functional.cosine_similarity(mm["embeds"].float(), r["feat32"].float(), dim=-1)
                print(f"VIDEO features vs PR-tower reference: PCC bf16 {pc(mm['embeds'], r['feat']):.6f} fp32 "
                      f"{pc(mm['embeds'], r['feat32']):.6f} (ref bf16 vs fp32 {pc(r['feat'], r['feat32']):.6f}); "
                      f"per-token cosine min {float(cs.min()):.5f} mean {float(cs.mean()):.5f}", flush=True)
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
    for size in (args.video or args.size).split(","):
        tag = size if "x" in size else f"{size}x{size}"
        ref = torch.load(f"{d}/{'v' if args.video else ''}ref_{tag}.pt", weights_only=False)
        ids = ref["input_ids"].reshape(-1)
        grid = tuple(int(v) for v in ref["grid"][0].tolist())
        if args.video:
            pos3, delta = prompt_positions(ids.tolist(), [], videos=[grid])
            assert torch.equal(torch.as_tensor(pos3), ref["pos"]) and delta == ref["delta"]
            idx = torch.nonzero(ids == 120124).reshape(-1)
            dev_feat = vit.encode_video(ref["pixel_values"].float().numpy(), grid)
        else:
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
                tail = slice(int(idx[-1]) + 1, None)  # the text after the image / video
                t_tail = float((lg[tail].argmax(-1) == r[tail].argmax(-1)).float().mean())
                msg.append(f"vs {rn}: PCC {pcc:.5f} top1 {top1:.4f} (after image {t_tail:.4f})")
            print(f"LMCMP {tag} {nl} layers, {name}: " + "; ".join(msg), flush=True)


if __name__ == "__main__":
    main()
