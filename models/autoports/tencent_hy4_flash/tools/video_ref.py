# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU reference of Hy4 video inputs, in three steps (three environments):

    make  (host python_env: PIL + font)       python tools/video_ref.py make [--frames 8 --size 640x480]
          -> DIR/video_<tag>.npy uint8 [T, H, W, 3] (a red ball crossing left -> right, "TT 4821" on top) + a png strip
    proc  (the serving image: vLLM 0.26, transformers 5.23, torchvision; Tencent's vLLM PR 68296 processor loaded from
          ~/m27-work/hy4/vllm-pr) -> DIR/vproc_<tag>.pt: input_ids, pixel_values_videos, video_grid_thw, timestamps and
          the PR's 3-D M-RoPE positions (HYV4VL get_mrope_input_positions, copied below), and checks
          tt/vision_host.py's video path against them bit for bit
    tower (venv-hy4: HF transformers PR 50903 ViT) -> DIR/vref_<tag>.pt: + the merged features (bf16 / fp32)

Video format (the PR; the HF transformers port differs: one start / pads / end block, no timestamps; the checkpoint's
processor_config carries the PR's timestamp keys): <video_start> then per frame "<{t:.1f} seconds>" <image_start>
(gh/2) x (gw/2 + 1) x <video_pad> <image_end>, then <video_end>; t = frame index / fps. Positions: text 1-D; each
frame's pads (s, s + row, s + col) as an image, s advancing by max(rows, cols) per frame, the timestamp / boundary
tokens between frames 1-D.
"""

from __future__ import annotations

import argparse
import os
import sys
import types

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "../../../.."))
DIR = os.path.expanduser("~/m27-work/hy4/vision")
QUESTION = "Describe what happens in this video: what object moves, its color, and in which direction?"


def make_frames(T=8, w=640, h=480):
    from PIL import Image, ImageDraw, ImageFont

    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", int(h * 0.14))
    out = []
    r = int(h * 0.12)
    for i in range(T):
        img = Image.new("RGB", (w, h), "white")
        d = ImageDraw.Draw(img)
        bb = d.textbbox((0, 0), "TT 4821", font=font)
        d.text(((w - (bb[2] - bb[0])) / 2 - bb[0], h * 0.05 - bb[1]), "TT 4821", fill="black", font=font)
        cx = int(r + (w - 2 * r) * i / max(1, T - 1))
        cy = int(h * 0.65)
        d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(220, 20, 20))
        out.append(np.asarray(img))
    return np.stack(out)


def pr_positions(input_tokens, video_grids, merge=2, image_start=120118, image_end=120119, video_start=120122,
                 video_end=120123, video_pad=120124):
    """Tencent vLLM PR 68296 HYV4VL get_mrope_input_positions, videos only (cat_extra_token 0, newline column on)."""
    tokens = torch.tensor(input_tokens, dtype=torch.long)
    positions = torch.arange(len(input_tokens)).repeat(3, 1)
    starts = torch.where(tokens == image_start)[0].tolist()
    ends = torch.where(tokens == image_end)[0].tolist()
    start_pos = start_index = mm_index = 0
    video_index = -1
    while mm_index < len(starts):
        pos = starts[mm_index] + 1
        if start_pos < pos:
            positions[:, start_pos:pos] = torch.arange(pos - start_pos) + start_index
            start_index += pos - start_pos
        video_index += 1
        t, h, w = (int(v) for v in video_grids[video_index])
        frames, rows, cols = t, h // merge, w // merge + 1
        count = rows * cols
        for frame in range(frames):
            begin = starts[mm_index + frame] + 1
            end = begin + count
            assert end == ends[mm_index + frame] and bool(torch.all(tokens[begin:end] == video_pad))
            positions[0, begin:end] = start_index
            positions[1, begin:end] = torch.arange(rows).repeat_interleave(cols) + start_index
            positions[2, begin:end] = torch.arange(cols).repeat(rows) + start_index
            start_index += max(rows, cols)
            if frame + 1 < frames:
                b0, b1 = ends[mm_index + frame], starts[mm_index + frame + 1] + 1
                positions[:, b0:b1] = torch.arange(b1 - b0) + start_index
                start_index += b1 - b0
        start_pos = ends[mm_index + frames - 1]
        mm_index += frames
    if start_pos < len(input_tokens):
        positions[:, start_pos:] = torch.arange(len(input_tokens) - start_pos) + start_index
    return positions, int(positions.max()) + 1 - len(input_tokens)


def load_pr():
    """Tencent's vLLM PR processor modules on top of the image's vLLM 0.26 (only their own config import is missing)."""
    pr = os.path.expanduser("~/m27-work/hy4/vllm-pr/vllm/transformers_utils")
    import importlib.util

    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    load("vllm.transformers_utils.configs.hy_v4", f"{pr}/configs/hy_v4.py")
    return load("hy4_pr_processor", f"{pr}/processors/hy_v4.py")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("make", "proc", "tower"))
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--size", default="640x480")
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--dir", default=DIR)
    args = ap.parse_args()
    w, h = (int(v) for v in args.size.split("x"))
    tag = f"{args.frames}f_{w}x{h}"
    os.makedirs(args.dir, exist_ok=True)
    if args.step == "make":
        v = make_frames(args.frames, w, h)
        np.save(f"{args.dir}/video_{tag}.npy", v)
        from PIL import Image

        Image.fromarray(np.concatenate(list(v[:: max(1, len(v) // 4)]), axis=1)).save(f"{args.dir}/video_{tag}.png")
        print("saved", f"{args.dir}/video_{tag}.npy", v.shape)
        return
    if args.step == "proc":
        snap = os.environ.get("SNAP", "tencent/Hy4-flash")
        P = load_pr()
        proc = P.HYV4VLProcessor.from_pretrained(snap)
        frames = np.load(f"{args.dir}/video_{tag}.npy")
        msgs = [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": QUESTION}]}]
        text = proc.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                                  reasoning_effort="no_think")
        meta = {"fps": args.fps, "total_num_frames": len(frames), "frames_indices": list(range(len(frames)))}
        enc = proc(text=[text], videos=[(torch.from_numpy(frames).permute(0, 3, 1, 2), meta)], return_tensors="pt")
        ids = enc["input_ids"][0]
        grid = enc["video_grid_thw"]
        pos, delta = pr_positions(ids.tolist(), grid.tolist())
        print("tokens", ids.numel(), "grid", grid.tolist(), "pv", tuple(enc["pixel_values_videos"].shape),
              "timestamps", enc["video_timestamps"].tolist(), "pads", int((ids == 120124).sum()), "delta", delta)
        print("decoded head:", repr(proc.tokenizer.decode(ids[:200])[-400:]))
        res = {"input_ids": ids, "pixel_values": enc["pixel_values_videos"], "grid": grid,
               "timestamps": enc["video_timestamps"], "pos": pos, "delta": delta, "text": text}
        sys.path.insert(0, REPO)
        try:
            from models.autoports.tencent_hy4_flash.tt import vision_host as vh

            pv, g, ts = vh.video_preprocess(frames, fps=args.fps)
            print("port video preprocess == PR:", torch.equal(torch.from_numpy(pv), enc["pixel_values_videos"].float()),
                  "grid", g, "timestamps", ts)
            tok = proc.tokenizer(text, add_special_tokens=False)["input_ids"]
            ids2 = vh.expand_video_tokens(tok, [(g, ts)], proc.tokenizer)
            print("port expand == PR:", ids2 == ids.tolist())
            p2, d2 = vh.prompt_positions(ids2, [], videos=[g])
            print("port positions == PR:", torch.equal(torch.as_tensor(p2), pos), d2, delta)
            # a fully decoded clip (vLLM's loaders set do_sample_frames then): the PR resamples it to 2 fps
            sfps = 4 * args.fps
            meta2 = {"fps": sfps, "total_num_frames": len(frames), "frames_indices": list(range(len(frames))),
                     "do_sample_frames": True}
            enc2 = proc(text=[text], videos=[(torch.from_numpy(frames).permute(0, 3, 1, 2), meta2)],
                        return_tensors="pt")
            sel = vh.sample_frames(len(frames), sfps)
            pv3, g3, ts3 = vh.video_preprocess(frames[sel], fps=sfps, frames_indices=sel)
            ids3 = vh.expand_video_tokens(tok, [(g3, ts3)], proc.tokenizer)
            print(f"port do_sample_frames (fps {sfps}) == PR: frames {sel}, pixels",
                  torch.equal(torch.from_numpy(pv3), enc2["pixel_values_videos"].float()), "timestamps",
                  ts3 == enc2["video_timestamps"].tolist(), "ids", ids3 == enc2["input_ids"][0].tolist())
        except (ImportError, AttributeError) as e:
            print("port video path not checked:", e)
        torch.save(res, f"{args.dir}/vproc_{tag}.pt")
        print("saved", f"{args.dir}/vproc_{tag}.pt")
        return
    # tower
    sys.path.insert(0, HERE)
    from vision_ref import load_vit

    res = torch.load(f"{args.dir}/vproc_{tag}.pt", weights_only=False)
    vit, _, _ = load_vit(torch.bfloat16)
    # the PR's HYV4VisionTransformer on a video = every frame as its own image: attention segmented per frame
    # (cu_seqlens), the position embedding repeated per frame, the merger batched over frames (temporal / spatial
    # patch 1: no pooling) -> T x (gh/2) x (gw/2 + 1) features in frame order
    T, gh, gw = (int(v) for v in res["grid"][0])
    pv = res["pixel_values"].reshape(T, gh * gw, -1)
    g1 = torch.tensor([[1, gh, gw]])

    def run(m, dt):
        return torch.cat([m(pv[i].to(dt), grid_thw=g1).pooler_output[0] for i in range(T)])

    with torch.no_grad():
        res["feat"] = run(vit, torch.bfloat16)
        vit32, _, _ = load_vit(torch.float32)
        res["feat32"] = run(vit32, torch.float32)
    a, b = res["feat"].flatten().double(), res["feat32"].flatten().double()
    print("features", tuple(res["feat"].shape), "bf16 vs fp32 PCC", float(torch.corrcoef(torch.stack([a, b]))[0, 1]))
    torch.save(res, f"{args.dir}/vref_{tag}.pt")
    print("saved", f"{args.dir}/vref_{tag}.pt")


if __name__ == "__main__":
    main()
