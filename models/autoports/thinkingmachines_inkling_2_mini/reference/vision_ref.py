# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Image-input fixtures from the transformers reference (CPU).

Vision (default): the vision tower alone (Inkling2VisionModel, the checkpoint's model.visual.* weights) on each test
image, through the HF processor (Inkling2Processor / Inkling2ImageProcessor). Saved per image:
  pixel_values [N, 40, 40, 3] fp32, image_grid_thw [1, 3]
  fp32: {"embed": [S, 1024] (after pre_layernorm), "layers": {i: [S, 1024]} (outputs of layers 0, 11, 22),
         "out": [N, 2304] (the projected image embeddings, one per image token)}
  bf16: the same with the weights in bf16 (as transformers serves the checkpoint): the reference's own noise floor

--full: the whole model (bf16, as transformers loads it) on a chat prompt with one image: the prompt ids, the
last-position logits [V] fp32, the image embeddings it spliced in, and its greedy continuation (--gen tokens).

usage (host venv with transformers >= 5.24):
  ~/inkling2-work/venv/bin/python reference/vision_ref.py [--out DIR] [--full IMAGE] [--gen 96]
"""
import argparse
import os
import re
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from tt.weights import snapshot_dir  # noqa: E402

FIX = os.path.expanduser("~/.cache/inkling2/fixtures/vision")
IMAGES = ["cat.jpg", "valley.png"]
QUESTION = "Describe this image in two sentences."
RENAMES = [
    (r"^model\.visual\.backbone\.embeddings\.position_embedding$", "embeddings.position_embedding.weight"),
    (r"^model\.visual\.backbone\.embeddings\.", "embeddings."),
    (r"^model\.visual\.backbone\.ln_pre\.", "pre_layernorm."),
    (r"^model\.visual\.backbone\.layers\.", "layers."),
    (r"^model\.visual\.proj_in\.", "merger.linear_1."),
    (r"^model\.visual\.proj_out\.", "merger.linear_2."),
    (r"\.ln_1\.", ".layernorm_before."),
    (r"\.ln_2\.", ".layernorm_after."),
    (r"\.mlp\.c_fc\.", ".mlp.fc1."),
    (r"\.mlp\.c_proj\.", ".mlp.fc2."),
    (r"\.ls_1\.gamma$", ".lambda_1"),
    (r"\.ls_2\.gamma$", ".lambda_2"),
]


def vision_state_dict(snap):
    from safetensors import safe_open

    sd = {}
    with safe_open(os.path.join(snap, "model-00001-of-00001.safetensors"), "pt") as f:
        for k in f.keys():
            if not k.startswith("model.visual."):
                continue
            n = k
            for a, b in RENAMES:
                n = re.sub(a, b, n)
            sd[n] = f.get_tensor(k).clone()
    return sd


def load_vision(snap, dtype):
    from transformers import AutoConfig
    from transformers.models.inkling2.modeling_inkling2 import Inkling2VisionModel

    cfg = AutoConfig.from_pretrained(snap).vision_config
    cfg._attn_implementation = "eager"
    m = Inkling2VisionModel(cfg)
    missing, unexpected = m.load_state_dict(vision_state_dict(snap), strict=False)
    missing = [k for k in missing if "inv_freq" not in k]
    assert not missing and not unexpected, (missing, unexpected)
    return m.to(dtype).eval()


def processor(snap):
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(snap)


def messages(image):
    return [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": QUESTION}]}]


def run_tower(m, pv, grid, dtype):
    cap = {}
    hooks = [m.pre_layernorm.register_forward_hook(lambda mod, i, o: cap.__setitem__("embed", o.float().clone()))]
    for li in (0, 11, len(m.layers) - 1):
        hooks.append(m.layers[li].register_forward_hook(
            lambda mod, i, o, li=li: cap.setdefault("layers", {}).__setitem__(li, o.float().clone())))
    with torch.no_grad():
        out = m(pixel_values=pv.to(dtype), image_grid_thw=grid).pooler_output
    for h in hooks:
        h.remove()
    cap["out"] = out.float()
    return cap


def pcc(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=FIX)
    ap.add_argument("--full", default=None, help="image file for the whole-model fixture")
    ap.add_argument("--gen", type=int, default=96)
    ap.add_argument("--threads", type=int, default=12)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    snap = snapshot_dir()
    from PIL import Image

    proc = processor(snap)
    if args.full:
        return full(snap, proc, args)
    m32 = load_vision(snap, torch.float32)
    m16 = load_vision(snap, torch.bfloat16)
    for name in IMAGES:
        img = Image.open(os.path.join(FIX, name))
        enc = proc.image_processor(images=[img], return_tensors="pt")
        pv, grid = enc["pixel_values"], enc["image_grid_thw"]
        t0 = time.time()
        r32 = run_tower(m32, pv, grid, torch.float32)
        r16 = run_tower(m16, pv, grid, torch.bfloat16)
        rec = {"pixel_values": pv, "image_grid_thw": grid, "fp32": r32, "bf16": r16}
        torch.save(rec, os.path.join(args.out, os.path.splitext(name)[0] + ".pt"))
        print(f"{name}: grid {grid.tolist()} pixels {tuple(pv.shape)} seq {r32['embed'].shape[0]} out {tuple(r32['out'].shape)} "
              f"bf16 vs fp32 pcc: embed {pcc(r16['embed'], r32['embed']):.6f} last layer "
              f"{pcc(r16['layers'][22], r32['layers'][22]):.6f} out {pcc(r16['out'], r32['out']):.6f} "
              f"({time.time() - t0:.0f}s)", flush=True)


def full(snap, proc, args):
    from PIL import Image
    from transformers import Inkling2LoopForConditionalGeneration

    img = Image.open(args.full)
    t0 = time.time()
    model = Inkling2LoopForConditionalGeneration.from_pretrained(snap, dtype=torch.bfloat16).eval()
    print(f"model loaded in {time.time() - t0:.0f}s", flush=True)
    rec = {}
    for effort in ("none",):
        enc = proc.apply_chat_template(messages(img), add_generation_prompt=True, tokenize=True, return_dict=True,
                                       return_tensors="pt", reasoning_effort=effort)
        ids = enc["input_ids"]
        with torch.no_grad():
            feats = model.get_image_features(enc["pixel_values"], enc["image_grid_thw"]).pooler_output
            out = model(**enc, logits_to_keep=1)
            t1 = time.time()
            gen = model.generate(**enc, max_new_tokens=args.gen, do_sample=False)
        cont = gen[0, ids.shape[1]:]
        rec[effort] = {"ids": ids[0], "pixel_values": enc["pixel_values"], "image_grid_thw": enc["image_grid_thw"],
                       "image_embeds": torch.cat(feats).float(), "logits": out.logits[0, -1].float(),
                       "greedy": cont, "text": proc.tokenizer.decode(cont)}
        print(f"[{effort}] prompt {ids.shape[1]} tokens; greedy ({time.time() - t1:.0f}s): {rec[effort]['text']!r}",
              flush=True)
    torch.save(rec, os.path.join(args.out, "full_" + os.path.splitext(os.path.basename(args.full))[0] + ".pt"))


if __name__ == "__main__":
    main()
