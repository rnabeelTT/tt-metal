# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU reference of the Hy4 vision path (HF transformers PR 50903 classes, vit.* weights only: ~0.9 GB).

    ~/m27-work/venv-hy4/bin/python models/autoports/tencent_hy4_flash/tools/vision_ref.py [--size 512] [--out DIR]

Writes DIR/ref_<WxH>.pt: the test image's processor outputs (pixel_values, image_grid_thw, the chat prompt's
input_ids / mm_token_type_ids / 3-D position ids), the hidden state entering every ViT block and leaving the last,
the merged features (the LM's image-token embeddings) in bf16 (HF's dtype) and fp32 (the noise floor), and checks
that tt/vision.py's host preprocessing equals the HF processor bit for bit.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "../../../.."))
SNAP = os.path.expanduser("~/.cache/huggingface/hub/models--tencent--Hy4-flash/snapshots/"
                          "d88c9dfd96ba4f2e6b92adba0b8e8f1c8696f959")
QUESTION = "What text is in the image? What shapes and colors do you see?"


def make_test_image(w: int = 512, h: int = 512):
    """White canvas: large black text "TT 4821" on top, a red filled circle bottom-left, a blue filled square bottom-
    right."""
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", int(h * 0.17))
    text = "TT 4821"
    bb = d.textbbox((0, 0), text, font=font)
    d.text(((w - (bb[2] - bb[0])) / 2 - bb[0], h * 0.08 - bb[1]), text, fill="black", font=font)
    r = int(min(w, h) * 0.17)
    cx, cy = int(w * 0.28), int(h * 0.68)
    d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(220, 20, 20))
    s = int(min(w, h) * 0.32)
    x0, y0 = int(w * 0.58), int(h * 0.68) - s // 2
    d.rectangle((x0, y0, x0 + s, y0 + s), fill=(20, 40, 220))
    return img


def load_vit(dtype=torch.bfloat16):
    from safetensors import safe_open
    from transformers.models.hy_v4_flash.configuration_hy_v4_flash import HYV4FlashConfig
    from transformers.models.hy_v4_flash.modeling_hy_v4_flash import HYV4FlashVisionTransformer
    import re

    cfg = HYV4FlashConfig.from_pretrained(SNAP)
    vc = cfg.vision_config
    vc._attn_implementation = "eager"
    with torch.device("meta"):
        vit = HYV4FlashVisionTransformer(vc)
    wm = json.load(open(os.path.join(SNAP, "model.safetensors.index.json")))["weight_map"]
    sd = {}
    for f in sorted({v for k, v in wm.items() if k.startswith("vit.")}):
        with safe_open(os.path.join(SNAP, f), "pt") as fh:
            for k in fh.keys():
                if k.startswith("vit."):
                    sd[k] = fh.get_tensor(k)
    ren = {}
    for k, v in sd.items():
        n = k
        n = n.replace("mlp.dense_h_to_4h", "mlp.fc1").replace("mlp.dense_4h_to_h", "mlp.fc2")
        n = re.sub(r"^vit\.layers\.(\d+)\.input_layernorm\.", r"vit.layers.\1.layer_norm1.", n)
        n = re.sub(r"^vit\.layers\.(\d+)\.post_attention_layernorm\.", r"vit.layers.\1.layer_norm2.", n)
        n = n.replace("perceive.norm.", "perceive.before_rms.").replace("perceive.proj.0.", "perceive.proj_conv.")
        n = n.replace("perceive.proj.2.", "perceive.proj_out.").replace("perceive.", "patch_merger.")
        ren[n[len("vit."):]] = v.to(dtype)
    missing = set(vit.state_dict()) - set(ren)
    assert not missing, missing
    vit.load_state_dict(ren, strict=True, assign=True)
    return vit.eval(), cfg, sd


def load_processor():
    """HYV4FlashProcessor with the PIL image backend (the venv has no torchvision: no video processor)."""
    from transformers import AutoTokenizer
    from transformers.models.hy_v4_flash.image_processing_pil_hy_v4_flash import HYV4FlashImageProcessorPil
    from transformers.models.hy_v4_flash.processing_hy_v4_flash import HYV4FlashProcessor

    pc = json.load(open(os.path.join(SNAP, "processor_config.json")))["image_processor"]
    ip = HYV4FlashImageProcessorPil(**{k: v for k, v in pc.items() if k not in ("image_processor_type", "device")})
    tok = AutoTokenizer.from_pretrained(SNAP)
    tmpl = open(os.path.join(SNAP, "chat_template.jinja")).read()
    HYV4FlashProcessor.get_attributes = classmethod(lambda cls: ["image_processor", "tokenizer"])
    try:
        return HYV4FlashProcessor(image_processor=ip, tokenizer=tok, chat_template=tmpl)
    except Exception:
        p = HYV4FlashProcessor.__new__(HYV4FlashProcessor)
        p.image_processor, p.tokenizer, p.video_processor, p.chat_template = ip, tok, None, tmpl
        p.image_token, p.video_token = tok.image_token, tok.video_token
        p.image_token_id, p.video_token_id = tok.convert_tokens_to_ids(tok.image_token), tok.convert_tokens_to_ids(tok.video_token)
        return p


def rope_index(cfg, input_ids, mm_token_type_ids, grid):
    from transformers.models.hy_v4_flash.modeling_hy_v4_flash import HYV4FlashModel

    ns = SimpleNamespace(config=cfg)
    ns.get_vision_position_ids = lambda *a, **k: HYV4FlashModel.get_vision_position_ids(ns, *a, **k)
    return HYV4FlashModel.get_rope_index(ns, input_ids, mm_token_type_ids, image_grid_thw=grid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="512", help="W or WxH")
    ap.add_argument("--out", default=os.path.expanduser("~/m27-work/hy4/vision"))
    ap.add_argument("--fp32", action="store_true", help="also run the tower in fp32 (noise floor)")
    args = ap.parse_args()
    w, h = (int(v) for v in (args.size.split("x") if "x" in args.size else (args.size, args.size)))
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(0)

    img = make_test_image(w, h)
    img.save(os.path.join(args.out, f"test_{w}x{h}.png"))
    proc = load_processor()
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": QUESTION}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, reasoning_effort="no_think")
    enc = proc(text=[text], images=[img], return_tensors="pt")
    print("image processor:", type(proc.image_processor).__name__)
    ids = enc["input_ids"]
    grid = enc["image_grid_thw"]
    print("prompt:", repr(text[:400]), "...")
    print("tokens", ids.shape[1], "grid", grid.tolist(), "pixel_values", tuple(enc["pixel_values"].shape),
          "image tokens", int((ids == 120120).sum()))
    # host preprocessing of the port vs HF
    sys.path.insert(0, REPO)
    from models.autoports.tencent_hy4_flash.tt.vision_host import preprocess, prompt_positions, expand_image_tokens

    pv, g = preprocess(img)
    same = torch.equal(torch.from_numpy(pv).float(), enc["pixel_values"].float())
    print("port preprocess == HF:", same, "max |d|", float((torch.from_numpy(pv).float() - enc["pixel_values"].float())
                                                       .abs().max()), "grid", g)

    t0 = time.time()
    vit, cfg, _ = load_vit(torch.bfloat16)
    print(f"vit loaded {time.time() - t0:.1f} s")
    pos, delta = rope_index(cfg, ids, enc["mm_token_type_ids"], grid)
    pos_port, delta_port = prompt_positions(ids[0].tolist(), [g])
    print("port positions == HF:", torch.equal(torch.as_tensor(pos_port), pos[:, 0]), "delta", int(delta), delta_port)
    print("port expand == HF:", expand_image_tokens(
        proc.tokenizer(proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                                reasoning_effort="no_think"), add_special_tokens=False)["input_ids"],
        [g]) == ids[0].tolist())
    states = []
    hooks = [blk.register_forward_pre_hook(lambda m, a, kw: states.append(a[0][0].clone()), with_kwargs=True)
             for blk in vit.layers]
    hooks.append(vit.patch_merger.register_forward_pre_hook(lambda m, a, kw: states.append(a[0][0].clone()),
                                                            with_kwargs=True))
    t0 = time.time()
    with torch.no_grad():
        out = vit(enc["pixel_values"].to(torch.bfloat16), grid_thw=grid)
    print(f"vit bf16 forward {time.time() - t0:.1f} s; features {tuple(out.pooler_output.shape)}")
    for hk in hooks:
        hk.remove()
    res = {"pixel_values": enc["pixel_values"], "grid": grid, "input_ids": ids[0], "mm": enc["mm_token_type_ids"][0],
           "pos": pos[:, 0], "delta": int(delta), "states": torch.stack(states).to(torch.bfloat16),
           "feat": out.pooler_output[0].to(torch.bfloat16), "text": text}
    if args.fp32:
        vit32, _, _ = load_vit(torch.float32)
        with torch.no_grad():
            f32 = vit32(enc["pixel_values"].float(), grid_thw=grid).pooler_output[0]
        res["feat32"] = f32
        a, b = f32.flatten().double(), res["feat"].flatten().double()
        print("bf16 vs fp32 tower PCC", float(torch.corrcoef(torch.stack([a, b]))[0, 1]))
    torch.save(res, os.path.join(args.out, f"ref_{w}x{h}.pt"))
    print("saved", os.path.join(args.out, f"ref_{w}x{h}.pt"))


if __name__ == "__main__":
    main()
