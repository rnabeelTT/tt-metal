# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host check of tt/mm.py (no device): an image and a tone through vLLM's Inkling processor, then our towers vs vLLM's
own tower modules (vllm/models/inkling/common/towers.py, bf16 weights on the CPU) on the same inputs; and build()'s
placement against the processor's expanded prompt.

usage (in the image): tools/run_dev.sh python models/autoports/thinkingmachines_inkling_small/tools/mm_tower_test.py
"""

import numpy as np
import torch

from models.autoports.thinkingmachines_inkling_small.tt import mm


def main():
    ck = mm.default_checkpoint()
    tw = mm.InklingTowers(ck)
    import json

    from vllm.transformers_utils.processors.inkling import InklingAudioFeatureExtractor, InklingImageProcessor

    cfg = json.load(open(f"{ck}/config.json"))
    ac = cfg["audio_config"]
    ip = InklingImageProcessor(patch_size=cfg["vision_config"]["patch_size"])
    fe = InklingAudioFeatureExtractor(params={"n_mels": ac["n_mel_bins"], "num_dmel_bins": ac["mel_vocab_size"],
                                              "dmel_min_value": ac["dmel_min_value"],
                                              "dmel_max_value": ac["dmel_max_value"]})
    rng = np.random.default_rng(0)
    from PIL import Image

    img = Image.fromarray(rng.integers(0, 255, (200, 280, 3), dtype=np.uint8))
    sr = 16000
    t = np.arange(sr * 2) / sr
    wav = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    out_i = ip.preprocess([img], return_tensors="pt")
    out_a = fe([wav])
    print("image processor keys:", list(out_i.keys()), "audio processor keys:", list(out_a.keys()))
    patches = torch.as_tensor(out_i["vision_patches_bthwc"])
    dmel = out_a["dmel_bins"]
    dmel = torch.as_tensor(dmel[0] if isinstance(dmel, (list, tuple)) else dmel)
    print(f"patches {tuple(patches.shape)}, dmel {tuple(dmel.shape)}")

    ours_i, ours_a = tw.image(patches), tw.audio(dmel)
    print(f"ours: image {tuple(ours_i.shape)}, audio {tuple(ours_a.shape)}")

    # vLLM's towers (CPU / native path), same checkpoint weights
    try:
        from vllm.models.inkling.common import towers as VT
        from vllm.models.inkling.common.towers import InklingAudio, InklingVision

        VT.plan_out_scales = mm.plan_out_scales  # (scipy is not in the image; the plan itself is checked on the weights)

        class _RMSNorm(torch.nn.Module):  # vLLM RMSNorm.forward_native's order, without its CustomOp config lookup
            def __init__(self, hidden, eps=1e-6):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(hidden))
                self.variance_epsilon = eps

            def forward(self, x):
                dt = x.dtype
                xf = x.float()
                xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)
                return xf.to(dt) * self.weight

        VT.RMSNorm = _RMSNorm
        from vllm.transformers_utils.config import get_config

        cfg = get_config(ck, trust_remote_code=True)
        vis = InklingVision(cfg.vision_config).to(torch.bfloat16)
        aud = InklingAudio(cfg.audio_config).to(torch.bfloat16)
        sd_v = {k.replace("model.visual.", "vision_encoder."): v for k, v in tw.w.items() if k.startswith("model.visual")}
        sd_a = {k.replace("model.audio.", ""): v for k, v in tw.w.items() if k.startswith("model.audio")}
        print("vision load:", vis.load_state_dict(sd_v, strict=True))
        print("audio load:", aud.load_state_dict(sd_a, strict=True))
        with torch.no_grad():
            ref_i = vis(patches.to(torch.bfloat16)).float()
            ref_a = aud(dmel).float()
        for name, a, b in (("image", ours_i.float(), ref_i), ("audio", ours_a.float(), ref_a)):
            cos = torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()
            rel = ((a - b).norm() / b.norm()).item()
            print(f"{name}: vs vLLM towers cosine {cos:.6f}, rel err {rel:.4f}, max|d| {(a - b).abs().max().item():.4g}")
    except Exception as e:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        print("vLLM tower comparison skipped:", e)

    img_id, aud_id = mm.placeholder_ids()
    n_i, n_a = int(ours_i.shape[0]), int(ours_a.shape[0])
    ids = [1, 2, 3] + [9] + [img_id] * n_i + [4, 5] + [8] + [aud_id] * n_a + [6, 7]
    items = [
        {"modality": "image", "identifier": "img0", "offset": 3, "length": n_i + 1,
         "data": {"pixel_values": patches, "num_patches": torch.tensor([n_i])}},
        {"modality": "audio", "identifier": "aud0", "offset": 6 + n_i, "length": n_a + 1,
         "data": {"input_audio_features": dmel, "num_audio_tokens": torch.tensor([n_a])}},
    ]
    b = mm.build(items, ids, tw)
    ok = (
        b["pos"].tolist() == [4 + i for i in range(n_i)] + [7 + n_i + i for i in range(n_a)]
        and torch.equal(b["rows"][:n_i], ours_i)
        and torch.equal(b["rows"][n_i:], ours_a)
    )
    print(f"build: {int(b['pos'].numel())} rows, keys {b['keys']}, placement {'OK' if ok else 'WRONG'}")
    no_off = [dict(it, offset=None, length=None) for it in items]
    b2 = mm.build(no_off, ids, tw)
    print(f"build without offsets: {'OK' if torch.equal(b2['pos'], b['pos']) else 'WRONG'}")


if __name__ == "__main__":
    main()
