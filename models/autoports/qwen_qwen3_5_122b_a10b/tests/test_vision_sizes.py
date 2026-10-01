# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Served image sizes: every padded patch bucket under the resolution cap is warmed (tt/vision.py warm_tower_sizes), so
real images of any size the capped processor produces - a 1080p frame, the largest image under the cap, a phone photo
resized by the cap, two images in one request - run the tower without compiling a program (the compile-after-capture
guard that stopped the served engine on 2026-10-01), and the tower still matches the HF visual at the large sizes."""
import glob
import os

import pytest
import torch

import ttnn
from models.autoports.qwen_qwen3_5_122b_a10b.tests.test_vision_tower import _image
from models.autoports.qwen_qwen3_5_122b_a10b.tt.model_config import Qwen35MoeModelArgs
from models.autoports.qwen_qwen3_5_122b_a10b.tt.vision import (
    build_hf_visual,
    max_patches,
    max_pixels,
    padded_patch_buckets,
    vision_args_for,
    warm_tower_sizes,
)
from models.common.utility_functions import comp_pcc


@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "l1_small_size": 24576, "trace_region_size": 64 * 1024 * 1024}],
    indirect=True,
)
def test_vision_sizes_covered_by_warmup(mesh_device):
    os.environ.setdefault("MESH_DEVICE", "P300x2")
    args = Qwen35MoeModelArgs(mesh_device, max_batch_size=1, max_seq_len=4096)
    visual = build_hf_visual(args.load_state_dict(), args.hf_config)
    vargs = vision_args_for(mesh_device, args)
    from models.demos.blackhole.qwen36.tt.vision.model import DropInVisionTransformer
    from models.tt_transformers.tt.ccl import TT_CCL

    tower = DropInVisionTransformer(visual, vargs, dtype=ttnn.bfloat8_b, tt_ccl=TT_CCL(mesh_device))  # served dtype
    vc = args.hf_config.vision_config
    patch_dim = int(vc.in_channels) * int(vc.temporal_patch_size) * int(vc.patch_size) ** 2
    merge = int(vc.spatial_merge_size) ** 2

    # 1. warm every bucket under the cap (what the served engine does before its trace captures)
    done = warm_tower_sizes(lambda pv, grid: tower(pv, grid), patch_dim, log=print)
    assert [d[0] for d in done] == padded_patch_buckets(), done
    ttnn.synchronize_device(mesh_device)
    n_prog = mesh_device.num_program_cache_entries()
    print(
        f"VISION_SIZES warmed {len(done)} buckets {[d[0] for d in done]} in {sum(d[1] for d in done):.1f}s; programs {n_prog}"
    )

    # 2. real images through the CAPPED processor (as TT_Qwen3_5MoeProcessingInfo serves them)
    from transformers import AutoImageProcessor

    snap = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3.5-122B-A10B/snapshots/*"))[0]
    proc = AutoImageProcessor.from_pretrained(snap, max_pixels=max_pixels())
    cases = {
        "1080p": [(1080, 1920)],
        "cap-max": [(2048, 2048)],
        "phone-photo": [(3024, 4032)],
        "two-images": [(768, 1024), (1080, 1920)],
    }
    for name, hws in cases.items():
        imgs = [_image(h, w, seed=h + w) for (h, w) in hws]
        inputs = proc(images=imgs, return_tensors="pt")
        pv, grid = inputs["pixel_values"].float(), inputs["image_grid_thw"].to(torch.int64)
        per_image = [int(g.prod()) for g in grid]
        assert max(per_image) <= max_patches(), (name, per_image, max_patches())
        tt_out = tower(pv, grid)
        ttnn.synchronize_device(mesh_device)
        n_now = mesh_device.num_program_cache_entries()
        print(f"VISION_SIZES {name}: grids {grid.tolist()} patches {per_image} -> programs {n_prog} -> {n_now}")
        assert n_now == n_prog, f"{name}: {n_now - n_prog} program(s) compiled for an image size inside the cap"
        if name in ("1080p", "two-images"):
            n_tokens = sum(per_image) // merge
            with torch.no_grad():
                ref = visual(pv, grid).pooler_output.float()
            got = ttnn.to_torch(tt_out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=-1)).float()
            got = got.reshape(-1, got.shape[-1])[:n_tokens]
            assert ref.shape == got.shape, (ref.shape, got.shape)
            ok, pcc = comp_pcc(ref, got, 0.85)  # the bfp8 tower's bar (test_vision_tower)
            print(f"VISION_SIZES {name}: {n_tokens} image tokens, PCC vs HF visual {pcc}")
            assert ok, (name, pcc)
        ttnn.deallocate(tt_out)
    print("VISION_SIZES all sizes under the cap served by the warmed programs")
