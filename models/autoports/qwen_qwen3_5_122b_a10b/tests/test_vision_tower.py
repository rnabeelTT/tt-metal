# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Vision tower on the (1,4) mesh vs the HF visual module built from the checkpoint's streamed ``model.visual.*`` weights
(no text model). Images go through the checkpoint's image processor; the TT output (merged tokens, hidden-fractured
over the mesh) must match the HF ``pooler_output`` row for row."""
import glob
import os

import numpy as np
import pytest
import torch
from loguru import logger

import ttnn
from models.autoports.qwen_qwen3_5_122b_a10b.tt.model_config import Qwen35MoeModelArgs
from models.autoports.qwen_qwen3_5_122b_a10b.tt.vision import build_hf_visual, vision_args_for
from models.common.utility_functions import comp_pcc


def _image(h, w, seed):
    from PIL import Image

    rng = np.random.default_rng(seed)
    # a smooth image (random noise gives near-zero attention structure): low-frequency blobs + noise
    yy, xx = np.mgrid[0:h, 0:w]
    base = np.stack([np.sin(yy / (37.0 + 11 * c)) * np.cos(xx / (23.0 + 7 * c)) for c in range(3)], -1)
    arr = ((base * 0.5 + 0.5) * 200 + rng.random((h, w, 3)) * 40).clip(0, 255).astype("uint8")
    return Image.fromarray(arr)


@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "l1_small_size": 24576, "trace_region_size": 64 * 1024 * 1024}],
    indirect=True,
)
@pytest.mark.parametrize("hw", [(448, 448), (896, 672)], ids=["448x448", "896x672"])
@pytest.mark.parametrize("wdtype", [ttnn.bfloat16, ttnn.bfloat8_b], ids=["bf16", "bfp8"])
def test_vision_tower_matches_hf(mesh_device, hw, wdtype):
    os.environ.setdefault("MESH_DEVICE", "P300x2")
    args = Qwen35MoeModelArgs(mesh_device, max_batch_size=1, max_seq_len=4096)
    depth = int(os.environ.get("QWEN35_VISION_TEST_DEPTH", "0") or 0)  # 0 = the full 27-block tower
    if depth:
        args.hf_config.vision_config.depth = depth
    visual = build_hf_visual(args.load_state_dict(), args.hf_config, strict=not depth)
    vargs = vision_args_for(mesh_device, args)
    if depth:
        vargs.hf_config.vision_config.depth = depth
    from models.demos.blackhole.qwen36.tt.vision.model import DropInVisionTransformer
    from models.tt_transformers.tt.ccl import TT_CCL

    tower = DropInVisionTransformer(visual, vargs, dtype=wdtype, tt_ccl=TT_CCL(mesh_device))

    from transformers import AutoImageProcessor

    snap = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3.5-122B-A10B/snapshots/*"))[0]
    proc = AutoImageProcessor.from_pretrained(snap)
    inputs = proc(images=_image(*hw, seed=hw[0]), return_tensors="pt")
    pixel_values, grid = inputs["pixel_values"].float(), inputs["image_grid_thw"]
    n_tokens = int(grid.prod()) // (args.hf_config.vision_config.spatial_merge_size**2)
    with torch.no_grad():
        ref = visual(pixel_values, grid).pooler_output.float()  # [n_tokens, 3072]
    tt_out = tower(pixel_values, grid)
    got = ttnn.to_torch(tt_out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=-1)).float()
    got = got.reshape(-1, got.shape[-1])[:n_tokens]
    assert ref.shape == got.shape, (ref.shape, got.shape)
    bar = 0.97 if wdtype == ttnn.bfloat16 else 0.85  # the in-tree 27-block tower test takes 0.91 at bfp8
    ok, pcc = comp_pcc(ref, got, bar)
    worst = min(comp_pcc(ref[i], got[i], 0.9)[1] for i in range(0, n_tokens, max(1, n_tokens // 16)))
    print(
        f"VISION_TOWER {wdtype} {hw[0]}x{hw[1]} grid {grid.tolist()} tokens {n_tokens}: PCC {pcc} (sampled rows min {worst:.4f})"
    )
    logger.info(f"VISION_TOWER {hw}: PCC {pcc}")
    assert ok, pcc
