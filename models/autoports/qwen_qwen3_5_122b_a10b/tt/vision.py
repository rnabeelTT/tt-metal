# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Vision tower for Qwen3.5-122B-A10B: the checkpoint's ``model.visual.*`` ViT (depth 27, hidden 1152, patch 16,
spatial merge 2 -> 3072-wide merged tokens) on the mesh through the in-tree qwen36 port
(``models/demos/blackhole/qwen36/tt/vision``: DropInVisionTransformer keeps the HF module for the patch embed and the
position-embedding interpolation on host and runs the 27 blocks + merger on device).

The HF visual module is built STANDALONE from the streaming state dict (333 tensors, ~0.9 GB bf16): the stock
``VisionModelArgs.reference_vision_model`` would ``from_pretrained`` the whole 250 GB checkpoint.
"""
from __future__ import annotations

import os

import torch
from loguru import logger

import ttnn


def build_hf_visual(state_dict, hf_config, dtype=torch.float32, strict=True):
    """``Qwen3_5MoeVisionModel`` with the checkpoint's weights, from ``state_dict.visual_keys()`` /
    ``visual_tensor()`` (``tt/weights.py::StreamingStateDict``). Float32 by default: the wrapper's host-side patch embed
    and position interpolation (and the PCC reference) run on it."""
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeVisionModel

    visual = Qwen3_5MoeVisionModel(hf_config.vision_config)
    depth = int(hf_config.vision_config.depth)
    keys = [k for k in state_dict.visual_keys() if not k.startswith("blocks.") or int(k.split(".")[1]) < depth]
    sd = {k: state_dict.visual_tensor(k).to(dtype) for k in keys}
    missing, unexpected = visual.load_state_dict(sd, strict=False)
    missing = [k for k in missing if "inv_freq" not in k]  # rotary inv_freq buffers are computed, not stored
    if strict:
        assert not missing and not unexpected, f"visual weights: missing {missing[:5]} unexpected {unexpected[:5]}"
    visual = visual.to(dtype).eval()
    logger.info(f"HF visual built from {len(sd)} streamed tensors ({dtype})")
    return visual


def _vision_args_class():
    """VisionModelArgs over OUR args class: the stock one derives the text parameters (ffn_dim_multiplier ...) that the
    MoE text config does not carry; Qwen35MoeModelArgs synthesizes them, VisionModelArgs then overrides the dims with
    the vision_config's."""
    from models.autoports.qwen_qwen3_5_122b_a10b.tt.model_config import Qwen35MoeModelArgs
    from models.demos.blackhole.qwen36.tt.vision.vision_model_config import VisionModelArgs

    class Qwen35VisionModelArgs(VisionModelArgs, Qwen35MoeModelArgs):
        pass

    return Qwen35VisionModelArgs


def vision_args_for(mesh_device, args):
    """Vision args on this mesh (reads HF_MODEL / the HF config for the vision_config)."""
    cls = _vision_args_class()
    return cls(mesh_device, max_batch_size=args.max_batch_size, max_seq_len=args.max_seq_len)


def vision_weight_dtype():
    """QWEN35_VISION_DTYPE = bfp8 (default; tower PCC 0.986 vs HF once the padding windows are in) | bf16 (its tower
    construction stalls in the merger's as_tensor on this host - under investigation)."""
    return {"bfp8": ttnn.bfloat8_b, "bf16": ttnn.bfloat16}[os.environ.get("QWEN35_VISION_DTYPE", "bfp8")]


def attach_vision(model, args, dtype=None, tt_ccl=None):
    """Build the HF visual from the streaming weights and attach the TT tower to ``model`` (idempotent)."""
    if getattr(model, "vision_model", None) is not None:
        return model.vision_model
    dtype = dtype or vision_weight_dtype()
    visual = build_hf_visual(args.load_state_dict(), args.hf_config)
    vargs = vision_args_for(model.mesh_device, args)
    if tt_ccl is not None:
        from models.demos.blackhole.qwen36.tt.vision.model import DropInVisionTransformer

        model.vision_args = vargs
        model.vision_model = DropInVisionTransformer(visual, vargs, dtype=dtype, tt_ccl=tt_ccl)
        return model.vision_model
    return model.init_vision_model(reference_visual=visual, vision_args=vargs, dtype=dtype)


# ---- served image sizes -------------------------------------------------------------------------------------------
# The tower pads every image to the next 2048-patch multiple (DropInVisionTransformer.forward: `(n // 2048 + 1) * 2048`)
# and its programs are keyed by that padded length, so a request whose image lands in an unwarmed bucket compiles after
# the traces are parked (the program guard then stops the engine: served image failure of 2026-10-01, a photo of 47k
# patches against a warm-up of 2048 / 4096). The server caps the image resolution at QWEN35_VISION_MAX_PIXELS (default
# 4 MP = 16384 patches = 4096 image tokens per image; the checkpoint's processor default is 16 MP) and warms every padded
# bucket up to that cap.
PATCH_BLOCK = 2048
DEFAULT_MAX_PIXELS = 16384 * 256  # 16 x 16 pixels per patch


def max_pixels():
    return int(os.environ.get("QWEN35_VISION_MAX_PIXELS", str(DEFAULT_MAX_PIXELS)) or DEFAULT_MAX_PIXELS)


def max_patches(cap_pixels=None):
    return (cap_pixels or max_pixels()) // 256


def padded_patch_buckets(cap_pixels=None):
    """Every padded sequence length the tower can see under the cap (the tower's formula is strictly above n)."""
    n_max = max_patches(cap_pixels)
    return [k * PATCH_BLOCK for k in range(1, n_max // PATCH_BLOCK + 2)]


def warm_tower_sizes(run_tower, patch_dim, cap_pixels=None, log=None):
    """Compile the tower's programs for every padded bucket under the cap with synthetic patches (one grid per bucket:
    32 rows x n/32 columns, n inside the bucket). ``run_tower(pixel_values, grid_thw)`` runs the tower and returns a
    device tensor (freed here). Returns the list of (padded length, seconds)."""
    import time

    out = []
    for padded in padded_patch_buckets(cap_pixels):
        n = padded - PATCH_BLOCK // 2  # lands in this bucket: (n // 2048 + 1) * 2048 == padded
        grid = torch.tensor([[1, 32, n // 32]], dtype=torch.int64)
        t0 = time.perf_counter()
        rows = run_tower(torch.randn(n, patch_dim), grid)
        ttnn.deallocate(rows)
        dt = time.perf_counter() - t0
        out.append((padded, dt))
        if log is not None:
            log(f"vision warm-up: {n} patches -> padded {padded} in {dt:.1f}s")
    return out
