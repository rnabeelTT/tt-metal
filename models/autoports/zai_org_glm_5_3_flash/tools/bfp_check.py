# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only check (no device) that tests/hf_ref/bfp.py reproduces the ttnn host block-float packing bit-exactly on
real checkpoint weights: for each tensor, ttnn.from_torch(W^T, bfloat4_b / bfloat8_b, TILE) -> to_torch against
bfp.bfp_round(W, fmt, dim=0)^T. Also the 4-D packed expert layout of tt/moe.py ([1, E, H, 2N] up | gate).

usage: python -m models.autoports.zai_org_glm_5_3_flash.tools.bfp_check --layer 4 --experts 0,57,191"""

import argparse

import torch

import ttnn
from models.autoports.zai_org_glm_5_3_flash.tests.hf_ref.bfp import bfp_round
from models.autoports.zai_org_glm_5_3_flash.tt.ckpt import CheckpointReader


def ttnn_round(wt: torch.Tensor, dtype) -> torch.Tensor:
    t = ttnn.from_torch(wt.contiguous(), dtype=dtype, layout=ttnn.TILE_LAYOUT)
    return ttnn.to_torch(t).float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=4)
    ap.add_argument("--experts", default="0,57,191")
    a = ap.parse_args()
    rd = CheckpointReader()
    sd = rd.layer_state(a.layer, lazy=True)
    names = []
    for e in [int(x) for x in a.experts.split(",")]:
        names += [f"mlp.experts.{e}.{p}.weight" for p in ("gate_proj", "up_proj", "down_proj")]
    names += [f"mlp.shared_experts.{p}.weight" for p in ("gate_proj", "up_proj", "down_proj")]
    if a.layer < 3:
        names = [f"mlp.{p}.weight" for p in ("gate_proj", "up_proj", "down_proj")]
    bad = 0
    for n in names:
        w = sd[n]() if callable(sd[n]) else sd[n]
        for fmt, dt in (("bfp4", ttnn.bfloat4_b), ("bfp8", ttnn.bfloat8_b)):
            ref = ttnn_round(w.T, dt)
            emu = bfp_round(w, fmt, dim=0).float().T
            diff = int((ref != emu).sum())
            bad += diff
            print(f"{n} {fmt}: {tuple(w.shape)} mismatches {diff} (max |d| {float((ref - emu).abs().max()):.3e})")
    # tt/moe.py layout: [1, E, H, 2N] = per expert [up^T | gate^T], from_torch on the 4-D tensor
    e0 = [int(x) for x in a.experts.split(",")][:2]
    if a.layer >= 3:
        g = [sd[f"mlp.experts.{e}.gate_proj.weight"] for e in e0]
        u = [sd[f"mlp.experts.{e}.up_proj.weight"] for e in e0]
        g = [t() if callable(t) else t for t in g]
        u = [t() if callable(t) else t for t in u]
        packed = torch.stack([torch.cat([ui.T, gi.T], dim=-1) for ui, gi in zip(u, g)]).unsqueeze(0)
        ref = ttnn.to_torch(ttnn.from_torch(packed, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT)).float()
        emu = torch.stack(
            [torch.cat([bfp_round(ui, "bfp4", 0).T, bfp_round(gi, "bfp4", 0).T], dim=-1) for ui, gi in zip(u, g)]
        ).unsqueeze(0)
        diff = int((ref != emu.float()).sum())
        bad += diff
        print(f"4-D packed up|gate bfp4 {tuple(packed.shape)}: mismatches {diff}")
    print(f"TOTAL mismatches {bad}")
    assert bad == 0


if __name__ == "__main__":
    main()
