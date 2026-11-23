# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Debug: the merged fused MoE (sub-shards as virtual experts) vs the per-sub-shard calls, layer 0, one die."""

import torch

import ttnn
from models.autoports.thinkingmachines_inkling_2_mini.tt import moe_fused as mf
from models.autoports.thinkingmachines_inkling_2_mini.tt.moe import MoE
from models.autoports.thinkingmachines_inkling_2_mini.tt.weights import Checkpoint, load_config


def pcc(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def main():
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    cfg, ck = load_config(), Checkpoint()
    li = 0
    moe = MoE(mesh, cfg, ck, li, expert_dtype=ttnn.bfloat8_b, cache_dir="/cache/wcache")
    mf.MERGE = False
    f0 = mf.FusedExperts(moe, ck, li, dtype=ttnn.bfloat8_b, cache_dir="/cache/wcache")
    mf.MERGE = True
    f1 = mf.FusedExperts(moe, ck, li, dtype=ttnn.bfloat8_b, cache_dir="/cache/wcache")
    assert f1.merged and not f0.merged
    # weights: virtual expert e * 3 + s of the stacked tensor == sub-shard s's expert e
    ug1 = ttnn.to_torch(f1.subs[0][1]).float()
    dn1 = ttnn.to_torch(f1.subs[0][2]).float()
    for s in range(3):
        ug0 = ttnn.to_torch(f0.subs[s][1]).float()
        dn0 = ttnn.to_torch(f0.subs[s][2]).float()
        for e in (0, 5, 127):
            print(f"sub {s} expert {e}: ug max|d| {(ug1[0, e * 3 + s] - ug0[0, e]).abs().max():.4g}, "
                  f"dn max|d| {(dn1[0, e * 3 + s] - dn0[0, e]).abs().max():.4g}", flush=True)
    torch.manual_seed(0)
    for M in (1, 2, 4):
        x = ttnn.from_torch(torch.randn(1, 1, M, cfg.hidden_size).to(torch.bfloat16), dtype=ttnn.bfloat16,
                            layout=ttnn.TILE_LAYOUT, device=mesh)
        a = ttnn.to_torch(f0.forward(x)).float().reshape(M, -1)
        b = ttnn.to_torch(f1.forward(x)).float().reshape(M, -1)
        print(f"M={M}: merged vs per-sub pcc {pcc(a, b):.6f}, norm ratio {b.norm() / a.norm():.4f}, "
              f"max|d| {(a - b).abs().max():.4g}", flush=True)
    ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
