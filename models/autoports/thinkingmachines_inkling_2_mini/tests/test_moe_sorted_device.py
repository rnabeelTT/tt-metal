# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Prefill MoE: the sorted dispatch (tt/moe.py _sorted) vs the masked union path on real layer inputs (the golden
fixture's post-attention-layernorm rows of layer 0, T = 64 / 128 / 256), layers 0 and 23; pcc and wall time."""

import os
import time

import torch

import ttnn
import models.autoports.thinkingmachines_inkling_2_mini.tt.moe as mm
from models.autoports.thinkingmachines_inkling_2_mini.tt.moe import MoE
from models.autoports.thinkingmachines_inkling_2_mini.tt.weights import Checkpoint, load_config


def pcc(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def main():
    N = int(os.environ.get("MESH", "1"))
    if N > 1:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, N))
    rep = ttnn.ReplicateTensorToMesh(mesh) if N > 1 else None
    d0 = (lambda t: ttnn.get_device_tensors(t)[0]) if N > 1 else (lambda t: t)
    cfg, ck = load_config(), Checkpoint()
    g = torch.load("/cache/fixtures/golden_t256.pt")
    xs = g["internals"]["layers.0.post_attention_layernorm"][0][0]  # [T, H]
    worst = 1.0
    for li in (0, 23):
        moe = MoE(mesh, cfg, ck, li, expert_dtype=ttnn.bfloat8_b, cache_dir="/cache/wcache")
        for T in [int(t) for t in os.environ.get("TS", "64,128,256").split(",")]:
            xt = xs.repeat(-(-T // xs.shape[0]), 1)[:T]  # (the fixture has 256 rows)
            x = ttnn.from_torch(xt.reshape(1, 1, T, -1).to(torch.bfloat16), dtype=ttnn.bfloat16,
                                layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep)
            res, tm = {}, {}
            for mode in (False, True):
                mm.SORTED = mode
                res[mode] = ttnn.to_torch(d0(moe.forward(x))).float().reshape(T, -1)
                ttnn.synchronize_device(mesh)
                t0 = time.time()
                for _ in range(3):
                    moe.forward(x)
                ttnn.synchronize_device(mesh)
                tm[mode] = (time.time() - t0) / 3 * 1e3
            p = pcc(res[True], res[False])
            worst = min(worst, p)
            print(f"layer {li} T={T}: sorted vs masked pcc {p:.6f}  max abs {(res[True] - res[False]).abs().max():.3e}"
                  f"  time masked {tm[False]:.1f} ms, sorted {tm[True]:.1f} ms", flush=True)
    ttnn.close_mesh_device(mesh)
    print(f"{'PASS' if worst > 0.999 else 'FAIL'}: worst pcc {worst:.6f}")


if __name__ == "__main__":
    main()
