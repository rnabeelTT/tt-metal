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
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    cfg, ck = load_config(), Checkpoint()
    g = torch.load("/cache/fixtures/golden_t256.pt")
    xs = g["internals"]["layers.0.post_attention_layernorm"][0][0]  # [T, H]
    worst = 1.0
    for li in (0, 23):
        moe = MoE(mesh, cfg, ck, li, expert_dtype=ttnn.bfloat8_b, cache_dir="/cache/wcache")
        for T in (64, 128, 256):
            x = ttnn.from_torch(xs[:T].reshape(1, 1, T, -1).to(torch.bfloat16), dtype=ttnn.bfloat16,
                                layout=ttnn.TILE_LAYOUT, device=mesh)
            res, tm = {}, {}
            for mode in (False, True):
                mm.SORTED = mode
                res[mode] = ttnn.to_torch(moe.forward(x)).float().reshape(T, -1)
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
