# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""tt/bconv.py (one-program batched conv step) vs the ttnn-op BatchedConv path, several steps, B in (1, 4, 32),
C in (2304, 1024), x bf16: outputs and carried states."""

import importlib
import os

import torch

import ttnn


def main():
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    worst = 0.0
    for C in (2304, 1024):
        for B in (1, 4, 16, 32):
            taps = torch.randn(C, 4) * 0.5
            res = {}
            for mode in ("0", "1"):
                os.environ["I2M_BCONV_KERNEL"] = mode
                import models.autoports.thinkingmachines_inkling_2_mini.tt.batched as bm

                importlib.reload(bm)
                conv = bm.BatchedConv(mesh, taps, None, C, B)
                torch.manual_seed(0)
                ys = []
                for step in range(5):
                    x = ttnn.from_torch(torch.randn(1, 1, B, C).to(torch.bfloat16), dtype=ttnn.bfloat16,
                                        layout=ttnn.TILE_LAYOUT, device=mesh)
                    ys.append(ttnn.to_torch(conv.step(x)).float())
                st = [ttnn.to_torch(t).float() for t in conv.s]
                res[mode] = (torch.stack(ys), st)
            dy = (res["0"][0] - res["1"][0]).abs().max().item()
            ds = max((a - b).abs().max().item() for a, b in zip(res["0"][1], res["1"][1]))
            worst = max(worst, dy, ds)
            print(f"C={C} B={B}: max |y diff| {dy:.2e}, max |state diff| {ds:.2e}", flush=True)
    ttnn.close_mesh_device(mesh)
    print(f"{'PASS' if worst < 0.02 else 'FAIL'}: worst diff {worst:.2e}")


if __name__ == "__main__":
    main()
