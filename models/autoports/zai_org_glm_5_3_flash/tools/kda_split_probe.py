# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage-3 review #1 / stage-2 hand-off: the KDA input projection stored as two tensors (qkv [4096 x 24576] + rest
[4096 x 8384]) instead of one packed [4096 x 32960], at the optimized policy's weight dtype (bfp8, HiFi4, fp32
accumulation, the decoder's default ttnn.linear program configs). Host-timed, synchronized, us per call:
  prefill 2048 rows: packed + slice(qkv) + untilize vs qkv + rest + untilize (the stage-2 narrower-matmul pattern)
  decode 1 / 32 rows: one packed matmul vs two matmuls (qkv, rest)
usage: python -m models.autoports.zai_org_glm_5_3_flash.tools.kda_split_probe"""

import torch

import ttnn
from models.autoports.zai_org_glm_5_3_flash.tests.harness import open_mesh
from models.autoports.zai_org_glm_5_3_flash.tools.fusing_probe import DRAM, _run


def main():
    mesh = open_mesh()
    g = torch.Generator().manual_seed(0)
    try:
        Hd, P, W = 4096, 32960, 24576
        ckc = ttnn.init_device_compute_kernel_config(
            mesh.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True
        )
        w = (torch.randn(Hd, P, generator=g) * 0.02).bfloat16()
        upw = lambda t: ttnn.from_torch(
            t, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=DRAM
        )
        tw, tq, tr = upw(w), upw(w[:, :W].contiguous()), upw(w[:, W:].contiguous())
        lin = lambda a, b: ttnn.linear(a, b, compute_kernel_config=ckc, memory_config=DRAM)
        RM = ttnn.ROW_MAJOR_LAYOUT
        for T in (2048, 32, 1):
            x = (torch.randn(1, 1, T, Hd, generator=g) * 0.05).bfloat16()
            tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=DRAM)
            if T == 2048:
                variants = {
                    "packed + slice + untilize": lambda: ttnn.to_layout(
                        ttnn.slice(lin(tx, tw), (0, 0, 0, 0), (1, 1, T, W), memory_config=DRAM), RM, memory_config=DRAM
                    ),
                    "qkv + rest + untilize": lambda: (lin(tx, tr), ttnn.to_layout(lin(tx, tq), RM, memory_config=DRAM))[
                        1
                    ],
                    "packed proj alone": lambda: lin(tx, tw),
                    "qkv proj alone": lambda: lin(tx, tq),
                    "rest proj alone": lambda: lin(tx, tr),
                }
                exact = "packed + slice + untilize"
            else:
                variants = {
                    "packed (one matmul)": lambda: ttnn.slice(
                        lin(tx, tw), (0, 0, 0, 0), (1, 1, T, W), memory_config=DRAM
                    ),
                    "qkv + rest (two matmuls)": lambda: (lin(tx, tr), lin(tx, tq))[1],
                    "packed proj alone": lambda: lin(tx, tw),
                }
                exact = "packed (one matmul)"
            _run(mesh, f"kda in-proj bfp8 rows {T}", variants, 10 if T == 2048 else 50, exact_vs=exact)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
