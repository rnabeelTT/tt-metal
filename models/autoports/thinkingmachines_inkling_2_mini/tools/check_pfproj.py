# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""tt/attention.py proj()'s prefill configs (pf_progcfg, I2M_PF_PROJ_CFG) vs ttnn's default config on one die: relative
error of each vs the fp32 torch product and of one vs the other, per (TP shard, rows, projection).
usage: tools/run_dev.sh python .../tools/check_pfproj.py   env: TPS (4,2,1), TS (256,512,1024,2048)"""

import os

import torch

import ttnn
from models.autoports.thinkingmachines_inkling_2_mini.tt import attention as A

NH, NKV, HD, DR, H = 24, 8, 128, 16, 2304


def main():
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    ckc = ttnn.init_device_compute_kernel_config(mesh.arch(), math_fidelity=ttnn.MathFidelity.HiFi2,
                                                 math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True)
    DRAM = ttnn.DRAM_MEMORY_CONFIG

    class W:
        pass

    for tp in [int(t) for t in os.environ.get("TPS", "4,2,1").split(",")]:
        nh, nkv = NH // tp, NKV // tp
        fq = nh * HD + nh * DR + 32 + nh * HD
        fq = -(-fq // 256) * 256
        for T in [int(t) for t in os.environ.get("TS", "256,512,1024,2048").split(",")]:
            for name, K, N, odt in (("q", H, fq, ttnn.float32), ("k", H, nkv * HD, None), ("o", nh * HD, H, None)):
                xh = torch.randn(1, 1, T, K)
                wh = torch.randn(1, 1, K, N) * K**-0.5
                x = ttnn.from_torch(xh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=DRAM)
                wt = ttnn.from_torch(wh, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=mesh,
                                     memory_config=DRAM)
                ref = (x_ := ttnn.to_torch(x).float()) @ ttnn.to_torch(wt).float()
                w = W()
                w.w_q = wt
                w.dram_sharded, w.ckc, w.mesh = False, ckc, mesh
                A.PF_PROJ_CFG = False
                d = ttnn.to_torch(A.proj(w, "q", x, odt)).float()
                A.PF_PROJ_CFG = True
                c = ttnn.to_torch(A.proj(w, "q", x, odt)).float()
                rel = lambda a, b: ((a - b).norm() / b.norm()).item()  # noqa: E731
                print(f"TP{tp} T={T} {name} K={K} N={N}: default {rel(d, ref):.2e} cfg {rel(c, ref):.2e} "
                      f"cfg vs default {rel(c, d):.2e} bitwise {torch.equal(c, d)}", flush=True)
                for t in (x, wt):
                    ttnn.deallocate(t)
    print("== check_pfproj done", flush=True)
    from models.autoports.thinkingmachines_inkling_2_mini.tt.clean_exit import close

    close(mesh)


if __name__ == "__main__":
    main()
