# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""tt/gatef.py GateFused against the op path (fp32 matmul g @ [Wf | Wg] -> tt/gatet.py GateTail): bitwise, timed.
env: ROWS (comma list, default 64,512,2048)"""

import os
import time

import torch

import ttnn
from models.autoports.thinkingmachines_inkling_2_mini.tests._devtime import devtime
from models.autoports.thinkingmachines_inkling_2_mini.tt.dmm import dmm
from models.autoports.thinkingmachines_inkling_2_mini.tt.gatef import GateFused
from models.autoports.thinkingmachines_inkling_2_mini.tt.gatet import GateTail

H, R = 2304, 64


def main():
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    ck = ttnn.init_device_compute_kernel_config(mesh.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
                                                math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False)
    up = lambda t, dt: ttnn.from_torch(t, dtype=dt, layout=ttnn.TILE_LAYOUT, device=mesh,  # noqa: E731
                                       memory_config=ttnn.DRAM_MEMORY_CONFIG)
    g_ = torch.Generator().manual_seed(0)
    wh = (torch.randn(1, 1, R, 2 * H, generator=g_) * 0.05).to(torch.bfloat16).float()
    w32, wbf = up(wh, ttnn.float32), up(wh, ttnn.bfloat16)
    gf, gt = GateFused(mesh, H, R), GateTail(mesh, H)
    ok = True
    for T in [int(v) for v in os.environ.get("ROWS", "64,512,2048").split(",")]:
        g = up(torch.randn(1, 1, T, R, generator=g_) * 3, ttnn.bfloat16)
        h = up(torch.randn(1, 1, T, H, generator=g_), ttnn.float32)
        m = up(torch.randn(1, 1, T, H, generator=g_), ttnn.float32)
        mc = ttnn.DRAM_MEMORY_CONFIG

        def old():
            fg = dmm(ttnn.typecast(g, ttnn.float32), w32, ck, dtype=ttnn.float32, memory_config=mc)
            return gt(fg, h, m, out_mc=mc)

        new = lambda: gf(g, wbf, h, m)  # noqa: E731
        a, b = ttnn.to_torch(old()).float(), ttnn.to_torch(new()).float()
        same = torch.equal(a, b)
        ok &= same
        print(f"T={T}: bitwise {same} maxdiff {(a - b).abs().max().item():.3g}", flush=True)
        dts = [devtime(mesh, f) for f in (new, old)]
        if dts[0] is not None:
            print(f"T={T}: GateFused {dts[0]:.1f} us, op path {dts[1]:.1f} us (device)", flush=True)
        res = []
        for f in (new, old):
            for _ in range(3):
                f()
            ttnn.synchronize_device(mesh)
            t0 = time.perf_counter()
            for _ in range(50):
                f()
            ttnn.synchronize_device(mesh)
            res.append((time.perf_counter() - t0) / 50 * 1e6)
        print(f"T={T}: GateFused {res[0]:.1f} us, op path {res[1]:.1f} us (wall per call)", flush=True)
    print("PASS" if ok else "FAIL", flush=True)
    print("== gatef done", flush=True)
    from models.autoports.thinkingmachines_inkling_2_mini.tt.clean_exit import close
    close(mesh)


if __name__ == "__main__":
    main()
