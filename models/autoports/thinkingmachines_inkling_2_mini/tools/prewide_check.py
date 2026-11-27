# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""PreKernel at 17..32 rows (two 16-row programs, PRE_ROW0) vs the 16-row program on each half's rows as their own
tensors / conv states: x (tail), xn (bound), the NQ outputs and the advanced conv states must be bitwise. One die.
Cases: NQ 1 / 3, with / without the tail, norm on / off, bound on / off. env: TS (17,24,32), NB (3)"""

import os

import torch

import ttnn
from models.autoports.thinkingmachines_inkling_2_mini.tt.batched import BatchedConv
from models.autoports.thinkingmachines_inkling_2_mini.tt.post import PostKernel
from models.autoports.thinkingmachines_inkling_2_mini.tt.pre import PreKernel


def main():
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    H, NB = 2304, int(os.environ.get("NB", "3"))
    pk = PostKernel(mesh, H, 1e-6)
    k = PreKernel(pk)
    L1 = ttnn.L1_MEMORY_CONFIG
    up = lambda t, dt, mc=L1: ttnn.from_torch(t.contiguous(), dtype=dt, layout=ttnn.TILE_LAYOUT, device=mesh,  # noqa: E731
                                              memory_config=mc)
    back = lambda t: ttnn.to_torch(t).float()  # noqa: E731
    torch.manual_seed(0)
    taps = torch.randn(H, 4) * 0.3
    rw = lambda t: pk.rows_weight(ttnn.from_torch(t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh))  # noqa
    qs = [rw(torch.randn(1, 1, 1, H) * 0.02) for _ in range(3)]
    nws = [rw(1 + 0.1 * torch.randn(1, 1, 1, H)) for _ in range(3)]
    bad = 0
    for T in [int(v) for v in os.environ.get("TS", "17,24,32").split(",")]:
        x, mo, resid = torch.randn(1, 1, T, H), torch.randn(1, 1, T, H), torch.randn(1, 1, T, H) * 3
        blocks = [torch.randn(1, 1, T, H) for _ in range(NB)]
        lgs = [torch.randn(1, 1, T, 128) for _ in range(NB)]
        st = [torch.randn(1, 1, T, H) for _ in range(3)]
        for NQ, tail, norm, bound in ((1, True, True, False), (3, True, True, True), (3, False, False, False),
                                      (1, False, False, True), (3, True, False, False)):
            def run(a, b):
                n = b - a
                conv = BatchedConv(mesh, taps, None, H, n) if tail else None
                if conv is not None:
                    conv.s = [up(s[:, :, a:b], ttnn.float32, ttnn.DRAM_MEMORY_CONFIG) for s in st]
                xin = up((mo if tail else x)[:, :, a:b], ttnn.bfloat16)
                tl = (xin, up(resid[:, :, a:b], ttnn.float32), conv) if tail else None
                o = k(xin, [up(t[:, :, a:b], ttnn.float32) for t in blocks],
                      [(up(t[:, :, a:b], ttnn.float32), 1) for t in lgs], qs[:NQ], nws[:NQ] if norm else None,
                      tail=tl, bound=bound)
                xo, xn, outs = o
                res = {"x": back(xo)[:, :, :n]} if tail else {}
                if bound:
                    res["xn"] = back(xn)[:, :, :n]
                for j, t in enumerate(outs):
                    res[f"o{j}"] = back(t)[:, :, :n]
                if conv is not None:
                    for j, s in enumerate(conv.s):
                        res[f"s{j}"] = back(s)[:, :, :n]
                return res

            got = run(0, T)
            h0, h1 = run(0, 16), run(16, T)
            for name in got:
                ref = torch.cat([h0[name], h1[name]], 2)
                d = (got[name] != ref).sum().item()
                bad += d
                if d:
                    print(f"T={T} NQ={NQ} tail={tail} norm={norm} bound={bound} {name}: mismatches {d} "
                          f"max|d| {(got[name] - ref).abs().max().item():.3g}")
            print(f"T={T} NQ={NQ} tail={tail} norm={norm} bound={bound}: {sorted(got)} checked", flush=True)
    print("PASS" if bad == 0 else f"FAIL {bad}")
    ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
