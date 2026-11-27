# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""PostKernel at 17..32 rows (two 16-row programs, POST_ROW0) vs the 16-row program on each half's rows as their own
tensors / conv states: resid, out and the advanced conv states must be bitwise. One die. env: TS (17,24,32), NB (3)"""

import os

import torch

import ttnn
from models.autoports.thinkingmachines_inkling_2_mini.tt.batched import BatchedConv
from models.autoports.thinkingmachines_inkling_2_mini.tt.post import PostKernel


def main():
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    H, NB = 2304, int(os.environ.get("NB", "3"))
    k = PostKernel(mesh, H, 1e-6)
    L1 = ttnn.L1_MEMORY_CONFIG
    up = lambda t, dt, mc=L1: ttnn.from_torch(t.contiguous(), dtype=dt, layout=ttnn.TILE_LAYOUT, device=mesh,  # noqa: E731
                                              memory_config=mc)
    back = lambda t: ttnn.to_torch(t).float()  # noqa: E731
    torch.manual_seed(0)
    taps = torch.randn(H, 4) * 0.3
    # (the query as rows: replicated over the 32 tile rows, like the model's rows_weight)
    q = k.rows_weight(ttnn.from_torch(torch.randn(1, 1, 1, H) * 0.02, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
                                      device=mesh))
    nw = k.rows_weight(ttnn.from_torch(1 + 0.1 * torch.randn(1, 1, 1, H), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
                                       device=mesh))
    bad = 0
    for T in [int(v) for v in os.environ.get("TS", "17,24,32").split(",")]:
        att, x = torch.randn(1, 1, T, H), torch.randn(1, 1, T, H)
        blocks = [torch.randn(1, 1, T, H) for _ in range(NB)]
        lgs = [torch.randn(1, 1, T, 64) for _ in range(NB)]
        st = [torch.randn(1, 1, T, H) for _ in range(3)]

        def run(a, b):
            n = b - a
            conv = BatchedConv(mesh, taps, None, H, n)
            conv.s = [up(s[:, :, a:b], ttnn.float32, ttnn.DRAM_MEMORY_CONFIG) for s in st]
            ins = (up(att[:, :, a:b], ttnn.bfloat16), up(x[:, :, a:b], ttnn.bfloat16), conv, q,
                   [up(t[:, :, a:b], ttnn.float32) for t in blocks], [(up(t[:, :, a:b], ttnn.float32), 1) for t in lgs])
            return conv, ins

        for norm in (False, True):
            conv, ins = run(0, T)
            r, o = k(*ins, nw=nw if norm else None)
            got = [back(r)[:, :, :T], back(o)[:, :, :T]] + [back(s)[:, :, :T] for s in conv.s]
            refs = []
            for a, b in ((0, 16), (16, T)):
                c2, i2 = run(a, b)
                r2, o2 = k(*i2, nw=nw if norm else None)
                refs.append([back(r2)[:, :, : b - a], back(o2)[:, :, : b - a]] + [back(s)[:, :, : b - a] for s in c2.s])
            for j, name in enumerate(["resid", "out", "s0", "s1", "s2"]):
                ref = torch.cat([refs[0][j], refs[1][j]], 2)
                d = (got[j] != ref).sum().item()
                bad += d
                print(f"T={T} norm={norm} {name}: mismatches {d} max|d| {(got[j] - ref).abs().max().item():.3g}")
        if os.environ.get("DIAG") == "1":  # (which input half 1's out came from)
            T2 = T - 16
            conv, ins = run(0, T)
            r, o = k(*ins)
            o1 = back(o)[:, :, 16:T]
            for bl, lgr in (((16, T), (16, T)), ((0, T2), (16, T)), ((16, T), (0, T2)), ((0, T2), (0, T2))):
                c2 = BatchedConv(mesh, taps, None, H, T2)
                c2.s = [up(s_[:, :, 16:T], ttnn.float32, ttnn.DRAM_MEMORY_CONFIG) for s_ in st]
                i2 = (up(att[:, :, 16:T], ttnn.bfloat16), up(x[:, :, 16:T], ttnn.bfloat16), c2, q,
                      [up(t[:, :, bl[0]:bl[1]], ttnn.float32) for t in blocks],
                      [(up(t[:, :, lgr[0]:lgr[1]], ttnn.float32), 1) for t in lgs])
                r2, o2 = k(*i2)
                print(f"DIAG T={T} blocks rows {bl} logits rows {lgr}: out mismatches {(back(o2)[:, :, :T2] != o1).sum().item()}")
    print("PASS" if bad == 0 else f"FAIL {bad}")
    ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
