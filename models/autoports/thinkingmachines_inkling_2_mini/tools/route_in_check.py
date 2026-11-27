# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""POST_ROUTE check (tt/kernels/post_route.hpp): the post program's router round with the routing inside writes the
same route table bits as the logits + qwen36 RouteRows (sigb, k 8, E 128), T = 1 (dup) / 2 / 4 rows, random inputs
and bias, many draws (OP=post) or the MLP gate (OP=gate). One die. env: N (draws, 20), OP"""

import os

import torch

import ttnn
from models.demos.blackhole.qwen36.tt.moe_decode.union import RouteRows, upload_selection_bias


def table(mesh, m):
    return ttnn.from_torch(torch.zeros(m, 32, dtype=torch.int32), dtype=ttnn.uint16, layout=ttnn.ROW_MAJOR_LAYOUT,
                           device=mesh, memory_config=ttnn.L1_MEMORY_CONFIG)


def main():
    from models.autoports.thinkingmachines_inkling_2_mini.tt.batched import BatchedConv
    from models.autoports.thinkingmachines_inkling_2_mini.tt.post import PostKernel, Routed
    from models.autoports.thinkingmachines_inkling_2_mini.tt.gate import GateKernel2

    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    n, op, E, k, H = int(os.environ.get("N", "20")), os.environ.get("OP", "post"), 128, 8, 2304
    pk = PostKernel(mesh, H, 1e-6)
    gk = GateKernel2(pk, 64) if op == "gate" else None
    l1 = lambda t, dt: ttnn.from_torch(t, dtype=dt, layout=ttnn.TILE_LAYOUT, device=mesh,  # noqa: E731
                                       memory_config=ttnn.L1_MEMORY_CONFIG)
    up = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)  # noqa: E731
    bad = 0
    for T in (1, 2, 4):
        m = max(T, 2)
        rr = RouteRows(mesh, E, k, m, router="sigb")
        conv = BatchedConv(mesh, torch.randn(H, 4) * 0.3, None, H, T) if op == "post" else None
        for it in range(n):
            g = torch.Generator().manual_seed(1000 * T + it)
            bias = upload_selection_bias(mesh, torch.randn(E, generator=g) * 0.05)
            rt = (up(torch.randn(1, 1, H, E, generator=g) * 0.02), up(torch.randn(1, 1, H, E, generator=g) * 1e-4))
            nw = pk.rows_weight(ttnn.from_torch(torch.ones(1, 1, 1, H), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
                                                device=mesh))
            ta, tb = table(mesh, m), table(mesh, m)
            if op == "post":
                att, x = l1(torch.randn(1, 1, T, H, generator=g), ttnn.bfloat16), l1(torch.randn(1, 1, T, H, generator=g), ttnn.bfloat16)
                q = ttnn.from_torch(torch.randn(1, 1, 32, H, generator=g) * 0.02, dtype=ttnn.float32,
                                    layout=ttnn.TILE_LAYOUT, device=mesh)
                blocks = [l1(torch.randn(1, 1, T, H, generator=g), ttnn.float32) for _ in range(3)]
                lgs = [(l1(torch.randn(1, 1, T, 64, generator=g), ttnn.float32), 1) for _ in range(3)]
                st = [ttnn.clone(s) for s in conv.s]
                _, _, lg = pk(att, x, conv, q, blocks, lgs, nw=nw, router=rt, dup=T == 1)
                for s, s0 in zip(conv.s, st):  # (the same conv states for the second run)
                    ttnn.copy(s0, s)
                _, _, rd = pk(att, x, conv, q, blocks, lgs, nw=nw, router=rt, dup=T == 1, route=(tb, bias, k))
            else:
                h, mm = l1(torch.randn(1, 1, T, H, generator=g), ttnn.float32), l1(torch.randn(1, 1, T, H, generator=g), ttnn.float32)
                wd = up(torch.randn(1, 1, 2 * H, 64, generator=g) * 0.05)
                wfg = up(torch.randn(1, 1, 64, 2 * H, generator=g) * 0.3)
                _, lg = gk(h, mm, wd, wfg, nw=nw, router=rt, dup=T == 1)
                _, rd = gk(h, mm, wd, wfg, nw=nw, router=rt, dup=T == 1, route=(tb, bias, k))
            assert isinstance(rd, Routed), type(rd)
            rr(lg, ta, bias=bias)
            a = ttnn.to_torch(ta).reshape(-1, 32)[:m]
            b = ttnn.to_torch(tb).reshape(-1, 32)[:m]
            if not torch.equal(a, b):
                bad += 1
                if bad < 4:
                    print("MISMATCH T", T, it, "\n", a[:, :25], "\n", b[:, :25])
        print(f"T={T}: {n} draws, mismatches so far {bad}")
    ttnn.close_mesh_device(mesh)
    print("ROUTE_IN_CHECK", "PASS" if bad == 0 else f"FAIL {bad}")


if __name__ == "__main__":
    main()
