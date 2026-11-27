# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""GateKernel2 at 17..32 rows (two 16-row halves: tt/gate.py halves / multi) vs the 16-row kernel run on each half's
rows as their own tensors: must be bitwise (the kernel is row-independent). One die. Also times each path (us / call,
eager, synchronized). env: TS (rows, default 17,24,32), N (timing calls, 20)"""

import os
import time

import torch

import ttnn
from models.autoports.thinkingmachines_inkling_2_mini.tt.gate import GateKernel, GateKernel2
from models.autoports.thinkingmachines_inkling_2_mini.tt.post import PostKernel


def main():
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    H, Rk = 2304, 64
    pk = PostKernel(mesh, H, 1e-6)
    k = GateKernel2(pk, Rk)
    old = GateKernel(mesh, H, Rk, 1e-6)
    L1 = ttnn.L1_MEMORY_CONFIG
    up = lambda t, dt, mc=None: ttnn.from_torch(t, dtype=dt, layout=ttnn.TILE_LAYOUT, device=mesh,  # noqa: E731
                                                memory_config=mc)
    back = lambda t: ttnn.to_torch(t).float()  # noqa: E731
    torch.manual_seed(0)
    gates = [(up(torch.randn(1, 1, 2 * H, Rk) * 0.05, ttnn.bfloat16), up(torch.randn(1, 1, Rk, 2 * H) * 0.3, ttnn.bfloat16))
             for _ in range(3)]
    nws = [pk.rows_weight(up(1 + 0.1 * torch.randn(1, 1, 1, H), ttnn.float32)) for _ in range(3)]
    bad = 0
    n = int(os.environ.get("N", "20"))
    for T in [int(v) for v in os.environ.get("TS", "17,24,32").split(",")]:
        hs = [torch.randn(1, 1, T, H) for _ in range(3)]
        ms = [torch.randn(1, 1, T, H) for _ in range(3)]
        h = [up(t, ttnn.float32, L1) for t in hs]
        m = [up(t, ttnn.float32, L1) for t in ms]
        sl = lambda ts, a, b: [up(t[:, :, a:b], ttnn.float32, L1) for t in ts]  # noqa: E731
        h0, m0, h1, m1 = sl(hs, 0, 16), sl(ms, 0, 16), sl(hs, 16, T), sl(ms, 16, T)
        for norm in (False, True):
            nw = nws[0] if norm else None
            o = k.halves(h[0], m[0], *gates[0], nw=nw)
            o = back(o[0] if norm else o)[:, :, :T]
            r0 = k(h0[0], m0[0], *gates[0], nw=nw)
            r1 = k(h1[0], m1[0], *gates[0], nw=nw)
            ref = torch.cat([back(r0[0] if norm else r0)[:, :, :16], back(r1[0] if norm else r1)[:, :, : T - 16]], 2)
            d = (o != ref).sum().item()
            bad += d
            print(f"halves T={T} norm={norm}: mismatches {d} max|d| {(o - ref).abs().max().item():.3g}")
        # multi (three gates with norms)
        args = [(h[i], m[i], *gates[i], nws[i]) for i in range(3)]
        outs = k.multi(args)
        # (reference: multi on each half's rows: gate i runs on rectangle i in both, the same per-core tiles)
        ref0 = k.multi([(h0[i], m0[i], *gates[i], nws[i]) for i in range(3)])
        ref1 = k.multi([(h1[i], m1[i], *gates[i], nws[i]) for i in range(3)])
        for i in range(3):
            o = back(outs[i][0])[:, :, :T]
            r0 = back(ref0[i][0])[:, :, :16]
            r1 = back(ref1[i][0])[:, :, : T - 16]
            d = (o != torch.cat([r0, r1], 2)).sum().item()
            bad += d
            print(f"multi T={T} gate {i}: mismatches {d}")
        # vs the old GateKernel (numerics differ: max |d| only)
        oo = back(old(h[0], m[0], *gates[0]))[:, :, :T]
        nn = back(k.halves(h[0], m[0], *gates[0]))[:, :, :T]
        print(f"vs GateKernel T={T}: max|d| {(oo - nn).abs().max().item():.3g} rel {((oo - nn).norm() / oo.norm()).item():.3g}")
        # timing (eager, synchronized per batch of n)
        for name, fn in (("GateKernel", lambda: old(h[0], m[0], *gates[0])),
                         ("halves", lambda: k.halves(h[0], m[0], *gates[0])),
                         ("halves+norm", lambda: k.halves(h[0], m[0], *gates[0], nw=nws[0])),
                         ("multi3+norm", lambda: k.multi(args))):
            fn()
            ttnn.synchronize_device(mesh)
            t0 = time.perf_counter()
            for _ in range(n):
                fn()
            ttnn.synchronize_device(mesh)
            print(f"  T={T} {name}: {(time.perf_counter() - t0) / n * 1e6:.1f} us/call (eager wall)")
    print("PASS" if bad == 0 else f"FAIL {bad}")
    ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
