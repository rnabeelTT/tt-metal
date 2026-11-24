# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""tt/gmm.py act mode (the activation and routing weights inside the gate|up grouped matmul, merged sub-shards) vs the
unfused path (gate|up grouped matmul per sub-shard + tt/moeact.py): bitwise on the real groups (single- and
double-buffered weights); then the down projection per sub-shard reading the act output's columns in place (bitwise)."""

import time

import torch

import ttnn
import models.autoports.thinkingmachines_inkling_2_mini.tt.gmm as gm
from models.autoports.thinkingmachines_inkling_2_mini.tt.gmm import GroupedMM
from models.autoports.thinkingmachines_inkling_2_mini.tt.moeact import MoEAct


def main():
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    torch.manual_seed(0)
    E, S, W, K, H, gs = 8, 3, 384, 2304, 2304, 32
    experts = sorted(torch.randint(0, E, (30,)).tolist())
    ids = torch.tensor(experts + [E, E])  # (padding groups)
    G = len(ids)
    real = G - 2
    dram = ttnn.DRAM_MEMORY_CONFIG
    up = lambda t, dt=ttnn.bfloat16, lay=ttnn.TILE_LAYOUT: ttnn.from_torch(  # noqa: E731
        t, dtype=dt, layout=lay, device=mesh, memory_config=dram)
    a = up(torch.randn(1, G, gs, K))
    ug = up(torch.randn(1, E * S, K, 2 * W) * 0.02, ttnn.bfloat8_b)
    dn = up(torch.randn(1, E * S, W, H) * 0.05, ttnn.bfloat8_b)
    tid = up(ids.reshape(1, G).to(torch.int32), ttnn.uint16, ttnn.ROW_MAJOR_LAYOUT)
    wts = up((torch.rand(1, 1, G * gs, 1) * 0.3).bfloat16().float())
    ttnn.synchronize_device(mesh)
    print("inputs up", flush=True)
    gmm, act = GroupedMM(mesh), MoEAct(mesh)
    alloc = lambda n, lay=ttnn.TILE_LAYOUT: ttnn.allocate_tensor_on_device(  # noqa: E731
        ttnn.Shape([1, G, gs, n]), ttnn.bfloat16, lay, mesh, dram)
    ok = True
    # unfused
    dins, downs = [], []
    for s in range(S):
        f = gmm(a, ug, tid, alloc(2 * W), E, wsub=(S, s))
        d = act(f, wts, alloc(W))
        dins.append(d)
        downs.append(gmm(d, dn, tid, alloc(H, ttnn.ROW_MAJOR_LAYOUT), E, wsub=(S, s)))
    ttnn.synchronize_device(mesh)
    print("unfused done", flush=True)
    ref = torch.cat([ttnn.to_torch(d).float() for d in dins], dim=-1)[0, :real]
    # fused
    fn = lambda: gmm(a, ug, tid, alloc(S * W), E, wsub=(S, 0), act=wts, nsub=S)  # noqa: E731
    gm.ACT_DB = "0"
    din = fn()
    ttnn.synchronize_device(mesh)
    print("fused done", flush=True)
    got = ttnn.to_torch(din).float()[0, :real]
    neq = (got != ref).sum().item()
    print(f"act: {neq} of {ref.numel()} differ, max abs diff {(got - ref).abs().max().item():.3e}, "
          f"ref absmax {ref.abs().max().item():.3e}", flush=True)
    ok &= neq == 0
    for s in range(S):
        d2 = gmm(din, dn, tid, alloc(H, ttnn.ROW_MAJOR_LAYOUT), E, wsub=(S, s), koff=s * W, K=W)
        r = ttnn.to_torch(downs[s]).float()[0, :real]
        g = ttnn.to_torch(d2).float()[0, :real]
        n2 = (g != r).sum().item()
        ttnn.synchronize_device(mesh)
        print(f"down s={s} (koff): {n2} differ", flush=True)
        ok &= n2 == 0
    # the double-buffered weight blocks (I2M_GMM_DB=1 path: h = 2) give the same bits
    gm.ACT_DB = "1"
    got2 = ttnn.to_torch(fn()).float()[0, :real]
    gm.ACT_DB = "auto"
    n3 = (got2 != ref).sum().item()
    print(f"act double-buffered: {n3} differ", flush=True)
    ok &= n3 == 0
    for name, f in (("unfused gate|up + act x3", lambda: [act(gmm(a, ug, tid, alloc(2 * W), E, wsub=(S, s)), wts,
                                                                 alloc(W)) for s in range(S)]), ("fused", fn)):
        f()
        ttnn.synchronize_device(mesh)
        t0 = time.perf_counter()
        for _ in range(5):
            f()
        ttnn.synchronize_device(mesh)
        print(f"{name}: {(time.perf_counter() - t0) / 5 * 1e3:.2f} ms", flush=True)
    ttnn.close_mesh_device(mesh)
    print("PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
