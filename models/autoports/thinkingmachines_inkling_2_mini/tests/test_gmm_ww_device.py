# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""tt/gmm.py's weight loads on the writer (I2M_GMM_WW) and the sub-shards' downs in one program (I2M_GMM_D3) vs the
reader loading the weights and one down program per sub-shard: bitwise on gate|up + act (single- and double-buffered
weights) and the three downs, with the slices' expert ranges balanced on device (I2M_GMM_BAL) or not, at a real-size
group count (E 128, T rows top-8, env T 2048; SKEW: uneven expert loads, default on)."""

import os
import time

import torch

import ttnn
import models.autoports.thinkingmachines_inkling_2_mini.tt.gmm as gm
from models.autoports.thinkingmachines_inkling_2_mini.tt.gmm import GroupedMM


def main():
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    torch.manual_seed(0)
    T, E, S, W, K, H, gs = int(os.environ.get("T", "2048")), 128, 3, 384, 2304, 2304, 32
    p = torch.exp(torch.randn(E)) if os.environ.get("SKEW", "1") == "1" else torch.ones(E)  # (uneven expert loads)
    cnt = torch.bincount(torch.multinomial(p, T * 8, replacement=True), minlength=E)
    cnt[5] = 0  # (an expert without rows)
    ids = torch.cat([torch.full((int(-(-c // gs)),), e) for e, c in enumerate(cnt.tolist())] + [torch.full((3,), E)])
    G = len(ids)
    dram = ttnn.DRAM_MEMORY_CONFIG
    up = lambda t, dt=ttnn.bfloat16, lay=ttnn.TILE_LAYOUT: ttnn.from_torch(  # noqa: E731
        t, dtype=dt, layout=lay, device=mesh, memory_config=dram)
    a = up(torch.randn(1, G, gs, K))
    ug = up(torch.randn(1, E * S, K, 2 * W) * 0.02, ttnn.bfloat8_b)
    dn = up(torch.randn(1, E * S, W, H) * 0.05, ttnn.bfloat8_b)
    tid = up(ids.reshape(1, G).to(torch.int32), ttnn.uint16, ttnn.ROW_MAJOR_LAYOUT)
    wts = up((torch.rand(1, 1, G * gs, 1) * 0.3).bfloat16().float())
    gmm = GroupedMM(mesh)
    alloc = lambda n, lay=ttnn.TILE_LAYOUT: ttnn.allocate_tensor_on_device(  # noqa: E731
        ttnn.Shape([1, G, gs, n]), ttnn.bfloat16, lay, mesh, dram)
    real = G - 3

    def run():
        din = gmm(a, ug, tid, alloc(S * W), E, wsub=(S, 0), act=wts, nsub=S)
        if gm.D3:
            ds = [alloc(H, ttnn.ROW_MAJOR_LAYOUT) for _ in range(S)]
            gmm(din, dn, tid, ds, E, wsub=(S, 0), K=W)
        else:
            ds = [gmm(din, dn, tid, alloc(H, ttnn.ROW_MAJOR_LAYOUT), E, wsub=(S, s), koff=s * W, K=W) for s in range(S)]
        # (non-act tile output and k0: the first sub-shard's down, K tiles [4, 12) of each expert)
        k = gmm(din, dn, tid, alloc(H), E, wsub=(S, 1), k0=4, koff=W + 128, K=W - 128)
        return [ttnn.to_torch(t).float()[0, :real] for t in [din] + ds + [k]]

    ok = True
    print(f"T={T} G={G}", flush=True)
    for db in ("0", "1"):
        gm.ACT_DB = db
        outs = {}
        for ww, d3, bal in (("0", False, 0), ("auto", True, 5), ("auto", True, 0), ("1", True, 2), ("1", False, 5)):
            gm.WW, gm.D3, gm.BAL = ww, d3, bal
            outs[(ww, d3, bal)] = run()
            ttnn.synchronize_device(mesh)
            t0 = time.perf_counter()
            for _ in range(3):
                run()
            ttnn.synchronize_device(mesh)
            print(f"DB={db} WW={ww} D3={int(d3)} BAL={bal}: {(time.perf_counter() - t0) / 3 * 1e3:.2f} ms per run (+ reads)",
                  flush=True)
            for i, (r, g) in enumerate(zip(outs[("0", False, 0)], outs[(ww, d3, bal)])):
                n = (r != g).sum().item()
                print(f"  out {i}: {n} of {r.numel()} differ (absmax {r.abs().max().item():.3e})", flush=True)
                ok &= n == 0
    ttnn.close_mesh_device(mesh)
    print("PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
