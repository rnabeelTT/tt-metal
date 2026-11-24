# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""tt/sampling.DeviceSampler (tt/kernels/tsample.cpp) vs the host sampler, on random logits (no model):
  1. every draw: the device token == argmax_j (logit_j + noise_j) over the host's candidates (ids < V, first max), for
     greedy (zero noise; rows with exact ties and padded vocab columns on top) and T * Gumbel noise, widths 1 / 4 / 8;
  2. distribution at T = 1 (and 0.7): DRAWS device draws vs the exact softmax over the candidates (HostSampler.dist) and
     vs DRAWS host-sampler draws: total variation and a chi-square test.
MESH > 1: the vocab-sharded head (each die its slice's top 32, all-gathered; the kernel merges them on every die, and
every die's token must agree).
env: DRAWS (20000), MESH (1)"""

import math
import os

import torch

import ttnn
from models.autoports.thinkingmachines_inkling_2_mini.tt.sampling import K, DeviceSampler, DeviceTopK, HostSampler

N = int(os.environ.get("MESH", "1"))
VP = (201024 + 32 * N - 1) // (32 * N) * (32 * N)
V = VP - 64  # (the last 64 columns: padded vocab, never sampled)


def gumbel(n, gen):
    u = torch.rand(n, K, generator=gen, dtype=torch.float64)
    return -torch.log(-torch.log(u.clamp(1e-300, 1.0 - 1e-16)))


def read(toks):
    """-> every die's tokens (they must agree)."""
    ts = [ttnn.to_torch(t).reshape(-1).long() for t in (ttnn.get_device_tensors(toks) if N > 1 else [toks])]
    assert all(torch.equal(t, ts[0]) for t in ts), ts
    return ts[0]


def main():
    draws = int(os.environ.get("DRAWS", "20000"))
    if N > 1:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, N))
    rep = ttnn.ReplicateTensorToMesh(mesh) if N > 1 else None
    shard = ttnn.ShardTensorToMesh(mesh, dim=3) if N > 1 else None
    tk = DeviceTopK(mesh, VP // N, V, sharded=N > 1)
    ds = DeviceSampler(tk)
    hs = HostSampler(V)
    gen = torch.Generator()
    gen.manual_seed(0)
    sp = lambda t: type("SP", (), {"temperature": [t] * 8, "top_k": [-1] * 8, "top_p": [1.0] * 8})()  # noqa: E731
    fails = 0
    for W in (1, 4, 8):
        lg = torch.randn(1, 1, W, VP, generator=gen) * 4
        lg[0, 0, 0, V + 3] = 100.0  # padded column on top of row 0
        if W > 1:  # an exact tie for the max in row 1
            lg[0, 0, 1, 77] = lg[0, 0, 1, 99] = 60.0
            if N > 1:  # (and across dies)
                lg[0, 0, 1, VP // N + 5] = 60.0
        if W > 2:  # a near-flat row
            lg[0, 0, 2] = torch.randn(VP, generator=gen) * 0.01
        x = ttnn.from_torch(lg, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=shard)
        toks = ttnn.from_torch(torch.zeros(1, W, dtype=torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT,
                               device=mesh, mesh_mapper=rep)
        parts = tk.device(x, W)
        idx, vals = tk.host(parts)
        n_ok = 0
        for trial in range(200):
            T = (0.0, 0.7, 1.0)[trial % 3]
            tg = (T * gumbel(W, gen)).float()
            ds.set_noise(tg)
            ds.device(parts, toks, W)
            got = read(toks)[:W]
            for r in range(W):
                ok = idx[r] < V
                s = vals[r].float() + tg[r]
                s[~ok] = -math.inf
                ref = int(idx[r][int(torch.argmax(s))])
                if T == 0.0:
                    ids, p = hs.dist(idx[r], vals[r], 0, sp(0.0))
                    assert p is None and int(ids[0]) == ref
                if int(got[r]) != ref:
                    fails += 1
                    if fails < 10:
                        print(f"W={W} trial {trial} row {r}: device {int(got[r])} host {ref}")
                else:
                    n_ok += 1
        print(f"W={W}: {n_ok} / {200 * W} draws identical to the host's Gumbel-max", flush=True)
    assert fails == 0, fails

    # distribution: one row with a spread of candidate probabilities
    lg = torch.full((1, 1, 1, VP), -20.0)
    lg[0, 0, 0, torch.randperm(V, generator=gen)[:40]] = torch.linspace(3.0, -1.0, 40)
    x = ttnn.from_torch(lg, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=shard)
    toks = ttnn.from_torch(torch.zeros(1, 1, dtype=torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT,
                           device=mesh, mesh_mapper=rep)
    parts = tk.device(x, 1)
    idx, vals = tk.host(parts)
    for T in (1.0, 0.7):
        ids, p = hs.dist(idx[0], vals[0], 0, sp(T))
        pos = {int(i): j for j, i in enumerate(ids)}
        dev = torch.zeros(len(ids), dtype=torch.float64)
        host = torch.zeros(len(ids), dtype=torch.float64)
        for _ in range(draws):
            ds.set_noise((T * gumbel(1, gen)).float())
            ds.device(parts, toks, 1)
            dev[pos[int(read(toks)[0])]] += 1
            host[pos[hs.sample(idx, vals, [(0, 0)], sp(T))[0]]] += 1
        e = p * draws
        m = e >= 5
        chi = float((((dev - e) ** 2) / e)[m].sum())
        dof = int(m.sum()) - 1
        z = (chi - dof) / math.sqrt(2 * dof)
        tv_exact = 0.5 * float((dev / draws - p).abs().sum())
        tv_host = 0.5 * float(((dev - host) / draws).abs().sum())
        tv_host_exact = 0.5 * float((host / draws - p).abs().sum())
        print(f"T={T}: {draws} draws over {len(ids)} candidates: TV(device, exact) {tv_exact:.4f}, TV(host, exact) "
              f"{tv_host_exact:.4f}, TV(device, host) {tv_host:.4f}; chi2 {chi:.1f} on {dof} dof (z {z:+.2f})",
              flush=True)
        assert abs(z) < 4, z
    ttnn.close_mesh_device(mesh)
    print("PASS", flush=True)


if __name__ == "__main__":
    main()
