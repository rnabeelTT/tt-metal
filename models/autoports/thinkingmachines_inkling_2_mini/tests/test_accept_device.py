# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""tt/sampling.DeviceAccept (tt/kernels/taccept.cpp) on random verify logits / drafter candidates (no model):
  1. bitwise: every trial's (emitted tokens, kept), the next drafter's anchor / commit count and the accept row == the
     Python twin tt/sampling.accept_ref on the host's merged candidates, greedy and T 0.7 / 1, J 0..3, every die;
     greedy also == HostSampler.spec;
  2. exactness (CPU, accept_ref): the first emitted token over DRAWS draws vs the target p_0 (the speculative rule's
     output is the target's distribution whatever q), and the second given the first kept, vs p_1; plus
     HostSampler.spec's draws as a yardstick.
MESH > 1: the vocab-sharded head (each die its slice's top 32, all-gathered; merged by the kernel on every die).
env: DRAWS (20000), TRIALS (300), MESH (1)"""

import math
import os
import types

import torch

import ttnn
from models.autoports.thinkingmachines_inkling_2_mini.tt.sampling import (
    K, DeviceAccept, DeviceTopK, HostSampler, accept_params, accept_ref)

N = int(os.environ.get("MESH", "1"))
VP = (201024 + 32 * N - 1) // (32 * N) * (32 * N)
V = VP - 64
R, D = 4, 3


def sp(t):
    return types.SimpleNamespace(temperature=[t], top_k=[-1], top_p=[1.0], seed=[-1])


def q_for(gen, row_ids, n=K):
    """drafter candidates: ids sharing some of the verify row's, fp32 softcapped-like logits, descending"""
    ids = torch.cat([row_ids[: n // 2], torch.randint(0, V, (n,), generator=gen)])
    ids = torch.unique_consecutive(ids)
    ids = list(dict.fromkeys(int(x) for x in ids))[:n]
    while len(ids) < n:
        x = int(torch.randint(0, V, (1,), generator=gen))
        if x not in ids:
            ids.append(x)
    lg, _ = torch.sort(torch.randn(n, generator=gen, dtype=torch.float64) * 3 + 5, descending=True)
    return torch.tensor(ids, dtype=torch.int64), lg.float()


def exactness(draws):
    gen = torch.Generator()
    gen.manual_seed(1)
    hs = HostSampler(V)
    for T in (1.0, 0.7):
        lg = torch.full((2, K), -30.0)
        idx = torch.stack([torch.randperm(V, generator=gen)[:K] for _ in range(2)])
        vals = torch.stack([torch.sort(torch.randn(K, generator=gen) * 1.5, descending=True)[0] for _ in range(2)])
        vals = vals.to(torch.bfloat16).float()
        qid0, qlg0 = q_for(gen, idx[0])
        qid1, qlg1 = q_for(gen, idx[1])
        p0 = hs.dist(idx[0], vals[0], 0, sp(T))
        p1 = hs.dist(idx[1], vals[1], 0, sp(T))
        qp0 = torch.softmax(qlg0.double() / T, -1)
        pos0 = {int(i): j for j, i in enumerate(p0[0])}
        pos1 = {int(i): j for j, i in enumerate(p1[0])}
        c0 = torch.zeros(len(pos0), dtype=torch.float64)
        c1 = torch.zeros(len(pos1), dtype=torch.float64)
        ch = torch.zeros(len(pos0), dtype=torch.float64)
        n1 = 0
        for _ in range(draws):
            d0 = int(qid0[int(torch.multinomial(qp0, 1, generator=gen))])
            u = torch.randint(0, 1 << 31, (16,), generator=gen).tolist()
            prm = accept_params(T, 1.0, 1, u[:8], u[8:])
            em, k = accept_ref(idx, vals, [d0], prm, torch.stack([qid0, qid1]), torch.stack([qlg0, qlg1]), V=V)
            c0[pos0[em[0]]] += 1
            if k == 1:
                c1[pos1[em[1]]] += 1
                n1 += 1
            hem, _ = hs.spec(idx, vals, 0, [d0], sp(T), q=[(qid0, qlg0)])
            ch[pos0[hem[0]]] += 1
        for name, c, p, n in (("first", c0, p0[1], draws), ("second | kept", c1, p1[1], n1), ("host first", ch, p0[1], draws)):
            e = p * n
            m = e >= 5
            chi = float((((c - e) ** 2) / e)[m].sum())
            dof = int(m.sum()) - 1
            z = (chi - dof) / math.sqrt(2 * dof)
            tv = 0.5 * float((c / n - p).abs().sum())
            print(f"T={T} {name}: {n} draws, TV vs target {tv:.4f}, chi2 {chi:.1f} on {dof} dof (z {z:+.2f})", flush=True)
            assert abs(z) < 4, z


def bitwise(trials):
    if N > 1:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, N))
    rep = ttnn.ReplicateTensorToMesh(mesh) if N > 1 else None
    shard = ttnn.ShardTensorToMesh(mesh, dim=3) if N > 1 else None
    u32 = lambda t: ttnn.from_torch(t.to(torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT,  # noqa
                                    device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=rep)
    dr = types.SimpleNamespace(cand_id=u32(torch.zeros(D, K)), cand_lg=ttnn.from_torch(
        torch.zeros(D, K), dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=rep), in_anchor=u32(torch.zeros(1, 1)),
        in_n=u32(torch.zeros(1, 1)))
    ver = types.SimpleNamespace(tokens=u32(torch.zeros(1, R)), acc=u32(torch.zeros(1, 32)))
    tk = DeviceTopK(mesh, VP // N, V, sharded=N > 1)
    da = DeviceAccept(tk, dr, ver, R)
    hs = HostSampler(V)
    gen = torch.Generator()
    gen.manual_seed(0)
    dies = lambda t: ttnn.get_device_tensors(t) if N > 1 else [t]  # noqa
    fails, kept = 0, [0] * (R)
    for trial in range(trials):
        lg = torch.randn(1, 1, R, VP, generator=gen) * 3
        hot = torch.randint(0, V, (R, 6), generator=gen)
        for r in range(R):  # a few peaked candidates per row (drafts that can be kept)
            lg[0, 0, r, hot[r]] = 12 + torch.randn(6, generator=gen) * (0.3 if trial % 2 else 2.0)
        if trial % 7 == 0:
            lg[0, 0, 0, V + 1] = 40.0  # padded column on top
        x = ttnn.from_torch(lg, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=shard)
        parts = tk.device(x, R)
        idx, vals = tk.host(parts)
        T = (0.0, 0.7, 1.0)[trial % 3]
        J = (3, 3, 2, 1, 0)[trial % 5]
        qid, qlg = zip(*[q_for(gen, idx[j]) for j in range(D)])
        qid, qlg = torch.stack(qid), torch.stack(qlg)
        drafts = []
        for j in range(D):  # the verify row's top candidate (often), else a draw from q
            if T == 0.0 or float(torch.rand(1, generator=gen)) < 0.5:
                drafts.append(int(idx[j][0]) if int(idx[j][0]) < V else int(idx[j][1]))
            else:
                drafts.append(int(qid[j][int(torch.multinomial(torch.softmax(qlg[j].double() / T, -1), 1, generator=gen))]))
        ttnn.copy_host_to_device_tensor(ttnn.from_torch(qid.to(torch.int32), dtype=ttnn.uint32,
                                                        layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=rep), dr.cand_id)
        ttnn.copy_host_to_device_tensor(ttnn.from_torch(qlg, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT,
                                                        mesh_mapper=rep), dr.cand_lg)
        ttnn.copy_host_to_device_tensor(ttnn.from_torch(torch.tensor([[777] + drafts], dtype=torch.int32),
                                                        dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT,
                                                        mesh_mapper=rep), ver.tokens)
        u = torch.randint(0, 1 << 31, (16,), generator=gen).tolist()
        prm = accept_params(T, 1.0, J, u[:8], u[8:])
        da.set(prm)
        da.device(parts)
        em, k, toks = da.result()
        ref = accept_ref(idx, vals, drafts[:J], prm, qid, qlg, V=V)
        if T == 0.0:
            assert hs.spec(idx, vals, 0, drafts[:J], sp(0.0)) == ref, (hs.spec(idx, vals, 0, drafts[:J], sp(0.0)), ref)
        ok = (em, k) == ref and toks == [777] + drafts
        for die in range(N):
            em2, k2, _ = da.result(None, die)
            an = int(ttnn.to_torch(dies(dr.in_anchor)[die]).reshape(-1)[0])
            nn = int(ttnn.to_torch(dies(dr.in_n)[die]).reshape(-1)[0])
            ac = ttnn.to_torch(dies(ver.acc)[die]).reshape(-1).tolist()
            ok = ok and (em2, k2) == (em, k) and an == em[-1] and nn == k + 1 and ac == [k] + [0] * 31
        kept[k] += 1
        if not ok:
            fails += 1
            if fails < 10:
                print(f"trial {trial} T={T} J={J}: device {em} {k} ref {ref} drafts {drafts}", flush=True)
    print(f"bitwise: {trials - fails} / {trials} trials identical (kept histogram {kept})", flush=True)
    # timing: the kernel alone, sampled, all drafts kept vs early stop
    import time

    for T in (0.0, 1.0):
        da.set(accept_params(T, 1.0, 3, [0] * 8, [0] * 8))  # (u = 0: every draft with p > 0 kept)
        ttnn.synchronize_device(mesh)
        t0 = time.perf_counter()
        for _ in range(50):
            da.device(parts)
        ttnn.synchronize_device(mesh)
        print(f"T={T}: {(time.perf_counter() - t0) / 50 * 1e6:.0f} us per call (eager, incl. dispatch)", flush=True)
    ttnn.close_mesh_device(mesh)
    assert fails == 0, fails


if __name__ == "__main__":
    if os.environ.get("CPU", "0") == "1":
        exactness(int(os.environ.get("DRAWS", "20000")))
    else:
        bitwise(int(os.environ.get("TRIALS", "300")))
        exactness(int(os.environ.get("DRAWS", "4000")))
    print("PASS", flush=True)
