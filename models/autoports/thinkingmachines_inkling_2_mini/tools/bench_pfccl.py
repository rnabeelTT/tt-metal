# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Prefill-chunk CCL costs on the (1, N) mesh, traced: ttnn.all_reduce of [1, 1, T, 2304] bf16 (the attention / MoE
output today) vs ttnn.reduce_scatter by rows (dim 2) and the all_gather of a T / N row slice; checks the sums.
usage: MESH=4 tools/run_dev.sh python .../tools/bench_pfccl.py   env: TS (row counts, default 512,1024,2048)"""

import os
import time

import torch

import ttnn

H = 2304


def main():
    N = int(os.environ.get("MESH", "4"))
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, N), trace_region_size=256 << 20)
    DRAM = ttnn.DRAM_MEMORY_CONFIG

    def bench(name, fn, n=10):
        out = fn()
        tid = ttnn.begin_trace_capture(mesh, cq_id=0)
        for _ in range(n):
            out = fn()
        ttnn.end_trace_capture(mesh, tid, cq_id=0)
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
        t0 = time.perf_counter()
        for _ in range(5):
            ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh)
        us = (time.perf_counter() - t0) / (5 * n) * 1e6
        print(f"{name:56s} {us:8.1f} us", flush=True)
        ttnn.release_trace(mesh, tid)
        return out

    for T in [int(t) for t in os.environ.get("TS", "512,1024,2048").split(",")]:
        parts = torch.randn(N, 1, T, H)
        x = ttnn.from_torch(parts.reshape(1, N, T, H).permute(0, 2, 1, 3).reshape(1, 1, T, N * H), dtype=ttnn.bfloat16,
                            layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=DRAM,
                            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=3))
        want = parts.to(torch.bfloat16).float().sum(0)[0]
        ar = bench(f"T={T} all_reduce", lambda: ttnn.all_reduce(x, cluster_axis=1, topology=ttnn.Topology.Linear,
                                                               memory_config=DRAM))
        got = ttnn.to_torch(ttnn.get_device_tensors(ar)[0]).float()[0, 0]
        print(f"   all_reduce max|d| {(got - want).abs().max():.4f}", flush=True)
        for lk in (2,):
            try:
                a2 = bench(f"T={T} all_reduce links={lk}",
                           lambda lk=lk: ttnn.all_reduce(x, cluster_axis=1, topology=ttnn.Topology.Linear,
                                                         memory_config=DRAM, num_links=lk))
                g2 = ttnn.to_torch(ttnn.get_device_tensors(a2)[0]).float()[0, 0]
                print(f"   links={lk} == links=1: {torch.equal(g2, got)}", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"T={T} all_reduce links={lk}: {type(e).__name__}: {str(e)[:300]}", flush=True)
        for lk in (1, 2):
            try:
                rs = bench(f"T={T} reduce_scatter dim2 links={lk}",
                           lambda lk=lk: ttnn.reduce_scatter(x, 2, cluster_axis=1, topology=ttnn.Topology.Linear,
                                                             memory_config=DRAM, num_links=lk))
                ds = ttnn.get_device_tensors(rs)
                got = torch.cat([ttnn.to_torch(d).float()[0, 0] for d in ds], 0)
                print(f"   rs shape {list(rs.shape)} max|d| {(got - want).abs().max():.4f}; vs AR equal "
                      f"{torch.equal(got, ttnn.to_torch(ttnn.get_device_tensors(ar)[0]).float()[0, 0])}", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"T={T} reduce_scatter links={lk}: {type(e).__name__}: {str(e)[:300]}", flush=True)
        xl = ttnn.from_torch(torch.randn(1, 1, T, H), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh,
                             memory_config=DRAM, mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=2))
        bench(f"T={T} all_gather of T/N rows", lambda: ttnn.all_gather(xl, 2, cluster_axis=1, memory_config=DRAM,
                                                                      topology=ttnn.Topology.Linear))
        try:
            bench(f"T={T} all_gather of T/N rows links=2", lambda: ttnn.all_gather(
                xl, 2, cluster_axis=1, memory_config=DRAM, topology=ttnn.Topology.Linear, num_links=2))
        except Exception as e:  # noqa: BLE001
            print(f"all_gather links=2: {type(e).__name__}: {str(e)[:300]}", flush=True)
        sm = ttnn.from_torch(torch.randn(1, 1, 64 * N, H), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh,
                             memory_config=DRAM, mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=2))
        bench("all_gather of 64 rows per die", lambda: ttnn.all_gather(sm, 2, cluster_axis=1, memory_config=DRAM,
                                                                       topology=ttnn.Topology.Linear))
    print("== bench_pfccl done", flush=True)
    from models.autoports.thinkingmachines_inkling_2_mini.tt.clean_exit import close

    close(mesh)


if __name__ == "__main__":
    main()
