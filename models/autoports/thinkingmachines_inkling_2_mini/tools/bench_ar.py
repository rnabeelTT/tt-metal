# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Decode all-reduce of a [1, 1, 32, 2304] bf16 partial over the (1, N) mesh, traced: ttnn.all_reduce (the current
path) vs ttnn.experimental.all_reduce_async on persistent width-sharded scratch (the qwen36 DecodeBoundary op), with
its reshards. Checks the sums too. usage: MESH=4 tools/run_dev.sh python .../tools/bench_ar.py"""

import os
import time

import torch

import ttnn

H = 2304


def main():
    N = int(os.environ.get("MESH", "4"))
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, N), trace_region_size=64 << 20)
    parts = torch.randn(N, 1, 32, H)
    x = ttnn.from_torch(parts.reshape(N, 32, H).unsqueeze(1).reshape(1, 1, 32 * N, H).reshape(1, 1, 32, H * N)
                        if False else parts.permute(1, 2, 0, 3).reshape(1, 32, N * H).unsqueeze(0),
                        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=ttnn.L1_MEMORY_CONFIG,
                        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=3))
    want = parts.to(torch.bfloat16).float().sum(0)[0]
    grid = mesh.compute_with_storage_grid_size()
    sem_cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})

    def check(name, out):
        got = ttnn.to_torch(ttnn.get_device_tensors(out)[0]).float().reshape(32, -1)[:, :H]
        print(f"   {name}: max|d| {(got - want).abs().max():.4f}", flush=True)

    def bench(name, fn):
        out = fn()
        check(name, out)
        tid = ttnn.begin_trace_capture(mesh, cq_id=0)
        for _ in range(20):
            out = fn()
        ttnn.end_trace_capture(mesh, tid, cq_id=0)
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
        t0 = time.perf_counter()
        for _ in range(10):
            ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh)
        us = (time.perf_counter() - t0) / 200 * 1e6
        print(f"{name:48s} {us:7.1f} us", flush=True)
        ttnn.release_trace(mesh, tid)

    bench("ttnn.all_reduce (DRAM out)", lambda: ttnn.all_reduce(x, cluster_axis=1, topology=ttnn.Topology.Linear,
                                                                 memory_config=ttnn.DRAM_MEMORY_CONFIG))
    for cx, cy in ((6, 2), (9, 2), (8, 3), (4, 3)):
        n = cx * cy
        if (H // 32) % n:
            continue
        mc = ttnn.create_sharded_memory_config(shape=(32, H // n), core_grid=ttnn.CoreGrid(x=cx, y=cy),
                                               strategy=ttnn.ShardStrategy.WIDTH,
                                               orientation=ttnn.ShardOrientation.ROW_MAJOR,
                                               use_height_and_width_as_shard_shape=True)
        smc = ttnn.create_sharded_memory_config(shape=(32, H * N // n), core_grid=ttnn.CoreGrid(x=cx, y=cy),
                                                strategy=ttnn.ShardStrategy.WIDTH,
                                                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                                                use_height_and_width_as_shard_shape=True)
        for dt in (ttnn.bfloat16, ttnn.bfloat8_b):
            scratch = ttnn.empty([1, 1, 32, H * N], dt, ttnn.TILE_LAYOUT, mesh, smc)
            sem = ttnn.create_global_semaphore(mesh, sem_cores, 0)

            def fn(mc=mc, scratch=scratch, sem=sem, dt=dt, reshard=True):
                p = ttnn.to_memory_config(x, mc, dtype=dt)
                r = ttnn.experimental.all_reduce_async(p, scratch, cluster_axis=1, mesh_device=mesh,
                                                       multi_device_global_semaphore=sem, memory_config=mc,
                                                       dtype=ttnn.bfloat16, topology=ttnn.Topology.Linear, num_links=1)
                ttnn.deallocate(p)
                if reshard:
                    o = ttnn.to_memory_config(r, ttnn.L1_MEMORY_CONFIG)
                    ttnn.deallocate(r)
                    return o
                return r

            try:
                bench(f"all_reduce_async {n} cores {dt} + reshards", fn)
            except Exception as e:  # noqa: BLE001
                print(f"all_reduce_async {n} cores {dt}: {type(e).__name__}: {str(e)[:200]}", flush=True)
    print("== bench_ar done", flush=True)
    from models.autoports.thinkingmachines_inkling_2_mini.tt.clean_exit import close

    close(mesh)


if __name__ == "__main__":
    main()
