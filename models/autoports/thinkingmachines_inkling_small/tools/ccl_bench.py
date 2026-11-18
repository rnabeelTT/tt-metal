# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""A decode layer boundary at Inkling's shapes (3 verify rows, H 4096, (1, 4) mesh), trace-replay time per call:
  rs+ag      the current contract: reduce_scatter_minimal_async of the partial, all_gather_async of the residual
  ar_bfp8    qwen36 DecodeBoundary.all_reduce (BFP8 partial in, bf16 out, 16-core residual layout)
  ar_bf16    the same all_reduce_async with a BF16 partial in
and each all-reduce's sum vs the exact (fp64) sum of the partials.

usage (in the image): tools/run_dev.sh python models/autoports/thinkingmachines_inkling_small/tools/ccl_bench.py
"""

import os
import sys
import time

import torch
import ttnn

from models.demos.blackhole.qwen36.tt.decode_boundary import DecodeBoundary, residual_memory_config

H, R = 4096, 3


def main():
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), l1_small_size=24576, trace_region_size=64 << 20)
    try:
        from models.tt_transformers.tt.ccl import TT_CCL

        ccl = TT_CCL(mesh)
        nd = mesh.get_num_devices()
        g = torch.Generator().manual_seed(0)
        parts = torch.randn(nd, 1, R, H, generator=g)
        exact = parts.double().sum(0)
        part = ttnn.from_torch(parts.reshape(nd, 1, R, H), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh,
                               memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0))
        topo = ttnn.Topology.Linear
        db = DecodeBoundary(mesh, H, topo)

        def rs_ag():
            red = ttnn.experimental.reduce_scatter_minimal_async(
                part, persistent_output_buffers=None, dim=3,
                multi_device_global_semaphore=ccl.get_and_cycle_rs_semaphore_handles(),
                barrier_semaphore=ccl.get_and_cycle_barrier_semaphore_handle(), num_links=1,
                memory_config=ttnn.DRAM_MEMORY_CONFIG, intermediate_memory_config=ttnn.DRAM_MEMORY_CONFIG,
                topology=topo, chunks_per_sync=10, num_workers_per_link=2, num_buffers_per_channel=2)
            out = ttnn.experimental.all_gather_async(
                red, persistent_output_buffer=None, dim=3,
                multi_device_global_semaphore=ccl.get_and_cycle_ag_semaphore_handles(), num_links=1, topology=topo,
                memory_config=ttnn.DRAM_MEMORY_CONFIG, barrier_semaphore=ccl.get_and_cycle_barrier_semaphore_handle(),
                chunks_per_sync=10, num_workers_per_link=2, num_buffers_per_channel=2)
            ttnn.deallocate(red)
            return out

        mc = residual_memory_config(H)
        scratch16 = ttnn.empty([1, 1, 32, H * nd], ttnn.bfloat16, ttnn.TILE_LAYOUT, mesh,
                               ttnn.create_sharded_memory_config(shape=(32, H * nd // 16), core_grid=ttnn.CoreGrid(x=8, y=2),
                                                                 strategy=ttnn.ShardStrategy.WIDTH,
                                                                 orientation=ttnn.ShardOrientation.ROW_MAJOR,
                                                                 use_height_and_width_as_shard_shape=True))
        grid = mesh.compute_with_storage_grid_size()
        sem16 = ttnn.create_global_semaphore(
            mesh, ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))}), 0)

        def ar(dtype):
            def f():
                p = ttnn.to_memory_config(part, mc, dtype=dtype)
                if dtype == ttnn.bfloat8_b:
                    scratch, sem = db.sites["mixer"]
                else:
                    scratch, sem = scratch16, sem16
                out = ttnn.experimental.all_reduce_async(p, scratch, cluster_axis=1, mesh_device=mesh,
                                                         multi_device_global_semaphore=sem, memory_config=mc,
                                                         dtype=ttnn.bfloat16, topology=topo, num_links=1)
                ttnn.deallocate(p)
                return out
            return f

        def bench(fn, reps=20, iters=20):
            ttnn.deallocate(fn())
            tid = ttnn.begin_trace_capture(mesh, cq_id=0)
            for _ in range(reps):
                ttnn.deallocate(fn())
            ttnn.end_trace_capture(mesh, tid, cq_id=0)
            ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
            t0 = time.time()
            for _ in range(iters):
                ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh)
            ttnn.release_trace(mesh, tid)
            return (time.time() - t0) / (reps * iters) * 1e6

        for name, fn in (("rs+ag", rs_ag), ("ar_bfp8", ar(ttnn.bfloat8_b)), ("ar_bf16", ar(ttnn.bfloat16))):
            try:
                out = fn()
                got = ttnn.to_torch(ttnn.get_device_tensors(out)[0]).double()[..., :R, :].reshape(R, H)
                ttnn.deallocate(out)
                err = (got - exact.reshape(R, H)).abs()
                t = bench(fn)
                print(f"{name:8s} {t:7.1f} us per boundary | vs exact sum: max {err.max().item():.3g} mean {err.mean().item():.3g}",
                      flush=True)
            except Exception as e:  # noqa: BLE001
                import traceback

                traceback.print_exc()
                print(f"{name}: failed {str(e).splitlines()[0][:150]}", flush=True)
    finally:
        sys.stdout.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
