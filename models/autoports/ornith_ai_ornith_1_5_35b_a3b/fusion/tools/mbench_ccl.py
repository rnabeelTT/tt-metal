"""Collective microbenchmark on the (1, 4) mesh (FABRIC_1D): the decode boundary's per-layer collectives.

  rs+ag: today's pattern per boundary: reduce-scatter of the [rows, H] partial (the mixer / MoE exit) and the next
         norm's all-gather of the fractured residual (tt_transformers ccl helpers, as the layer calls them)
  ar:    one all_reduce_async of the partial (persistent scratch buffer + global semaphore, the gpt-oss decode form)

Traced, REPS calls per trace, time per call (includes dispatch).

    python mbench_ccl.py [rows ...]
"""

import sys
import time

import torch

import ttnn
from models.tt_transformers.tt.ccl import TT_CCL, tt_all_gather, tt_all_reduce

H = 2048
REPS, ITERS = 20, 10


def bench(dev, label, fn):
    fn()
    tid = ttnn.begin_trace_capture(dev, cq_id=0)
    for _ in range(REPS):
        fn()
    ttnn.end_trace_capture(dev, tid, cq_id=0)
    ttnn.execute_trace(dev, tid, cq_id=0, blocking=True)
    t0 = time.time()
    for _ in range(ITERS):
        ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(dev)
    ttnn.release_trace(dev, tid)
    print(f"  {label:44s} {(time.time() - t0) / (ITERS * REPS) * 1e6:7.1f} us/call", flush=True)


def main():
    rows_list = [int(v) for v in sys.argv[1:]] or [1, 8, 32]
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), l1_small_size=24576, trace_region_size=64 << 20)
    try:
        ccl = TT_CCL(mesh)
        rep = ttnn.ReplicateTensorToMesh(mesh)
        nd = mesh.get_num_devices()
        grid = mesh.compute_with_storage_grid_size()
        sem_cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
        for rows in rows_list:
            print(f"rows {rows}")
            x = ttnn.from_torch(torch.randn(1, 1, rows, H), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh,
                                memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=rep)

            def rs_ag():
                part = ttnn.clone(x, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # tt_all_reduce deallocates its input
                frac = tt_all_reduce(part, mesh, ccl, cluster_axis=0, dim=3, topology=ttnn.Topology.Ring,
                                     memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16)
                full = tt_all_gather(frac, mesh, ccl, cluster_axis=None, dim=3, topology=ttnn.Topology.Ring,
                                     memory_config=ttnn.L1_MEMORY_CONFIG)
                ttnn.deallocate(frac)
                ttnn.deallocate(full)

            bench(mesh, "clone + reduce-scatter + all-gather", rs_ag)
            bench(mesh, "clone only", lambda: ttnn.deallocate(ttnn.clone(x, memory_config=ttnn.DRAM_MEMORY_CONFIG)))

            for cores in (8, 16, 32):  # 32: the decode norm's act_shard_hidden layout (4 x 8 cores, [32, 64] shards)
                shard = ttnn.create_sharded_memory_config(
                    shape=(32, H // cores), core_grid=ttnn.CoreGrid(x=min(cores, 8), y=max(1, cores // 8)),
                    strategy=ttnn.ShardStrategy.WIDTH, orientation=ttnn.ShardOrientation.ROW_MAJOR,
                    use_height_and_width_as_shard_shape=True,
                )
                scratch_cfg = ttnn.create_sharded_memory_config(
                    shape=(32, H * nd // cores), core_grid=ttnn.CoreGrid(x=min(cores, 8), y=max(1, cores // 8)),
                    strategy=ttnn.ShardStrategy.WIDTH, orientation=ttnn.ShardOrientation.ROW_MAJOR,
                    use_height_and_width_as_shard_shape=True,
                )
                for dt in (ttnn.bfloat8_b,):
                    xs = ttnn.to_memory_config(x, shard, dtype=dt)
                    scratch = ttnn.empty([1, 1, 32, H * nd], dt, ttnn.TILE_LAYOUT, mesh, scratch_cfg)
                    sem = ttnn.create_global_semaphore(mesh, sem_cores, 0)
                    try:
                        def ar():
                            out = ttnn.experimental.all_reduce_async(
                                xs, scratch, cluster_axis=1, mesh_device=mesh, multi_device_global_semaphore=sem,
                                memory_config=shard, dtype=ttnn.bfloat16, topology=ttnn.Topology.Ring, num_links=1,
                            )
                            ttnn.deallocate(out)

                        bench(mesh, f"all_reduce_async {cores} cores {'bf16' if dt == ttnn.bfloat16 else 'bfp8'} in", ar)
                    except Exception as e:  # noqa: BLE001 - probe
                        print(f"  all_reduce_async {cores} cores {dt}: {str(e).splitlines()[0][:150]}")
                    ttnn.deallocate(xs)
                    ttnn.deallocate(scratch)
            ttnn.deallocate(x)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
