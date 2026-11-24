# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Decode rms_norm on one fp32 row [1, 1, 32, 2304], traced: interleaved (current) vs width-sharded multi-core
(LayerNormShardedMultiCoreProgramConfig) on 8 / 24 / 36 / 72 cores, alone and with the reshard in + out.
Prints us per norm and the max diff vs the interleaved result."""

import time

import torch

import ttnn

REPS, ITERS, H = 20, 10, 2304


def timed(mesh, fn):
    out = fn()
    ttnn.synchronize_device(mesh)
    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
    for _ in range(REPS):
        fn()
    ttnn.end_trace_capture(mesh, tid, cq_id=0)
    ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
    t0 = time.perf_counter()
    for _ in range(ITERS):
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh)
    dt = (time.perf_counter() - t0) / (REPS * ITERS) * 1e6
    ttnn.release_trace(mesh, tid)
    return dt, out


def main():
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=64 << 20)
    ckc = ttnn.init_device_compute_kernel_config(mesh.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
                                                 math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False)
    x = ttnn.from_torch(torch.randn(1, 1, 32, H), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh,
                        memory_config=ttnn.L1_MEMORY_CONFIG)
    wt = torch.rand(1, 1, 1, H) + 0.5
    w = ttnn.from_torch(wt, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh)
    t, ref = timed(mesh, lambda: ttnn.rms_norm(x, epsilon=1e-6, weight=w, compute_kernel_config=ckc))
    ref_t = ttnn.to_torch(ref).float()
    print(f"interleaved: {t:.1f} us", flush=True)
    # sharded variants need the weight as [1, 1, H/32, 32] row-major (the sharded norm's gamma layout)
    wrm = ttnn.from_torch(wt.reshape(1, 1, H // 32, 32), dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh)
    for cx, cy in ((8, 1), (8, 3), (9, 4), (9, 8)):
        n = cx * cy
        bw = H // n // 32
        mc = ttnn.create_sharded_memory_config(shape=(32, H // n), core_grid=ttnn.CoreGrid(x=cx, y=cy),
                                               strategy=ttnn.ShardStrategy.WIDTH,
                                               orientation=ttnn.ShardOrientation.ROW_MAJOR,
                                               use_height_and_width_as_shard_shape=True)
        pc = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(cx, cy), subblock_w=max(d for d in range(1, 5) if bw % d == 0),
            block_h=1, block_w=bw, inplace=False)
        try:
            xs = ttnn.to_memory_config(x, mc)
            for wname, ww in (("tile w", w), ("rm w", wrm)):
                try:
                    ta, o = timed(mesh, lambda: ttnn.rms_norm(xs, epsilon=1e-6, weight=ww, program_config=pc,
                                                              memory_config=mc, compute_kernel_config=ckc))
                    d = (ttnn.to_torch(o).float() - ref_t).abs().max().item()
                    tb, _ = timed(mesh, lambda: ttnn.to_memory_config(ttnn.rms_norm(
                        ttnn.to_memory_config(x, mc), epsilon=1e-6, weight=ww, program_config=pc, memory_config=mc,
                        compute_kernel_config=ckc), ttnn.L1_MEMORY_CONFIG))
                    print(f"{n} cores ({wname}): norm {ta:.1f} us, with reshard in+out {tb:.1f} us, max diff {d:.2e}",
                          flush=True)
                except Exception as e:  # noqa: BLE001
                    print(f"{n} cores ({wname}): err {str(e)[:120]}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"{n} cores: err {str(e)[:120]}", flush=True)
    ttnn.close_mesh_device(mesh)
    print("== bench done")


if __name__ == "__main__":
    main()
