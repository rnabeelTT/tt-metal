# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Microbenchmark: a [32, K] x [K, N] decode linear, DRAM-interleaved (the port's day-0 form) vs DRAM-width-sharded
(tt/common.DecodeLinear), on the (1, 4) mesh; device time from repeated trace replays.

    python -m models.autoports.tencent_hy4_flash.tools.bench_dsh
"""

from __future__ import annotations

import time

import torch
import ttnn

from models.autoports.tencent_hy4_flash.tt.common import ckc


def timed(mesh, fn, n=200):
    out = fn()
    ttnn.deallocate(out)
    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
    outs = [fn() for _ in range(10)]
    ttnn.end_trace_capture(mesh, tid, cq_id=0)
    ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
    t0 = time.perf_counter()
    for _ in range(n // 10):
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh)
    dt = (time.perf_counter() - t0) / n
    ttnn.release_trace(mesh, tid)
    return dt * 1e6, outs[-1]


def main():
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=200_000_000)
    try:
        rep = ttnn.ReplicateTensorToMesh(mesh)
        for K, N, dt, fid in [(3072, 1024 + 576, ttnn.bfloat16, ttnn.MathFidelity.HiFi4),
                              (3072, 1024 + 576 + 128 + 32 + 4096, ttnn.bfloat8_b, ttnn.MathFidelity.HiFi2),
                              (3072, 1024 + 576 + 128 + 32 + 4096, ttnn.bfloat16, ttnn.MathFidelity.HiFi4),
                              (3072, 1536 + 768, ttnn.bfloat16, ttnn.MathFidelity.HiFi4),
                              (4096, 3072, ttnn.bfloat8_b, ttnn.MathFidelity.HiFi2),
                              (1536, 3072, ttnn.bfloat16, ttnn.MathFidelity.HiFi4),
                              (3072, 30208, ttnn.bfloat16, ttnn.MathFidelity.HiFi4),
                              (3072, 30208, ttnn.bfloat8_b, ttnn.MathFidelity.HiFi2)]:
            torch.manual_seed(0)
            w = torch.randn(K, N) / K**0.5
            x = torch.randn(1, 1, 32, K)
            k = ckc(mesh, fid, fp32=True)
            xt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep,
                                 memory_config=ttnn.L1_MEMORY_CONFIG)
            wi = ttnn.from_torch(w.reshape(1, 1, K, N), dtype=dt, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep)
            t_i, oi = timed(mesh, lambda: ttnn.matmul(xt, wi, compute_kernel_config=k, memory_config=ttnn.L1_MEMORY_CONFIG))
            res = {}
            from models.demos.blackhole.qwen36.tt.decode_linear.ops import DecodeLinear as StreamLinear

            for rd in (4, 6, 8):
                try:
                    op = StreamLinear(mesh, K, N, dt, readers=rd, fidelity=fid)
                    ws = op.weight([w] * 4)
                    t_d, od = timed(mesh, lambda: op(xt, ws, 1))
                    a = ttnn.to_torch(ttnn.get_device_tensors(oi)[0]).float()[0, 0, 0]
                    b = ttnn.to_torch(ttnn.get_device_tensors(od)[0]).float().reshape(-1, N)[0]
                    res[rd] = (t_d, float((a - b).abs().max()), bool(torch.equal(a, b)))
                    ttnn.deallocate(ws)
                except Exception as e:  # noqa: BLE001
                    print(f"  readers {rd}: {str(e)[:120]}")
            print(f"K {K} N {N} {dt}: interleaved {t_i:7.1f} us | stream " +
                  " | ".join(f"{nc} readers {t:6.1f} us (max diff {d:.2e}, equal {e})" for nc, (t, d, e) in res.items()),
                  flush=True)
            ttnn.deallocate(wi)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
