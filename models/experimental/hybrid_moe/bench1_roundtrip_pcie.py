# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Hybrid-MoE POC, microbenchmark 1 (QB2, 4x Blackhole, mesh (1,4)).

(A) Per-layer host round trip: what a trace split at every MoE layer costs versus one trace.
    Each "layer" is two traced segments. Between them the host reads back a small routing-sized
    tensor and writes a hidden-sized result, which is what exact hybrid decode needs per layer.
(B) Host -> device write bandwidth into a preallocated DRAM tensor, bfp4 expert-slice sizes,
    sharded across the 4 dies, on CQ0 idle and on CQ1 while a trace loops on CQ0.
"""

import json
import statistics as st
import sys
import time

import torch
import ttnn

H = 4096  # 397B hidden size (the 122B is 3072); decode activation [1,1,32,H]
LAYERS = 60
REPS = 5


def now():
    return time.perf_counter()


def main():
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=256 << 20, num_command_queues=2)
    rep = ttnn.ReplicateTensorToMesh(mesh)
    res = {}
    try:
        # ---------------- (A) round trip ----------------
        x = ttnn.from_torch(
            torch.randn(1, 1, 32, H), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep
        )
        # a decode-sized matmul per segment so each segment does real device work (~tens of us)
        w = ttnn.from_torch(
            torch.randn(H, 1024) * 0.01, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep
        )
        w2 = ttnn.from_torch(
            torch.randn(1024, H) * 0.01, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep
        )
        host_in_dev = ttnn.from_torch(
            torch.zeros(1, 1, 32, H), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep
        )
        host_in_host = ttnn.from_torch(
            torch.randn(1, 1, 32, H), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=rep
        )

        def seg_a():
            h = ttnn.linear(x, w)
            small = ttnn.slice(h, [0, 0, 0, 0], [1, 1, 32, 32])  # stands in for top-k ids/scores
            return h, small

        def seg_b(h):
            y = ttnn.linear(h, w2)
            return ttnn.add(y, host_in_dev)

        # compile
        h, small = seg_a()
        y = seg_b(h)
        ttnn.synchronize_device(mesh)

        tid_ab = ttnn.begin_trace_capture(mesh, cq_id=0)
        h_ab, small_ab = seg_a()
        y_ab = seg_b(h_ab)
        ttnn.end_trace_capture(mesh, tid_ab, cq_id=0)

        tid_a = ttnn.begin_trace_capture(mesh, cq_id=0)
        h_a, small_a = seg_a()
        ttnn.end_trace_capture(mesh, tid_a, cq_id=0)
        tid_b = ttnn.begin_trace_capture(mesh, cq_id=0)
        y_b = seg_b(h_a)
        ttnn.end_trace_capture(mesh, tid_b, cq_id=0)

        die0 = lambda t: ttnn.get_device_tensors(t)[0]  # noqa: E731

        def run(kind):
            ttnn.synchronize_device(mesh)
            t0 = now()
            for _ in range(LAYERS):
                if kind == "one_trace":
                    ttnn.execute_trace(mesh, tid_ab, cq_id=0, blocking=False)
                    continue
                ttnn.execute_trace(mesh, tid_a, cq_id=0, blocking=False)
                if kind in ("split_read", "split_read_write"):
                    ttnn.to_torch(die0(small_a))  # blocks until segment A finished
                if kind in ("split_write", "split_read_write"):
                    ttnn.copy_host_to_device_tensor(host_in_host, host_in_dev, cq_id=0)
                ttnn.execute_trace(mesh, tid_b, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh)
            return (now() - t0) * 1e6 / LAYERS  # us per layer

        kinds = ["one_trace", "split", "split_read", "split_write", "split_read_write"]
        for k in kinds:
            run(k)  # warm
        a = {k: st.median([run(k) for _ in range(REPS)]) for k in kinds}
        a["overhead_us_per_layer"] = a["split_read_write"] - a["one_trace"]
        a["overhead_ms_per_token_60_layers"] = a["overhead_us_per_layer"] * 60 / 1000
        res["roundtrip_us_per_layer"] = a
        print(json.dumps(a, indent=1), flush=True)

        # ---------------- (B) host -> device bandwidth ----------------
        bw = {}
        loop_trace = tid_ab
        for per_die_mb_target in (1.75, 16, 128):
            # bfp4_b ~= 0.5625 B/elem; shape [1,1,4096, W*4] sharded on the last dim
            wcols = int(per_die_mb_target * 1e6 / 0.5625 / 4096 // 32 * 32)
            src = torch.randn(1, 1, 4096, wcols * 4, dtype=torch.bfloat16)
            host_t = ttnn.from_torch(
                src, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT, mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=-1)
            )
            dev_t = ttnn.from_torch(
                src,
                dtype=ttnn.bfloat4_b,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=-1),
            )
            del src
            total_bytes = 4096 * wcols * 4 * 0.5625
            n = max(3, int(512e6 // total_bytes))
            for cq, busy in ((0, False), (1, False), (1, True)):
                ttnn.synchronize_device(mesh)
                if busy:
                    for _ in range(400):  # keep CQ0 busy for the whole measurement
                        ttnn.execute_trace(mesh, loop_trace, cq_id=0, blocking=False)
                t0 = now()
                for _ in range(n):
                    ttnn.copy_host_to_device_tensor(host_t, dev_t, cq_id=cq)
                ttnn.synchronize_device(mesh, cq_id=cq) if cq else ttnn.synchronize_device(mesh)
                dt = now() - t0
                key = f"{per_die_mb_target}MB_per_die_cq{cq}{'_busy' if busy else ''}"
                bw[key] = {
                    "GB_per_s_total": round(total_bytes * n / dt / 1e9, 2),
                    "ms_per_copy": round(dt * 1e3 / n, 3),
                }
                ttnn.synchronize_device(mesh)
                print(key, bw[key], flush=True)
            ttnn.deallocate(dev_t)
        res["h2d_bandwidth"] = bw
    finally:
        ttnn.close_mesh_device(mesh)
    out = sys.argv[1] if len(sys.argv) > 1 else "bench1.json"
    with open(out, "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
