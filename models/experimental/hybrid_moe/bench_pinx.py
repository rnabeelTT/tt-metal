# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Microbenchmark: host -> device writes of one expert-sized bfp4 weight (per die ~1.3 MB gate|up, ~0.66 MB down):
pinned zero-copy (pinx write_at into a slot of a [1, C, H, W] tensor) vs copy_host_to_device_tensor; host enqueue time
and completed time. usage (in the image): python bench_pinx.py"""

import sys
import time

sys.path.insert(0, "/poc")
import torch  # noqa: E402

import ttnn  # noqa: E402
from hybrid.pin import Pinx  # noqa: E402


def main():
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), l1_small_size=24576, trace_region_size=10_000_000)
    try:
        tp = 4
        H, W, C = 3072, 768, int(__import__("os").environ.get("BENCH_C", "8"))
        kw = dict(layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat4_b)
        big = ttnn.from_torch(torch.zeros(1, C, H, tp * W), mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=-1), **kw)
        one = ttnn.from_torch(torch.randn(1, 1, H, tp * W), mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=-1), **kw)
        host = ttnn.from_torch(
            torch.randn(1, 1, H, tp * W),
            dtype=ttnn.bfloat4_b,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=-1),
        )
        px = Pinx(mesh)
        d_one, n = px.dst(one)
        d_big, _ = px.dst(big)
        rid, _ = px.region(8 * tp * n)
        px.read(d_one, rid, [d * n for d in range(tp)])
        for name, fn in (
            ("pinx write_at (slot 3)", lambda i: px.write_at(d_big, rid, [d * n for d in range(tp)], (i % C) * n, n)),
            ("pinx write whole", lambda i: px.write(d_one, rid, [d * n for d in range(tp)])),
            ("copy_host_to_device", lambda i: ttnn.copy_host_to_device_tensor(host, one)),
        ):
            for nw in (1, 4):
                ttnn.synchronize_device(mesh)
                t0 = time.perf_counter()
                for i in range(nw):
                    fn(i)
                t1 = time.perf_counter()
                ttnn.synchronize_device(mesh)
                t2 = time.perf_counter()
                reps = []
                for _ in range(10):
                    ttnn.synchronize_device(mesh)
                    a = time.perf_counter()
                    for i in range(nw):
                        fn(i)
                    b = time.perf_counter()
                    ttnn.synchronize_device(mesh)
                    reps.append((b - a, time.perf_counter() - a))
                e = sorted(r[0] for r in reps)[5] * 1e6 / nw
                c = sorted(r[1] for r in reps)[5] * 1e6 / nw
                print(
                    f"{name:26s} x{nw}: enqueue {e:7.1f} us, complete {c:7.1f} us per write "
                    f"({n / 1e6:.2f} MB per die, {tp * n / c / 1e3:.1f} GB/s all dies)",
                    flush=True,
                )
        # many large pinned regions (the model pins ~34 GB in 42 per-layer regions): write from each in turn
        import os

        nreg, rsz = int(os.environ.get("NREG", "12")), int(os.environ.get("RSZ_GB", "2")) << 30
        regs = []
        for _ in range(nreg):
            r, buf = px.region(rsz)
            buf[:] = 1 if os.environ.get("TOUCH_ALL") == "1" else 0  # touch (all pages: like the model)
            buf[: tp * n] = 1
            regs.append(r)
        print(f"pinned {nreg} x {rsz >> 30} GB", flush=True)
        for nw in (1, 4):
            reps = []
            for k in range(10):
                ttnn.synchronize_device(mesh)
                a = time.perf_counter()
                for i in range(nw):
                    px.write_at(d_big, regs[(k * nw + i) % nreg], [d * n for d in range(tp)], (i % C) * n, n)
                b = time.perf_counter()
                ttnn.synchronize_device(mesh)
                reps.append((b - a, time.perf_counter() - a))
            e = sorted(r[0] for r in reps)[5] * 1e6 / nw
            c = sorted(r[1] for r in reps)[5] * 1e6 / nw
            print(
                f"{'many regions, rotating':26s} x{nw}: enqueue {e:7.1f} us, complete {c:7.1f} us per write", flush=True
            )
        # with a concurrent DRAM-streaming CPU load (the hybrid's CPU experts run while the fetch writes are enqueued)
        import threading

        import numpy as np

        stop = [False]
        arrs = [np.ones(1 << 28, dtype=np.uint8) for _ in range(8)]  # 8 x 256 MB

        def hog(a):
            while not stop[0]:
                a.sum()  # numpy releases the GIL in the reduction loop

        ths = [threading.Thread(target=hog, args=(a,), daemon=True) for a in arrs]
        for t in ths:
            t.start()
        time.sleep(0.5)
        for nw in (1, 4):
            reps = []
            for k in range(10):
                ttnn.synchronize_device(mesh)
                a = time.perf_counter()
                for i in range(nw):
                    px.write_at(d_big, regs[(k * nw + i) % nreg], [d * n for d in range(tp)], (i % C) * n, n)
                b = time.perf_counter()
                ttnn.synchronize_device(mesh)
                reps.append((b - a, time.perf_counter() - a))
            e = sorted(r[0] for r in reps)[5] * 1e6 / nw
            c = sorted(r[1] for r in reps)[5] * 1e6 / nw
            print(
                f"{'with 8-thread CPU load':26s} x{nw}: enqueue {e:7.1f} us, complete {c:7.1f} us per write", flush=True
            )
        stop[0] = True
        time.sleep(0.2)

        def timed(label, before=None, nw=2):
            reps = []
            for k in range(10):
                ttnn.synchronize_device(mesh)
                if before is not None:
                    before()
                a = time.perf_counter()
                for i in range(nw):
                    px.write_at(d_big, regs[(k * nw + i) % nreg], [d * n for d in range(tp)], (i % C) * n, n)
                b = time.perf_counter()
                ttnn.synchronize_device(mesh)
                reps.append(b - a)
            print(f"{label:34s}: enqueue {sorted(reps)[5] * 1e6 / nw:7.1f} us per write", flush=True)

        # a captured trace replayed (non-blocking) just before the writes, and one finished before them
        x = ttnn.from_torch(
            torch.randn(1, 1, 32, 4096),
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            dtype=ttnn.bfloat16,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )

        def body():
            y = x
            for _ in range(20):
                y = ttnn.add(y, x)
            return y

        body()
        tid = ttnn.begin_trace_capture(mesh, cq_id=0)
        body()
        ttnn.end_trace_capture(mesh, tid, cq_id=0)
        timed("after a finished trace replay", lambda: (ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)))
        timed("behind a running trace replay", lambda: ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False))
        # GIL: a Python thread doing pure-Python work
        stop2 = [False]

        def pyhog():
            v = 0
            while not stop2[0]:
                for i in range(1000):
                    v += i

        th = threading.Thread(target=pyhog, daemon=True)
        th.start()
        timed("with a pure-Python thread (GIL)")
        stop2[0] = True
        ttnn.release_trace(mesh, tid)
        # the model's sequence: CPU experts started on hostk's C thread (int4 library streaming at full DRAM bandwidth),
        # then the fetch writes, then the wait
        from hybrid import hostk

        hostk.load(4)
        d = torch.load("/hyblib/minimax_m27_int4_s152/layer_10.pt", mmap=True, weights_only=True)
        NE = 16
        lib = {k: d[k][:NE].clone().contiguous() for k in ("gu_p", "gu_sz", "dn_p", "dn_sz")}
        lib["gs"] = d["gs"]
        Hh = int(lib["dn_p"].shape[1])
        X = torch.randn(4, Hh)
        rows = torch.arange(4)
        Wt = torch.rand(4, 4)
        out = torch.zeros(4, Hh)
        for nw in (4,):
            reps, cpu = [], []
            for k in range(10):
                ttnn.synchronize_device(mesh)
                rk = rows + (k % 4) * 4
                job = hostk.AsyncMoe(hostk.moe_prepared(lib, X, rk, Wt, out), out, (X, rk, Wt))
                a = time.perf_counter()
                for i in range(nw):
                    px.write_at(d_big, regs[(k * nw + i) % nreg], [d * n for d in range(tp)], (i % C) * n, n)
                b = time.perf_counter()
                job.result()
                cpu.append(time.perf_counter() - a)
                ttnn.synchronize_device(mesh)
                reps.append(b - a)
            print(
                f"{'with hostk running (4 experts)':34s}: enqueue {sorted(reps)[5] * 1e6 / nw:7.1f} us per write, "
                f"CPU job done after {sorted(cpu)[5] * 1e6:7.1f} us",
                flush=True,
            )
        px.release()
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()
