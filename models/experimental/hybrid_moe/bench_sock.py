# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Microbenchmark: per-layer host <-> device round trips through sockets inside ONE trace (a candidate replacement for
the hybrid's per-layer trace segments + readback + pinned uploads). Per simulated layer the trace runs: send_async_d2h
of a page from device 0 (the routing logits and activations the host step reads), then recv_async_h2d of a page on each
of the 4 devices (the crafted routing row / host partial). The host loop answers each layer: D2H read, then the 4 H2D
writes. Reports microseconds per layer round trip for each H2D mode and page size.
usage (in the image): python bench_sock.py"""

import time

import torch

import ttnn

LAYERS = 42


def run(mesh, h2d_mode, d2h_page, h2d_page, passes=20):
    ndev = mesh.get_num_devices()
    core = ttnn.CoreCoord(0, 0)
    d2h = ttnn.D2HSocket(mesh, ttnn.MeshCoreCoord(ttnn.MeshCoordinate(0, 0), core), 4 * d2h_page)
    d2h.set_page_size(d2h_page)
    h2d = []
    for i in range(ndev):
        s = ttnn.H2DSocket(
            mesh, ttnn.MeshCoreCoord(ttnn.MeshCoordinate(0, i), core), ttnn.BufferType.L1, 4 * h2d_page, h2d_mode
        )
        s.set_page_size(h2d_page)
        h2d.append(s)
    rm = dict(dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, memory_config=ttnn.L1_MEMORY_CONFIG)
    src = ttnn.from_torch(torch.arange(d2h_page // 4, dtype=torch.int32).reshape(1, -1), **rm)
    dst = ttnn.from_torch(torch.zeros(1, h2d_page // 4, dtype=torch.int32), **rm)
    rd = torch.zeros(d2h_page // 4, dtype=torch.int32)
    wr = torch.arange(h2d_page // 4, dtype=torch.int32)

    def body():
        for _ in range(LAYERS):
            ttnn.experimental.send_async_d2h(src, d2h)
            for s in h2d:
                ttnn.experimental.recv_async_h2d(dst, s)

    def host():
        for _ in range(LAYERS):
            d2h.read_tensor(rd)
            for s in h2d:
                s.write_tensor(wr)

    # eager pass first (programs compiled), then a trace of one pass
    body()
    host()
    ttnn.synchronize_device(mesh)
    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
    body()
    ttnn.end_trace_capture(mesh, tid, cq_id=0)
    ts = []
    for _ in range(passes):
        t0 = time.perf_counter()
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
        host()
        ttnn.synchronize_device(mesh)
        ts.append(time.perf_counter() - t0)
    ttnn.release_trace(mesh, tid)
    ts.sort()
    us = ts[len(ts) // 2] / LAYERS * 1e6
    print(f"SOCK mode {h2d_mode} d2h page {d2h_page} h2d page {h2d_page}: {us:.1f} us per layer round trip", flush=True)


def main():
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), l1_small_size=24576, trace_region_size=64 << 20)
    try:
        for mode in (ttnn.H2DMode.HOST_PUSH, ttnn.H2DMode.DEVICE_PULL):
            for d2h_page, h2d_page in ((4096, 2048), (32768, 2048), (32768, 24576)):
                try:
                    run(mesh, mode, d2h_page, h2d_page)
                except Exception as e:  # a mode / size the build rejects
                    print(
                        f"SOCK mode {mode} {d2h_page}/{h2d_page}: failed ({str(e).splitlines()[0][:160]})", flush=True
                    )
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
