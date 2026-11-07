# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""hybrid/pin/pinx.so on the device: raw bytes of a bfp4 staging tensor read into pinned host memory, written back
zero-copy (exact round trip), and the write time vs ttnn.copy_host_to_device_tensor (the hugepage path).
usage (in the image): python dbg_pinx.py <out dir>"""
import ctypes
import mmap
import sys
import time

sys.path.insert(0, "/poc")
import numpy as np  # noqa: E402
import torch  # noqa: E402

import ttnn  # noqa: E402
from hybrid.pin.nbptr import cpp_ptr  # noqa: E402

ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4))
try:
    lib = ctypes.CDLL("/poc/hybrid/pin/pinx.so")
    lib.pinx_pin.restype = ctypes.c_int
    lib.pinx_pin.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64]
    lib.pinx_dst.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64]
    lib.pinx_write.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_int,
    ]
    lib.pinx_read.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_uint64)]
    lib.pinx_dst_size.restype = ctypes.c_uint64
    mp = ctypes.c_void_p(cpp_ptr(mesh))
    H, Wl, tp = 3072, 768, 4
    sh = ttnn.ShardTensorToMesh(mesh, dim=-1)
    st = ttnn.from_torch(
        torch.zeros(1, 1, H, tp * Wl),
        dtype=ttnn.bfloat4_b,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        mesh_mapper=sh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tiles = (H // 32) * (Wl // 32)
    page = 576  # bfp4 tile: 1024 x 4 bits + 64 exponents
    n = tiles * page
    dst = lib.pinx_dst(mp, st.buffer_address(), n, page)
    print(f"staging {hex(st.buffer_address())} {n} B per die, dst {dst} (size {lib.pinx_dst_size(dst)})", flush=True)
    size = 64 << 20
    mm = mmap.mmap(-1, size)  # page-aligned anonymous memory
    buf = np.frombuffer(mm, dtype=np.uint8)
    region = lib.pinx_pin(mp, ctypes.c_void_p(buf.ctypes.data), size)
    print(f"pinned region {region}", flush=True)
    content = torch.randn(1, 1, H, tp * Wl) * 0.02
    ttnn.copy_host_to_device_tensor(
        ttnn.from_torch(content, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT, mesh_mapper=sh), st
    )
    want = ttnn.to_torch(st, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=-1)).float()
    offs = (ctypes.c_uint64 * tp)(*[d * n for d in range(tp)])
    print("read rc", lib.pinx_read(mp, dst, region, offs), flush=True)
    ttnn.copy_host_to_device_tensor(
        ttnn.from_torch(torch.zeros(1, 1, H, tp * Wl), dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT, mesh_mapper=sh),
        st,
    )
    print("write rc", lib.pinx_write(mp, dst, region, offs, 0), flush=True)
    ttnn.synchronize_device(mesh)
    got = ttnn.to_torch(st, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=-1)).float()
    print(f"RESULT zero-copy round trip exact: {torch.equal(got, want)}", flush=True)
    host_t = ttnn.from_torch(content, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT, mesh_mapper=sh)
    for name, fn in (
        ("hugepage copy", lambda: ttnn.copy_host_to_device_tensor(host_t, st)),
        ("zero-copy", lambda: lib.pinx_write(mp, dst, region, offs, 0)),
    ):
        ts = []
        for _ in range(40):
            t0 = time.perf_counter()
            fn()
            ttnn.synchronize_device(mesh)
            ts.append(time.perf_counter() - t0)
        ms = 1e3 * sorted(ts)[20]
        print(f"RESULT {name}: {ms:.3f} ms for {tp * n / 1e6:.2f} MB -> {tp * n / ms / 1e6:.1f} GB/s", flush=True)
    # host time only (the call returns before the device finishes): what the host step pays
    ts = []
    for _ in range(40):
        t0 = time.perf_counter()
        lib.pinx_write(mp, dst, region, offs, 0)
        ts.append(time.perf_counter() - t0)
        ttnn.synchronize_device(mesh)
    print(f"RESULT zero-copy host call: {1e3 * sorted(ts)[20]:.3f} ms", flush=True)
finally:
    ttnn.close_mesh_device(mesh)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
