# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""ctypes wrapper of pinx.so (zero-copy host -> device writes from pinned host memory; see pinx.cpp)."""

import ctypes
import mmap
import os
import time

import numpy as np

from .nbptr import cpp_ptr

_SO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pinx.so")
TILE_BYTES = {"BFLOAT4_B": 576, "BFLOAT8_B": 1088}  # one 32x32 tile: mantissas + 64 shared exponents


def tile_bytes(dtype):
    name = str(dtype).split(".")[-1]
    if name not in TILE_BYTES:
        raise ValueError(f"pinx: no tile size for {dtype}")
    return TILE_BYTES[name]


class Pinx:
    def __init__(self, mesh):
        lib = ctypes.CDLL(_SO)
        u64, vp = ctypes.c_uint64, ctypes.c_void_p
        lib.pinx_pin.argtypes = [vp, vp, u64]
        lib.pinx_dst.argtypes = [vp, u64, u64, u64]
        lib.pinx_write.argtypes = [vp, ctypes.c_int, ctypes.c_int, ctypes.POINTER(u64), ctypes.c_int]
        lib.pinx_write_at.argtypes = [vp, ctypes.c_int, ctypes.c_int, ctypes.POINTER(u64), ctypes.c_int, u64, u64]
        lib.pinx_read.argtypes = [vp, ctypes.c_int, ctypes.c_int, ctypes.POINTER(u64)]
        lib.pinx_dst_size.restype = u64
        lib.pinx_dst_size.argtypes = [ctypes.c_int]
        self.lib, self.mesh = lib, mesh
        self.mp = ctypes.c_void_p(cpp_ptr(mesh))
        self.tp = len(mesh.get_device_ids())
        self.keep = []  # the mmaps backing pinned regions

    def region(self, size):
        """A page-aligned host region of `size` bytes, pinned for the mesh: (region id, uint8 numpy view)."""
        mm = mmap.mmap(-1, size)
        buf = np.frombuffer(mm, dtype=np.uint8)
        rid = self.lib.pinx_pin(self.mp, ctypes.c_void_p(buf.ctypes.data), size)
        if rid < 0:
            raise RuntimeError(f"pinx: the {size}-byte region could not be mapped to the NOC")
        self.keep.append(mm)
        return rid, buf

    def dst(self, tensor):
        """A destination view of an interleaved DRAM tile tensor (its per-device bytes)."""
        shp = [int(v) for v in tensor.shape]  # per device
        tiles = (shp[-2] // 32) * (shp[-1] // 32) * int(np.prod(shp[:-2]))
        tb = tile_bytes(tensor.dtype)
        return self.lib.pinx_dst(self.mp, tensor.buffer_address(), tiles * tb, tb), tiles * tb

    def dst_rm(self, tensor):
        """A destination view of an interleaved DRAM row-major bf16 tensor (one page per row)."""
        shp = [int(v) for v in tensor.shape]  # per device
        if "BFLOAT16" not in str(tensor.dtype):
            raise ValueError(f"pinx: row-major destinations are bf16 here, got {tensor.dtype}")
        page = shp[-1] * 2
        size = int(np.prod(shp[:-1])) * page
        return self.lib.pinx_dst(self.mp, tensor.buffer_address(), size, page), size

    def _offs(self, offsets):
        if len(offsets) != self.tp:
            raise ValueError(f"pinx: {len(offsets)} offsets for {self.tp} devices")
        return (ctypes.c_uint64 * self.tp)(*[int(o) for o in offsets])

    def write(self, dst, region, offsets, cq_id=0):
        rc = self.lib.pinx_write(self.mp, dst, region, self._offs(offsets), cq_id)
        if rc != 0:
            raise RuntimeError(f"pinx_write returned {rc} (dst {dst}, region {region}, offsets {list(offsets)})")

    def write_at(self, dst, region, offsets, dst_offset, size, cq_id=0):
        """Region bytes [offsets[d], + size) into the destination's device-local bytes [dst_offset, + size)."""
        rc = self.lib.pinx_write_at(self.mp, dst, region, self._offs(offsets), cq_id, int(dst_offset), int(size))
        if rc != 0:
            raise RuntimeError(
                f"pinx_write_at returned {rc} (dst {dst}, region {region}, offsets {list(offsets)}, "
                f"dst_offset {dst_offset}, size {size})"
            )

    def read(self, dst, region, offsets):
        rc = self.lib.pinx_read(self.mp, dst, region, self._offs(offsets))
        if rc != 0:
            raise RuntimeError(f"pinx_read returned {rc} (dst {dst}, region {region}, offsets {list(offsets)})")

    def release(self):
        self.lib.pinx_release()
        self.keep = []


class PinnedRows:
    """Small replicated row-major bf16 uploads (the hybrid's crafted logits rows and host partial per layer) written
    from slices of one pinned host region: no host tensor conversion, no copy into the command-queue hugepage. Each
    (key, destination) owns its slice, rewritten only when the same key is uploaded again (the next pass of that layer,
    after the device consumed the previous write)."""

    def __init__(self, px, size=16 << 20):
        self.px = px
        self.rid, self.buf = px.region(size)
        self.next = 0
        self.slices = {}  # (key, destination address) -> host offset
        self.dsts = {}  # destination address -> (pinx destination, bytes)
        self.t_copy = self.t_write = 0.0  # cumulative seconds: staging copies, pinned write calls

    def put(self, key, dst, t):
        """t: a torch bf16 tensor with the destination's per-device element count (replicated to every device)."""
        addr = dst.buffer_address()
        d = self.dsts.get(addr)
        if d is None:
            d = self.dsts[addr] = self.px.dst_rm(dst)
        n = t.numel() * 2
        if n != d[1]:
            raise ValueError(f"PinnedRows: {n} bytes for a {d[1]}-byte destination {list(dst.shape)} (key {key})")
        off = self.slices.get((key, addr))
        if off is None:
            off = self.slices[(key, addr)] = self.next
            self.next += -(-n // 4096) * 4096
            if self.next > len(self.buf):
                raise RuntimeError(f"PinnedRows: region of {len(self.buf)} bytes is full ({len(self.slices)} slices)")
        import torch

        t0 = time.perf_counter()
        torch.from_numpy(self.buf[off : off + n]).view(torch.bfloat16).copy_(t.reshape(-1))
        t1 = time.perf_counter()
        self.px.write(d[0], self.rid, [off] * self.px.tp)
        self.t_copy += t1 - t0
        self.t_write += time.perf_counter() - t1
