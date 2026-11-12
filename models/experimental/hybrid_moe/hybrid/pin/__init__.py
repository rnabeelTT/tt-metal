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
        lib.pinx_write_at2.argtypes = [
            vp,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(u64),
            ctypes.c_int,
            u64,
            u64,
            ctypes.c_int,
        ]
        lib.pinx_read_at.argtypes = [vp, ctypes.c_int, ctypes.c_int, ctypes.POINTER(u64), ctypes.c_int, u64, u64]
        lib.pinx_dst_size.restype = u64
        lib.pinx_dst_size.argtypes = [ctypes.c_int]
        self.lib, self.mesh = lib, mesh
        self.mp = ctypes.c_void_p(cpp_ptr(mesh))
        self.tp = len(mesh.get_device_ids())
        self.keep = []  # the mmaps backing pinned regions
        # HYB_PINX_ZEROCOPY=1: writes read the pinned region straight over the NOC. Off by default: measured on the
        # QB2 (P300x2, cache/test_pinx_write2.py), 5-15% of zero-copy slot writes never landed on 3 of the 4 dies
        # (CQ0 and CQ1 alike) while the copy path (bytes through the command queue) never failed in the same test
        self.zerocopy = os.environ.get("HYB_PINX_ZEROCOPY", "0") == "1"
        self.by_rid = {}  # region id -> its uint8 view

    def region(self, size):
        """A page-aligned host region of `size` bytes, pinned for the mesh: (region id, uint8 numpy view)."""
        mm = mmap.mmap(-1, size)
        buf = np.frombuffer(mm, dtype=np.uint8)
        rid = self.lib.pinx_pin(self.mp, ctypes.c_void_p(buf.ctypes.data), size)
        if rid < 0:
            raise RuntimeError(f"pinx: the {size}-byte region could not be mapped to the NOC")
        self.keep.append(mm)
        self.by_rid[rid] = buf  # region ids are process-wide (every Pinx shares pinx.so's table)
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
        if not self.zerocopy:
            return self.write_at2(dst, region, offsets, 0, 0, cq_id=cq_id, pinned=False)
        rc = self.lib.pinx_write(self.mp, dst, region, self._offs(offsets), cq_id)
        if rc != 0:
            raise RuntimeError(f"pinx_write returned {rc} (dst {dst}, region {region}, offsets {list(offsets)})")

    def write_at(self, dst, region, offsets, dst_offset, size, cq_id=0):
        """Region bytes [offsets[d], + size) into the destination's device-local bytes [dst_offset, + size)."""
        if not self.zerocopy:
            return self.write_at2(dst, region, offsets, dst_offset, size, cq_id=cq_id, pinned=False)
        rc = self.lib.pinx_write_at(self.mp, dst, region, self._offs(offsets), cq_id, int(dst_offset), int(size))
        if rc != 0:
            raise RuntimeError(
                f"pinx_write_at returned {rc} (dst {dst}, region {region}, offsets {list(offsets)}, "
                f"dst_offset {dst_offset}, size {size})"
            )

    def write_at2(self, dst, region, offsets, dst_offset, size, cq_id=0, pinned=True):
        """write_at through either path: pinned=False copies the bytes through the command queue (at enqueue)."""
        rc = self.lib.pinx_write_at2(
            self.mp, dst, region, self._offs(offsets), cq_id, int(dst_offset), int(size), 1 if pinned else 0
        )
        if rc != 0:
            raise RuntimeError(f"pinx_write_at2 returned {rc} (dst {dst}, region {region}, dst_offset {dst_offset})")

    def read(self, dst, region, offsets):
        rc = self.lib.pinx_read(self.mp, dst, region, self._offs(offsets))
        if rc != 0:
            raise RuntimeError(f"pinx_read returned {rc} (dst {dst}, region {region}, offsets {list(offsets)})")

    def read_at(self, dst, region, offsets, src_offset, size, cq_id=0):
        """Region bytes [offsets[d], + size) <- the destination's device-local bytes [src_offset, + size). Blocking."""
        rc = self.lib.pinx_read_at(self.mp, dst, region, self._offs(offsets), cq_id, int(src_offset), int(size))
        if rc != 0:
            raise RuntimeError(
                f"pinx_read_at returned {rc} (dst {dst}, region {region}, offsets {list(offsets)}, "
                f"src_offset {src_offset}, size {size})"
            )

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
        self.put_fill(key, dst, lambda v: v.copy_(t.reshape(v.shape)))

    def put_fill(self, key, dst, fill):
        """fill(view): writes the upload into ``view``, a torch bf16 [rows, width] view of the pinned slice (the
        destination's per-device shape), e.g. converting from fp32 in place; then the slice is written to ``dst``."""
        addr = dst.buffer_address()
        d = self.dsts.get(addr)
        if d is None:
            d = self.dsts[addr] = self.px.dst_rm(dst)
        n = d[1]
        off = self.slices.get((key, addr))
        if off is None:
            off = self.slices[(key, addr)] = self.next
            self.next += -(-n // 4096) * 4096
            if self.next > len(self.buf):
                raise RuntimeError(f"PinnedRows: region of {len(self.buf)} bytes is full ({len(self.slices)} slices)")
        import torch

        w = int(dst.shape[-1])
        t0 = time.perf_counter()
        fill(torch.from_numpy(self.buf[off : off + n]).view(torch.bfloat16).view(n // (2 * w), w))
        t1 = time.perf_counter()
        self.px.write(d[0], self.rid, [off] * self.px.tp)
        self.t_copy += t1 - t0
        self.t_write += time.perf_counter() - t1
