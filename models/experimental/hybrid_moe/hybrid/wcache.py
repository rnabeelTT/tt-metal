# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Raw device-image cache of the routed-expert weights (the port's ``moe.EXPERT_WEIGHT_HOOK``; HYB_WCACHE=<dir>).

A cold build converts each MoE layer's experts from the FP8 checkpoint (dequant, 5-bit rounding, host bfp4 packing:
~20 s per layer, single-threaded under the GIL) and this cache then reads the uploaded tensors' per-die bytes back
(pinned, ``hybrid.pin.Pinx``) into ``<dir>/L<layer>_<key>.gu|dn``. A warm build allocates the same interleaved DRAM
tensors and writes those bytes straight back: the device holds the exact bytes of the conversion, so numerics cannot
change. The key hashes everything that decides the bytes: the layer, its slot -> expert ids (``HYB.plan``: the static
set, LRU fill and sink of a hybrid layer, or all experts of a full layer), the shapes / dtypes / padding / rounding
suffix the port reports, the TP width and the port. A different key is a miss (the layer converts and adds its image
next to the others), never a wrong load. Only 32x32-tile bfp4 / bfp8 weights are cached."""

from __future__ import annotations

import glob
import hashlib
import json
import os
import time

import ttnn

VERSION = 1
_DT = {"DataType.BFLOAT4_B": ttnn.bfloat4_b, "DataType.BFLOAT8_B": ttnn.bfloat8_b}


class ExpertImageCache:
    def __init__(self, root, resident_fn, tag=""):
        self.root, self.resident_fn, self.tag = root, resident_fn, tag
        os.makedirs(root, exist_ok=True)
        self.px = None
        self.regions = {}  # bytes -> (region id, uint8 view)
        self.hits = self.misses = 0
        self.t_load = self.t_store = 0.0

    # ------------------------------------------------------------------ helpers
    def _key(self, layer, spec):
        ids = [int(e) for e in self.resident_fn(layer)]
        blob = json.dumps({"v": VERSION, "tag": self.tag, "layer": int(layer), "ids": ids, **spec}, sort_keys=True)
        return hashlib.sha1(blob.encode()).hexdigest()[:16]

    def _paths(self, layer, key):
        base = os.path.join(self.root, f"L{int(layer):02d}_{key}")
        return base + ".gu", base + ".dn"

    def _pinx(self, mesh):
        if self.px is None:
            from .pin import Pinx

            self.px = Pinx(mesh)
        return self.px

    def _region(self, size):
        if size not in self.regions:
            self.regions[size] = self.px.region(size)
        return self.regions[size]

    @staticmethod
    def _cacheable(spec):
        return spec["gate_up_tile"] == "None" and spec["gate_up_dtype"] in _DT and spec["down_dtype"] in _DT

    def _shapes(self, spec):
        E, H, Nlp = spec["E"], spec["H"], spec["Nlp"]
        return [1, E, H, 2 * Nlp], [1, E, Nlp, H]  # per device: gate|up columns, down rows

    def _known(self, layer):
        try:
            self.resident_fn(layer)
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------ hook interface
    def load(self, mesh, layer, spec):
        if layer is None or not self._cacheable(spec) or not self._known(layer):
            return None
        pgu, pdn = self._paths(layer, self._key(layer, spec))
        if not (os.path.exists(pgu) and os.path.exists(pdn)):
            self.misses += 1
            return None
        t0 = time.perf_counter()
        px = self._pinx(mesh)
        tp = px.tp
        out = []
        for path, shape, dt in zip((pgu, pdn), self._shapes(spec), (spec["gate_up_dtype"], spec["down_dtype"])):
            t = ttnn.allocate_tensor_on_device(
                ttnn.Shape(shape), _DT[dt], ttnn.TILE_LAYOUT, mesh, ttnn.DRAM_MEMORY_CONFIG
            )
            dst, n = px.dst(t)
            size = os.path.getsize(path)
            if size != tp * n:
                raise RuntimeError(f"wcache: {path} holds {size} bytes, expected {tp} x {n} (shape {shape}, {dt})")
            rid, buf = self._region(tp * n)
            with open(path, "rb", buffering=0) as f:
                got = f.readinto(memoryview(buf)[: tp * n])
            if got != tp * n:
                raise RuntimeError(f"wcache: short read of {path}: {got} of {tp * n} bytes")
            px.write(dst, rid, [d * n for d in range(tp)])
            ttnn.synchronize_device(mesh)  # the region is reused by the next weight
            out.append(t)
        self.hits += 1
        self.t_load += time.perf_counter() - t0
        return tuple(out)

    def store(self, mesh, layer, spec, up_gate, down):
        if layer is None or not self._cacheable(spec) or not self._known(layer):
            return
        t0 = time.perf_counter()
        key = self._key(layer, spec)
        for old in glob.glob(os.path.join(self.root, f"L{int(layer):02d}_*.tmp")):
            os.remove(old)  # a partial write of an earlier, interrupted build
        px = self._pinx(mesh)
        tp = px.tp
        ttnn.synchronize_device(mesh)
        for path, t, shape in zip(self._paths(layer, key), (up_gate, down), self._shapes(spec)):
            if [int(v) for v in t.shape] != shape:
                raise RuntimeError(f"wcache: layer {layer} tensor shape {list(t.shape)}, expected per device {shape}")
            dst, n = px.dst(t)
            rid, buf = self._region(tp * n)
            px.read(dst, rid, [d * n for d in range(tp)])
            tmp = path + ".tmp"
            with open(tmp, "wb", buffering=0) as f:
                f.write(memoryview(buf)[: tp * n])
            os.replace(tmp, path)
        self.t_store += time.perf_counter() - t0

    def summary(self):
        return (
            f"HYB_WCACHE: {self.hits} layers loaded from images ({self.t_load:.1f} s), {self.misses} converted "
            f"(image writes {self.t_store:.1f} s)"
        )
