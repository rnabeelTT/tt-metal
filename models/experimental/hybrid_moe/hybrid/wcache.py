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
        self.hits = self.misses = self.gathered = self.gather_skipped = 0
        self.t_load = self.t_store = self.t_gather = 0.0
        self.specs = {}  # layer -> the spec of its last build (a full layer's, for gather())
        self.gather_src = {}  # layer -> (up_gate, down) of its full build, kept by an elastic demotion (gather())
        self.gather_check = os.environ.get("HYB_GATHER_CHECK") == "1"

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
        src = self.gather_src.pop(layer, None)
        if src is not None:
            out = self.gather(mesh, layer, spec, *src)
            if out is not None:
                self.specs[layer] = spec
                return out
        self.specs[layer] = spec
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

    def gather(self, mesh, layer, spec, up_gate, down):
        """An elastic demotion's hybrid tensors from the layer's full ones, on the device: slot s holds expert
        resident_fn(layer)[s] (ttnn.slice along the expert dim), an empty slot (id < 0: LRU / sink) a zero expert, and
        ttnn.concat stacks them in slot order. Tile-aligned slices copy whole bfp tiles, so the bytes are the full build's
        (cache/test_expert_gather.py: exact). Only when the full build's spec matches this one but for the expert count
        (dtypes, padding, rounding suffix); otherwise None (the image / conversion path). Frees the full tensors.
        HYB_GATHER_CHECK=1: the result is compared byte for byte with this layer's hybrid image file, when it exists."""
        full = self.specs.get(layer)
        ids = [int(e) for e in self.resident_fn(layer)]
        same = full is not None and {k: v for k, v in full.items() if k != "E"} == {
            k: v for k, v in spec.items() if k != "E"
        }
        if not same or len(ids) != spec["E"] or not self._gather_fits(mesh, up_gate, len(ids)):
            for t in (up_gate, down):
                ttnn.deallocate(t)
            return None
        t0 = time.perf_counter()
        out = []
        for t in (up_gate, down):
            shp = [int(v) for v in t.shape]  # per device [1, E_full, rows, cols]
            zero, parts = None, []
            for e in ids:
                if e < 0:
                    if zero is None:
                        zero = ttnn.zeros(
                            [1, 1, shp[2], shp[3]],
                            dtype=t.dtype,
                            layout=ttnn.TILE_LAYOUT,
                            device=mesh,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        )
                    parts.append(zero)
                else:
                    parts.append(
                        ttnn.slice(t, [0, e, 0, 0], [1, e + 1, shp[2], shp[3]], memory_config=ttnn.DRAM_MEMORY_CONFIG)
                    )
            g = ttnn.concat(parts, dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            for p in {id(p): p for p in parts}.values():
                ttnn.deallocate(p)
            ttnn.deallocate(t)
            out.append(g)
        ttnn.synchronize_device(mesh)
        self.gathered += 1
        self.t_gather += time.perf_counter() - t0
        if self.gather_check:
            self._check_gather(mesh, layer, spec, out)
        return tuple(out)

    def _gather_fits(self, mesh, up_gate, n_slots):
        """Whether the DRAM has room for a gather next to the full tensors it reads: the slices plus the contiguous
        concat output of the larger (gate|up) tensor, with 25% margin. The image path frees the full tensors first, so
        a tight DRAM (many elastic switches fragment it) takes that path instead."""
        mv = ttnn.get_memory_view(mesh, ttnn.BufferType.DRAM)
        from .pin import tile_bytes

        shp = [int(v) for v in up_gate.shape]  # per device [1, E_full, rows, cols]
        out = n_slots * (shp[2] // 32) * (shp[3] // 32) * tile_bytes(up_gate.dtype) // int(mv.num_banks)
        fits = (
            int(mv.total_bytes_free_per_bank) >= 2.5 * out
            and int(mv.largest_contiguous_bytes_free_per_bank) >= 1.25 * out
        )
        if not fits:
            self.gather_skipped += 1
        return fits

    def _check_gather(self, mesh, layer, spec, out):
        paths = self._paths(layer, self._key(layer, spec))
        if not all(os.path.exists(p) for p in paths):
            print(f"HYB_GATHER_CHECK: layer {layer}: no hybrid image to compare with", flush=True)
            return
        px = self._pinx(mesh)
        tp = px.tp
        for path, t in zip(paths, out):
            dst, n = px.dst(t)
            rid, buf = self._region(tp * n)
            px.read(dst, rid, [d * n for d in range(tp)])
            with open(path, "rb") as f:
                ref = f.read()
            if bytes(memoryview(buf)[: tp * n]) != ref:
                raise RuntimeError(f"HYB_GATHER_CHECK: layer {layer}: gathered bytes differ from {path}")
        print(f"HYB_GATHER_CHECK: layer {layer}: gathered tensors identical to the hybrid image", flush=True)

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
            f"(image writes {self.t_store:.1f} s), {self.gathered} gathered on the device ({self.t_gather:.1f} s, "
            f"{self.gather_skipped} skipped for DRAM room)"
        )
