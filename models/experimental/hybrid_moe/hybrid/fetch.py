# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Fetched experts (the FreeToken "hybrid" policy on the fused hybrid MoE): per layer, D of the C device slots are an
LRU cache. At every decode step a share f of the layer's missed experts is copied host -> device into LRU slots and
computed by the device kernels with the resident ones; the CPU computes the rest at the same time. A fetched expert
stays until evicted, so a miss that recurs over the next steps (consecutive tokens share many experts) is a hit.

Slots per layer: [0, S) static (the plan's top S = C - 1 - D), [S, S + D) LRU, C - 1 the all-zero sink.

Device side, direct (HYB_FETCH_DIRECT=1, default with HYB_PINX): the host step writes each fetched expert from its
pinned host copy straight into its LRU slot of the layer's weights (one zero-copy PCIe write per weight, ordered on
CQ0 ahead of the segment that computes the experts); nothing runs inside the trace and the sink is never written.
Otherwise, per fused layer and step (inside the trace, after the host step that routed the layer): F_MAX
paged_fill_cache writes of the model-wide staging buffers into the layer's [C, 1, H, W] views of its up_gate / down
weights, at the slots in a device page table [F_MAX, 1] the host step wrote; staging rows with nothing to fetch point
at the sink, which a last pair of writes from all-zero staging resets.

Host side: the device-format (bfp4, TP-sharded, rounded like the port's fused weights) copies of every expert that can
miss, loaded from HYB_DEV_EXPERTS (build_dev_experts.py). Which misses to fetch: the most recently active first
(FreeToken's recency rule; ties to the lower id); how many: f x misses, rounded stochastically and capped by F_MAX.

env: HYB_LRU (D slots per layer, 0 = off), HYB_FETCH_FRAC (f, default 0.4), HYB_FETCH_MAX (F_MAX, default 2),
HYB_DEV_EXPERTS (dir of <L>_<e>_gu.tensorbin / <L>_<e>_dn.tensorbin), HYB_FETCH_DIRECT (default 1 with HYB_PINX),
HYB_FETCH_ASYNC (direct only, needs a second command queue): the fetch writes go on CQ1 and the fetched experts become
resident from the NEXT pass (commit(): CQ0 waits for CQ1's writes before the pass's replay); this pass the CPU
computes them with the other misses. The device never waits for a transfer, which overlaps the next segments.

Per-request re-plan (HYB_REPLAN = swaps per layer, 0 = off; needs the direct fetch): after a prefill of at least
HYB_REPLAN_MIN tokens, each hybrid layer swaps up to HYB_REPLAN static experts the prompt never routed to (least train
mass first) for the non-static experts the prompt routed the most mass to. A victim's device bytes are read back into
a pinned victim store (HYB_REPLAN_STORE experts per layer) unless the cold library already holds them; a victim with
no host int4 row (``nohost``) is never computed on the CPU: every decode miss of one is fetched into an LRU slot, and
prefill streams it to the device whatever its token count.
"""

from __future__ import annotations

import os
import time
from collections import OrderedDict

import numpy as np
import torch

import ttnn


class FetchCache:
    def __init__(self):
        self.D = int(os.environ.get("HYB_LRU", "0") or 0)
        self.f = float(os.environ.get("HYB_FETCH_FRAC", "0.4"))
        self.F = int(os.environ.get("HYB_FETCH_MAX", "2"))
        self.dir = os.environ.get("HYB_DEV_EXPERTS", "")
        if self.D and not self.dir:
            raise ValueError("HYB_LRU needs HYB_DEV_EXPERTS (the device-format copies of the experts that can miss)")
        if self.D and not (0 < self.F <= self.D):
            raise ValueError(f"HYB_FETCH_MAX={self.F} must be in 1..HYB_LRU={self.D}")
        self.rng = np.random.default_rng(int(os.environ.get("HYB_FETCH_SEED", "0")))
        self.lru = {}  # layer -> OrderedDict expert -> slot (least recent first)
        self.free = {}  # layer -> free LRU slots
        self.stamp = {}  # layer -> [E] last step each expert was routed (-1: never)
        self.host = {}  # (layer, expert) -> (gu, dn) host tensors in the device format (until finalize)
        self.step = 0
        self.dev = None  # model-wide staging, created on the first layer (before any capture)
        # HYB_PINX: the device-format experts as raw device bytes in host memory pinned for the mesh (hybrid/pin):
        # fetch / stream writes read them straight over PCIe (no CPU memcpy into the command-queue hugepage)
        self.pinx_on = bool(self.dir) and os.environ.get("HYB_PINX", "1") == "1"
        self.px = None
        self.direct = self.pinx_on and os.environ.get("HYB_FETCH_DIRECT", "1") == "1"
        self.rows = None  # pin.PinnedRows after finalize (HYB_PIN_UPLOADS)
        self.async_on = self.direct and os.environ.get("HYB_FETCH_ASYNC", "0") == "1"
        self.pending = []  # async: (layer, expert, slot) written on CQ1, resident after the next commit()
        self.mods = {}  # layer -> its fused MoE module (direct: the weights the fetched experts are written into)
        self.wdst = {}  # layer -> pinx destinations of its (up_gate, down) weights (direct)
        self.raw = {}  # layer -> (region id, {expert: index}, gate|up bytes per die, down bytes per die)
        self.shapes = None  # per-device ([1, 1, H, 2 Nl], [1, 1, Nl, H]) and dtypes of one expert (note_shapes)
        self.replan_n = int(os.environ.get("HYB_REPLAN", "0") or 0)
        self.replan_min = int(os.environ.get("HYB_REPLAN_MIN", "256"))
        self.store_n = int(os.environ.get("HYB_REPLAN_STORE", str(self.replan_n + self.replan_n // 2)))
        if self.replan_n and (not self.direct or self.async_on):
            raise ValueError("HYB_REPLAN needs the synchronous direct fetch (HYB_FETCH_DIRECT=1, HYB_FETCH_ASYNC=0)")
        self.store = {}  # layer -> (region id, {expert: store index}) victims' device bytes (pinned)
        # HYB_WCACHE: a layer's pinned region (raw device bytes of its missable experts) cached as one file, read
        # straight into the region on later builds / elastic demotions instead of loading and converting the copies
        self.pin_cache = os.environ.get("HYB_WCACHE", "")
        self.raw_files = {}  # layer -> (cached region file, experts) found by load_host, pinned by finalize
        # HYB_FETCH_POLICY=dyn (FreeToken-style, choose_dyn): a per-pass budget of fetched experts (HYB_FETCH_BUDGET,
        # spread over the layers by their share of recent misses) and value-aware admission: per layer a decaying use
        # score per expert (HYB_FETCH_DECAY per step); a miss is fetched only if it outscores the weakest evictable LRU
        # resident by HYB_FETCH_MARGIN. "fixed" (default): a fraction f of the misses, at most F per layer, recency order.
        # HYB_PINX_BGWRITE (default 1): the async (CQ1) fetch copies are issued from pinx's background thread
        # (pinx_write_async: the copy path's host memcpy leaves this thread); commit() flushes them before its CQ1 event.
        # runs/m27/fb_bg1 vs fb_bg0: short 55.0 -> 57.2, 8K 41.2 -> 42.6 tok/s; log-probs identical (lp_bg)
        self.bgwrite = os.environ.get("HYB_PINX_BGWRITE", "1") == "1"
        self.policy = os.environ.get("HYB_FETCH_POLICY", "fixed")
        self.budget = float(os.environ.get("HYB_FETCH_BUDGET", "8"))
        self.decay = float(os.environ.get("HYB_FETCH_DECAY", "0.9"))
        self.margin = float(os.environ.get("HYB_FETCH_MARGIN", "0.25"))
        self.score = {}  # layer -> [E] decaying use score (dyn)
        self.miss_ema = {}  # layer -> EMA of misses per step (dyn: the budget split)
        v = os.environ.get("HYB_FETCH_VERIFY", "")
        self.verify_layer = int(v) if v else None  # debug: read back one layer's LRU slots at commits (_verify)
        self.nohost = {}  # layer -> experts off the device with no host int4 row (always fetched, never on the CPU)

    @property
    def on(self):
        return self.D > 0

    # -- setup ------------------------------------------------------------------------------------------------------
    def plan_layer(self, layer, E, C, static_ids):
        """The slot list of a layer: static experts, D empty LRU slots (zero experts), the sink."""
        S = C - 1 - self.D
        self.__dict__.setdefault("_static_n", {})[layer] = S
        self.lru[layer] = OrderedDict()
        self.free[layer] = list(range(S, S + self.D))
        self.stamp[layer] = np.full(E, -1, dtype=np.int64)
        return [int(i) for i in static_ids[:S]] + [-1] * self.D + [-1]

    def load_host(self, layer, experts):
        """The device-format copies of a layer's missable experts (host RAM; read once at build). A layer whose copies
        are already pinned (an earlier elastic demotion) is not read again."""
        if layer in self.raw:
            return
        path = self._pin_path(layer, experts)
        if path is not None and os.path.exists(path):  # their pinned bytes are cached: read straight into the region
            self.raw_files[layer] = (path, sorted(int(e) for e in experts))
            return
        for e in experts:
            p = os.path.join(self.dir, f"{layer}_{e}")
            self.host[(layer, e)] = (ttnn.load_tensor(p + "_gu.tensorbin"), ttnn.load_tensor(p + "_dn.tensorbin"))

    def _pin_path(self, layer, experts):
        if not (self.pin_cache and self.pinx_on and self.shapes is not None):
            return None
        import hashlib
        import json

        gsh, dsh, gdt, ddt, mesh = self.shapes
        key = json.dumps(
            [
                1,
                int(layer),
                sorted(int(e) for e in experts),
                gsh,
                dsh,
                str(gdt),
                str(ddt),
                len(mesh.get_device_ids()),
                os.path.abspath(self.dir),
            ]
        )
        return os.path.join(self.pin_cache, f"pin_L{int(layer):02d}_{hashlib.sha1(key.encode()).hexdigest()[:16]}.raw")

    def _pin_layers(self, layers, conv):
        """Pinned regions of the given layers' device-format experts (finalize / finalize_more): from the region cache
        file when load_host found one, else each copy written to the staging tensors and read back raw (then cached)."""
        px, tp = self.px, self.px.tp
        (dg, ng), (dd, nd) = px.dst(conv[0]), px.dst(conv[1])
        n_cached = 0
        for L in sorted(layers):
            if L in self.raw:
                continue
            cached = self.raw_files.pop(L, None)
            if cached is not None:
                path, experts = cached
                size = len(experts) * tp * (ng + nd)
                if os.path.getsize(path) != size:
                    raise RuntimeError(f"pinned-region cache {path}: {os.path.getsize(path)} bytes, expected {size}")
                rid, buf = px.region(size)
                with open(path, "rb", buffering=0) as f:
                    got = f.readinto(memoryview(buf)[:size])
                if got != size:
                    raise RuntimeError(f"pinned-region cache {path}: short read {got} of {size}")
                n_cached += 1
            else:
                experts = sorted(e for (l2, e) in self.host if l2 == L)
                size = len(experts) * tp * (ng + nd)
                rid, buf = px.region(size)
                for i, e in enumerate(experts):
                    gu, dn = self.host.pop((L, e))
                    base = i * tp * (ng + nd)
                    ttnn.copy_host_to_device_tensor(gu, conv[0])
                    px.read(dg, rid, [base + d * ng for d in range(tp)])
                    ttnn.copy_host_to_device_tensor(dn, conv[1])
                    px.read(dd, rid, [base + tp * ng + d * nd for d in range(tp)])
                path = self._pin_path(L, experts)
                if path is not None and experts:
                    tmp = path + ".tmp"
                    with open(tmp, "wb", buffering=0) as f:
                        f.write(memoryview(buf)[:size])
                    os.replace(tmp, path)
            self.raw[L] = (rid, {e: i for i, e in enumerate(experts)}, ng, nd)
        return n_cached

    def note_shapes(self, mod):
        if self.shapes is None:
            ug, dn = mod.up_gate, mod.down
            gs, ds = [int(v) for v in ug.shape], [int(v) for v in dn.shape]
            self.shapes = ([1, 1, gs[2], gs[3]], [1, 1, ds[2], ds[3]], ug.dtype, dn.dtype, mod.device)

    def finalize(self, loader=None, stream=None):
        """After the model build, before any trace capture (HYB_PINX): every loaded device-format expert -> raw device
        bytes in a pinned per-layer host region (written to a staging tensor the regular way, read back raw), the host
        tensors dropped; destination views for the fetch / stream staging."""
        if not self.pinx_on or self.shapes is None:
            return
        if loader is not None:
            loader.submit(lambda: None).result()  # the queued load_host calls are done
        from .pin import Pinx

        gsh, dsh, gdt, ddt, mesh = self.shapes
        self.mesh = mesh
        px = self.px = Pinx(mesh)
        tp = px.tp
        kw = dict(layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        conv_gu = ttnn.from_torch(
            torch.zeros(gsh[0], gsh[1], gsh[2], tp * gsh[3]),
            dtype=gdt,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=-1),
            **kw,
        )
        conv_dn = ttnn.from_torch(
            torch.zeros(dsh[0], dsh[1], tp * dsh[2], dsh[3]),
            dtype=ddt,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=-2),
            **kw,
        )
        (dg, ng), (dd, nd) = px.dst(conv_gu), px.dst(conv_dn)
        t0 = time.perf_counter()
        layers = sorted({L for L, _ in self.host} | set(self.raw_files))
        n_cached = self._pin_layers(layers, (conv_gu, conv_dn))
        ttnn.deallocate(conv_gu)
        ttnn.deallocate(conv_dn)
        if self.replan_n:
            t1 = time.perf_counter()
            for L in layers:
                self.store[L] = (px.region(self.store_n * tp * (ng + nd))[0], {})
                self.nohost[L] = set()
            print(f"HYB_REPLAN: {self.store_n} store slots x {len(layers)} layers ({time.perf_counter() - t1:.1f} s)")
        if os.environ.get("HYB_PIN_UPLOADS", "1") == "1":
            from .pin import PinnedRows

            self.rows = PinnedRows(px)  # the per-layer crafted rows / host partial uploads (patch._put)
        if self.direct:
            for L, mod in self.mods.items():
                self.wdst[L] = (px.dst(mod.up_gate)[0], px.dst(mod.down)[0])
        elif self.dev is not None:
            self.dev["pdst"] = [(px.dst(g)[0], px.dst(d)[0]) for g, d in zip(self.dev["gu"], self.dev["dn"])]
        if stream is not None and stream.dev is not None:
            sg, sd = stream.dev["stage"]
            stream.dev["pdst"] = (px.dst(sg)[0], px.dst(sd)[0])
            # the whole staging cache, so a prefill upload can write slot j at byte offset j * one expert
            stream.dev["gdst"] = (px.dst(stream.dev["gu"])[0], px.dst(stream.dev["dn"])[0])
        print(
            f"HYB_PINX: {sum(len(v[1]) for v in self.raw.values())} experts of {len(layers)} layers pinned "
            f"({time.perf_counter() - t0:.1f} s)",
            flush=True,
        )

    def finalize_more(self, layers, loader=None):
        """finalize() for layers that became hybrid after it ran (elastic demotion, hybrid/elastic.py): their
        device-format experts pinned as finalize() does (a layer pinned by an earlier demotion keeps its region) and the
        direct-fetch destinations of their new weights."""
        if not self.pinx_on:
            return
        if self.px is None:
            raise RuntimeError("fetch.finalize_more before finalize()")
        if loader is not None:
            loader.submit(lambda: None).result()  # the queued load_host calls are done
        todo = [L for L in layers if L not in self.raw]
        if todo:
            gsh, dsh, gdt, ddt, mesh = self.shapes
            tp = self.px.tp
            kw = dict(layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            conv = (
                ttnn.from_torch(
                    torch.zeros(gsh[0], gsh[1], gsh[2], tp * gsh[3]),
                    dtype=gdt,
                    mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=-1),
                    **kw,
                ),
                ttnn.from_torch(
                    torch.zeros(dsh[0], dsh[1], tp * dsh[2], dsh[3]),
                    dtype=ddt,
                    mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=-2),
                    **kw,
                ),
            )
            self._pin_layers(todo, conv)
            ttnn.deallocate(conv[0])
            ttnn.deallocate(conv[1])
            if self.replan_n:
                ng, nd = self.raw[todo[0]][2], self.raw[todo[0]][3]
                for L in todo:
                    if L not in self.store:
                        self.store[L] = (self.px.region(self.store_n * tp * (ng + nd))[0], {})
                        self.nohost[L] = set()
        if self.direct:
            for L in layers:
                mod = self.mods[L]
                self.wdst[L] = (self.px.dst(mod.up_gate)[0], self.px.dst(mod.down)[0])

    def drop_layer(self, layer):
        """A hybrid layer that becomes full again (elastic promotion): its weights' fetch destinations and LRU state go
        (its pinned library region is kept for a later demotion). Call with no fetch in flight."""
        if any(p[0] == layer for p in self.pending):
            raise RuntimeError(f"fetch.drop_layer({layer}) with fetches in flight")
        for d in (self.wdst, self.mods, self.lru, self.free, self.stamp, self.__dict__.get("_static_n", {})):
            d.pop(layer, None)

    def has_expert(self, layer, e):
        return (
            (layer, e) in self.host
            or (layer in self.raw and e in self.raw[layer][1])
            or (layer in self.store and e in self.store[layer][1])
        )

    def raw_offsets(self, layer, e):
        """(region, gate|up offsets per die, down offsets per die) of a pinned expert (cold library or victim store)."""
        rid, idx, ng, nd = self.raw[layer]
        if e not in idx:
            rid, idx = self.store[layer]
        tp = self.px.tp
        base = idx[e] * tp * (ng + nd)
        return rid, [base + d * ng for d in range(tp)], [base + tp * ng + d * nd for d in range(tp)]

    def replan(self, layer, mass, slot_of, train_mass, host_row, st):
        """The per-request re-plan of one hybrid layer (see the module doc); ``mass`` [E] = the prefill's routed weight
        per expert, ``host_row`` [E] = the int4 library row (-1: none). Slot writes on CQ0, ahead of the next segments.
        Returns the number of swaps."""
        S = self._static_n[layer]
        rid_s, sidx = self.store[layer]
        nohost = self.nohost[layer]
        static = np.flatnonzero((slot_of >= 0) & (slot_of < S))
        lru = self.lru[layer]
        cand = [
            int(e)
            for e in np.argsort(-mass, kind="stable")
            if mass[e] > 0 and not (0 <= slot_of[e] < S) and self.has_expert(layer, int(e))
        ][: self.replan_n]
        vict = [int(e) for e in static[np.argsort(train_mass[static], kind="stable")] if mass[e] == 0]
        cold = self.raw[layer][1]
        n = 0
        dg, dd = self.wdst[layer]
        ng, nd = self.raw[layer][2], self.raw[layer][3]
        tp = self.px.tp
        t0 = time.perf_counter()
        for c in cand:
            v = next((x for x in vict if x in cold or x in sidx or len(sidx) < self.store_n), None)
            if v is None:
                break
            vict.remove(v)
            s = int(slot_of[v])
            if v not in cold and v not in sidx:  # its device bytes into the store before the slot is overwritten
                i = len(sidx)
                base = i * tp * (ng + nd)
                self.px.read_at(dg, rid_s, [base + d * ng for d in range(tp)], s * ng, ng)
                self.px.read_at(dd, rid_s, [base + tp * ng + d * nd for d in range(tp)], s * nd, nd)
                sidx[v] = i
                st["replan_reads"] = st.get("replan_reads", 0) + 1
            slot_of[v] = -1
            if host_row[v] < 0:
                nohost.add(v)
            rid, og, od = self.raw_offsets(layer, c)
            self.px.write_at(dg, rid, og, s * ng, ng)
            self.px.write_at(dd, rid, od, s * nd, nd)
            if c in lru:  # it leaves its LRU slot for the static one
                self.free[layer].append(lru.pop(c))
            slot_of[c] = s
            nohost.discard(c)
            n += 1
        st["replan_swaps"] = st.get("replan_swaps", 0) + n
        st["t_replan"] = st.get("t_replan", 0.0) + time.perf_counter() - t0
        return n

    def write_expert(self, layer, e, dst_pair, host_pair):
        """One expert into a staging pair: zero-copy from the pinned region, else the regular host -> device copy."""
        if self.px is not None and layer in self.raw:
            rid, og, od = self.raw_offsets(layer, e)
            self.px.write(dst_pair[0], rid, og)
            self.px.write(dst_pair[1], rid, od)
            return
        gu, dn = self.host[(layer, e)]
        ttnn.copy_host_to_device_tensor(gu, host_pair[0])
        ttnn.copy_host_to_device_tensor(dn, host_pair[1])

    def release(self):
        if self.px is not None:
            self.px.release()
            self.px = None

    def init_layer(self, mod):
        """Per fused layer, before any capture: the [C, 1, *, *] views of its expert weights, and the model-wide
        staging on the first call."""
        ug, dn = mod.up_gate, mod.down
        gs, ds = [int(v) for v in ug.shape], [int(v) for v in dn.shape]
        if self.direct:
            self.mods[mod.layer_num] = mod
            return
        mod._fetch_views = (ttnn.reshape(ug, [gs[1], 1, gs[2], gs[3]]), ttnn.reshape(dn, [ds[1], 1, ds[2], ds[3]]))
        if self.dev is not None:
            return
        mesh = self.mesh = mod.device
        kw = dict(layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        g_sh, d_sh = ttnn.ShardTensorToMesh(mesh, dim=-1), ttnn.ShardTensorToMesh(mesh, dim=-2)

        def stage(shape, mapper, dt):
            return ttnn.from_torch(torch.zeros(*shape), dtype=dt, mesh_mapper=mapper, **kw)

        tp = len(mesh.get_device_ids())  # shapes above are per device; the staging is built from the full tensor
        gsh, dsh = [1, 1, gs[2], tp * gs[3]], [1, 1, tp * ds[2], ds[3]]
        gdt, ddt = ug.dtype, dn.dtype
        rep = ttnn.ReplicateTensorToMesh(mesh)
        self.dev = {
            "gu": [stage(gsh, g_sh, gdt) for _ in range(self.F)],
            "dn": [stage(dsh, d_sh, ddt) for _ in range(self.F)],
            "zero_gu": stage(gsh, g_sh, gdt),
            "zero_dn": stage(dsh, d_sh, ddt),
            "pt": ttnn.from_torch(
                torch.zeros(self.F, 1, dtype=torch.int32),
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=mesh,
                mesh_mapper=rep,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            ),
            "pt_sink": ttnn.from_torch(
                torch.full((1, 1), gs[1] - 1, dtype=torch.int32),
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=mesh,
                mesh_mapper=rep,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            ),
            "C": gs[1],
        }

    def commit(self, slot_of):
        """Async fetch, before a pass's replay (outside any capture): CQ0 waits for every write CQ1 has queued, then the
        experts written since the last commit become resident (slot_of: HYB.slot_of, layer -> [E] slots)."""
        if self.verify_layer is not None:
            self._verify(slot_of)
        if not self.pending:
            return
        if self.bgwrite:
            self.px.flush()  # every background copy enqueued on CQ1 before the event that covers them
        ev = ttnn.record_event(self.mesh, cq_id=1)
        ttnn.wait_for_event(0, ev)
        for layer, e, s in self.pending:
            if self.lru[layer].get(e) == s:  # not evicted again before the commit
                slot_of[layer][e] = s
        self.pending = []

    def _verify(self, slot_of):
        """HYB_FETCH_VERIFY=<layer> (debug): every resident LRU expert of that layer read back from its device slot and
        compared with its pinned bytes (gate|up and down, every die); mismatches are counted and printed."""
        L = self.verify_layer
        if L not in self.wdst or L not in self.raw:
            return
        self._vn = getattr(self, "_vn", 0) + 1
        if self._vn % 8:
            return
        import numpy as np

        ttnn.synchronize_device(self.mesh)
        px, tp = self.px, self.px.tp
        rid, idx, ng, nd = self.raw[L]
        if not hasattr(self, "_vreg"):
            self._vreg = px.region(tp * (ng + nd))
        vrid, vbuf = self._vreg
        dg, dd = self.wdst[L]
        bad = checked = 0
        for e, s in list(self.lru[L].items()):
            if slot_of[L][e] != s:
                continue  # pending (not yet resident)
            px.read_at(dg, vrid, [d * ng for d in range(tp)], s * ng, ng)
            px.read_at(dd, vrid, [tp * ng + d * nd for d in range(tp)], s * nd, nd)
            got = np.frombuffer(vbuf, dtype=np.uint8, count=tp * (ng + nd))
            r2, og, od = self.raw_offsets(L, e)
            want = self._region_bytes(r2)
            ok = all(np.array_equal(got[d * ng : (d + 1) * ng], want[og[d] : og[d] + ng]) for d in range(tp)) and all(
                np.array_equal(got[tp * ng + d * nd : tp * ng + (d + 1) * nd], want[od[d] : od[d] + nd])
                for d in range(tp)
            )
            checked += 1
            bad += 0 if ok else 1
            if not ok and bad <= 3:  # what the slot holds instead
                parts = []
                for name, base_got, off_want, n in (("gu", 0, og, ng), ("dn", tp * ng, od, nd)):
                    for d in range(tp):
                        g = got[base_got + d * n : base_got + (d + 1) * n]
                        if np.array_equal(g, want[off_want[d] : off_want[d] + n]):
                            continue
                        who = [
                            e2
                            for e2, i2 in idx.items()
                            if np.array_equal(
                                g,
                                want[
                                    i2 * tp * (ng + nd)
                                    + (0 if name == "gu" else tp * ng)
                                    + d * n : i2 * tp * (ng + nd)
                                    + (0 if name == "gu" else tp * ng)
                                    + (d + 1) * n
                                ],
                            )
                        ]
                        nz = int(np.count_nonzero(g))
                        parts.append(
                            f"{name}{d}:{'zero' if nz == 0 else ('expert ' + str(who) if who else f'other ({int((g != want[off_want[d]:off_want[d] + n]).sum())} bytes differ)')}"
                        )
                pend = [p for p in self.pending if p[0] == L]
                print(f"  slot {s} expert {e}: {' '.join(parts)}; pending {pend[:4]}", flush=True)
        print(
            f"HYB_FETCH_VERIFY layer {L} commit {self._vn}: {checked} resident LRU experts checked, {bad} mismatched",
            flush=True,
        )

    def _verify_now(self, L, e, s, cq):
        import numpy as np

        ttnn.synchronize_device(self.mesh)  # both queues drained: the write has executed
        px, tp = self.px, self.px.tp
        rid, idx, ng, nd = self.raw[L]
        if not hasattr(self, "_vreg"):
            self._vreg = px.region(tp * (ng + nd))
        vrid, vbuf = self._vreg
        dg, dd = self.wdst[L]
        px.read_at(dg, vrid, [d * ng for d in range(tp)], s * ng, ng)
        px.read_at(dd, vrid, [tp * ng + d * nd for d in range(tp)], s * nd, nd)
        got = np.frombuffer(vbuf, dtype=np.uint8, count=tp * (ng + nd))
        r2, og, od = self.raw_offsets(L, e)
        want = self._region_bytes(r2)
        gu_ok = [bool(np.array_equal(got[d * ng : (d + 1) * ng], want[og[d] : og[d] + ng])) for d in range(tp)]
        dn_ok = [
            bool(np.array_equal(got[tp * ng + d * nd : tp * ng + (d + 1) * nd], want[od[d] : od[d] + nd]))
            for d in range(tp)
        ]
        self._wn = getattr(self, "_wn", 0) + 1
        if self._wn <= 12 or not (all(gu_ok) and all(dn_ok)):
            print(
                f"HYB_FETCH_VERIFY write: layer {L} expert {e} -> slot {s} on CQ{cq}: gate|up ok {gu_ok} down ok {dn_ok}",
                flush=True,
            )

    def _region_bytes(self, rid):
        pass

        return self.px.by_rid[rid]

    # -- device side (traced) -----------------------------------------------------------------------------------------
    def device_writes(self, mod):
        """The staged experts into their slots, then the sink back to zero (traced with the layer's experts)."""
        if self.direct:
            return  # the host step wrote the fetched experts into their slots
        d = self.dev
        gv, dv = mod._fetch_views
        for j in range(self.F):
            ttnn.experimental.paged_fill_cache(gv, d["gu"][j], d["pt"], batch_idx=j)
            ttnn.experimental.paged_fill_cache(dv, d["dn"][j], d["pt"], batch_idx=j)
        ttnn.experimental.paged_fill_cache(gv, d["zero_gu"], d["pt_sink"], batch_idx=0)
        ttnn.experimental.paged_fill_cache(dv, d["zero_dn"], d["pt_sink"], batch_idx=0)

    # -- host side (between trace segments) ---------------------------------------------------------------------------
    def choose(self, layer, ids, slot_of):
        """After routing a layer's step (ids [m, k]): LRU hits refreshed, the fetch set chosen, slot_of updated in place
        (fetched experts now resident at their LRU slots). Returns (fetched ids, their slots) for stage()."""
        if self.policy == "dyn":
            return self.choose_dyn(layer, ids, slot_of)
        self.step += 1
        lru, free, stamp = self.lru[layer], self.free[layer], self.stamp[layer]
        picked = np.unique(ids)
        for e in picked:
            if e in lru:
                lru.move_to_end(e)
        miss = [int(e) for e in picked if slot_of[e] < 0]
        nh = self.nohost.get(layer)
        must = [e for e in miss if e in nh] if nh else []  # no host int4 row: always fetched
        if must:
            miss = [e for e in miss if e not in nh]
        x = self.f * len(miss)
        nf = min(int(x) + int(self.rng.random() < x - int(x)), self.F, len(miss))
        fetched, slots = [], []
        if nf or must:
            miss.sort(key=lambda e: (-stamp[e], e))  # most recently active first
            miss = must + miss
            nf += len(must)
            protect = set(int(e) for e in picked)
            for e in miss[:nf]:
                if free:
                    s = free.pop()
                else:
                    victim = next((v for v in lru if v not in protect), None)
                    if victim is None:
                        if e in must:
                            raise RuntimeError(
                                f"layer {layer}: expert {e} has no host copy and every LRU slot holds a pick of this "
                                f"step ({len(lru)} slots, picks {sorted(protect)})"
                            )
                        break
                    s = lru.pop(victim)
                    slot_of[victim] = -1
                lru[e] = s
                if self.async_on:
                    self.pending.append((layer, e, s))  # resident from the next pass on (commit)
                else:
                    slot_of[e] = s
                fetched.append(e)
                slots.append(s)
        stamp[picked] = self.step
        return fetched, slots

    def choose_dyn(self, layer, ids, slot_of):
        """choose() under HYB_FETCH_POLICY=dyn: the layer's share of the per-pass budget, candidates by use score, each
        admitted only into a free slot or over a resident it outscores (by the margin); the evicted resident goes back
        to the host (slot_of -1). Same contract as choose()."""
        self.step += 1
        lru, free, stamp = self.lru[layer], self.free[layer], self.stamp[layer]
        E = len(slot_of)
        sc = self.score.get(layer)
        if sc is None:
            sc = self.score[layer] = np.zeros(E, dtype=np.float32)
        sc *= self.decay
        picked = np.unique(ids)
        np.add.at(sc, ids.reshape(-1), 1.0)
        for e in picked:
            if e in lru:
                lru.move_to_end(e)
        miss = [int(e) for e in picked if slot_of[e] < 0]
        nh = self.nohost.get(layer)
        must = [e for e in miss if e in nh] if nh else []
        if must:
            miss = [e for e in miss if e not in nh]
        ema = self.miss_ema[layer] = 0.9 * self.miss_ema.get(layer, float(len(miss))) + 0.1 * len(miss)
        total = sum(self.miss_ema.values()) or 1.0
        share = self.budget * ema / total
        nf = min(int(share) + int(self.rng.random() < share - int(share)), self.F, len(miss))
        fetched, slots = [], []
        if nf or must:
            miss.sort(key=lambda e: (-sc[e], -stamp[e], e))
            protect = set(int(e) for e in picked)
            for e in must + miss[:nf]:
                if free:
                    s = free.pop()
                else:
                    cands = [v for v in lru if v not in protect]
                    if not cands:
                        if e in must:
                            raise RuntimeError(f"layer {layer}: expert {e} has no host copy and no evictable LRU slot")
                        break
                    victim = min(cands, key=lambda v: sc[v])
                    if e not in must and sc[e] <= sc[victim] * (1.0 + self.margin):
                        continue  # not worth the eviction
                    s = lru.pop(victim)
                    slot_of[victim] = -1
                lru[e] = s
                if self.async_on:
                    self.pending.append((layer, e, s))
                else:
                    slot_of[e] = s
                fetched.append(e)
                slots.append(s)
        stamp[picked] = self.step
        return fetched, slots

    def stage(self, layer, fetched, slots, st):
        """The fetched experts into the staging buffers and their slots into the page table (host -> device copies on
        CQ0, ahead of the segment that runs device_writes)."""
        t0 = time.perf_counter()
        if self.direct:
            if layer not in self.wdst:
                raise RuntimeError(f"layer {layer}: direct fetch before finalize() (no pinned weight destinations)")
            dg, dd = self.wdst[layer]
            ng, nd = self.raw[layer][2], self.raw[layer][3]
            for e, s in zip(fetched, slots):
                rid, og, od = self.raw_offsets(layer, e)
                w0 = time.perf_counter()
                cq = 1 if self.async_on else 0
                if self.bgwrite and cq == 1:  # HYB_PINX_BGWRITE: the copies run on pinx's worker thread
                    self.px.write_async(dg, rid, og, s * ng, ng, cq_id=1)
                    self.px.write_async(dd, rid, od, s * nd, nd, cq_id=1)
                else:
                    self.px.write_at(dg, rid, og, s * ng, ng, cq_id=cq)
                    self.px.write_at(dd, rid, od, s * nd, nd, cq_id=cq)
                st["t_fetch_w"] = st.get("t_fetch_w", 0.0) + time.perf_counter() - w0
                if layer == self.verify_layer:  # debug: read the slot straight back once the write has run
                    self._verify_now(layer, e, s, cq)
            st["fetched"] += len(fetched)
            st["t_fetch"] += time.perf_counter() - t0
            return
        d = self.dev
        for j, (e, s) in enumerate(zip(fetched, slots)):
            self.write_expert(layer, e, d["pdst"][j] if "pdst" in d else None, (d["gu"][j], d["dn"][j]))
        rows = tuple(slots) + (d["C"] - 1,) * (self.F - len(slots))  # unused rows: the sink (reset right after)
        st["fetched"] += len(fetched)
        if rows == d.get("pt_last"):  # the model-wide table already holds these slots (e.g. no fetch twice in a row)
            st["t_fetch"] += time.perf_counter() - t0
            return
        d["pt_last"] = rows
        pt = torch.tensor(rows, dtype=torch.int32).reshape(self.F, 1)
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(
                pt, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh)
            ),
            d["pt"],
        )
        st["t_fetch"] += time.perf_counter() - t0
