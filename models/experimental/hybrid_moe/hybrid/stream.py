# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Prefill streaming of missed experts (FreeToken's prefill idea, scaled to what a chunk actually routes to): in an
eager prefill MoE, every missed expert with at least HYB_STREAM_MIN tokens is copied host -> device into a model-wide
staging weight tensor ([1, NS, H, W] bfp4, the port's fused layout) and computed there by a second routed-expert call
over the staging; the CPU keeps only the experts with a few tokens. A missed expert costs one ~8 MB copy (~0.4 ms)
instead of a CPU GEMM over its tokens, which for long prompts is the difference between seconds and minutes.

The device-format copies come from HYB_DEV_EXPERTS (build_dev_experts.py; hybrid/fetch.py loads them).
env: HYB_STREAM (1 = on when HYB_DEV_EXPERTS is set), HYB_STREAM_MIN (tokens, default 4), HYB_STREAM_SLOTS (NS, 128),
HYB_STREAM_ROUNDS (staging rounds per prefill layer step, default 1: the experts past rounds x (NS - 1) stay on the CPU),
HYB_STREAM_CQ1 (1: the copies go on a second command queue, so the first round's copy runs during the resident experts),
HYB_STREAM_MERGE (1: the first round shares one routed call with the resident experts, patch._merged_round).
"""

from __future__ import annotations

import os
import time

import torch

import ttnn


class PrefillStream:
    def __init__(self, fetch):
        self.fetch = fetch  # the device-format host copies (FetchCache.host)
        self.on = bool(fetch.dir) and os.environ.get("HYB_STREAM", "1") == "1"
        self.min_tokens = int(os.environ.get("HYB_STREAM_MIN", "4"))
        self.NS = int(os.environ.get("HYB_STREAM_SLOTS", "128"))
        self.rounds = int(os.environ.get("HYB_STREAM_ROUNDS", "1"))
        self.cq1 = os.environ.get("HYB_STREAM_CQ1", "0") == "1"
        self.merge = os.environ.get("HYB_STREAM_MERGE", "1") == "1"
        # HYB_PF_PREFETCH_MIN: from this prompt length on (0: never) the staging is two banks, alternating over the
        # hybrid layers, and a layer's first prefill step copies the NEXT hybrid layer's whole non-resident set into
        # its bank on CQ1 (prefetch): at these lengths a layer uses nearly every non-resident expert and its compute
        # outlasts the copies, which then run off the critical path. Needs HYB_STREAM_CQ1.
        self.prefetch_min = int(os.environ.get("HYB_PF_PREFETCH_MIN", "0"))
        self.content = [None] * (self.NS - 1)  # (layer, expert) held by each staging slot (planned; copies in order)
        self.where = {}  # (layer, expert) -> slot
        self.reused = 0  # planned experts already in the staging (no copy)
        self.dev = None
        self.layers = []  # the hybrid layers (stream.init), ascending: the forward order of a prefill pass
        self.banked = False  # this prefill prefetches (begin_prefill)
        self.cur = None  # the layer of the latest prefill step
        self.last_read = {}  # bank -> CQ0 event after the latest step that read the bank

    def init(self, mod):
        """Model-wide staging, on the first fused layer (before any capture); per layer, its routed-call attributes."""
        if self.dev is None:
            mesh = mod.device
            tp = len(mesh.get_device_ids())
            gs, ds = [int(v) for v in mod.up_gate.shape], [int(v) for v in mod.down.shape]  # per device
            kw = dict(layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)

            def zeros(shape, dim, dt):
                return ttnn.from_torch(
                    torch.zeros(*shape), dtype=dt, mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=dim), **kw
                )

            gu = zeros([1, self.NS, gs[2], tp * gs[3]], -1, mod.up_gate.dtype)
            dn = zeros([1, self.NS, tp * ds[2], ds[3]], -2, mod.down.dtype)
            rep = ttnn.ReplicateTensorToMesh(mesh)
            self.dev = {
                "gu": gu,
                "dn": dn,
                "views": (ttnn.reshape(gu, [self.NS, 1, gs[2], gs[3]]), ttnn.reshape(dn, [self.NS, 1, ds[2], ds[3]])),
                "stage": (
                    zeros([1, 1, gs[2], tp * gs[3]], -1, mod.up_gate.dtype),
                    zeros([1, 1, tp * ds[2], ds[3]], -2, mod.down.dtype),
                ),
                "pt": ttnn.from_torch(
                    torch.zeros(1, 1, dtype=torch.int32),
                    dtype=ttnn.int32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    device=mesh,
                    mesh_mapper=rep,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                ),
                "ones": ttnn.from_torch(
                    torch.ones(1, 1, 1, self.NS, dtype=torch.bfloat16),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    device=mesh,
                    mesh_mapper=rep,
                ),
                "mesh": mesh,
            }
        d = self.dev
        if mod.layer_num not in self.layers:
            self.layers = sorted(self.layers + [mod.layer_num])
        mod._stream_attrs = {
            "up_gate": d["gu"],
            "down": d["dn"],
            "ones_mask": d["ones"],
            "num_experts": self.NS,
            "_pairs_masks": {},
            "_dispatch": None,
        }

    def plan(self, layer, idh, miss, other=None):
        """Prefill picks idh [rows, k] (torch long) and their miss mask -> rounds [(round mask [rows, k], staging index
        per pick, the round's experts in staging order)]: the missed experts with >= min_tokens tokens that have host
        copies, most tokens first, NS - 1 per round and at most ``rounds`` rounds. Each round is one upload into the
        staging and one routed call over it. ``other`` = the staging index of every pick outside the round (default
        NS - 1, a slot kept all-zero; the sorted dispatch takes a sentinel id >= NS, which it does not dispatch)."""
        ids = idh[miss]
        if ids.numel() == 0:
            return None
        e, n = torch.unique(ids, return_counts=True)
        keep = [
            (int(c), int(x))
            for x, c in zip(e.tolist(), n.tolist())
            if (c >= self.min_tokens or x in self.fetch.nohost.get(layer, ())) and self.fetch.has_expert(layer, x)
        ]
        if not keep:
            return None
        keep.sort(reverse=True)
        per = len(self._slots(layer))  # slot NS - 1 stays empty: the dummy index of the picks outside a round
        other = self.NS - 1 if other is None else int(other)
        out = []
        for r in range(min(self.rounds, -(-len(keep) // per))):
            experts = [x for _, x in keep[r * per : (r + 1) * per]]
            slots, writes = self._assign(layer, experts)
            pos = torch.full((int(idh.max()) + 1,), -1, dtype=torch.long)
            pos[torch.tensor(experts)] = torch.tensor(slots)
            sidx = pos[idh.clamp(0, pos.numel() - 1)]
            rmask = miss & (sidx >= 0)
            out.append((rmask, torch.where(rmask, sidx, torch.full_like(sidx, other)), writes))
        return out

    def _slots(self, layer):
        """The staging slots a layer may use: all (NS - 1: the last stays the all-zero dummy), or its bank."""
        if not self.banked:
            return range(self.NS - 1)
        half = self.NS // 2
        b = self.layers.index(layer) % 2
        return range(b * half, min((b + 1) * half, self.NS - 1))

    def begin_prefill(self, tokens):
        """A prefill of `tokens` positions starts: bank the staging and prefetch when it is long enough."""
        self.banked = bool(self.prefetch_min) and tokens >= self.prefetch_min and self.can_prefetch()
        self.cur = None
        self.last_read = {}

    def can_prefetch(self):
        return self.cq1 and self.dev is not None and self.dev.get("gdst") is not None and self.fetch.px is not None

    def step(self, layer, nonresident, st):
        """Before a prefill step of `layer`: on its first step, the next hybrid layer's prefetch (banked prefills)."""
        if layer == self.cur:
            return
        self.cur = layer
        if not self.banked:
            return
        i = self.layers.index(layer)
        if i + 1 >= len(self.layers):
            return
        nxt = self.layers[i + 1]
        if nxt not in self.fetch.raw:
            return
        experts = [e for e in nonresident(nxt) if self.fetch.has_expert(nxt, e)][: len(self._slots(nxt))]
        _, writes = self._assign(nxt, experts)
        if not writes:
            return
        ev = self.last_read.get(self.layers.index(nxt) % 2)
        if ev is not None:  # the bank's previous reader (the hybrid layer two back) has run
            ttnn.wait_for_event(1, ev)
        self.upload(nxt, writes, st, cq_id=1)
        st["prefetched"] = st.get("prefetched", 0) + len(writes)

    def mark_read(self, layer, mesh):
        """After a prefill step's routed calls over the staging are enqueued (banked prefills)."""
        if self.banked:
            self.last_read[self.layers.index(layer) % 2] = ttnn.record_event(mesh, cq_id=0)

    def _assign(self, layer, experts):
        """Staging slots for one round's experts -> (slot per expert, [(slot, expert)] to copy). An expert of this layer
        already in the staging keeps its slot (no copy: consecutive calls of one layer, the layer-major prefill, or a
        prefetch). The others take the slots no expert of the round holds: empty first, then other layers', then this
        layer's. A banked prefill keeps each layer in its bank (_slots)."""
        content, where = self.content, self.where
        slots = [where.get((layer, e), -1) for e in experts]
        allowed = self._slots(layer)
        for i, s in enumerate(slots):
            if s >= 0 and s not in allowed:  # staged outside this layer's bank (an earlier, unbanked prefill): re-copy
                content[s] = None
                del where[(layer, experts[i])]
                slots[i] = -1
        held = {s for s in slots if s >= 0}
        free = sorted(
            (s for s in allowed if s not in held),
            key=lambda s: 0 if content[s] is None else (1 if content[s][0] != layer else 2),
        )
        writes = []
        fi = 0
        for i, e in enumerate(experts):
            if slots[i] >= 0:
                continue
            s = free[fi]
            fi += 1
            if content[s] is not None:
                del where[content[s]]
            content[s] = (layer, e)
            where[(layer, e)] = s
            slots[i] = s
            writes.append((s, e))
        self.reused += len(experts) - len(writes)
        return slots, writes

    def can_cq1(self, layer):
        """The upload of this layer's experts can go on CQ1: pinned zero-copy writes (the other path runs device ops)."""
        return self.cq1 and self.dev.get("gdst") is not None and self.fetch.px is not None and layer in self.fetch.raw

    def upload(self, layer, writes, st, cq_id=0):
        """A round's copies [(slot, expert)] into the staging (plan / _assign). With the pinned library this is one
        zero-copy write per weight straight into the slot, on queue ``cq_id``. Otherwise each expert lands in the
        single-slot stage and a paged fill copies it into its slot (CQ0 only)."""
        t0 = time.perf_counter()
        d = self.dev
        fetch = self.fetch
        gdst = d.get("gdst")
        if gdst is not None and fetch.px is not None and layer in fetch.raw:
            dg, dd = gdst
            ng, nd = fetch.raw[layer][2], fetch.raw[layer][3]
            for j, e in writes:
                rid, og, od = fetch.raw_offsets(layer, e)
                fetch.px.write_at(dg, rid, og, j * ng, ng, cq_id=cq_id)
                fetch.px.write_at(dd, rid, od, j * nd, nd, cq_id=cq_id)
            st["t_stream"] += time.perf_counter() - t0
            st["streamed"] += len(writes)
            return
        if cq_id != 0:
            raise RuntimeError(f"layer {layer}: stream upload on CQ{cq_id} needs the pinned library (HYB_PINX)")
        gv, dv = d["views"]
        sg, sd = d["stage"]
        rep = ttnn.ReplicateTensorToMesh(d["mesh"])
        for j, e in writes:
            fetch.write_expert(layer, e, d.get("pdst"), (sg, sd))
            ttnn.copy_host_to_device_tensor(
                ttnn.from_torch(
                    torch.tensor([[j]], dtype=torch.int32),
                    dtype=ttnn.int32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    mesh_mapper=rep,
                ),
                d["pt"],
            )
            ttnn.experimental.paged_fill_cache(gv, sg, d["pt"], batch_idx=0)
            ttnn.experimental.paged_fill_cache(dv, sd, d["pt"], batch_idx=0)
        st["t_stream"] += time.perf_counter() - t0
        st["streamed"] += len(writes)
