# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Prefill: device-format copies of the experts that have none, converted from the host int4 library on the fly.

The device-format copies (HYB_DEV_EXPERTS) cover only HYB_PIN_TOP missable experts per layer (pinned host RAM); a
prefill's picks of the others ("cold") ran on the CPU, which for long prompts was most of the prefill time. Here a
layer's cold experts are converted from the resident int4 library to bfp4 device images (hostk.to_bfp4, the
build_dev_experts.py layout) on worker threads one hybrid layer ahead, into one bank of a two-bank pinned ring, and
registered with the fetch cache (FetchCache.cold) for the layer's prefill steps, so the prefill stream (stream.py)
copies them into its staging and computes them on the device like any other missed expert. A bank is overwritten only
after the CQ1 copies that read it have completed (an event per bank). The registration is cleared at the end of the
prefill: decode never fetches from the ring.

Numerics: int4 (group 32) -> bfp4 is a second quantization (gate|up / down weight error vs the FP8 checkpoint
12.1% / 15.8% against 9.5% / 12.0% for the FP8 -> bfp4 copies; the CPU path's int4 alone is smaller).

env: HYB_PF_COLD (1 = on), HYB_PF_COLD_MIN (prompt tokens from which it runs, 1024), HYB_PF_COLD_THREADS (8),
HYB_PF_COLD_SIG (significant bits kept before the bfp4 packing, 24 = no pre-rounding).
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import ttnn

from . import hostk


class ColdConvert:
    def __init__(self, hyb):
        self.hyb = hyb
        self.on = os.environ.get("HYB_PF_COLD", "1") == "1"
        self.min_tokens = int(os.environ.get("HYB_PF_COLD_MIN", "1024"))
        self.sig = int(os.environ.get("HYB_PF_COLD_SIG", "24"))
        self.nthreads = int(os.environ.get("HYB_PF_COLD_THREADS", "8"))
        self.pool = None
        self.active = False
        self.banks = {}  # bank -> (region id, numpy view, capacity in experts)
        self.jobs = {}  # layer -> (bank, experts, futures)
        self.bank_layer = {}  # bank -> the layer converted into it
        self.bank_ev = {}  # bank -> CQ1 event after the latest copies that read it
        self.cur = None

    # ---------------------------------------------------------------- per prefill
    def begin(self, tokens):
        fetch, stream = self.hyb.fetch, self.hyb.stream
        self._clear()
        self.active = bool(
            self.on
            and tokens >= self.min_tokens
            and stream.on
            and stream.dev is not None
            and fetch.px is not None
            and fetch.raw
            and self.hyb.lib_dir
            and self.hyb.lib_resident
        )
        self.cur = None

    def end(self):
        for _, _, futs in self.jobs.values():
            for f in futs:
                f.result()
        self._clear()
        self.active = False

    def _clear(self):
        self.hyb.fetch.cold.clear()
        self.jobs.clear()
        self.bank_layer.clear()

    # ---------------------------------------------------------------- per layer step
    def step(self, layer):
        """Before a prefill step of `layer`: on its first step, its conversions are waited for and registered, and the
        next hybrid layer's start."""
        if not self.active or layer == self.cur:
            return
        self.cur = layer
        layers = self.hyb.stream.layers
        if layer not in layers:
            return
        self._ready(layer)
        i = layers.index(layer)
        if i + 1 < len(layers):
            self._start(layers[i + 1])

    def mark(self, layer, mesh):
        """After a prefill step's copies are enqueued: the CQ1 event the bank's next reuse waits for."""
        if self.active and layer in self.jobs:
            self.bank_ev[self.jobs[layer][0]] = ttnn.record_event(mesh, cq_id=1)

    # ---------------------------------------------------------------- internals
    def _cold_ids(self, layer):
        hyb, fetch = self.hyb, self.hyb.fetch
        lib = hyb.lib_layer(layer)
        pinned = fetch.raw[layer][1] if layer in fetch.raw else {}
        miss = np.flatnonzero(hyb.slot_of[layer] < 0)
        return [(int(e), int(lib["row"][e])) for e in miss if int(e) not in pinned and int(lib["row"][e]) >= 0]

    def _geometry(self, layer):
        fetch = self.hyb.fetch
        L0 = layer if layer in fetch.raw else next(iter(fetch.raw))
        _, _, ng, nd = fetch.raw[L0]
        return fetch.px.tp, ng, nd

    def _bank(self, b, need):
        cur = self.banks.get(b)
        if cur is not None and cur[2] >= need:
            return cur
        tp, ng, nd = self._geometry(next(iter(self.hyb.fetch.raw)))
        cap = max(need, max((len(self._cold_ids(L)) for L in self.hyb.stream.layers), default=need))
        rid, buf = self.hyb.fetch.px.region(cap * tp * (ng + nd))  # pinned once per bank, kept for the process
        self.banks[b] = (rid, buf, cap)
        return self.banks[b]

    def _start(self, layer):
        if layer in self.jobs:
            return
        b = self.hyb.stream.layers.index(layer) % 2
        prev = self.bank_layer.get(b)
        if prev is not None:  # the bank's previous layer: its conversions done, its copies completed on CQ1
            for f in self.jobs.pop(prev)[2]:
                f.result()
            self.hyb.fetch.cold.pop(prev, None)
            ev = self.bank_ev.pop(b, None)
            if ev is not None:
                ttnn.event_synchronize(ev)
        ids = self._cold_ids(layer)
        tp, ng, nd = self._geometry(layer)
        rid, buf, _ = self._bank(b, len(ids))
        per = tp * (ng + nd)
        lib = self.hyb.lib_layer(layer)
        if self.pool is None:
            self.pool = ThreadPoolExecutor(self.nthreads, thread_name_prefix="hyb-cold")
        base = buf.ctypes.data

        def conv(i, row):
            n = hostk.to_bfp4(lib, row, tp, base + i * per, sig=self.sig)
            if n != per:
                raise RuntimeError(f"layer {layer}: converted expert has {n} bytes, the device copy {per}")

        futs = [self.pool.submit(conv, i, row) for i, (_, row) in enumerate(ids)]
        self.jobs[layer] = (b, [e for e, _ in ids], futs)
        self.bank_layer[b] = layer
        self.hyb.st["pf_cold_conv"] = self.hyb.st.get("pf_cold_conv", 0) + len(ids)

    def _ready(self, layer):
        if layer not in self.jobs:
            self._start(layer)
        b, experts, futs = self.jobs[layer]
        t0 = time.perf_counter()
        for f in futs:
            f.result()
        self.hyb.st["t_cold_wait"] = self.hyb.st.get("t_cold_wait", 0.0) + time.perf_counter() - t0
        self.hyb.fetch.cold[layer] = (self.banks[b][0], {e: i for i, e in enumerate(experts)})
