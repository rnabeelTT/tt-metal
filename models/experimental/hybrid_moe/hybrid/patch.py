# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Runtime patches: C static expert slots (+ D dynamic slots) on device, host compute for missed experts.

Env (read at install):
  HYB_SLOTS        C, static expert slots per MoE layer (bfp4, the model's routed precision; default all E)
  HYB_DYN_SLOTS    D, dynamic expert slots per MoE layer (bfp8: ttnn.fill_cache cannot write bfp4); 0 = off
  HYB_PROMOTE      max promotions written per layer per decode step (default 2)
  HYB_FUSED        1 (default): decode through the fused MoE kernels (C slots incl. an all-zero sink slot)
  HYB_BOUNDARY     0 (default; 1 with HYB_FUSED): replicated residual + fabric decode boundary (production layout)
  HYB_SKIP         routing-weight threshold: missed experts below it are dropped (default 0: none)
  HYB_PLAN_MIN_SCORE  plan counts only picks with routing weight >= this (default HYB_SKIP; needs recorded scores)
  HYB_SINGLE_ROWS  1 (default): one decode row runs the 2-row fused kernels (accuracy, see _kernel_rows)
  HYB_FULL_LAYERS  comma list of layers that hold all E experts on the device and run the port's own MoE decode (no
                   host step, no trace split): the device memory the host-heavy layers buy back (sim_full_layers.py)
  HYB_DEVCRAFT     1: the fused kernels' crafted logits rows are built on the device (_dc_craft), the host reads the
                   router logits without blocking and routes / computes the misses while the device runs the experts
                   (one host round trip per layer on the device's critical path instead of two)
  HYB_OVERLAP      1 (default): traced decode splits after the resident experts, so the host experts overlap them
  HYB_PROFILE      1: synchronize before each readback to time the device segment wait separately (t_sync)
  HYB_PLAN         .npz with routing (routing.npz layout) -> static set = the C most-routed experts per layer
  HYB_LIB          int4 host expert library dir (build_int4_library.py); otherwise bf16 dequant from the checkpoint
  HYB_LIB_RESIDENT 1 (default): copy each layer's non-static int4 experts into RAM (during the build; with MTP, after the head)
  HYB_DECODE_ROWS  valid rows of a decode step (default 1: single user)
  HYB_THREADS      host torch threads (default 8)
  HYB_HOSTK        1 (default): one layer's missed experts in one call of the fused C kernel (hybrid/hostk; build it
                   on the host with `python -m hybrid.hostk`); 0: per-expert torch int4 matmuls
  HYB_HOSTK_THREADS threads of that kernel's pool (default 3: leaves DRAM bandwidth for the pinned uploads)
  HYB_PLAN_WEIGHT  count (default) | mass: rank the plan's experts by picks or by summed routing weight
  HYB_PORT         the patched autoport (default qwen_qwen3_5_122b_a10b; minimaxai_minimax_m2_7)
  HYB_LOWBIT       missed experts whose largest routing weight in the step is below this (and not skipped) read only
                   the top 2 bits of their int4 weights (hostk bit-plane layout: half the weight bytes, ~0.5 relative
                   output error instead of the 1.0 of skipping); 0 (default): every computed miss is full int4
"""

from __future__ import annotations

import gc
import math
import os
import sys
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

import ttnn

from . import hfpath, hostk

# The autoport the overlay patches (models/autoports/<HYB_PORT>/tt): the Qwen3.5-122B port (default, serving the 397B)
# or a port of the same lineage, e.g. minimaxai_minimax_m2_7 (sigmoid + bias router, no shared expert).
PORT = os.environ.get("HYB_PORT", "qwen_qwen3_5_122b_a10b")


def _port(mod):
    import importlib

    return importlib.import_module(f"models.autoports.{PORT}.tt.{mod}")


class HybridState:
    @property
    def sock(self):
        """Socket mode for the decode captures from now on (HYB_SOCKET; switchable at run time between captures). The
        fetch follows: asynchronous (CQ1 writes, resident at the next replay's commit) while it is on."""
        return self._sock

    @sock.setter
    def sock(self, on):
        if on and not self._sock_env:
            raise ValueError("HYB.sock needs HYB_SOCKET=1 at startup (the sockets are created in the eager warm-up)")
        self._sock = bool(on)
        if self.fetch.on:
            self.fetch.commit(self.slot_of)  # writes of the previous mode land first
            self.fetch.async_on = self._sock

    @property
    def hostk_threads(self):
        """Host-kernel threads in use (0 = the whole pool); settable at run time (hostk.set_active)."""
        return getattr(self, "_hostk_threads", 0)

    @hostk_threads.setter
    def hostk_threads(self, n):
        hostk.set_active(int(n))
        self._hostk_threads = int(n)

    def __init__(self):
        self.E = None
        self.C = int(os.environ.get("HYB_SLOTS", "0") or 0)
        self.D = int(os.environ.get("HYB_DYN_SLOTS", "0") or 0)
        # layers whose MoE holds all E experts on the device and runs the port's own decode (no host step)
        self.full_layers = {int(v) for v in os.environ.get("HYB_FULL_LAYERS", "").split(",") if v.strip()}
        self._full_build = False  # set while such a layer builds: its fused decode keeps the port's own router
        self.promote = int(os.environ.get("HYB_PROMOTE", "2"))
        self.profile = os.environ.get("HYB_PROFILE") == "1"
        self.overlap = os.environ.get("HYB_OVERLAP", "1") == "1"
        self.devcraft = os.environ.get("HYB_DEVCRAFT", "0") == "1"
        self.single_via_rows = os.environ.get("HYB_SINGLE_ROWS", "1") == "1"
        self.filler = float(os.environ.get("HYB_FILLER", "-1000"))  # crafted logit of unselected slots
        # HYB_CRAFT_SHIFT=1: each crafted row's slot logits (log routing weights) shifted so the row's largest is 0
        # before the bf16 upload: the kernels' top-k softmax is shift-invariant, and bf16 keeps ~8 bits of relative
        # precision, so log weights near 0 lose far less absolute precision than ln(0.1) = -2.3 (+-0.016: ~1.6% weight)
        self.craft_shift = os.environ.get("HYB_CRAFT_SHIFT", "0") == "1"
        self.rows_max = int(os.environ.get("QWEN35_MOE_FUSED_ROWS_MAX", "1") or 1)  # fused multi-row decode width
        self.fused = os.environ.get("HYB_FUSED", "1") == "1"
        self.skip = float(os.environ.get("HYB_SKIP", "0"))  # missed experts below this routing weight are not computed
        self.lowbit = float(os.environ.get("HYB_LOWBIT", "0"))  # computed misses below this weight: hi 2 bits only
        # static plan counts only picks with at least this routing weight (default HYB_SKIP: skipped misses are free)
        self.plan_min_score = float(
            os.environ.get("HYB_PLAN_MIN_SCORE", str(self.skip))
        )  # fused MoE decode kernels (moe_decode/fused.py)
        self.decode_rows = int(os.environ.get("HYB_DECODE_ROWS", "1"))
        # HYB_PLAN_WEIGHT: "count" (default: picks, thresholded by HYB_PLAN_MIN_SCORE) or "mass" (summed routed weight)
        self.plan_weight = os.environ.get("HYB_PLAN_WEIGHT", "count")
        if self.plan_weight not in ("count", "mass"):
            raise ValueError(f"HYB_PLAN_WEIGHT={self.plan_weight}: expected count or mass")
        self.plan_path = hfpath.resolve(os.environ.get("HYB_PLAN", ""))
        self.lib_dir = hfpath.resolve(os.environ.get("HYB_LIB", ""))
        self._bf16_cache = OrderedDict()  # (layer, expert) -> bf16 (gate|up, down), without an int4 library
        self.sd = None
        self.slot_of = {}  # layer -> np.int32[E] static slot (-1 = none)
        self.dyn_of = {}  # layer -> np.int32[E] dynamic slot (-1 = none)
        self.dyn_lru = {}  # layer -> OrderedDict(expert -> dyn slot), LRU first
        self.dyn_free = {}  # layer -> list of free dyn slots
        self.pending = {}  # (layer, e) -> Future of prepared host tensors
        self.resident = {}
        self._freq = None
        self.crafted_rm = {}  # fused hybrid: rows -> the model-wide crafted logits rows (row-major DRAM), _fused_step1
        self.lib = {}  # layer -> future of the layer's host library
        self.router_kind = "softmax"
        self.sel_bias = {}  # layer -> [E] selection bias (sigmoid_bias routers)
        self.lib_resident = os.environ.get("HYB_LIB_RESIDENT", "1") == "1"
        self.hostk = os.environ.get("HYB_HOSTK", "1") == "1"
        # Prefill chunks of at most this many rows run the host kernel in tiles of hostk.MAXR; longer ones go through
        # _hostk_packed (pairs sorted by expert, each expert's weights read about once). Default 16: a 64-token prompt's
        # 128-row bucket packed instead of tiled, 0.79 -> 0.72 s prefill (runs/m27/pf_pad).
        self.prefill_hostk_rows = int(os.environ.get("HYB_PREFILL_HOSTK_ROWS", "16"))
        self.pf_mask_pad = os.environ.get("HYB_PF_MASK_PAD", "1") == "1"  # _decide_core: skip the padding rows' misses
        # sorted prefill MoE calls: picks outside the call get a sentinel id (no dispatched row) instead of a
        # zero-weight row of a real slot
        self.prefill_sentinel = os.environ.get("HYB_PREFILL_SENTINEL", "1") == "1"
        self.merge_check = os.environ.get("HYB_MERGE_CHECK", "0") == "1"  # see _merged_round
        # HYB_PF_SPLIT=1: a prefill layer step computes its fewest-token missed experts on the CPU (packed int4 kernel,
        # concurrently with the device) while the CPU estimate stays below the remaining staging copies (_cpu_split).
        # Costs in ms (pcie_bench.py / hostk_bench.py on the QB2 host): a staging copy per expert, the CPU per expert
        # + per pair, and the fixed CPU-side cost of a step (activation readback, result upload and add). Off by default:
        # on the full model it gains nothing at 512-2K (runs/m27/split_full: the device waits on the CPU).
        self.pf_split = os.environ.get("HYB_PF_SPLIT", "0") == "1"
        self.pf_host_async = os.environ.get("HYB_PF_HOST_ASYNC", "1") == "1"  # prefill host experts on HYB.pool
        self.pf_copy_ms = float(os.environ.get("HYB_PF_COPY_MS", "0.28"))
        self.pf_cpu_ms = [float(v) for v in os.environ.get("HYB_PF_CPU_MS", "0.3,0.02").split(",")]
        self.pf_cpu_fixed_ms = [
            float(v) for v in os.environ.get("HYB_PF_CPU_FIXED_MS", "0.5,0.0005").split(",")
        ]  # + per row
        # a longer step is a full segment of a layer-major group, whose staging copies serve the group's later segments
        # (a per-step copy cost overstates them): its misses stay on the device
        self.pf_split_max_rows = int(os.environ.get("HYB_PF_SPLIT_MAX_ROWS", "2048"))
        # HYB_CROUTE (default 1 with hostk): the decode host step's routing and crafted rows in C (hostk_route / _craft)
        self.croute = self.hostk and os.environ.get("HYB_CROUTE", "1") == "1"
        # HYB_HOSTK_ASYNC (default 1 with hostk): the CPU experts of a decode step start on hostk's own C thread instead
        # of a Python thread-pool worker, whose GIL hand-offs stalled the main thread's device calls (bench_pinx.py)
        self.hostk_async = self.hostk and os.environ.get("HYB_HOSTK_ASYNC", "1") == "1"
        if self.hostk:
            hostk.load(int(os.environ.get("HYB_HOSTK_THREADS", "3")))  # 3: fixed-text A/B runs/m27/fb_thr, +2-3% over 4
            if os.environ.get("HYB_PREFETCH", "0") == "1":
                hostk.touch_init(int(os.environ.get("HYB_PF_THREADS", "2")))
        if self.lowbit > 0 and not self.hostk:
            raise ValueError(f"HYB_LOWBIT={self.lowbit} needs the fused host kernel (HYB_HOSTK=1)")
        # with an MTP head, the RAM copies of the library start only once the head is built (_install_mtp_head): the
        # head's 512 full-precision experts take ~44 GB of host RAM while they are repacked, on top of ~136 GB of copies
        self.defer_lib = self.lib_resident and int(os.environ.get("QWEN35_MTP", "0") or 0) > 0
        self.loader = ThreadPoolExecutor(max_workers=1)
        self.lock = threading.Lock()
        self.pool = ThreadPoolExecutor(max_workers=int(os.environ.get("HYB_PREP_WORKERS", "2")))
        from .fetch import FetchCache

        self._sock_env = os.environ.get("HYB_SOCKET", "0") == "1"  # sockets created (the eager warm-up) when set
        self._sock = False
        self.fetch = FetchCache()  # HYB_LRU: fetched experts in LRU slots (hybrid/fetch.py)
        if self._sock_env:
            self.sock = True
        self.cpu_pool = ThreadPoolExecutor(max_workers=1)  # HYB_LRU: the CPU experts run while the fetches copy
        from .stream import PrefillStream

        # HYB_EARLY_CPU (default 1): a decode host step starts the CPU experts before it builds and uploads the routing
        # row (fixed-text A/B, runs/m27/fb_early: CPU wait 6-11 -> 0.2-2.6 ms per pass, +4-5% tok/s)
        self.early_cpu = os.environ.get("HYB_EARLY_CPU", "1") == "1"
        self.np_prepare = os.environ.get("HYB_NP_PREPARE", "1") == "1"  # decode CPU-job arguments built with NumPy
        # HYB_SOCKET=1: a captured decode step stays ONE trace; each hybrid layer exchanges its host step's data through
        # host <-> device sockets inside it (_HybSockets) instead of being cut into segments with readbacks and pinned
        # uploads; fetched experts are written on CQ1 and resident from the next replay (FetchCache async)
        self.socks = None
        # HYB_SOCK_ROWS_FIRST (default 1): crafted rows sent before the activations are read and the CPU experts start
        # (runs/m27/fb_rows: short 48.5 -> 50.2 tok/s, 8K 35.9 -> 38.2)
        self.sock_rows_first = os.environ.get("HYB_SOCK_ROWS_FIRST", "1") == "1"
        # HYB_SOCK_LATE_CHOOSE (default 1): socket mode picks the layer's fetches after its rows are sent
        self.sock_late_choose = os.environ.get("HYB_SOCK_LATE_CHOOSE", "1") == "1"
        self.sock_late_stage = os.environ.get("HYB_SOCK_LATE_STAGE", "1") == "1"
        # HYB_PREFETCH (socket mode): at each hybrid layer the host predicts the NEXT hybrid layer's missed experts
        # (that layer's router on this layer's MoE input: sigmoid + bias top-HYB_PF_TOPK per row) and, once this
        # layer's partial is sent, reads their int4 weights on HYB_PF_THREADS threads (hostk_touch) while the device
        # runs the next attention: the next CPU experts then read L2 / L3 instead of DRAM. Cache contents only, no
        # numerics change. HYB_PF_MAXMB caps the bytes (L3 is 32 MiB).
        self.prefetch = os.environ.get("HYB_PREFETCH", "0") == "1"
        self.pf_topk = int(os.environ.get("HYB_PF_TOPK", "8"))
        self.pf_maxmb = float(os.environ.get("HYB_PF_MAXMB", "24"))
        self.gate_wT = {}  # layer -> [E, H] bf16 (uint16) router weight (HYB_PREFETCH)
        # HYB_ELASTIC_DEMOTE (hybrid/elastic.py): layers whose feed-forward can be rebuilt at runtime as full or hybrid;
        # their decoder-layer builder arguments are kept (ff_build), the switcher is set by DFlashServing.warm
        self.elastic_demote = {int(v) for v in os.environ.get("HYB_ELASTIC_DEMOTE", "").split(",") if v.strip()}
        self.ff_build = {}  # layer -> (decoder layer, mesh, args, cache path, tt_ccl)
        self.elastic = None
        self.pf_pred = {}  # layer -> [E] int32: 1 = predicted host expert (written by the prefetch threads)
        self.pf_next = None  # (next layer, MoE input [m, H] fp32) to predict + touch after this layer's partial
        self.pf_keep = None  # the arrays the running prefetch job reads
        self.pf_mass = {}  # HYB_REPLAN: layer -> [E] routed weight per expert over the latest prefill's real rows
        self.pf_valid = None  # real rows of the prefill segment the current layer call runs (bucket padding excluded)
        self.replan_tokens = 0  # prompt tokens of the prefill awaiting its re-plan (0: none pending)
        self.stream = PrefillStream(self.fetch)  # prefill: missed experts with enough tokens computed on the device
        self.record = None
        self.enabled = True
        self.reset_stats()

    def reset_stats(self):
        self.st = {
            "picks": 0,
            "hits": 0,
            "dyn_hits": 0,
            "steps": 0,
            "promoted": 0,
            "t_read": 0.0,
            "t_host": 0.0,
            "t_up": 0.0,
            "t_promote": 0.0,
            "prefill_rows": 0,
            "prefill_miss_pairs": 0,
            "t_sync": 0.0,
            "t_up_conv": 0.0,
            "t_lib_load": 0.0,
            "dec_picks": 0,
            "dec_hits": 0,
            "pf_act": 0,
            "pf_pred": 0,
            "pf_hit": 0,
            "pf_mb": 0.0,
            "t_lg2rows": 0.0,
            "t_pf": 0.0,
            "skipped": 0,
            "skipped_mass": 0.0,
            "lowbit": 0,
            "host_experts": 0,
            "t_route": 0.0,
            "t_craft": 0.0,
            "t_fetch": 0.0,
            "fetched": 0,
            "t_stream": 0.0,
            "streamed": 0,
            "t_wait": 0.0,  # host step 2 waiting for the CPU experts
            "t_exec": 0.0,  # trace segment enqueues (seg.py)
            "t_cb": 0.0,  # host callbacks between segments, wall (seg.py)
        }
        from .seg import SEG

        SEG.host_st = self.st
        # a replay of a socket-mode trace: the fetched experts written on CQ1 during the previous replay become
        # resident (CQ0 waits for them) before it is enqueued
        SEG.pre_replay = lambda: self.fetch.commit(self.slot_of) if self.fetch.on else None

    # -- static plan ----------------------------------------------------------------------------------------------
    def plan(self, layer):
        if layer in self.resident:
            return self.resident[layer]
        E, C = self.E, self.C
        if layer in self.full_layers:  # every expert on the device, in id order
            ids = list(range(E))
            self.resident[layer] = ids
            self.slot_of[layer] = np.arange(E, dtype=np.int32)
            self.dyn_of[layer] = np.full(E, -1, dtype=np.int32)
            self.dyn_lru[layer] = OrderedDict()
            self.dyn_free[layer] = []
            return ids
        if self._freq is None and self.plan_path and os.path.exists(self.plan_path):
            z = np.load(self.plan_path)
            layers = [int(v) for v in z["layers"]]
            f = np.zeros((max(layers) + 1, E))
            for k in z.files:
                if k.startswith("p"):
                    a = z[k]
                    # with scores (s<k>) and a threshold, count only the picks the host would compute on a miss
                    sk = "s" + k[1:]
                    if self.plan_weight == "mass" and sk in z.files:  # rank by routed weight (sim_m27.py's order)
                        w = z[sk].astype(np.float64)
                    elif self.plan_min_score > 0 and sk in z.files:
                        w = (z[sk] >= self.plan_min_score).astype(np.float64)
                    else:
                        w = np.ones(a.shape)
                    for li, L in enumerate(layers):
                        np.add.at(f[L], a[:, li, :].ravel(), w[:, li, :].ravel())
            self._freq = f
            # an explicit per-layer ranking (plan_from_routing.py <sim plan>: the order the host int4 library and the
            # device images were built from) wins over re-ranking the routing: the resident set then matches the
            # library's cold set by construction (Inkling: the shared experts 256, 257 pinned first)
            self._order = z["order"] if "order" in z.files else None
        n_real = C - 1 if self.fused else C  # fused: the last slot is the all-zero sink (expert id -1)
        if getattr(self, "_order", None) is not None and layer < self._order.shape[0]:
            rank = [int(v) for v in self._order[layer]]
        elif self._freq is not None and layer < self._freq.shape[0] and self._freq[layer].sum() > 0:
            rank = list(np.argsort(-self._freq[layer], kind="stable"))
        else:
            rank = list(range(E))
        if self.fetch.on:  # static top C - 1 - D, then the D LRU slots (empty: zero experts), then the sink
            if not self.fused:
                raise ValueError("HYB_LRU needs the fused hybrid (HYB_FUSED=1)")
            ids = self.fetch.plan_layer(layer, E, C, rank)
            n_real = C - 1 - self.fetch.D
        else:
            ids = [int(i) for i in rank[:n_real]] + ([-1] if self.fused else [])
        self.resident[layer] = ids
        so = np.full(E, -1, dtype=np.int32)
        so[ids[:n_real]] = np.arange(n_real, dtype=np.int32)
        self.slot_of[layer] = so
        self.dyn_of[layer] = np.full(E, -1, dtype=np.int32)
        self.dyn_lru[layer] = OrderedDict()
        self.dyn_free[layer] = list(range(self.D))
        return ids

    # -- host experts ---------------------------------------------------------------------------------------------
    @property
    def pf_order(self):
        """The hybrid layers in forward order (HYB_PREFETCH predicts each one's successor)."""
        o = getattr(self, "_pf_order", None)
        if o is None or len(o) != len(self.gate_wT):
            o = self._pf_order = sorted(self.gate_wT)
        return o

    def lib_layer(self, layer):
        f = self.lib.get(layer)
        if f is None:
            f = self.lib[layer] = self.loader.submit(self._load_lib, layer)
        return f.result()

    def lib_layer_async(self, layer):
        if layer not in self.lib:
            self.lib[layer] = self.loader.submit(self._load_lib, layer)

    def _load_lib(self, layer):
        """The int4 experts of one layer that can miss (every expert not in the static set), copied out of the mmap
        into process RAM: through the page cache the 225 GiB library is mostly evicted by the checkpoint load, and a
        miss then reads its expert from NVMe. row[e] = the expert's row in the copies (-1: static, never on the host).
        """
        if layer in self.full_layers or not self.lib_dir:  # every expert on the device, or no int4 library (HYB_LIB
            return {"row": np.full(self.E, -1, dtype=np.int64)}  # unset: misses in bf16 from the checkpoint): no rows
        t0 = time.perf_counter()
        d = torch.load(os.path.join(self.lib_dir, f"layer_{layer}.pt"), mmap=True, weights_only=True)
        # a library may hold a subset of the experts (build_int4_library.py HYB_LIB_COLD): file row of each expert id
        pos = np.full(self.E, -1, dtype=np.int64)
        lib_ids = d["experts"].numpy() if "experts" in d else np.arange(self.E)
        pos[lib_ids] = np.arange(len(lib_ids))
        if not self.lib_resident:
            if self.lowbit > 0:
                raise ValueError(f"HYB_LOWBIT={self.lowbit} needs the RAM-resident library (HYB_LIB_RESIDENT=1)")
            d["row"] = pos
            return d
        cold = np.flatnonzero(self.slot_of[layer] < 0) if layer in self.slot_of else np.arange(self.E)
        if (pos[cold] < 0).any():
            raise RuntimeError(
                f"layer {layer}: experts {cold[pos[cold] < 0][:8].tolist()}... can miss on the device but are not in "
                f"the host library {self.lib_dir} ({len(lib_ids)} experts)"
            )
        idx = torch.from_numpy(pos[cold])
        out = {k: d[k].index_select(0, idx).contiguous() for k in ("gu_p", "gu_sz", "dn_p", "dn_sz")}
        out["gs"] = d["gs"]
        if self.lowbit > 0:  # bit-plane layout, in place on the RAM copies (same bytes, read by hostk)
            hostk.planarize(out["gu_p"])
            hostk.planarize(out["dn_p"])
            out["planar"] = True
        row = np.full(self.E, -1, dtype=np.int64)
        row[cold] = np.arange(len(cold))
        out["row"] = row
        if os.environ.get("HYB_MLOCK", "1") == "1":  # resident, never swapped (also pre-faults every page)
            for k in ("gu_p", "gu_sz", "dn_p", "dn_sz"):
                _mlock(out[k])
        self.st["t_lib_load"] += time.perf_counter() - t0
        return out

    def host_prepare(self, layer, X, ids, sc, miss_mask):
        """host_compute for a worker thread: the routing tensors and the kernel arguments built here (caller's
        thread); returns (run, out): run() makes only the C call (no GIL), after which out [R, H] fp32 holds the
        result. None when the int4 kernel does not apply (the caller falls back to host_compute)."""
        R, H = X.shape
        if not (self.hostk and self.lib_dir and R <= hostk.MAXR) or not miss_mask.any():
            return None
        out = torch.zeros(R, H, dtype=torch.float32)
        lib = self.lib_layer(layer)
        ex = torch.unique(ids[miss_mask])
        rows = torch.from_numpy(lib["row"][ex.numpy()])
        if (rows < 0).any():
            raise RuntimeError(f"layer {layer}: experts {ex[rows < 0].tolist()} missed on the device but are static")
        W = ((ids[None] == ex[:, None, None]) & miss_mask[None]).float().mul_(sc.float()[None]).sum(-1)
        lo = (W.amax(1) < self.lowbit).to(torch.int8) if lib.get("planar") else None
        Xf = X.float().contiguous()
        rows, W = rows.contiguous(), W.contiguous()
        call = hostk.moe_prepared(lib, Xf, rows, W, out, lo=lo)
        self.st["host_experts"] += len(ex)
        self.st["lowbit"] += int(lo.sum()) if lo is not None else 0
        keep = (Xf, rows, W, lo, lib)  # alive until the call has run (run.keep)

        def run():
            if run.keep is None:
                raise RuntimeError("host_prepare: run() called twice")
            t0 = time.perf_counter()
            call()
            self.st["t_host"] += time.perf_counter() - t0
            run.keep = None  # the closure held the argument tensors (keep) until the call had run
            return out

        run.keep = keep

        def start():  # HYB_HOSTK_ASYNC: the same call on hostk's C launcher thread (no Python worker thread)
            run.keep = None

            def done(dt):
                self.st["t_host"] += dt

            return hostk.AsyncMoe(call, out, keep, on_done=done)

        run.start = start
        return run

    def host_prepare_np(self, layer, X, ids, sc, miss):
        """host_prepare for the decode step's NumPy routing (ids [R, k] int, sc [R, k] float, miss [R, k] bool; X [R, H]
        torch bf16): the same kernel arguments built with NumPy (the torch version's ~10 small ops per layer cost more
        than the arrays they touch). None when the int4 kernel does not apply."""
        R, H = X.shape
        if not (self.hostk and self.lib_dir and R <= hostk.MAXR) or not miss.any():
            return None
        lib = self.lib_layer(layer)
        if lib.get("planar"):
            return self.host_prepare(
                layer,
                X,
                torch.from_numpy(ids.astype(np.int64)),
                torch.from_numpy(sc.astype(np.float32)),
                torch.from_numpy(miss),
            )
        ex = np.unique(ids[miss])
        rows_np = lib["row"][ex]
        if (rows_np < 0).any():
            raise RuntimeError(f"layer {layer}: experts {ex[rows_np < 0].tolist()} missed on the device but are static")
        r_i, k_i = np.nonzero(miss)
        W = np.zeros((len(ex), R), dtype=np.float32)
        np.add.at(W, (np.searchsorted(ex, ids[r_i, k_i]), r_i), sc[r_i, k_i].astype(np.float32))
        out = torch.zeros(R, H, dtype=torch.float32)
        Xf = X.float().contiguous()
        rows, Wt = torch.from_numpy(np.ascontiguousarray(rows_np, dtype=np.int64)), torch.from_numpy(W)
        call = hostk.moe_prepared(lib, Xf, rows, Wt, out)
        self.st["host_experts"] += len(ex)
        keep = (Xf, rows, Wt, lib)

        def start():
            def done(dt):
                self.st["t_host"] += dt

            return hostk.AsyncMoe(call, out, keep, on_done=done)

        return start

    def _hostk_rows(self, layer, X, ids, sc, miss_mask, out):
        """One host-kernel call for X [R, H] with R <= hostk.MAXR, written into out. A tile with no misses is a no-op."""
        if not miss_mask.any():
            return
        if X.shape[0] > hostk.MAXR:
            raise RuntimeError(f"layer {layer}: host kernel tile has {X.shape[0]} rows, limit {hostk.MAXR}")
        lib = self.lib_layer(layer)
        ex = torch.unique(ids[miss_mask])
        rows = torch.from_numpy(lib["row"][ex.numpy()])
        if (rows < 0).any():
            raise RuntimeError(f"layer {layer}: experts {ex[rows < 0].tolist()} missed on the device but are static")
        W = ((ids[None] == ex[:, None, None]) & miss_mask[None]).float().mul_(sc.float()[None]).sum(-1)  # [E, R]
        lo = (W.amax(1) < self.lowbit).to(torch.int8) if lib.get("planar") else None
        hostk.moe(lib, X.float().contiguous(), rows.contiguous(), W.contiguous(), out, lo=lo)
        self.st["host_experts"] += len(ex)
        self.st["lowbit"] += int(lo.sum()) if lo is not None else 0

    def _hostk_packed(self, layer, X, ids, sc, miss_mask, out):
        """Prefill rows through the int4 kernel: the missed (row, expert) pairs sorted by expert and packed MAXR per
        call, each pair its own kernel row (block-diagonal weights), so an expert's weights are read about once."""
        lib = self.lib_layer(layer)
        r_all, j_all = miss_mask.nonzero(as_tuple=True)
        e_all = ids[r_all, j_all]
        order = torch.argsort(e_all, stable=True)
        r_all, e_all, w_all = r_all[order], e_all[order], sc[r_all, j_all][order].float()
        Xf = X.float()
        M = hostk.MAXR
        for c0 in range(0, len(e_all), M):
            r, e, w = r_all[c0 : c0 + M], e_all[c0 : c0 + M], w_all[c0 : c0 + M]
            ex, inv = torch.unique(e, return_inverse=True)
            rows = torch.from_numpy(lib["row"][ex.numpy()])
            if (rows < 0).any():
                raise RuntimeError(
                    f"layer {layer}: experts {ex[rows < 0].tolist()} missed on the device but are static"
                )
            W = torch.zeros(len(ex), len(r))
            W[inv, torch.arange(len(r))] = w
            lo = (W.amax(1) < self.lowbit).to(torch.int8) if lib.get("planar") else None
            oc = torch.zeros(len(r), X.shape[1], dtype=torch.float32)
            hostk.moe(lib, Xf[r].contiguous(), rows.contiguous(), W, oc, lo=lo)
            out.index_add_(0, r, oc)
            self.st["lowbit"] += int(lo.sum()) if lo is not None else 0
        self.st["host_experts"] += int(torch.unique(e_all).numel())

    def host_compute(self, layer, X, ids, sc, miss_mask):
        """X [R, H] bf16, ids/sc [R, k], miss_mask [R, k] -> [R, H] fp32: the missed experts' weighted sum."""
        R, H = X.shape
        out = torch.zeros(R, H, dtype=torch.float32)
        if not miss_mask.any():
            return out
        t0 = time.perf_counter()
        if self.hostk and self.lib_dir and self.lib_resident and R > max(hostk.MAXR, self.prefill_hostk_rows):
            self._hostk_packed(layer, X, ids, sc, miss_mask, out)
            self.st["t_host"] += time.perf_counter() - t0
            return out
        # Decode and MTP verify are one tile. A short prefill chunk is several tiles of the same kernel (rows are
        # independent). Longer prefill chunks use the torch GEMMs below.
        if self.hostk and self.lib_dir and R <= max(hostk.MAXR, self.prefill_hostk_rows):
            if R <= hostk.MAXR:
                self._hostk_rows(layer, X, ids, sc, miss_mask, out)
            else:
                for s in range(0, R, hostk.MAXR):
                    e = min(s + hostk.MAXR, R)
                    self._hostk_rows(layer, X[s:e], ids[s:e], sc[s:e], miss_mask[s:e], out[s:e])
            self.st["t_host"] += time.perf_counter() - t0
            return out
        for e in torch.unique(ids[miss_mask]).tolist():
            sel = (ids == e) & miss_mask
            rows = sel.any(dim=1).nonzero().flatten()
            w_r = (sc * sel).sum(dim=1)[rows]
            xr = X[rows].contiguous()
            if self.lib_dir:
                lib = self.lib_layer(layer)
                gs = int(lib["gs"])
                r = int(lib["row"][e])
                if r < 0:
                    raise RuntimeError(f"layer {layer}: expert {e} missed on the device but is in the static set")
                if lib.get("planar"):  # prefill rows: back in torch's pack for its GEMMs (low-weight rows: hi bits)
                    y = torch.empty(len(rows), X.shape[1], dtype=torch.float32)
                    lo_r = w_r < self.lowbit
                    self.st["lowbit"] += int(lo_r.sum())
                    for m in (~lo_r, lo_r):
                        if m.any():
                            y[m] = _int4_expert(xr[m], *_torch_pack(lib, r, hi_only=bool(m is lo_r)), gs).float()
                else:
                    y = _int4_expert(xr, lib["gu_p"][r], lib["gu_sz"][r], lib["dn_p"][r], lib["dn_sz"][r], gs)
            else:
                gu_dn = self._bf16_cache.get((layer, e))
                if gu_dn is None:  # no int4 library: bf16 from the checkpoint (FP8 dequant), recent experts kept
                    with self.lock:
                        gu_dn = self.sd.expert(layer, e)
                    self._bf16_cache[(layer, e)] = gu_dn
                    while len(self._bf16_cache) > 512:
                        self._bf16_cache.popitem(last=False)
                else:
                    self._bf16_cache.move_to_end((layer, e))
                gu, dn = gu_dn
                n = dn.shape[1]
                h = xr @ gu.T
                a = (torch.nn.functional.silu(h[:, :n].float()) * h[:, n:].float()).to(torch.bfloat16)
                y = a @ dn.T
            out.index_add_(0, rows, y.float() * w_r[:, None])
        self.st["t_host"] += time.perf_counter() - t0
        return out

    # -- dynamic slots: background preparation of device-format expert weights ------------------------------------
    def prep(self, moe_meta, layer, e):
        """Worker thread: expert e of `layer` -> host ttnn tensors (bfp8) in the MoE's packed TP layout."""
        tp, Nl, mesh = moe_meta
        with self.lock:
            gu, dn = self.sd.expert(layer, e)  # [2N, H] (gate; up), [H, N]
        N = dn.shape[1]
        gate_t, up_t = gu[:N].T, gu[N:].T  # [H, N]
        packed = torch.cat(
            [torch.cat([up_t[:, d * Nl : (d + 1) * Nl], gate_t[:, d * Nl : (d + 1) * Nl]], dim=-1) for d in range(tp)],
            dim=-1,
        ).reshape(1, 1, gu.shape[1], 2 * N)
        down_t = dn.T.contiguous().reshape(1, 1, N, dn.shape[0])
        h_gu = ttnn.from_torch(
            packed.contiguous(),
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=-1),
        )
        h_dn = ttnn.from_torch(
            down_t, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=-2)
        )
        return h_gu, h_dn


_MLOCK_WARNED = False


def _mlock(t):
    """mlock a CPU tensor's storage (the host expert library: the page cache of the checkpoint load otherwise pushes
    it to swap, and a swapped-in expert costs the host step milliseconds). The lock needs a memlock rlimit as large as
    the library (docker: --ulimit memlock=<bytes>); a container started without one (Docker's default is 8 MiB) gets
    the pages pre-faulted instead (MADV_POPULATE_READ), resident but swappable, with one warning. HYB_MLOCK=strict
    raises when the lock is refused."""
    import ctypes

    global _MLOCK_WARNED
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    n = t.numel() * t.element_size()
    if not n or libc.mlock(ctypes.c_void_p(t.data_ptr()), ctypes.c_size_t(n)) == 0:
        return
    err = ctypes.get_errno()
    if os.environ.get("HYB_MLOCK") == "strict":
        raise RuntimeError(f"mlock of {n} bytes failed (errno {err}): raise the memlock limit or set HYB_MLOCK=0")
    if not _MLOCK_WARNED:
        _MLOCK_WARNED = True
        sys.stderr.write(
            f"[hybrid] mlock refused (errno {err}, memlock limit too low): the host expert library is pre-faulted but "
            "not locked; under memory pressure the CPU experts can slow down (raise the memlock limit to lock it)\n"
        )
    page = 4096
    start = t.data_ptr() // page * page  # madvise needs a page-aligned start
    libc.madvise(ctypes.c_void_p(start), ctypes.c_size_t(n + t.data_ptr() - start), 22)  # MADV_POPULATE_READ


TILE_W = 32
HYB = HybridState()
_orig_forward = None  # Qwen35MoE.forward before install() (the MTP head's MoE keeps it)


def _int4_expert(x, gu_p, gu_sz, dn_p, dn_sz, gs):
    h = torch._weight_int4pack_mm_for_cpu(x, gu_p, gs, gu_sz)
    n = h.shape[1] // 2
    a = (torch.nn.functional.silu(h[:, :n].float()) * h[:, n:].float()).to(torch.bfloat16)
    return torch._weight_int4pack_mm_for_cpu(a, dn_p, gs, dn_sz)


def _torch_pack(lib, r, hi_only):
    """Library row r of a bit-plane library, in torch's int4 pack: (gu_p, gu_sz, dn_p, dn_sz). hi_only: the lo planes
    zeroed (q = 4h) and every zero point raised by 1.5 scales, i.e. hostk's hi-only q ~ 4h + 1.5."""
    out = []
    for p, sz in ((lib["gu_p"][r], lib["gu_sz"][r]), (lib["dn_p"][r], lib["dn_sz"][r])):
        p = p.clone()
        if hi_only:
            K = p.shape[-1] * 2
            p.view(-1, 2, K * 16)[:, 1] = 0
            s = sz.float()
            sz = torch.stack([s[..., 0], s[..., 1] + 1.5 * s[..., 0]], -1).to(sz.dtype)
        hostk.planarize(p, inverse=True)
        out += [p, sz]
    return out


def _die0(t):
    return ttnn.get_device_tensors(t)[0]


def _log_stats(replays):
    """HYB_LOG_EVERY: the host-step counters accumulated since the last log, per replayed segmented trace."""
    from .seg import SEG

    st, prev = HYB.st, getattr(HYB, "_log_prev", None) or {}
    n = SEG.log_every
    d = {k: st[k] - prev.get(k, 0) for k in st if isinstance(st[k], (int, float))}
    ms = {k[2:]: round(1e3 * v / n, 2) for k, v in d.items() if k.startswith("t_") and v}
    print(
        f"HYB_STATS replays={replays} per replay ms={ms} steps/replay={d.get('steps', 0) / n:.1f} "
        f"hit={d.get('dec_hits', 0) / max(1, d.get('dec_picks', 0)):.3f} skipped/replay={d.get('skipped', 0) / n:.1f} "
        f"host experts/replay={d.get('host_experts', 0) / n:.1f} lowbit/replay={d.get('lowbit', 0) / n:.1f}",
        flush=True,
    )
    HYB._log_prev = dict(st)


def install():
    from . import hostmem

    # before anything loads: does the host have the free RAM this profile's expert copies need (HYB_RAM_CHECK)
    hostmem.check_startup(
        HYB.lib_dir, hfpath.resolve(os.environ.get("HYB_DEV_EXPERTS", "")), HYB.full_layers, HYB.elastic_demote
    )
    torch.set_num_threads(int(os.environ.get("HYB_THREADS", "8")))
    from .seg import SEG

    SEG.on_log = _log_stats
    mc, moe_mod, od = _port("model_config"), _port("moe"), _port("optimized_decoder")

    def load_state_dict(self):
        HYB.E = int(self.num_experts)
        if not HYB.C or HYB.C > HYB.E:
            HYB.C = HYB.E
        if PORT == "qwen_qwen3_5_122b_a10b":
            from .state import HybridStateDict

            sd = HybridStateDict(
                self.CKPT_DIR, linear_q_dim=self.linear_q_dim, linear_k_dim=self.linear_k_dim, resident_fn=HYB.plan
            )
        else:
            from .state import hybrid_fused_state_dict

            sd = hybrid_fused_state_dict(PORT, self.CKPT_DIR, resident_fn=HYB.plan)
        HYB.sd = sd
        return sd

    mc.Qwen35MoeModelArgs.load_state_dict = load_state_dict
    if os.environ.get("HYB_WCACHE"):  # raw device images of the routed experts (hybrid/wcache.py)
        from .wcache import ExpertImageCache

        HYB.wcache = ExpertImageCache(os.environ["HYB_WCACHE"], resident_fn=lambda L: HYB.resident[L], tag=PORT)
        # on the module that DEFINES Qwen35MoE (its __init__ reads the global there): a port whose moe module only
        # re-exports the shared class (Inkling) would otherwise get the attribute on the wrong module, silently
        import sys as _sys

        _moe = _port("moe")
        _moe.EXPERT_WEIGHT_HOOK = HYB.wcache
        _sys.modules[_moe.Qwen35MoE.__module__].EXPERT_WEIGHT_HOOK = HYB.wcache

    orig_init = od.Qwen35MoEOptimized.__init__

    def init(self, mesh_device, args, state_dict, tensor_cache_path=None, tt_ccl=None, layer_num=None):
        E = int(args.num_experts)
        if layer_num is not None and layer_num >= int(getattr(args, "n_layers", 1 << 30)):
            # the MTP head's MoE (virtual layer n_layers): all experts on device, stock forward
            orig_init(self, mesh_device, args, state_dict, tensor_cache_path, tt_ccl=tt_ccl, layer_num=layer_num)
            self._hyb_on = False
            return
        if layer_num in HYB.full_layers:  # HYB_FULL_LAYERS: all experts on the device, the port's own forward
            HYB.plan(layer_num)
            HYB._full_build = True
            try:
                orig_init(self, mesh_device, args, state_dict, tensor_cache_path, tt_ccl=tt_ccl, layer_num=layer_num)
            finally:
                HYB._full_build = False
            self._hyb_on = False
            return
        HYB.plan(layer_num)
        if HYB.lib_dir and not HYB.defer_lib:
            HYB.lib_layer_async(layer_num)  # load this layer's host library while the rest of the model builds
        args.num_experts = HYB.C
        try:
            orig_init(self, mesh_device, args, state_dict, tensor_cache_path, tt_ccl=tt_ccl, layer_num=layer_num)
        finally:
            args.num_experts = E
        self.num_experts = HYB.C
        self.E_full = E
        HYB.router_kind = getattr(args, "router_kind", "softmax")
        if HYB.router_kind == "sigmoid_bias":  # the host's sigmoid + bias selection (_route_host) needs the bias
            HYB.sel_bias[layer_num] = state_dict["gate.e_score_correction_bias"].float().numpy()
            if HYB.prefetch:
                w = state_dict["gate.weight"].to(torch.bfloat16).contiguous()
                HYB.gate_wT[layer_num] = w.view(torch.int16).numpy().view(np.uint16)
        elif HYB.router_kind == "inkling":
            # Inkling: select 6 of the 256 routed on sigmoid + bias, then weight those 6 and the 2 shared experts
            # (experts 256, 257: always picked) by their normalised sigmoids (the port's _route, tt/optimized_decoder.py)
            HYB.sel_bias[layer_num] = state_dict["gate.e_score_correction_bias"].float().numpy()
            HYB.n_routed = int(args.n_routed_experts)
            HYB.top_k_routed = int(args.top_k_routed)
        elif HYB.router_kind != "softmax":
            raise ValueError(f"router_kind {HYB.router_kind!r}: the hybrid routes softmax, sigmoid_bias or inkling")
        if int(self.router_w.shape[-1]) != E:
            raise RuntimeError(f"router width {self.router_w.shape} != E {E}")
        self._hyb_fused = False
        if HYB.fused and HYB.C < E:
            if getattr(self, "_fused", None) is None:
                raise RuntimeError(
                    f"layer {layer_num}: HYB_FUSED=1 but the fused MoE decode was not built "
                    f"(QWEN35_MOE_FUSED_DECODE={os.environ.get('QWEN35_MOE_FUSED_DECODE')}, C={HYB.C})"
                )
            if HYB.D:
                raise RuntimeError("HYB_FUSED=1 does not support dynamic slots (HYB_DYN_SLOTS)")
            self._hyb_fused = True
        self._static_attrs = None
        self._dyn_attrs = None
        self._hyb_bufs = {}  # T -> persistent device buffers the host step writes (allocated before any capture)
        if HYB.fetch.dir and self._hyb_fused:  # device-format copies of the experts that can miss (fetch, stream)
            HYB.fetch.note_shapes(self)
            if HYB.fetch.on:
                HYB.fetch.init_layer(self)
            if HYB.stream.on:
                HYB.stream.init(self)
            missable = np.flatnonzero(HYB.slot_of[layer_num] < 0).tolist()
            self._fetch_load = HYB.loader.submit(HYB.fetch.load_host, layer_num, missable)
        if HYB.D and HYB.C < E:
            D, tp, H, N, Nl = HYB.D, self.num_devices, self.hidden, self.inter, self.inter_local
            rep = ttnn.ReplicateTensorToMesh(mesh_device)
            sh_last = ttnn.ShardTensorToMesh(mesh_device, dim=-1)
            sh_m2 = ttnn.ShardTensorToMesh(mesh_device, dim=-2)
            kw = {"layout": ttnn.TILE_LAYOUT, "device": mesh_device, "memory_config": ttnn.DRAM_MEMORY_CONFIG}
            gu = ttnn.from_torch(torch.zeros(1, D, H, tp * 2 * Nl), dtype=ttnn.bfloat8_b, mesh_mapper=sh_last, **kw)
            dn = ttnn.from_torch(torch.zeros(1, D, N, H), dtype=ttnn.bfloat8_b, mesh_mapper=sh_m2, **kw)
            ones = ttnn.from_torch(
                torch.ones(1, 1, 1, D, dtype=torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=mesh_device,
                mesh_mapper=rep,
            )
            gs, ds = [int(v) for v in gu.shape], [int(v) for v in dn.shape]
            self._dyn_views = (ttnn.reshape(gu, [gs[1], 1, gs[2], gs[3]]), ttnn.reshape(dn, [ds[1], 1, ds[2], ds[3]]))
            # promotion staging: allocated once, written with copy_host_to_device_tensor (no device allocation later)
            self._stage = (
                ttnn.from_torch(torch.zeros(1, 1, H, tp * 2 * Nl), dtype=ttnn.bfloat8_b, mesh_mapper=sh_last, **kw),
                ttnn.from_torch(torch.zeros(1, 1, N, H), dtype=ttnn.bfloat8_b, mesh_mapper=sh_m2, **kw),
            )
            self._dyn_attrs = {
                "up_gate": gu,
                "down": dn,
                "ones_mask": ones,
                "num_experts": D,
                "_pairs_masks": {},
                "_dispatch": None,
            }
            self._static_attrs = {k: getattr(self, k, None) for k in self._dyn_attrs}
            if self._static_attrs["_pairs_masks"] is None:
                self._static_attrs["_pairs_masks"] = {}
            self._meta = (tp, Nl, mesh_device)

    od.Qwen35MoEOptimized.__init__ = init
    if HYB.elastic_demote:
        orig_bff = od.Qwen35MoeOptimizedDecoderLayer._build_feed_forward

        def build_ff(self, mesh_device, args, state_dict, layer_num, tensor_cache_path, tt_ccl):
            if layer_num in HYB.elastic_demote:  # not the layer's state dict: it holds the loaded weights
                HYB.ff_build[layer_num] = (self, mesh_device, args, tensor_cache_path, tt_ccl)
            return orig_bff(self, mesh_device, args, state_dict, layer_num, tensor_cache_path, tt_ccl)

        od.Qwen35MoeOptimizedDecoderLayer._build_feed_forward = build_ff

    from models.demos.blackhole.qwen36.tt.moe_decode import fused as moe_fused

    f_orig = moe_fused.FusedMoEDecode.__init__

    def f_init(fself, mesh_device, args, *, num_experts, router_weight, shared_gate_weight, hidden, **kw):
        if HYB._full_build:  # a HYB_FULL_LAYERS layer: the port's fused decode as is
            return f_orig(
                fself,
                mesh_device,
                args,
                num_experts=num_experts,
                router_weight=router_weight,
                shared_gate_weight=shared_gate_weight,
                hidden=hidden,
                **kw,
            )
        if kw.get("router") == "sigb" and HYB.enabled and HYB.fused:
            # the hybrid feeds the kernels crafted rows of log-weights, whose softmax top-k reproduces the model's
            # normalized sigmoid weights (they sum to 1 over the top-k like softmax's); the host does the sigmoid + bias
            # selection on the E-wide logits of the router built below
            if router_weight is None:
                raise ValueError("HYB_FUSED with a sigmoid-bias router needs the port to pass router_weight")
            kw = {k: v for k, v in kw.items() if k not in ("selection_bias", "router_logits")}
            kw["router"] = "softmax"
        E_true = int(router_weight.shape[0])
        f_orig(
            fself,
            mesh_device,
            args,
            num_experts=num_experts,
            router_weight=router_weight[:num_experts],
            shared_gate_weight=shared_gate_weight,
            hidden=hidden,
            **kw,
        )
        if E_true == num_experts:
            return
        # the kernels rank the C slots; the router (run by _forward_fused, its logits read on the host) stays E-wide
        from models.demos.blackhole.qwen36.tt import tp_common as tpc

        ttnn.deallocate(fself.router_w)
        rw = torch.zeros(hidden, E_true + TILE_W)
        rw[:, :E_true] = router_weight.T.float()
        if shared_gate_weight is not None:  # no shared expert (MiniMax-M2): column E stays 0, never read
            rw[:, E_true] = shared_gate_weight.reshape(-1).float()
        fself.router_w = ttnn.from_torch(
            rw,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        fself.router_pc = tpc.create_matmul_1d_decode_progcfg(
            TILE_W,
            hidden,
            E_true + TILE_W,
            (E_true + TILE_W) // TILE_W,
            fp32_acc=True,
            grid_w=mesh_device.compute_with_storage_grid_size().x,
            in0_block_w=16 if (hidden // TILE_W) % 16 == 0 else None,
        )

    moe_fused.FusedMoEDecode.__init__ = f_init
    global _orig_forward
    _orig_forward = moe_mod.Qwen35MoE.forward
    moe_mod.Qwen35MoE.forward = _forward
    # prefill stays eager: the per-bucket prefill traces would capture the MoE host step (decode-only by design);
    # without a bucket trace the model's prefill falls back to its stock (eager) path
    gen_mod = _port("generator")

    gen_mod.Qwen35MoeGenerator._warm_prefill_traces = lambda self: None
    single = gen_mod.Qwen35MoeGenerator._prefill_single

    def prefill_single(self, tokens, *a, **kw):
        HYB.stream.begin_prefill(int(tokens.shape[1]))
        _begin_replan(int(tokens.shape[1]))
        return single(self, tokens, *a, **kw)

    gen_mod.Qwen35MoeGenerator._prefill_single = prefill_single
    model_cls = _port("model").Qwen35MoeModel
    all_logits = model_cls.prefill_all_logits

    def prefill_all_logits(self, token_ids, *a, **kw):
        HYB.stream.begin_prefill(int(token_ids.shape[-1]))
        _begin_replan(int(token_ids.shape[-1]))
        return all_logits(self, token_ids, *a, **kw)

    model_cls.prefill_all_logits = prefill_all_logits
    stage_inputs, bucket_layer = model_cls._stage_bucket_inputs, model_cls._bucket_layer

    def stage_bucket_inputs(self, bi, token_ids, page_table, actual_len, *a, **kw):
        bi["hyb_valid"] = int(actual_len)
        return stage_inputs(self, bi, token_ids, page_table, actual_len, *a, **kw)

    def bucket_layer_(self, layer, x, bi, chunk_start):
        HYB.pf_valid = bi.get("hyb_valid")
        try:
            return bucket_layer(self, layer, x, bi, chunk_start)
        finally:
            HYB.pf_valid = None

    model_cls._stage_bucket_inputs = stage_bucket_inputs
    model_cls._bucket_layer = bucket_layer_
    if HYB.defer_lib:
        _install_mtp_head()


def _install_mtp_head():
    """Start the deferred host-library loads (HybridState.defer_lib) once the MTP head is built and its state freed."""
    mtp_mod = _port("mtp")

    h_orig = mtp_mod.Qwen35MtpHead.__init__

    def h_init(self, *a, **kw):
        h_orig(self, *a, **kw)
        gc.collect()
        if HYB.lib_dir:
            for layer in sorted(HYB.slot_of):
                HYB.lib_layer_async(layer)

    mtp_mod.Qwen35MtpHead.__init__ = h_init


def _routed(self, x4, scores, ids, T, mode):
    if mode == "decode" or (mode is None and T <= self.GATHER_MAX_TOKENS):
        if T <= self.GATHER_MAX_TOKENS:
            return self._routed_gather(x4, scores, ids, T)
        return self._routed_masked(x4, scores, ids, T, chunk=32)
    return self._routed_masked(x4, scores, ids, T, chunk=self.PREFILL_CHUNK)


def _routed_with(self, attrs, x4, sc, ids, T, mode):
    """_routed over another expert tensor set (attrs: up_gate / down / ones_mask / num_experts and the per-set
    dispatch caches), the layer's own restored after."""
    own = {k: getattr(self, k, None) for k in attrs}
    for k, v in attrs.items():
        setattr(self, k, v)
    try:
        return _routed(self, x4, sc, ids, T, mode)
    finally:
        for k in ("_pairs_masks", "_dispatch"):
            if k in attrs:
                attrs[k] = getattr(self, k, None)
        for k, v in own.items():
            setattr(self, k, v)


def _routed_dyn(self, x4, d_sc, d_ids, T, mode):
    for k in ("_pairs_masks", "_dispatch"):
        self._static_attrs[k] = getattr(self, k, None)
    for k, v in self._dyn_attrs.items():
        setattr(self, k, v)
    try:
        return _routed(self, x4, d_sc, d_ids, T, mode)
    finally:
        for k in ("_pairs_masks", "_dispatch"):
            self._dyn_attrs[k] = getattr(self, k, None)
        for k, v in self._static_attrs.items():
            setattr(self, k, v)


def _finish(self, out, orig, partial):
    if self.num_devices > 1 and partial:
        return ttnn.reshape(out, orig) if list(out.shape) != orig else out
    if self.num_devices > 1:
        from models.tt_transformers.tt.ccl import tt_all_reduce

        out = tt_all_reduce(
            out,
            self.device,
            self.tt_ccl,
            cluster_axis=0,
            dim=3,
            topology=self.args.ccl_topology(),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
        )
        orig = orig[:-1] + [out.shape[-1]]
    if list(out.shape) != orig:
        out = ttnn.reshape(out, orig)
    return out


def _plus_shared(self, routed, x4, free=True):
    """routed + the shared expert; a port without one (``has_shared`` False, MiniMax-M2) returns routed itself."""
    if not getattr(self, "has_shared", True):
        return routed
    if not free:
        return ttnn.add(routed, self._shared(x4), memory_config=ttnn.DRAM_MEMORY_CONFIG)
    return _add(routed, self._shared(x4))


def _add(a, b):
    c = ttnn.add(a, b, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(a)
    ttnn.deallocate(b)
    return c


def _fused_bufs(self, m):
    """The fused hybrid's persistent host-written buffers for an m-row decode step (allocated in the eager warm-up,
    before any capture): the host partial as row-major rows [1, 1, m, H] (tiled on the device), and the model-wide
    crafted logits rows [1, 1, m, C + 32]."""
    from .seg import SEG

    B = self._hyb_bufs.get(("fused", m))
    if B is None:
        if SEG.capturing:
            raise RuntimeError(f"layer {self.layer_num}: no fused hybrid buffers for {m} rows; run an eager step first")
        H = int(self.hidden)
        B = {"host_rm": _rm_buffer(self.device, m, H), "shapes": {"T": m, "H": H}}
        if HYB._sock_env:  # the socket partial: chip 0 receives the full host partial, the other chips keep zeros
            B["host_sock"] = _rm_buffer(self.device, m, H)
            if HYB.socks is None:
                HYB.socks = _HybSockets(self.device, H)
        self._hyb_bufs[("fused", m)] = B
    mk = _kernel_rows(m)
    if mk not in HYB.crafted_rm:
        HYB.crafted_rm[mk] = _rm_buffer(self.device, mk, HYB.C + TILE_W)
    return B


class _HybSockets:
    """HYB_SOCKET: the host <-> device sockets of the fused hybrid's decode steps, shared by every hybrid layer (pages
    flow in layer order): from chip 0 the routing logits rows (fp32, E + 32 wide) and the activation rows (bf16, H);
    to every chip the crafted logits rows (bf16, C + 32); to chip 0 the host partial rows (bf16, H, undivided: the
    other chips' partial buffers stay zero and the layer's reduce sums the chips). One worker core each."""

    def __init__(self, mesh, H):
        ndev = mesh.get_num_devices()
        mcc = lambda d, x: ttnn.MeshCoreCoord(ttnn.MeshCoordinate(0, d), ttnn.CoreCoord(x, 0))  # noqa: E731
        # FIFO depths in pages: device -> host FIFOs live in pinned host memory (deep: the device never waits on the
        # host's reads); host -> device FIFOs in device L1 (the only buffer type they support), whose range the L1
        # allocator reserves on every core: 64-page FIFOs (~420 KB) left too little L1 for the prefill matmuls'
        # circular buffers (~400 KB free above them), 8 pages take ~52 KB
        d2h_depth, h2d_depth = 64, int(os.environ.get("HYB_SOCKET_H2D_DEPTH", "8"))
        h2d_buf = ttnn.BufferType.L1

        def d2h(core, page):
            s = ttnn.D2HSocket(mesh, mcc(0, core), d2h_depth * page)
            s.set_page_size(page)
            return s

        def h2d(dev, core, page):
            s = ttnn.H2DSocket(mesh, mcc(dev, core), h2d_buf, h2d_depth * page, ttnn.H2DMode.HOST_PUSH)
            s.set_page_size(page)
            return s

        E, C = HYB.E, HYB.C
        self.lg = d2h(0, (E + TILE_W) * 4)
        self.x = d2h(1, H * 2)
        self.rows = [h2d(d, 2, (C + TILE_W) * 2) for d in range(ndev)]
        self.part = h2d(0, 3, H * 2)
        self.lg_buf, self.x_buf = {}, {}
        self.warmed = set()  # row counts whose socket programs ran eagerly (_sock_warm)

    def read_lg(self, m):
        b = self.lg_buf.get(m)
        if b is None:
            b = self.lg_buf[m] = torch.empty(m, HYB.E + TILE_W, dtype=torch.float32)
        self.lg.read_tensor(b)
        return b.numpy()

    def read_x(self, m, H):
        b = self.x_buf.get(m)
        if b is None:
            b = self.x_buf[m] = torch.empty(m, H, dtype=torch.bfloat16)
        self.x.read_tensor(b)
        return b


def _sock_warm(self, logits, h_x, B, m, mk):
    """One eager round of the socket ops for an m-row step (their programs compiled before any capture: a capture
    cannot load new binaries), paired with the host side; the received rows are zeros and the buffers are reset."""
    S, H = HYB.socks, B["shapes"]["H"]
    ttnn.experimental.send_async_d2h(logits, S.lg)
    ttnn.experimental.send_async_d2h(h_x, S.x)
    S.read_lg(m)
    S.read_x(m, H)
    for r in S.rows:
        ttnn.experimental.recv_async_h2d(HYB.crafted_rm[mk], r)
        r.write_tensor(torch.zeros(mk, HYB.C + TILE_W, dtype=torch.bfloat16))
    ttnn.experimental.recv_async_h2d(B["host_sock"], S.part)
    S.part.write_tensor(torch.zeros(m, H, dtype=torch.bfloat16))
    ttnn.synchronize_device(self.device)
    S.warmed.add(m)


def _sock_step1(self, B, m):
    """HYB_SOCKET host step 1 of a layer (the device waits in recv_async_h2d for its crafted rows): the routing logits
    from the socket, routing and the fetch choice, then (HYB.sock_rows_first) the crafted rows to every chip before the
    activations are read and the CPU experts started (the device resumes sooner), or after; the fetch writes last."""
    t0 = time.perf_counter()
    S, L, H = HYB.socks, self.layer_num, B["shapes"]["H"]
    read0 = HYB.st["t_read"]
    lg = S.read_lg(m)
    HYB.st["t_read"] += time.perf_counter() - t0
    ids, sc, slot, res, miss = _route_host(self, lg, m)
    fetch = None
    # the fetch choice waits until the rows are out: the fetch is asynchronous here (resident from the next replay), so
    # it changes neither this step's slots nor its misses (victims are never this step's picks)
    late = HYB.fetch.on and HYB.fetch.async_on and HYB.sock_late_choose
    if HYB.fetch.on and not late:
        fetch = HYB.fetch.choose(L, ids, HYB.slot_of[L])
        slot = HYB.slot_of[L][ids]
        res = slot >= 0
        miss = miss & ~res
    if HYB.sock_rows_first:
        _sock_rows(self, S, L, lg, ids, sc, m)
    t1 = time.perf_counter()
    HYB.st["t_lg2rows"] += t1 - t0 - HYB.st["t_read"] + read0  # logits in hand -> rows sent (the device waits)
    xh = S.read_x(m, H)  # always consumed: the device sends every step
    HYB.st["t_read"] += time.perf_counter() - t1
    job = None
    if miss.any():
        start = HYB.host_prepare_np(L, xh, ids, sc, miss)
        if start is not None:
            job = start()
        else:
            j = _job(self, xh.clone(), ids, sc, miss)
            job = HYB.cpu_pool.submit(lambda j=j: HYB.host_compute(*j))
    B["job"] = job
    if HYB.prefetch:
        _pf_predict(L, xh, miss, ids, m)
    if not HYB.sock_rows_first:
        _sock_rows(self, S, L, lg, ids, sc, m)
    if late:
        fetch = HYB.fetch.choose(L, ids, HYB.slot_of[L])
    if fetch is not None:
        # after the rows (the device waits only for them): CQ1 writes, resident from the next replay (FetchCache.commit),
        # enqueued while the device runs this layer's experts and the next layer's attention; HYB_SOCK_LATE_STAGE: after
        # the partial instead (_sock_step2), so the copies' host-DRAM reads do not overlap the CPU experts' streaming
        if HYB.sock_late_stage and HYB.fetch.async_on:
            B["fetch_late"] = (L, fetch)
        else:
            HYB.fetch.stage(L, *fetch, HYB.st)
    HYB.st["t_s1"] = HYB.st.get("t_s1", 0.0) + time.perf_counter() - t0


def _pf_predict(L, xh, miss, ids, m):
    """HYB_PREFETCH: score this layer's prediction (made by the prefetch threads at the previous hybrid layer) against
    its actual misses, then queue the next hybrid layer's prediction from this layer's MoE input xh [m, H]
    (_pf_touch starts it once this layer's partial is sent)."""
    st = HYB.st
    pred = HYB.pf_pred.pop(L, None)
    if pred is not None:
        act = np.unique(ids[miss]) if miss.any() else np.zeros(0, dtype=np.int64)
        st["pf_act"] += len(act)
        st["pf_pred"] += int(pred.sum())
        st["pf_hit"] += int(pred[act].sum())
    order = HYB.pf_order
    i = order.index(L) if L in order else -1
    if 0 <= i < len(order) - 1:
        HYB.pf_next = (order[i + 1], np.ascontiguousarray(xh[:m].float().numpy()))


def _pf_touch(Ln, x):
    """Start the prefetch threads on layer Ln: its router on x, top-HYB_PF_TOPK per row, read the predicted host
    experts' int4 rows (the RAM library copies) up to HYB_PF_MAXMB."""
    t0 = time.perf_counter()
    lib = HYB.lib_layer(Ln)
    E = HYB.E
    rows = lib["row"]
    host = (rows >= 0) & (HYB.slot_of[Ln] < 0)
    eptr = np.zeros((E, 4), dtype=np.uint64)
    elen = np.zeros(4, dtype=np.int64)
    for q, key in enumerate(("gu_p", "gu_sz", "dn_p", "dn_sz")):
        t = lib[key]
        elen[q] = t[0].numel() * t.element_size()
        eptr[host, q] = np.uint64(t.data_ptr()) + rows[host].astype(np.uint64) * np.uint64(elen[q])
    pred = np.zeros(E, dtype=np.int32)
    hostk.touch_predict(x, HYB.gate_wT[Ln], HYB.sel_bias[Ln], HYB.pf_topk, eptr, elen, HYB.pf_maxmb * 1e6, pred)
    HYB.pf_keep = (x, eptr, elen, pred)
    HYB.pf_pred[Ln] = pred
    HYB.st["t_pf"] += time.perf_counter() - t0


def _sock_rows(self, S, L, lg, ids, sc, m):
    """The crafted logits rows of a routed socket-mode step to every chip."""
    t1 = time.perf_counter()
    E, C = HYB.E, HYB.C
    mk = _kernel_rows(m)
    row = np.empty((1, 1, m, C + TILE_W), dtype=np.float32)
    hostk.craft(ids, sc.astype(np.float32, copy=False), HYB.slot_of[L], C, HYB.filler, lg, E, row)
    if HYB.craft_shift:
        _shift_rows(row, C)
    r = torch.from_numpy(row.reshape(m, -1))
    if mk > m:
        r = r.repeat_interleave(mk // m, dim=0)
    r = r.to(torch.bfloat16).contiguous()
    for s_ in S.rows:
        s_.write_tensor(r)
    HYB.st["t_up"] += time.perf_counter() - t1


def _sock_step2(self, B, m):
    """HYB_SOCKET host step 2 of a layer: the CPU experts' partial (zeros without misses) to chip 0."""
    t0 = time.perf_counter()
    job = B["job"]
    B["job"] = None
    if job is None:
        host = torch.zeros(m, B["shapes"]["H"], dtype=torch.float32)
    else:
        t1 = time.perf_counter()
        host = job.result()
        HYB.st["t_wait"] += time.perf_counter() - t1
    t1 = time.perf_counter()
    HYB.socks.part.write_tensor(host.to(torch.bfloat16).contiguous())
    HYB.st["t_up"] += time.perf_counter() - t1
    if HYB.pf_next is not None:
        _pf_touch(*HYB.pf_next)
        HYB.pf_next = None
    late = B.pop("fetch_late", None)
    if late is not None:
        HYB.fetch.stage(late[0], *late[1], HYB.st)
    HYB.st["t_s2"] = HYB.st.get("t_s2", 0.0) + time.perf_counter() - t0


def _shift_rows(row, C):
    """HYB_CRAFT_SHIFT: the slot columns [0, C) of crafted rows [1, 1, m, C + 32] shifted so each row's largest is 0
    (in place; column C, the shared-expert gate logit, is not part of the slot softmax and stays)."""
    sl = row[..., :C]
    sl -= sl.max(axis=-1, keepdims=True)


def _kernel_rows(m):
    """Rows the fused kernels run for an m-row step: one decode row goes through the 2-row (union) kernels, which match
    the sparse-matmul path (PCC 0.998 per row, test_rows.py) where the one-row kernels on crafted logits did not
    (0.93-0.97); HYB_SINGLE_ROWS=0 keeps the one-row kernels."""
    return 2 if (m == 1 and HYB.single_via_rows) else m


def _rm_buffer(mesh_device, rows, width):
    return ttnn.from_torch(
        torch.zeros(1, 1, rows, width),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _route_host(self, lg, m, ids=None):
    """True top-k routing of m rows from the E-wide logits lg [m, >= E] (numpy): ids / scores [m, k], each pick's
    static slot, the resident mask and the host-computed misses (not skipped); updates the hit and skip counters.
    ids: the selection made on the device (HYB_DEVCRAFT, sigmoid_bias), used as given."""
    E, k, L = HYB.E, self.top_k, self.layer_num
    t0 = time.perf_counter()
    z = lg[:, :E]
    if HYB.router_kind == "sigmoid_bias" and ids is None and HYB.croute:  # one C call (hostk_route)
        ids, sc = hostk.route(lg, k, HYB.sel_bias[L])
    elif HYB.router_kind == "sigmoid_bias":  # MiniMax-M2 / GLM: select on sigmoid + bias, weight by normalized sigmoid
        sg = 1.0 / (1.0 + np.exp(-z))
        if ids is None:
            ids = np.argpartition(-(sg + HYB.sel_bias[L]), k, axis=1)[:, :k]
        elif (ids < 0).any() or (ids >= E).any():
            raise RuntimeError(f"layer {L}: device-selected expert ids out of range: {ids.tolist()}")
        sw = np.take_along_axis(sg, ids, 1)
        sc = sw / sw.sum(1, keepdims=True)
    elif HYB.router_kind == "inkling":
        sg = 1.0 / (1.0 + np.exp(-z))
        R, kr = HYB.n_routed, HYB.top_k_routed
        if ids is None:
            picks = np.argpartition(-(sg[:, :R] + HYB.sel_bias[L]), kr, axis=1)[:, :kr]
            ids = np.concatenate([picks, np.broadcast_to(np.arange(R, E), (z.shape[0], E - R))], axis=1)
        sw = np.take_along_axis(sg, ids, 1)
        sc = sw / sw.sum(1, keepdims=True)
    else:
        ids = np.argpartition(-z, k, axis=1)[:, :k]
        zs = np.take_along_axis(z, ids, 1)
        e = np.exp(zs - zs.max(1, keepdims=True))
        sc = e / e.sum(1, keepdims=True)  # softmax over the top-k logits == softmax over all E, top-k, renormalized
    slot = HYB.slot_of[L][ids]
    res = slot >= 0
    miss = ~res
    st = HYB.st
    nh = int(res.sum())
    st["picks"] += m * k
    st["hits"] += nh
    st["dec_picks"] += m * k
    st["dec_hits"] += nh
    st["steps"] += 1
    if HYB.record is not None:
        HYB.record.append((L, torch.from_numpy(ids[0].astype(np.int64)), torch.from_numpy(sc[0].astype(np.float32))))
    if HYB.skip > 0:
        drop = miss & (sc < HYB.skip)
        st["skipped"] += int(drop.sum())
        st["skipped_mass"] += float(sc[drop].sum())
        miss = miss & ~drop
    HYB.st["t_route"] += time.perf_counter() - t0
    return ids, sc, slot, res, miss


def _fused_step1(self, x, logits, B):
    t0 = time.perf_counter()
    _fused_step1_body(self, x, logits, B)
    HYB.st["t_s1"] = HYB.st.get("t_s1", 0.0) + time.perf_counter() - t0


def _fused_step1_body(self, x, logits, B):
    """Host step after the router (m decode rows): true routing from the E-wide logits rows, then the crafted logits
    rows the fused kernels rank over the C slots: ln(score) at each routed resident expert's slot, ln(missed mass) at
    the all-zero sink slot C-1, -1000 elsewhere, and the shared-expert gate logit at column C. The kernels' top-k +
    softmax over the selected logits then reproduce the true weights of the resident experts (the sink absorbs the
    missed mass). x / logits: row-major [1, 1, m, *] in DRAM."""
    E, C = HYB.E, HYB.C
    m = int(B["shapes"]["T"])
    if HYB.profile:
        t0 = time.perf_counter()
        ttnn.synchronize_device(self.device)
        HYB.st["t_sync"] += time.perf_counter() - t0
    t0 = time.perf_counter()
    lg = ttnn.to_torch(_die0(logits)).reshape(m, -1).float().numpy()
    HYB.st["t_read"] += time.perf_counter() - t0
    if HYB.replan_tokens:
        _run_replan()
    ids, sc, slot, res, miss = _route_host(self, lg, m)
    fetch = None
    if HYB.fetch.on:  # a share of the misses goes to LRU slots: resident from here on
        t1 = time.perf_counter()
        fetch = HYB.fetch.choose(self.layer_num, ids, HYB.slot_of[self.layer_num])
        HYB.st["t_choose"] = HYB.st.get("t_choose", 0.0) + time.perf_counter() - t1
        slot = HYB.slot_of[self.layer_num][ids]
        res = slot >= 0
        miss = miss & ~res
    if not HYB.early_cpu:
        _upload_row(self, B, lg, ids, sc, slot, res, m)
    if miss.any():
        t0 = time.perf_counter()
        xh = ttnn.to_torch(_die0(x)).reshape(m, -1).to(torch.bfloat16)
        HYB.st["t_read"] += time.perf_counter() - t0
        fast = fetch is not None and HYB.hostk_async and HYB.np_prepare
        B["job"] = (xh, ids, sc, miss) if fast else _job(self, xh, ids, sc, miss)
    else:
        B["job"] = None
    if fetch is not None and HYB.hostk_async and HYB.np_prepare and B["job"] is not None:  # NumPy fast path
        HYB.fetch.stage(self.layer_num, *fetch, HYB.st)
        xh, ids_np, sc_np, miss_np = B["job"]
        t1 = time.perf_counter()
        start = HYB.host_prepare_np(self.layer_num, xh, ids_np, sc_np, miss_np)
        if start is not None:
            B["job"] = start()
        else:
            j = _job(self, xh, ids_np, sc_np, miss_np)
            B["job"] = HYB.cpu_pool.submit(lambda j=j: HYB.host_compute(*j))
        HYB.st["t_prep"] = HYB.st.get("t_prep", 0.0) + time.perf_counter() - t1
    elif fetch is not None:
        # the fetch writes are enqueued BEFORE the CPU experts start: with the int4 kernel streaming host DRAM, each
        # pinned write's enqueue took ~390 us instead of ~15 us (bench_pinx.py)
        HYB.fetch.stage(self.layer_num, *fetch, HYB.st)
        if B["job"] is not None:  # the CPU experts run while the fetched ones copy (the worker only makes the C call)
            run = HYB.host_prepare(*B["job"])
            if run is not None and HYB.hostk_async:
                B["job"] = run.start()
            else:
                B["job"] = HYB.cpu_pool.submit(run if run is not None else (lambda j=B["job"]: HYB.host_compute(*j)))
    if HYB.early_cpu:  # HYB_EARLY_CPU: the routing row after the CPU experts started (the CPU job is the long pole)
        _upload_row(self, B, lg, ids, sc, slot, res, m)


def _upload_row(self, B, lg, ids, sc, slot, res, m):
    """The crafted logits rows of a routed step into the fused kernels' input (see _fused_step1)."""
    E, C = HYB.E, HYB.C
    t0 = time.perf_counter()
    row = np.empty((1, 1, m, C + TILE_W), dtype=np.float32)
    if HYB.croute and HYB.skip == 0:
        hostk.craft(ids, sc.astype(np.float32, copy=False), HYB.slot_of[self.layer_num], C, HYB.filler, lg, E, row)
    else:
        row.fill(HYB.filler)
        row[..., C:] = 0.0
        row[0, 0, :, C] = lg[:, E]  # shared-expert gate logit (router column E)
        r_i, k_i = np.nonzero(res)
        row[0, 0, r_i, slot[r_i, k_i]] = np.log(sc[r_i, k_i])
        miss_mass = (sc * ~res).sum(1)  # computed and skipped misses: their routing mass goes to the zero sink
        has = miss_mass > 1e-6
        row[0, 0, has, C - 1] = np.log(miss_mass[has])
    if HYB.craft_shift:
        _shift_rows(row, C)
    HYB.st["t_craft"] += time.perf_counter() - t0
    t0 = time.perf_counter()
    mk = _kernel_rows(m)
    r2 = torch.from_numpy(row.reshape(m, -1))

    def fill(v):  # fp32 -> bf16 straight into the pinned slice
        if mk > m:  # one row through the rows kernels: the extra rows repeat it (same experts, no extra weight reads)
            v.copy_(r2.repeat_interleave(mk // m, dim=0))
        else:
            v.copy_(r2)

    if not _put_fill(self, HYB.crafted_rm[mk], fill):
        if mk > m:
            row = np.repeat(row, mk, axis=2)
        _put(self, HYB.crafted_rm[mk], torch.from_numpy(row).to(torch.bfloat16), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
    HYB.st["t_up"] += time.perf_counter() - t0


def _begin_replan(tokens):
    """A prefill starts: its routing mass is collected for the re-plan of the decode that follows (HYB_REPLAN)."""
    if not HYB.fetch.replan_n:
        return
    HYB.pf_mass = {}
    HYB.replan_tokens = tokens if tokens >= HYB.fetch.replan_min else 0


def _run_replan():
    """At the first decode step after a long enough prefill: every hybrid layer's static slots re-planned from the
    prompt's routing (FetchCache.replan). Called from the first hybrid layer's host step of that decode pass, before
    its routing: the layers before it in the pass hold every expert (no swaps), and the slot writes queue on CQ0
    ahead of this layer's next segment."""
    from .seg import SEG

    if SEG.capturing:
        raise RuntimeError("HYB_REPLAN: a host step ran during a trace capture")
    t0 = time.perf_counter()
    n = 0
    for L, mass in HYB.pf_mass.items():
        n += HYB.fetch.replan(L, mass, HYB.slot_of[L], HYB._freq[L], HYB.lib_layer(L)["row"], HYB.st)
    print(
        f"HYB_REPLAN: {HYB.replan_tokens}-token prompt, {n} swaps over {len(HYB.pf_mass)} layers, "
        f"{HYB.st.get('replan_reads', 0)} store reads so far ({(time.perf_counter() - t0) * 1e3:.0f} ms)",
        flush=True,
    )
    HYB.replan_tokens = 0
    HYB.pf_mass = {}


def _job(self, xh, ids, sc, miss):
    return (
        self.layer_num,
        xh,
        torch.from_numpy(ids.astype(np.int64)),
        torch.from_numpy(sc.astype(np.float32)),
        torch.from_numpy(miss),
    )


def _dc_consts(self):
    """HYB_DEVCRAFT per-layer constants (replicated, DRAM tiles): P [E+32, C+32] one-hot expert -> static slot and
    router column E (shared-expert gate) -> column C; bias [1, C+32] = filler at the slots no expert maps to (not the
    sink); keep [1, E+32] lifts the non-expert columns above any top-k threshold; nonres [1, E] = 1 for experts with
    no slot; sinkhot [1, C+32] = 1 at the sink slot C-1."""
    from .seg import SEG

    c = getattr(self, "_dc", None)
    if c is not None:
        return c
    if SEG.capturing:
        raise RuntimeError(f"layer {self.layer_num}: HYB_DEVCRAFT constants not built; run an eager step first")
    E, C, L, W = HYB.E, HYB.C, self.layer_num, HYB.E + TILE_W
    so = HYB.slot_of[L]
    res = np.flatnonzero(so >= 0)
    if (so[res] >= C - 1).any():
        raise RuntimeError(f"layer {L}: a resident expert maps to slot >= C-1 (the sink): {so[res].max()}")
    P = torch.zeros(W, C + TILE_W)
    P[torch.from_numpy(res), torch.from_numpy(so[res].astype(np.int64))] = 1.0
    P[E, C] = 1.0
    mapped = torch.zeros(C, dtype=torch.bool)
    mapped[torch.from_numpy(so[res].astype(np.int64))] = True
    mapped[C - 1] = True
    bias = torch.zeros(1, C + TILE_W)
    bias[0, :C][~mapped] = HYB.filler
    keep = torch.zeros(1, W)
    keep[0, E:] = 1e4
    nonres = torch.from_numpy((so < 0).astype(np.float32)).reshape(1, E)
    sinkhot = torch.zeros(1, C + TILE_W)
    sinkhot[0, C - 1] = 1.0
    rep = ttnn.ReplicateTensorToMesh(self.device)
    sbias = None
    if HYB.router_kind == "sigmoid_bias":  # selection bias, FP32 (the selection compares sigmoids ~1e-3 apart)
        sbias = ttnn.from_torch(
            torch.from_numpy(HYB.sel_bias[L]).float().reshape(1, 1, 1, E),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=rep,
        )
    c = self._dc = {
        k: ttnn.from_torch(
            v.reshape(1, 1, *v.shape).to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=rep,
        )
        for k, v in (("P", P), ("bias", bias), ("keep", keep), ("nonres", nonres), ("sinkhot", sinkhot))
    }
    c["sbias"] = sbias
    return c


def _dc_craft(self, logits, m, mk):
    """HYB_DEVCRAFT: the crafted rows from the router logits tile [1, 1, m, E+32] on the device. Per row: threshold t
    = the k-th largest expert logit; resident experts at or above t keep their raw logit at their slot (the kernels'
    softmax is shift-invariant), every other slot gets the filler, the sink gets mx + ln(sum over the missed top-k of
    exp(z - mx)) (mx = the largest logit), column C the shared-expert gate logit. Returns the L1 tile [1, 1, mk, C+32].
    """
    c = _dc_consts(self)
    if HYB.router_kind == "sigmoid_bias":
        return _dc_craft_sigmoid(self, c, logits, m, mk)
    E, k = HYB.E, self.top_k
    dram = ttnn.DRAM_MEMORY_CONFIG
    zE = ttnn.slice(logits, [0, 0, 0, 0], [1, 1, m, E])
    vals, idx = ttnn.topk(zE, k=k, dim=-1, largest=True, sorted=True)
    ttnn.deallocate(idx)
    t = ttnn.slice(vals, [0, 0, 0, k - 1], [1, 1, m, k])
    mx = ttnn.slice(vals, [0, 0, 0, 0], [1, 1, m, 1])
    ttnn.deallocate(vals)
    # sink: the missed top-k mass
    condE = ttnn.ge(zE, t)
    zmE = ttnn.where(condE, zE, HYB.filler)
    ttnn.deallocate(condE)
    ttnn.deallocate(zE)
    sh = ttnn.sub(zmE, mx)
    ttnn.deallocate(zmE)
    ex = ttnn.exp(sh)
    ttnn.deallocate(sh)
    exm = ttnn.mul(ex, c["nonres"])
    ttnn.deallocate(ex)
    ssum = ttnn.sum(exm, dim=-1, keepdim=True)
    ttnn.deallocate(exm)
    lsum = ttnn.log(ttnn.add(ssum, 1e-30))
    ttnn.deallocate(ssum)
    sink = ttnn.add(lsum, mx)
    ttnn.deallocate(lsum)
    ttnn.deallocate(mx)
    # resident top-k logits (and the shared gate) to their slots
    lifted = ttnn.add(logits, c["keep"])
    cond = ttnn.ge(lifted, t)
    ttnn.deallocate(lifted)
    ttnn.deallocate(t)
    zm = ttnn.where(cond, logits, HYB.filler)
    ttnn.deallocate(cond)
    f = self._fused
    slots = ttnn.matmul(zm, c["P"], compute_kernel_config=f.router_ckc, memory_config=dram)
    ttnn.deallocate(zm)
    sb = ttnn.add(slots, c["bias"])
    ttnn.deallocate(slots)
    st = ttnn.mul(sink, c["sinkhot"])
    ttnn.deallocate(sink)
    l1 = ttnn.L1_MEMORY_CONFIG
    if mk == m:
        out = ttnn.add(sb, st, memory_config=l1)
    else:  # one row through the rows kernels: the extra rows repeat it
        one = ttnn.add(sb, st, memory_config=dram)
        out = ttnn.concat([one] * (mk // m), dim=2, memory_config=l1)
        ttnn.deallocate(one)
    ttnn.deallocate(sb)
    ttnn.deallocate(st)
    return out


def _dc_craft_sigmoid(self, c, logits, m, mk):
    """HYB_DEVCRAFT for a sigmoid + bias router (MiniMax-M2): per row, select the top k of sigmoid(z) + bias (FP32),
    weight by sigmoid(z) normalized over the k. The crafted rows: ln(sigmoid(z)) at each selected resident expert's
    slot, the sink = ln(sum of sigmoid(z) over the selected misses), filler elsewhere, column C = logit column E (0
    without a shared expert): the kernels' softmax over those entries is sigmoid(z) / (sum over the k) for every
    resident pick (the sink takes the missed share). Returns (crafted L1 tile [1, 1, mk, C+32], the selected expert
    ids [1, 1, m, k] row-major: the host routes the misses from the SAME selection, never its own ranking)."""
    E, k = HYB.E, self.top_k
    dram = ttnn.DRAM_MEMORY_CONFIG
    zE = ttnn.slice(logits, [0, 0, 0, 0], [1, 1, m, E])  # FP32
    sg = ttnn.sigmoid(zE)
    ttnn.deallocate(zE)
    sel = ttnn.add(sg, c["sbias"])
    vals, idx = ttnn.topk(sel, k=k, dim=-1, largest=True, sorted=True)
    t = ttnn.slice(vals, [0, 0, 0, k - 1], [1, 1, m, k])
    ttnn.deallocate(vals)
    cond = ttnn.ge(sel, t)
    ttnn.deallocate(sel)
    ttnn.deallocate(t)
    # sink: ln(sum of the selected misses' sigmoids)
    sgm = ttnn.where(cond, sg, 0.0)
    ms = ttnn.sum(ttnn.mul(sgm, ttnn.typecast(c["nonres"], ttnn.float32)), dim=-1, keepdim=True)
    ttnn.deallocate(sgm)
    sink = ttnn.typecast(ttnn.log(ttnn.add(ms, 1e-30)), ttnn.bfloat16)
    ttnn.deallocate(ms)
    # selected experts: ln(sigmoid), the rest filler; the padding columns (E: 0 without a shared expert) as they are
    lsg = ttnn.log(sg)
    ttnn.deallocate(sg)
    zm = ttnn.where(cond, lsg, HYB.filler)
    ttnn.deallocate(cond)
    ttnn.deallocate(lsg)
    pad = ttnn.slice(logits, [0, 0, 0, E], [1, 1, m, E + TILE_W])
    full = ttnn.typecast(ttnn.concat([zm, pad], dim=-1), ttnn.bfloat16)
    ttnn.deallocate(zm)
    ttnn.deallocate(pad)
    f = self._fused
    slots = ttnn.matmul(full, c["P"], compute_kernel_config=f.router_ckc, memory_config=dram)
    ttnn.deallocate(full)
    sb = ttnn.add(slots, c["bias"])
    ttnn.deallocate(slots)
    st = ttnn.mul(sink, c["sinkhot"])
    ttnn.deallocate(sink)
    l1 = ttnn.L1_MEMORY_CONFIG
    if mk == m:
        out = ttnn.add(sb, st, memory_config=l1)
    else:
        one = ttnn.add(sb, st, memory_config=dram)
        out = ttnn.concat([one] * (mk // m), dim=2, memory_config=l1)
        ttnn.deallocate(one)
    ttnn.deallocate(sb)
    ttnn.deallocate(st)
    ids = ttnn.to_layout(idx, ttnn.ROW_MAJOR_LAYOUT, memory_config=dram)
    ttnn.deallocate(idx)
    return out, ids


def _dc_read(self, x, logits, B, ids=None):
    """HYB_DEVCRAFT host step 1: queue the reads of the router logits and the MoE input rows (row-major DRAM) and
    return at once, so the experts are queued right behind them. ids: the device's selection (sigmoid_bias)."""
    t0 = time.perf_counter()
    B["dc"] = (
        ttnn.from_device(logits, blocking=False),
        ttnn.from_device(x, blocking=False),
        ttnn.from_device(ids, blocking=False) if ids is not None else None,
        ttnn.record_event(self.device, 0),
    )
    HYB.st["t_read"] += time.perf_counter() - t0


def _dc_step2(self, B):
    """HYB_DEVCRAFT host step 2 (the device runs the resident experts meanwhile): wait for the reads, route, compute
    the misses, write the host partial (_host_step2)."""
    h_lg, h_x, h_ids, ev = B.pop("dc")
    m = int(B["shapes"]["T"])
    t0 = time.perf_counter()
    ttnn.event_synchronize(ev)
    lg = ttnn.to_torch(ttnn.get_device_tensors(h_lg)[0]).reshape(m, -1).float().numpy()
    sel = None
    if h_ids is not None:
        sel = ttnn.to_torch(ttnn.get_device_tensors(h_ids)[0]).reshape(m, -1)[:, : self.top_k].long().numpy()
    HYB.st["t_read"] += time.perf_counter() - t0
    ids, sc, slot, res, miss = _route_host(self, lg, m, ids=sel)
    if miss.any():
        xh = ttnn.to_torch(ttnn.get_device_tensors(h_x)[0]).reshape(m, -1).to(torch.bfloat16)
        B["job"] = _job(self, xh, ids, sc, miss)
    else:
        B["job"] = None
    _host_step2(self, B)


def _forward_fused(self, x4, orig, partial, logits=None):
    """Decode (one row) through the fused MoE kernels with C slots: router (E-wide) | host step 1 | fused gate-up/down
    on the crafted logits + shared expert | host step 2 (missed experts) | + host partial, then the layer's reduce:
    reduce-scatter (partial False), the replicated-residual boundary's partial (True), or, for the fabric boundary
    ("fabric"), the host partial added in place into the kernels' model-wide L1 output that the boundary reads.
    logits: the E-wide router row when the fabric boundary computed it (QWEN35_FB_ROUTER)."""
    from .seg import SEG

    f = self._fused
    dram = ttnn.DRAM_MEMORY_CONFIG
    if logits is None:
        logits = ttnn.linear(
            x4,
            f.router_w,
            compute_kernel_config=f.router_ckc,
            program_config=f.router_pc,
            memory_config=dram,
            # sigmoid + bias selection compares sigmoid values ~1e-3 apart: FP32 logits, as the model computes them
            dtype=ttnn.float32 if HYB.router_kind in ("sigmoid_bias", "inkling") else None,
        )
    m = int(x4.shape[-2])
    mk = _kernel_rows(m)
    crafted = _dc_craft(self, logits, m, mk) if HYB.devcraft else None
    dc_ids = None
    if isinstance(crafted, tuple):
        crafted, dc_ids = crafted
    # the host step reads one row of each (row-major, DRAM: they stay allocated for the trace's life, no live L1)
    lg_rm = ttnn.to_layout(logits, ttnn.ROW_MAJOR_LAYOUT, memory_config=dram)
    ttnn.deallocate(logits)
    logits = lg_rm
    B = _fused_bufs(self, m)
    h_x = ttnn.to_layout(x4, ttnn.ROW_MAJOR_LAYOUT, memory_config=dram)
    if HYB.devcraft:  # reads queued behind the router, the experts right behind them; the host routes meanwhile
        step1 = lambda: _dc_read(self, h_x, logits, B, dc_ids)  # noqa: E731
        step2 = lambda: _dc_step2(self, B)  # noqa: E731
    elif HYB.overlap:  # host experts in a second step, while the device runs the fused experts
        step1 = lambda: _fused_step1(self, h_x, logits, B)  # noqa: E731
        step2 = lambda: _host_step2(self, B)  # noqa: E731
    else:  # one host step per layer: routing + host experts, then a single device segment
        step1 = lambda: (_fused_step1(self, h_x, logits, B), _host_step2(self, B))  # noqa: E731
        step2 = None
    if HYB._sock_env and not SEG.capturing and m not in HYB.socks.warmed:
        _sock_warm(self, logits, h_x, B, m, mk)
    sock = SEG.capturing and HYB.sock
    if sock:
        if HYB.devcraft or not HYB.overlap or not HYB.fetch.direct:
            raise ValueError("HYB_SOCKET needs the two-step host path with the direct fetch (HYB_DEVCRAFT=0)")
        S = HYB.socks
        ttnn.experimental.send_async_d2h(logits, S.lg)
        ttnn.experimental.send_async_d2h(h_x, S.x)
        for r in S.rows:
            ttnn.experimental.recv_async_h2d(HYB.crafted_rm[mk], r)
        SEG.post(lambda: _sock_step1(self, B, m))
    else:
        step1() if not SEG.capturing else SEG.split(step1)
    if HYB.fetch.on:  # the experts staged by step 1 into their LRU slots (and the sink back to zero)
        HYB.fetch.device_writes(self)
    sug, sdn = f._shared_tensors(self.up_gate, self.down)
    # the kernels read the crafted rows as an L1 BF16 tile (their face reads need L1 alignment: never a DRAM tile)
    if crafted is None:
        crafted = ttnn.to_layout(HYB.crafted_rm[mk], ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
    if mk == 1:
        b = f.buf
        f.gate_up(x4, crafted, self.up_gate, sug, b.act, b.route, bias=f.bias, ep_offset=f.ep_offset)
        f.down_op(b.act, b.route, self.down, sdn, b.out)
        part = b.out
    else:  # FusedMoEDecode.rows on the crafted rows (its own router matmul skipped)
        from models.demos.blackhole.qwen36.tt.moe_decode.union import act_buffer

        route, gu, dn, bb = f.rows_ops(mk)
        route(crafted, bb.table, bias=f.bias)
        _H, Il, Ek, kk = f._key[:4]
        act = act_buffer(f._mesh, Il, Ek, kk, mk)
        gu(x4, bb.table, bb.identity, self.up_gate, sug, act, ep_offset=f.ep_offset)
        dn(act, bb.table, self.down, sdn, bb.out, ep_offset=f.ep_offset)
        ttnn.deallocate(act)
        part = bb.out if mk == m else ttnn.slice(bb.out, [0, 0, 0, 0], [1, 1, m, int(bb.out.shape[-1])])
    ttnn.deallocate(crafted)
    if sock:
        ttnn.experimental.recv_async_h2d(B["host_sock"], HYB.socks.part)
        SEG.post(lambda: _sock_step2(self, B, m))
    elif step2 is not None:
        step2() if not SEG.capturing else SEG.split(step2)
    if not SEG.capturing:  # in a trace these stay allocated: the host steps read them at replay
        ttnn.deallocate(logits)
        ttnn.deallocate(h_x)
        if dc_ids is not None:
            ttnn.deallocate(dc_ids)
    host = ttnn.to_layout(B["host_sock" if sock else "host_rm"], ttnn.TILE_LAYOUT, memory_config=dram)
    if partial == "fabric":
        if mk != 1:  # the boundary reads (and never frees) the one-row kernels' model-wide L1 output
            raise NotImplementedError(
                "fused hybrid on the fabric boundary needs the one-row kernels: HYB_SINGLE_ROWS=0"
            )
        out = ttnn.add(part, host, output_tensor=part)
        ttnn.deallocate(host)
        return out
    out = ttnn.add(part, host, memory_config=dram)
    ttnn.deallocate(host)
    if mk != m:
        ttnn.deallocate(part)
    if partial:
        from models.demos.blackhole.qwen36.tt.decode_boundary import decode_boundary

        part = decode_boundary(self.args, self.device).to_partial(out)
        ttnn.deallocate(out)
        return ttnn.reshape(part, orig) if list(part.shape) != orig else part
    if self.num_devices > 1:
        red = ttnn.experimental.reduce_scatter_minimal_async(
            out,
            persistent_output_buffers=None,
            dim=3,
            multi_device_global_semaphore=self.tt_ccl.get_and_cycle_rs_semaphore_handles(),
            barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(),
            num_links=self.tt_ccl.get_num_links(0),
            memory_config=dram,
            intermediate_memory_config=dram,
            topology=self.args.ccl_topology(),
            chunks_per_sync=10,
            num_workers_per_link=2,
            num_buffers_per_channel=2,
        )
        ttnn.deallocate(out)
        out = red
        orig = orig[:-1] + [out.shape[-1]]
    if list(out.shape) != orig:
        out = ttnn.reshape(out, orig)
    return out


def _host_inputs(x4, scores, ids):
    """DRAM copies of what the host step reads (x, scores, ids); the L1 originals are freed. In a trace these stay
    allocated for the trace's life, and the router leaves scores/ids in L1 at decode: 60 layers of live L1 tensors
    clash with later programs' static circular buffers."""
    return _to_dram(x4, free=False), _to_dram(scores, free=True), _to_dram(ids, free=True)


def _to_dram(t, free):
    # to_memory_config returns its input when the config already matches, so only L1 tensors are copied (and freed)
    if t.memory_config().buffer_type != ttnn.BufferType.L1:
        return t
    d = ttnn.to_memory_config(t, ttnn.DRAM_MEMORY_CONFIG)
    if free:
        ttnn.deallocate(t)
    return d


def _forward(self, x, mode=None, partial=False, logits=None):
    from .seg import SEG

    if not getattr(self, "_hyb_on", True):
        if partial is False and logits is None:  # ports whose MoE forward takes only (x, mode) (MiniMax-M2)
            return _orig_forward(self, x, mode=mode)
        return _orig_forward(self, x, mode=mode, partial=partial, logits=logits)

    orig = list(x.shape)
    H = orig[-1]
    T = math.prod(orig[:-1])
    if self.num_devices > 1 and x.is_sharded():
        x = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG)
    x4 = x if (len(orig) == 4 and orig[2] == T) else ttnn.reshape(x, (1, 1, T, H))
    if getattr(self, "_hyb_fused", False) and mode != "prefill" and (T == 1 or 2 <= T <= HYB.rows_max):
        return _forward_fused(self, x4, orig, partial, logits)
    if partial and HYB.enabled and HYB.C < HYB.E and T == 1:
        raise RuntimeError(f"layer {self.layer_num}: one-row decode boundary (partial={partial!r}) needs HYB_FUSED=1")
    scores, ids = self._route(x4)
    if not (HYB.enabled and HYB.C < HYB.E):
        routed = _routed(self, x4, scores, ids, T, mode)
        ttnn.deallocate(scores)
        ttnn.deallocate(ids)
        return _finish(self, _plus_shared(self, routed, x4), orig, partial)

    if SEG.capturing and mode == "prefill":
        raise RuntimeError(f"layer {self.layer_num}: prefill MoE inside a trace capture (prefill must stay eager)")
    if SEG.capturing:
        # traced decode: end the segment after the router; the host step runs between the segments at replay and
        # writes this layer's persistent buffers, which the next segment reads
        B = self._hyb_bufs.get(T)
        if B is None:
            raise RuntimeError(f"layer {self.layer_num}: no hybrid buffers for T={T}; run one eager decode step first")
        h_x, h_sc, h_ids = _host_inputs(x4, scores, ids)
        SEG.split(lambda: _host_step(self, h_x, h_sc, h_ids, T, "decode", B))
        routed = _routed(self, x4, B["sc"], B["ids"], T, mode)
        if self._dyn_attrs is not None:
            routed = ttnn.add(
                routed, _routed_dyn(self, x4, B["dsc"], B["dids"], T, mode), memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
        out = _plus_shared(self, routed, x4, free=False)
        if HYB.overlap:
            SEG.split(lambda: _host_step2(self, B))  # host experts run while the segment above executes
        out = ttnn.add(out, B["host"], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return _finish(self, out, orig, partial)

    # eager: same decision, uploaded as fresh tensors; decode also (re)creates the persistent buffers for capture
    if mode != "prefill":
        # same DRAM copies as the capture path, so their programs are compiled before capture
        h_x, h_sc, h_ids = _host_inputs(x4, scores, ids)
        dec = _decide(self, h_x, h_sc, h_ids, T, mode)
        scores, ids = h_sc, h_ids
        if h_x is not x4:
            ttnn.deallocate(h_x)
        B = _bufs(self, T, scores, ids, x4)
        _write_bufs(self, B, dec)
        routed = _routed(self, x4, B["sc"], B["ids"], T, mode)
        if self._dyn_attrs is not None:
            routed = ttnn.add(
                routed, _routed_dyn(self, x4, B["dsc"], B["dids"], T, mode), memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
        out = _plus_shared(self, routed, x4, free=False)
        out = ttnn.add(out, B["host"], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(scores)
        ttnn.deallocate(ids)
        return _finish(self, out, orig, partial)
    # the next hybrid layer's prefetch goes on CQ1 before this step's routing readback blocks on the device
    HYB.stream.step(self.layer_num, _nonresident_hot_first, HYB.st)
    dec = _decide(self, x4, scores, ids, T, mode, defer_host=HYB.pf_host_async)
    # the host experts run on a worker thread (the int4 kernel releases the GIL) while the device computes below
    host_fut = HYB.pool.submit(HYB.host_compute, *dec["job"]) if dec["host"] is None else None
    t1 = time.perf_counter()
    stream = list(dec.get("stream") or ())
    # HYB_STREAM_MERGE: the resident experts and the first staging round in one routed call (_merged_round)
    merged = (
        _merged_round(self, dec, stream[0], T) if (stream and HYB.stream.merge and _prefill_sentinel(self, T)) else None
    )
    if merged is not None:
        dec = dict(dec, ids=merged[0], sc=merged[1])
        stream = stream[1:]
    n_ids = _upload_like(self, dec["ids"], ids, torch.int32)
    n_sc = _upload_like(self, dec["sc"], scores, torch.bfloat16)
    rounds = [
        (writes, _upload_like(self, sidx, ids, torch.int32), _upload_like(self, ssc, scores, torch.bfloat16))
        for writes, sidx, ssc in stream
    ]
    ttnn.deallocate(ids)
    ttnn.deallocate(scores)
    HYB.st["t_up"] += time.perf_counter() - t1
    HYB._pf_t = time.perf_counter()
    # sentinel picks are gathered from the dispatch's trailing padding group, which the skip-padding expert matmuls
    # never write: the sorted path zero-fills its down output for them
    self._sorted_zero_out = _prefill_sentinel(self, T)
    cq1 = bool(rounds or merged) and HYB.stream.can_cq1(self.layer_num) and not HYB.profile
    first = merged[2] if merged is not None else (rounds[0][0] if rounds else None)
    copied = _stream_copy(self, first) if cq1 else None  # runs on CQ1 during the resident experts below
    try:
        if merged is not None:
            writes0, gids = merged[2], merged[3]
            if cq1:
                wait = lambda: ttnn.wait_for_event(0, copied)  # noqa: E731
            else:
                wait = lambda: HYB.stream.upload(self.layer_num, writes0, HYB.st)  # noqa: E731
            st = self._stream_attrs
            second = (st["up_gate"], st["down"], HYB.stream.NS, gids, wait)
            routed = self._routed_sorted(x4, n_sc, n_ids, T, second=second)
            HYB.st["stream_rounds"] = HYB.st.get("stream_rounds", 0) + 1
            if rounds and cq1:
                copied = _stream_copy(self, rounds[0][0])  # the next round's copy, after the merged call's reads
        else:
            routed = _routed(self, x4, n_sc, n_ids, T, mode)
        ttnn.deallocate(n_sc)
        ttnn.deallocate(n_ids)
        _pf_mark(self, "t_pf_routed")
        for r, (writes, s_ids, s_sc) in enumerate(rounds):  # the streamed experts: the routed call over the staging
            if cq1:
                ttnn.wait_for_event(0, copied if r == 0 else _stream_copy(self, writes))
            else:
                HYB.stream.upload(self.layer_num, writes, HYB.st)
            _pf_mark(self, "t_pf_upload")
            r2 = _routed_with(self, self._stream_attrs, x4, s_sc, s_ids, T, mode)
            ttnn.deallocate(s_sc)
            ttnn.deallocate(s_ids)
            routed = _add(routed, r2)
            _pf_mark(self, "t_pf_sroute")
    finally:
        self._sorted_zero_out = False
    HYB.stream.mark_read(self.layer_num, self.device)
    HYB.st["stream_rounds"] = HYB.st.get("stream_rounds", 0) + len(rounds)
    out = _plus_shared(self, routed, x4)
    if host_fut is not None:
        t0 = time.perf_counter()
        dec["host"] = host_fut.result()
        HYB.st["t_host_wait"] = HYB.st.get("t_host_wait", 0.0) + time.perf_counter() - t0
    rows_nz = (dec["host"] != 0).any(1).nonzero().reshape(-1)
    if rows_nz.numel():
        t0 = time.perf_counter()
        out = _add(out, _host_rows_dense(self, dec["host"], rows_nz, [int(s) for s in out.shape], out.dtype))
        HYB.st["t_up"] += time.perf_counter() - t0
        _pf_mark(self, "t_pf_hostadd")
    out = _finish(self, out, orig, partial)
    _pf_mark(self, "t_pf_reduce")
    return out


def _nonresident_hot_first(layer):
    """A layer's prefill-missed experts, most routed first (the plan's frequencies; expert order without a plan)."""
    miss = np.flatnonzero(HYB.slot_of[layer] < 0)
    f = HYB._freq
    if f is not None and layer < f.shape[0]:
        miss = miss[np.argsort(-f[layer][miss], kind="stable")]
    return miss.tolist()


def _host_rows_dense(self, host, rows_nz, shp, dtype):
    """The host experts' partial [R, H] fp32 (nonzero on rows_nz only) as a replicated device tensor of shape shp
    (tiled): the nonzero rows go up as a small bf16 table with one trailing zero row, and an embedding lookup of
    the per-row table index (the zero row elsewhere) expands it on the device. Each die adds 1 / tp of it (the
    caller's all-reduce sums the dies)."""
    R, H = host.shape
    n = int(rows_nz.numel())
    table = torch.zeros(n + 1, H, dtype=torch.bfloat16)
    table[:n] = (host[rows_nz] / self.num_devices).to(torch.bfloat16)
    rows = math.prod(shp[:-1])
    idx = torch.full((1, rows), n, dtype=torch.int32)
    idx[0, rows_nz] = torch.arange(n, dtype=torch.int32)
    kw = dict(
        device=self.device, mesh_mapper=ttnn.ReplicateTensorToMesh(self.device), memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    t_dev = ttnn.from_torch(table, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, **kw)
    i_dev = ttnn.from_torch(idx, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, **kw)
    dense = ttnn.embedding(i_dev, t_dev, layout=ttnn.TILE_LAYOUT, dtype=dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(t_dev)
    ttnn.deallocate(i_dev)
    return ttnn.reshape(dense, shp) if list(dense.shape) != shp else dense


def _merged_round(self, dec, round0, T):
    """One routed call over the resident experts and a staging round (optimized_decoder._routed_sorted ``second``):
    a staged pick gets id C + its staging index. Returns (ids, scores, the round's copies, group_ids) with group_ids
    the port's callable: the two per-set group-id lists, built on the host from the dispatch layout of
    moe_sorted_dispatch (experts ascending, ceil(count / group) groups each, the rest padding). HYB_MERGE_CHECK=1
    compares them with the dispatch's own list on every call (by default on the first call only)."""
    writes, sidx, ssc = round0
    C, NS, k = HYB.C, HYB.stream.NS, self.top_k
    staged = sidx < NS
    ids = torch.where(staged, C + sidx, dec["ids"])
    sc = torch.where(staged, ssc, dec["sc"])
    gs = int(self.opt.sorted_group_rows)
    G = _port("moe_dispatch").dispatch_rows(T, k, C + NS, gs) // gs
    e, n = torch.unique(ids[ids < C + NS], return_counts=True)
    gid = torch.full((G,), SENTINEL_ID, dtype=torch.int32)
    used = torch.repeat_interleave(e, -(-n // gs)).to(torch.int32)
    if used.numel() > G:
        raise RuntimeError(f"layer {self.layer_num}: {used.numel()} expert groups > {G} dispatch groups (T {T})")
    gid[: used.numel()] = used
    sent = torch.full_like(gid, SENTINEL_ID)
    gid_a = torch.where(gid < C, gid, sent)
    gid_b = torch.where((gid >= C) & (gid < C + NS), gid - C, sent)

    def group_ids(m):
        ref = m["group_ids"]
        if HYB.merge_check or not HYB.st.get("merge_checked"):
            got = ttnn.to_torch(_die0(ref)).reshape(-1).to(torch.int32)
            if got.shape != gid.shape or not torch.equal(got, gid):
                bad = (got != gid).nonzero().reshape(-1)[:8].tolist() if got.shape == gid.shape else None
                raise RuntimeError(
                    f"layer {self.layer_num}: host group ids differ from the dispatch's (T {T}, G {G}, got "
                    f"{tuple(got.shape)}, first differing groups {bad}: host {gid[bad].tolist() if bad else None} "
                    f"dispatch {got[bad].tolist() if bad else None})"
                )
            HYB.st["merge_checked"] = 1

        def up(t):
            return ttnn.from_torch(
                t.reshape(1, G),
                dtype=ref.dtype,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.device,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        return up(gid_a), up(gid_b)

    return ids, sc, writes, group_ids


def _stream_copy(self, writes):
    """A round's copies into the staging on CQ1, after every CQ0 command queued so far (the staging's last reader is
    the previous streamed call on CQ0). Returns the CQ1 event CQ0 waits on before computing from the staging. With no
    copies (every expert staged, e.g. prefetched) the event only orders CQ0 after the copies already on CQ1."""
    if writes:
        ttnn.wait_for_event(1, ttnn.record_event(self.device, cq_id=0))
        HYB.stream.upload(self.layer_num, writes, HYB.st, cq_id=1)
    return ttnn.record_event(self.device, cq_id=1)


def _pf_mark(self, key):
    """HYB_PROFILE: device time since the previous mark of this prefill layer step (synchronizes)."""
    if not HYB.profile:
        return
    ttnn.synchronize_device(self.device)
    now = time.perf_counter()
    HYB.st[key] = HYB.st.get(key, 0.0) + now - HYB._pf_t
    HYB._pf_t = now


def _upload_like(self, t_host, ref, dtype_torch):
    return ttnn.from_torch(
        t_host.to(dtype_torch).reshape([int(s) for s in ref.shape]),
        dtype=ref.dtype,
        layout=ref.layout,
        device=self.device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _bufs(self, T, scores, ids, x4):
    """Persistent device buffers for a decode width T (allocated in the eager warm-up, before any capture)."""
    B = self._hyb_bufs.get(T)
    if B is not None:
        return B
    rep = ttnn.ReplicateTensorToMesh(self.device)

    def like(ref, dtype_torch):
        return ttnn.from_torch(
            torch.zeros([int(s) for s in ref.shape], dtype=dtype_torch),
            dtype=ref.dtype,
            layout=ref.layout,
            device=self.device,
            mesh_mapper=rep,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    B = {"ids": like(ids, torch.int32), "sc": like(scores, torch.bfloat16), "meta": (ids, scores)}
    if self._dyn_attrs is not None:
        B["dids"] = like(ids, torch.int32)
        B["dsc"] = like(scores, torch.bfloat16)
    H = int(x4.shape[-1])
    B["host"] = ttnn.from_torch(
        torch.zeros(1, 1, T, H),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=self.device,
        mesh_mapper=rep,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    B["shapes"] = {"ids": [int(s) for s in ids.shape], "sc": [int(s) for s in scores.shape], "T": T, "H": H}
    B["dtypes"] = {"ids": (ids.dtype, ids.layout), "sc": (scores.dtype, scores.layout)}
    self._hyb_bufs[T] = B
    return B


def _write_bufs(self, B, dec):
    """Host -> the persistent buffers (copy_host_to_device_tensor: no device allocation)."""
    t0 = time.perf_counter()
    rep = ttnn.ReplicateTensorToMesh(self.device)
    sh = B["shapes"]
    (idt, ilay), (sdt, slay) = B["dtypes"]["ids"], B["dtypes"]["sc"]

    def put(dst, t, dt, lay):
        _put(self, dst, t, dt, lay)

    put(B["ids"], dec["ids"].to(torch.int32).reshape(sh["ids"]), idt, ilay)
    put(B["sc"], dec["sc"].to(torch.bfloat16).reshape(sh["sc"]), sdt, slay)
    if "dids" in B:
        put(B["dids"], dec["dids"].to(torch.int32).reshape(sh["ids"]), idt, ilay)
        put(B["dsc"], dec["dsc"].to(torch.bfloat16).reshape(sh["sc"]), sdt, slay)
    if dec["host"] is not None:
        _put_host(self, B, dec["host"], put)
        B["host_zero"] = False
    HYB.st["t_up"] += time.perf_counter() - t0


def _put_host(self, B, host, put):
    sh = B["shapes"]
    n, inv = host.shape[0], 1.0 / self.num_devices

    def fill(v):  # the per-device partial, fp32 -> bf16 in the pinned slice; the rows past the host's are zero
        torch.mul(host, inv, out=v[:n])
        if v.shape[0] > n:
            v[n:].zero_()

    if "host_rm" in B and _put_fill(self, B["host_rm"], fill):
        return
    if "host_rm" in B:  # fused: row-major rows [1, 1, T, H] (decode)
        full = torch.zeros(1, 1, sh["T"], sh["H"], dtype=torch.float32)
        full.view(-1, sh["H"])[: host.shape[0]] = host / self.num_devices  # per-device partial
        put(B["host_rm"], full.to(torch.bfloat16), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
        return
    full = torch.zeros(1, 1, sh["T"], sh["H"], dtype=torch.float32)
    full.view(-1, sh["H"])[: host.shape[0]] = host / self.num_devices  # per-device partial
    put(B["host"], full.to(torch.bfloat16), ttnn.bfloat16, ttnn.TILE_LAYOUT)


def _put_fill(self, dst, fill):
    """A pinned row-major bf16 upload written in place by fill(view) (PinnedRows.put_fill); False without pinned rows."""
    rows = HYB.fetch.rows
    if rows is None:
        return False
    c0, w0 = rows.t_copy, rows.t_write
    rows.put_fill(self.layer_num, dst, fill)
    HYB.st["t_up_cp"] = HYB.st.get("t_up_cp", 0.0) + rows.t_copy - c0
    HYB.st["t_up_w"] = HYB.st.get("t_up_w", 0.0) + rows.t_write - w0
    return True


def _put(self, dst, t, dt, lay):
    rows = HYB.fetch.rows
    if rows is not None and lay == ttnn.ROW_MAJOR_LAYOUT and dt == ttnn.bfloat16:  # pinned zero-copy (HYB_PIN_UPLOADS)
        c0, w0 = rows.t_copy, rows.t_write
        rows.put(self.layer_num, dst, t)
        HYB.st["t_up_cp"] = HYB.st.get("t_up_cp", 0.0) + rows.t_copy - c0
        HYB.st["t_up_w"] = HYB.st.get("t_up_w", 0.0) + rows.t_write - w0
        return
    t1 = time.perf_counter()
    h = ttnn.from_torch(t, dtype=dt, layout=lay, mesh_mapper=ttnn.ReplicateTensorToMesh(self.device))
    HYB.st["t_up_conv"] += time.perf_counter() - t1
    ttnn.copy_host_to_device_tensor(h, dst)


def _host_step(self, x4, scores, ids, T, mode, B):
    """Between segments a and b: routing -> slot buffers. The missed experts' host compute is left in B["job"] for
    _host_step2, which runs after segment b has been queued (the device computes the resident experts meanwhile)."""
    dec = _decide(self, x4, scores, ids, T, mode, defer_host=HYB.overlap)
    _write_bufs(self, B, dec)
    B["job"] = dec["job"] if dec["host"] is None else False


def _host_step2(self, B):
    t0 = time.perf_counter()
    _host_step2_body(self, B)
    HYB.st["t_s2"] = HYB.st.get("t_s2", 0.0) + time.perf_counter() - t0


def _host_step2_body(self, B):
    job, B["job"] = B["job"], None
    if job is False:  # computed (and written) in _host_step
        return
    if job is None:  # no misses: the host partial must still be zeroed once
        if B.get("host_zero"):
            return
        host = torch.zeros(1, B["shapes"]["H"], dtype=torch.float32)
    elif hasattr(job, "result"):  # started in step 1 (HYB_LRU)
        t0 = time.perf_counter()
        host = job.result()
        HYB.st["t_wait"] += time.perf_counter() - t0
    else:
        host = HYB.host_compute(*job)
    t0 = time.perf_counter()
    _put_host(self, B, host, lambda d, t, dt, lay: _put(self, d, t, dt, lay))
    B["host_zero"] = job is None
    HYB.st["t_up"] += time.perf_counter() - t0


def _apply_promotions(self, L, protect):
    """Write prepared experts of layer L into dynamic slots (LRU victims, never one routed this step), through the
    preallocated staging tensors (no device allocation: safe between trace segments)."""
    t0 = time.perf_counter()
    done = [(e, f) for (l, e), f in list(HYB.pending.items()) if l == L and f.done()]
    n = 0
    for e, f in done:
        if n >= HYB.promote:
            break
        del HYB.pending[(L, e)]
        if HYB.dyn_of[L][e] >= 0 or HYB.slot_of[L][e] >= 0:
            continue
        lru, free = HYB.dyn_lru[L], HYB.dyn_free[L]
        if free:
            slot = free.pop()
        else:
            victim = next((v for v in lru if v not in protect), None)
            if victim is None:
                continue
            slot = lru.pop(victim)
            HYB.dyn_of[L][victim] = -1
        h_gu, h_dn = f.result()
        gv, dv = self._dyn_views
        sg, sd = self._stage
        ttnn.copy_host_to_device_tensor(h_gu, sg)
        ttnn.copy_host_to_device_tensor(h_dn, sd)
        ttnn.fill_cache(gv, sg, slot)
        ttnn.fill_cache(dv, sd, slot)
        HYB.dyn_of[L][e] = slot
        lru[e] = slot
        n += 1
    HYB.st["promoted"] += n
    HYB.st["t_promote"] += time.perf_counter() - t0


def _decide(self, x4, scores, ids, T, mode, defer_host=False):
    """Read back the routing, apply pending promotions (decode), split picks into static / dynamic / host, compute
    the host part. Returns torch tensors: ids, sc (static), dids, dsc (dynamic), host [valid, H] fp32."""
    L = self.layer_num
    k = self.top_k
    if HYB.profile:  # split "wait for the device segment" from the readback itself
        t0 = time.perf_counter()
        ttnn.synchronize_device(self.device)
        HYB.st["t_sync"] += time.perf_counter() - t0
    t0 = time.perf_counter()
    idh = ttnn.to_torch(_die0(ids)).reshape(-1, k).long()
    sch = ttnn.to_torch(_die0(scores)).reshape(-1, k).float()
    HYB.st["t_read"] += time.perf_counter() - t0
    return _decide_core(self, x4, idh, sch, mode, defer_host)


SENTINEL_ID = 0xFFFF  # an expert id >= any slot count: moe_sorted_dispatch counts no row for it


def _prefill_sentinel(self, rows):
    """Whether this prefill MoE call takes the port's expert-sorted path (optimized_decoder._routed_masked), so the
    picks of experts outside a call can carry SENTINEL_ID. The masked path scatters ids into a dense [T, E] matrix and
    needs in-range ids."""
    opt = getattr(self, "opt", None)
    return (
        HYB.prefill_sentinel
        and opt is not None
        and getattr(opt, "prefill_mode", None) == "sorted"
        and opt.sorted_min_tokens <= rows <= opt.sorted_max_tokens
    )


def _cpu_split(L, idh, miss):
    """The missed experts of a prefill layer step to compute on the CPU (tensor of ids) or None: fewest tokens first,
    while the CPU estimate (a + b * pairs per expert, plus the step cost c0 + c1 * rows) stays at or below the copy
    time of the experts left to stage. Experts already in the staging cost no copy and stay on the device."""
    e, n = torch.unique(idh[miss], return_counts=True)
    staged = HYB.stream.where
    cand = [(int(c), int(x)) for x, c in zip(e.tolist(), n.tolist()) if (L, int(x)) not in staged]
    if not cand:
        return None
    cand.sort()
    a, b = HYB.pf_cpu_ms
    copy_left = HYB.pf_copy_ms * len(cand)
    cpu_t = HYB.pf_cpu_fixed_ms[0] + HYB.pf_cpu_fixed_ms[1] * idh.shape[0]
    pick = []
    for c, x in cand:
        t = a + b * c
        if cpu_t + t > copy_left - HYB.pf_copy_ms:
            break
        cpu_t += t
        copy_left -= HYB.pf_copy_ms
        pick.append(x)
    if not pick:
        return None
    HYB.st["pf_cpu_experts"] = HYB.st.get("pf_cpu_experts", 0) + len(pick)
    return torch.tensor(pick, dtype=idh.dtype)


def _decide_core(self, x4, idh, sch, mode, defer_host=False):
    """Routing on the host (ids/scores [rows, k]) -> static / dynamic / host split and the host part (see _decide)."""
    L = self.layer_num
    k = self.top_k
    t0 = time.perf_counter()
    rows = idh.shape[0]
    valid = rows if mode == "prefill" else min(rows, HYB.decode_rows)
    HYB.st["t_read"] += time.perf_counter() - t0
    use_dyn = self._dyn_attrs is not None
    if use_dyn and mode != "prefill":
        _apply_promotions(self, L, set(idh[:valid].flatten().tolist()))
    ic = idh.clamp(0, HYB.E - 1)
    s_slot = torch.from_numpy(HYB.slot_of[L]).long()[ic]
    # prefill: the grouped (sorted-dispatch) sparse matmul cannot hold a bfp8 expert block in L1, so the bfp8 dynamic
    # slots serve decode only; in prefill a dynamic-resident expert counts as a miss (host)
    d_slot = torch.from_numpy(HYB.dyn_of[L]).long()[ic] if (use_dyn and mode != "prefill") else torch.full_like(ic, -1)
    in_s, in_d = s_slot >= 0, (s_slot < 0) & (d_slot >= 0)
    miss = ~(in_s | in_d)
    v_miss = miss[:valid]
    if mode == "prefill" and HYB.pf_mask_pad and HYB.pf_valid is not None and HYB.pf_valid < valid:
        # the bucket's padding rows (after the segment's real rows; no real row attends to them): their missed
        # experts are neither computed on the CPU nor streamed (their picks take the sink / sentinel)
        v_miss = v_miss.clone()
        v_miss[HYB.pf_valid :] = False
    HYB.st["picks"] += int(valid * k)
    HYB.st["hits"] += int(valid * k - v_miss.sum())
    if mode != "prefill":
        HYB.st["dec_picks"] += int(valid * k)
        HYB.st["dec_hits"] += int(valid * k - v_miss.sum())
    HYB.st["dyn_hits"] += int(in_d[:valid].sum())
    if mode == "prefill":
        HYB.st["prefill_rows"] += valid
        HYB.st["prefill_miss_pairs"] += int(v_miss.sum())
        if HYB.replan_tokens and L in HYB.fetch.store:
            n = valid if HYB.pf_valid is None else min(valid, HYB.pf_valid)
            m = HYB.pf_mass.setdefault(L, np.zeros(HYB.E))
            np.add.at(m, idh[:n].numpy().ravel(), sch[:n].numpy().ravel())
    else:
        HYB.st["steps"] += 1
        if HYB.record is not None:
            HYB.record.append((L, idh[0].clone(), sch[0].clone()))
        if use_dyn:
            lru = HYB.dyn_lru[L]
            for e in idh[:valid][in_d[:valid]].tolist():
                lru.move_to_end(e)
            for e in set(idh[:valid][v_miss].tolist()):
                if (L, e) not in HYB.pending:
                    HYB.pending[(L, e)] = HYB.pool.submit(HYB.prep, self._meta, L, e)
    if HYB.skip > 0:  # drop missed experts whose routing weight is below the threshold (their output is lost)
        drop = v_miss & (sch[:valid] < HYB.skip)
        HYB.st["skipped"] += int(drop.sum())
        HYB.st["skipped_mass"] += float(sch[:valid][drop].sum())
        v_miss = v_miss & ~drop
    sentinel = mode == "prefill" and _prefill_sentinel(self, rows)
    stream = None
    if mode == "prefill" and HYB.stream.on and bool(v_miss.any()):
        to_stage = v_miss
        if HYB.pf_split and HYB.hostk and HYB.lib_dir and HYB.lib_resident and rows <= HYB.pf_split_max_rows:
            cpu = _cpu_split(L, idh[:valid], v_miss)
            if cpu is not None:
                to_stage = v_miss & ~torch.isin(idh[:valid], cpu)
        rounds = HYB.stream.plan(L, idh[:valid], to_stage, other=SENTINEL_ID if sentinel else None)
        if rounds is not None:  # these picks run on the device from the staging (hybrid/stream.py)
            stream = []
            for rmask, sidx, writes in rounds:
                v_miss = v_miss & ~rmask
                stream.append((writes, sidx, torch.where(rmask, sch[:valid], torch.zeros_like(sch[:valid]))))
    if bool(v_miss.any()):
        t1 = time.perf_counter()
        xh = ttnn.to_torch(_die0(x4)).reshape(-1, x4.shape[-1])[:valid].to(torch.bfloat16)
        HYB.st["t_read"] += time.perf_counter() - t1
        job = (L, xh, idh[:valid], sch[:valid], v_miss)
        host = None if defer_host else HYB.host_compute(*job)
    else:
        job = None
        host = torch.zeros(valid, int(x4.shape[-1]), dtype=torch.float32)
    # non-resident picks need SOME index: the all-zero sink slot when there is one (fused: C - 1), never a real
    # expert's slot. The prefill routing matrix is a scatter of (index, score) per pick, and a zero-score pick aimed at
    # slot 0 could overwrite a real pick of the slot-0 expert in the same row. The sorted prefill dispatch takes the
    # sentinel instead: it gives the pick no row, where a slot index would compute a zero-weight row of that slot.
    z = torch.full_like(s_slot, SENTINEL_ID if sentinel else (HYB.C - 1 if HYB.fused else 0))
    zf = torch.zeros_like(sch)
    return {
        "ids": torch.where(in_s, s_slot, z),
        "sc": torch.where(in_s, sch, zf),
        "dids": torch.where(in_d, d_slot, z),
        "dsc": torch.where(in_d, sch, zf),
        "host": host,
        "job": job,
        "stream": stream,
    }
