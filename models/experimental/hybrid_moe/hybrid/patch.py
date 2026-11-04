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
  HYB_OVERLAP      1 (default): traced decode splits after the resident experts, so the host experts overlap them
  HYB_PROFILE      1: synchronize before each readback to time the device segment wait separately (t_sync)
  HYB_PLAN         .npz with routing (routing.npz layout) -> static set = the C most-routed experts per layer
  HYB_LIB          int4 host expert library dir (build_int4_library.py); otherwise bf16 dequant from the checkpoint
  HYB_LIB_RESIDENT 1 (default): copy each layer's non-static int4 experts into RAM during the build (0: mmap only)
  HYB_DECODE_ROWS  valid rows of a decode step (default 1: single user)
  HYB_THREADS      host torch threads (default 8)
"""

from __future__ import annotations

import math
import os
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

import ttnn


class HybridState:
    def __init__(self):
        self.E = None
        self.C = int(os.environ.get("HYB_SLOTS", "0") or 0)
        self.D = int(os.environ.get("HYB_DYN_SLOTS", "0") or 0)
        self.promote = int(os.environ.get("HYB_PROMOTE", "2"))
        self.profile = os.environ.get("HYB_PROFILE") == "1"
        self.overlap = os.environ.get("HYB_OVERLAP", "1") == "1"
        self.fused = os.environ.get("HYB_FUSED", "1") == "1"
        self.skip = float(os.environ.get("HYB_SKIP", "0"))  # missed experts below this routing weight are not computed
        # static plan counts only picks with at least this routing weight (default HYB_SKIP: skipped misses are free)
        self.plan_min_score = float(os.environ.get("HYB_PLAN_MIN_SCORE", str(self.skip)))  # fused MoE decode kernels (moe_decode/fused.py)
        self.decode_rows = int(os.environ.get("HYB_DECODE_ROWS", "1"))
        self.plan_path = os.environ.get("HYB_PLAN", "")
        self.lib_dir = os.environ.get("HYB_LIB", "")
        self.sd = None
        self.slot_of = {}  # layer -> np.int32[E] static slot (-1 = none)
        self.dyn_of = {}  # layer -> np.int32[E] dynamic slot (-1 = none)
        self.dyn_lru = {}  # layer -> OrderedDict(expert -> dyn slot), LRU first
        self.dyn_free = {}  # layer -> list of free dyn slots
        self.pending = {}  # (layer, e) -> Future of prepared host tensors
        self.resident = {}
        self._freq = None
        self.crafted_rm = None  # fused hybrid: the model-wide crafted logits row (row-major DRAM), see _fused_step1
        self.lib = {}  # layer -> future of the layer's host library
        self.lib_resident = os.environ.get("HYB_LIB_RESIDENT", "1") == "1"
        self.loader = ThreadPoolExecutor(max_workers=1)
        self.lock = threading.Lock()
        self.pool = ThreadPoolExecutor(max_workers=int(os.environ.get("HYB_PREP_WORKERS", "2")))
        self.record = None
        self.enabled = True
        self.reset_stats()

    def reset_stats(self):
        self.st = {"picks": 0, "hits": 0, "dyn_hits": 0, "steps": 0, "promoted": 0, "t_read": 0.0, "t_host": 0.0,
                   "t_up": 0.0, "t_promote": 0.0, "prefill_rows": 0, "prefill_miss_pairs": 0, "t_sync": 0.0, "t_up_conv": 0.0, "t_lib_load": 0.0, "dec_picks": 0, "dec_hits": 0, "skipped": 0,
                   "skipped_mass": 0.0, "t_route": 0.0, "t_craft": 0.0}

    # -- static plan ----------------------------------------------------------------------------------------------
    def plan(self, layer):
        if layer in self.resident:
            return self.resident[layer]
        E, C = self.E, self.C
        if self._freq is None and self.plan_path and os.path.exists(self.plan_path):
            z = np.load(self.plan_path)
            layers = [int(v) for v in z["layers"]]
            f = np.zeros((max(layers) + 1, E))
            for k in z.files:
                if k.startswith("p"):
                    a = z[k]
                    # with scores (s<k>) and a threshold, count only the picks the host would compute on a miss
                    sk = "s" + k[1:]
                    w = (z[sk] >= self.plan_min_score).astype(np.float64) if (self.plan_min_score > 0 and sk in z.files) \
                        else np.ones(a.shape)
                    for li, L in enumerate(layers):
                        np.add.at(f[L], a[:, li, :].ravel(), w[:, li, :].ravel())
            self._freq = f
        n_real = C - 1 if self.fused else C  # fused: the last slot is the all-zero sink (expert id -1)
        if self._freq is not None and layer < self._freq.shape[0] and self._freq[layer].sum() > 0:
            ids = list(np.argsort(-self._freq[layer], kind="stable")[:n_real])
        else:
            ids = list(range(n_real))
        ids = [int(i) for i in ids] + ([-1] if self.fused else [])
        self.resident[layer] = ids
        so = np.full(E, -1, dtype=np.int32)
        so[ids[:n_real]] = np.arange(n_real, dtype=np.int32)
        self.slot_of[layer] = so
        self.dyn_of[layer] = np.full(E, -1, dtype=np.int32)
        self.dyn_lru[layer] = OrderedDict()
        self.dyn_free[layer] = list(range(self.D))
        return ids

    # -- host experts ---------------------------------------------------------------------------------------------
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
        miss then reads its expert from NVMe. row[e] = the expert's row in the copies (-1: static, never on the host)."""
        t0 = time.perf_counter()
        d = torch.load(os.path.join(self.lib_dir, f"layer_{layer}.pt"), mmap=True, weights_only=True)
        if not self.lib_resident:
            d["row"] = np.arange(self.E, dtype=np.int64)
            return d
        cold = np.flatnonzero(self.slot_of[layer] < 0) if layer in self.slot_of else np.arange(self.E)
        idx = torch.from_numpy(cold)
        out = {k: d[k].index_select(0, idx).contiguous() for k in ("gu_p", "gu_sz", "dn_p", "dn_sz")}
        out["gs"] = d["gs"]
        row = np.full(self.E, -1, dtype=np.int64)
        row[cold] = np.arange(len(cold))
        out["row"] = row
        self.st["t_lib_load"] += time.perf_counter() - t0
        return out

    def host_compute(self, layer, X, ids, sc, miss_mask):
        """X [R, H] bf16, ids/sc [R, k], miss_mask [R, k] -> [R, H] fp32: the missed experts' weighted sum."""
        R, H = X.shape
        out = torch.zeros(R, H, dtype=torch.float32)
        if not miss_mask.any():
            return out
        t0 = time.perf_counter()
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
                h = torch._weight_int4pack_mm_for_cpu(xr, lib["gu_p"][r], gs, lib["gu_sz"][r])
                n = h.shape[1] // 2
                a = (torch.nn.functional.silu(h[:, :n].float()) * h[:, n:].float()).to(torch.bfloat16)
                y = torch._weight_int4pack_mm_for_cpu(a, lib["dn_p"][r], gs, lib["dn_sz"][r])
            else:
                with self.lock:
                    gu, dn = self.sd.expert(layer, e)
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
        h_gu = ttnn.from_torch(packed.contiguous(), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT,
                               mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=-1))
        h_dn = ttnn.from_torch(down_t, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT,
                               mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=-2))
        return h_gu, h_dn


TILE_W = 32
HYB = HybridState()


def _die0(t):
    return ttnn.get_device_tensors(t)[0]


def install():
    torch.set_num_threads(int(os.environ.get("HYB_THREADS", "8")))
    from models.autoports.qwen_qwen3_5_122b_a10b.tt import model_config as mc
    from models.autoports.qwen_qwen3_5_122b_a10b.tt import moe as moe_mod
    from models.autoports.qwen_qwen3_5_122b_a10b.tt import optimized_decoder as od

    from .state import HybridStateDict

    def load_state_dict(self):
        HYB.E = int(self.num_experts)
        if not HYB.C or HYB.C > HYB.E:
            HYB.C = HYB.E
        sd = HybridStateDict(
            self.CKPT_DIR, linear_q_dim=self.linear_q_dim, linear_k_dim=self.linear_k_dim, resident_fn=HYB.plan
        )
        HYB.sd = sd
        return sd

    mc.Qwen35MoeModelArgs.load_state_dict = load_state_dict

    orig_init = od.Qwen35MoEOptimized.__init__

    def init(self, mesh_device, args, state_dict, tensor_cache_path=None, tt_ccl=None, layer_num=None):
        E = int(args.num_experts)
        HYB.plan(layer_num)
        if HYB.lib_dir:
            HYB.lib_layer_async(layer_num)  # load this layer's host library while the rest of the model builds
        args.num_experts = HYB.C
        try:
            orig_init(self, mesh_device, args, state_dict, tensor_cache_path, tt_ccl=tt_ccl, layer_num=layer_num)
        finally:
            args.num_experts = E
        self.num_experts = HYB.C
        self.E_full = E
        if int(self.router_w.shape[-1]) != E:
            raise RuntimeError(f"router width {self.router_w.shape} != E {E}")
        self._hyb_fused = False
        if HYB.fused and HYB.C < E:
            if getattr(self, "_fused", None) is None:
                raise RuntimeError(f"layer {layer_num}: HYB_FUSED=1 but the fused MoE decode was not built "
                                   f"(QWEN35_MOE_FUSED_DECODE={os.environ.get('QWEN35_MOE_FUSED_DECODE')}, C={HYB.C})")
            if HYB.D:
                raise RuntimeError("HYB_FUSED=1 does not support dynamic slots (HYB_DYN_SLOTS)")
            self._hyb_fused = True
        self._static_attrs = None
        self._dyn_attrs = None
        self._hyb_bufs = {}  # T -> persistent device buffers the host step writes (allocated before any capture)
        if HYB.D and HYB.C < E:
            D, tp, H, N, Nl = HYB.D, self.num_devices, self.hidden, self.inter, self.inter_local
            rep = ttnn.ReplicateTensorToMesh(mesh_device)
            sh_last = ttnn.ShardTensorToMesh(mesh_device, dim=-1)
            sh_m2 = ttnn.ShardTensorToMesh(mesh_device, dim=-2)
            kw = {"layout": ttnn.TILE_LAYOUT, "device": mesh_device, "memory_config": ttnn.DRAM_MEMORY_CONFIG}
            gu = ttnn.from_torch(torch.zeros(1, D, H, tp * 2 * Nl), dtype=ttnn.bfloat8_b, mesh_mapper=sh_last, **kw)
            dn = ttnn.from_torch(torch.zeros(1, D, N, H), dtype=ttnn.bfloat8_b, mesh_mapper=sh_m2, **kw)
            ones = ttnn.from_torch(torch.ones(1, 1, 1, D, dtype=torch.bfloat16), dtype=ttnn.bfloat16,
                                   layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device, mesh_mapper=rep)
            gs, ds = [int(v) for v in gu.shape], [int(v) for v in dn.shape]
            self._dyn_views = (ttnn.reshape(gu, [gs[1], 1, gs[2], gs[3]]), ttnn.reshape(dn, [ds[1], 1, ds[2], ds[3]]))
            # promotion staging: allocated once, written with copy_host_to_device_tensor (no device allocation later)
            self._stage = (
                ttnn.from_torch(torch.zeros(1, 1, H, tp * 2 * Nl), dtype=ttnn.bfloat8_b, mesh_mapper=sh_last, **kw),
                ttnn.from_torch(torch.zeros(1, 1, N, H), dtype=ttnn.bfloat8_b, mesh_mapper=sh_m2, **kw),
            )
            self._dyn_attrs = {"up_gate": gu, "down": dn, "ones_mask": ones, "num_experts": D, "_pairs_masks": {},
                               "_dispatch": None}
            self._static_attrs = {k: getattr(self, k, None) for k in self._dyn_attrs}
            if self._static_attrs["_pairs_masks"] is None:
                self._static_attrs["_pairs_masks"] = {}
            self._meta = (tp, Nl, mesh_device)

    od.Qwen35MoEOptimized.__init__ = init

    from models.demos.blackhole.qwen36.tt.moe_decode import fused as moe_fused

    f_orig = moe_fused.FusedMoEDecode.__init__

    def f_init(fself, mesh_device, args, *, num_experts, router_weight, shared_gate_weight, hidden, **kw):
        E_true = int(router_weight.shape[0])
        f_orig(fself, mesh_device, args, num_experts=num_experts, router_weight=router_weight[:num_experts],
               shared_gate_weight=shared_gate_weight, hidden=hidden, **kw)
        if E_true == num_experts:
            return
        # the kernels rank the C slots; the router (run by _forward_fused, its logits read on the host) stays E-wide
        from models.demos.blackhole.qwen36.tt import tp_common as tpc

        ttnn.deallocate(fself.router_w)
        rw = torch.zeros(hidden, E_true + TILE_W)
        rw[:, :E_true] = router_weight.T.float()
        rw[:, E_true] = shared_gate_weight.reshape(-1).float()
        fself.router_w = ttnn.from_torch(rw, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device,
                                         mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
                                         memory_config=ttnn.DRAM_MEMORY_CONFIG)
        fself.router_pc = tpc.create_matmul_1d_decode_progcfg(
            TILE_W, hidden, E_true + TILE_W, (E_true + TILE_W) // TILE_W, fp32_acc=True,
            grid_w=mesh_device.compute_with_storage_grid_size().x,
            in0_block_w=16 if (hidden // TILE_W) % 16 == 0 else None,
        )

    moe_fused.FusedMoEDecode.__init__ = f_init
    moe_mod.Qwen35MoE.forward = _forward

    # prefill stays eager: the per-bucket prefill traces would capture the MoE host step (decode-only by design);
    # without a bucket trace the model's prefill falls back to its stock (eager) path
    from models.autoports.qwen_qwen3_5_122b_a10b.tt import generator as gen_mod

    gen_mod.Qwen35MoeGenerator._warm_prefill_traces = lambda self: None


def _routed(self, x4, scores, ids, T, mode):
    if mode == "decode" or (mode is None and T <= self.GATHER_MAX_TOKENS):
        if T <= self.GATHER_MAX_TOKENS:
            return self._routed_gather(x4, scores, ids, T)
        return self._routed_masked(x4, scores, ids, T, chunk=32)
    return self._routed_masked(x4, scores, ids, T, chunk=self.PREFILL_CHUNK)


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

        out = tt_all_reduce(out, self.device, self.tt_ccl, cluster_axis=0, dim=3, topology=self.args.ccl_topology(),
                            memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16)
        orig = orig[:-1] + [out.shape[-1]]
    if list(out.shape) != orig:
        out = ttnn.reshape(out, orig)
    return out


def _add(a, b):
    c = ttnn.add(a, b, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(a)
    ttnn.deallocate(b)
    return c


def _fused_bufs(self):
    """The fused hybrid's persistent host-written buffers (allocated in the eager warm-up, before any capture): the host
    partial as one row-major row [1, 1, 1, H] (tiled on the device), and the model-wide crafted logits row."""
    from .seg import SEG

    B = self._hyb_bufs.get("fused")
    if B is None:
        if SEG.capturing:
            raise RuntimeError(f"layer {self.layer_num}: no fused hybrid buffers; run one eager decode step first")
        H = int(self.hidden)
        B = {"host_rm": _rm_buffer(self.device, H), "shapes": {"T": 1, "H": H}}
        self._hyb_bufs["fused"] = B
    if HYB.crafted_rm is None:
        HYB.crafted_rm = _rm_buffer(self.device, HYB.C + TILE_W)
    return B


def _rm_buffer(mesh_device, width):
    return ttnn.from_torch(torch.zeros(1, 1, 1, width), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT,
                           device=mesh_device, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
                           memory_config=ttnn.DRAM_MEMORY_CONFIG)


def _fused_step1(self, x, logits, B):
    """Host step after the router (one decode row): true routing from the E-wide logits row, then the crafted logits
    row the fused kernel ranks over the C slots: ln(score) at each routed resident expert's slot, ln(missed mass) at
    the all-zero sink slot C-1, -1000 elsewhere, and the shared-expert gate logit at column C. The kernel's top-k +
    softmax over the selected logits then reproduces the true weights of the resident experts (the sink absorbs the
    missed mass). x / logits: row-major single rows in DRAM."""
    E, C, k, L = HYB.E, HYB.C, self.top_k, self.layer_num
    if HYB.profile:
        t0 = time.perf_counter()
        ttnn.synchronize_device(self.device)
        HYB.st["t_sync"] += time.perf_counter() - t0
    t0 = time.perf_counter()
    lg = ttnn.to_torch(_die0(logits)).reshape(-1).float().numpy()
    HYB.st["t_read"] += time.perf_counter() - t0
    t0 = time.perf_counter()
    z = lg[:E]
    ids = np.argpartition(-z, k)[:k]
    e = np.exp(z[ids] - z[ids].max())
    sc = e / e.sum()  # softmax over the top-k logits == softmax over all E, top-k, renormalized
    slot = HYB.slot_of[L][ids]
    res = slot >= 0
    miss = ~res
    st = HYB.st
    st["picks"] += k
    st["hits"] += int(res.sum())
    st["dec_picks"] += k
    st["dec_hits"] += int(res.sum())
    st["steps"] += 1
    if HYB.record is not None:
        HYB.record.append((L, torch.from_numpy(ids.astype(np.int64)), torch.from_numpy(sc.astype(np.float32))))
    if HYB.skip > 0:
        drop = miss & (sc < HYB.skip)
        st["skipped"] += int(drop.sum())
        st["skipped_mass"] += float(sc[drop].sum())
        miss = miss & ~drop
    HYB.st["t_route"] += time.perf_counter() - t0
    t0 = time.perf_counter()
    row = np.full((1, 1, 1, C + TILE_W), -1000.0, dtype=np.float32)
    row[..., C:] = 0.0
    row[..., C] = lg[E]  # shared-expert gate logit (router column E)
    row[0, 0, 0, slot[res]] = np.log(sc[res])
    miss_mass = float(sc[~res].sum())  # computed and skipped misses: their routing mass goes to the zero sink
    if miss_mass > 1e-6:
        row[..., C - 1] = math.log(miss_mass)
    HYB.st["t_craft"] += time.perf_counter() - t0
    t0 = time.perf_counter()
    _put(self, HYB.crafted_rm, torch.from_numpy(row).to(torch.bfloat16), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
    HYB.st["t_up"] += time.perf_counter() - t0
    if miss.any():
        t0 = time.perf_counter()
        xh = ttnn.to_torch(_die0(x)).reshape(1, -1).to(torch.bfloat16)
        HYB.st["t_read"] += time.perf_counter() - t0
        it = torch.from_numpy(ids.astype(np.int64))[None]
        B["job"] = (L, xh, it, torch.from_numpy(sc.astype(np.float32))[None], torch.from_numpy(miss)[None])
    else:
        B["job"] = None


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
        logits = ttnn.linear(x4, f.router_w, compute_kernel_config=f.router_ckc, program_config=f.router_pc,
                             memory_config=dram)
    # the host step reads one row of each (row-major, DRAM: they stay allocated for the trace's life, no live L1)
    lg_rm = ttnn.to_layout(logits, ttnn.ROW_MAJOR_LAYOUT, memory_config=dram)
    ttnn.deallocate(logits)
    logits = lg_rm
    B = _fused_bufs(self)
    h_x = ttnn.to_layout(x4, ttnn.ROW_MAJOR_LAYOUT, memory_config=dram)
    if HYB.overlap:  # host experts in a second step, while the device runs the fused experts
        step1 = lambda: _fused_step1(self, h_x, logits, B)  # noqa: E731
        step2 = lambda: _host_step2(self, B)  # noqa: E731
    else:  # one host step per layer: routing + host experts, then a single device segment
        step1 = lambda: (_fused_step1(self, h_x, logits, B), _host_step2(self, B))  # noqa: E731
        step2 = None
    step1() if not SEG.capturing else SEG.split(step1)
    b = f.buf
    sug, sdn = f._shared_tensors(self.up_gate, self.down)
    # the kernels read the crafted row as an L1 BF16 tile (its face reads need L1 alignment: never a DRAM tile)
    crafted = ttnn.to_layout(HYB.crafted_rm, ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
    f.gate_up(x4, crafted, self.up_gate, sug, b.act, b.route, bias=f.bias, ep_offset=f.ep_offset)
    f.down_op(b.act, b.route, self.down, sdn, b.out)
    ttnn.deallocate(crafted)
    if step2 is not None:
        step2() if not SEG.capturing else SEG.split(step2)
    if not SEG.capturing:  # in a trace these stay allocated: the host steps read them at replay
        ttnn.deallocate(logits)
        ttnn.deallocate(h_x)
    host = ttnn.to_layout(B["host_rm"], ttnn.TILE_LAYOUT, memory_config=dram)
    if partial == "fabric":
        out = ttnn.add(b.out, host, output_tensor=b.out)
        ttnn.deallocate(host)
        return out
    out = ttnn.add(b.out, host, memory_config=dram)
    ttnn.deallocate(host)
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

    orig = list(x.shape)
    H = orig[-1]
    T = math.prod(orig[:-1])
    if self.num_devices > 1 and x.is_sharded():
        x = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG)
    x4 = x if (len(orig) == 4 and orig[2] == T) else ttnn.reshape(x, (1, 1, T, H))
    if getattr(self, "_hyb_fused", False) and mode != "prefill" and T == 1:
        return _forward_fused(self, x4, orig, partial, logits)
    if partial and HYB.enabled and HYB.C < HYB.E and T == 1:
        raise RuntimeError(f"layer {self.layer_num}: one-row decode boundary (partial={partial!r}) needs HYB_FUSED=1")
    scores, ids = self._route(x4)
    if not (HYB.enabled and HYB.C < HYB.E):
        routed = _routed(self, x4, scores, ids, T, mode)
        ttnn.deallocate(scores)
        ttnn.deallocate(ids)
        return _finish(self, _add(routed, self._shared(x4)), orig, partial)

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
            routed = ttnn.add(routed, _routed_dyn(self, x4, B["dsc"], B["dids"], T, mode),
                              memory_config=ttnn.DRAM_MEMORY_CONFIG)
        out = ttnn.add(routed, self._shared(x4), memory_config=ttnn.DRAM_MEMORY_CONFIG)
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
            routed = ttnn.add(routed, _routed_dyn(self, x4, B["dsc"], B["dids"], T, mode),
                              memory_config=ttnn.DRAM_MEMORY_CONFIG)
        out = ttnn.add(routed, self._shared(x4), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        out = ttnn.add(out, B["host"], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(scores)
        ttnn.deallocate(ids)
        return _finish(self, out, orig, partial)
    dec = _decide(self, x4, scores, ids, T, mode)
    t1 = time.perf_counter()
    n_ids = _upload_like(self, dec["ids"], ids, torch.int32)
    n_sc = _upload_like(self, dec["sc"], scores, torch.bfloat16)
    ttnn.deallocate(ids)
    ttnn.deallocate(scores)
    HYB.st["t_up"] += time.perf_counter() - t1
    routed = _routed(self, x4, n_sc, n_ids, T, mode)
    ttnn.deallocate(n_sc)
    ttnn.deallocate(n_ids)
    out = _add(routed, self._shared(x4))
    if bool(dec["host"].abs().sum() > 0):
        t0 = time.perf_counter()
        shp = [int(s) for s in out.shape]
        full = torch.zeros(shp, dtype=torch.float32)
        full.view(-1, shp[-1])[: dec["host"].shape[0]] = dec["host"] / self.num_devices
        hdev = ttnn.from_torch(full, dtype=out.dtype, layout=out.layout, device=self.device,
                               mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
                               memory_config=ttnn.DRAM_MEMORY_CONFIG)
        out = _add(out, hdev)
        HYB.st["t_up"] += time.perf_counter() - t0
    return _finish(self, out, orig, partial)


def _upload_like(self, t_host, ref, dtype_torch):
    return ttnn.from_torch(t_host.to(dtype_torch).reshape([int(s) for s in ref.shape]), dtype=ref.dtype,
                           layout=ref.layout, device=self.device, mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
                           memory_config=ttnn.DRAM_MEMORY_CONFIG)


def _bufs(self, T, scores, ids, x4):
    """Persistent device buffers for a decode width T (allocated in the eager warm-up, before any capture)."""
    B = self._hyb_bufs.get(T)
    if B is not None:
        return B
    rep = ttnn.ReplicateTensorToMesh(self.device)

    def like(ref, dtype_torch):
        return ttnn.from_torch(torch.zeros([int(s) for s in ref.shape], dtype=dtype_torch), dtype=ref.dtype,
                               layout=ref.layout, device=self.device, mesh_mapper=rep,
                               memory_config=ttnn.DRAM_MEMORY_CONFIG)

    B = {"ids": like(ids, torch.int32), "sc": like(scores, torch.bfloat16), "meta": (ids, scores)}
    if self._dyn_attrs is not None:
        B["dids"] = like(ids, torch.int32)
        B["dsc"] = like(scores, torch.bfloat16)
    H = int(x4.shape[-1])
    B["host"] = ttnn.from_torch(torch.zeros(1, 1, T, H), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                device=self.device, mesh_mapper=rep, memory_config=ttnn.DRAM_MEMORY_CONFIG)
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
    if "host_rm" in B:  # fused: one row-major row (decode, one valid row)
        put(B["host_rm"], (host[:1] / self.num_devices).reshape(1, 1, 1, sh["H"]).to(torch.bfloat16), ttnn.bfloat16,
            ttnn.ROW_MAJOR_LAYOUT)
        return
    full = torch.zeros(1, 1, sh["T"], sh["H"], dtype=torch.float32)
    full.view(-1, sh["H"])[: host.shape[0]] = host / self.num_devices  # per-device partial
    put(B["host"], full.to(torch.bfloat16), ttnn.bfloat16, ttnn.TILE_LAYOUT)


def _put(self, dst, t, dt, lay):
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
    job, B["job"] = B["job"], None
    if job is False:  # computed (and written) in _host_step
        return
    if job is None:  # no misses: the host partial must still be zeroed once
        if B.get("host_zero"):
            return
        host = torch.zeros(1, B["shapes"]["H"], dtype=torch.float32)
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
    d_slot = (
        torch.from_numpy(HYB.dyn_of[L]).long()[ic] if (use_dyn and mode != "prefill") else torch.full_like(ic, -1)
    )
    in_s, in_d = s_slot >= 0, (s_slot < 0) & (d_slot >= 0)
    miss = ~(in_s | in_d)
    v_miss = miss[:valid]
    HYB.st["picks"] += int(valid * k)
    HYB.st["hits"] += int(valid * k - v_miss.sum())
    if mode != "prefill":
        HYB.st["dec_picks"] += int(valid * k)
        HYB.st["dec_hits"] += int(valid * k - v_miss.sum())
    HYB.st["dyn_hits"] += int(in_d[:valid].sum())
    if mode == "prefill":
        HYB.st["prefill_rows"] += valid
        HYB.st["prefill_miss_pairs"] += int(v_miss.sum())
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
    if bool(v_miss.any()):
        t1 = time.perf_counter()
        xh = ttnn.to_torch(_die0(x4)).reshape(-1, x4.shape[-1])[:valid].to(torch.bfloat16)
        HYB.st["t_read"] += time.perf_counter() - t1
        job = (L, xh, idh[:valid], sch[:valid], v_miss)
        host = None if defer_host else HYB.host_compute(*job)
    else:
        job = None
        host = torch.zeros(valid, int(x4.shape[-1]), dtype=torch.float32)
    z = torch.zeros_like(s_slot)
    zf = torch.zeros_like(sch)
    return {"ids": torch.where(in_s, s_slot, z), "sc": torch.where(in_s, sch, zf),
            "dids": torch.where(in_d, d_slot, z), "dsc": torch.where(in_d, sch, zf), "host": host, "job": job}
