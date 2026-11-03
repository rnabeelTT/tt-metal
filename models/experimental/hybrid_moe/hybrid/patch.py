# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Runtime patches: C static expert slots (+ D dynamic slots) on device, host compute for missed experts.

Env (read at install):
  HYB_SLOTS        C, static expert slots per MoE layer (bfp4, the model's routed precision; default all E)
  HYB_DYN_SLOTS    D, dynamic expert slots per MoE layer (bfp8: ttnn.fill_cache cannot write bfp4); 0 = off
  HYB_PROMOTE      max promotions written per layer per decode step (default 2)
  HYB_PLAN         .npz with routing (routing.npz layout) -> static set = the C most-routed experts per layer
  HYB_LIB          int4 host expert library dir (build_int4_library.py); otherwise bf16 dequant from the checkpoint
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
        self.lib = {}
        self.lock = threading.Lock()
        self.pool = ThreadPoolExecutor(max_workers=int(os.environ.get("HYB_PREP_WORKERS", "2")))
        self.record = None
        self.enabled = True
        self.reset_stats()

    def reset_stats(self):
        self.st = {"picks": 0, "hits": 0, "dyn_hits": 0, "steps": 0, "promoted": 0, "t_read": 0.0, "t_host": 0.0,
                   "t_up": 0.0, "t_promote": 0.0, "prefill_rows": 0, "prefill_miss_pairs": 0}

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
                    for li, L in enumerate(layers):
                        np.add.at(f[L], a[:, li, :].ravel(), 1)
            self._freq = f
        if self._freq is not None and layer < self._freq.shape[0] and self._freq[layer].sum() > 0:
            ids = list(np.argsort(-self._freq[layer], kind="stable")[:C])
        else:
            ids = list(range(C))
        ids = [int(i) for i in ids]
        self.resident[layer] = ids
        so = np.full(E, -1, dtype=np.int32)
        so[ids] = np.arange(len(ids), dtype=np.int32)
        self.slot_of[layer] = so
        self.dyn_of[layer] = np.full(E, -1, dtype=np.int32)
        self.dyn_lru[layer] = OrderedDict()
        self.dyn_free[layer] = list(range(self.D))
        return ids

    # -- host experts ---------------------------------------------------------------------------------------------
    def lib_layer(self, layer):
        d = self.lib.get(layer)
        if d is None:
            d = torch.load(os.path.join(self.lib_dir, f"layer_{layer}.pt"), mmap=True, weights_only=True)
            self.lib[layer] = d
        return d

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
                h = torch._weight_int4pack_mm_for_cpu(xr, lib["gu_p"][e], gs, lib["gu_sz"][e])
                n = h.shape[1] // 2
                a = (torch.nn.functional.silu(h[:, :n].float()) * h[:, n:].float()).to(torch.bfloat16)
                y = torch._weight_int4pack_mm_for_cpu(a, lib["dn_p"][e], gs, lib["dn_sz"][e])
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
        args.num_experts = HYB.C
        try:
            orig_init(self, mesh_device, args, state_dict, tensor_cache_path, tt_ccl=tt_ccl, layer_num=layer_num)
        finally:
            args.num_experts = E
        self.num_experts = HYB.C
        self.E_full = E
        if int(self.router_w.shape[-1]) != E:
            raise RuntimeError(f"router width {self.router_w.shape} != E {E}")
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
        ttnn.copy_host_to_device_tensor(ttnn.from_torch(t, dtype=dt, layout=lay, mesh_mapper=rep), dst)

    put(B["ids"], dec["ids"].to(torch.int32).reshape(sh["ids"]), idt, ilay)
    put(B["sc"], dec["sc"].to(torch.bfloat16).reshape(sh["sc"]), sdt, slay)
    if "dids" in B:
        put(B["dids"], dec["dids"].to(torch.int32).reshape(sh["ids"]), idt, ilay)
        put(B["dsc"], dec["dsc"].to(torch.bfloat16).reshape(sh["sc"]), sdt, slay)
    full = torch.zeros(1, 1, sh["T"], sh["H"], dtype=torch.float32)
    full.view(-1, sh["H"])[: dec["host"].shape[0]] = dec["host"] / self.num_devices  # per-device partial
    put(B["host"], full.to(torch.bfloat16), ttnn.bfloat16, ttnn.TILE_LAYOUT)
    HYB.st["t_up"] += time.perf_counter() - t0


def _host_step(self, x4, scores, ids, T, mode, B):
    _write_bufs(self, B, _decide(self, x4, scores, ids, T, mode))


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


def _decide(self, x4, scores, ids, T, mode):
    """Read back the routing, apply pending promotions (decode), split picks into static / dynamic / host, compute
    the host part. Returns torch tensors: ids, sc (static), dids, dsc (dynamic), host [valid, H] fp32."""
    L = self.layer_num
    k = self.top_k
    t0 = time.perf_counter()
    idh = ttnn.to_torch(_die0(ids)).reshape(-1, k).long()
    sch = ttnn.to_torch(_die0(scores)).reshape(-1, k).float()
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
    HYB.st["dyn_hits"] += int(in_d[:valid].sum())
    if mode == "prefill":
        HYB.st["prefill_rows"] += valid
        HYB.st["prefill_miss_pairs"] += int(v_miss.sum())
    else:
        HYB.st["steps"] += 1
        if HYB.record is not None:
            HYB.record.append((L, idh[0].clone()))
        if use_dyn:
            lru = HYB.dyn_lru[L]
            for e in idh[:valid][in_d[:valid]].tolist():
                lru.move_to_end(e)
            for e in set(idh[:valid][v_miss].tolist()):
                if (L, e) not in HYB.pending:
                    HYB.pending[(L, e)] = HYB.pool.submit(HYB.prep, self._meta, L, e)
    if bool(v_miss.any()):
        t1 = time.perf_counter()
        xh = ttnn.to_torch(_die0(x4)).reshape(-1, x4.shape[-1])[:valid].to(torch.bfloat16)
        HYB.st["t_read"] += time.perf_counter() - t1
        host = HYB.host_compute(L, xh, idh[:valid], sch[:valid], v_miss)
    else:
        host = torch.zeros(valid, int(x4.shape[-1]), dtype=torch.float32)
    z = torch.zeros_like(s_slot)
    zf = torch.zeros_like(sch)
    return {"ids": torch.where(in_s, s_slot, z), "sc": torch.where(in_s, sch, zf),
            "dids": torch.where(in_d, d_slot, z), "dsc": torch.where(in_d, sch, zf), "host": host}
