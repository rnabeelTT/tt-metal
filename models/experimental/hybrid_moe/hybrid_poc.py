# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Hybrid-MoE POC, step 3: Qwen3.5-122B-A10B on the QB2 with only a fraction of the routed experts treated as
device-resident; the rest are computed on the host CPU from the HF checkpoint and added to the device MoE output.

Shortcuts (it is a POC): eager decode (no trace); the device keeps all experts but a non-resident expert's routing
weight is zeroed, which is numerically what a cache miss does; host experts run in bf16 torch (the device uses bfp4).
Resident set = the globally most-routed (layer, expert) pairs from step 2's routing.npz.

Modes, one model build: baseline (no host) -> hybrid at each RESIDENT fraction. Per mode: the greedy tokens (compared
with baseline), the in-situ hit rate, and the per-layer host overhead (readback, host compute, upload+add).
"""

import json
import os
import sys
import time

import numpy as np
import torch
from safetensors import safe_open

import ttnn

os.environ.setdefault("HF_MODEL", "Qwen/Qwen3.5-122B-A10B")
from models.autoports.qwen_qwen3_5_122b_a10b.tt import fused_decoder as fd  # noqa: E402
from models.autoports.qwen_qwen3_5_122b_a10b.tt import moe as moe_mod  # noqa: E402
from models.autoports.qwen_qwen3_5_122b_a10b.tt.generator import build_generator  # noqa: E402

sys.path.insert(0, "/poc")
from capture_routing import MESH_PARAMS, PROMPTS  # noqa: E402

N_PROMPTS = int(os.environ.get("POC_PROMPTS", "4"))
NEW_TOKENS = int(os.environ.get("POC_NEW_TOKENS", "128"))
MODES = [(m.split(":")[0], float(m.split(":")[1])) for m in os.environ.get("POC_MODES", "static:0.5,dyn:0.5,dyn:0.3").split(",")]
E = 256
torch.set_num_threads(8)

# ---------------- host expert weights (lazy, mmapped safetensors) ----------------
SNAP = [d for d in os.scandir("/hf/hub/models--Qwen--Qwen3.5-122B-A10B/snapshots")][0].path
WMAP = json.load(open(f"{SNAP}/model.safetensors.index.json"))["weight_map"]
_files = {}


def _slice(key, e):
    fn = WMAP[key]
    if fn not in _files:
        _files[fn] = safe_open(f"{SNAP}/{fn}", framework="pt")
    return _files[fn].get_slice(key)[e]


_wcache = {}


def host_expert(layer, e):
    k = (layer, e)
    if k not in _wcache:
        p = f"model.language_model.layers.{layer}.mlp.experts."
        gu = _slice(p + "gate_up_proj", e).to(torch.bfloat16)  # [2N, H] = [gate; up]
        dn = _slice(p + "down_proj", e).to(torch.bfloat16)  # [H, N]
        _wcache[k] = (gu, dn)
    return _wcache[k]


# ---------------- state ----------------
S = {"resident": None, "lru": None, "misses": 0, "hits": 0, "picks": 0, "t_read": [], "t_host": [], "t_add": [], "pending": {}}
_depth = [0]
die0 = lambda t: ttnn.get_device_tensors(t)[0]  # noqa: E731


def wrap_route(cls):
    orig = cls._route

    def _route(self, x4):
        _depth[0] += 1
        try:
            scores, ids = orig(self, x4)
        finally:
            _depth[0] -= 1
        if _depth[0] or x4.shape[-2] > 32 or S["resident"] is None:
            return scores, ids
        mesh = self.device
        ttnn.synchronize_device(mesh)
        t0 = time.perf_counter()
        idh = ttnn.to_torch(die0(ids)).reshape(-1, self.top_k)[0].long()
        sch = ttnn.to_torch(die0(scores)).reshape(-1, self.top_k)[0].float()
        xh = ttnn.to_torch(die0(x4)).reshape(-1, x4.shape[-1])[0].to(torch.bfloat16)
        t1 = time.perf_counter()
        L = self.layer_num
        miss = [j for j in range(self.top_k) if (L, int(idh[j])) not in S["resident"]]
        S["picks"] += self.top_k
        S["hits"] += self.top_k - len(miss)
        S["t_read"].append(t1 - t0)
        S["misses"] += len(miss)
        lru = S["lru"]
        if lru is not None:
            for j in range(self.top_k):
                key = (L, int(idh[j]))
                if key in lru:
                    lru.move_to_end(key)
                else:  # the miss is fetched/promoted after this step
                    lru[key] = None
                    S["resident"].add(key)
                    if len(lru) > S["cap"]:
                        old, _ = lru.popitem(last=False)
                        S["resident"].discard(old)
        if not miss:
            return scores, ids
        ws = [host_expert(L, int(idh[j])) for j in miss]  # weight fetch (disk/page cache) excluded from timing
        t2 = time.perf_counter()
        acc = torch.zeros(x4.shape[-1], dtype=torch.float32)
        for j, (gu, dn) in zip(miss, ws):
            n = dn.shape[1]
            h = gu @ xh
            a = torch.nn.functional.silu(h[:n].float()) * h[n:].float()
            acc += sch[j] * (dn @ a.to(torch.bfloat16)).float()
        S["t_host"].append(time.perf_counter() - t2)
        # zero the missed experts' routing weights on device
        mask = torch.ones(list(scores.shape), dtype=torch.float32)
        mask[..., miss] = 0.0
        mdev = ttnn.from_torch(
            mask,
            dtype=scores.dtype,
            layout=scores.layout,
            device=mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            memory_config=scores.memory_config(),
        )
        masked = ttnn.multiply(scores, mdev, memory_config=scores.memory_config())
        ttnn.deallocate(scores)
        S["pending"][id(self)] = acc
        return masked, ids

    cls._route = _route


orig_forward = moe_mod.Qwen35MoE.forward


def forward(self, x, mode=None):
    out = orig_forward(self, x, mode)
    acc = S["pending"].pop(id(self), None)
    if acc is None:
        return out
    t0 = time.perf_counter()
    shp = list(out.shape)
    full = torch.zeros(shp[:-1] + [shp[-1] * self.num_devices], dtype=torch.float32)
    full.view(-1, full.shape[-1])[0] = acc
    hdev = ttnn.from_torch(
        full,
        dtype=out.dtype,
        layout=out.layout,
        device=self.device,
        mesh_mapper=ttnn.ShardTensorToMesh(self.device, dim=-1),
        memory_config=out.memory_config(),
    )
    res = ttnn.add(out, hdev, memory_config=out.memory_config())
    ttnn.synchronize_device(self.device)
    S["t_add"].append(time.perf_counter() - t0)
    return res


moe_mod.Qwen35MoE.forward = forward
wrap_route(moe_mod.Qwen35MoE)
wrap_route(fd.Qwen35MoEFused)


def resident_ranked(frac):
    z = np.load("/poc/routing/routing.npz")
    layers = list(z["layers"])
    f = np.zeros((len(layers), E))
    for k in z.files:
        if k.startswith("p") and int(k[1:]) >= N_PROMPTS:  # hot set from prompts NOT used here (no leakage)
            a = z[k]
            for li in range(len(layers)):
                np.add.at(f[li], a[:, li, :].ravel(), 1)
    budget = int(frac * len(layers) * E)
    flat = np.argsort(-f.ravel(), kind="stable")[:budget]
    return [(int(layers[i // E]), int(i % E)) for i in flat]


def main():
    out_dir = sys.argv[1]
    ttnn.set_fabric_config(MESH_PARAMS["fabric_config"])
    mesh = ttnn.open_mesh_device(
        ttnn.MeshShape(1, 4),
        l1_small_size=MESH_PARAMS["l1_small_size"],
        trace_region_size=MESH_PARAMS["trace_region_size"],
    )
    results = {}
    try:
        gen = build_generator(
            "models/autoports/qwen_qwen3_5_122b_a10b",
            mesh,
            max_seq_len=4096,
            warm_decode_trace=False,
            warm_prefill_trace=False,
        )
        tok = gen.tokenizer
        prompts = []
        for p in PROMPTS[:N_PROMPTS]:
            chat = tok.apply_chat_template(
                [{"role": "user", "content": p}], add_generation_prompt=True, enable_thinking=False, tokenize=False
            )
            prompts.append(tok.encode(chat, add_special_tokens=False))
        base = None
        import collections

        for mode in [None] + MODES:
            frac = None if mode is None else mode[1]
            name = "baseline_eager" if mode is None else f"{mode[0]}_{frac}"
            ranked = [] if mode is None else resident_ranked(frac)
            S.update(
                resident=None if mode is None else set(ranked),
                lru=collections.OrderedDict((k, None) for k in ranked) if mode and mode[0] == "dyn" else None,
                cap=len(ranked),
                misses=0,
                hits=0,
                picks=0,
                t_read=[],
                t_host=[],
                t_add=[],
            )
            outs, t0 = [], time.time()
            for ids in prompts:
                outs.append([int(t) for t in gen.generate(list(ids), NEW_TOKENS, enable_trace=False)])
            wall = time.time() - t0
            r = {"wall_s": round(wall, 1), "tokens": outs, "eager_ms_per_token": round(1e3 * wall / (len(prompts) * NEW_TOKENS), 2)}
            if frac is not None:
                ms = lambda v: round(1e3 * float(np.mean(v)), 4) if v else 0.0  # noqa: E731
                r.update(
                    hit_rate=round(S["hits"] / max(1, S["picks"]), 4),
                    misses_per_token=round(S["misses"] / (len(prompts) * NEW_TOKENS), 2),
                    layers_with_miss_frac=round(len(S["t_host"]) / max(1, len(S["t_read"])), 4),
                    ms_readback_per_layer=ms(S["t_read"]),
                    ms_host_per_missing_layer=ms(S["t_host"]),
                    ms_upload_add_per_missing_layer=ms(S["t_add"]),
                )
                agree = []
                for a, b in zip(base, outs):
                    first_div = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), len(a))
                    agree.append(first_div)
                r["first_divergence_per_prompt"] = agree
                r["texts"] = [tok.decode(o) for o in outs]
            else:
                base = outs
                r["texts"] = [tok.decode(o) for o in outs]
            results[name] = r
            print(name, {k: v for k, v in r.items() if k not in ("tokens", "texts")}, flush=True)
            json.dump(results, open(os.path.join(out_dir, "hybrid_poc.json"), "w"), indent=1)
        # traced decode baseline in the same build: per-token decode time from two generation lengths
        S.update(resident=None, lru=None)
        ids = prompts[0]
        gen.generate(list(ids), 8, enable_trace=True)  # capture + warm
        tt = {}
        for n in (16, 144):
            t0 = time.time()
            gen.generate(list(ids), n, enable_trace=True)
            tt[n] = time.time() - t0
        results["traced_baseline"] = {"ms_per_token": round(1e3 * (tt[144] - tt[16]) / 128, 3)}
        print("traced_baseline", results["traced_baseline"], flush=True)
        json.dump(results, open(os.path.join(out_dir, "hybrid_poc.json"), "w"), indent=1)
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()
