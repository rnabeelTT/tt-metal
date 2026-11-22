# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Device-format copies of the experts that can miss, for HYB_LRU fetches (hybrid/fetch.py): per (layer, expert) the
TP-sharded fused layout of the port (minimaxai_minimax_m2_7/tt/moe.py: gate|up packed per die as [up_d | gate_d],
down rows per die), rounded to 5 significant bits like the fused weights (moe_decode.fused.round_sig; or first onto the bfp4 grid by
HYB_BFP4_ROUND, bfp4_round.py, like the port's EXPERT_ROUND_HOOK) and converted to bfp4 per die; the 4 die shards are aggregated into one host tensor and dumped as <out>/<L>_<e>_gu.tensorbin /
<L>_<e>_dn.tensorbin. ttnn's bfp4 packing needs the cluster (run in the image with the device attached; no mesh is
opened, but run it while no other job uses the device).

usage: python build_dev_experts.py <hf snapshot dir> <out dir> <plan.npz>:<S> [layers: a-b or a list a,b,c]
       (every expert outside each layer's top S of the plan's `order`)
"""

import json
import os
import sys
import time

import numpy as np
import torch
from safetensors import safe_open

import ttnn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bfp4_round import round_experts  # noqa: E402
from build_int4_library import dequant  # noqa: E402

TP, SIG_BITS = 4, 5


def round_sig(t, bits=SIG_BITS):
    drop = 24 - bits
    i = t.float().contiguous().view(torch.int32)
    i = (i + (1 << (drop - 1))) & ~((1 << drop) - 1)
    return i.view(torch.float32)


def main():
    snap, out, spec = sys.argv[1], sys.argv[2], sys.argv[3]
    os.makedirs(out, exist_ok=True)
    torch.set_num_threads(int(os.environ.get("THREADS", "8")))
    plan_path, S = spec.rsplit(":", 1)
    order = np.load(plan_path)["order"]
    wm = json.load(open(os.path.join(snap, "model.safetensors.index.json")))["weight_map"]
    layers = range(order.shape[0])
    if len(sys.argv) > 4 and "," in sys.argv[4]:  # an explicit list, e.g. the hybrid layers of a profile
        layers = sorted(int(v) for v in sys.argv[4].split(","))
    elif len(sys.argv) > 4:
        a, b = sys.argv[4].split("-")
        layers = range(int(a), int(b) + 1)
    handles = {}
    # checkpoint key style (as build_int4_library.py HYB_LIB_KEYS): (prefix, (gate, up, down) names)
    style = os.environ.get("HYB_LIB_KEYS", "minimax_m2")
    pre_fmt, names = {
        "minimax_m2": ("model.layers.{L}.block_sparse_moe.experts.{e}.", ("w1", "w3", "w2")),
        "qwen35": ("model.language_model.layers.{L}.mlp.experts.{e}.", ("gate_proj", "up_proj", "down_proj")),
    }[style]

    def get(key):
        def h(k):
            sh = wm[k]
            if sh not in handles:
                handles[sh] = safe_open(os.path.join(snap, sh), framework="pt")
            return handles[sh].get_tensor(k)

        t = h(key)
        return dequant(t, h(key + "_scale_inv")) if key + "_scale_inv" in wm else t.float()

    # per-layer dtypes from the served precision policy (QWEN35_PRECISION_CONFIG: weight_groups.routed_moe gate_up /
    # down, layer_exceptions): a fetched copy must be byte-compatible with the slot it lands in (e.g. Inkling's fast
    # profile: bfp8 down projections in layers 2-17). Without a policy: bfp4 everywhere (the M2.7 package).
    # HYB_DEV_POLICY (else QWEN35_PRECISION_CONFIG): the copies' format. Prefill streaming stages every layer's experts in
    # one model-wide staging tensor in the first hybrid layer's format, so a mixed-dtype profile builds its copies in
    # that one format (Inkling's fast profile: all_gu4dn8.json); the decode fetch writes into the layers' own slots
    pp = os.environ.get("HYB_DEV_POLICY") or os.environ.get("QWEN35_PRECISION_CONFIG")
    pol = json.load(open(pp)) if pp else {}
    DT = {"bfp4": ttnn.bfloat4_b, "bfp8": ttnn.bfloat8_b}

    def layer_dtype(L, which):
        r = pol.get("weight_groups", {}).get("routed_moe", {})
        r = {**r, **pol.get("layer_exceptions", {}).get(str(L), {}).get("routed_moe", {})}
        return DT[r.get(which, "bfp4")]

    def dump(path, full, dim, dtype=ttnn.bfloat4_b):
        n = full.shape[dim] // TP
        shards = [
            ttnn.from_torch(full.narrow(dim, d * n, n).contiguous(), dtype=dtype, layout=ttnn.TILE_LAYOUT)
            for d in range(TP)
        ]
        tmp = path[: -len(".tensorbin")] + ".tmp.tensorbin"  # ttnn insists on the extension
        ttnn.dump_tensor(tmp, ttnn.from_host_shards(shards, ttnn.MeshShape(1, TP)))
        os.replace(tmp, path)

    port_sd = None
    if os.environ.get("HYB_LIB_KEYS", "").startswith(
        "port:"
    ):  # the port owns the checkpoint layout (as build_int4_library)
        import importlib

        port_sd = importlib.import_module(
            f"models.autoports.{os.environ['HYB_LIB_KEYS'][5:]}.tt.weights"
        ).StreamingStateDict(snap)

    # HYB_DEV_PARTS: which copies to write (default "gu,dn"; "dn": the own-format downs of HYB_FETCH_DN_EXPERTS, whose
    # gate/up the main copies already hold)
    parts = set(os.environ.get("HYB_DEV_PARTS", "gu,dn").split(","))
    def one(L, e):  # (gate [N,H], up [N,H], down [H,N]) of one expert
        if port_sd is not None:  # the port owns the checkpoint layout
            gu, down = port_sd.one_expert(L, e)
            gu, down = gu.float(), down.float()
            return gu[: gu.shape[0] // 2], gu[gu.shape[0] // 2 :], down
        pre = pre_fmt.format(L=L, e=e)
        return tuple(get(pre + n + ".weight") for n in names)

    for L in layers:
        t0 = time.time()
        ids = [
            e
            for e in sorted(order[L, int(S) :].tolist())
            if not os.path.exists(os.path.join(out, f"{L}_{e}_dn.tensorbin"))
        ]
        rounded = None
        if ids and os.environ.get("HYB_BFP4_ROUND"):  # the layer's experts round together (the GPTQ methods batch them)
            w = [one(L, e) for e in ids]
            gu = torch.stack([torch.cat([g, u]) for g, u, _ in w])  # [n, 2N, H]
            dn = torch.stack([d for _, _, d in w])  # [n, H, N]
            del w
            qg = round_experts(L, "gu", gu, experts=ids)  # grid values, or None (the default rounding)
            qd = round_experts(L, "dn", dn, experts=ids, gate_up=gu)
            gu, dn = (gu if qg is None else qg), (dn if qd is None else qd)
            N = gu.shape[1] // 2
            rounded = lambda j: (gu[j, :N], gu[j, N:], dn[j])  # noqa: E731
        for j, e in enumerate(ids):  # without HYB_BFP4_ROUND one expert at a time (a layer of fp32 experts is many GB)
            p = os.path.join(out, f"{L}_{e}")
            gate, up, down = rounded(j) if rounded else one(L, e)  # [N,H] [N,H] [H,N]
            N, H = gate.shape
            Nl = N // TP
            gate_t, up_t = gate.T, up.T  # [H, N]
            packed = torch.cat(
                [torch.cat([up_t[:, d * Nl : (d + 1) * Nl], gate_t[:, d * Nl : (d + 1) * Nl]], -1) for d in range(TP)],
                -1,
            )  # [H, TP * 2 Nl]
            if "gu" in parts:
                dump(p + "_gu.tensorbin", round_sig(packed).reshape(1, 1, H, -1), -1, layer_dtype(L, "gate_up"))
            dump(p + "_dn.tensorbin", round_sig(down.T.contiguous()).reshape(1, 1, N, H), -2, layer_dtype(L, "down"))
        handles.clear()
        print(f"layer {L}: {len(ids)} experts in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
