# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Device-format copies of the experts that can miss, for HYB_LRU fetches (hybrid/fetch.py): per (layer, expert) the
TP-sharded fused layout of the port (minimaxai_minimax_m2_7/tt/moe.py: gate|up packed per die as [up_d | gate_d],
down rows per die), rounded to 5 significant bits like the fused weights (moe_decode.fused.round_sig) and converted to
bfp4 per die; the 4 die shards are aggregated into one host tensor and dumped as <out>/<L>_<e>_gu.tensorbin /
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

    def get(key):
        def h(k):
            sh = wm[k]
            if sh not in handles:
                handles[sh] = safe_open(os.path.join(snap, sh), framework="pt")
            return handles[sh].get_tensor(k)

        t = h(key)
        return dequant(t, h(key + "_scale_inv")) if key + "_scale_inv" in wm else t.float()

    def dump(path, full, dim):
        n = full.shape[dim] // TP
        shards = [
            ttnn.from_torch(full.narrow(dim, d * n, n).contiguous(), dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT)
            for d in range(TP)
        ]
        tmp = path[: -len(".tensorbin")] + ".tmp.tensorbin"  # ttnn insists on the extension
        ttnn.dump_tensor(tmp, ttnn.from_host_shards(shards, ttnn.MeshShape(1, TP)))
        os.replace(tmp, path)

    port_sd = None
    if os.environ.get("HYB_LIB_KEYS", "").startswith("port:"):  # the port owns the checkpoint layout (as build_int4_library)
        import importlib

        port_sd = importlib.import_module(
            f"models.autoports.{os.environ['HYB_LIB_KEYS'][5:]}.tt.weights"
        ).StreamingStateDict(snap)

    for L in layers:
        t0 = time.time()
        ids = sorted(order[L, int(S) :].tolist())
        done = 0
        for e in ids:
            p = os.path.join(out, f"{L}_{e}")
            if os.path.exists(p + "_dn.tensorbin"):
                continue
            if port_sd is not None:
                gu, down = port_sd.one_expert(L, e)
                gu, down = gu.float(), down.float()
                gate, up = gu[: gu.shape[0] // 2], gu[gu.shape[0] // 2 :]
            else:
                pre = f"model.layers.{L}.block_sparse_moe.experts.{e}."
                gate, up, down = get(pre + "w1.weight"), get(pre + "w3.weight"), get(pre + "w2.weight")  # [N,H] [N,H] [H,N]
            N, H = gate.shape
            Nl = N // TP
            gate_t, up_t = gate.T, up.T  # [H, N]
            packed = torch.cat(
                [torch.cat([up_t[:, d * Nl : (d + 1) * Nl], gate_t[:, d * Nl : (d + 1) * Nl]], -1) for d in range(TP)],
                -1,
            )  # [H, TP * 2 Nl]
            dump(p + "_gu.tensorbin", round_sig(packed).reshape(1, 1, H, -1), -1)
            dump(p + "_dn.tensorbin", round_sig(down.T.contiguous()).reshape(1, 1, N, H), -2)
            done += 1
        handles.clear()
        print(f"layer {L}: {done} of {len(ids)} experts in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
