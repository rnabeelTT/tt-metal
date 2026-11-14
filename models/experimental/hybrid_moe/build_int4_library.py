# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Build the host int4 expert library for the hybrid prototype.

For every MoE layer and expert of the checkpoint: dequantize (FP8 block or bf16) -> group-wise int4 (asymmetric,
group GS along K) in torch's CPU tinygemm layout (_convert_weight_to_int4pack_for_cpu), so the host computes
    h = _weight_int4pack_mm_for_cpu(x, gu_p, GS, gu_sz)   # x @ [gate; up]^T  -> [r, 2N]
    y = _weight_int4pack_mm_for_cpu(a, dn_p, GS, dn_sz)   # a @ down^T        -> [r, H]
One file per layer: <out>/layer_<L>.pt with gu_p, gu_sz, dn_p, dn_sz stacked over experts.

usage: build_int4_library.py <hf snapshot dir> <out dir> [layers: a range e.g. 0-59, or a list e.g. 0,1,3]
env: HYB_LIB_KEYS qwen35 (default) | minimax_m2 (block_sparse_moe.experts.E.w1/w3/w2 = gate/up/down);
     HYB_LIB_COLD=<plan.npz>:<C> builds only the experts outside each layer's top C of the plan's `order` and
     stores their ids as "experts" (row i = expert experts[i]); otherwise every expert, row = expert id.
"""

import json
import os
import re
import sys
import time

import numpy as np
import torch
from safetensors import safe_open

GS = int(os.environ.get("HYB_INT4_GS", "32"))  # hostk needs gs dividing 64 (32 / 64); the shipped libraries are 32
BLOCK = 128


def dequant(w, s):
    O, I = w.shape
    sf = s.float().repeat_interleave(BLOCK, 0)[:O].repeat_interleave(BLOCK, 1)[:, :I]
    return w.float() * sf


def quant_pack(w_out_in):
    """w [out, in] (fp32) -> (packed, scales_and_zeros) for x @ w.T with _weight_int4pack_mm_for_cpu."""
    w = w_out_in  # [N, K] in torch-test terms (b^T); groups along K
    N, K = w.shape
    g = w.reshape(N, K // GS, GS)
    mx = g.amax(dim=2, keepdim=True)
    mn = g.amin(dim=2, keepdim=True)
    scales = (mx - mn).clamp(min=1e-6) / 15
    zeros = mn + scales * 8
    q = ((g - mn) / scales).round().clamp_(0, 15).to(torch.int32).reshape(N, K)
    packed = torch._convert_weight_to_int4pack_for_cpu(q, 2)
    sz = torch.cat([scales.reshape(N, K // GS, 1), zeros.reshape(N, K // GS, 1)], 2).transpose(0, 1).contiguous()
    return packed, sz.to(torch.bfloat16)


def main():
    snap, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    torch.set_num_threads(int(os.environ.get("THREADS", "8")))
    wm = json.load(open(os.path.join(snap, "model.safetensors.index.json")))["weight_map"]
    style = os.environ.get("HYB_LIB_KEYS", "qwen35")
    if style == "minimax_m2":
        pat = re.compile(r"^model\.layers\.(\d+)\.block_sparse_moe\.experts\.(\d+)\.(w1|w2|w3)\.weight$")
        pre, names = "model.layers.{L}.block_sparse_moe.experts.{e}.", ("w1", "w3", "w2")
    elif style == "qwen35":
        pat = re.compile(
            r"^model\.language_model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$"
        )
        pre, names = "model.language_model.layers.{L}.mlp.experts.{e}.", ("gate_proj", "up_proj", "down_proj")
    else:
        raise ValueError(f"HYB_LIB_KEYS={style}: expected qwen35 or minimax_m2")
    cold = None
    if os.environ.get("HYB_LIB_COLD"):
        plan_path, C = os.environ["HYB_LIB_COLD"].rsplit(":", 1)
        order = np.load(plan_path)["order"]
        cold = {L: sorted(order[L, int(C) :].tolist()) for L in range(order.shape[0])}
    experts = {}
    for k in wm:
        m = pat.match(k)
        if m:
            experts.setdefault(int(m.group(1)), set()).add(int(m.group(2)))
    layers = sorted(experts)
    if len(sys.argv) > 3 and "," in sys.argv[3]:  # an explicit list, e.g. the hybrid layers of a profile
        keep = {int(v) for v in sys.argv[3].split(",")}
        layers = [l for l in layers if l in keep]
    elif len(sys.argv) > 3:
        a, b = sys.argv[3].split("-")
        layers = [l for l in layers if int(a) <= l <= int(b)]
    handles = {}

    def get(key):
        sh = wm[key]
        if sh not in handles:
            handles[sh] = safe_open(os.path.join(snap, sh), framework="pt")
        t = handles[sh].get_tensor(key)
        sk = key + "_scale_inv"
        if sk in wm:
            sh2 = wm[sk]
            if sh2 not in handles:
                handles[sh2] = safe_open(os.path.join(snap, sh2), framework="pt")
            return dequant(t, handles[sh2].get_tensor(sk))
        return t.float()

    for L in layers:
        path = os.path.join(out, f"layer_{L}.pt")
        if os.path.exists(path):
            continue
        t0 = time.time()
        ids = cold[L] if cold is not None else list(range(max(experts[L]) + 1))
        gu_p, gu_sz, dn_p, dn_sz = [], [], [], []
        for e in ids:
            p = pre.format(L=L, e=e)
            gate, up, down = (get(p + n + ".weight") for n in names)
            a, b = quant_pack(torch.cat([gate, up], 0))  # [2N, H]
            c, d = quant_pack(down)  # [H, N]
            gu_p.append(a)
            gu_sz.append(b)
            dn_p.append(c)
            dn_sz.append(d)
        torch.save(
            {
                "gu_p": torch.stack(gu_p),
                "gu_sz": torch.stack(gu_sz),
                "dn_p": torch.stack(dn_p),
                "dn_sz": torch.stack(dn_sz),
                "gs": GS,
                **({"experts": torch.tensor(ids)} if cold is not None else {}),
            },
            path + ".tmp",
        )
        os.replace(path + ".tmp", path)
        handles.clear()
        print(f"layer {L}: {len(ids)} experts in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
