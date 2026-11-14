# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""CPU emulation of the device's block-float expert weights, per MoE layer: the MoE output with experts quantised as
bfp4 / bfp8 (and int4-g32 as the host library computes them) vs the bf16 experts, on the HF fixtures' MoE inputs
(fixtures_inkling.py ... moe_all) with the fixtures' own routing. Much faster than the device sweep (no weight
conversion on device per policy); tools/sweep_moe_precision.py on a few layers checks that it tracks the device.

Block float as ttnn stores a [in, out] weight in 32x32 tiles: 16 consecutive values of a tile row (the out dimension, at
one in index) share an 8-bit exponent; bfp8 keeps sign + 7 mantissa bits, bfp4 sign + 3, round to nearest.

usage (host venv): python emulate_moe_precision.py <fixture dir> <out.json> [layers a-b] [T]
"""

import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
from models.autoports.thinkingmachines_inkling_small.tt.weights import StreamingStateDict  # noqa: E402

_build = sys.modules[__name__]


def bfp(w_out_in: torch.Tensor, mbits: int) -> torch.Tensor:
    """Block-float of a torch [out, in] weight as the device stores its transpose: blocks of 16 along `out`."""
    W = w_out_in.float().T.contiguous()  # [in, out] (device layout)
    I, O = W.shape
    B = W.reshape(I, O // 16, 16)
    e = torch.floor(torch.log2(B.abs().amax(-1, keepdim=True).clamp(min=1e-38)))
    step = torch.exp2(e - (mbits - 1))
    lim = 2 ** mbits - 1
    q = (torch.round(B / step).clamp(-lim, lim) * step).reshape(I, O)
    return q.T.contiguous()


def int4_g32(w_out_in: torch.Tensor, gs: int = 32) -> torch.Tensor:
    """The host library's asymmetric int4, groups of gs along `in` (build_int4_library.quant_pack), dequantised."""
    w = w_out_in.float()
    N, K = w.shape
    g = w.reshape(N, K // gs, gs)
    mx, mn = g.amax(-1, keepdim=True), g.amin(-1, keepdim=True)
    sc = (mx - mn).clamp(min=1e-6) / 15
    q = ((g - mn) / sc).round().clamp(0, 15)
    return (q * sc + mn).reshape(N, K)


QUANT = {"bf16": lambda w: w.float(), "bfp8": lambda w: bfp(w, 7), "bfp4": lambda w: bfp(w, 3), "int4": int4_g32,
         "gu4_dn8": None}


def moe_out(x, ids, w, experts, qgu, qdn):
    """sum_j w[t, j] * E_ids[t, j](x[t]) with quantisers qgu / qdn; experts: e -> (gate_up [2I, H], down [H, I])."""
    out = torch.zeros(x.shape[0], x.shape[1])
    cache = {}
    for e in ids.unique().tolist():
        gu, dn = experts(e)
        cache[e] = (qgu(gu), qdn(dn))
    for t in range(x.shape[0]):
        for j in range(ids.shape[1]):
            gu, dn = cache[int(ids[t, j])]
            g, u = (gu @ x[t]).chunk(2)
            out[t] += w[t, j] * (dn @ (torch.nn.functional.silu(g) * u))
    return out


def main():
    fx_dir, out_path = sys.argv[1], sys.argv[2]
    a, b = (sys.argv[3].split("-") if len(sys.argv) > 3 else ("2", "41"))
    T = int(sys.argv[4]) if len(sys.argv) > 4 else 128
    torch.set_num_threads(int(os.environ.get("THREADS", "16")))
    sd = StreamingStateDict()
    res = json.load(open(out_path)) if os.path.exists(out_path) else {}
    for li in range(int(a), int(b) + 1):
        if str(li) in res:
            continue
        fx = torch.load(os.path.join(fx_dir, f"layer{li}.pt"))
        x = fx["mlp_in"][:T].to(torch.bfloat16).float()
        # the port's routing (8 experts incl. the shared, weights normalised; expert_scale folded into down)
        rt = sd.ck.get(f"model.llm.layers.{li}.mlp.gate.weight").float()
        bias = sd.ck.get(f"model.llm.layers.{li}.mlp.gate.bias").float()
        lg = x @ rt.T
        picks = torch.topk(lg[:, :256].sigmoid() + bias, 6).indices
        ids = torch.cat([picks, torch.tensor([256, 257]).expand(T, -1)], 1)
        wt = lg.gather(1, ids).sigmoid()
        wt = wt / wt.sum(1, keepdim=True)
        experts = lambda e: sd.one_expert(li, e)  # noqa: E731
        ref = moe_out(x, ids, wt, experts, QUANT["bf16"], QUANT["bf16"])
        r = {}
        for name, (qg, qd) in {"bfp4": (QUANT["bfp4"],) * 2, "gu4_dn8": (QUANT["bfp4"], QUANT["bfp8"]),
                               "bfp8": (QUANT["bfp8"],) * 2, "int4": (QUANT["int4"],) * 2}.items():
            y = moe_out(x, ids, wt, experts, qg, qd)
            p = torch.nn.functional.cosine_similarity(y - y.mean(1, keepdim=True), ref - ref.mean(1, keepdim=True))
            r[name] = {"pcc_mean": p.mean().item(), "pcc_worst": p.min().item()}
        res[str(li)] = r
        print(f"layer {li:2d} " + "  ".join(f"{k} {v['pcc_mean']:.5f}" for k, v in r.items()), flush=True)
        json.dump(res, open(out_path, "w"), indent=1)


if __name__ == "__main__":
    main()
