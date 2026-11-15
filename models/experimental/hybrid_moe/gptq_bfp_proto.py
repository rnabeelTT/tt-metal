# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Prototype: GPTQ-style error-compensated rounding into the device's bfp4 grid (same storage format, chosen values),
vs round-to-nearest bfp4 / gu4-dn8 / bfp8, on one MoE layer's fixture inputs (emulate_moe_precision.py's MoE output PCC
against bf16 experts). Calibrates on the first CAL tokens (layer-level input Hessians), scores the rest.

Device layout: a [in, out] weight in tiles; 16 consecutive outputs at one input index share an exponent; bfp4 keeps
sign + 3 mantissa bits. GPTQ walks the input dimension: at input k, the row W[k, :] is rounded to its blocks' grid
(exponent from the current, error-updated values) and the rounding error is pushed onto the inputs not yet rounded
through the inverse Hessian's Cholesky factor (the standard GPTQ update), in blocks of 128 inputs.

usage (host venv): python gptq_bfp_proto.py <fixture dir> <layer> [CAL=384]
"""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from emulate_moe_precision import QUANT, bfp, moe_out  # noqa: E402

from models.autoports.thinkingmachines_inkling_small.tt.weights import StreamingStateDict  # noqa: E402


def round_row(r, mbits):
    """r [out] -> its bfp grid values (blocks of 16 along out)."""
    B = r.reshape(-1, 16)
    e = torch.floor(torch.log2(B.abs().amax(-1, keepdim=True).clamp(min=1e-38)))
    step = torch.exp2(e - (mbits - 1))
    lim = 2 ** mbits - 1
    return (torch.round(B / step).clamp(-lim, lim) * step).reshape(-1)


def gptq_bfp(w_out_in, H, mbits=3, damp=0.01, blk=128):
    """GPTQ into the bfp grid: w [out, in], H [in, in] (X^T X of the inputs) -> the rounded [out, in]."""
    W = w_out_in.float().T.contiguous().clone()  # [in, out]: one row per input index
    n = W.shape[0]
    H = H.clone().double()
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    W[dead] = 0
    H += damp * torch.mean(torch.diag(H)) * torch.eye(n, dtype=H.dtype)
    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True).float()
    Q = torch.zeros_like(W)
    for i1 in range(0, n, blk):
        i2 = min(i1 + blk, n)
        W1 = W[i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        Err = torch.zeros_like(W1)
        Hi = Hinv[i1:i2, i1:i2]
        for i in range(i2 - i1):
            q = round_row(W1[i], mbits)
            Q1[i] = q
            err = (W1[i] - q) / Hi[i, i]
            W1[i:] -= torch.outer(Hi[i, i:], err)
            Err[i] = err
        Q[i1:i2] = Q1
        W[i2:] -= Hinv[i1:i2, i2:].T @ Err
    return Q.T.contiguous()


def hessians(calib_dir, li, routing, j):
    """(layer-level X^T X / n, {expert: (X_e^T X_e, n_e)}, X) from the calibration dump (ref_inkling REF_MLPIN)."""
    X = torch.load(os.path.join(calib_dir, f"mlpin_L{li}.pt")).reshape(-1, 4096).float()
    ids = torch.from_numpy(routing["ids"][:, j].reshape(-1, routing["ids"].shape[-1]).astype("int64"))
    return X, ids


def expert_H(X, ids, e, Hl, nmin=512):
    """Expert e's input Hessian: its routed tokens' X^T X, blended with the layer's when it saw few tokens."""
    if e >= 256:  # shared experts: every token
        return Hl * X.shape[0]
    rows = (ids == e).any(-1).nonzero().reshape(-1)
    Xe = X[rows].double()
    He = Xe.T @ Xe
    n = rows.numel()
    if n < nmin:
        He = He + Hl * (nmin - n)
    return He


def main():
    fx_dir, li = sys.argv[1], int(sys.argv[2])
    calib = os.environ.get("GPTQ_CALIB")
    CAL = int(sys.argv[3]) if len(sys.argv) > 3 else 384
    torch.set_num_threads(int(os.environ.get("THREADS", "8")))
    sd = StreamingStateDict()
    fx = torch.load(os.path.join(fx_dir, f"layer{li}.pt"))
    xa = fx["mlp_in"].to(torch.bfloat16).float()
    rt = sd.ck.get(f"model.llm.layers.{li}.mlp.gate.weight").float()
    bias = sd.ck.get(f"model.llm.layers.{li}.mlp.gate.bias").float()

    def route(x):
        lg = x @ rt.T
        picks = torch.topk(lg[:, :256].sigmoid() + bias, 6).indices
        ids = torch.cat([picks, torch.tensor([256, 257]).expand(x.shape[0], -1)], 1)
        wt = lg.gather(1, ids).sigmoid()
        return ids, wt / wt.sum(1, keepdim=True)

    if calib:  # the calibration dump: every token of the training texts; score on the whole fixture text
        import numpy as np

        rz = np.load(os.path.join(calib, "..", "ref", "train_calib", "routing.npz")) if os.path.exists(
            os.path.join(calib, "..", "ref", "train_calib", "routing.npz")) else np.load(os.environ["GPTQ_ROUTING"])
        j = list(rz["layers"]).index(li)
        Xc, ids_c = hessians(calib, li, rz, j)
        xe = xa
        print(f"calibration: {Xc.shape[0]} tokens", flush=True)
    else:
        Xc, xe = xa[:CAL], xa[CAL:]
        ids_c, _ = route(Xc)
    ids_e, wt_e = route(xe)
    Hl = (Xc.T.double() @ Xc.double()) / Xc.shape[0]
    experts = lambda e: sd.one_expert(li, e)  # noqa: E731
    need = ids_e.unique().tolist()
    gq = {}
    for e in need:  # GPTQ per expert: gate/up on its routed inputs, down on its activations of them
        gu, dn = experts(e)
        gu, dn = gu.float(), dn.float()
        He = expert_H(Xc, ids_c, e, Hl)
        gq_gu = gptq_bfp(gu, He)
        rows = Xc if e >= 256 else Xc[(ids_c == e).any(-1)]
        if rows.shape[0] < 64:
            rows = Xc[:4096]
        g, u = (rows @ gu.T).chunk(2, dim=-1)
        a = torch.nn.functional.silu(g) * u
        gq_dn = gptq_bfp(dn, a.T.double() @ a.double())
        gq[e] = (gq_gu, gq_dn)
    ref = moe_out(xe, ids_e, wt_e, experts, QUANT["bf16"], QUANT["bf16"])
    def pcc(y):
        p = torch.nn.functional.cosine_similarity(y - y.mean(1, keepdim=True), ref - ref.mean(1, keepdim=True))
        return p.mean().item(), p.min().item()

    rows = {
        "bfp4 (RTN)": moe_out(xe, ids_e, wt_e, experts, QUANT["bfp4"], QUANT["bfp4"]),
        "gu4_dn8 (RTN)": moe_out(xe, ids_e, wt_e, experts, QUANT["bfp4"], QUANT["bfp8"]),
        "bfp8": moe_out(xe, ids_e, wt_e, experts, QUANT["bfp8"], QUANT["bfp8"]),
        "bfp4 GPTQ": moe_out(xe, ids_e, wt_e, lambda e: gq[e], lambda w: w, lambda w: w),
    }
    for k, y in rows.items():
        m, w = pcc(y)
        print(f"layer {li} {k:14s} pcc mean {m:.5f} worst {w:.5f}", flush=True)


if __name__ == "__main__":
    main()
