# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Error-aware rounding of the routed experts to the device's bfp4 grid (HYB_BFP4_ROUND), ahead of ttnn's packer.

bfp4_b as the experts are stored: per input row of the [in, out] device layout, 16 consecutive output values share one
exponent e; each value is sign + 3-bit magnitude m in 0..7, worth m * 2^(e - 2). The default conversion rounds each
weight to 5 significant bits (moe_decode.fused.round_sig) and lets ttnn's packer pick e from the group's largest value
and round to nearest. Values already on that grid pack exactly (checked on a p300 board, ~/m27-pkg/bfp4q/t_pack.py), so
any rounding that returns grid values changes only which grid point each weight takes:

    rtne     per group, e = the largest value's exponent or one less (that value then clips to 7 steps), whichever has
             the smaller squared error; round to nearest within it.
    gptq<d>  GPTQ over the input columns with a calibration Hessian H = sum x x^T (damping d * mean diag H); each column
             is rounded as above and its error is pushed into the columns not yet rounded. Needs <dir>/H_<L>_<kind>.pt.

HYB_BFP4_ROUND: unset / "" -> the default conversion; "rtne" -> rtne everywhere; a .json path -> {"default": method,
"layers": {"<L>": {"gu": method, "dn": method}}} with HYB_BFP4_H=<dir> holding the Hessians gptq methods need.
Weights are [E, out, in] (gate|up: [E, 2N, H]; down: [E, H, N]): groups run along `out`, as in the device layout.
"""

from __future__ import annotations

import hashlib
import json
import os

import torch


def _cfg():
    spec = os.environ.get("HYB_BFP4_ROUND", "")
    if not spec:
        return None
    if spec.endswith(".json"):
        with open(spec) as f:
            c = json.load(f)
    else:
        c = {"default": spec, "layers": {}}
    c["hdir"] = os.environ.get("HYB_BFP4_H", "")
    return c


def method(layer, kind, cfg=None):
    """The method for (layer, 'gu' | 'dn'), or None for the default conversion."""
    cfg = cfg if cfg is not None else _cfg()
    if cfg is None:
        return None
    return cfg["layers"].get(str(layer), {}).get(kind, cfg["default"])


def tag(layer, cfg=None):
    """Cache-key suffix of a layer's rounding ('' for the default conversion): a changed method or Hessian is a new key."""
    cfg = cfg if cfg is not None else _cfg()
    if cfg is None:
        return ""
    ms = (method(layer, "gu", cfg), method(layer, "dn", cfg))
    if all(m is None or m == "default" for m in ms):
        return ""
    h = hashlib.sha1(repr(ms).encode())
    for kind, m in zip(("gu", "dn"), ms):
        if m and m.startswith("gptq"):
            p = os.path.join(cfg["hdir"], f"H_{layer}_{kind}.pt")
            st = os.stat(p)
            h.update(f"{p}:{st.st_size}:{int(st.st_mtime)}".encode())
    return ".q" + h.hexdigest()[:10]


def q_groups(v, search=True):
    """v [..., 16] -> the nearest bfp4 grid values (search: also the exponent one lower, per group, least squared
    error)."""
    e = torch.floor(torch.log2(v.abs().amax(-1, keepdim=True).clamp_min(1e-30)))
    best = best_err = None
    for de in (0, -1) if search else (0,):
        step = torch.pow(2.0, e + de - 2)
        q = torch.clamp(torch.round(v / step), -7, 7) * step
        err = ((q - v) ** 2).sum(-1, keepdim=True)
        if best is None:
            best, best_err = q, err
        else:
            better = err < best_err
            best = torch.where(better, q, best)
            best_err = torch.where(better, err, best_err)
    return best


def rtne(W):
    """W [..., out, in] -> grid values, groups of 16 along out."""
    O, I = W.shape[-2], W.shape[-1]
    Wt = W.float().transpose(-1, -2).reshape(*W.shape[:-2], I, O // 16, 16)
    return q_groups(Wt).reshape(*W.shape[:-2], I, O).transpose(-1, -2).contiguous()


def gptq(W, H, damp, block=128):
    """W [rows, in] (rows: any number of `out` groups of 16, e.g. many experts stacked) with one shared H [in, in].
    Works on W transposed, so the column being rounded and the ones its error updates are contiguous rows (2x faster
    than the row-major loop, same values)."""
    Wt = W.float().t().contiguous()  # [in, rows]
    H = H.double().clone()
    n = H.shape[0]
    dead = torch.diag(H) == 0
    if dead.all():
        raise ValueError(f"gptq: the Hessian is all zero ({n} columns): no calibration tokens reached this weight")
    H[dead, dead] = 1
    Wt[dead] = 0
    H += damp * torch.mean(torch.diag(H)) * torch.eye(n, dtype=H.dtype)
    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True).float()
    Qt = torch.empty_like(Wt)
    R = Wt.shape[1]
    for i1 in range(0, n, block):
        i2 = min(i1 + block, n)
        W1 = Wt[i1:i2].clone()
        Err1 = torch.empty_like(W1)
        Hi = Hinv[i1:i2, i1:i2]
        for i in range(i2 - i1):
            w = W1[i]
            q = q_groups(w.view(R // 16, 16)).view(R)
            Qt[i1 + i] = q
            err = (w - q) / Hi[i, i]
            W1[i:] -= Hi[i, i:, None] * err[None]
            Err1[i] = err
        Wt[i2:] -= Hinv[i1:i2, i2:].t() @ Err1
    return Qt.t()


def round_experts(layer, kind, W, cfg=None, chunk=32):
    """W [E, out, in] -> the layer's rounding of it as grid values [E, out, in] in W's dtype (bf16 holds every bfp4 grid
    value exactly), or None for the default."""
    cfg = cfg if cfg is not None else _cfg()
    m = method(layer, kind, cfg)
    if m is None or m == "default":
        return None
    if m == "rtne":
        return torch.cat([rtne(W[e0 : e0 + chunk]).to(W.dtype) for e0 in range(0, W.shape[0], chunk)])
    if m.startswith("gptq"):
        damp = float(m[4:])
        H = torch.load(os.path.join(cfg["hdir"], f"H_{layer}_{kind}.pt"))
        E, O, I = W.shape
        out = []
        for e0 in range(0, E, chunk):
            Wc = W[e0 : e0 + chunk].float().reshape(-1, I)  # shared H: experts stack along the rows
            out.append(gptq(Wc, H, damp).reshape(-1, O, I).to(W.dtype))
        return torch.cat(out)
    raise ValueError(f"HYB_BFP4_ROUND: unknown method {m!r} for layer {layer} {kind}")
