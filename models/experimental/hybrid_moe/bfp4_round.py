# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Error-aware rounding of the routed experts to the device's bfp4 grid (HYB_BFP4_ROUND), ahead of ttnn's packer.

bfp4_b as the experts are stored: per input row of the [in, out] device layout, 16 consecutive output values share one
exponent e; each value is sign + 3-bit magnitude m in 0..7, worth m * 2^(e - 2). The default conversion rounds each
weight to 5 significant bits (moe_decode.fused.round_sig) and lets ttnn's packer pick e from the group's largest value
and round to nearest. Values already on that grid pack exactly (checked on a p300 board, ~/m27-pkg/bfp4q/t_pack.py), so
any rounding that returns grid values changes only which grid point each weight takes:

    rtne      per group, e = the largest value's exponent or one less (that value then clips to 7 steps), whichever
              has the smaller squared error; round to nearest within it.
    gptq<d>   GPTQ over the input columns with the layer's calibration Hessian H = sum x x^T over every calibration
              token (damping d * mean diag H); each column is rounded as above and its error is pushed into the
              columns not yet rounded. gate|up only (its input is the layer's MoE input, shared by all experts).
    gptqe<d>  GPTQ with the expert's own Hessian: the calibration tokens routed to it (gate|up: their MoE inputs; down:
              silu(x g^T) * (x u^T) through the expert's unrounded gate|up). rtne for an expert with fewer than
              MIN_TOK routed tokens.

HYB_BFP4_ROUND: unset / "" -> the default conversion; "rtne" -> rtne everywhere; a .json path -> {"default": method or
{"gu": method, "dn": method}, "layers": {"<L>": {"gu": method, "dn": method}}}. gptq methods read the calibration
capture HYB_BFP4_CAL=<dir>/cal_<L>.pt: {"h": [T, H] the layer's MoE inputs, "idx": [T, k] the routed expert ids}.
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
    c["cal"] = os.environ.get("HYB_BFP4_CAL", "")
    return c


def method(layer, kind, cfg=None):
    """The method for (layer, 'gu' | 'dn'), or None for the default conversion."""
    cfg = cfg if cfg is not None else _cfg()
    if cfg is None:
        return None
    d = cfg["default"]
    return cfg["layers"].get(str(layer), {}).get(kind, d[kind] if isinstance(d, dict) else d)


def uses_cal(cfg=None):
    """Whether any layer's method needs the calibration capture (a gptq method)."""
    cfg = cfg if cfg is not None else _cfg()
    if cfg is None:
        return False
    d = cfg["default"]
    ms = list(d.values()) if isinstance(d, dict) else [d]
    ms += [m for kinds in cfg["layers"].values() for m in kinds.values()]
    return any(m.startswith("gptq") for m in ms)


def tag(layer, cfg=None):
    """Cache-key suffix of a layer's rounding ('' for the default conversion): a changed method or Hessian is a new key."""
    cfg = cfg if cfg is not None else _cfg()
    if cfg is None:
        return ""
    ms = (method(layer, "gu", cfg), method(layer, "dn", cfg))
    if all(m is None or m == "default" for m in ms):
        return ""
    h = hashlib.sha1(repr(ms).encode())
    if any(m and m.startswith("gptq") for m in ms):
        p = _cal_path(layer, cfg)
        st = os.stat(p)
        h.update(f"{p}:{st.st_size}:{int(st.st_mtime)}".encode())
    return ".q" + h.hexdigest()[:10]


MIN_TOK = 64
_CAL = {}


def _cal_path(layer, cfg):
    if not cfg["cal"]:
        raise ValueError(
            f"HYB_BFP4_ROUND: layer {layer} uses a gptq method, but HYB_BFP4_CAL (the capture dir) is unset"
        )
    return os.path.join(cfg["cal"], f"cal_{layer}.pt")


def _cal(layer, cfg):
    """(h [T, H] float, idx [T, k] long) of the layer's calibration capture (the last layer's stays loaded)."""
    p = _cal_path(layer, cfg)
    if _CAL.get("path") != p:
        c = torch.load(p)
        _CAL.clear()
        _CAL.update(path=p, h=c["h"].float(), idx=c["idx"].long())
    return _CAL["h"], _CAL["idx"]


def q_groups(v, search=True):
    """v [..., 16] -> the nearest bfp4 grid values (search: also the exponent one lower, per group, least squared
    error)."""
    _, ex = torch.frexp(v.abs().amax(-1, keepdim=True))  # amax = m * 2^ex, m in [0.5, 1): floor(log2 amax) = ex - 1
    step = torch.ldexp(torch.ones_like(v[..., :1]), ex - 3)  # 2^(e - 2); an all-zero group rounds to 0 either way
    q = torch.clamp(torch.round(v / step), -7, 7) * step
    if not search:
        return q
    step = step * 0.5
    q1 = torch.clamp(torch.round(v / step), -7, 7) * step
    better = ((q1 - v) ** 2).sum(-1, keepdim=True) < ((q - v) ** 2).sum(-1, keepdim=True)
    return torch.where(better, q1, q)


def rtne(W):
    """W [..., out, in] -> grid values, groups of 16 along out."""
    O, I = W.shape[-2], W.shape[-1]
    Wt = W.float().transpose(-1, -2).reshape(*W.shape[:-2], I, O // 16, 16)
    return q_groups(Wt).reshape(*W.shape[:-2], I, O).transpose(-1, -2).contiguous()


def gptq(W, H, damp, block=256, sub=32):
    """GPTQ of W [E, rows, in] (rows: any number of `out` groups of 16) with H [E or 1, in, in] (one per batch entry,
    or one shared): the experts of a chunk run each column step together. Works on W transposed, so the column
    being rounded and the ones its error updates are contiguous; inside a block of `block` columns the error reaches
    the next `sub` columns directly and the rest of the block, then the columns after it, by in-place GEMMs."""
    if W.dim() == 2:
        return gptq(W[None], H[None], damp, block, sub)[0]
    Wt = W.float().transpose(1, 2).contiguous()  # [E, n, R]
    E, n, R = Wt.shape
    H = H.double().clone()
    d = torch.diagonal(H, dim1=1, dim2=2)
    dead = d == 0
    if dead.all(-1).any():
        raise ValueError(f"gptq: a Hessian is all zero ({n} columns): no calibration tokens reached this weight")
    d[dead] = 1
    if H.shape[0] == E:
        Wt[dead] = 0
    else:
        Wt[:, dead[0]] = 0
    d += damp * d.mean(-1, keepdim=True)
    U = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True).float()
    diag = torch.diagonal(U, dim1=1, dim2=2).expand(E, n)
    Ut = U.transpose(1, 2).contiguous().expand(E, n, n)  # lower: Ut[k, i] = U[i, k]
    Qt = torch.empty_like(Wt)
    Err = torch.empty(E, block, R)
    for i1 in range(0, n, block):
        i2 = min(i1 + block, n)
        W1 = Wt[:, i1:i2]
        for j1 in range(0, i2 - i1, sub):
            j2 = min(j1 + sub, i2 - i1)
            for i in range(j1, j2):
                w = W1[:, i]
                q = q_groups(w.view(E, R // 16, 16)).view(E, R)
                Qt[:, i1 + i] = q
                err = Err[:, i]
                torch.sub(w, q, out=err)
                err /= diag[:, i1 + i, None]
                if i + 1 < j2:
                    W1[:, i + 1 : j2] -= Ut[:, i1 + i + 1 : i1 + j2, i1 + i, None] * err[:, None]
            if j2 < i2 - i1:
                W1[:, j2:].baddbmm_(Ut[:, i1 + j2 : i2, i1 + j1 : i1 + j2], Err[:, j1:j2], alpha=-1)
        if i2 < n:
            Wt[:, i2:].baddbmm_(Ut[:, i2:, i1:i2], Err[:, : i2 - i1], alpha=-1)
    return Qt.transpose(1, 2)


def round_experts(layer, kind, W, cfg=None, chunk=32, experts=None, gate_up=None):
    """W [E, out, in] -> the layer's rounding of it as grid values [E, out, in] in W's dtype (bf16 holds every bfp4 grid
    value exactly), or None for the default. experts: the global expert ids of W's rows (default 0..E-1); gate_up: the
    same experts' unrounded gate|up [E, 2N, H], which gptqe on down needs."""
    cfg = cfg if cfg is not None else _cfg()
    m = method(layer, kind, cfg)
    if m is None or m == "default":
        return None
    E, O, I = W.shape
    if m == "rtne":
        return torch.cat([rtne(W[e0 : e0 + chunk]).to(W.dtype) for e0 in range(0, E, chunk)])
    if m.startswith("gptqe"):
        damp = float(m[5:])
        h, idx = _cal(layer, cfg)
        if kind == "dn" and gate_up is None:
            raise ValueError(f"HYB_BFP4_ROUND: gptqe on layer {layer} down needs the experts' gate|up (gate_up=)")
        ids = list(range(E) if experts is None else experts)
        out = torch.empty_like(W)
        Hs, js = [], []
        for j, e in enumerate(ids):
            x = h[(idx == int(e)).any(-1)]
            if x.shape[0] < MIN_TOK:
                out[j] = rtne(W[j : j + 1])[0].to(W.dtype)
                continue
            if kind == "dn":
                g = gate_up[j].float()
                N = g.shape[0] // 2
                x = torch.nn.functional.silu(x @ g[:N].t()) * (x @ g[N:].t())
            Hs.append(x.t() @ x)
            js.append(j)
            if len(js) == chunk:
                out[js] = gptq(W[js], torch.stack(Hs), damp).to(W.dtype)
                Hs, js = [], []
        if js:
            out[js] = gptq(W[js], torch.stack(Hs), damp).to(W.dtype)
        return out
    if m.startswith("gptq"):
        if kind != "gu":
            raise ValueError(f"HYB_BFP4_ROUND: {m!r} (layer-wide Hessian) is for gate|up only; use gptqe on down")
        damp = float(m[4:])
        h, _ = _cal(layer, cfg)
        H = h.t() @ h
        out = []
        for e0 in range(0, E, chunk):
            Wc = W[e0 : e0 + chunk].reshape(1, -1, I)  # shared H: experts stack along the rows
            out.append(gptq(Wc, H[None], damp)[0].reshape(-1, O, I).to(W.dtype))
        return torch.cat(out)
    raise ValueError(f"HYB_BFP4_ROUND: unknown method {m!r} for layer {layer} {kind}")
