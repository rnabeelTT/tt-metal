# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Hy4 hyper-connections (HF HYV4FlashHyperConnection / HYV4FlashHyperHead), fp32 like the reference.

The residual is ONE replicated bf16 tensor X [1, hc, T, H] (the 4 streams stacked on dim 1). One sublayer:

    r     = rsqrt(mean(X^2 over all hc * H) + eps)          (fp32)
    mix   = (sum_i X_i @ fn_i^T) * r                         [T, 2 * hc]   (fn_i: stream i's columns of fn)
    pre   = sigmoid(mix[:hc] * s_pre + b_pre) + hc_eps
    post  = mag * sigmoid(mix[hc:] * s_post + b_post) + hc_eps
    in    = bf16(sum_i pre_i * X_i)                          -> the sublayer (after its own RMSNorm)
    X_i'  = bf16(post_i * out + X_i)                         (fp32 math)

As batched ops over the stream dim: one batched matmul for the mixes, one broadcast multiply + stream-sum for the
input, one broadcast multiply-add for the update. The head (HyperHead) is the `pre` half only, with its own weights.
"""

from __future__ import annotations

import os

import torch
import ttnn

from models.autoports.tencent_hy4_flash.tt.common import TILE, ckc, upload

KC = 256  # split-K chunk of the decode mix (8 tiles per core: hc * H / KC = 48 cores at H = 3072)


def _ssum(x, mem):
    """Sum over the stream dim (1) in one program (ttnn.sum(dim=1) is transpose + reduce + transpose)."""
    if os.environ.get("HY4_HC_FASTREDUCE", "1") == "1":
        return ttnn.experimental.fast_reduce_nc(x, dims=[1], memory_config=mem)
    return ttnn.sum(x, dim=1, keepdim=True, memory_config=mem)


class HyperConnection:
    def __init__(self, mesh, fn: torch.Tensor, base: torch.Tensor, scale: torch.Tensor, hc: int, hidden: int,
                 rms_eps: float, hc_eps: float, magnitude: float | None):
        """fn [M, hc * H] (M = 2 hc for a sublayer, hc for the head), base [M], scale [2] (sublayer) or [1] (head)."""
        self.mesh, self.hc, self.H = mesh, hc, hidden
        self.eps, self.hc_eps, self.mag = float(rms_eps), float(hc_eps), magnitude
        M = int(fn.shape[0])
        self.M = M
        fn = fn.float()
        w = torch.zeros(1, hc, hidden, TILE, dtype=torch.float32)
        for i in range(hc):
            w[0, i, :, :M] = fn[:, i * hidden : (i + 1) * hidden].T
        self.fnT = upload(mesh, w, ttnn.float32)  # [1, hc, H, 32]
        sc = torch.zeros(TILE, dtype=torch.float32)
        bs = torch.zeros(TILE, dtype=torch.float32)
        scale = scale.float().reshape(-1)
        if M == 2 * hc:
            sc[:hc], sc[hc:M] = scale[0], scale[1]
        else:
            sc[:M] = scale[0]
        bs[:M] = base.float().reshape(-1)
        self.sc = upload(mesh, sc.reshape(1, 1, 1, TILE), ttnn.float32)
        self.bs = upload(mesh, bs.reshape(1, 1, 1, TILE), ttnn.float32)
        # post = mag * sig + eps, pre = sig + eps on the mixing columns: one fused affine on the sigmoid row
        a = torch.ones(TILE, dtype=torch.float32)
        if M == 2 * hc:
            a[hc:M] = float(magnitude)
        self.aff_a = upload(mesh, a.reshape(1, 1, 1, TILE), ttnn.float32)
        self.ckc = ckc(mesh, fp32=True)
        self.inv_n = 1.0 / float(hc * hidden)
        # decode (one 32-row tile row): split-K form. [1, hc, 32, H] and [1, hc * H / KC, 32, KC] share their tile pages
        # (stream s, tile j -> page s * H/32 + j), so a zero-copy view makes the mix one batched matmul with
        # hc * H / KC output tiles (one per core) + one reduce over the batch, instead of hc output tiles on hc cores
        self.splitk = os.environ.get("HY4_HC_SPLITK", "1") == "1" and hidden % KC == 0
        if self.splitk:
            nb = hc * hidden // KC
            wk = torch.zeros(1, nb, KC, TILE, dtype=torch.float32)
            for i in range(hc):
                blk = fn[:, i * hidden : (i + 1) * hidden].T  # [H, M]
                for c in range(hidden // KC):
                    wk[0, i * (hidden // KC) + c, :, :M] = blk[c * KC : (c + 1) * KC]
            self.fnK = upload(mesh, wk, ttnn.float32)
            self.nb = nb
            key = (id(mesh), nb)
            if key not in HyperConnection._ONES:  # sum of squares: the same split, a ones column (shared)
                ones = torch.zeros(1, nb, KC, TILE, dtype=torch.float32)
                ones[..., 0] = 1.0
                HyperConnection._ONES[key] = upload(mesh, ones, ttnn.float32)
            self.onesK = HyperConnection._ONES[key]

    _ONES = {}

    def _weights_splitk(self, xf, mem):
        """xf fp32 [1, hc, 32, H] -> w [1, 1, 32, 32] (as _weights), via the split-K views."""
        T = int(xf.shape[-2])
        xv = ttnn.experimental.view(xf, (1, self.nb, T, KC))
        mix = ttnn.experimental.fast_reduce_nc(
            ttnn.matmul(xv, self.fnK, dtype=ttnn.float32, compute_kernel_config=self.ckc, memory_config=mem),
            dims=[1], memory_config=mem)
        sq = ttnn.multiply(xf, xf, memory_config=mem)
        sv = ttnn.experimental.view(sq, (1, self.nb, T, KC))
        ss = ttnn.experimental.fast_reduce_nc(
            ttnn.matmul(sv, self.onesK, dtype=ttnn.float32, compute_kernel_config=self.ckc, memory_config=mem),
            dims=[1], memory_config=mem)  # column 0 = sum of squares
        ttnn.deallocate(sq)
        ss0 = ttnn.slice(ss, (0, 0, 0, 0), (1, 1, T, 1), memory_config=mem)
        ttnn.deallocate(ss)
        r = ttnn.rsqrt(ttnn.add(ttnn.multiply(ss0, self.inv_n, memory_config=mem), self.eps, memory_config=mem),
                       memory_config=mem)
        mix = ttnn.multiply(mix, r, memory_config=mem)
        sig = ttnn.sigmoid(ttnn.add(ttnn.multiply(mix, self.sc, memory_config=mem), self.bs, memory_config=mem),
                           memory_config=mem)
        return ttnn.add(ttnn.multiply(sig, self.aff_a, memory_config=mem), self.hc_eps, memory_config=mem)

    def _weights(self, xf, mem):
        """xf fp32 [1, hc, T, H] -> w [1, 1, T, 32] fp32: cols 0..hc-1 = pre, hc..2hc-1 = post (pad meaningless)."""
        mix = ttnn.matmul(xf, self.fnT, dtype=ttnn.float32, compute_kernel_config=self.ckc, memory_config=mem)
        mix = _ssum(mix, mem)  # [1, 1, T, 32]
        # NOT fast_reduce_nc on the [1, hc, T, 1] row sums: it returned wrong sums there (2026-11-19: every layer's
        # output at PCC ~0.3 vs the reference, garbage generations); ttnn.sum over the stream dim is exact
        ss = ttnn.sum(ttnn.sum(ttnn.multiply(xf, xf, memory_config=mem), dim=-1, keepdim=True, memory_config=mem),
                      dim=1, keepdim=True, memory_config=mem)  # [1, 1, T, 1]
        r = ttnn.rsqrt(ttnn.add(ttnn.multiply(ss, self.inv_n, memory_config=mem), self.eps, memory_config=mem),
                       memory_config=mem)
        mix = ttnn.multiply(mix, r, memory_config=mem)
        sig = ttnn.sigmoid(ttnn.add(ttnn.multiply(mix, self.sc, memory_config=mem), self.bs, memory_config=mem),
                           memory_config=mem)
        return ttnn.add(ttnn.multiply(sig, self.aff_a, memory_config=mem), self.hc_eps, memory_config=mem)

    def _per_stream(self, w, c0, mem):
        """columns c0 .. c0 + hc - 1 of w [1, 1, T, 32] -> [1, hc, T, 1] (stream on dim 1)."""
        T = int(w.shape[-2])
        cols = ttnn.slice(w, (0, 0, 0, c0), (1, 1, T, c0 + self.hc), memory_config=mem)  # [1, 1, T, hc]
        return ttnn.permute(cols, (0, 3, 2, 1), memory_config=mem)  # [1, hc, T, 1]

    def pre(self, X, mem=None):
        """X [1, hc, T, H] bf16 -> (sublayer input bf16 [1, 1, T, H], post weights [1, hc, T, 1] fp32 or None)."""
        mem = mem or ttnn.DRAM_MEMORY_CONFIG
        xf = ttnn.typecast(X, ttnn.float32, memory_config=mem)
        w = self._weights_splitk(xf, mem) if self.splitk and int(X.shape[-2]) == TILE else self._weights(xf, mem)
        pre = self._per_stream(w, 0, mem)
        s = _ssum(ttnn.multiply(xf, pre, memory_config=mem), mem)
        ttnn.deallocate(xf)
        out = ttnn.typecast(s, ttnn.bfloat16, memory_config=mem)
        post = self._per_stream(w, self.hc, mem) if self.M == 2 * self.hc else None
        return out, post

    def post(self, X, out, post, mem=None):
        """X' = bf16(post * out + X) (fp32 math): X [1, hc, T, H], out [1, 1, T, H], post [1, hc, T, 1]."""
        mem = mem or ttnn.DRAM_MEMORY_CONFIG
        of = ttnn.typecast(out, ttnn.float32, memory_config=mem)
        y = ttnn.add(ttnn.multiply(post, of, memory_config=mem), ttnn.typecast(X, ttnn.float32, memory_config=mem),
                     memory_config=mem)
        return ttnn.typecast(y, ttnn.bfloat16, memory_config=mem)


def sublayer_hc(mesh, sd, prefix, cfg):
    """The attention / MLP hyper-connection of one layer: sd keys `{prefix}.hc_pre.hc_fn|hc_base|hc_scale`."""
    return HyperConnection(
        mesh,
        sd[f"{prefix}.hc_pre.hc_fn"],
        sd[f"{prefix}.hc_pre.hc_base"],
        sd[f"{prefix}.hc_pre.hc_scale"],
        cfg.hc_mult,
        cfg.hidden,
        cfg.rms_eps,
        cfg.hc_eps,
        cfg.hc_magnitude,
    )


def head_hc(mesh, w, cfg):
    return HyperConnection(
        mesh,
        w.get("model.hc_head.hc_head_fn"),
        w.get("model.hc_head.hc_head_base"),
        w.get("model.hc_head.hc_head_scale"),
        cfg.hc_mult,
        cfg.hidden,
        cfg.rms_eps,
        cfg.hc_eps,
        None,
    )
