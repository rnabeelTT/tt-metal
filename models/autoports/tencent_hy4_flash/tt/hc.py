# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Hy4 hyper-connections (HF HYV4FlashHyperConnection / HYV4FlashHyperHead), fp32 like the reference.

The residual is a list of hc_mult (4) replicated bf16 streams X_i [1, 1, T, H]. One sublayer:

    r     = rsqrt(mean(concat(X)^2) + eps)                 (fp32, over all 4 * H)
    mix   = (concat(X) @ fn^T) * r                          [T, 2 * hc]
    pre   = sigmoid(mix[:hc] * s_pre + b_pre) + hc_eps
    post  = mag * sigmoid(mix[hc:] * s_post + b_post) + hc_eps
    in    = bf16(sum_i pre_i * X_i)                         -> the sublayer (after its own RMSNorm)
    X_i'  = bf16(post_i * out + X_i)                        (fp32 math)

The head (HyperHead) is the `pre` half only (no post), with its own fn / base / scale.
"""

from __future__ import annotations

import torch
import ttnn

from models.autoports.tencent_hy4_flash.tt.common import TILE, ckc, rms_scale_fp32, upload


class HyperConnection:
    def __init__(self, mesh, fn: torch.Tensor, base: torch.Tensor, scale: torch.Tensor, hc: int, hidden: int,
                 rms_eps: float, hc_eps: float, magnitude: float | None):
        """fn [M, hc * H] (M = 2 hc for a sublayer, hc for the head), base [M], scale [2] (sublayer) or [1] (head)."""
        self.mesh, self.hc, self.H = mesh, hc, hidden
        self.eps, self.hc_eps, self.mag = float(rms_eps), float(hc_eps), magnitude
        M = int(fn.shape[0])
        self.M = M
        fn = fn.float()
        # per-stream transposed slices, columns padded to one tile: X_i @ fnT_i summed over i == concat(X) @ fn^T
        self.fnT = []
        for i in range(hc):
            w = torch.zeros(hidden, TILE, dtype=torch.float32)
            w[:, :M] = fn[:, i * hidden : (i + 1) * hidden].T
            self.fnT.append(upload(mesh, w.reshape(1, 1, hidden, TILE), ttnn.float32))
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
        self.ckc = ckc(mesh, fp32=True)

    def _weights(self, xs_f32, mem):
        """-> sig [1, 1, T, 32] fp32 = sigmoid(mix * scale + base) (pad columns meaningless)."""
        mix = None
        for x, w in zip(xs_f32, self.fnT):
            m = ttnn.matmul(x, w, dtype=ttnn.float32, compute_kernel_config=self.ckc, memory_config=mem)
            mix = m if mix is None else ttnn.add(mix, m, memory_config=mem)
        r = rms_scale_fp32(xs_f32, self.eps, self.hc * self.H, mem)
        mix = ttnn.multiply(mix, r, memory_config=mem)
        mix = ttnn.add(ttnn.multiply(mix, self.sc, memory_config=mem), self.bs, memory_config=mem)
        return ttnn.sigmoid(mix, memory_config=mem)

    def _col(self, sig, j, mem):
        T = int(sig.shape[-2])
        return ttnn.slice(sig, (0, 0, 0, j), (1, 1, T, j + 1), memory_config=mem)

    def pre(self, xs, mem=None):
        """-> (sublayer input bf16 [1, 1, T, H], post weights list of [1, 1, T, 1] fp32 or None for the head)."""
        mem = mem or ttnn.DRAM_MEMORY_CONFIG
        xf = [ttnn.typecast(x, ttnn.float32, memory_config=mem) for x in xs]
        sig = self._weights(xf, mem)
        acc = None
        for i, x in enumerate(xf):
            p = ttnn.add(self._col(sig, i, mem), self.hc_eps, memory_config=mem)
            t = ttnn.multiply(x, p, memory_config=mem)
            acc = t if acc is None else ttnn.add(acc, t, memory_config=mem)
        out = ttnn.typecast(acc, ttnn.bfloat16, memory_config=mem)
        post = None
        if self.M == 2 * self.hc:
            post = [
                ttnn.add(ttnn.multiply(self._col(sig, self.hc + i, mem), self.mag, memory_config=mem), self.hc_eps,
                         memory_config=mem)
                for i in range(self.hc)
            ]
        return out, post

    def post(self, xs, out, post, mem=None):
        """X_i' = bf16(post_i * out + X_i) (fp32 math)."""
        mem = mem or ttnn.DRAM_MEMORY_CONFIG
        of = ttnn.typecast(out, ttnn.float32, memory_config=mem)
        res = []
        for x, p in zip(xs, post):
            y = ttnn.add(ttnn.multiply(of, p, memory_config=mem), ttnn.typecast(x, ttnn.float32, memory_config=mem),
                         memory_config=mem)
            res.append(ttnn.typecast(y, ttnn.bfloat16, memory_config=mem))
        return res


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
