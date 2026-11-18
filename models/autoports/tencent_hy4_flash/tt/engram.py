# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Hy4 Engram memory (HF HYV4FlashEngramLayer), device half.

The host computes, per token, the 16 hashed n-gram rows of this layer's table (FP8 round trip as trained) and their
per-head RMSNorms (models/experimental/hybrid_moe/engram_store.py), and uploads E [T, 1536] bf16. The device then does

    v      = value_proj(E)                                   (1536 -> 3072, bias)
    main   = pre_gamma * rmsnorm(v)                          (no weight, eps = fp32 eps)
    key_i  = norm_key_i(key_proj_i(E))                       i = 0..3 (one per stream)
    q_i    = norm_query_i(X_i)
    g_i    = sigmoid(signed_sqrt(sum(key_i * q_i) / sqrt(3072)))
    X_i   += (g_i + scale_i) * main                          (bf16 add, as the reference)

All of the norms are torch.nn.RMSNorm with eps=None on bf16 inputs: torch computes them in fp32 with eps =
finfo(fp32).eps (1.19e-7) and applies the weight before one rounding (not the bf16 eps: that moves outputs by up to 3.7).
Positions whose token is the pad id get no Engram residual at all (the reference masks the whole output).
"""

from __future__ import annotations

import torch
import ttnn

from models.autoports.tencent_hy4_flash.tt.common import ckc, upload

NORM_EPS = float(torch.finfo(torch.float32).eps)


class Hy4EngramDevice:
    def __init__(self, mesh, cfg, sd):
        """sd: the layer's `engram_layer.` tensors (prefix stripped)."""
        self.mesh, self.cfg = mesh, cfg
        self.hc = cfg.hc_mult
        H = cfg.hidden
        bf = lambda t: t.to(torch.bfloat16)  # noqa: E731
        self.ckc = ckc(mesh, fp32=True)
        self.wv = upload(mesh, bf(sd["value_proj.weight"]).T.reshape(1, 1, -1, H))
        self.bv = upload(mesh, bf(sd["value_proj.bias"]).reshape(1, 1, 1, H))
        hc = self.hc
        st = lambda key: torch.stack([bf(sd[key.format(i)]) for i in range(hc)])  # noqa: E731
        self.wk = upload(mesh, st("key_projs.{}.weight").transpose(1, 2).reshape(1, hc, -1, H))  # [1, 4, 1536, H]
        self.bk = upload(mesh, st("key_projs.{}.bias").reshape(1, hc, 1, H))
        self.nk = upload(mesh, st("norm_key.{}.weight").float().reshape(1, hc, 1, H), ttnn.float32)
        self.nq = upload(mesh, st("norm_query.{}.weight").float().reshape(1, hc, 1, H), ttnn.float32)
        self.gamma = upload(mesh, bf(sd["unified_pre_gamma"]).float().reshape(1, 1, 1, H), ttnn.float32)
        self.sc = upload(mesh, st("unified_per_hc_scales.{}").reshape(1, hc, 1, H))
        self.inv_sqrt_h = H**-0.5

    def _rms(self, x, w, mem):
        """torch.nn.RMSNorm(eps=None) on a bf16 input: fp32 math, the weight applied in fp32, one rounding at the end
        (left fp32 here: the consumers keep computing in fp32). w: fp32 [1, hc or 1, 1, H]."""
        xf = ttnn.typecast(x, ttnn.float32, memory_config=mem) if x.dtype != ttnn.float32 else x
        n = ttnn.rms_norm(xf, epsilon=NORM_EPS, compute_kernel_config=self.ckc, memory_config=mem)
        return ttnn.multiply(n, w, memory_config=mem)

    def forward(self, X, emb, mem=None, keep=None):
        """X: the streams [1, hc, Tp, 3072] bf16; emb: [1, 1, Tp, 1536] bf16 (host rows, zero for padding rows); keep:
        optional [1, 1, Tp, 1] bf16 0/1 (0 = a masked position: no Engram residual). Returns the updated streams."""
        mem = mem or ttnn.DRAM_MEMORY_CONFIG
        Tp = int(X.shape[-2])
        v = ttnn.linear(emb, self.wv, bias=self.bv, compute_kernel_config=self.ckc, memory_config=mem)
        main = ttnn.typecast(self._rms(v, self.gamma, mem), ttnn.bfloat16, memory_config=mem)  # gamma * rmsnorm(v)
        ttnn.deallocate(v)
        e4 = ttnn.repeat(emb, ttnn.Shape((1, self.hc, 1, 1)), memory_config=mem)
        k = ttnn.add(ttnn.matmul(e4, self.wk, compute_kernel_config=self.ckc, memory_config=mem), self.bk,
                     memory_config=mem)  # [1, hc, Tp, H] bf16
        ttnn.deallocate(e4)
        kn = self._rms(k, self.nk, mem)
        ttnn.deallocate(k)
        qn = self._rms(X, self.nq, mem)
        g = ttnn.sum(ttnn.multiply(kn, qn, memory_config=mem), dim=-1, keepdim=True, memory_config=mem)  # [1,hc,Tp,1]
        ttnn.deallocate(kn)
        ttnn.deallocate(qn)
        g = ttnn.multiply(g, self.inv_sqrt_h, memory_config=mem)
        mag = ttnn.sqrt(ttnn.clamp(ttnn.abs(g, memory_config=mem), min=1e-6, memory_config=mem), memory_config=mem)
        g = ttnn.sigmoid(ttnn.multiply(mag, ttnn.sign(g, memory_config=mem), memory_config=mem), memory_config=mem)
        g = ttnn.typecast(g, ttnn.bfloat16, memory_config=mem)
        coef = ttnn.add(self.sc, g, memory_config=mem)  # [1, hc, Tp, H] = g_i + scale_i
        if keep is not None:
            coef = ttnn.multiply(coef, keep, memory_config=mem)
        out = ttnn.add(X, ttnn.multiply(coef, main, memory_config=mem), memory_config=mem)
        ttnn.deallocate(coef)
        ttnn.deallocate(main)
        return out
