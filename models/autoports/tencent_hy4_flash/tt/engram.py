# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Hy4 Engram memory (HF HYV4FlashEngramLayer), device half.

The host computes, per token, the 16 hashed n-gram rows of this layer's table (FP8 round trip as trained) and their
per-head RMSNorms (models/experimental/hybrid_moe/engram_store.py), and uploads E [T, 1536] bf16. The device then does

    v      = value_proj(E)                                   (1536 -> 3072, bias)
    main   = pre_gamma * rmsnorm(v)                          (no weight, eps = bf16 eps)
    key_i  = norm_key_i(key_proj_i(E))                       i = 0..3 (one per stream)
    q_i    = norm_query_i(X_i)
    g_i    = sigmoid(signed_sqrt(sum(key_i * q_i) / sqrt(3072)))
    X_i   += (g_i + scale_i) * main                          (bf16 add, as the reference)

All of the norms are torch.nn.RMSNorm with eps=None on bf16 inputs, i.e. eps = finfo(bf16).eps = 2^-7.
"""

from __future__ import annotations

import torch
import ttnn

from models.autoports.tencent_hy4_flash.tt.common import ckc, upload

BF16_EPS = float(torch.finfo(torch.bfloat16).eps)


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
        self.wk = [upload(mesh, bf(sd[f"key_projs.{i}.weight"]).T.reshape(1, 1, -1, H)) for i in range(self.hc)]
        self.bk = [upload(mesh, bf(sd[f"key_projs.{i}.bias"]).reshape(1, 1, 1, H)) for i in range(self.hc)]
        self.nk = [upload(mesh, bf(sd[f"norm_key.{i}.weight"]).reshape(1, 1, 1, H)) for i in range(self.hc)]
        self.nq = [upload(mesh, bf(sd[f"norm_query.{i}.weight"]).reshape(1, 1, 1, H)) for i in range(self.hc)]
        self.gamma = upload(mesh, bf(sd["unified_pre_gamma"]).reshape(1, 1, 1, H))
        self.sc = [upload(mesh, bf(sd[f"unified_per_hc_scales.{i}"]).reshape(1, 1, 1, H)) for i in range(self.hc)]
        self.inv_sqrt_h = H**-0.5

    def _rms(self, x, w, mem):
        return ttnn.rms_norm(x, epsilon=BF16_EPS, weight=w, compute_kernel_config=self.ckc, memory_config=mem)

    def forward(self, xs, emb, mem=None):
        """xs: the 4 streams [1,1,Tp,3072] bf16; emb: [1,1,Tp,1536] bf16 (host rows, zero for padding rows).
        Returns the updated streams."""
        mem = mem or ttnn.DRAM_MEMORY_CONFIG
        v = ttnn.linear(emb, self.wv, bias=self.bv, compute_kernel_config=self.ckc, memory_config=mem)
        main = self._rms(v, self.gamma, mem)  # gamma * rmsnorm(v): the unweighted norm's output times the gamma
        ttnn.deallocate(v)
        out = []
        for i, x in enumerate(xs):
            k = ttnn.linear(emb, self.wk[i], bias=self.bk[i], compute_kernel_config=self.ckc, memory_config=mem)
            kn = self._rms(k, self.nk[i], mem)
            ttnn.deallocate(k)
            qn = self._rms(x, self.nq[i], mem)
            g = ttnn.sum(ttnn.multiply(kn, qn, memory_config=mem), dim=-1, keepdim=True, memory_config=mem)
            ttnn.deallocate(kn)
            ttnn.deallocate(qn)
            g = ttnn.multiply(ttnn.typecast(g, ttnn.float32, memory_config=mem), self.inv_sqrt_h, memory_config=mem)
            mag = ttnn.sqrt(ttnn.clamp(ttnn.abs(g, memory_config=mem), min=1e-6, memory_config=mem), memory_config=mem)
            g = ttnn.sigmoid(ttnn.multiply(mag, ttnn.sign(g, memory_config=mem), memory_config=mem), memory_config=mem)
            g = ttnn.typecast(g, ttnn.bfloat16, memory_config=mem)  # [Tp, 1]
            coef = ttnn.add(self.sc[i], g, memory_config=mem)  # [Tp, 3072] = g_i + scale_i
            out.append(ttnn.add(x, ttnn.multiply(coef, main, memory_config=mem), memory_config=mem))
            ttnn.deallocate(coef)
        ttnn.deallocate(main)
        return out
