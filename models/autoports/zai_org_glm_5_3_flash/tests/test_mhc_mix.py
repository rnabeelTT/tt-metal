# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""ttnn.experimental.deepseek_prefill.mhc_mix (split-K mix projection + Gram partials) and the fused decoder's finalize
(sum over chunks, Gram diagonals -> rsqrt(mean(x^2) + eps)) vs torch: (x @ fn^T) * rsqrt(mean(x^2) + eps)."""

import pytest
import torch

import ttnn
from models.autoports.zai_org_glm_5_3_flash.tt.fused_decoder import mix_chunks

K, EPS = 16384, 1e-5


def _pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm()))


@pytest.mark.parametrize("T", [32, 64, 2048])
def test_mhc_mix(device, T):
    g = torch.Generator().manual_seed(T)
    x = torch.randn(T, K, generator=g) * 0.02
    x[:, ::97] *= 30  # outlier channels, as the real stream
    fn = torch.zeros(K, 32)
    fn[:, :24] = torch.randn(K, 24, generator=g) * 0.01
    ref = (x @ fn) * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS)
    up = lambda t: ttnn.from_torch(t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    C = mix_chunks(T, K // 32)
    part = ttnn.experimental.deepseek_prefill.mhc_mix(up(x.reshape(1, 1, T, K)), up(fn), C)
    assert list(part.shape) == [1, C, T, 64]
    p = ttnn.to_torch(part).reshape(C, T, 64).sum(0)
    gram_diag = p[:, 32:].reshape(T // 32, 32, 32).diagonal(dim1=1, dim2=2).reshape(T)
    got = p[:, :32] * torch.rsqrt(gram_diag.reshape(T, 1) / K + EPS)
    ss_err = (gram_diag / x.pow(2).sum(-1) - 1).abs().max().item()
    pc = _pcc(got[:, :24], ref[:, :24])
    print(f"T={T} C={C}: mixes PCC {pc:.7f}, sum-of-squares max rel err {ss_err:.2e}")
    # the Gram matmul reads fp32 at TF32 precision: sum of squares ~1.5e-3 relative (scale rsqrt: ~8e-4), below the
    # HF stream's own bf16 rounding (~4e-3)
    assert pc > 0.99999 and ss_err < 2e-3
