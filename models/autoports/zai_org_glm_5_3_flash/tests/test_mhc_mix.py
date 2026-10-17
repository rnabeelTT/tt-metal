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


@pytest.mark.parametrize("fn_bf16", [True, False], ids=["fn_bf16", "fn_fp32"])
@pytest.mark.parametrize("T", [32, 64, 2048])
def test_mhc_mix(device, T, fn_bf16):
    g = torch.Generator().manual_seed(T)
    x = torch.randn(T, K, generator=g) * 0.02
    x[:, ::97] *= 30  # outlier channels, as the real stream
    fn = torch.zeros(K, 32)
    fn[:, :24] = torch.randn(K, 24, generator=g) * 0.01
    if fn_bf16:  # the model's projection weights are bf16-valued (HyperConnection: fn.to(bfloat16).float())
        fn = fn.bfloat16().float()
    ref = (x.double() @ fn.double()) * torch.rsqrt(x.double().pow(2).mean(-1, keepdim=True) + EPS)
    up = lambda t: ttnn.from_torch(t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    C = mix_chunks(T, K // 32)
    part = ttnn.experimental.deepseek_prefill.mhc_mix(up(x.reshape(1, 1, T, K)), up(fn), C)
    assert list(part.shape) == [1, C, T, 64]
    p = ttnn.to_torch(part).reshape(C, T, 64).sum(0)
    gram_diag = p[:, 32:].reshape(T // 32, 32, 32).diagonal(dim1=1, dim2=2).reshape(T)
    got = p[:, :32] * torch.rsqrt(gram_diag.reshape(T, 1) / K + EPS)
    ss_err = (gram_diag.double() / x.double().pow(2).sum(-1) - 1).abs().max().item()
    pc = _pcc(got[:, :24], ref[:, :24])
    mix_err = ((got[:, :24].double() - ref[:, :24]).abs().max() / ref[:, :24].abs().max()).item()
    print(
        f"T={T} C={C} fn {'bf16' if fn_bf16 else 'fp32'}: mixes PCC {pc:.9f} max rel err {mix_err:.2e}, "
        f"sum-of-squares max rel err {ss_err:.2e}"
    )
    # x is split hi + lo in the kernel (no TF32 read of the fp32 stream; fn is used at srcA precision, exact for bf16
    # weights). What remains is the FPU's own reduced-precision sum of the 16 products inside each tile face
    # (~2^-12 of the largest, measured with ttnn.matmul probes): sum of squares ~3e-4 (1.5e-3 with the TF32 read)
    assert pc > 0.99999 and ss_err < 1e-3
    if fn_bf16:
        assert mix_err < 1e-3
