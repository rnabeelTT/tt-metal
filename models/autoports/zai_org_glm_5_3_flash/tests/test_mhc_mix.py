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


@pytest.mark.parametrize("T", [1, 8])
def test_mixes_split_k_padded_rows(T):
    """The fused decoder's own finalize (FusedHyperConnection._mixes_split_k) at a non-aligned row count, with NaN / inf
    in the tile's padding rows (the decode case with idle padding rows; the stage-2 DSA NaN bug lived here). Checks:
    finite output, bit-identical to the zero-padded run, mixes vs float64, and the signed sum-of-squares error next to
    the functional composite's (multiply + mean). Column 0 of fn is a one-hot (bf16-exact 1.0), so the implied scale is
    out[:, 0] / x[:, K0]."""
    from models.autoports.zai_org_glm_5_3_flash.tests.harness import close_mesh, open_mesh
    from models.autoports.zai_org_glm_5_3_flash.tt.fused_decoder import FusedFlags, FusedHyperConnection
    from models.autoports.zai_org_glm_5_3_flash.tt.mhc import HcConfig, HyperConnection

    K0 = 5
    g = torch.Generator().manual_seed(100 + T)
    x = torch.randn(T, K, generator=g) * 0.02
    x[:, ::97] *= 30
    x[:, K0] = 0.5 + torch.rand(T, generator=g)
    fn = torch.randn(24, K, generator=g) * 0.01
    fn[0] = 0.0
    fn[0, K0] = 1.0
    cfg = HcConfig(n=4, sinkhorn_iters=20, eps=1e-6, norm_eps=EPS)
    mesh = open_mesh()
    try:
        hc = FusedHyperConnection(HyperConnection(mesh, cfg, fn, torch.zeros(24), torch.ones(3)), FusedFlags())
        hc.prepare([T])

        def stream(pad):
            full = torch.full((32, K), pad)
            full[T + 1 :: 2] = float("inf")
            full[:T] = x
            t = ttnn.from_torch(full.reshape(1, 1, 32, K), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh)
            return ttnn.reshape(t, (1, 1, T, K), (1, 1, 32, K))  # logical T rows; the padding rows keep their values

        out_nan = ttnn.to_torch(hc._mixes_split_k(stream(float("nan")))).reshape(T, 32)
        out_zero = ttnn.to_torch(hc._mixes_split_k(stream(0.0))).reshape(T, 32)
        assert torch.isfinite(out_nan).all()
        same = torch.equal(out_nan, out_zero)
        fnb = fn.bfloat16().double()
        ss_ref = x.double().pow(2).sum(-1)
        ref = (x.double() @ fnb.t()) * torch.rsqrt(ss_ref / K + EPS).reshape(T, 1)
        pc = _pcc(out_nan[:, :24], ref)
        scale = out_nan[:, 0].double() / x[:, K0].double()
        ss_fused = K * (scale.pow(-2) - EPS)
        e_f = ss_fused / ss_ref - 1
        # functional composite on the same rows (clean padding: the composite reduces logical columns only)
        tx = ttnn.from_torch(x.reshape(1, 1, T, K), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh)
        ms = ttnn.mean(ttnn.multiply(tx, tx), dim=-1, keepdim=True)
        e_c = ttnn.to_torch(ms).reshape(T).double() * K / ss_ref - 1
        print(
            f"T={T}: finite, bit-identical to zero padding: {same}, mixes PCC {pc:.9f}; sum-of-squares signed rel err "
            f"fused mean {e_f.mean().item():+.2e} max |.| {e_f.abs().max().item():.2e} (via the rsqrt scale) | "
            f"functional composite mean {e_c.mean().item():+.2e} max |.| {e_c.abs().max().item():.2e}"
        )
        assert same and pc > 0.99999 and e_f.abs().max().item() < 1e-3
    finally:
        close_mesh(mesh)
