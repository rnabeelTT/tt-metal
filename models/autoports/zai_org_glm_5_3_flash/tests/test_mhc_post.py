# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""ttnn.experimental.deepseek_prefill.mhc_post: streams'_j = post_j * y + sum_i comb[i, j] * streams_i (packed fp32 mHC
streams) vs an fp64 torch evaluation of the identical fp32 inputs. The streams, post and comb are fp32 and must not be
read at TF32 precision: bar ~1e-6 relative (fp32 path); the bf16 path is checked against the bf16-rounded inputs."""

import pytest
import torch

import ttnn
from models.demos.deepseek_v3_d_p.reference.mhc.mhc_reference import sinkhorn_knopp
from models.demos.deepseek_v3_d_p.tt.mhc.tt_mhc import build_bcast_consts

N = 4


def _pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm()))


def _ref(y, res, post, comb, C):
    y, res, post, comb = (t.double() for t in (y, res, post, comb))
    T = y.shape[-2]
    streams = res.reshape(T, N, C)
    cmat = comb.reshape(T, N, N)
    out = post.reshape(T, N, 1) * y.reshape(T, 1, C) + torch.einsum("tij,tic->tjc", cmat, streams)
    return out.reshape(1, 1, T, N * C)


@pytest.mark.parametrize("dtype", [ttnn.float32, ttnn.bfloat16], ids=["fp32", "bf16"])
@pytest.mark.parametrize("T,C", [(32, 4096), (96, 1024), (2048, 4096)])
def test_mhc_post(device, T, C, dtype):
    g = torch.Generator().manual_seed(T + C)
    # residual streams carry large outlier channels (the real stream), the sub-block output is smaller
    res = torch.randn(1, 1, T, N * C, generator=g)
    res[..., ::97] *= 30
    y = torch.randn(1, 1, T, C, generator=g) * 0.3
    post = 2 * torch.sigmoid(torch.randn(1, 1, T, N, generator=g))
    comb = sinkhorn_knopp(torch.randn(T, N, N, generator=g), 20, 1e-6).reshape(1, 1, T, N * N)
    tdt = torch.float32 if dtype == ttnn.float32 else torch.bfloat16
    q = lambda t: t.to(tdt).float()
    up = lambda t: ttnn.from_torch(t.to(tdt), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    consts = up(build_bcast_consts(N).reshape(N * N, 32, 32))
    out = ttnn.experimental.deepseek_prefill.mhc_post(up(y), up(res), up(post), up(comb), consts, N)
    got = ttnn.to_torch(out).float().reshape(1, 1, T, N * C).double()
    ref = _ref(q(y), q(res), q(post), q(comb), C)
    err = (got - ref).abs()
    upd = got - q(res).double()  # the residual update the reviewer measured: post*y + (comb^T - I) streams
    upd_ref = ref - q(res).double()
    p = _pcc(got, ref)
    scale = float((got.flatten() @ ref.flatten()) / (ref.flatten() @ ref.flatten()))
    upd_scale = float((upd.flatten() @ upd_ref.flatten()) / (upd_ref.flatten() @ upd_ref.flatten()))
    max_rel = float(err.max() / ref.abs().max())
    rms_rel = float(err.pow(2).mean().sqrt() / ref.pow(2).mean().sqrt())
    upd_rel = float((upd - upd_ref).pow(2).mean().sqrt() / upd_ref.pow(2).mean().sqrt())
    print(
        f"T={T} C={C} {'fp32' if dtype == ttnn.float32 else 'bf16'}: PCC {p:.9f} scale {scale:.9f} "
        f"max rel err {max_rel:.2e} rms rel err {rms_rel:.2e}; residual update: scale {upd_scale:.7f} "
        f"rms rel err {upd_rel:.2e}"
    )
    if dtype == ttnn.float32:
        assert max_rel < 1e-5 and abs(upd_scale - 1) < 1e-5, (max_rel, upd_scale)
    else:  # output rounded to bf16
        assert p > 0.99999 and max_rel < 1e-2
