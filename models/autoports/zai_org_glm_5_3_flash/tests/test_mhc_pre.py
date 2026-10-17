# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""ttnn.experimental.deepseek_prefill.mhc_pre: h = sum_i pre[:, i] * residual[:, i*C:(i+1)*C] (packed mHC streams)."""

import pytest
import torch

import ttnn
from models.demos.deepseek_v3_d_p.tt.mhc.tt_mhc import build_bcast_consts

N, C = 4, 4096


def _pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm()))


@pytest.mark.parametrize(
    "dtype,out_dtype", [(ttnn.float32, None), (ttnn.float32, ttnn.bfloat16), (ttnn.bfloat16, None)]
)
@pytest.mark.parametrize("T", [32, 96, 2048])
def test_mhc_pre(device, T, dtype, out_dtype):
    torch.manual_seed(0)
    res = torch.randn(1, 1, T, N * C) * 0.05
    pre = torch.sigmoid(torch.randn(1, 1, T, N)) + 1e-6
    tdt = torch.float32 if dtype == ttnn.float32 else torch.bfloat16
    up = lambda t: ttnn.from_torch(t.to(tdt), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    consts = up(build_bcast_consts(N).reshape(N * N, 32, 32))
    kw = {"output_dtype": out_dtype} if out_dtype is not None else {}
    h = ttnn.experimental.deepseek_prefill.mhc_pre(up(res), up(pre), consts, N, **kw)
    assert list(h.shape) == [1, 1, T, C]
    assert h.dtype == (out_dtype or dtype)
    got = ttnn.to_torch(h).float()
    ref_q = sum(pre.to(tdt).double()[..., i : i + 1] * res.to(tdt).double()[..., i * C : (i + 1) * C] for i in range(N))
    p = _pcc(got, ref_q)
    err = (got - ref_q).abs().max().item()
    rel = err / ref_q.abs().max().item()
    scale = float((got.double().flatten() @ ref_q.double().flatten()) / ref_q.double().flatten().pow(2).sum())
    print(
        f"T={T} dtype={dtype} out={out_dtype}: PCC {p:.9f} scale {scale:.9f} max|err| {err:.2e} max rel err {rel:.2e}"
    )
    assert p > (0.99999 if dtype == ttnn.float32 and out_dtype is None else 0.9999)
    if dtype == ttnn.float32 and out_dtype is None:  # fp32 end to end: no TF32 read of the streams / pre
        assert rel < 1e-5, rel
