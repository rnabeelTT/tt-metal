# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""ttnn.experimental.deepseek_prefill.kda_decode_conv (stage 5: the fused KDA decode conv step, which replaces the
unfused chain of tt/fused_decoder.FusedGlmKDA: multiply, 3 x typecast + addcmul, SiLU, select, 3 shift selects) vs a
torch fp64 reference on the identical bf16 / fp32 inputs: qkv to fp32 accuracy, the history update bit-exact (an idle
slot's NaN history comes back as Inf: the bf16 packer's NaN encoding), NaN rows of an idle slot kept out of qkv. The
in-layer equivalence with the unfused chain is covered by the decoder A/B (tools/mc_decode_ab.py --compare) and suite.
"""

import pytest
import torch

import ttnn

W, P = 6144, 8352


def _dev(dev, t, dtype):
    return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=dev)


@pytest.mark.parametrize("B", [2, 32])
def test_kda_decode_conv(device, B):
    g = torch.Generator().manual_seed(B)
    proj = torch.randn(1, 1, B, P, generator=g).bfloat16()
    hist = [torch.randn(1, 1, B, W, generator=g).bfloat16() for _ in range(3)]
    taps = [torch.randn(1, 1, 1, W, generator=g) * 0.5 for _ in range(4)]
    active = (torch.rand(1, 1, B, 1, generator=g) > 0.3).float()
    active[0, 0, 0, 0] = 1.0
    if B > 1:
        active[0, 0, 1, 0] = 0.0
        hist[0][0, 0, 1, :5] = float("nan")  # an idle slot's NaN history must not reach qkv
    # reference
    cur = proj[..., :W].double()
    acc = cur * taps[3].double() + sum(hist[j].double() * taps[j].double() for j in range(3))
    q_ref = torch.where(active.bool(), acc * torch.sigmoid(acc), torch.zeros_like(acc))
    a = active.bool()
    h_ref = [torch.where(a, hist[1], hist[0]), torch.where(a, hist[2], hist[1]), torch.where(a, proj[..., :W], hist[2])]

    hd = [_dev(device, h, ttnn.bfloat16) for h in hist]
    td = [_dev(device, t, ttnn.float32) for t in taps]
    q = ttnn.experimental.deepseek_prefill.kda_decode_conv(
        _dev(device, proj, ttnn.bfloat16), *hd, *td, _dev(device, active, ttnn.float32), W
    )
    qt = ttnn.to_torch(q).double()
    assert qt.shape == q_ref.shape
    assert torch.isfinite(qt).all()
    err = float((qt - q_ref).abs().max() / q_ref.abs().max())
    print(f"B={B}: qkv max rel err {err:.2e}")
    assert err < 1e-5, err
    ok = True
    for j in range(3):
        got = ttnn.to_torch(hd[j]).float()
        ref = h_ref[j].float()
        # an idle slot's non-finite history stays non-finite: the bf16 packer writes NaN as Inf (the unfused bf16
        # `where` keeps the NaN payload); the select keeps it out of qkv either way
        both_bad = ~got.isfinite() & ~ref.isfinite()
        diff = ~(both_bad | (got == ref))
        n = int(diff.sum())
        if n:
            idx = diff.nonzero()[:4].tolist()
            print(
                f"B={B}: h{j} {n} differing elements, e.g. {[(i, float(got[tuple(i)]), float(ref[tuple(i)])) for i in idx]}"
            )
            ok = False
        else:
            print(f"B={B}: h{j} exact (non-finite idle rows stay non-finite)")
    assert ok
