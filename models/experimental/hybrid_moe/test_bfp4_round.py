# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Host test of bfp4_round (no device): outputs lie on the bfp4 grid the packer keeps exactly (and that the 5-bit
pre-rounding leaves unchanged), exponent search never loses to plain round-to-nearest per group, GPTQ lowers the
output error on correlated inputs, and the per-layer config / cache tag behave.
usage: python -m pytest test_bfp4_round.py"""

import json
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bfp4_round as B  # noqa: E402


def round_sig(t, bits=5):  # moe_decode.fused.round_sig
    drop = 24 - bits
    i = t.float().contiguous().view(torch.int32)
    i = (i + (1 << (drop - 1))) & ~((1 << drop) - 1)
    return i.view(torch.float32)


def on_grid(Q):
    """Q [..., out, in]: every group of 16 along out is m * 2^(e - 2) with |m| <= 7 for one e."""
    g = Q.float().transpose(-1, -2).reshape(-1, 16)
    amax = g.abs().amax(-1, keepdim=True)
    nz = amax[:, 0] > 0
    e = torch.floor(torch.log2(amax[nz]))
    m = g[nz] / torch.pow(2.0, e - 2)
    return bool((m == torch.round(m)).all() and (m.abs() <= 7).all())


def plain_rtn(W):
    O, I = W.shape[-2], W.shape[-1]
    Wt = round_sig(W).transpose(-1, -2).reshape(*W.shape[:-2], I, O // 16, 16)
    return B.q_groups(Wt, search=False).reshape(*W.shape[:-2], I, O).transpose(-1, -2)


def test_rtne_on_grid_and_never_worse():
    torch.manual_seed(0)
    W = torch.randn(3, 64, 48) * torch.rand(3, 64, 1) * 0.05
    Q = B.rtne(W)
    assert Q.shape == W.shape and on_grid(Q)
    assert torch.equal(round_sig(Q), Q.float())  # the conversion's 5-bit rounding is a no-op on grid values
    assert torch.equal(Q.bfloat16().float(), Q)  # bf16 holds them exactly
    err_s = ((Q - W) ** 2).transpose(-1, -2).reshape(-1, 16).sum(-1)
    err_p = ((plain_rtn(W) - W) ** 2).transpose(-1, -2).reshape(-1, 16).sum(-1)
    assert (err_s <= err_p + 1e-12).all()  # per group: emax is one of the two candidates
    assert err_s.sum() < err_p.sum()


def test_gptq_on_grid_and_lower_output_error():
    torch.manual_seed(1)
    I, O, T = 96, 64, 4096
    mix = torch.randn(I, I) * 0.3 + torch.eye(I)
    X = torch.randn(T, I) @ mix  # correlated inputs
    W = torch.randn(O, I) * 0.02
    H = X.t() @ X
    Q = B.gptq(W, H, damp=0.01)
    assert on_grid(Q)
    out = lambda M: ((X @ (M - W).t()) ** 2).sum()  # noqa: E731
    assert out(Q) < 0.9 * out(B.rtne(W))


def test_gptq_rejects_empty_hessian():
    with pytest.raises(ValueError):
        B.gptq(torch.randn(16, 8), torch.zeros(8, 8), damp=0.1)


def test_config_methods_and_tag(tmp_path, monkeypatch):
    W = torch.randn(2, 32, 16)
    monkeypatch.delenv("HYB_BFP4_ROUND", raising=False)
    assert B.round_experts(0, "gu", W) is None and B.tag(0) == ""
    monkeypatch.setenv("HYB_BFP4_ROUND", "rtne")
    assert torch.equal(B.round_experts(0, "gu", W), B.rtne(W)) and not B.uses_cal()
    assert B.tag(0).startswith(".q")
    torch.save(
        {"h": torch.randn(256, 16).bfloat16(), "idx": torch.zeros(256, 2, dtype=torch.uint8)}, tmp_path / "cal_3.pt"
    )
    cfg = {"default": "rtne", "layers": {"3": {"gu": "gptq0.1"}, "5": {"gu": "default", "dn": "default"}}}
    (tmp_path / "r.json").write_text(json.dumps(cfg))
    monkeypatch.setenv("HYB_BFP4_ROUND", str(tmp_path / "r.json"))
    monkeypatch.setenv("HYB_BFP4_CAL", str(tmp_path))
    assert B.method(3, "gu") == "gptq0.1" and B.method(3, "dn") == "rtne" and B.uses_cal()
    assert B.round_experts(5, "gu", W) is None and B.tag(5) == ""
    assert B.tag(3) != B.tag(0) and on_grid(B.round_experts(3, "gu", W))
    Wb = W.bfloat16()
    assert B.round_experts(3, "gu", Wb).dtype == torch.bfloat16
    (tmp_path / "d.json").write_text(json.dumps({"default": {"gu": "rtne", "dn": "gptq0.1"}, "layers": {}}))
    monkeypatch.setenv("HYB_BFP4_ROUND", str(tmp_path / "d.json"))
    assert B.method(7, "gu") == "rtne" and B.method(7, "dn") == "gptq0.1"
    with pytest.raises(ValueError):  # the layer-wide Hessian is gate|up's input only
        B.round_experts(3, "dn", W)


def test_gptqe_per_expert(tmp_path, monkeypatch):
    """gptqe: each expert's GPTQ uses only the tokens routed to it (down: through its unrounded gate|up); an expert with
    too few tokens falls back to rtne; a subset of experts (build_dev_experts) rounds exactly like the whole layer."""
    torch.manual_seed(2)
    H, N, T = 32, 16, 600
    h = (torch.randn(T, H) @ (torch.randn(H, H) * 0.3 + torch.eye(H))).bfloat16()
    idx = torch.zeros(T, 2, dtype=torch.uint8)
    idx[:, 0], idx[:, 1] = 0, 3  # experts 0 and 3 see every token, 1 sees 10 (< MIN_TOK), 2 none
    idx[:10, 1] = 1
    torch.save({"h": h, "idx": idx}, tmp_path / "cal_4.pt")
    (tmp_path / "e.json").write_text(json.dumps({"default": {"gu": "gptqe0.1", "dn": "gptqe0.3"}, "layers": {}}))
    monkeypatch.setenv("HYB_BFP4_ROUND", str(tmp_path / "e.json"))
    monkeypatch.setenv("HYB_BFP4_CAL", str(tmp_path))
    gu = torch.randn(4, 2 * N, H) * 0.05
    dn = torch.randn(4, H, N) * 0.05
    qg = B.round_experts(4, "gu", gu)
    qd = B.round_experts(4, "dn", dn, gate_up=gu)
    assert on_grid(qg) and on_grid(qd)
    for e in (1, 2):
        assert torch.equal(qg[e], B.rtne(gu[e : e + 1])[0]) and torch.equal(qd[e], B.rtne(dn[e : e + 1])[0])
    x = h.float()
    a = torch.nn.functional.silu(x @ gu[0, :N].t()) * (x @ gu[0, N:].t())
    err = lambda M, W, inp: ((inp @ (M - W).t()) ** 2).sum()  # noqa: E731
    assert err(qg[0], gu[0], x) < err(B.rtne(gu[:1])[0], gu[0], x)
    assert err(qd[0], dn[0], a) < err(B.rtne(dn[:1])[0], dn[0], a)
    sub = B.round_experts(4, "dn", dn[[3]], experts=[3], gate_up=gu[[3]])
    assert torch.allclose(sub[0], qd[3])
    with pytest.raises(ValueError):
        B.round_experts(4, "dn", dn)  # down without the gate|up it reads
