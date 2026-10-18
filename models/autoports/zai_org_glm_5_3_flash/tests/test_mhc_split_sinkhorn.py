# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""ttnn.experimental.deepseek_prefill.mhc_split_sinkhorn (mHC parametrization: pre / post / Sinkhorn comb) vs the fp32
torch reference (models/demos/deepseek_v3_d_p/reference/mhc/mhc_reference.py::parametrize, evaluated in fp64 on the
identical fp32 inputs). The kernel's fp32 operands must keep full precision (no TF32 reads), so the bar is ~1e-6
relative."""

import pytest
import torch

import ttnn
from models.autoports.zai_org_glm_5_3_flash.tt.mhc import HcConfig
from models.demos.deepseek_v3_d_p.reference.mhc.mhc_reference import MHCConfig, parametrize
from models.demos.deepseek_v3_d_p.tt.mhc.tt_mhc import build_consts

N, ITERS, HC_EPS = 4, 20, 1e-6


def _pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm()))


def _stats(got, ref):
    got, ref = got.double().flatten(), ref.double().flatten()
    err = (got - ref).abs()
    return dict(
        pcc=_pcc(got, ref),
        scale=float((got @ ref) / (ref @ ref)),
        max_rel=float(err.max() / ref.abs().max()),
        max_elem_rel=float((err / ref.abs().clamp_min(1e-30)).max()),
    )


@pytest.mark.parametrize("bf16_scale", [True, False], ids=["bf16_scale", "fp32_scale"])
@pytest.mark.parametrize("T", [32, 96, 2048])
def test_mhc_split_sinkhorn(device, T, bf16_scale):
    g = torch.Generator().manual_seed(T + bf16_scale)
    mix = (2 + N) * N
    # mixes as the model makes them (rms-normalized projection: O(1) logits), a / b like the GLM checkpoint
    mixes = torch.randn(T, mix, generator=g)
    scale = torch.rand(3, generator=g) * 0.5 + 0.5
    base = torch.randn(mix, generator=g) * 0.5
    if bf16_scale:  # GLM rounds base / scale to bf16 at load
        scale, base = scale.bfloat16().float(), base.bfloat16().float()
    cfg = HcConfig(n=N, sinkhorn_iters=ITERS, eps=HC_EPS, norm_eps=1e-5)
    consts = build_consts(cfg, scale, base)
    up = lambda t: ttnn.from_torch(t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    pre, post, comb = ttnn.experimental.deepseek_prefill.mhc_split_sinkhorn(
        up(mixes), up(consts.reshape(8, 32, 32)), N, ITERS, HC_EPS
    )
    rcfg = MHCConfig(n=N, sinkhorn_iters=ITERS, eps=HC_EPS)
    pre_r, post_r, comb_r = parametrize(mixes.double(), scale.double(), base.double(), rcfg)
    got = {
        "pre": ttnn.to_torch(pre).float()[:T, :N],
        "post": ttnn.to_torch(post).float()[:T, :N],
        "comb": ttnn.to_torch(comb).float()[:T, : N * N],
    }
    ref = {"pre": pre_r, "post": post_r, "comb": comb_r.reshape(T, N * N)}
    worst = 0.0
    for k in ("pre", "post", "comb"):
        s = _stats(got[k], ref[k])
        worst = max(worst, s["max_rel"])
        print(
            f"T={T} {'bf16' if bf16_scale else 'fp32'} a/b {k}: PCC {s['pcc']:.9f} scale {s['scale']:.9f} "
            f"max rel err {s['max_rel']:.2e} max elementwise rel err {s['max_elem_rel']:.2e}"
        )
    # doubly stochastic: column sums exactly 1 after the final column normalization
    cs = got["comb"].double().reshape(T, N, N).sum(-2)
    print(f"T={T}: comb column-sum max |1 - s| {(cs - 1).abs().max().item():.2e}")
    assert worst < 1e-5, worst


def test_mhc_split_sinkhorn_sharded(device):
    """L1 height-sharded mixes (the zero-copy path: CBs alias the shards) must match the interleaved path bit-for-bit."""
    g = torch.Generator().manual_seed(5)
    grid = device.compute_with_storage_grid_size()
    T = grid.x * grid.y * 2 * 32
    mix = (2 + N) * N
    mixes = torch.zeros(T, 32)
    mixes[:, :mix] = torch.randn(T, mix, generator=g)
    scale, base = torch.rand(3, generator=g) * 0.5 + 0.5, torch.randn(mix, generator=g) * 0.5
    cfg = HcConfig(n=N, sinkhorn_iters=ITERS, eps=HC_EPS, norm_eps=1e-5)
    consts = ttnn.from_torch(
        build_consts(cfg, scale, base).reshape(8, 32, 32), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device
    )
    mem = ttnn.create_sharded_memory_config(
        [T, 32], ttnn.CoreGrid(y=grid.y, x=grid.x), ttnn.ShardStrategy.HEIGHT, ttnn.ShardOrientation.ROW_MAJOR
    )
    sharded = ttnn.from_torch(mixes, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem)
    inter = ttnn.from_torch(mixes[:, :mix], dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    a = ttnn.experimental.deepseek_prefill.mhc_split_sinkhorn(sharded, consts, N, ITERS, HC_EPS)
    b = ttnn.experimental.deepseek_prefill.mhc_split_sinkhorn(inter, consts, N, ITERS, HC_EPS)
    rcfg = MHCConfig(n=N, sinkhorn_iters=ITERS, eps=HC_EPS)
    pre_r, post_r, comb_r = parametrize(mixes[:, :mix].double(), scale.double(), base.double(), rcfg)
    for k, x, y, w, ref in zip(("pre", "post", "comb"), a, b, (N, N, N * N), (pre_r, post_r, comb_r.reshape(T, -1))):
        xs, ys = ttnn.to_torch(x).float()[:T, :w], ttnn.to_torch(y).float()[:T, :w]
        s = _stats(xs, ref)
        print(
            f"sharded T={T} {k}: max |sharded - interleaved| {(xs - ys).abs().max().item():.2e}, "
            f"max rel err vs fp64 {s['max_rel']:.2e}"
        )
        assert torch.equal(xs, ys) and s["max_rel"] < 1e-5


@pytest.mark.parametrize("T", [1, 8, 16, 17, 32])
def test_mhc_split_sinkhorn_faces(device, monkeypatch, T):
    """MHC_SINKHORN_FACES=1 (the decode path that ships: face 0 at T <= 16, faces 0 / 2 above) gives the same used
    entries as the all-faces kernel, bit for bit, at HiFi3 permutation sums (the shipped fidelity)."""
    g = torch.Generator().manual_seed(100 + T)
    mix = (2 + N) * N
    mixes = torch.randn(T, mix, generator=g)
    scale = (torch.rand(3, generator=g) * 0.5 + 0.5).bfloat16().float()
    base = (torch.randn(mix, generator=g) * 0.5).bfloat16().float()
    consts = build_consts(HcConfig(n=N, sinkhorn_iters=ITERS, eps=HC_EPS, norm_eps=1e-5), scale, base)
    up = lambda t: ttnn.from_torch(t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    monkeypatch.setenv("MHC_SINKHORN_PERM_FIDELITY", "3")
    outs = {}
    for faces in ("0", "1", "1"):  # faces on twice: a repeat call must not see stale DST state
        monkeypatch.setenv("MHC_SINKHORN_FACES", faces)
        pre, post, comb = ttnn.experimental.deepseek_prefill.mhc_split_sinkhorn(
            up(mixes), up(consts.reshape(8, 32, 32)), N, ITERS, HC_EPS
        )
        got = [
            ttnn.to_torch(pre).float()[:T, :N],
            ttnn.to_torch(post).float()[:T, :N],
            ttnn.to_torch(comb).float()[:T, : N * N],
        ]
        outs.setdefault(faces, []).append(got)
    ref = outs["0"][0]
    for run in outs["1"]:
        for name, a, b in zip(("pre", "post", "comb"), ref, run):
            assert torch.isfinite(b).all(), (T, name)
            assert torch.equal(a, b), (T, name, float((a - b).abs().max()))
