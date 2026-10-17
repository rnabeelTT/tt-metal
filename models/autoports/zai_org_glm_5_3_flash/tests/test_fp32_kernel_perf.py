# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Rough device time of the fp32 custom kernels at decode shapes (T = 32 tokens / B = 32 users): a synchronized loop of
back-to-back launches, wall time per call. Only a before/after comparison of the kernels' own arithmetic; not a
profiler measurement."""

import time

import pytest
import torch

import ttnn
from models.autoports.zai_org_glm_5_3_flash.tt.fused_decoder import mix_chunks
from models.autoports.zai_org_glm_5_3_flash.tt.mhc import HcConfig
from models.demos.deepseek_v3_d_p.reference.mhc.mhc_reference import sinkhorn_knopp
from models.demos.deepseek_v3_d_p.tt.mhc.tt_mhc import build_bcast_consts, build_consts

N, C, T, ITERS = 4, 4096, 32, 200


def _time(device, fn):
    warm = [fn() for _ in range(5)]
    ttnn.synchronize_device(device)
    del warm
    outs = []
    t0 = time.perf_counter()
    for _ in range(ITERS):
        outs.append(fn())
    ttnn.synchronize_device(device)
    dt = (time.perf_counter() - t0) / ITERS * 1e6
    return dt


def _up(device, t):
    return ttnn.from_torch(t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)


@pytest.mark.parametrize("op", ["kda_recurrent_decode", "mhc_split_sinkhorn", "mhc_post", "mhc_pre", "mhc_mix"])
def test_fp32_kernel_perf(device, op):
    g = torch.Generator().manual_seed(0)
    if op == "kda_recurrent_decode":
        B, H, K = 32, 64, 128
        S = _up(device, torch.randn(B, H, K, K, generator=g) * 0.05)
        qkv = _up(device, torch.randn(1, 1, B, 3 * H * K, generator=g))
        decay = _up(device, torch.rand(1, 1, B, H * K, generator=g) * 0.1 + 0.9)
        beta = _up(device, torch.sigmoid(torch.randn(1, 1, B, H, generator=g)))
        fn = lambda: ttnn.experimental.kda.recurrent_decode(S, qkv, decay, beta, K**-0.5)
    elif op == "mhc_split_sinkhorn":
        cfg = HcConfig(n=N, sinkhorn_iters=20, eps=1e-6, norm_eps=1e-5)
        consts = _up(device, build_consts(cfg, torch.ones(3), torch.randn(24, generator=g)).reshape(8, 32, 32))
        mixes = _up(device, torch.randn(T, 24, generator=g))
        fn = lambda: ttnn.experimental.deepseek_prefill.mhc_split_sinkhorn(mixes, consts, N, 20, 1e-6)
    elif op == "mhc_post":
        consts = _up(device, build_bcast_consts(N).reshape(N * N, 32, 32))
        y = _up(device, torch.randn(1, 1, T, C, generator=g))
        res = _up(device, torch.randn(1, 1, T, N * C, generator=g))
        post = _up(device, 2 * torch.sigmoid(torch.randn(1, 1, T, N, generator=g)))
        comb = _up(device, sinkhorn_knopp(torch.randn(T, N, N, generator=g), 20, 1e-6).reshape(1, 1, T, N * N))
        fn = lambda: ttnn.experimental.deepseek_prefill.mhc_post(y, res, post, comb, consts, N)
    elif op == "mhc_pre":
        consts = _up(device, build_bcast_consts(N).reshape(N * N, 32, 32))
        res = _up(device, torch.randn(1, 1, T, N * C, generator=g))
        pre = _up(device, torch.sigmoid(torch.randn(1, 1, T, N, generator=g)))
        fn = lambda: ttnn.experimental.deepseek_prefill.mhc_pre(res, pre, consts, N)
    else:
        Kd = N * C
        x = _up(device, torch.randn(1, 1, T, Kd, generator=g))
        fnw = _up(device, torch.randn(Kd, 32, generator=g) * 0.01)
        chunks = mix_chunks(T, Kd // 32)
        fn = lambda: ttnn.experimental.deepseek_prefill.mhc_mix(x, fnw, chunks)
    us = _time(device, fn)
    print(f"PERF {op}: {us:.1f} us/call (synchronized loop of {ITERS})")
