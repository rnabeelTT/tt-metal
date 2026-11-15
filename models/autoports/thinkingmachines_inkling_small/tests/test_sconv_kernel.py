# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The fused short-conv programs (tt/sconv_kernels) against the fp32 host reference and the ttnn-op path: verify rows
(R = 1..4) + commit(n), decode steps; then us per call, traced, both paths. One device.
usage: python tests/test_sconv_kernel.py"""
import os
import time

import torch

import ttnn

os.environ["INKLING_SCONV_KERNEL"] = "1"
from models.autoports.thinkingmachines_inkling_small.tt import sconv as S  # noqa: E402

dev = ttnn.open_device(device_id=0, trace_region_size=100_000_000)
ok = True


def bf(t):
    return t.to(torch.bfloat16).float()


def host(t):
    return ttnn.to_torch(t).float()


def timed(f, n=200):
    f()
    tid = ttnn.begin_trace_capture(dev, cq_id=0)
    f()
    ttnn.end_trace_capture(dev, tid, cq_id=0)
    ttnn.execute_trace(dev, tid, cq_id=0, blocking=True)
    t = time.perf_counter()
    for _ in range(n):
        ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(dev)
    ttnn.release_trace(dev, tid)
    return (time.perf_counter() - t) / n * 1e6


try:
    torch.manual_seed(0)
    for C in (1024, 256):
        w = bf(torch.randn(C, 4) * 0.5)
        sc = S.ShortConv(dev, w, None, C)
        assert sc.fused
        state = None
        sc.reset()
        for step, (R, n) in enumerate([(1, 1), (3, 2), (4, 4), (2, 1), (4, 3)]):
            x = bf(torch.randn(R, C) * 3)
            xt = ttnn.from_torch(x.reshape(1, 1, R, C), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
            y = host(sc.rows(xt, commit_all=False)).reshape(R, C)
            ref, _ = S.host_reference(x, w, state)
            sc.commit_rows(n)
            _, state = S.host_reference(x[:n], w, state)
            st = host(sc.st).reshape(3, C)
            ey = (y - bf(ref)).abs().max().item()
            es = (st - state).abs().max().item()
            good = ey <= 2 ** -6 * ref.abs().max().item() and es == 0
            ok &= good
            print(f"C {C} step {step} R {R} commit {n}: max|y - ref| {ey:.3g} (|ref| {ref.abs().max():.3g}), "
                  f"state diff {es:.3g} {'ok' if good else 'FAIL'}", flush=True)
        for _ in range(3):  # decode steps
            x = bf(torch.randn(1, C) * 3)
            xt = ttnn.from_torch(x.reshape(1, 1, 1, C), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
            y = host(sc.decode(xt)).reshape(1, C)
            ref, state = S.host_reference(x, w, state)
            ey = (y - bf(ref)).abs().max().item()
            es = (host(sc.st).reshape(3, C) - state).abs().max().item()
            good = ey <= 2 ** -6 * ref.abs().max().item() and es == 0
            ok &= good
            print(f"C {C} decode: max|y - ref| {ey:.3g}, state diff {es:.3g} {'ok' if good else 'FAIL'}", flush=True)
        for R in (1, 3, 4):
            xt = ttnn.from_torch(torch.randn(1, 1, R, C), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)

            def fk(sc=sc, xt=xt):
                y = sc.rows(xt, commit_all=False)
                sc.commit_rows(1)
                return y

            sc.fused = False
            t_old = timed(fk)
            sc.fused = True
            t_new = timed(fk)
            print(f"C {C} R {R}: rows + commit {t_old:.1f} us (ttnn ops) -> {t_new:.1f} us (fused)", flush=True)
    print("ALL OK" if ok else "SOME FAILED")
finally:
    ttnn.close_device(dev)
