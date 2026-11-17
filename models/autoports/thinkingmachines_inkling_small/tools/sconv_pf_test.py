# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""ShortConv.prefill_commit (sconv_kernels/pf_*.cpp) vs the ttnn-op path it replaces (rows + commit_final), on the
device: outputs and states compared bit for bit over consecutive chunks with random states and a padded last chunk,
at the K/V width and the hidden width per die; then the host time per call (queued, no wait) and the traced device
time per call of both paths.

usage (in the image): tools/run_dev.sh python models/autoports/thinkingmachines_inkling_small/tools/sconv_pf_test.py
"""

import sys
import time

import torch
import ttnn

from models.autoports.thinkingmachines_inkling_small.tt import sconv as S
from models.autoports.thinkingmachines_inkling_small.tt.sconv import ShortConv


def old_path(sc, x, n):
    R = int(x.shape[2])
    pf, S.PF_KERNEL = S.PF_KERNEL, False
    try:
        return sc.prefill_commit(x, n)
    finally:
        S.PF_KERNEL = pf
    del R


def host(t):
    return ttnn.to_torch(ttnn.get_device_tensors(t)[0]).float()


def timed(mesh, fn, reps=50):
    fn()
    ttnn.synchronize_device(mesh)
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    t_host = (time.perf_counter() - t0) / reps
    ttnn.synchronize_device(mesh)
    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
    fn()
    ttnn.end_trace_capture(mesh, tid, cq_id=0)
    ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
    t0 = time.perf_counter()
    for _ in range(reps):
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh)
    t_dev = (time.perf_counter() - t0) / reps
    ttnn.release_trace(mesh, tid)
    return t_host, t_dev


def main():
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), l1_small_size=24576, trace_region_size=200_000_000)
    rep = ttnn.ReplicateTensorToMesh(mesh)
    bad = 0
    try:
        torch.manual_seed(0)
        for C in (256, 1024):
            w = (torch.randn(C, 4) * 0.3).bfloat16().float()
            a, b = ShortConv(mesh, w, rep, C), ShortConv(mesh, w, rep, C)
            st = torch.randn(1, 1, 3, C).bfloat16().float()
            for sc in (a, b):
                ttnn.copy_host_to_device_tensor(
                    ttnn.from_torch(st, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=rep), sc.st
                )
            for R, n in ((1536, None), (1536, None), (128, None), (1536, 1000), (128, 37), (128, 2), (1536, 1)):
                xt = torch.randn(1, 1, R, C) * 2
                x = ttnn.from_torch(xt, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep)
                st0 = host(a.st).reshape(3, C)
                ya = a.prefill_commit(x, n)
                yb = old_path(b, x, n)
                ref, _ = S.host_reference(xt.bfloat16().float().reshape(R, C), w, st0)
                rb = ref.bfloat16().float().reshape(1, 1, R, C)
                print(
                    f"   vs the fp32 reference rounded to bf16 (RNE): kernel mismatches "
                    f"{(host(ya) != rb).sum().item()}, ttnn ops mismatches {(host(yb) != rb).sum().item()}",
                    flush=True,
                )
                da = (host(ya) != host(yb)).sum().item()
                ds = (host(a.st) != host(b.st)).sum().item()
                mx = (host(ya) - host(yb)).abs().max().item()
                ok = da == 0 and ds == 0
                bad += not ok
                print(
                    f"C={C} R={R} n={n}: y mismatches {da} (max |d| {mx:.3g}), state mismatches {ds} "
                    f"{'OK' if ok else 'DIFF'}",
                    flush=True,
                )
                # resync the states so the next case starts equal even after a mismatch
                ttnn.copy(b.st, a.st)
            for R in (1536,):
                x = ttnn.from_torch(
                    torch.randn(1, 1, R, C), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep
                )
                h1, d1 = timed(mesh, lambda: ttnn.deallocate(a.prefill_commit(x)))
                h0, d0 = timed(mesh, lambda: ttnn.deallocate(old_path(b, x, None)))
                print(
                    f"C={C} R={R}: kernel host {1e6 * h1:.0f} us / device {1e6 * d1:.0f} us | ttnn ops host "
                    f"{1e6 * h0:.0f} us / device {1e6 * d0:.0f} us",
                    flush=True,
                )
        print(f"RESULT {'PASS' if bad == 0 else f'FAIL ({bad} cases)'}", flush=True)
    except Exception:  # noqa: BLE001 - printed before the hard exit below
        import traceback

        traceback.print_exc()
    finally:
        sys.stdout.flush()
        import os

        os._exit(0)


if __name__ == "__main__":
    main()
