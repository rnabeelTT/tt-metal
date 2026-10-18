# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Sinkhorn permutation-sum fidelity (MHC_SINKHORN_PERM_FIDELITY = 4 / 3 / 2) and the face-restricted normalizations
(MHC_SINKHORN_FACES = 1): bit-identity of pre / post / comb against the HiFi4 (all-faces) kernel on the same inputs,
error against the fp64 reference, and the device time of one decode-sized call (one token tile, one core) from traced
replays. HiFi2 is not exact and needs MHC_SINKHORN_ALLOW_INEXACT=1 (set here).

  python -m models.autoports.zai_org_glm_5_3_flash.tools.sinkhorn_fidelity_probe
"""

from __future__ import annotations

import json
import os
import time

import torch

import ttnn
from models.autoports.zai_org_glm_5_3_flash.tt.mhc import HcConfig
from models.demos.deepseek_v3_d_p.reference.mhc.mhc_reference import MHCConfig, parametrize
from models.demos.deepseek_v3_d_p.tt.mhc.tt_mhc import build_consts

N, ITERS, EPS = 4, 20, 1e-6


def run(dev, mixes, consts):
    up = lambda t: ttnn.from_torch(t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
    outs = ttnn.experimental.deepseek_prefill.mhc_split_sinkhorn(up(mixes), up(consts), N, ITERS, EPS)
    return [ttnn.to_torch(o).float() for o in outs]


def timed(dev, mixes, consts, reps=200):
    up = lambda t: ttnn.from_torch(t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
    m, c = up(mixes), up(consts)
    o = ttnn.experimental.deepseek_prefill.mhc_split_sinkhorn(m, c, N, ITERS, EPS)  # compile
    for t in o:
        ttnn.deallocate(t)
    tid = ttnn.begin_trace_capture(dev, cq_id=0)
    o = ttnn.experimental.deepseek_prefill.mhc_split_sinkhorn(m, c, N, ITERS, EPS)
    ttnn.end_trace_capture(dev, tid, cq_id=0)
    ttnn.execute_trace(dev, tid, cq_id=0, blocking=True)
    t0 = time.time()
    for _ in range(reps):
        ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(dev)
    us = (time.time() - t0) / reps * 1e6
    ttnn.release_trace(dev, tid)
    return us


def main():
    dev = ttnn.open_device(device_id=0, trace_region_size=16 << 20)
    os.environ["MHC_SINKHORN_ALLOW_INEXACT"] = "1"
    res = []
    try:
        for T in (1, 8, 16, 17, 32, 96, 2048):
            g = torch.Generator().manual_seed(T)
            mixes = torch.randn(T, (2 + N) * N, generator=g)
            scale = (torch.rand(3, generator=g) * 0.5 + 0.5).bfloat16().float()
            base = (torch.randn((2 + N) * N, generator=g) * 0.5).bfloat16().float()
            consts = build_consts(HcConfig(n=N, sinkhorn_iters=ITERS, eps=EPS, norm_eps=1e-5), scale, base).reshape(
                8, 32, 32
            )
            pre_r, post_r, comb_r = parametrize(
                mixes.double(), scale.double(), base.double(), MHCConfig(n=N, sinkhorn_iters=ITERS, eps=EPS)
            )
            ref = [pre_r, post_r, comb_r.reshape(T, N * N)]
            outs = {}
            for f in ("4", "3", "2", "3f"):
                os.environ["MHC_SINKHORN_PERM_FIDELITY"] = f[0]
                os.environ["MHC_SINKHORN_FACES"] = "1" if f.endswith("f") else "0"
                got = run(dev, mixes, consts)
                again = run(dev, mixes, consts)  # a second call: no stale DST state may change the result
                got = [got[0][:T, :N], got[1][:T, :N], got[2][:T, : N * N]]
                outs[f] = got
                err = max(float((g_.double() - r).abs().max() / r.abs().max()) for g_, r in zip(got, ref))
                rec = {
                    "T": T,
                    "perm_fidelity": int(f[0]),
                    "faces": f.endswith("f"),
                    "repeat_identical": all(
                        torch.equal(a, b)
                        for a, b in zip(got, [again[0][:T, :N], again[1][:T, :N], again[2][:T, : N * N]])
                    ),
                    "finite": all(torch.isfinite(g_).all().item() for g_ in got),
                    "max_rel_err_vs_fp64": err,
                    "bit_identical_to_hifi4": all(torch.equal(a, b) for a, b in zip(got, outs["4"])),
                    "us_per_call": timed(dev, mixes, consts) if T in (1, 32) else None,
                }
                print("SK", json.dumps(rec))
                res.append(rec)
    finally:
        for k in ("MHC_SINKHORN_PERM_FIDELITY", "MHC_SINKHORN_FACES", "MHC_SINKHORN_ALLOW_INEXACT"):
            os.environ.pop(k, None)
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
