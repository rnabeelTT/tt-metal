# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Warm prefill time (TTFT without sampling) per prompt length: each length runs twice, the second is reported.

    python -m models.autoports.tencent_hy4_flash.tools.prefill_bench --lens 200,1000,2000,4000 [--ctx 32768]
        [--profile 2048]   (cProfile the host side of one warm 2048-token prefill, top functions)
"""

from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import time

import torch
import ttnn
from loguru import logger


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lens", default="200,1000,2000,4000")
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--layers", default="")
    ap.add_argument("--no-engram", action="store_true")
    ap.add_argument("--profile", type=int, default=0)
    ap.add_argument("--reps", type=int, default=2)
    args = ap.parse_args()
    from models.autoports.tencent_hy4_flash.tools.generate import build, open_mesh

    mesh = open_mesh()
    try:
        model = build(mesh, args)
        g = torch.Generator().manual_seed(0)
        res = {}
        for L in [int(x) for x in args.lens.split(",")]:
            ids = torch.randint(1000, 100000, (L,), generator=g)
            ts = []
            for r in range(args.reps):
                model.reset()
                t0 = time.time()
                lg = model.prefill(ids, args.chunk)
                ttnn.synchronize_device(mesh)
                ts.append(time.time() - t0)
            res[L] = ts[-1]
            logger.info(f"prefill {L} tokens: {[round(t, 2) for t in ts]} s -> {L / ts[-1]:.0f} tok/s")
        if args.profile:
            ids = torch.randint(1000, 100000, (args.profile,), generator=g)
            model.reset()
            pr = cProfile.Profile()
            pr.enable()
            model.prefill(ids, args.chunk)
            ttnn.synchronize_device(mesh)
            pr.disable()
            s = io.StringIO()
            pstats.Stats(pr, stream=s).sort_stats("tottime").print_stats(35)
            print(s.getvalue()[:9000])
        print("RESULT " + " ".join(f"{L}:{t:.2f}s" for L, t in res.items()))
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
