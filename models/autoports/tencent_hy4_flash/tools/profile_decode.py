# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Time the model build, a prefill and warm decode steps; cProfile the decode steps (host view).

    python -m models.autoports.tencent_hy4_flash.tools.profile_decode [--steps 16] [--layers 0,1,2] [--prompt-len 200]
"""

from __future__ import annotations

import argparse
import cProfile
import io
import os
import pstats
import time

import torch
import ttnn
from loguru import logger


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--layers", default="")
    ap.add_argument("--prompt-len", type=int, default=200)
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--no-engram", action="store_true")
    args = ap.parse_args()
    from models.autoports.tencent_hy4_flash.tools.generate import build, open_mesh

    mesh = open_mesh()
    try:
        t0 = time.time()
        model = build(mesh, args)
        logger.info(f"build {time.time() - t0:.1f} s")
        ids = torch.randint(1000, 100000, (args.prompt_len,))
        for rnd in range(2):
            model.reset()
            t0 = time.time()
            for c0 in range(0, ids.numel(), args.chunk):
                lg = model.forward(ids[c0 : c0 + args.chunk])
            ttnn.synchronize_device(mesh)
            logger.info(f"prefill round {rnd}: {ids.numel()} tokens {time.time() - t0:.2f} s")
            ts = []
            tok = int(torch.argmax(lg))
            pr = cProfile.Profile()
            for i in range(args.steps):
                if rnd == 1 and i == 2:
                    pr.enable()
                t1 = time.time()
                lg = model.forward(torch.tensor([tok]))
                tok = int(torch.argmax(lg))
                ts.append(time.time() - t1)
            pr.disable()
            logger.info(f"decode round {rnd}: per step ms {[round(t * 1000) for t in ts]}")
        s = io.StringIO()
        pstats.Stats(pr, stream=s).sort_stats("tottime").print_stats(30)
        print(s.getvalue()[:8000])
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
