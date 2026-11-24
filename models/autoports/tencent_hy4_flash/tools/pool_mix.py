# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Mixed long / short serving sequence on the batch engine (shared pool, tt/pool.py), served-like (warm-up + capture
first): per phase decode tok/s and the program-cache entries / trace captures it added (a serve-time compile releases
every trace). HY4_PROG_TRACE=1 names each serve-time program.

    HY4_POOL_TOKENS=172032 HY4_SPEC_K=4 python -m models.autoports.tencent_hy4_flash.tools.pool_mix --ctx 161792 \
        --phases 32768,131072,8x64,8x64
"""

from __future__ import annotations

import argparse
import os
import time

import torch
import ttnn
from loguru import logger


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", type=int, default=8)
    ap.add_argument("--ctx", type=int, default=161792)
    ap.add_argument("--phases", default="32768,131072,8x64,8x64", help="N: one user with an N-token prompt; UxN: U users")
    ap.add_argument("--steps", type=int, default=96)
    args = ap.parse_args()
    from models.autoports.tencent_hy4_flash.tools.generate import open_mesh
    from models.autoports.tencent_hy4_flash.tt.config import Hy4Config, checkpoint_dir
    from models.autoports.tencent_hy4_flash.tt.engine_batch import Hy4BatchEngine
    from models.autoports.tencent_hy4_flash.tt.model import Hy4Model
    from models.experimental.hybrid_moe.engram_store import EngramRows

    mesh = open_mesh()
    try:
        engram = EngramRows(os.environ.get("HY4_ENGRAM_DIR", os.path.expanduser("~/.cache/hybrid-moe/hy4_engram")),
                            checkpoint_dir())
        model = Hy4Model(mesh, Hy4Config.load(), max_ctx=args.ctx + 64, chunk=2048, engram=engram,
                         cache_dir=os.environ.get("HY4_TENSOR_CACHE", os.path.expanduser("~/.cache/hybrid-moe/hy4_tt")),
                         slots=args.users)
        if hasattr(engram.store, "wait_ram"):
            engram.store.wait_ram()
        eng = Hy4BatchEngine(model, k=int(os.environ.get("HY4_SPEC_K", "0") or 0))
        eng.warm()
        eng.capture()
        dec = eng.dec
        g = torch.Generator().manual_seed(3)
        live = []
        for ph in args.phases.split(","):
            for s in live:
                eng.finish(s)
            u, n = (int(x) for x in ph.split("x")) if "x" in ph else (1, int(ph))
            p0, c0 = int(mesh.num_program_cache_entries()), len(dec.traces) + len(dec.btraces)
            t0 = time.time()
            live = []
            reqs = []
            for _ in range(u):
                s = eng.alloc(taken=live)
                live.append(s)
                reqs.append((s, torch.randint(1000, 100000, (n,), generator=g).tolist(), None))
            eng.prefill_many(reqs) if u > 1 else eng.prefill(reqs[0][0], reqs[0][1])
            tp = time.time() - t0
            p1 = int(mesh.num_program_cache_entries())
            toks, t1 = 0, time.time()
            for i in range(args.steps):
                if i == 8:
                    toks, t1 = 0, time.time()
                out = eng.step({s: (None, None, False) for s in live})
                toks += sum(len(v) for v in out.values())
            dt = time.time() - t1
            logger.info(f"POOL_MIX phase {ph}: prefill {tp:.2f} s, decode {toks / dt:.1f} tok/s total "
                        f"({toks / dt / u:.1f} per user); programs +{p1 - p0} prefill +"
                        f"{int(mesh.num_program_cache_entries()) - p1} decode; traces {c0} -> "
                        f"{len(dec.traces) + len(dec.btraces)}; ext {dict(model.ext)}; {eng.pool_stats}")
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
