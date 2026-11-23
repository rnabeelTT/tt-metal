# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Batched decode quality gate + timing: every text a user (its own slot / lane), the first P tokens prefilled, the
rest teacher-forced through batched passes (one row per user per pass). Writes OUT_DIR/dev_<TAG>_<name>.npz like
tools/teacher_force.py (compare with models/experimental/hybrid_moe/compare_hy4.py).

    python -m models.autoports.tencent_hy4_flash.tools.batch_gate TEXTS REF_DIR OUT_DIR TAG --decode-from 32
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import ttnn
from loguru import logger


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("texts")
    ap.add_argument("ref_dir")
    ap.add_argument("out_dir")
    ap.add_argument("tag")
    ap.add_argument("--names", default="")
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--layers", default="")
    ap.add_argument("--no-engram", action="store_true")
    ap.add_argument("--decode-from", type=int, default=32)
    ap.add_argument("--users", type=int, default=0, help="users (texts cycled; default: one per text)")
    ap.add_argument("--pack", action="store_true", help="re-prefill every user in one packed pass before decoding "
                    "(the prefill positions' scores stay the serial ones; decode reads the packed caches)")
    ap.add_argument("--rows", type=int, default=1, help="rows per user per pass (verify-like: consecutive tokens)")
    ap.add_argument("--group", type=int, default=0, help="users per pass (default all; groups take turns)")
    ap.add_argument("--frag", action="store_true", help="shared pool (tt/pool.py) with fragmented extents (reverse "
                    "order, 4096-token holes); a third into the decode the largest extent moves to the free tail, two "
                    "thirds in the pool is fully compacted (overlapping slides): outputs must be bit-identical")
    args = ap.parse_args()
    seqs = json.load(open(args.texts))
    names = [n for n in (args.names.split(",") if args.names else sorted(seqs))]
    U = args.users or len(names)
    users = [names[u % len(names)] for u in range(U)]
    os.environ["HY4_SLOTS"] = str(U)
    if args.frag:
        from models.autoports.tencent_hy4_flash.tt.pool import rup

        need = [rup(max(len(seqs[n]), args.decode_from + 2048) + 384 + 2048) for n in users]
        os.environ["HY4_POOL_TOKENS"] = str(rup(sum(need) + 4096 * U + max(need), 2048))
    from models.autoports.tencent_hy4_flash.tools.generate import build, open_mesh
    from models.autoports.tencent_hy4_flash.tt.decode_batch import Hy4BatchDecoder

    os.makedirs(args.out_dir, exist_ok=True)
    mesh = open_mesh()
    try:
        model = build(mesh, args)
        if model.engram is not None and hasattr(model.engram.store, "wait_ram"):
            model.engram.store.wait_ram()
        dec = Hy4BatchDecoder(model, lanes=max(8, U), trace=True)
        P = args.decode_from
        po = None
        if args.frag:
            from models.autoports.tencent_hy4_flash.tt.pool import PoolOps

            assert model.pooled, "--frag: the model is not pooled"
            po = PoolOps(model, dec)
            po.warm()
            b = 0
            for u in reversed(range(U)):  # slot U-1 first: bases in reverse slot order, a hole after each
                model.ext[u] = (b, need[u])
                b += need[u] + 4096
            logger.info(f"frag: pool {model.pool_tokens}, extents {dict(model.ext)}")
        ids, lps, lane = [], [], []
        for u, name in enumerate(users):
            t = torch.tensor(seqs[name], dtype=torch.long)
            ids.append(t)
            model.set_slot(u)
            model.reset()
            out = []
            for c0 in range(0, P, args.chunk):
                lg = model.forward(t[c0 : min(P, c0 + args.chunk)], logits_rows="all")
                out.append(torch.log_softmax(lg.float(), dim=-1))
            lps.append(out)
            lane.append(dec.add(u, P, t[:P].tolist()))
        if args.pack:
            plan = model.pack_plan([P] * U) or (1 << (U - 1).bit_length(), 128)
            for l in lane:
                dec.drop(l)
            t0 = time.perf_counter()
            model.forward_packed([(u, ids[u][:P]) for u in range(U)], plan)
            logger.info(f"packed prefill of {U} x {P} tokens (plan {plan}) in {time.perf_counter() - t0:.2f} s")
            lane = [dec.add(u, P, ids[u][:P].tolist()) for u in range(U)]
        steps = max(int(t.numel()) for t in ids) - P
        times = []
        N, G = max(1, args.rows), args.group or U
        for i in range(0, steps, N):
            if po is not None and i in (steps // 3 // N * N, 2 * steps // 3 // N * N) and i > 0:
                ttnn.synchronize_device(mesh)
                if i == steps // 3 // N * N:
                    u = max(range(U), key=lambda v: model.ext[v][1])
                    nb = po.pa.fit(model.ext[u][1])
                    logger.info(f"frag: step {i}: slot {u} {model.ext[u]} -> base {nb}")
                    po.move(u, nb)
                else:
                    logger.info(f"frag: step {i}: compaction {po.pa.compaction()}")
                    po.compact()
                ttnn.synchronize_device(mesh)
                logger.info(f"frag: extents {dict(model.ext)}; {po.stats}")
            for g0 in range(0, U, G):
                grp = [u for u in range(g0, min(U, g0 + G))]
                if not any(P + i < ids[u].numel() for u in grp):
                    continue
                # (N > 1: every lane of the group N rows a pass, as a verify pass: a text's tail padded with token 0,
                # a finished text fed padding rows until its group finishes; only the text's rows are scored)
                rows = {}
                for u in grp:
                    real = [int(x) for x in ids[u][P + i : P + i + N]]
                    if N == 1 and not real:
                        continue
                    rows[lane[u]] = real + [0] * (N - len(real))
                t0 = time.perf_counter()
                out = dec.run_batch(rows)
                times.append((len(rows), time.perf_counter() - t0))
                for u in grp:
                    n = max(0, min(N, ids[u].numel() - P - i))
                    if lane[u] in out and n:
                        lps[u].append(torch.log_softmax(out[lane[u]][:n].float(), dim=-1))
        for u, name in enumerate(users):
            if u >= len(names):
                continue
            ref = np.load(os.path.join(args.ref_dir, f"ref_{name}.npz"))
            top_ids = ref["top_ids"].astype(np.int64)
            lp = torch.cat(lps[u])
            T = ids[u].numel() - 1
            lp = lp[:T]
            nll = (-lp[torch.arange(T), ids[u][1:]]).numpy()
            np.savez(os.path.join(args.out_dir, f"dev_{args.tag}_{name}.npz"), top1=lp.argmax(-1).numpy(),
                     lp_at_ref=torch.gather(lp, 1, torch.from_numpy(top_ids[:T])).numpy(), nll=nll)
            logger.info(f"{name}: nll {nll.mean():.4f}")
        full = [t for n, t in times[8:] if n == min(U, G)]
        if full:
            med = sorted(full)[len(full) // 2]
            if N == 1 and G >= U:
                print(f"BATCH {U} users: median pass {med * 1e3:.2f} ms -> {U / med:.1f} tok/s total, "
                      f"{1 / med:.1f} tok/s per user", flush=True)
            else:
                print(f"BATCH {U} users ({min(U, G)} x {N} rows a pass): median pass {med * 1e3:.2f} ms", flush=True)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
