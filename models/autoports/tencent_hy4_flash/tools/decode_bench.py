# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Traced decode step time on a layer subset, with the logits saved for bit-exact A/B between code versions.

    python -m models.autoports.tencent_hy4_flash.tools.decode_bench --layers 0,1,2,3 [--rows 1] [--steps 32]
        [--save a.pt] [--ref a.pt]

Prefills `prompt-len` fixed random tokens, then feeds the next tokens teacher forced (fixed ids, so two code versions
see the same inputs) through Hy4Decoder (traced). Reports the median wall time per pass with a device sync after each
replay (HY4_TIME_STEPS=1: the device time of the pass + the launch), and the host-side split.
"""

from __future__ import annotations

import argparse
import os
import time

import torch
import ttnn
from loguru import logger


def pcc(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm() + 1e-30))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="0,1,2,3")
    ap.add_argument("--prompt-len", type=int, default=100)
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--rows", type=int, default=1)
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--no-engram", action="store_true")
    ap.add_argument("--eager", action="store_true")
    ap.add_argument("--save", default="")
    ap.add_argument("--ref", default="")
    ap.add_argument("--prof-host", action="store_true", help="cProfile the host side of the warm passes")
    args = ap.parse_args()
    os.environ.setdefault("HY4_TIME_STEPS", "1")
    from models.autoports.tencent_hy4_flash.tools.generate import build, open_mesh
    from models.autoports.tencent_hy4_flash.tt.decode import Hy4Decoder

    mesh = open_mesh()
    try:
        t0 = time.time()
        model = build(mesh, args)
        logger.info(f"build {time.time() - t0:.0f} s")
        if model.engram is not None and hasattr(model.engram.store, "wait_ram"):
            t0 = time.time()
            model.engram.store.wait_ram()
            logger.info(f"engram RAM wait {time.time() - t0:.0f} s")
        def l1(tag):
            mv = ttnn.get_memory_view(mesh, ttnn.BufferType.L1)
            logger.info(f"L1 {tag}: {mv.total_bytes_allocated_per_bank / 1024:.0f} KB allocated per bank, largest free "
                        f"{mv.largest_contiguous_bytes_free_per_bank / 1024:.0f} KB")

        l1("after build")
        g = torch.Generator().manual_seed(0)
        P, S, R = args.prompt_len, args.steps, args.rows
        ids = torch.randint(1000, 100000, (P + S * R,), generator=g)
        model.reset()
        model.prefill(ids[:P])
        l1("after prefill")
        dec = Hy4Decoder(model, trace=not args.eager)
        l1("after decoder init")
        dec.start(P)
        dec.set_history(ids[:P].tolist())
        outs, ts, parts = [], [], []
        import cProfile
        import pstats

        pr = cProfile.Profile()
        for i in range(S):
            if args.prof_host and i == 4:
                pr.enable()
            if os.environ.get("HY4_SIGNPOST") == "1" and i == S - 1:
                from tracy import signpost

                ttnn.synchronize_device(mesh)
                signpost("last_step")
            t1 = time.time()
            try:
                o = dec.run([int(t) for t in ids[P + i * R : P + (i + 1) * R]], logits_rows="all")
            except RuntimeError:
                l1("at the failure")
                try:
                    ttnn.device.dump_device_memory_state(ttnn.get_device_tensors(model.lm)[0].device(), "hy4_fail_")
                except Exception as e:  # noqa: BLE001
                    logger.info(f"dump failed: {e}")
                raise
            ts.append(time.time() - t1)
            outs.append(o.clone())
            if getattr(dec, "timing", None) is not None:
                parts.append(dec.timing)
        pr.disable()
        if args.prof_host:
            pstats.Stats(pr).sort_stats("tottime").print_stats(25)
        warm = sorted(ts[3:]) if len(ts) > 4 else sorted(ts)
        med = warm[len(warm) // 2]
        msg = f"layers {args.layers} rows {R}: median pass {med * 1e3:.2f} ms (min {warm[0] * 1e3:.2f})"
        if parts:
            p = [sorted(x)[len(x) // 2] * 1e3 for x in zip(*parts[3:] or parts)]
            msg += f"; host inputs {p[0]:.2f} copies {p[1]:.2f} device+launch {p[2]:.2f} read {p[3]:.2f}"
        print("BENCH " + msg, flush=True)
        lg = torch.stack(outs)  # [S, R, V]
        if args.save:
            torch.save({"ids": ids, "logits": lg}, args.save)
        if args.ref:
            ref = torch.load(args.ref)["logits"]
            n = min(len(ref), len(lg))
            same = bool(torch.equal(ref[:n], lg[:n]))
            p = min(pcc(ref[i], lg[i]) for i in range(n))
            t1 = (ref[:n].argmax(-1) == lg[:n].argmax(-1)).float().mean().item()
            print(f"BENCH vs ref: bit-identical {same}; min pcc {p:.6f}; top-1 agreement {t1:.3f}", flush=True)
        dec.release()
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
