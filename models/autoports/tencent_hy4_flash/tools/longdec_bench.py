# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Steady-state decode speed vs context, one user (the served NSEQS=1 engine, tt/engine.py), in one process.

Per context length L: prefill L random tokens, a few spec steps (the re-captures a long prefill triggers), then for each
setting (--ab: ';'-separated env assignments, e.g. "HY4_IDX_TOPK_SPLIT=1;HY4_IDX_TOPK_SPLIT=auto"; traces released
between settings) at the same positions and tokens:
  - plain 1-row passes (Hy4Decoder.run, traced): engine wall per pass, host split, logits (A/B: max |dlogit|, top-1),
  - replay-alone device time of the 1-row and 5-row (k = 4 verify) traces and of a draft trace,
and at the first L the 1-row replay with the score width forced to every larger bucket (the bucket-dependent cost).
The spec step time on a random prompt is per step (acceptance ~0), not tokens / s of real text.
    python -m models.autoports.tencent_hy4_flash.tools.longdec_bench --ctx 131072 --lens 12000,32768,65536,131000
Lines start with "LDB ".
"""
from __future__ import annotations

import argparse
import os
import time

import torch
import ttnn
from loguru import logger


def med(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2] if xs else float("nan")


def replay_alone(mesh, tid, n=10):
    ts = []
    for _ in range(n):
        ttnn.synchronize_device(mesh)
        t0 = time.perf_counter()
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh)
        ts.append(time.perf_counter() - t0)
    return med(ts) * 1e3


ALL_KEYS = set()


def apply_env(s):
    for k in ALL_KEYS:  # (a setting is exactly its assignments: every key another setting sets is unset first)
        os.environ.pop(k, None)
    for kv in [x for x in s.split(",") if x]:
        k, v = kv.split("=", 1)
        os.environ[k] = v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, default=131072)
    ap.add_argument("--lens", default="12000,32768,65536,131000")
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--layers", default="")
    ap.add_argument("--engram", action="store_true")
    ap.add_argument("--no-warm", action="store_true")
    ap.add_argument("--ab", default="", help="';'-separated settings (each a ','-list of VAR=val)")
    ap.add_argument("--text", action="store_true", help="natural-text prompts (tech_reports + a summary request): "
                    "spec acceptance per context on real text, greedy, 96 steps")
    ap.add_argument("--force", default="1", help="1: forced-bucket sweep at the first length (first setting)")
    args = ap.parse_args()
    from models.autoports.tencent_hy4_flash.tools.generate import open_mesh
    from models.autoports.tencent_hy4_flash.tt.config import Hy4Config, checkpoint_dir
    from models.autoports.tencent_hy4_flash.tt.engine import Hy4Engine
    from models.autoports.tencent_hy4_flash.tt.model import Hy4Model

    settings = args.ab.split(";") if args.ab else [""]
    ALL_KEYS.update(kv.split("=", 1)[0] for st in settings for kv in st.split(",") if kv)
    apply_env(settings[0])
    k = int(os.environ.get("HY4_SPEC_K", "4") or 0)
    mesh = open_mesh()
    try:
        cfg = Hy4Config.load()
        engram = None
        if args.engram:
            from models.experimental.hybrid_moe.engram_store import EngramRows

            engram = EngramRows(os.environ.get("HY4_ENGRAM_DIR", os.path.expanduser("~/.cache/hybrid-moe/hy4_engram")),
                                checkpoint_dir())
        cache_dir = os.environ.get("HY4_TENSOR_CACHE", os.path.expanduser("~/.cache/hybrid-moe/hy4_tt"))
        layers = [int(x) for x in args.layers.split(",")] if args.layers else None
        model = Hy4Model(mesh, cfg, max_ctx=args.ctx + 64, chunk=2048, engram=engram, cache_dir=cache_dir, slots=1,
                         **({"layers": layers} if layers else {}))
        eng = Hy4Engine(model, k)
        dec = eng.dec
        print(f"LDB buckets (groups) {dec.buckets}; spec k {k}; layers {len(model.layers)}", flush=True)
        t0 = time.time()
        if not args.no_warm:
            eng.warm()
        print(f"LDB warm-up {time.time() - t0:.0f} s", flush=True)
        g = torch.Generator().manual_seed(1)
        corpus = None
        if args.text:
            import glob

            from tokenizers import Tokenizer

            tok = Tokenizer.from_file(os.path.expanduser("~/m27-work/hy4/tokenizer.json"))
            fs = sorted(glob.glob(os.path.expanduser("~/hy4-flash/tech_reports/**/*.md"), recursive=True))
            corpus = tok.encode("\n\n".join(open(f, errors="ignore").read() for f in fs)).ids
            ask = tok.encode("\n\nNow write a long, detailed technical summary of the documents above, in "
                             "well-structured prose of at least 600 words.\n").ids
        forced = False
        for li, L in enumerate(int(x) for x in args.lens.split(",") if x):
            apply_env(settings[0])
            ids = torch.randint(1000, 100000, (L,), generator=g).tolist()
            if corpus is not None:
                ids = corpus[: L - len(ask)] + ask
            t0 = time.time()
            eng.prefill(ids)
            print(f"LDB L={L} prefill {time.time() - t0:.2f} s", flush=True)
            if k > 0:
                for _ in range(10):  # (re-captures: a long prefill compiles programs -> the traces are released)
                    eng.step()
                ts, acc0, em0 = [], eng.stats.get("accepted", 0), eng.stats["emitted"]
                for _ in range(96 if corpus is not None else 16):
                    t1 = time.perf_counter()
                    eng.step()
                    ts.append(time.perf_counter() - t1)
                em = (eng.stats["emitted"] - em0) / len(ts)
                print(f"LDB L={L} spec step median {med(ts) * 1e3:.2f} ms (emitted {em:.2f} per step on this random "
                      f"prompt; {settings[0] or 'default'})", flush=True)
            P = eng._pos
            ref = vref = None
            for si_, s in enumerate(settings):
              try:
                for kv in [x for x in s.split(",") if x]:
                    os.environ.pop(kv.split("=", 1)[0], None)
                apply_env(s)
                dec.release()
                pos = P
                for _ in range(4):  # eager + capture + replays
                    dec.run([1000 + pos % 5000], p0=pos)
                    pos += 1
                ts, parts, lgs = [], [], []
                for _ in range(args.steps):
                    t1 = time.perf_counter()
                    lgs.append(dec.run([1000 + pos % 5000], p0=pos).float().clone())
                    ts.append(time.perf_counter() - t1)
                    if getattr(dec, "timing", None) is not None:
                        parts.append(dec.timing)
                    pos += 1
                p = [med(x) * 1e3 for x in zip(*parts)] if parts else [0, 0, 0, 0]
                tag = f"L={L} [{s or 'default'}]"
                print(f"LDB {tag} plain pass median {med(ts) * 1e3:.2f} ms ({1 / med(ts):.2f} tok/s); host inputs "
                      f"{p[0]:.2f} copies {p[1]:.2f} launch {p[2]:.2f} read(wait) {p[3]:.2f}", flush=True)
                lg = torch.stack(lgs)
                if ref is None:
                    ref = lg
                else:
                    d = (lg - ref).abs().max().item()
                    t1a = (lg.argmax(-1) == ref.argmax(-1)).float().mean().item()
                    print(f"LDB {tag} vs [{settings[0] or 'default'}]: bit-identical {bool(torch.equal(lg, ref))}; "
                          f"max |dlogit| {d:.4f}; top-1 agreement {t1a:.3f}", flush=True)
                key1 = next((kk for kk in dec.traces if kk[0] == 1), None)
                if key1 is not None:
                    print(f"LDB {tag} replay alone {key1}: {replay_alone(mesh, dec.traces[key1]):.2f} ms", flush=True)
                if k > 0:  # a k + 1-row verify pass (positions pos..pos+k), captured then replayed alone
                    vl = []
                    for _ in range(3):
                        vl.append(dec.run([1000 + (pos + j) % 5000 for j in range(k + 1)], logits_rows="all",
                                          p0=pos).float().clone())
                    vl = torch.stack(vl)
                    if vref is None:
                        vref = vl
                    else:
                        print(f"LDB {tag} verify rows vs [{settings[0] or 'default'}]: bit-identical "
                              f"{bool(torch.equal(vl, vref))}; max |dlogit| {(vl - vref).abs().max().item():.4f}; top-1 "
                              f"agreement {(vl.argmax(-1) == vref.argmax(-1)).float().mean().item():.3f}", flush=True)
                    keyv = next((kk for kk in dec.traces if kk[0] == k + 1), None)
                    if keyv is not None:
                        print(f"LDB {tag} replay alone {keyv}: {replay_alone(mesh, dec.traces[keyv]):.2f} ms",
                              flush=True)
                    try:
                        for _ in range(2):
                            dec.draft(1000 + pos % 5000, pos, 0)
                        dk = next(iter(dec.dtraces), None)
                        if dk is not None:
                            print(f"LDB {tag} draft replay alone {dk}: {replay_alone(mesh, dec.dtraces[dk]):.2f} ms",
                                  flush=True)
                    except Exception as e:  # noqa: BLE001
                        print(f"LDB {tag} draft timing failed: {e}", flush=True)
                if L >= 2100 and not forced and args.force == "1":
                    for b in dec.buckets:
                        dec.min_bucket = b
                        for _ in range(3):
                            dec.run([1000 + pos % 5000], p0=pos)
                            pos += 1
                        kb = (1, True, dec.si.bucket)
                        if kb in dec.traces:
                            print(f"LDB {tag} forced bucket {b}: replay alone {replay_alone(mesh, dec.traces[kb]):.2f} ms",
                                  flush=True)
                    dec.min_bucket = 0
              except Exception as e:  # noqa: BLE001
                print(f"LDB L={L} [{s}] FAILED: {e!r}", flush=True)
                try:
                    ttnn.synchronize_device(mesh)
                except Exception:  # noqa: BLE001
                    pass
            forced = forced or (L >= 2100)
            eng._n = None
            apply_env(settings[0])
            dec.release()
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
