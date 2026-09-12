#!/usr/bin/env python3
"""Reproduce the decode non-determinism documented in ../NONDETERMINISM.md.

Two modes, both on one device, batch 1, with GDN + KV state zeroed before every trial:

  --mode trials   Run the SAME decode step several times and report the first routed
                  expert bank whose INPUTS are bitwise identical but whose OUTPUT is
                  not. This is the evidence that the router is not at fault: x, the
                  routing weights and the expert indices all compare exactly equal
                  while the bank's result does not.

  --mode b2b      Call each bank TWICE in a row inside one decode step with the
                  identical operand tensors, and report how many banks answered the
                  same question two different ways. Same process, same step, nothing
                  in between.

THE EFFECT IS INTERMITTENT. A clean run is not evidence of a fix: one six-trial run
came out 6/6 identical and a rerun of the identical script flipped. Repeat before
concluding anything. It is also suppressed by heavy instrumentation, so do not add
readbacks beyond what this script already does and expect the same behaviour.

Usage:
    source ~/ornith-bringup/env.sh; activate_venv
    export ORNITH_WEIGHTS=<weights>
    python tools/repro_nondeterminism.py --mode trials --trials 4
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib

import torch

import ttnn
from models.autoports.ornith_ai_ornith_1_5_35b_a3b.tt.model import OrnithModel
from models.autoports.ornith_ai_ornith_1_5_35b_a3b.tt.model_config import OrnithModelArgs
from models.autoports.ornith_ai_ornith_1_5_35b_a3b.tt.weight_mapping import load_ornith_state_dict
from models.demos.blackhole.qwen36.tt.model_config import GDN_CONV1D_L1_SMALL_SIZE


def _grab(t):
    try:
        return ttnn.to_torch(t).float().clone()
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("trials", "b2b"), default="trials")
    ap.add_argument("--trials", type=int, default=4)
    ap.add_argument("--weights", default=os.environ.get("ORNITH_WEIGHTS"))
    a = ap.parse_args()
    if not a.weights:
        raise SystemExit("set --weights or ORNITH_WEIGHTS")

    W = pathlib.Path(a.weights)
    mesh = ttnn.open_mesh_device(
        ttnn.MeshShape(1, 1),
        l1_small_size=GDN_CONV1D_L1_SMALL_SIZE,
        num_command_queues=1,
        trace_region_size=1024 * 1024 * 1024,
    )
    cfg = json.loads((W / "config.json").read_text())["text_config"]
    args = OrnithModelArgs(mesh, max_batch_size=1, max_seq_len=200)
    model = OrnithModel(
        mesh,
        args,
        load_ornith_state_dict(W, cfg),
        tensor_cache_path=pathlib.Path(args.weight_cache_path(args.routed_expert_dtype)),
    )

    BLOCK, NB = 64, 64

    def fresh():
        model.free_kv_caches()
        model.allocate_kv_caches([NB, args.n_local_kv_heads, BLOCK, args.head_dim], ttnn.bfloat16, batch_size=1)

    rec = []
    on = [False]
    wrapped = 0
    for i, layer in enumerate(model.layers):
        bank = getattr(getattr(layer, "feed_forward", None), "bank", None)
        if bank is None or getattr(bank, "_repro_wrapped", False):
            continue
        orig = bank.forward

        def make(orig=orig, i=i):
            def f(x, scores, indices, layer_id=None, *rest, **kw):
                if not on[0]:
                    return orig(x, scores, indices, layer_id=layer_id, *rest, **kw)
                if a.mode == "trials":
                    pre = (_grab(x), _grab(scores), _grab(indices))
                    out = orig(x, scores, indices, layer_id=layer_id, *rest, **kw)
                    rec[-1].append((f"L{i:02d}", pre[0], pre[1], pre[2], _grab(out)))
                    return out
                out = orig(x, scores, indices, layer_id=layer_id, *rest, **kw)
                out2 = orig(x, scores, indices, layer_id=layer_id, *rest, **kw)
                t1, t2 = _grab(out), _grab(out2)
                d = None if (t1 is None or t2 is None) else (t1 - t2).abs().max().item()
                rec[-1].append((f"L{i:02d}", d))
                try:
                    ttnn.deallocate(out2)
                except Exception:
                    pass
                return out

            return f

        bank.forward = make()
        bank._repro_wrapped = True
        wrapped += 1
    print(f"instrumented {wrapped} routed-expert banks", flush=True)

    tok = torch.full((1, 1), 1234, dtype=torch.long)
    pos = torch.full((1,), 128, dtype=torch.long)
    fresh()
    model.decode(tok, pos)
    ttnn.synchronize_device(mesh)  # warm / compile, not measured

    for _ in range(a.trials):
        fresh()
        rec.append([])
        on[0] = True
        model.decode(tok, pos)
        ttnn.synchronize_device(mesh)
        on[0] = False

    if a.mode == "b2b":
        print("\nSame bank, same operands, called twice in a row:")
        any_bad = False
        for t, r in enumerate(rec):
            bad = [(tag, d) for tag, d in r if d and d > 0]
            any_bad = any_bad or bool(bad)
            note = f"   e.g. {bad[0][0]} diff={bad[0][1]:g}" if bad else ""
            print(f"  trial {t}: {len(bad)}/{len(r)} banks answered differently{note}")
        print("REPRODUCED" if any_bad else "clean this run -- intermittent, repeat")
    else:
        print(f"\nFirst bank with identical inputs but a different output, over {a.trials} trials:")
        names = ["x_in", "scores", "indices", "bank_out"]
        n = min(len(r) for r in rec) if rec else 0
        found = False
        for j in range(n):
            tag = rec[0][j][0]
            worst = {}
            for c in range(1, 5):
                base = rec[0][j][c]
                if base is None:
                    continue
                d = 0.0
                for t in range(1, a.trials):
                    other = rec[t][j][c]
                    if other is not None:
                        d = max(d, (base - other).abs().max().item())
                worst[names[c - 1]] = d
            if any(v > 0 for v in worst.values()):
                print("  " + tag + ": " + "  ".join(f"{k}={v:g}" for k, v in worst.items()))
                print("REPRODUCED" if worst.get("bank_out", 0) > 0 else "diverged upstream of the bank")
                found = True
                break
        if not found:
            print("  clean this run -- intermittent, repeat")

    ttnn.close_mesh_device(mesh)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
