# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""ttnn.experimental.moe_sorted_dispatch: bit-exact check against the counting_mm dispatch (the op-free reference
path of tt/moe_dispatch.py) and the op's own time (median of back-to-back calls, synchronized).

    python -m models.autoports.ornith_ai_ornith_1_5_35b_a3b.tools.dispatch_kernel_bench [--lens 500,2048]
"""
import argparse
import statistics
import time

import torch

import ttnn
from models.autoports.ornith_ai_ornith_1_5_35b_a3b.tt import moe_dispatch as md


def to_torch(t):
    return ttnn.to_torch(t).reshape(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lens", default="1,32,500,1024,2048")
    ap.add_argument("--E", type=int, default=256)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--reps", type=int, default=20)
    a = ap.parse_args()
    dev = ttnn.open_device(device_id=0)
    E, K = a.E, a.k
    ok_all = True
    for T in [int(x) for x in a.lens.split(",")]:
        for skew in (False, True):
            g = torch.Generator().manual_seed(T + skew)
            logits = torch.randn(T, E, generator=g)
            if skew:  # hot experts: a few experts take most rows (long runs of one expert)
                logits[:, :4] += 4.0
            sc, ids = torch.topk(torch.softmax(logits, -1), K, dim=-1)
            sc = sc / sc.sum(-1, keepdim=True)
            def inputs():
                return (ttnn.from_torch(ids.to(torch.int32).reshape(1, 1, T, K), dtype=ttnn.uint16,
                                        layout=ttnn.TILE_LAYOUT, device=dev),
                        ttnn.from_torch(sc.to(torch.bfloat16).reshape(1, 1, T, K), dtype=ttnn.bfloat16,
                                        layout=ttnn.TILE_LAYOUT, device=dev))
            outs = {}
            for mode in ("counting_mm", "kernel"):
                md.DISPATCH_MODE = mode
                md.UNDISPATCH_KERNEL = False
                disp = md.SortedDispatch(dev, E, K)
                m = disp.build(*inputs(), T)
                outs[mode] = {kk: to_torch(m[kk]).float() for kk in ("src_tok", "weights", "group_ids", "undispatch")}
            ok = True
            for kk in outs["kernel"]:
                x, y = outs["kernel"][kk], outs["counting_mm"][kk]
                if kk == "group_ids":  # counting modes mark padding groups with 0, the kernel with pad_group_id (0)
                    pass
                if x.shape != y.shape or not torch.equal(x, y):
                    ok = False
                    bad = (x != y).nonzero()[:5].reshape(-1).tolist() if x.shape == y.shape else "shape"
                    print(f"  MISMATCH T={T} skew={skew} {kk}: {bad}", flush=True)
            ok_all &= ok
            # op time
            R = T * K
            Rd = md.dispatch_rows(T, K, E, 32)
            op = getattr(ttnn, "moe_sorted_dispatch", None) or ttnn._ttnn.operations.experimental.moe_sorted_dispatch
            ts = []
            ids_tt, sc_tt = inputs()
            for r in range(a.reps + 2):
                ttnn.synchronize_device(dev)
                t0 = time.perf_counter()
                o = op(ids_tt, sc_tt, E, K, 32, Rd, memory_config=ttnn.DRAM_MEMORY_CONFIG, pad_group_id=0)
                ttnn.synchronize_device(dev)
                ts.append(time.perf_counter() - t0)
                for t in o:
                    ttnn.deallocate(t)
            print(f"DISPATCH_BENCH T={T} skew={int(skew)} exact={ok} op median {statistics.median(ts[2:]) * 1e6:.0f} us",
                  flush=True)
    print(f"DISPATCH_BENCH all_exact={ok_all}", flush=True)
    ttnn.close_device(dev)


if __name__ == "__main__":
    main()
