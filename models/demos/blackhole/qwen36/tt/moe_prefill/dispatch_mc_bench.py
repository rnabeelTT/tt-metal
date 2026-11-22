# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""dispatch_mc vs ttnn.experimental.moe_sorted_dispatch: bit-exact on every output, and both ops' device time.

    python -m models.demos.blackhole.qwen36.tt.moe_prefill.dispatch_mc_bench [--lens 1,20,128,500,2048,3000,8192]
"""
import argparse
import statistics
import time

import torch

import ttnn
from models.demos.blackhole.qwen36.tt.moe_prefill.dispatch_mc import dispatch_mc


def rows(T, k, E, group=32):
    R = T * k
    return -(-R // 32) * 32 + (-(-(group - 1) * E // group)) * group + group


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lens", default="1,20,128,500,2048,3000,8192")
    ap.add_argument("--shapes", default="256x8,512x10,128x8")  # ExK: Ornith-35B/Qwen3.5-122B, Coder-Next, Coder-30B
    ap.add_argument("--reps", type=int, default=10)
    a = ap.parse_args()
    dev = ttnn.open_device(device_id=0)
    op = getattr(ttnn, "moe_sorted_dispatch", None) or ttnn._ttnn.operations.experimental.moe_sorted_dispatch
    ok_all = True
    for shp in a.shapes.split(","):
        E, K = (int(v) for v in shp.split("x"))
        for T in [int(x) for x in a.lens.split(",")]:
            for case in ("plain", "skew", "sentinel"):
                g = torch.Generator().manual_seed(T * 7 + E + len(case))
                logits = torch.randn(T, E, generator=g)
                if case == "skew":
                    logits[:, :4] += 4.0
                sc, ids = torch.topk(torch.softmax(logits, -1), K, dim=-1)
                sc = sc / sc.sum(-1, keepdim=True)
                if case == "sentinel":  # 4 dies: experts off this die become the sentinel E
                    ids = torch.where(ids % 4 == 1, ids, torch.full_like(ids, E))
                Rd = rows(T, K, E)
                ids_tt = ttnn.from_torch(ids.to(torch.int32).reshape(1, 1, T, K), dtype=ttnn.uint16,
                                         layout=ttnn.TILE_LAYOUT, device=dev)
                sc_tt = ttnn.from_torch(sc.to(torch.bfloat16).reshape(1, 1, T, K), dtype=ttnn.bfloat16,
                                        layout=ttnn.TILE_LAYOUT, device=dev)
                try:
                    ref = op(ids_tt, sc_tt, E, K, 32, Rd, memory_config=ttnn.DRAM_MEMORY_CONFIG, pad_group_id=0)
                except RuntimeError as ex:  # the single-core op holds everything in one core's L1
                    print(f"DMC_BENCH E={E} k={K} T={T} {case}: single-core op unavailable ({str(ex).splitlines()[0][:80]})",
                          flush=True)
                    ttnn.deallocate(ids_tt)
                    ttnn.deallocate(sc_tt)
                    continue
                ok = True
                for rep in range(3):  # a few times: stale semaphores / races would show as a later mismatch
                    got = dispatch_mc(ids_tt, sc_tt, E, K, 32, Rd, pad_group_id=0)
                    for name, x, y in zip(("src_tok", "weights", "group_ids", "undispatch"), got, ref):
                        xt, yt = ttnn.to_torch(x).reshape(-1), ttnn.to_torch(y).reshape(-1)
                        if not torch.equal(xt.view(torch.int16) if xt.dtype == torch.bfloat16 else xt,
                                           yt.view(torch.int16) if yt.dtype == torch.bfloat16 else yt):
                            ok = False
                            bad = (xt != yt).nonzero()[:5].reshape(-1).tolist()
                            print(f"  MISMATCH E={E} k={K} T={T} {case} rep{rep} {name}: at {bad}", flush=True)
                    for t in got:
                        ttnn.deallocate(t)
                for t in ref:
                    ttnn.deallocate(t)
                ok_all &= ok
                times = {}
                for name, fn in (("single", lambda: op(ids_tt, sc_tt, E, K, 32, Rd, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                                                         pad_group_id=0)),
                                 ("mc", lambda: dispatch_mc(ids_tt, sc_tt, E, K, 32, Rd, pad_group_id=0))):
                    ts = []
                    for r in range(a.reps + 2):
                        ttnn.synchronize_device(dev)
                        t0 = time.perf_counter()
                        o = fn()
                        ttnn.synchronize_device(dev)
                        ts.append(time.perf_counter() - t0)
                        for t in o:
                            ttnn.deallocate(t)
                    times[name] = statistics.median(ts[2:]) * 1e6
                print(f"DMC_BENCH E={E} k={K} T={T} {case}: exact={ok} single {times['single']:.0f} us  "
                      f"mc {times['mc']:.0f} us (host wall incl. launch)", flush=True)
                ttnn.deallocate(ids_tt)
                ttnn.deallocate(sc_tt)
    print(f"DMC_BENCH all_exact={ok_all}", flush=True)
    ttnn.close_device(dev)


if __name__ == "__main__":
    main()
