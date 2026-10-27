"""Device time of an MTP verify's GDN mixer on the fused decode ops, one die (Ornith TP4 shapes):

    B4      one GdnDecode call over 4 users (what a 4-row fused decode step costs today; NOT sequential rows)
    B1      one call over 1 user (the plain single-user step)
    chain4  4 sequential rows of ONE user: slice row r of the in-proj output -> GdnDecode(B=1) on the user's state ->
            state + conv-carry snapshot after rows 0..2 (commit = pick one) -> the 4 output rows concatenated
    commit  restore one snapshot (state + 3 conv rows)

Each variant is traced and replayed N times (wall / N with one sync at the end = device time).

    python bench_gdn_rows_chain.py [N]
"""

import sys
import time

import torch

import ttnn
from models.demos.blackhole.qwen36.tt.gdn_decode.ops import TILE, GdnDecode

Nk, Nv, D = 4, 8, 128  # Ornith per die on 4 dies
R = 4  # k = 3 drafts + the current token


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    dev = ttnn.open_device(device_id=0, l1_small_size=0, trace_region_size=64 << 20)
    try:
        qkv = 2 * Nk * D + Nv * D
        P = qkv + Nv * D + 2 * TILE
        to = lambda x, dtype=ttnn.bfloat16, mc=ttnn.DRAM_MEMORY_CONFIG: ttnn.from_torch(
            x, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=mc)
        ops = {B: GdnDecode(dev, Nk, Nv, D, D, P, D**-0.5, max_batch=B) for B in (1, R)}
        taps = [to(torch.randn(1, 1, qkv) * 0.5) for _ in range(4)]
        consts = ops[1].prepare(taps, to(-torch.rand(1, 1, Nv)), to(torch.randn(1, 1, Nv)), to(torch.ones(1, 1, D)))
        proj = {B: to(torch.randn(1, B, P), mc=ttnn.L1_MEMORY_CONFIG) for B in (1, R)}
        conv = {B: [to(torch.randn(1, B, qkv)) for _ in range(4)] for B in (1, R)}
        state = {B: to(torch.randn(B, Nv, D, D) * 0.05) for B in (1, R)}
        snap_rec = [ttnn.clone(state[1]) for _ in range(R - 1)]
        snap_conv = [[ttnn.clone(c) for c in conv[1][1:]] for _ in range(R - 1)]

        def b4():
            ttnn.deallocate(ops[R](proj[R], conv[R], *consts, state[R], R))

        def b1():
            ttnn.deallocate(ops[1](proj[1], conv[1], *consts, state[1], 1))

        def chain4():
            ys = []
            for r in range(R):
                p = ttnn.slice(proj[R], (0, r, 0), (1, r + 1, P), memory_config=ttnn.L1_MEMORY_CONFIG)
                ys.append(ops[1](p, conv[1], *consts, state[1], 1))
                ttnn.deallocate(p)
                if r < R - 1:
                    ttnn.copy(state[1], snap_rec[r])
                    for c, s in zip(conv[1][1:], snap_conv[r]):
                        ttnn.copy(c, s)
            y = ttnn.concat(ys, dim=1, memory_config=ttnn.L1_MEMORY_CONFIG)
            for t in ys:
                ttnn.deallocate(t)
            ttnn.deallocate(y)

        def commit():
            ttnn.copy(snap_rec[1], state[1])
            for s, c in zip(snap_conv[1], conv[1][1:]):
                ttnn.copy(s, c)

        for name, fn in (("B4", b4), ("B1", b1), ("chain4", chain4), ("commit", commit)):
            fn()  # compile
            ttnn.synchronize_device(dev)
            tid = ttnn.begin_trace_capture(dev, cq_id=0)
            fn()
            ttnn.end_trace_capture(dev, tid, cq_id=0)
            for _ in range(5):
                ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(dev)
            t0 = time.perf_counter()
            for _ in range(n):
                ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(dev)
            us = (time.perf_counter() - t0) / n * 1e6
            ttnn.release_trace(dev, tid)
            print(f"GDNCHAIN {name}: {us:.1f} us per call ({n} replays)", flush=True)
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
