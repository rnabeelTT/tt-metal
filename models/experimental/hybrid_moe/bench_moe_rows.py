# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Microbenchmark: the multi-row fused MoE (qwen36 moe_decode/union.py) at MiniMax-M2.7's per-die hybrid shapes (H 3072,
expert width 384 per die, C = 192 slots, k = 8, 4 verify rows, bfp4, routed only), on one chip. The rows' logits share
a common part so their union is realistic (~20-24 slots of 32 picks). Prints the trace-replay time per call of
route + gate|up + down for each reader / chunk variant. usage (in the image): python bench_moe_rows.py"""

import os
import sys
import time

sys.path.insert(0, "/opt/tt-metal")
import torch  # noqa: E402

import ttnn  # noqa: E402
from models.demos.blackhole.qwen36.tt.moe_decode.union import (  # noqa: E402
    ExpertDownRows,
    ExpertGateUpRows,
    RouteRows,
    RowsBuffers,
    act_buffer,
)

H, I, E, K, ROWS = 3072, 384, 192, 8, 4
DT = ttnn.bfloat4_b


def main():
    dev = ttnn.open_device(device_id=0, l1_small_size=24576, trace_region_size=64 << 20)
    dram = ttnn.DRAM_MEMORY_CONFIG
    up = lambda t: ttnn.from_torch(t, dtype=DT, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=dram)  # noqa: E731
    g = torch.Generator().manual_seed(0)
    ug = up(torch.randn(1, E, H, 2 * I, generator=g) * 0.03)
    dn = up(torch.randn(1, E, I, H, generator=g) * 0.03)
    xr = torch.zeros(1, 1, 32, H)
    xr[0, 0, :ROWS] = torch.randn(ROWS, H, generator=g)
    x = ttnn.from_torch(
        xr, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    base = torch.randn(E, generator=g) * 3
    lr = torch.zeros(1, 1, 32, E + 32)
    lr[0, 0, :ROWS, :E] = base + torch.randn(ROWS, E, generator=g) * 1.5
    union = len({int(e) for r in range(ROWS) for e in torch.topk(lr[0, 0, r, :E], K).indices})
    lg = ttnn.from_torch(
        lr, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    print(f"rows {ROWS}, union {union} of {ROWS * K} picks", flush=True)

    def bench(fn, reps=20, iters=20):
        fn()
        tid = ttnn.begin_trace_capture(dev, cq_id=0)
        for _ in range(reps):
            fn()
        ttnn.end_trace_capture(dev, tid, cq_id=0)
        ttnn.execute_trace(dev, tid, cq_id=0, blocking=True)
        t0 = time.time()
        for _ in range(iters):
            ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(dev)
        ttnn.release_trace(dev, tid)
        return (time.time() - t0) / (reps * iters) * 1e6

    variants = [(None, None, 8), (None, None, 8)]  # the port's defaults (twice: the first call is cold)
    variants += [(gr, dr, c) for gr in (None, 3) for dr in (3, 4) for c in (2, 3, 4, 6)]
    variants += [(None, None, 8)]
    env_split = os.environ.get("QWEN35_MOE_SPLIT_PAIR", "0")
    for gr, dr, chunk in variants:
        try:
            buf = RowsBuffers(dev, H, I, E, K, ROWS, DT, act=act_buffer(dev, I, E, K, ROWS))
            route = RouteRows(dev, E, K, ROWS)
            gu = ExpertGateUpRows(dev, H, I, E, K, DT, ROWS, readers=gr, shared=False)
            dn_op = ExpertDownRows(dev, H, I, E, K, DT, ROWS, readers=dr, chunk=chunk, shared=False)
        except Exception as e:  # a variant the layout rules reject
            print(f"gu {gr} dn {dr} chunk {chunk}: rejected ({str(e).splitlines()[0][:100]})", flush=True)
            continue
        t_gu = bench(lambda: (route(lg, buf.table), gu(x, buf.table, buf.identity, ug, ug, buf.act)))
        t_all = bench(
            lambda: (
                route(lg, buf.table),
                gu(x, buf.table, buf.identity, ug, ug, buf.act),
                dn_op(buf.act, buf.table, dn, dn, buf.out),
            )
        )
        print(
            f"split {env_split} gu readers {gu.readers} ({len(gu.assign)} cores) dn readers {getattr(dn_op, 'readers', dr)} "
            f"({len(getattr(dn_op, 'assign', ()))} cores) chunk {dn_op.chunk}: route+gate|up {t_gu:6.1f} us, all {t_all:6.1f} us",
            flush=True,
        )
    ttnn.close_device(dev)


if __name__ == "__main__":
    main()
