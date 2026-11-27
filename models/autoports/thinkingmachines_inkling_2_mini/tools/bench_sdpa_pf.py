# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Prefill SDPA (inkling_sdpa_decode, packed-prefill arguments) device time on one die with a TP's per-die heads: one
sequence of T rows from position P0, items of 8 rows, pt_per_item, mi SDPA_MI, pair tables, bf16 caches.
env: TP (4), T (2048), P0 (0), W (0,1024), REPS (5); device time needs TT_METAL_DEVICE_PROFILER=1 (tests/_devtime.py),
else host wall after a synchronize."""

import os
import time

import torch

import ttnn
from models.autoports.thinkingmachines_inkling_2_mini.tests._devtime import devtime
from models.autoports.thinkingmachines_inkling_2_mini.tt import relbias
from models.autoports.thinkingmachines_inkling_2_mini.tt.attn.inkling_sdpa import inkling_sdpa_decode

TP = int(os.environ.get("TP", "4"))
NH, NKV, HD, DREL, BLK, R = 24 // TP, 8 // TP, 128, 16, 64, 8
MI = int(os.environ.get("I2M_SDPA_MI", "4"))


def main():
    dev = ttnn.open_device(device_id=0)
    T, P0 = int(os.environ.get("T", "2048")), int(os.environ.get("P0", "0"))
    reps = int(os.environ.get("REPS", "5"))
    torch.manual_seed(0)
    dram = ttnn.DRAM_MEMORY_CONFIG
    P = P0 + T
    nb = (P + BLK - 1) // BLK
    tl = lambda x, dt=ttnn.bfloat16: ttnn.from_torch(x, dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev,  # noqa: E731
                                                     memory_config=dram)
    K = tl(torch.randn(nb + 1, NKV, BLK, HD))
    V = tl(torch.randn(nb + 1, NKV, BLK, HD))
    n_items = T // R
    pt = torch.randperm(nb + 1)[:nb].to(torch.int32).reshape(1, -1).expand(n_items, -1).contiguous()
    ptt = ttnn.from_torch(pt, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, memory_config=dram)
    cur = ttnn.from_torch(torch.arange(P0, P, dtype=torch.int32).reshape(n_items, R), dtype=ttnn.int32,
                          layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, memory_config=dram)
    q = tl(torch.nn.functional.normalize(torch.randn(1, T, NH, HD), dim=-1) * 20.0)
    rq_h = torch.zeros(1, T, NH, 32)
    rq_h[..., :DREL] = torch.randn(T, NH, DREL) * 0.5
    rq = tl(rq_h)
    sk = tl(torch.randn(1, T, NH, 1).expand(1, T, NH, 32).contiguous())
    Tb = relbias.build_tables(torch.randn(DREL, 1024) * 0.5, HD**-0.5)
    table = tl(Tb.reshape(1, 32, 32, -1))
    table2 = tl(relbias.build_pair_tables(Tb).reshape(1, 32, 32, -1))
    for W in [int(v) for v in os.environ.get("W", "0,1024").split(",")]:
        f = lambda: inkling_sdpa_decode(q, K, V, cur, ptt, rq, table, HD**-0.5, W, 1024, sink=sk,  # noqa: E731
                                        out_memory_config=dram, pt_per_item=True, mi=MI, table2=table2, cargs=True)
        out = f()
        ref = ttnn.to_torch(out).float()
        ttnn.deallocate(out)
        dt = devtime(dev, lambda: ttnn.deallocate(f()), reps=reps)
        ws = []
        for _ in range(reps):
            ttnn.synchronize_device(dev)
            t0 = time.perf_counter()
            o = f()
            ttnn.synchronize_device(dev)
            ws.append((time.perf_counter() - t0) * 1e3)
            ttnn.deallocate(o)
        ws.sort()
        print(f"TP{TP} T={T} P0={P0} W={W}: device {dt if dt is None else round(dt / 1e3, 3)} ms, wall median "
              f"{ws[len(ws) // 2]:.3f} ms, out sum {ref.double().sum().item():.6e}", flush=True)
    ttnn.close_device(dev)


if __name__ == "__main__":
    main()
