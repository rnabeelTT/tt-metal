# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The sorted prefill MoE's row gather (tt/moe.py _sorted, GATHER_L1) on one die: T rows of H, random top-k routing over
E experts; device time of ttnn.embedding (TILE out, DRAM), ttnn.moe_gather_rows, the L1 row-major copy of x, and a
DRAM clone of a same-size tile tensor (write bandwidth reference); outputs compared on the real groups.
env: T (2048), REPS (5), GK (ttnn.moe_gather_rows on: 1). Device time needs TT_METAL_DEVICE_PROFILER=1."""

import os

import torch

import ttnn
from models.autoports.minimaxai_minimax_m2_7.tt.moe_dispatch import SortedDispatch
from models.autoports.thinkingmachines_inkling_2_mini.tests._devtime import devtime

H, E, K = 2304, 128, 8


def main():
    dev = ttnn.open_device(device_id=0)
    T, reps = int(os.environ.get("T", "2048")), int(os.environ.get("REPS", "5"))
    torch.manual_seed(0)
    DRAM, L1 = ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG
    x = ttnn.from_torch(torch.randn(1, 1, T, H), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                        memory_config=DRAM)
    ids = torch.stack([torch.randperm(E)[:K] for _ in range(T)]).reshape(1, 1, T, K)
    ids_t = ttnn.from_torch(ids.to(torch.int32), dtype=ttnn.uint16, layout=ttnn.TILE_LAYOUT, device=dev)
    w_t = ttnn.from_torch(torch.rand(1, 1, T, K), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    sd = SortedDispatch(dev, E, K, group=32)
    sd.need_mask = False
    m = sd.build(ids_t, w_t, T)
    Rd, G = m["Rd"], m["G"]
    gid = ttnn.to_torch(m["group_ids"]).flatten()[:G].to(torch.int64)
    real = int((gid < E).sum())
    print(f"T={T}: Rd {Rd}, G {G}, real groups {real}", flush=True)
    x_rm = ttnn.to_layout(ttnn.reshape(x, (T, H)), ttnn.ROW_MAJOR_LAYOUT, memory_config=L1)

    def emb():
        return ttnn.reshape(ttnn.embedding(m["src_tok"], x_rm, layout=ttnn.TILE_LAYOUT, memory_config=DRAM),
                            (1, Rd // 32, 32, H))

    def gk():
        return ttnn.moe_gather_rows(x_rm, m["src_tok"], m["group_ids"], E, memory_config=DRAM)

    ref = ttnn.to_torch(emb()).reshape(G, 32, H)
    fs = {"to_layout L1": lambda: ttnn.to_layout(ttnn.reshape(x, (T, H)), ttnn.ROW_MAJOR_LAYOUT, memory_config=L1),
          "embedding": emb}
    if os.environ.get("GK", "1") == "1":
        got = ttnn.to_torch(gk()).reshape(G, 32, H)
        ok = torch.equal(got[gid < E], ref[gid < E])
        print(f"moe_gather_rows == embedding on the real groups: {ok}", flush=True)
        fs["moe_gather_rows"] = gk
    big = ttnn.from_torch(torch.zeros(1, 1, Rd, H), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                          memory_config=DRAM)
    fs["clone (DRAM, Rd x H)"] = lambda: ttnn.clone(big, memory_config=DRAM)
    for name, f in fs.items():
        dt = devtime(dev, lambda: ttnn.deallocate(f()), reps=reps)
        mb = Rd * H * 2 / 1e6
        print(f"{name}: device {dt if dt is None else round(dt, 1)} us"
              + ("" if dt is None or "L1" in name else f" ({mb / dt * 1e3:.0f} GB/s out)"), flush=True)
    ttnn.close_device(dev)


if __name__ == "__main__":
    main()
