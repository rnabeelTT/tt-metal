# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""tt/undisp.py against the per-sub-shard ttnn.moe_gather_add_rows + ttnn.add path: bitwise, T in 64 .. 2048, S 2 / 3."""

import time

import torch

import ttnn
from models.autoports.thinkingmachines_inkling_2_mini.tt.undisp import Undispatch


def main():
    dev = ttnn.open_device(device_id=0)
    op = Undispatch(dev)
    H, k, fails = 2304, 8, 0
    for T, S in ((64, 3), (512, 3), (1000, 2), (1024, 3), (2048, 3), (2048, 2)):
        torch.manual_seed(T + S)
        Rd = (T * k // 32 + 128) * 32
        rows_h = [torch.randn(Rd, H) * 4 for _ in range(S)]
        und_h = torch.stack([torch.randperm(Rd)[:k] for _ in range(T)]).reshape(1, T * k).to(torch.int32)
        rows = [ttnn.from_torch(r.reshape(1, Rd // 32, 32, H), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT,
                                device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG) for r in rows_h]
        und = ttnn.from_torch(und_h, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev,
                              memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ref = None
        for r in rows:
            y = ttnn.moe_gather_add_rows(ttnn.reshape(r, (Rd, H)), und, T, k, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ref = y if ref is None else ttnn.add(ref, y)
        ref = ttnn.to_torch(ttnn.reshape(ref, (1, 1, T, H)))
        got = op(rows, und, T, k)
        g = ttnn.to_torch(got)
        ok = torch.equal(g, ref)
        fails += not ok
        ttnn.synchronize_device(dev)
        t0 = time.time()
        for _ in range(10):
            ttnn.deallocate(op(rows, und, T, k))
        ttnn.synchronize_device(dev)
        t1 = time.time()
        for _ in range(10):
            acc = None
            for r in rows:
                y = ttnn.moe_gather_add_rows(ttnn.reshape(r, (Rd, H)), und, T, k, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                acc = y if acc is None else ttnn.add(acc, y)
        ttnn.synchronize_device(dev)
        t2 = time.time()
        print(f"T={T} S={S}: bitwise {ok} (max diff {(g.float() - ref.float()).abs().max().item():.3e}); "
              f"merged {(t1 - t0) * 100:.3f} ms, old {(t2 - t1) * 100:.3f} ms (wall)", flush=True)
    ttnn.close_device(dev)
    print("PASS" if not fails else "FAIL")


if __name__ == "__main__":
    main()
