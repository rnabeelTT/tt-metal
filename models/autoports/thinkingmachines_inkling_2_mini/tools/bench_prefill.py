# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Prefill chunk timing: host enqueue time (no sync) vs synced wall time per chunk size, so host-bound and device-bound
chunks can be told apart. usage: MESH=4 SIZES=128,256,1024 tools/run_dev.sh python .../tools/bench_prefill.py"""

import os
import time

import torch

import ttnn
from models.autoports.thinkingmachines_inkling_2_mini.tt.model import Inkling2Model


def main():
    N = int(os.environ.get("MESH", "1"))
    if N > 1:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, N), trace_region_size=int(os.environ.get("TRACE_MB", "0")) << 20)
    chunk = int(os.environ.get("CHUNK", "1024"))
    dt = {"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b}
    m = Inkling2Model(mesh, max_seq_len=8192, expert_dtype=dt[os.environ.get("EXPERT_DT", "bfp8")],
                      kv_dtype=dt[os.environ.get("KV_DT", "bf16")], prefill_chunk=chunk, slots=int(os.environ.get("B", "1")))
    sizes = [int(s) for s in os.environ.get("SIZES", "128,256,512,1024").split(",")]
    for T in sizes:
        ids = torch.arange(1000, 1000 + T)
        for _ in range(2):  # compile
            m.reset()
            m.forward(ids, 0, valid=T, slot=0, last_only=True)
        ttnn.synchronize_device(mesh)
        enq, wall = [], []
        for _ in range(3):
            m.reset()
            ttnn.synchronize_device(mesh)
            t0 = time.perf_counter()
            m.forward(ids, 0, valid=T, slot=0, last_only=True)
            t1 = time.perf_counter()
            ttnn.synchronize_device(mesh)
            t2 = time.perf_counter()
            enq.append(t1 - t0)
            wall.append(t2 - t0)
        print(f"T={T:5d}: enqueue {min(enq) * 1e3:7.1f} ms  wall {min(wall) * 1e3:7.1f} ms  "
              f"({T / min(wall):.0f} tok/s)", flush=True)
    print("== bench_prefill done", flush=True)
    from models.autoports.thinkingmachines_inkling_2_mini.tt.clean_exit import close

    close(mesh)


if __name__ == "__main__":
    main()
