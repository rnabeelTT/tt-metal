# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage-3 review #3 P1-2: the KDA in-proj split DRAM-sharded decode matmul (tt/optimized_decoder._KdaInSplitDS)
standalone, for a geometry (--cores --in0 --readers): synthetic bfp8 weight [4096, 32960], 32 decode rows, HiFi2; 20
calls, output PCC vs the interleaved ttnn.linear, host-timed us per call. Run under TT_METAL_WATCHER=10 (with
TT_METAL_LOGGER_LEVEL=Warning so the watcher prints the asserting core / RISC / kernel / line) to check a geometry.

usage: python -m models.autoports.zai_org_glm_5_3_flash.tools.s3r3_split_watch --cores 64 --in0 2 --readers 2"""

import argparse
import time

import torch

import ttnn
from models.autoports.zai_org_glm_5_3_flash.tests.harness import open_mesh, pcc
from models.autoports.zai_org_glm_5_3_flash.tt.optimized_decoder import _ckc, _KdaInSplitDS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cores", type=int, default=64)
    ap.add_argument("--in0", type=int, default=2)
    ap.add_argument("--readers", type=int, default=2)
    ap.add_argument("--n", type=int, default=20)
    a = ap.parse_args()
    mesh = open_mesh()
    try:
        g = torch.Generator().manual_seed(0)
        K, N = 4096, 32960
        w = ttnn.from_torch(
            (torch.randn(1, 1, K, N, generator=g) / K**0.5).bfloat16(),
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        x = ttnn.from_torch(
            torch.randn(1, 1, 32, K, generator=g).bfloat16(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ref = ttnn.to_torch(ttnn.linear(x, w, compute_kernel_config=_ckc(mesh, "HiFi2"))).float()
        mm = _KdaInSplitDS(w, mesh, "HiFi2", (a.cores, a.in0, a.readers))
        out = ttnn.to_torch(mm(x)).float()
        ttnn.synchronize_device(mesh)
        t = time.perf_counter()
        for _ in range(a.n):
            ttnn.deallocate(mm(x))
        ttnn.synchronize_device(mesh)
        us = (time.perf_counter() - t) / a.n * 1e6
        print(
            f"split DS cores {a.cores} in0 {a.in0} readers {a.readers}: PCC {pcc(out, ref):.6f} {us:.1f} us/call (host)",
            flush=True,
        )
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
