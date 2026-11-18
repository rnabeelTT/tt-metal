# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Decode-row projections (verify rows / decode: x [1, 1, R <= 32, K], DRAM-interleaved weight [K, N] per die) with a
1D multicast program config: every output column tile on its own core group, the activation multicast, in0 blocks of
8 K tiles (4 for short K).

ttnn.linear without a program config picks a config that reads the weight at ~90-220 GB/s on these shapes; the 1D
config reads it at ~310-330 GB/s (tools/linear_bench.py, one chip: QKV 4096 x 1664 bfp8 80.3 -> 23.5 us, o_proj
1024 x 4096 13.4 vs 20.5 us). In the model (tools/pass_dump.py A/B of a verify pass): o_proj bit-identical, QKV within
a few bf16 ulps of the logits (mean |d| 0.006, same argmax).
"""

from __future__ import annotations

import os

import ttnn

# INKLING_ROWS_MM_PC: "qkv,o" (default: the attention projections), "1" every call site (+ "dense": the dense layers'
# HiFi4 MLP, whose output the config changes: pass logits mean |d| 0.12 for ~0.2 ms per pass), "0" none
_ENV = os.environ.get("INKLING_ROWS_MM_PC", "qkv,o")
ON = _ENV != "0"
TAGS = None if _ENV == "1" else set(_ENV.split(","))
MAX_CORES = 64
_PC = {}


def _program_config(mesh, K, N):
    key = (id(mesh), K, N)
    pc = _PC.get(key)
    if pc is None:
        grid = mesh.compute_with_storage_grid_size()
        kt, nt = K // 32, N // 32
        cap = min(MAX_CORES, grid.x * grid.y)
        ncores = max(d for d in range(1, cap + 1) if nt % d == 0)
        bw = max(d for d in (8, 4, 2, 1) if kt % d == 0 and (d <= 4 or kt >= 64))
        pc = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(grid.x, grid.y),
            in0_block_w=bw,
            out_subblock_h=1,
            out_subblock_w=1,
            per_core_M=1,
            per_core_N=nt // ncores,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=True,
        )
        _PC[key] = pc
    return pc


def rows_linear(x, w, mesh, compute_kernel_config, memory_config, dtype=None, tag=None):
    """ttnn.linear(x, w) for x of at most 32 rows (one tile row), through the 1D config; longer x: the plain call."""
    R, K, N = int(x.shape[-2]), int(w.shape[-2]), int(w.shape[-1])
    kw = dict(compute_kernel_config=compute_kernel_config, memory_config=memory_config)
    if dtype is not None:
        kw["dtype"] = dtype
    if not ON or (TAGS is not None and tag not in TAGS) or R > 32 or K % 32 or N % 32 or x.is_sharded():
        return ttnn.linear(x, w, **kw)
    return ttnn.linear(x, w, program_config=_program_config(mesh, K, N), **kw)
