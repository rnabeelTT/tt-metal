# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Decode matmuls (<= 32 rows per batch item, few batch items) with an explicit 1D mcast_in0 program config on an
interleaved weight (qwen36 tp_common.create_matmul_1d_decode_progcfg): the default config picks too few cores / small
blocks at one tile row. tools/bench_proj.py (P150, bfp8, HiFi2, traced): k / v 34 -> 13 us, o 60 -> 24, q|r|s|g 58 -> 50,
gate down (K 4608, N 256) 104 -> 20. Wider inputs (prefill) keep the default config.

I2M_DMM (default 1), I2M_DMM_CORES (48; the LM head uses all cores)."""

from __future__ import annotations

import math
import os

import ttnn

ON = os.environ.get("I2M_DMM", "1") == "1"
CORES = int(os.environ.get("I2M_DMM_CORES", "48"))
_cfg = {}


def _progcfg(mesh, m, k, n):
    key = (m, k, n)
    if key not in _cfg:
        from models.demos.blackhole.qwen36.tt import tp_common as tpc

        grid = mesh.compute_with_storage_grid_size()
        nt = math.ceil(n / 32)
        nc = min(grid.x * grid.y if nt >= 2048 else CORES, nt, grid.x * grid.y)
        _cfg[key] = tpc.create_matmul_1d_decode_progcfg(m, k, n, nc, grid_w=grid.x)
    return _cfg[key]


def dmm(x, w, ckc, dtype=None, memory_config=None, linear=False):
    """x [1, b, R, K] (TILE, b * ceil(R / 32) <= 8 row tiles) @ w [1, 1, K, N] -> [1, b, R, N]."""
    sh = [int(d) for d in x.shape]
    kw = {"compute_kernel_config": ckc}
    if dtype is not None:
        kw["dtype"] = dtype
    if memory_config is not None:
        kw["memory_config"] = memory_config
    fn = ttnn.linear if linear else ttnn.matmul
    lead = math.prod(sh[:-2])
    rows = lead * math.ceil(sh[-2] / 32) * 32
    if ON and sh[-2] <= 32 and rows <= 256 and not x.is_sharded():
        kw["program_config"] = _progcfg(x.device(), rows, sh[-1], int(w.shape[-1]))
    return fn(x, w, **kw)
