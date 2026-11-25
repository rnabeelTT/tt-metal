# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Mix logits from packed queries (I2M_LG_PACK): the logits matmul against the queries packed 32 per tile column
(query i in column i) instead of each query repeated over its 32 columns (32x less fp32 weight read), then
tt/kernels/lgx.cpp copies each query's column over its 32 columns: the same [1, n, T, nq * 32] layout and values
(an output element of the matmul depends only on its row and its weight column)."""

from __future__ import annotations

import hashlib

import torch

import ttnn

from .bconv import KDIR, _cb, _ks, _range_set, _rt

CORES = 32
ON = __import__("os").environ.get("I2M_LG_PACK", "1") == "1"
# I2M_LG_FOLD (default 1): no expand program; the decode mix readers take the packed logits and copy each column
# over its tile themselves (tt/kernels/lgpk.hpp)
FOLD = ON and __import__("os").environ.get("I2M_LG_FOLD", "1") == "1"
DEF = [("LG_PK", "1")] if FOLD else []
# I2M_LG_BW (default 24): the fp32 logit matmuls' K block (tiles; 8 before): fewer in0 hand-offs, same values
BW = int(__import__("os").environ.get("I2M_LG_BW", "24"))


def pack_cols(cols, mesh, rep):
    """cols [H, nq] -> [1, 1, H, ceil(nq / 32) * 32] fp32 (query i in column i, zero padding)."""
    H, nq = cols.shape
    p = torch.zeros(H, -(-nq // 32) * 32)
    p[:, :nq] = cols
    return ttnn.from_torch(p.reshape(1, 1, H, -1).contiguous(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh,
                           memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=rep)


def host_cols(t, nq):
    """A [1, 1, H, >= nq] device tensor (replicated) -> its first nq columns [H, nq] on the host."""
    d = ttnn.get_device_tensors(t)[0] if t.device().get_num_devices() > 1 else t
    return ttnn.to_torch(d).float().reshape(-1, int(t.shape[-1]))[:, :nq]


class LogitExpand:
    def __init__(self, mesh):
        self.mesh = mesh
        g = mesh.compute_with_storage_grid_size()
        self.gx, self.G = g.x, g.x * g.y

    def __call__(self, x, nq, out_mc=ttnn.L1_MEMORY_CONFIG):
        """x [1, n, T, Wp * 32] fp32 (packed logits) -> [1, n, T, nq * 32] fp32 (query q over tile q's columns).
        Or x: a list of n (packed [1, 1, T, W] tensor, first logit): row block s from logits off_s .. off_s + nq - 1
        of its own tensor (all of one memory config)."""
        if isinstance(x, (list, tuple)):
            srcs = list(x)
            n, x = len(srcs), srcs[0][0]
            T, W = int(x.shape[2]), int(x.shape[3])
            src = [v for t, off in srcs for v in (t.buffer_address(), 0, int(off))]
        else:
            n, T, W = int(x.shape[1]), int(x.shape[2]), int(x.shape[3])
            srcs = None
            src = [v for s in range(n) for v in (x.buffer_address(), s * (W // 32), 0)]
        assert x.dtype == ttnn.float32 and x.layout == ttnn.TILE_LAYOUT and T <= 32 and not x.is_sharded()
        out = ttnn.allocate_tensor_on_device(ttnn.Shape([1, n, T, nq * 32]), ttnn.float32, ttnn.TILE_LAYOUT, self.mesh,
                                             out_mc)
        items = n * nq
        nc = min(CORES, items)
        cores = [ttnn.CoreCoord(i % self.gx, i // self.gx) for i in range(nc)]
        crs = _range_set(cores)
        rt = []
        for i in range(nc):
            a, b = items * i // nc, items * (i + 1) // nc
            rt.append([out.buffer_address(), a, b - a] + src)
        ct = [0, n, nq] + ttnn.TensorAccessorArgs(x).get_compile_time_args() \
            + ttnn.TensorAccessorArgs(out).get_compile_time_args()
        k = ttnn.KernelDescriptor(
            kernel_source=_ks(KDIR, "lgx.cpp"), core_ranges=crs, compile_time_args=ct, runtime_args=_rt(cores, rt),
            defines=[("LGX_T", str(T))],
            config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_0))
        prog = ttnn.ProgramDescriptor(kernels=[k], semaphores=[], cbs=[_cb(crs, 0, ttnn.float32, 4096, 3)])
        key = ("i2m_lgx2", n, T, W, nq, nc, repr(x.memory_config()), repr(out_mc))
        prog.custom_program_hash = int.from_bytes(hashlib.sha1(repr(key).encode()).digest()[:8], "little") & ((1 << 63) - 1)
        ttnn.generic_op(([t for t, _ in srcs] if srcs else [x]) + [out], prog)
        return out
