# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3-Flash manifold-constrained hyper-connections (hc_mult = 4) around every attention and FFN sub-block.

HF (`Glm5NextTextHyperConnection`), per token, X = the 4 residual streams packed as [1,1,T,4*4096]:
    f    = X / rms(X)  (fp32, no weight, eps = rms_norm_eps)
    mix  = f @ fn^T                                              [T, 24]
    pre  = sigmoid(s0*mix[0:4]   + base[0:4]) + hc_eps
    post = 2*sigmoid(s1*mix[4:8] + base[4:8])
    comb = Sinkhorn(softmax(s2*mix[8:24] + base[8:24]) + hc_eps)  (columns first; 20 col / 19 row passes)
    x    = sum_n pre_n X_n                                      -> sub-block input
    X'_n = post_n * y + sum_a comb[a, n] X_a                    -> residual update
The parametrization and the post mix are the fused kernels ttnn.experimental.deepseek_prefill.mhc_split_sinkhorn and
mhc_post (DeepSeek-V4 mHC; same math, see models/demos/deepseek_v3_d_p/reference/mhc). The streams stay fp32 on
device. HF loads `base` / `scale` (fp32 in the checkpoint) as bf16; this module rounds them the same way.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.deepseek_v3_d_p.tt.mhc.tt_mhc import (
    W,
    _cols,
    _compute_kernel_config,
    _mix,
    _streams,
    build_bcast_consts,
    build_consts,
)

DRAM = ttnn.DRAM_MEMORY_CONFIG


@dataclass(frozen=True)
class HcConfig:
    n: int
    sinkhorn_iters: int
    eps: float
    norm_eps: float


def hc_config(tc: dict) -> HcConfig:
    return HcConfig(
        n=tc["hc_mult"], sinkhorn_iters=tc["hc_sinkhorn_iters"], eps=tc["hc_eps"], norm_eps=tc["rms_norm_eps"]
    )


def expand_streams(h: ttnn.Tensor, n: int) -> ttnn.Tensor:
    """Embedding [1,1,T,C] -> n identical streams [1,1,T,n*C] (fp32)."""
    if h.dtype != ttnn.float32:
        h = ttnn.typecast(h, ttnn.float32)
    return ttnn.repeat(h, [1, 1, 1, n])


def collapse_streams(x: ttnn.Tensor, n: int) -> ttnn.Tensor:
    """HF hc_head: the unweighted mean of the n streams [1,1,T,n*C] -> [1,1,T,C]."""
    streams = _streams(x, n)
    y = streams[0]
    for s in streams[1:]:
        y = ttnn.add(y, s, memory_config=DRAM)
    return ttnn.multiply(y, 1.0 / n, memory_config=DRAM)


class HyperConnection(LightweightModule):
    """One mHC wrapper (attn_hc or ffn_hc) of one decoder layer."""

    def __init__(self, mesh_device, cfg: HcConfig, fn: torch.Tensor, base: torch.Tensor, scale: torch.Tensor):
        self.cfg = cfg
        self.n = cfg.n
        self.ckc = _compute_kernel_config()
        rep = ttnn.ReplicateTensorToMesh(mesh_device)
        # HF semantics: fn bf16 (applied in fp32), base / scale rounded to bf16 at load
        fn = fn.to(torch.bfloat16).float()
        base = base.to(torch.bfloat16).float()
        scale = scale.to(torch.bfloat16).float()
        mix = fn.shape[0]
        fn_T = torch.zeros(fn.shape[1], W)  # [n*C, 32]: columns >= mix are zero
        fn_T[:, :mix] = fn.t()
        up = lambda t, shape: ttnn.from_torch(
            t.reshape(shape).contiguous(),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=rep,
        )
        self.fn_T = up(fn_T, (1, 1, fn.shape[1], W))
        self.consts = up(build_consts(cfg, scale, base), (8, W, W))
        self.bcast = up(build_bcast_consts(self.n), (self.n * self.n, W, W))

    def pre(self, x: ttnn.Tensor):
        """x [1,1,T,n*C] fp32 -> (h [1,1,T,C] fp32, post [1,1,T,n], comb [1,1,T,n*n])."""
        T = x.shape[-2]
        mixes = ttnn.matmul(x, self.fn_T, compute_kernel_config=self.ckc, memory_config=DRAM)
        ms = ttnn.mean(ttnn.multiply(x, x, memory_config=DRAM), dim=-1, keepdim=True)
        mixes = ttnn.multiply(mixes, ttnn.rsqrt(ttnn.add(ms, self.cfg.norm_eps)), memory_config=DRAM)
        ttnn.deallocate(ms)
        pre, post, comb = ttnn.experimental.deepseek_prefill.mhc_split_sinkhorn(
            mixes, self.consts, self.n, self.cfg.sinkhorn_iters, self.cfg.eps
        )
        ttnn.deallocate(mixes)
        r4 = lambda t: ttnn.reshape(t, (1, 1, T, t.shape[-1]))
        h = _mix(_streams(x, self.n), _cols(r4(pre), self.n))
        return h, r4(post), r4(comb)

    def post(self, y: ttnn.Tensor, residual: ttnn.Tensor, post: ttnn.Tensor, comb: ttnn.Tensor) -> ttnn.Tensor:
        """y [1,1,T,C] (sub-block output) -> new streams [1,1,T,n*C] fp32."""
        if y.dtype != ttnn.float32:
            y = ttnn.typecast(y, ttnn.float32)
        out = ttnn.experimental.deepseek_prefill.mhc_post(y, residual, post, comb, self.bcast, self.n)
        return ttnn.reshape(out, tuple(residual.shape))  # the kernel emits rank 2 [T, n*C]
