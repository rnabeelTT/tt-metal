# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Mesh helpers shared by the Hy4 modules: uploads (replicated / sharded), compute-kernel configs, the all-reduce.

Layout contract of the port (TP over a (1, N) mesh, N = 4 on a QB2): the residual (the 4 hyper-connection streams) and
every sublayer input are REPLICATED on all dies; column-parallel weights shard their output dim, row-parallel weights
their input dim, and each sublayer ends in one all-reduce that leaves its output replicated.
"""

from __future__ import annotations

import os

import torch
import ttnn

TILE = 32


def num_devices(mesh) -> int:
    return int(mesh.get_num_devices())


def rep(mesh):
    return ttnn.ReplicateTensorToMesh(mesh) if num_devices(mesh) > 1 else None


def shard(mesh, dim):
    return ttnn.ShardTensorToMesh(mesh, dim=dim) if num_devices(mesh) > 1 else None


def _kw(mapper):
    return {"mesh_mapper": mapper} if mapper is not None else {}


def upload(mesh, t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mapper="rep", mem=None, cache=None):
    """torch -> device. mapper: "rep", ("shard", dim), or a ttnn mapper. cache: optional tensor-cache file name."""
    if mapper == "rep":
        mapper = rep(mesh)
    elif isinstance(mapper, tuple):
        mapper = shard(mesh, mapper[1])
    mem = mem or ttnn.DRAM_MEMORY_CONFIG
    if cache is not None:
        return ttnn.as_tensor(
            t.contiguous(), dtype=dtype, layout=layout, device=mesh, memory_config=mem, cache_file_name=cache, **_kw(mapper)
        )
    return ttnn.from_torch(t.contiguous(), dtype=dtype, layout=layout, device=mesh, memory_config=mem, **_kw(mapper))


def host(t, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, mesh=None):
    """A host-side (unplaced) mesh tensor, replicated: for copy_host_to_device_tensor into persistent inputs."""
    return ttnn.from_torch(t.contiguous(), dtype=dtype, layout=layout, **_kw(rep(mesh) if mesh is not None else None))


def to_torch0(t) -> torch.Tensor:
    """Device 0's copy of a (replicated) tensor."""
    return ttnn.to_torch(ttnn.get_device_tensors(t)[0])


def to_torch_cat(t, dim) -> torch.Tensor:
    """Concatenate every die's shard along dim (sharded tensors)."""
    return torch.cat([ttnn.to_torch(x) for x in ttnn.get_device_tensors(t)], dim=dim)


def ckc(mesh, fidelity=ttnn.MathFidelity.HiFi4, fp32=True, approx=False, l1_acc=False):
    return ttnn.init_device_compute_kernel_config(
        mesh.arch(), math_fidelity=fidelity, math_approx_mode=approx, fp32_dest_acc_en=fp32, packer_l1_acc=l1_acc
    )


class AllReduce:
    """Sum of a per-die partial over the (1, N) mesh, result replicated on every die.

    HY4_ALLREDUCE=composite (default): ttnn.all_reduce (fabric). Kept behind one class so the collective can be
    swapped (fused reduce-scatter + all-gather, or the CCL handles of the serving framework) without touching modules.
    """

    def __init__(self, mesh):
        self.mesh = mesh
        self.n = num_devices(mesh)
        self.topology = ttnn.Topology.Linear
        self.mode = os.environ.get("HY4_ALLREDUCE", "composite")

    def __call__(self, x, mem=None):
        if self.n == 1:
            return x
        mem = mem or ttnn.DRAM_MEMORY_CONFIG
        return ttnn.all_reduce(x, cluster_axis=1, topology=self.topology, memory_config=mem)


class AllGather:
    def __init__(self, mesh):
        self.mesh = mesh
        self.n = num_devices(mesh)

    def __call__(self, x, dim, mem=None):
        if self.n == 1:
            return x
        return ttnn.all_gather(
            x, dim=dim, cluster_axis=1, topology=ttnn.Topology.Linear, memory_config=mem or ttnn.DRAM_MEMORY_CONFIG
        )


def rms_scale_fp32(x_list, eps, n_total, mem=None):
    """rsqrt(mean over all of x_list's last dims of x^2 + eps) in fp32: [.., T, 1]. x_list tensors are fp32."""
    acc = None
    for x in x_list:
        s = ttnn.sum(ttnn.multiply(x, x, memory_config=mem), dim=-1, keepdim=True, memory_config=mem)
        acc = s if acc is None else ttnn.add(acc, s, memory_config=mem)
    m = ttnn.multiply(acc, 1.0 / float(n_total), memory_config=mem)
    return ttnn.rsqrt(ttnn.add(m, eps, memory_config=mem), memory_config=mem)


class RMSNorm:
    """HF HYV4FlashRMSNorm: y = w * bf16(x_f32 * rsqrt(mean(x^2) + eps)). Replicated weight, bf16 in/out."""

    def __init__(self, mesh, weight: torch.Tensor, eps: float):
        self.mesh, self.eps = mesh, float(eps)
        self.w = upload(mesh, weight.to(torch.bfloat16).reshape(1, 1, 1, -1), ttnn.bfloat16)
        self.ckc = ckc(mesh)

    def __call__(self, x, mem=None):
        return ttnn.rms_norm(x, epsilon=self.eps, weight=self.w, compute_kernel_config=self.ckc, memory_config=mem)
