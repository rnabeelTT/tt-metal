# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Gated full-attention decode head prep for 1-32 users as one ``ttnn.generic_op`` program (Qwen3.5 / Qwen3-Next
attention layers, per tensor-parallel die), between the fused QKV projection and the paged cache update / SDPA decode:

From row b of the fused [q | k | v | gate] projection ([1, 1, B, (2 NH + 2 NKV) HD] TILE bf16) it builds, with the
heads of a user as the rows of its tiles (the layout SDPA decode and paged_update_cache read):

* q  [1, B, NH, HD]: per-head RMSNorm x (1 + w_q), then the HF rotate-half RoPE on the first rope_dim dims;
* k  [1, B, NKV, HD]: the same with w_k;
* v  [1, B, NKV, HD]: copied;
* gate [1, B, NH, HD]: sigmoid(gate) (the post-SDPA output gate).

Work items: (user, part) with part 0 = q, 1 = k + v, 2 = gate. cos / sin are the decode tables [1, B, 1, rope_dim]
(TILE, row 0 of each user's tile row). Constants (``prepare``): one [2, 1, 32, HD] FP32 tensor, row 0 of slice 0 = 1 +
w_q, of slice 1 = 1 + w_k. Outputs are persistent per width (``outputs``; build every width before trace capture).
"""

from __future__ import annotations

from pathlib import Path

import torch

import ttnn
from models.demos.blackhole.qwen36.tt.gdn_decode.ops import (
    TILE,
    _bits,
    _cb,
    _cores,
    _ndev,
    _per_device,
    _range_set,
    _rt,
    _split,
)

KERNEL_DIR = Path(__file__).parent / "kernels"
NUM_CBS = 64
# prep program CB ids (kernels/prep_compute.cpp)
CB = dict(x=0, w=1, cs=2, ones=3, sq=4, r=5, xn=6, out=7, v=8)
UNPACK_FP32 = ("w", "r", "xn")  # sq is a matmul operand (Default unpack)


def _dm(source, cores, ct, rt, riscv, noc):
    return ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / source), core_ranges=_range_set(cores), compile_time_args=ct,
        runtime_args=_rt(cores, rt),
        config=ttnn.DataMovementConfigDescriptor(processor=riscv, noc=noc),
    )


class AttnDecodePrep:
    """One instance per model (shape set); the layers share its output buffers (each layer consumes them before the next
    writes them)."""

    def __init__(self, mesh_device, num_heads, num_kv_heads, head_dim, rope_dim, max_batch=32, eps=1e-6):
        self.mesh = mesh_device
        self.NH, self.NKV, self.HD, self.rd = num_heads, num_kv_heads, head_dim, rope_dim
        assert head_dim % TILE == 0 and rope_dim == 2 * TILE, "rotate-half over the first two tiles of a head"
        assert num_heads <= TILE and num_kv_heads <= TILE and max_batch <= TILE
        self.Ht = head_dim // TILE
        self.eps = eps
        self.max_batch = max_batch
        self._rep = {"mesh_mapper": ttnn.ReplicateTensorToMesh(mesh_device)} if _ndev(mesh_device) > 1 else {}
        # matmul operand for the per-row sum of squares: column 0 all ones
        ones = torch.zeros(1, 1, TILE, TILE)
        ones[..., :, 0] = 1.0
        self.ones = ttnn.from_torch(ones, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device,
                                    memory_config=ttnn.DRAM_MEMORY_CONFIG, **self._rep)
        self._out = {}

    def prepare(self, q_norm_w, k_norm_w):
        """(1 + w) norm weights as ttnn tensors or torch vectors of HD -> the [2, 1, 32, HD] FP32 constant."""
        def vec(t):
            if isinstance(t, torch.Tensor):
                return t.reshape(-1).float()
            return _per_device(t)[0].reshape(-1).float()
        w = torch.zeros(2, 1, TILE, self.HD)
        w[0, 0, 0] = vec(q_norm_w)[: self.HD]
        w[1, 0, 0] = vec(k_norm_w)[: self.HD]
        return ttnn.from_torch(w, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=self.mesh,
                               memory_config=ttnn.DRAM_MEMORY_CONFIG, **self._rep)

    def outputs(self, B):
        """(q, k, v, gate) for width B: [1, B, heads, HD] bf16 TILE, L1 interleaved, persistent."""
        if B not in self._out:
            mk = lambda h: ttnn.from_torch(torch.zeros(1, B, h, self.HD), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                           device=self.mesh, memory_config=ttnn.L1_MEMORY_CONFIG, **self._rep)
            self._out[B] = (mk(self.NH), mk(self.NKV), mk(self.NKV), mk(self.NH))
        return self._out[B]

    def __call__(self, qkv, cos, sin, const, B):
        q, k, v, g = self.outputs(B)
        units = 3 * B
        grid = self.mesh.compute_with_storage_grid_size()
        ncores = min(units, grid.x * grid.y)
        cores = _cores(self.mesh, ncores)
        work = _split(units, ncores)
        cr = _range_set(cores)
        Ht = self.Ht
        f32, bf = (ttnn.float32, 4096), (ttnn.bfloat16, 2048)
        sizes = dict(x=(bf, 2 * Ht), w=(f32, 2 * Ht), cs=(bf, 8), ones=(bf, 1), sq=(f32, Ht), r=(f32, 1), xn=(f32, 2),
                     out=(bf, 2 * Ht), v=(bf, 2 * Ht))
        cbs = [_cb(cr, CB[n], dt, tb, c) for n, ((dt, tb), c) in sizes.items()]
        wq = qkv.shape[-1] // TILE  # tiles per qkv row
        rd_ct = [Ht, self.NH, self.NKV, wq, CB["x"], CB["w"], CB["cs"], CB["ones"], CB["v"]]
        for t in (qkv, cos, sin, const, self.ones):
            rd_ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        wr_ct = [Ht, CB["out"], CB["v"]]
        for t in (q, k, v, g):
            wr_ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        rd_rt = [[qkv.buffer_address(), cos.buffer_address(), sin.buffer_address(), const.buffer_address(),
                  self.ones.buffer_address(), s, c] for s, c in work]
        wr_rt = [[q.buffer_address(), k.buffer_address(), v.buffer_address(), g.buffer_address(), s, c]
                 for s, c in work]
        cp_ct = [Ht, _bits(self.eps), _bits(1.0 / self.HD)]
        utd = [ttnn.UnpackToDestMode.Default] * NUM_CBS
        for n in UNPACK_FP32:
            utd[CB[n]] = ttnn.UnpackToDestMode.UnpackToDestFp32
        cfg = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True,
                                           math_approx_mode=False)
        cfg.unpack_to_dest_mode = utd
        kernels = [
            _dm("prep_reader.cpp", cores, rd_ct, rd_rt, ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.NOC_0),
            _dm("prep_writer.cpp", cores, wr_ct, wr_rt, ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.NOC_1),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "prep_compute.cpp"), core_ranges=cr, compile_time_args=cp_ct,
                runtime_args=_rt(cores, [[s, c] for s, c in work]), config=cfg,
            ),
        ]
        program = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
        ttnn.generic_op([qkv, cos, sin, const, self.ones, q, k, v, g], program)
        return q, k, v, g


class AttnDecodePost:
    """attn_out ([1, B, NH, HD] from SDPA decode, heads as rows) x gate ([1, B, NH, HD], sigmoid already applied) ->
    the user's row of the out-projection input [1, B, NH * HD] (bf16 TILE L1, persistent per width): one work item per
    user, the product's row h of tile j written to row b of tile h * Ht + j (two 32-byte face-row writes)."""

    def __init__(self, mesh_device, num_heads, head_dim):
        self.mesh = mesh_device
        self.NH, self.HD, self.Ht = num_heads, head_dim, head_dim // TILE
        self._rep = {"mesh_mapper": ttnn.ReplicateTensorToMesh(mesh_device)} if _ndev(mesh_device) > 1 else {}
        self._y = {}

    def output(self, B):
        if B not in self._y:
            self._y[B] = ttnn.from_torch(torch.zeros(1, B, self.NH * self.HD), dtype=ttnn.bfloat16,
                                         layout=ttnn.TILE_LAYOUT, device=self.mesh,
                                         memory_config=ttnn.L1_MEMORY_CONFIG, **self._rep)
        return self._y[B]

    def __call__(self, attn_out, gate, B):
        y = self.output(B)
        grid = self.mesh.compute_with_storage_grid_size()
        ncores = min(B, grid.x * grid.y)
        cores = _cores(self.mesh, ncores)
        work = _split(B, ncores)
        cr = _range_set(cores)
        Ht = self.Ht
        bf = (ttnn.bfloat16, 2048)
        cbs = [_cb(cr, 0, *bf, 2 * Ht), _cb(cr, 1, *bf, 2 * Ht), _cb(cr, 2, *bf, 2 * Ht)]
        rd_ct = [Ht, 0, 1]
        for t in (attn_out, gate):
            rd_ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        wr_ct = [Ht, self.NH, 2]
        wr_ct += ttnn.TensorAccessorArgs(y).get_compile_time_args()
        cfg = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True,
                                           math_approx_mode=False)
        kernels = [
            _dm("post_reader.cpp", cores, rd_ct, [[attn_out.buffer_address(), gate.buffer_address(), s, c]
                                                  for s, c in work], ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.NOC_0),
            _dm("post_writer.cpp", cores, wr_ct, [[y.buffer_address(), s, c] for s, c in work],
                ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.NOC_1),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "post_compute.cpp"), core_ranges=cr, compile_time_args=[Ht],
                runtime_args=_rt(cores, [[c] for _, c in work]), config=cfg,
            ),
        ]
        program = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
        ttnn.generic_op([attn_out, gate, y], program)
        return y
