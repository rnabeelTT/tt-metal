# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Partial sums of squares for a full-width QK RMSNorm (MiniMax-M2: one norm over all heads' q, one over all k, across
the tensor-parallel dies) as one ``ttnn.generic_op`` program per die: from the die's fused [q | k | v] projection
([1, 1, B <= 32, W] TILE bf16), P FP32 tiles [1, 1, 32, P * 32]: tile i < Pq = the row sums of squares of q's tile
columns [i * cq, (i + 1) * cq), tile Pq = the same over k's columns (every column of a tile holds the row's sum). After
an all_gather of these over the dies, ``AttnDecodePrep(norm="full")`` adds them up per user and normalizes."""

from __future__ import annotations

from pathlib import Path

import torch

import ttnn
from models.demos.blackhole.qwen36.tt.gdn_decode.ops import TILE, _cb, _cores, _ndev, _range_set, _rt

KERNEL_DIR = Path(__file__).parent / "kernels"
CB = dict(x=0, ones=1, sq=2, out=3)


def _dm(source, cores, ct, rt, riscv, noc):
    return ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / source),
        core_ranges=_range_set(cores),
        compile_time_args=ct,
        runtime_args=_rt(cores, rt),
        config=ttnn.DataMovementConfigDescriptor(processor=riscv, noc=noc),
    )


class QKStats:
    def __init__(self, mesh_device, q_width, k_width, q_chunks=6):
        """q_width / k_width: the die's q / k columns (multiples of 32); q split over q_chunks cores (+1 for k)."""
        self.mesh = mesh_device
        self.qt, self.kt = q_width // TILE, k_width // TILE
        if q_width % TILE or k_width % TILE or self.qt % q_chunks:
            raise ValueError(f"QKStats: q {q_width} / k {k_width} not split into {q_chunks} whole-tile chunks")
        self.Pq, self.P = q_chunks, q_chunks + 1
        self.cq = self.qt // q_chunks
        rep = {"mesh_mapper": ttnn.ReplicateTensorToMesh(mesh_device)} if _ndev(mesh_device) > 1 else {}
        self.ones = ttnn.from_torch(
            torch.ones(1, 1, TILE, TILE),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            **rep,
        )

    @property
    def parts(self):
        return self.Pq, self.P

    def __call__(self, qkv):
        out = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, TILE, self.P * TILE]), ttnn.float32, ttnn.TILE_LAYOUT, self.mesh, ttnn.DRAM_MEMORY_CONFIG
        )
        cores = _cores(self.mesh, self.P)
        cr = _range_set(cores)
        cnt_max = max(self.cq, self.kt)
        cbs = [
            _cb(cr, CB["x"], ttnn.bfloat16, 2048, cnt_max),
            _cb(cr, CB["ones"], ttnn.bfloat16, 2048, 1),
            _cb(cr, CB["sq"], ttnn.float32, 4096, cnt_max),
            _cb(cr, CB["out"], ttnn.float32, 4096, 1),
        ]
        spans = [(i * self.cq, self.cq) for i in range(self.Pq)] + [(self.qt, self.kt)]
        rd_ct = [CB["x"], CB["ones"]]
        for t in (qkv, self.ones):
            rd_ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        wr_ct = [CB["out"]] + ttnn.TensorAccessorArgs(out).get_compile_time_args()
        cfg = ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False
        )
        kernels = [
            _dm(
                "qkstats_reader.cpp",
                cores,
                rd_ct,
                [[qkv.buffer_address(), self.ones.buffer_address(), c0, n] for c0, n in spans],
                ttnn.DataMovementProcessor.RISCV_1,
                ttnn.NOC.NOC_0,
            ),
            _dm(
                "qkstats_writer.cpp",
                cores,
                wr_ct,
                [[out.buffer_address(), i] for i in range(self.P)],
                ttnn.DataMovementProcessor.RISCV_0,
                ttnn.NOC.NOC_1,
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "qkstats_compute.cpp"),
                core_ranges=cr,
                compile_time_args=[CB["x"], CB["ones"], CB["sq"], CB["out"]],
                runtime_args=_rt(cores, [[n] for _, n in spans]),
                config=cfg,
            ),
        ]
        ttnn.generic_op([qkv, self.ones, out], ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))
        return out
