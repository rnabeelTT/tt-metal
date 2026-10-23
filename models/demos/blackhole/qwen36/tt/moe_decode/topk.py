# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Fused decode router top-k + softmax for one token (Qwen3.5-MoE family: softmax router, renormalized top-k).

Replaces the ttnn chain topk (route-k prep / large indices / finish) -> 2 x slice -> softmax -> untilize (7 ops,
~25 us per layer on Blackhole) with one ``ttnn.generic_op`` on one core (kernels/router_topk.cpp).

Input: the router logits [1, 1, 32, E] BF16 TILE (row 0 = the token). Outputs: expert ids [1, 1, 1, k] UINT16
ROW_MAJOR (the sparse matmul ``indices`` layout) and weights [1, 1, 1, k] BF16 TILE, both L1 interleaved. Ties
go to the lower expert id; the softmax runs in Q16 fixed point (~1e-5 relative) and rounds to BF16.
"""

from __future__ import annotations

from pathlib import Path

import torch

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"


class RouterTopK:
    def __init__(self, mesh_device, num_experts: int, k: int, core=(0, 0)):
        assert num_experts % 32 == 0 and k <= 16, (num_experts, k)
        self.mesh_device = mesh_device
        self.E, self.k = num_experts, k
        self.core = ttnn.CoreCoord(*core)
        self.cores = ttnn.CoreRangeSet([ttnn.CoreRange(self.core, self.core)])
        self.ids_page_bytes = 2 * k
        self.scr_bytes = 2048 + 2 * num_experts + 64

    def alloc_outputs(self):
        """(ids, scores) output tensors (host upload: build them before any trace capture)."""
        rep = {"mesh_mapper": ttnn.ReplicateTensorToMesh(self.mesh_device)} if getattr(self.mesh_device, "get_num_devices", lambda: 1)() > 1 else {}
        ids = ttnn.from_torch(
            torch.zeros(1, 1, 1, self.k, dtype=torch.int32),
            dtype=ttnn.uint16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            **rep,
        )
        scores = ttnn.from_torch(
            torch.zeros(1, 1, 1, self.k, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            **rep,
        )
        return ids, scores

    def __call__(self, logits, ids, scores):
        cb_scr = 0
        cbs = [
            ttnn.CBDescriptor(
                total_size=self.scr_bytes,
                core_ranges=self.cores,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(buffer_index=cb_scr, data_format=ttnn.bfloat16, page_size=self.scr_bytes)
                ],
            )
        ]
        ct = [cb_scr, self.E // 32, self.k, self.ids_page_bytes]
        for t in (logits, ids, scores):
            ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        rt = ttnn.RuntimeArgs()
        rt[self.core.x][self.core.y] = [logits.buffer_address(), ids.buffer_address(), scores.buffer_address()]
        kernel = ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "router_topk.cpp"),
            core_ranges=self.cores,
            compile_time_args=ct,
            runtime_args=rt,
            config=ttnn.DataMovementConfigDescriptor(
                processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_0
            ),
        )
        program = ttnn.ProgramDescriptor(kernels=[kernel], semaphores=[], cbs=cbs)
        ttnn.generic_op([logits, ids, scores], program)
        return ids, scores
