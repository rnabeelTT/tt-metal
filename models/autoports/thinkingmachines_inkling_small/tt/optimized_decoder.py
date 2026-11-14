# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Inkling MoE block and decoder layer on the qwen35 autoport framework (module / class names as the hybrid overlay
expects: ``Qwen35MoEOptimized``, ``Qwen35MoeOptimizedDecoderLayer``).

Router (transformers `InklingTopkRouter`), with l = x @ W_gate^T over 258 rows (256 routed + 2 shared):

    picks   = top-6 of sigmoid(l[:256]) + bias
    weights = sigmoid(l[picks ∪ {256, 257}]) / sum(same 8) * route_scale * global_scale
    (= softmax of the 8 logsigmoids, HF's form: exp(logsigmoid(x)) = sigmoid(x))

The last two weights are the shared experts' gammas (HF multiplies them inside the shared SwiGLU before the linear
down projection, which equals scaling the shared expert's output). So the shared experts are experts 256, 257 of the
fused expert tensors and every token computes 8 weighted experts; the routed-expert paths of the base class need no
change. ``_route`` returns those 8 (weight, id) pairs.
"""

from __future__ import annotations

import os

import torch

import ttnn
from models.autoports.minimaxai_minimax_m2_7.tt.optimized_decoder import Qwen35MoEOptimized as _M27MoE

os.environ.setdefault("QWEN35_ROUTER_DRAM_SHARDED", "0")  # 258-wide router: the plain interleaved linear
os.environ.setdefault("QWEN35_MOE_FUSED_DECODE", "0")  # the fused decode kernels assume softmax / sigmoid-bias routers


class Qwen35MoEOptimized(_M27MoE):
    """Inkling MoE: 256 routed + 2 shared experts as one 258-expert block, router_kind "inkling"."""

    def __init__(self, mesh_device, args, state_dict, tensor_cache_path=None, tt_ccl=None, layer_num=None):
        super().__init__(mesh_device, args, state_dict, tensor_cache_path, tt_ccl=tt_ccl, layer_num=layer_num)
        self.n_routed = int(args.n_routed_experts)
        self.n_shared = int(args.n_shared_experts)
        self.top_k_routed = int(args.top_k_routed)
        assert self.num_experts == self.n_routed + self.n_shared and self.top_k == self.top_k_routed + self.n_shared
        tp = self.num_devices
        rep = {"mesh_mapper": ttnn.ReplicateTensorToMesh(mesh_device)} if tp > 1 else {}
        b = state_dict["gate.e_score_correction_bias"].float().reshape(1, 1, 1, -1)
        self.inkling_bias = ttnn.from_torch(b, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh_device, **rep)
        self.inkling_scale = float(args.route_scale) * float(state_dict["gate.global_scale"].float().reshape(-1)[0])
        self._shared_ids = {}

    def _shared_id_rows(self, T):
        """[1, 1, T, n_shared] uint32 ids (256, 257) for T rows (cached: decode row counts are built before tracing)."""
        t = self._shared_ids.get(T)
        if t is None:
            ids = torch.arange(self.n_routed, self.n_routed + self.n_shared, dtype=torch.int32).expand(T, -1)
            tp = self.num_devices
            t = ttnn.from_torch(
                ids.reshape(1, 1, T, -1).contiguous(), dtype=ttnn.uint32, layout=ttnn.TILE_LAYOUT, device=self.device,
                **({"mesh_mapper": ttnn.ReplicateTensorToMesh(self.device)} if tp > 1 else {}),
            )
            self._shared_ids[T] = t
        return t

    def _route(self, x4):
        """-> (weights bf16 [1, 1, T, 8], ids uint16 [1, 1, T, 8]): the 6 picks then the 2 shared experts."""
        T = int(x4.shape[-2])
        mc = ttnn.L1_MEMORY_CONFIG if T <= 32 else ttnn.DRAM_MEMORY_CONFIG
        self._route_persistent = False
        logits = self._router_logits(x4, mc, dtype=ttnn.float32)  # [1, 1, T, 258]
        sig = ttnn.sigmoid(logits, memory_config=mc)
        ttnn.deallocate(logits)
        R, S = self.n_routed, self.n_shared
        routed = ttnn.slice(sig, (0, 0, 0, 0), (1, 1, T, R), memory_config=mc)
        shared = ttnn.slice(sig, (0, 0, 0, R), (1, 1, T, R + S), memory_config=mc)
        ttnn.deallocate(sig)
        choice = ttnn.add(routed, self.inkling_bias, memory_config=mc)
        vals, ids = ttnn.topk(choice, k=self.top_k_routed, dim=-1, largest=True, sorted=True, memory_config=mc)
        ttnn.deallocate(vals)
        ttnn.deallocate(choice)
        w6 = ttnn.gather(routed, dim=-1, index=ids, memory_config=mc)  # [1, 1, T, 6] fp32 sigmoids of the picks
        ttnn.deallocate(routed)
        w8 = ttnn.concat([w6, shared], dim=-1, memory_config=mc)
        ttnn.deallocate(w6)
        ttnn.deallocate(shared)
        denom = ttnn.sum(w8, dim=-1, keepdim=True, memory_config=mc)
        denom = ttnn.multiply(denom, 1.0 / self.inkling_scale, memory_config=mc)
        wn = ttnn.div(w8, denom, memory_config=mc)
        ttnn.deallocate(w8)
        ttnn.deallocate(denom)
        scores = ttnn.typecast(wn, ttnn.bfloat16, memory_config=mc)
        ttnn.deallocate(wn)
        ids32 = ttnn.typecast(ids, ttnn.uint32, memory_config=mc) if ids.dtype != ttnn.uint32 else ids
        ids8 = ttnn.concat([ids32, self._shared_id_rows(T)], dim=-1, memory_config=mc)
        if ids32 is not ids:
            ttnn.deallocate(ids32)
        ttnn.deallocate(ids)
        ids16 = ttnn.typecast(ids8, ttnn.uint16, memory_config=mc)
        ttnn.deallocate(ids8)
        return scores, ids16


# the decoder layer under the name the shared code / overlay use
from models.autoports.thinkingmachines_inkling_small.tt.layer import InklingDecoderLayer as Qwen35MoeOptimizedDecoderLayer  # noqa: E402,F401
