# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Fused decoder layer for Qwen/Qwen3.5-122B-A10B (stage 2 of the autoport).

Same public contract as :class:`~.functional_decoder.FunctionalDecoder`; the graph is rewritten with
numerically equivalent fused forms (see doc/fused_decoder/README.md for the before/after evidence):

* Router: ``linear -> softmax -> topk -> sum -> div -> typecast -> untilize`` (10 ops) becomes one router
  matmul plus the dedicated ``generalized_moe_gate`` kernel (``models/common/modules/moe/TTMoEGate``:
  top-k + softmax-over-selected, which equals softmax-then-renormalize exactly).
* Shared expert: the three shared-input matmuls (gate, up, sigmoid-gate scalar) are one packed matmul
  ``[H, S | S | 1]`` followed by slices (shared-LHS rewrite); SiLU is folded into the multiply.
* Routed experts (prefill / masked groups): the routing weights scale the expert intermediate ``[E, C, N]``
  BEFORE the (linear) down projection instead of its ``[E, C, H]`` output, 3x less elementwise work, exact.
* Attention: q/k/v projections share their input and become one ``[H, 2*Hq*D + 2*Hkv*D]`` matmul + slices
  (``AttentionConfig.fuse_qkv``, consumed by the shared qwen36 gated-attention kernel).

Every rewrite is gated by :class:`FusionFlags` so each can be A/B measured against the functional graph.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import ttnn
from models.autoports.qwen_qwen3_5_122b_a10b.tt.functional_decoder import FunctionalDecoder
from models.autoports.qwen_qwen3_5_122b_a10b.tt.layer import Qwen35MoeDecoderLayer
from models.autoports.qwen_qwen3_5_122b_a10b.tt.moe import Qwen35MoE
from models.demos.blackhole.qwen36.utils.substate import substate


@dataclass(frozen=True)
class FusionFlags:
    router_gate_op: bool = True  # TTMoEGate (generalized_moe_gate kernel) instead of the softmax/topk chain
    shared_expert_packed: bool = True  # one packed [gate|up|sg] matmul for the shared expert
    weight_before_down: bool = True  # scale expert intermediates (not outputs) by routing weights in masked groups
    attn_fused_qkv: bool = True  # one fused q|k|v projection in the gated-attention kernel

    @classmethod
    def none(cls):
        return cls(False, False, False, False)


# TTMoEGate keeps four persistent L1-sharded buffers per instance (score-correction bias, input indices, output scores,
# output indices: one 32x32 tile per core for `buffer_rows` = all 110 cores, ~10.5 KB of L1 per core per layer). They are
# layer-independent constants / scratch outputs consumed right after the gate op, so one set is shared by every layer of a
# MODEL: with 48 layers the per-layer copies held 395 KB of L1 per core and the 2048-token prefill matmul programs no
# longer fit their circular buffers beside them (stage-6 full-model bring-up, 2026-09-24). The share is scoped to the
# model's args object (one per model build): a process that builds several models on one mesh (pytest sessions) must
# not hand a later model the first model's - by then deallocated - buffers (seen as wrong logits in the generator tests).
_GATE_SHARED_ATTRS = ("tt_bias", "tt_input_indices", "tt_output", "tt_output_indices")


def _share_gate_buffers(gate, args):
    key = (getattr(gate, "_buffer_rows", None), getattr(gate, "num_blocks", None), gate.use_fallback)
    cache = getattr(args, "_qwen35_shared_gate_buffers", None)
    if cache is None:
        cache = {}
        args._qwen35_shared_gate_buffers = cache
    shared = cache.get(key)
    if shared is None:
        cache[key] = {a: getattr(gate, a) for a in _GATE_SHARED_ATTRS if getattr(gate, a, None) is not None}
        return
    for a, t in shared.items():
        mine = getattr(gate, a, None)
        if mine is not None and mine is not t:
            ttnn.deallocate(mine)
        setattr(gate, a, t)


class Qwen35MoEFused(Qwen35MoE):
    """MoE block with the router / shared-expert / weighting rewrites."""

    def __init__(self, mesh_device, args, state_dict, tensor_cache_path=None, tt_ccl=None, layer_num=None):
        self.flags: FusionFlags = getattr(args, "fusion_flags", FusionFlags())
        super().__init__(mesh_device, args, state_dict, tensor_cache_path, tt_ccl=tt_ccl, layer_num=layer_num)
        tp = self.num_devices
        cache = tensor_cache_path
        if self.flags.router_gate_op:
            from models.common.modules.moe.tt_moe_gate import TTMoEGate
            from models.common.modules.moe.tt_moe_gate_config import TTMoEGateConfig

            self.gate = TTMoEGate(
                mesh_device,
                TTMoEGateConfig(
                    num_routed_experts=self.num_experts,
                    select_experts_k=self.top_k,
                    hidden_size=self.hidden,
                    n_group=1,
                    score_func="softmax",
                    routed_scaling_factor=1.0,
                    # The gate's derived 2D program config is sized for one M tile (decode batch <= 32);
                    # prefill routes hundreds of rows through the same object, so let ttnn pick the matmul
                    # program per shape. (The router matmul itself is ~5% of the fused layer at 70 us on 8 cores;
                    # the optimized decoder installs a decode-only program config for it.)
                    gate_matmul_auto_program_config=False,
                ),
                state_dict["gate.weight"].T.contiguous(),
            )
            if os.environ.get("QWEN35_SHARE_GATE_BUFFERS", "1") == "1":
                _share_gate_buffers(self.gate, args)
            # The functional router weight (1.5 MB) is kept: prefill rows use it (see _route); the gate op serves the
            # single-tile decode rows only.
        if self.flags.shared_expert_packed:
            import torch

            S = self.shared_inter
            Sl = S // tp
            w1 = state_dict["shared_expert.gate_proj.weight"].T  # [H, S]
            w3 = state_dict["shared_expert.up_proj.weight"].T  # [H, S]
            wg = state_dict["shared_expert_gate.weight"].T  # [H, 1]
            # per TP shard: [gate_d | up_d | sg] so each device slices its own columns; sg replicated.
            packed = torch.cat(
                [
                    torch.cat([w1[:, d * Sl : (d + 1) * Sl], w3[:, d * Sl : (d + 1) * Sl], wg], dim=-1)
                    for d in range(tp)
                ],
                dim=-1,
            ).contiguous()
            self.shared_packed_width = 2 * Sl + 1
            kw = {"mesh_mapper": ttnn.ShardTensorToMesh(mesh_device, dim=-1)} if tp > 1 else {}
            self.shared_packed = ttnn.as_tensor(
                packed,
                dtype=self.prec.shared,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                cache_file_name=str(cache / "mlp.shared_expert.packed") if cache else None,
                **kw,
            )
            for name in ("shared_w1", "shared_w3", "shared_gate_w"):
                ttnn.deallocate(getattr(self, name))
                setattr(self, name, None)

    # ---- router -------------------------------------------------------------------------------------
    def _route(self, x4):
        # The gate op is used for the single-user decode row only. With several valid rows its output was measured
        # bit-NONDETERMINISTIC (5 of 29 identical T=37 prefills and 4 of 11 identical B=32 decodes differed at bf16 ulp
        # level; 0 of 29 with this functional router; 0 of 29+29 for the single-row decode) — see
        # doc/optimized_decoder/work_log.md (2026-09-24 06:35 / 07:05).
        phys = list(x4.padded_shape)
        if not self.flags.router_gate_op or x4.shape[2] != 1 or phys[-2] != 32 or any(d != 1 for d in phys[:-2]):
            return super()._route(x4)
        weights, ids = self.gate.forward(x4)  # [T,1,1,k] bf16 / uint16, ROW_MAJOR
        T = weights.shape[0]
        # Metadata-only views into the [1,1,T,k] layout the expert paths consume.
        scores = ttnn.reshape(weights, (1, 1, T, self.top_k))
        ids = ttnn.reshape(ids, (1, 1, T, self.top_k))
        return scores, ids

    def _routing_matrix(self, scores, ids, T):
        if self.flags.router_gate_op:
            # scatter wants tiled operands; the gate op returns row-major views.
            if scores.layout != ttnn.TILE_LAYOUT:
                scores = ttnn.to_layout(scores, ttnn.TILE_LAYOUT)
            if ids.layout != ttnn.TILE_LAYOUT:
                ids = ttnn.to_layout(ids, ttnn.TILE_LAYOUT)
        return super()._routing_matrix(scores, ids, T)

    # ---- shared expert ----------------------------------------------------------------------------------
    def _shared(self, x4):
        if not self.flags.shared_expert_packed:
            return super()._shared(x4)
        mc = ttnn.DRAM_MEMORY_CONFIG
        Sl = self.shared_inter // self.num_devices
        T = x4.shape[2]
        fused = ttnn.linear(x4, self.shared_packed, compute_kernel_config=self.shared_ckc, memory_config=mc)
        gate = ttnn.slice(fused, [0, 0, 0, 0], [1, 1, T, Sl], memory_config=mc)
        up = ttnn.slice(fused, [0, 0, 0, Sl], [1, 1, T, 2 * Sl], memory_config=mc)
        sg = ttnn.slice(fused, [0, 0, 0, 2 * Sl], [1, 1, T, 2 * Sl + 1], memory_config=mc)
        ttnn.deallocate(fused)
        hidden = ttnn.multiply(up, gate, input_tensor_b_activations=[ttnn.UnaryOpType.SILU], memory_config=mc)
        ttnn.deallocate(gate)
        ttnn.deallocate(up)
        out = ttnn.linear(hidden, self.shared_w2, compute_kernel_config=self.shared_ckc, memory_config=mc)
        ttnn.deallocate(hidden)
        # sigmoid folded into the multiply as an input activation (op merge: one program instead of two)
        out = ttnn.multiply(out, sg, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID], memory_config=mc)
        ttnn.deallocate(sg)
        return out

    # ---- routed experts, masked groups ------------------------------------------------------------------
    def _routed_masked(self, x4, scores, ids, T, chunk):
        if not self.flags.weight_before_down:
            return super()._routed_masked(x4, scores, ids, T, chunk)
        E, Nl, H = self.num_experts, self.inter_local, self.hidden
        dram = ttnn.DRAM_MEMORY_CONFIG
        routing = self._routing_matrix(scores, ids, T)
        pad = (-T) % chunk
        if pad:
            x4 = ttnn.pad(x4, [(0, 0), (0, 0), (0, pad), (0, 0)], value=0.0)
            routing = ttnn.pad(routing, [(0, 0), (0, 0), (0, pad), (0, 0)], value=0.0)
        Tp = T + pad
        from models.autoports.qwen_qwen3_5_122b_a10b.tt.moe import sparse_matmul_program_config

        cfg_gu = (
            self.cfg_gate_up_masked
            if chunk == self.PREFILL_CHUNK
            else sparse_matmul_program_config(
                chunk, 2 * Nl, H, target_cores=32, in0_block_w_cap=64, tile_w=self.tile_gu.tile_shape[1]
            )
        )
        cfg_dn = (
            self.cfg_down_masked
            if chunk == self.PREFILL_CHUNK
            else sparse_matmul_program_config(chunk, H, Nl, target_cores=32, in0_block_w_cap=32)
        )
        outs = []
        for s0 in range(0, Tp, chunk):
            x_c = ttnn.slice(x4, [0, 0, s0, 0], [1, 1, s0 + chunk, H], memory_config=dram)
            r_c = ttnn.slice(routing, [0, 0, s0, 0], [1, 1, s0 + chunk, E], memory_config=dram)
            mask = ttnn.to_layout(
                ttnn.sum(r_c, dim=2, keepdim=True, memory_config=dram), ttnn.ROW_MAJOR_LAYOUT, memory_config=dram
            )
            fused = ttnn.sparse_matmul(
                x_c,
                self.up_gate,
                sparsity=mask,
                nnz=None,
                memory_config=dram,
                output_tile=self.tile_gu,
                program_config=cfg_gu,
                compute_kernel_config=self.routed_ckc,
                dtype=self.prec.act_dtype,
            )
            ttnn.deallocate(x_c)
            fused = ttnn.reshape(fused, (1, E, chunk, 2 * Nl))
            din = ttnn.swiglu(fused, memory_config=dram)  # [1,E,chunk,Nl]
            ttnn.deallocate(fused)
            # routing weights on the intermediate (linear down projection follows -> exact)
            weights_c = ttnn.permute(r_c, (0, 3, 2, 1), memory_config=dram)  # [1,E,chunk,1]
            din = ttnn.multiply(din, weights_c, memory_config=dram)
            ttnn.deallocate(weights_c)
            ttnn.deallocate(r_c)
            down = ttnn.sparse_matmul(
                din,
                self.down,
                sparsity=mask,
                nnz=None,
                is_input_a_sparse=True,
                is_input_b_sparse=True,
                memory_config=dram,
                output_tile=self.tile,
                program_config=cfg_dn,
                compute_kernel_config=self.routed_ckc,
                dtype=self.prec.act_dtype,
            )  # [1,E,chunk,H]
            ttnn.deallocate(din)
            ttnn.deallocate(mask)
            outs.append(ttnn.experimental.fast_reduce_nc(ttnn.reshape(down, (1, E, chunk, H)), dims=[1]))
            ttnn.deallocate(down)
        ttnn.deallocate(routing)
        acc = outs[0] if len(outs) == 1 else ttnn.concat(outs, dim=2, memory_config=dram)
        if len(outs) > 1:
            for o in outs:
                ttnn.deallocate(o)
        acc = ttnn.reshape(acc, (1, 1, Tp, H))
        if pad:
            acc = ttnn.slice(acc, [0, 0, 0, 0], [1, 1, T, H], memory_config=dram)
        return acc


class Qwen35MoeFusedDecoderLayer(Qwen35MoeDecoderLayer):
    def _build_feed_forward(self, mesh_device, args, state_dict, layer_num, tensor_cache_path, tt_ccl):
        if not args.is_moe_layer(layer_num):
            return super()._build_feed_forward(mesh_device, args, state_dict, layer_num, tensor_cache_path, tt_ccl)
        mlp_state = substate(state_dict, f"layers.{layer_num}.mlp")
        mlp_cache = (tensor_cache_path / f"layers.{layer_num}") if tensor_cache_path else None
        return Qwen35MoEFused(mesh_device, args, mlp_state, mlp_cache, tt_ccl=tt_ccl, layer_num=layer_num)


class FusedDecoder(FunctionalDecoder):
    """Decoder layer with the fused graph. Same contract as FunctionalDecoder."""

    LAYER_CLS = Qwen35MoeFusedDecoderLayer

    @classmethod
    def from_state_dict(cls, state_dict, *, hf_config, layer_idx, mesh_device, fusion_flags=None, **kwargs):
        flags = fusion_flags or FusionFlags()
        return super().from_state_dict(
            state_dict,
            hf_config=hf_config,
            layer_idx=layer_idx,
            mesh_device=mesh_device,
            _args_hook=lambda args: (
                setattr(args, "fusion_flags", flags),
                setattr(args, "attn_fuse_qkv", flags.attn_fused_qkv),
            ),
            **kwargs,
        )
