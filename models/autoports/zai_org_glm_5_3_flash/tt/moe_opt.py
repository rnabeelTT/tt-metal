# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Optimized MoE block of the GLM-5.3-Flash autoport: the fused / optimized MoE of the GLM-4.5-Air / Qwen3.5 autoports
(models/autoports/zai_org_glm_4_5_air/tt/{fused_decoder,optimized_decoder}.py) with the GLM-5.3 SwiGLU clamp at every
activation site and fp32 destination accumulation on the routed expert projections.

Paths: decode rows go through the gather (<= 4 rows) or the pairs path; prefill rows through the expert-sorted
dispatch + grouped sparse matmuls (tt/moe_dispatch.py). The base class's masked path is not used for prefill: its
sparse_matmul is biased by a bf16 accumulator (+5 % on the K = 4096 projection) and returns wrong results with fp32
accumulation on Blackhole (tools/sparse_mm_probe.py, 2026-10-14).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, replace

import torch
from loguru import logger

import ttnn
from models.demos.blackhole.qwen36.tt import tp_common as tpc

from .moe import TILE, MoEPrecision, Qwen35MoE, sparse_matmul_program_config
from .moe_dispatch import SortedDispatch, gather_rows, undispatch_rows

# Qwen3-Coder-Next perf pass: at one decode token the pairs path repeats the TILE row block on the batch dim instead of
# the row-major to_layout / repeat / tilize round trip (two tilizes, ~24 us per layer).
PAIRS_T1_TILE_REPEAT = os.environ.get("QWEN35_PAIRS_T1_TILE_REPEAT", "0") == "1"


def sorted_rows_cap() -> int:
    """GLM53_MOE_SORTED_ROWS = N: the expert-sorted prefill path runs the routed experts of a chunk wider than N rows in
    N-row blocks (bit-identical; caps the worst-case dispatch buffers, Qwen35MoEOptimized._routed_sorted_split).
    0 = whole chunk. Read per call so a probe can A/B it in one process."""
    return int(os.environ.get("GLM53_MOE_SORTED_ROWS", str(SORTED_ROWS_DEFAULT)) or 0)


SORTED_ROWS_DEFAULT = 1024


@dataclass(frozen=True)
class FusionFlags:
    # TTMoEGate (generalized_moe_gate kernel) instead of the softmax/topk chain; it only ever ran for the single-row
    # decode (see _route) and is the root cause of the width-1 decode non-determinism (perf pass, doc/perf_pass/): OFF by
    # default since 2026-09-29; QWEN35_ROUTER_GATE_OP=1 re-enables it for experiments.
    router_gate_op: bool = os.environ.get("QWEN35_ROUTER_GATE_OP", "0") == "1"
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

    def _swiglu_clamped(self, fused, memory_config):
        """ttnn.swiglu on a packed [up | gate] projection after the swiglu_limit clamp; `fused` stays owned by the caller
        (it may be a persistent output buffer)."""
        if self.swiglu_limit is None:
            return ttnn.swiglu(fused, memory_config=memory_config)
        if getattr(self, "fused_swiglu_act", False) and tuple(fused.tile.tile_shape) == (32, 32):
            # fused decoder: clamps and SiLU as input activations of one multiply (2 slices + 1 multiply instead of
            # minimum + tensor maximum + swiglu's slice / slice / silu / multiply; bit-identical, tools/fusing_probe.py:
            # 1307 -> 665 us at 32 decode users, 4635 -> 2352 us at 2048 prefill rows)
            L = float(self.swiglu_limit)
            shp = list(fused.shape)
            nl = shp[-1] // 2
            up = ttnn.slice(fused, [0] * len(shp), shp[:-1] + [nl], memory_config=memory_config)
            gate = ttnn.slice(fused, [0] * (len(shp) - 1) + [nl], shp, memory_config=memory_config)
            out = ttnn.multiply(
                up,
                gate,
                input_tensor_a_activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.HARDTANH, -L, L)],
                input_tensor_b_activations=[
                    ttnn.UnaryWithParam((ttnn.UnaryOpType.MINIMUM, L)),
                    ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU),
                ],
                memory_config=memory_config,
            )
            ttnn.deallocate(up)
            ttnn.deallocate(gate)
            return out
        c = ttnn.minimum(fused, float(self.swiglu_limit))
        c2 = ttnn.maximum(c, self._clamp_lo)
        ttnn.deallocate(c)
        out = ttnn.swiglu(c2, memory_config=memory_config)
        ttnn.deallocate(c2)
        return out

    def _clamp_gate_up(self, gate, up, mc):
        """swiglu_limit on separate gate / up slices (gate <= L, up in [-L, L]); frees the inputs."""
        if self.swiglu_limit is None:
            return gate, up
        L = float(self.swiglu_limit)
        g = ttnn.minimum(gate, L, memory_config=mc)
        u = ttnn.clamp(up, min=-L, max=L, memory_config=mc)
        ttnn.deallocate(gate)
        ttnn.deallocate(up)
        return g, u

    def __init__(self, mesh_device, args, state_dict, tensor_cache_path=None, tt_ccl=None, layer_num=None):
        self.flags: FusionFlags = getattr(args, "fusion_flags", FusionFlags())
        super().__init__(mesh_device, args, state_dict, tensor_cache_path, tt_ccl=tt_ccl, layer_num=layer_num)
        tp = self.num_devices
        cache = tensor_cache_path
        if self.flags.router_gate_op:
            assert self.router_kind == "softmax", "the TTMoEGate gate op is wired for the softmax router only"
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
            self.shared_packed_width = 2 * Sl + (1 if self.shared_gated else 0)

            def packed_shared():
                w1 = state_dict["shared_expert.gate_proj.weight"].T  # [H, S]
                w3 = state_dict["shared_expert.up_proj.weight"].T  # [H, S]
                # per TP shard: [gate_d | up_d | sg] so each device slices its own columns; sg replicated. Ungated shared
                # expert (GLM): [gate_d | up_d].
                wg = [state_dict["shared_expert_gate.weight"].T] if self.shared_gated else []  # [H, 1]
                return torch.cat(
                    [
                        torch.cat([w1[:, d * Sl : (d + 1) * Sl], w3[:, d * Sl : (d + 1) * Sl], *wg], dim=-1)
                        for d in range(tp)
                    ],
                    dim=-1,
                ).contiguous()

            name = "mlp.shared_expert.packed" if self.shared_gated else "mlp.shared_expert.packed_ng"
            kw = {"mesh_mapper": ttnn.ShardTensorToMesh(mesh_device, dim=-1)} if tp > 1 else {}
            # tt/weight_cache.WeightCache: the final device tensor, state_dict read on a miss
            if hasattr(cache, "store"):
                self.shared_packed = cache.get(
                    name,
                    lambda: ttnn.from_torch(
                        packed_shared(),
                        dtype=self.prec.shared,
                        layout=ttnn.TILE_LAYOUT,
                        device=mesh_device,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        **kw,
                    ),
                )
            else:
                self.shared_packed = ttnn.as_tensor(
                    packed_shared(),
                    dtype=self.prec.shared,
                    layout=ttnn.TILE_LAYOUT,
                    device=mesh_device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    cache_file_name=str(cache / name) if cache else None,
                    **kw,
                )
            for name in ("shared_w1", "shared_w3", "shared_gate_w"):
                if getattr(self, name) is not None:
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
        if getattr(self.args, "moe_gate_fast", False) and self.gate.num_blocks == 1 and self.gate.n_group != 8:
            weights, ids = self._gate_fast_forward(x4)  # [T,1,1,k] bf16 / uint16, ROW_MAJOR
        else:
            weights, ids = self.gate.forward(x4)  # [T,1,1,k] bf16 / uint16, ROW_MAJOR
        T = weights.shape[0]
        # Metadata-only views into the [1,1,T,k] layout the expert paths consume.
        scores = ttnn.reshape(weights, (1, 1, T, self.top_k))
        ids = ttnn.reshape(ids, (1, 1, T, self.top_k))
        return scores, ids

    def _gate_fast_buffers(self, bpi):
        """The gate op's bias / input-index / output / output-index tensors for a ``bpi``-token iteration, built ONCE
        (sliced out of the TTMoEGate persistent buffers) and shared by every layer of the model (keyed on args). The op
        requires all five tensors to have the same leading (token) shape, so the stock forward slices them per step."""
        g = self.gate
        key = (bpi, g.num_blocks)
        cache = getattr(self.args, "_qwen35_gate_fast_bufs", None)
        if cache is None:
            cache = {}
            self.args._qwen35_gate_fast_bufs = cache
        bufs = cache.get(key)
        if bufs is None:
            mem_in = g._sharded_mem_config(bpi, (g.num_blocks * 32, 32))
            mem_out = g._sharded_mem_config(bpi, (32, 32))
            bias = ttnn.slice(g.tt_bias, [0, 0, 0], [bpi, 16, 16], memory_config=mem_in)
            in_idx = ttnn.slice(g.tt_input_indices, [0, 0, 0], [bpi, 16, 16], memory_config=mem_in)
            out = ttnn.slice(g.tt_output, [0, 0, 0], [bpi, 32, 32], memory_config=mem_out)
            out_idx = ttnn.slice(g.tt_output_indices, [0, 0, 0], [bpi, 32, 32], memory_config=mem_out)
            bufs = {"mem_in": mem_in, "bias": bias, "in_idx": in_idx, "out": out, "out_idx": out_idx}
            cache[key] = bufs
        return bufs

    def _gate_fast_forward(self, x4):
        """Decode router through the gate op with the per-step constant traffic removed (stage 7, ``moe_gate_fast``).

        ``TTMoEGate.forward`` slices its four persistent buffers (bias, input indices, output scores, output indices)
        down to the iteration batch EVERY step (4 ops) and moves both outputs sharded->interleaved before slicing the
        top-k columns (2 ops). Here the per-iteration tensors are built once (:meth:`_gate_fast_buffers`) and the
        top-k columns are sliced straight out of the sharded row-major output buffers into L1 when the slice op
        supports it. ``moe_router_l1`` puts the router logits in L1 so the (bpi,16,16) relayout runs on the L1 path.
        Numerics are the stock path's: same op, same inputs, same output rows."""
        g = self.gate
        logits = ttnn.matmul(
            x4,
            g.tt_gate_weight,
            compute_kernel_config=g.compute_kernel_config,
            program_config=g.matmul_program_config,
            memory_config=ttnn.L1_MEMORY_CONFIG if getattr(self.args, "moe_router_l1", False) else None,
        )
        if g.score_transform == "softmax":
            logits = ttnn.softmax(logits, dim=-1)
        bpi = logits.shape[2]  # one iteration: decode rows (32) <= cores and <= buffer rows
        assert bpi <= min(g.num_device_cores, g._buffer_rows), (bpi, g.num_device_cores, g._buffer_rows)
        bufs = self._gate_fast_buffers(bpi)
        cur = ttnn.reshape(logits, (bpi, 16, 16))
        cur = ttnn.to_memory_config(cur, memory_config=bufs["mem_in"])  # height-shard: one token per core
        ttnn.deallocate(logits)
        w, idx = ttnn.experimental.deepseek.moe.generalized_moe_gate(
            cur,
            bias_tensor=bufs["bias"],
            input_indices_tensor=bufs["in_idx"],
            output_tensor=bufs["out"],
            output_indices_tensor=bufs["out_idx"],
            eps=g.eps,
            scaling_factor=g.scaling_factor,
            enable_sigmoid=g.enable_sigmoid,
            topk=g.k,
            output_softmax=g.output_softmax,
        )
        ttnn.deallocate(cur)
        T = x4.shape[2]
        k = g.k
        if not hasattr(self, "_gate_fast_direct_slice"):
            # first (eager) call decides whether ttnn.slice reads the height-sharded row-major output buffers straight
            # into interleaved L1; the decision is fixed before any trace capture (same path every step).
            try:
                weights = ttnn.slice(w, [0, 0, 0], [T, 1, k], memory_config=ttnn.L1_MEMORY_CONFIG)
                indices = ttnn.slice(idx, [0, 0, 0], [T, 1, k], memory_config=ttnn.L1_MEMORY_CONFIG)
                self._gate_fast_direct_slice = True
            except Exception as e:  # noqa: BLE001 - op support probe
                logger.warning(
                    f"moe_gate_fast: sharded->interleaved slice unsupported ({str(e)[:120]}); using S2I + slice"
                )
                self._gate_fast_direct_slice = False
        if self._gate_fast_direct_slice:
            weights = ttnn.slice(w, [0, 0, 0], [T, 1, k], memory_config=ttnn.L1_MEMORY_CONFIG)
            indices = ttnn.slice(idx, [0, 0, 0], [T, 1, k], memory_config=ttnn.L1_MEMORY_CONFIG)
        else:
            w_il = ttnn.to_memory_config(w, memory_config=ttnn.L1_MEMORY_CONFIG)
            i_il = ttnn.to_memory_config(idx, memory_config=ttnn.L1_MEMORY_CONFIG)
            weights = ttnn.slice(w_il, [0, 0, 0], [T, 1, k], memory_config=ttnn.L1_MEMORY_CONFIG)
            indices = ttnn.slice(i_il, [0, 0, 0], [T, 1, k], memory_config=ttnn.L1_MEMORY_CONFIG)
            ttnn.deallocate(w_il)
            ttnn.deallocate(i_il)
        return ttnn.view(weights, (T, 1, 1, k)), ttnn.view(indices, (T, 1, 1, k))

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
        sg = (
            ttnn.slice(fused, [0, 0, 0, 2 * Sl], [1, 1, T, 2 * Sl + 1], memory_config=mc) if self.shared_gated else None
        )
        ttnn.deallocate(fused)
        gate, up = self._clamp_gate_up(gate, up, mc)
        hidden = ttnn.multiply(up, gate, input_tensor_b_activations=[ttnn.UnaryOpType.SILU], memory_config=mc)
        ttnn.deallocate(gate)
        ttnn.deallocate(up)
        out = ttnn.linear(hidden, self.shared_w2, compute_kernel_config=self.shared_ckc, memory_config=mc)
        ttnn.deallocate(hidden)
        if sg is None:
            return out
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
        from models.autoports.zai_org_glm_4_5_air.tt.moe import sparse_matmul_program_config

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
            din = self._swiglu_clamped(fused, dram)  # [1,E,chunk,Nl]
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


@dataclass(frozen=True)
class OptimizationConfig:
    # --- precision / fidelity policy (per tensor group) ---
    routed_gate_up_dtype: object = ttnn.bfloat4_b
    routed_down_dtype: object = ttnn.bfloat4_b
    routed_fidelity: object = ttnn.MathFidelity.LoFi
    shared_dtype: object = ttnn.bfloat8_b
    shared_fidelity: object = ttnn.MathFidelity.HiFi2
    router_dtype: object = ttnn.bfloat16  # router logits matmul weight (never bfp4: routing numerics)
    router_fidelity: object = ttnn.MathFidelity.HiFi4
    act_dtype: object = ttnn.bfloat16
    attn_weight_dtype: object = ttnn.bfloat8_b
    gdn_weight_dtype: object = ttnn.bfloat8_b
    kv_dtype: object = (
        ttnn.bfloat8_b
    )  # -4.5% attention decode at 16K context with k_chunk 256, PCC-neutral (sweep kv_final)
    # --- sparse matmul geometry ---
    gather_gate_up_cores: int = 64
    gather_gate_up_in0: int = 32
    gather_gate_up_output_tile: tuple | None = (
        None  # e.g. (32, 16): 16-wide output tiles -> 2x the gate/up grid (stage 5)
    )
    gather_down_cores: int = 64
    gather_down_in0: int = 32
    masked_chunk: int = 64
    masked_gate_up_cores: int = 32
    masked_gate_up_in0: int = 64
    masked_down_cores: int = 32
    masked_down_in0: int = 32
    gather_max_tokens: int = 4
    # decode "pairs" path (perf pass): one group-parallel sparse matmul per projection over all (token, expert) pairs
    # of the step instead of the per-token gather loop / the serial masked path; 0 = off (QWEN35_MOE_PAIRS_MAX_TOKENS).
    # Layer-level traced decode per step (GDN / attention layer): w4 1.078 -> 0.985 / 0.968 -> 0.877 ms, w8 1.987 -> 1.430 /
    # 1.906 -> 1.282 ms; 48-layer traced step w4 50.8 -> 46.6 ms, w8 79.8 -> 67.8 ms; bit-reproducible at widths 2-8.
    # perf pass turn 2 adopted 8; 2026-10-01 traced per-layer probe at width 16: pairs 1.843 / 2.078 ms (attn / GDN layer
    # step) vs masked 2.212 / 2.288 -> 16; at width 32 the 256 pair groups lose (3.17 / 3.59 vs 2.60 / 2.73): masked stays
    pairs_max_tokens: int = int(os.environ.get("QWEN35_MOE_PAIRS_MAX_TOKENS", "32"))  # Coder-Next: pairs to 32 rows
    # Qwen3-Coder-Next (512 experts, top-10, Nl = 128 per die): below this many decode rows the 64-core gather path beats
    # the pairs path's one core per (token, expert) group (width 1: 25.9 vs 27.4 ms/step, 48 layers)
    pairs_min_tokens: int = int(
        os.environ.get("QWEN35_MOE_PAIRS_MIN_TOKENS", "2")
    )  # with the topk route: 24.36 -> 22.72 ms
    # routed-expert prefill: "masked" (per-group expert union, stage 1/2) or "sorted" (expert-sorted dispatch: every 32-row
    # group is one expert; see tt/moe_dispatch.py); geometry of the two compact sparse matmuls
    prefill_mode: str = "sorted"
    sorted_min_tokens: int = int(
        os.environ.get("QWEN35_SORTED_MIN_TOKENS", "512")
    )  # below this many tokens the masked path took over (per-expert padding dominated with the multicast factory;
    # re-evaluated in the post-stage-12 perf pass with the group-parallel factory: QWEN35_SORTED_MIN_TOKENS sweeps it)
    sorted_group_rows: int = 32  # rows per expert block (multiple of 32): fewer weight re-reads vs more padding
    # kernel dispatch + grouped sparse matmul: mark the trailing padding groups with a sentinel expert id and skip them
    # in the expert matmuls (no weight / activation reads, no output writes); ~16 % of the groups at 2048 tokens
    skip_pad_groups: bool = os.environ.get("QWEN35_SKIP_PAD_GROUPS", "1") == "1"
    sorted_max_tokens: int = (
        4096  # largest prefill chunk routed to the sorted path (index math exact to 7196; checked to 4096)
    )
    sorted_gate_up_cores: int = 64
    sorted_gate_up_in0: int = 32
    sorted_down_cores: int = 64
    sorted_down_in0: int = 32
    # stage 12: group-parallel sparse matmul (MatmulMultiCoreReuseProgramConfig + `indices`): the expert groups are split
    # across the whole core grid (each core reads its groups' rows and the whole [K, N] expert block) instead of every
    # core walking every group with only the small per-device N split. in0_block_w per projection (K tiles per step).
    sorted_grouped: bool = os.environ.get("QWEN35_SORTED_GROUPED", "1") == "1"
    grouped_gate_up_in0: int = 4
    grouped_down_in0: int = 4
    grouped_subblock_w: int = 8
    # --- decode matmul program configs (round 2, from the stage-2 profile) ---
    # router gate matmul [32,H]x[H,256]: "ttnn" (auto), "derived" (TTMoEGate 2D-mcast derivation), "1d8" (1D, 8 cores)
    router_decode_pc: str = "derived"
    # shared expert packed [H -> 2S+1] and down [S -> H] decode matmuls: 1D small-grid config on N cores (None = ttnn auto)
    shared_decode_cores: int | None = 44
    # attention qkv/o and GDN mega/out decode projections: "default" (ttnn auto, interleaved weights),
    # "1d" (1D mcast_in0 small grid, interleaved weights), "dram_sharded" (DRAM width-sharded weights +
    # DRAM-sharded kernel, activation width-sharded in L1 on the matching grid)
    dense_decode_mode: str = "dram_sharded"
    dense_decode_cores_in: int = 64  # 1d: cores for the wide in-projections (qkv / mega)
    dense_decode_cores_out: int = 96  # 1d: cores for the out-projections (N = hidden)
    dense_dram_readers: int = 1  # dram_sharded: num_workers_per_dram_bank (1..3 on Blackhole)
    dense_decode_in0: int | None = None  # 1d: in0_block_w override (None = helper default 8; must divide K tiles)
    dense_decode_mode_attn: str | None = None  # per-kind override of dense_decode_mode (attention layers)
    dense_decode_mode_gdn: str | None = None  # per-kind override of dense_decode_mode (GDN layers)
    # roles placed DRAM-sharded in dram_sharded mode; the others keep the 1d config. The GDN mega in-projection stays 1d: its
    # bank shard (81 tiles) needs 3 readers to fit L1 and then loses to the 1d config (sweep dense_ds2).
    dense_ds_roles: tuple = ("qkv", "o")
    dense_ds_cores: int | None = (
        None  # DRAM-sharded activation/compute grid (None = largest common divisor grid, 32 here)
    )
    # GDN mega in-projection split: the qkv columns [3072 x 12288] (48 tiles per bank) run DRAM-sharded, the a|b|z columns
    # keep the 1D config, outputs concatenated (the whole 81-tile bank shard does not fit L1 with < 3 readers)
    mega_split_ds: bool = False
    mega_split_readers: int = 1
    # GDN recurrent state kept in fp32 with an fp32 decode step (bf16 default = qwen36 single-device behaviour)
    gdn_state_fp32: bool = False
    # --- layouts / composite ops ---
    decode_norm_sharded: bool = True
    # attention layers: hand the sharded norm output straight to the DRAM-sharded qkv matmul (its activation grid adopts the
    # norm's shard grid when legal) instead of sharded->interleaved->sharded
    attn_norm_out_sharded: bool = True
    sdpa_decode_q_chunk: int | None = 32
    sdpa_decode_k_chunk: int | None = 256
    fusion: FusionFlags = FusionFlags()

    def with_(self, **kw):
        return replace(self, **kw)

    def moe_precision(self) -> MoEPrecision:
        return MoEPrecision(
            routed_gate_up=self.routed_gate_up_dtype,
            routed_down=self.routed_down_dtype,
            shared=self.shared_dtype,
            router=self.router_dtype,
            routed_fidelity=self.routed_fidelity,
            shared_fidelity=self.shared_fidelity,
            router_fidelity=self.router_fidelity,
            act_dtype=self.act_dtype,
        )

    def describe(self) -> dict:
        d = {}
        for k, v in self.__dict__.items():
            d[k] = (
                str(v).replace("DataType.", "").replace("MathFidelity.", "")
                if not isinstance(v, (int, bool, type(None), FusionFlags))
                else (v.__dict__ if isinstance(v, FusionFlags) else v)
            )
        return d


# The selected policy (filled in by the stage-3 sweeps; see doc/optimized_decoder/README.md).
SELECTED = OptimizationConfig()


def _is_single_tile_rows(x) -> bool:
    """True for the batch-1 decode activation (one 32-row tile of tokens, any leading dims == 1)."""
    phys = list(x.padded_shape)
    return phys[-2] == 32 and math.prod(phys[:-2]) == 1


def _fp32_acc(ckc) -> bool:
    return bool(getattr(ckc, "fp32_dest_acc_en", False))


def dense_weight_memcfgs(mode: str, roles: dict, ds_roles=("qkv", "mega", "o")) -> dict | None:
    """Per-role weight memory configs for the dense projections; ``roles`` = {role: (K, N)}."""
    if mode != "dram_sharded":
        return None
    return {role: tpc.create_dram_sharded_mem_config(k, n) for role, (k, n) in roles.items() if role in ds_roles}


class DenseDecodeMatmul:
    """Decode-matmul hook (x, weight, role, ckc, mc) -> out | None for the T==1 dense projections of the token
    mixers (attention "qkv"/"o", GDN "mega"/"o"). Returns None (-> plain ``ttnn.linear``) for anything that is not
    the batch-1 single-tile decode activation, so batched decode and prefill keep the default path."""

    def __init__(self, opt: OptimizationConfig, grid_w: int, mode: str | None = None):
        self.opt = opt
        self.grid_w = grid_w
        self.mode = mode or opt.dense_decode_mode
        self._pc = {}

    def _cores(self, role):
        return self.opt.dense_decode_cores_out if role == "o" else self.opt.dense_decode_cores_in

    def _ds_configs(self, role, K, N, readers):
        """(program config, activation shard config) for the DRAM-sharded decode kernel on a [K, N] width-sharded weight."""
        k_tiles = K // 32
        n_tiles = tpc._roundup(N, 32 * tpc.DRAM_CORES) // 32
        g = math.gcd(k_tiles, n_tiles)
        legal = [c for c in range(1, 65) if g % c == 0 and any(c % r == 0 and c // r <= 8 for r in range(1, 9))]
        num_cores = self.opt.dense_ds_cores if self.opt.dense_ds_cores in legal else max(legal)
        rows = next(r for r in range(8, 0, -1) if num_cores % r == 0 and num_cores // r <= 8)
        cols = num_cores // rows
        pc = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
            in0_block_w=tpc._find_largest_divisor(k_tiles // num_cores),
            per_core_M=1,
            per_core_N=n_tiles // num_cores,
            fused_activation=None,
        )
        tiles_per_bank = n_tiles // tpc.DRAM_CORES
        if readers > 1 and tiles_per_bank % readers == 0:  # legal only when the bank shard splits evenly
            pc.num_workers_per_dram_bank = readers
        elif readers > 1:
            logger.warning(f"dram_sharded {role}: {tiles_per_bank} tiles/bank not divisible by {readers} readers -> 1")
        act_cfg = ttnn.create_sharded_memory_config(
            shape=(32, K // num_cores),
            core_grid=ttnn.CoreGrid(x=cols, y=rows),
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        return pc, act_cfg

    def _ds_matmul(self, x, w_sharded, pc, act_cfg, ckc, out_mc):
        same = x.is_sharded() and x.memory_config() == act_cfg
        x_sh = x if same else ttnn.to_memory_config(x, act_cfg)
        out = ttnn.linear(
            x_sh,
            w_sharded,
            compute_kernel_config=ckc,
            program_config=pc,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
        )
        if not same:
            ttnn.deallocate(x_sh)
        out_il = ttnn.to_memory_config(out, out_mc)
        ttnn.deallocate(out)
        return out_il

    def _mega_split(self, x, w, ckc, out_mc):
        """GDN mega in-projection as DS(qkv columns) + 1D(a|b|z columns), concatenated along the output dim."""
        K, N = int(w.shape[-2]), int(w.shape[-1])
        qkv_n = self.mega_qkv_n
        key = ("mega_split", id(w))
        if key not in self._pc:
            w_qkv = ttnn.slice(w, (0, 0), (K, qkv_n))
            w_qkv_ds = ttnn.to_memory_config(w_qkv, tpc.create_dram_sharded_mem_config(K, qkv_n))
            ttnn.deallocate(w_qkv)
            w_rest = ttnn.slice(w, (0, qkv_n), (K, N))  # interleaved
            pc_ds, act_cfg = self._ds_configs("mega_qkv", K, qkv_n, self.opt.mega_split_readers)
            pc_1d = tpc.create_matmul_1d_decode_progcfg(
                32, K, N - qkv_n, self.opt.dense_decode_cores_in, fp32_acc=_fp32_acc(ckc), grid_w=self.grid_w
            )
            self._pc[key] = (w_qkv_ds, w_rest, pc_ds, act_cfg, pc_1d)
        w_qkv_ds, w_rest, pc_ds, act_cfg, pc_1d = self._pc[key]
        out_qkv = self._ds_matmul(x, w_qkv_ds, pc_ds, act_cfg, ckc, out_mc)
        x_in = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG) if x.is_sharded() else x
        out_rest = ttnn.linear(x_in, w_rest, compute_kernel_config=ckc, program_config=pc_1d, memory_config=out_mc)
        if x_in is not x:
            ttnn.deallocate(x_in)
        out = ttnn.concat([out_qkv, out_rest], dim=-1, memory_config=out_mc)
        ttnn.deallocate(out_qkv)
        ttnn.deallocate(out_rest)
        return out

    def __call__(self, x, w, role, ckc, mc):
        K, N = int(w.shape[-2]), int(w.shape[-1])
        out_mc = mc if mc is not None else ttnn.L1_MEMORY_CONFIG
        mode = self.mode
        if role == "mega" and self.opt.mega_split_ds and _is_single_tile_rows(x) and not w.is_sharded():
            return self._mega_split(x, w, ckc, out_mc)
        if mode == "dram_sharded" and not w.is_sharded():
            mode = "1d"  # role kept interleaved (dense_ds_roles) -> 1D config
        if not _is_single_tile_rows(x):
            if mode == "dram_sharded" and w.is_sharded():
                # Prefill / batched rows on a DRAM width-sharded weight: the auto-selected matmul rejects sharded operands
                # and the qwen36 2D prefill config overflows L1 at the single-chip widths, so prefill runs on an
                # interleaved copy of the weight (made once on device). Measurement path: it doubles the weight footprint
                # of these roles, so a production selection must fit a prefill program config instead.
                key = ("il", id(w))
                if key not in self._pc:
                    self._pc[key] = ttnn.to_memory_config(w, ttnn.DRAM_MEMORY_CONFIG)
                return ttnn.linear(x, self._pc[key], compute_kernel_config=ckc, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            return None
        if mode == "1d":
            key = ("1d", role, K, N, _fp32_acc(ckc))
            if key not in self._pc:
                pc = tpc.create_matmul_1d_decode_progcfg(
                    32, K, N, self._cores(role), fp32_acc=_fp32_acc(ckc), grid_w=self.grid_w
                )
                if self.opt.dense_decode_in0:
                    assert (K // 32) % self.opt.dense_decode_in0 == 0, (K, self.opt.dense_decode_in0)
                    pc.in0_block_w = self.opt.dense_decode_in0
                self._pc[key] = pc
            x_in = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG) if x.is_sharded() else x
            out = ttnn.linear(x_in, w, compute_kernel_config=ckc, program_config=self._pc[key], memory_config=out_mc)
            if x_in is not x:
                ttnn.deallocate(x_in)
            return out
        if mode == "dram_sharded":
            if x.is_sharded():
                # adopt the producer's width-shard grid when it divides K and N evenly (no reshard)
                spec = x.memory_config().shard_spec
                cores = spec.grid.num_cores()
                k_tiles = K // 32
                n_tiles = tpc._roundup(N, 32 * tpc.DRAM_CORES) // 32
                if spec.shape[1] * cores == K and k_tiles % cores == 0 and n_tiles % cores == 0:
                    key = ("ds_adopt", role, K, N, cores)
                    if key not in self._pc:
                        self._pc[key] = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                            in0_block_w=tpc._find_largest_divisor(k_tiles // cores),
                            per_core_M=1,
                            per_core_N=n_tiles // cores,
                            fused_activation=None,
                        )
                    return self._ds_matmul(x, w, self._pc[key], x.memory_config(), ckc, out_mc)
            key = ("ds", role, K, N)
            if key not in self._pc:
                self._pc[key] = self._ds_configs(role, K, N, self.opt.dense_dram_readers)
            pc, act_cfg = self._pc[key]
            return self._ds_matmul(x, w, pc, act_cfg, ckc, out_mc)
        return None


class Qwen35MoEOptimized(Qwen35MoEFused):
    """Fused MoE block with config-driven precision and sparse-matmul geometry."""

    def __init__(self, mesh_device, args, state_dict, tensor_cache_path=None, tt_ccl=None, layer_num=None):
        opt: OptimizationConfig = getattr(args, "opt", SELECTED)
        self.opt = opt
        args.moe_precision = opt.moe_precision()
        args.gather_gate_up_output_tile = opt.gather_gate_up_output_tile  # read by the base loader (weight tile)
        self.PREFILL_CHUNK = opt.masked_chunk
        self.GATHER_MAX_TOKENS = int(
            os.environ.get("QWEN35_MOE_GATHER_MAX_TOKENS", opt.gather_max_tokens)
        )  # probe knob
        super().__init__(mesh_device, args, state_dict, tensor_cache_path, tt_ccl=tt_ccl, layer_num=layer_num)
        for t in range(1, int(opt.pairs_max_tokens) + 1):
            self._pairs_mask(t * self.top_k, self.num_experts)
        H, Nl = self.hidden, self.inter_local
        tile_gu = opt.gather_gate_up_output_tile
        if tile_gu:
            self.tile_gu = ttnn.Tile(list(tile_gu))
        self.cfg_gate_up_gather = sparse_matmul_program_config(
            1,
            2 * Nl,
            H,
            target_cores=opt.gather_gate_up_cores,
            in0_block_w_cap=opt.gather_gate_up_in0,
            tile_w=(tile_gu[1] if tile_gu else 32),
        )
        self.cfg_down_gather = sparse_matmul_program_config(
            1, H, Nl, target_cores=opt.gather_down_cores, in0_block_w_cap=opt.gather_down_in0
        )
        self.cfg_gate_up_masked = sparse_matmul_program_config(
            opt.masked_chunk,
            2 * Nl,
            H,
            target_cores=opt.masked_gate_up_cores,
            in0_block_w_cap=opt.masked_gate_up_in0,
            tile_w=(tile_gu[1] if tile_gu else 32),
        )
        self.cfg_down_masked = sparse_matmul_program_config(
            opt.masked_chunk, H, Nl, target_cores=opt.masked_down_cores, in0_block_w_cap=opt.masked_down_in0
        )
        grid_w = mesh_device.compute_with_storage_grid_size().x
        # router gate matmul program config (decode rows only; prefill keeps ttnn's per-shape choice)
        self._gate_pc_decode = None
        if self.flags.router_gate_op and opt.router_decode_pc != "ttnn":
            if opt.router_decode_pc == "derived":
                self._gate_pc_decode = self.gate._matmul_program_config(self.gate._derive_program_config(), True)
            elif opt.router_decode_pc == "1d8":
                self._gate_pc_decode = tpc.create_matmul_1d_decode_progcfg(
                    32,
                    H,
                    self.gate._padded_experts,
                    8,
                    fp32_acc=_fp32_acc(self.gate.compute_kernel_config),
                    grid_w=grid_w,
                )
            else:
                raise ValueError(f"unknown router_decode_pc {opt.router_decode_pc}")
        # shared expert decode matmuls: 1D small-grid configs
        self._shared_pc = None
        if self.flags.shared_expert_packed and opt.shared_decode_cores:
            Sl = self.shared_inter // self.num_devices
            fp32 = _fp32_acc(self.shared_ckc)
            self._shared_pc = (
                tpc.create_matmul_1d_decode_progcfg(
                    32, H, self.shared_packed_width, opt.shared_decode_cores, fp32_acc=fp32, grid_w=grid_w
                ),
                tpc.create_matmul_1d_decode_progcfg(32, Sl, H, opt.shared_decode_cores, fp32_acc=fp32, grid_w=grid_w),
            )

    # ---- routed experts, pairs dispatch (decode, few tokens) ---------------------------------------------------
    def _decode_sorted(self, T):
        """QWEN35_DECODE_SORTED_MIN_ROWS = N (default 16; 0 = off): decode steps of N or more rows take the expert-sorted
        path (dispatch kernel + grouped resident sparse matmuls + gather-add un-dispatch) instead of the pairs / masked
        paths. Decode-width profile 2026-10-01: at 16 users the MoE is ~65 % of the layer step and reads one expert
        block per (user, expert) PAIR (128 reads) through the pairs path's relayout glue; sorting reads each distinct
        expert once. Measured 2026-10-01 with the multi-core gather kernel + the row-major down projection: per-layer step
        at 16 rows 1.99 / 1.87 ms (GDN / attn) vs 2.13 / 1.90 pairs-masked, at 32 rows 2.38 / 2.16 vs 2.72 / 2.60; at 8
        rows the sorted path still loses (1.74 vs 1.32), hence the threshold."""
        if getattr(self, "mtp_rows_mode", False):
            # the MTP verify pass / head rows run on a 32-row block with a few real rows: their path is chosen by
            # QWEN35_MTP_MOE_ROWS (pairs), not by the row count (served c=1 TPOT 18.6 -> 26.5 ms when this rule took
            # the padded block to the sorted path, 2026-10-01 21:14)
            return False
        # Qwen3-Coder-Next (512 experts, top-10), SERVED A/B on 4 dies (real routing, TPOT at 8 / 16 / 32 users): sorted
        # from 16 rows 47 / 97 / 109 ms, pairs to 32 rows 47 / 72 / 115 ms, sorted from 32 rows 47 / 72 / 109 ms. The
        # synthetic-routing layer test favoured pairs at 32 too (uniform ids hide the sorted path's one-read-per-expert
        # win); the served numbers decide: pairs through 16 rows, sorted at 32 (16 on the 122B port).
        n = int(os.environ.get("QWEN35_DECODE_SORTED_MIN_ROWS", "32") or 0)
        return 0 < n <= T and self.opt.prefill_mode == "sorted"

    def _routed_gather(self, x4, scores, ids, T):
        if self._decode_sorted(T):
            return self._routed_sorted(x4, scores, ids, T)
        if int(self.opt.pairs_min_tokens) <= T <= int(self.opt.pairs_max_tokens) and T > 0:
            if getattr(self, "rows_sorted", False) and self.opt.prefill_mode == "sorted":
                # MTP rows mode: the expert-sorted path (dispatch kernel, two grouped sparse matmuls with the padding
                # groups skipped, gather-add un-dispatch) instead of the pairs path's ~18 relayout ops
                return self._routed_sorted(x4, scores, ids, T)
            return self._routed_pairs(x4, scores, ids, T)
        return super()._routed_gather(x4, scores, ids, T)

    def _routed_pairs(self, x4, scores, ids, T):
        """All (token, expert) pairs of a decode step as G = T*k groups of one tile-padded row each: the group-parallel
        sparse matmul (whole groups per core over the grid) does gate/up and down in ONE call each; the routing weights
        and the sum over the k experts of every token follow as a batch-dim multiply + reduction. Replaces the per-token
        gather loop (widths 2-4) and the serial masked path (width 8) with ~12 ops per layer."""
        k, Nl, H, E = self.top_k, self.inter_local, self.hidden, self.num_experts
        G = T * k
        dram = ttnn.DRAM_MEMORY_CONFIG
        RM = ttnn.ROW_MAJOR_LAYOUT
        cfg_gu, cfg_dn = self._grouped_configs(TILE, H, Nl)
        # activation rows -> one padded tile per pair (row of token t repeated k times), ids -> one uint16 stick
        if T == 1 and PAIRS_T1_TILE_REPEAT and x4.layout == ttnn.TILE_LAYOUT and tuple(x4.shape) == (1, 1, 1, H):
            # one token: repeat its (tile-padded) row block k times on the batch dim, no row-major round trip
            x_pairs = ttnn.repeat(x4, ttnn.Shape([1, k, 1, 1]), memory_config=dram)  # [1,G,1,H] TILE
        else:
            x_rm = ttnn.to_layout(x4, RM, memory_config=dram)  # [1,1,T,H]
            x_rows = ttnn.reshape(x_rm, (1, T, 1, H))
            x_rep = ttnn.repeat(x_rows, ttnn.Shape([1, 1, k, 1]), memory_config=dram)  # [1,T,k,H]: row t repeated k
            ttnn.deallocate(x_rm)
            x_pairs = ttnn.to_layout(ttnn.reshape(x_rep, (1, G, 1, H)), ttnn.TILE_LAYOUT, memory_config=dram)
            ttnn.deallocate(x_rep)
        # ids [1,1,T,k] -> one uint16 stick [1,G]: the data-moving reshape takes uint32 (not uint16), the final
        # [1,1,1,G] -> [1,G] reshape is a view
        ids_u32 = ttnn.typecast(ids, ttnn.uint32) if ids.dtype != ttnn.uint32 else ids
        ids_rm = ttnn.to_layout(ids_u32, RM, memory_config=dram) if ids_u32.layout != RM else ids_u32
        ids_row = ttnn.reshape(ids_rm, (1, 1, 1, G))
        ids_g = ttnn.reshape(ttnn.typecast(ids_row, ttnn.uint16), (1, G))
        tmp_ids = [t for t in (ids_u32, ids_rm, ids_row) if t is not ids]
        mask = self._pairs_mask(G, E)
        fused = ttnn.sparse_matmul(
            x_pairs,
            self.up_gate,
            sparsity=mask,
            indices=ids_g,
            is_input_b_sparse=True,
            memory_config=dram,
            program_config=cfg_gu,
            compute_kernel_config=self.routed_ckc,
            dtype=self.prec.act_dtype,
        )  # compact [1,G,1,2Nl]
        ttnn.deallocate(x_pairs)
        din = self._swiglu_clamped(fused, dram)  # [1,G,1,Nl]
        ttnn.deallocate(fused)
        down = ttnn.sparse_matmul(
            din,
            self.down,
            sparsity=mask,
            indices=ids_g,
            is_input_a_sparse=False,
            is_input_b_sparse=True,
            memory_config=dram,
            program_config=cfg_dn,
            compute_kernel_config=self.routed_ckc,
            dtype=self.prec.act_dtype,
        )  # compact [1,G,1,H]
        ttnn.deallocate(din)
        ttnn.deallocate(ids_g)
        seen = set()
        for t in tmp_ids:
            if id(t) not in seen:
                seen.add(id(t))
                ttnn.deallocate(t)
        # routing weights: [1,1,T,k] -> one value per pair batch [1,G,1,1]
        sc_rm = ttnn.to_layout(scores, RM, memory_config=dram)
        sc_g = ttnn.to_layout(ttnn.reshape(sc_rm, (1, G, 1, 1)), ttnn.TILE_LAYOUT, memory_config=dram)
        ttnn.deallocate(sc_rm)
        weighted = ttnn.multiply(down, sc_g, memory_config=dram)  # [1,G,1,H]
        ttnn.deallocate(down)
        ttnn.deallocate(sc_g)
        # swiglu reports the tile-padded row count (32) as its logical rows; the padded rows are zeros through the
        # chain, so keep them in the batch-dim split and trim one logical row after the k-sum
        rows = weighted.shape[-2]
        per_tok = ttnn.reshape(weighted, (T, k, rows, H))  # batch-dim split, no data movement
        acc = ttnn.sum(per_tok, dim=1, keepdim=True, compute_kernel_config=self.routed_ckc)  # [T,1,1,H]
        ttnn.deallocate(weighted)
        if acc.shape[-2] != 1:  # the NC reduction keeps the tile-padded row count: keep the one logical row
            acc_t = ttnn.slice(acc, [0, 0, 0, 0], [T, 1, 1, H], memory_config=dram)
            ttnn.deallocate(acc)
            acc = acc_t
        if T == 1:
            return acc  # [1,1,1,H] already
        # T tokens in T separate padded tiles -> one [1,1,T,H] tile block (row-major round trip)
        acc_rm = ttnn.to_layout(acc, RM, memory_config=dram)
        ttnn.deallocate(acc)
        out = ttnn.to_layout(ttnn.reshape(acc_rm, (1, 1, T, H)), ttnn.TILE_LAYOUT, memory_config=dram)
        ttnn.deallocate(acc_rm)
        return out

    def _pairs_mask(self, G, E):
        """The op still validates a sparsity operand of batch length G x E (unused in indexed mode): one device tensor
        per pair count, shared by all layers of a build (cached on the model args) and built at construction for every
        pairs width, so the decode path never uploads from the host."""
        cache = self.args.__dict__.setdefault("_pairs_masks", {})
        if (G, E) not in cache:
            cache[(G, E)] = ttnn.from_torch(
                torch.ones(1, G, 1, E, dtype=torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                **({"mesh_mapper": ttnn.ReplicateTensorToMesh(self.device)} if self.num_devices > 1 else {}),
            )
        return cache[(G, E)]

    # ---- routed experts, sorted dispatch (prefill) ----------------------------------------------------------
    def _routed_masked(self, x4, scores, ids, T, chunk):
        if chunk == TILE and T <= 32 and self._decode_sorted(T):
            return self._routed_sorted(
                x4, scores, ids, T
            )  # decode step wider than the gather limit (see _decode_sorted)
        if chunk == TILE and 0 < T <= int(self.opt.pairs_max_tokens):
            if getattr(self, "rows_sorted", False) and self.opt.prefill_mode == "sorted":
                return self._routed_sorted(x4, scores, ids, T)  # MTP rows mode (see _routed_gather)
            return self._routed_pairs(x4, scores, ids, T)  # decode step wider than the gather limit
        if self.opt.prefill_mode != "sorted" or not (self.opt.sorted_min_tokens <= T <= self.opt.sorted_max_tokens):
            return super()._routed_masked(x4, scores, ids, T, chunk)
        return self._routed_sorted(x4, scores, ids, T)

    def _routed_sorted(self, x4, scores, ids, T):
        E, Nl, H, k = self.num_experts, self.inter_local, self.hidden, self.top_k
        opt, act = self.opt, self.prec.act_dtype
        gs = opt.sorted_group_rows
        self._ensure_dispatch()
        sub = sorted_rows_cap()
        if sub and T > sub:
            return self._routed_sorted_split(x4, scores, ids, T, sub)
        m = self._dispatch.build(ids, scores, T)
        return self._routed_sorted_built(x4, scores, ids, T, m)

    def _routed_sorted_split(self, x4, scores, ids, T, sub):
        """The routed experts of a T-row chunk in row blocks of `sub` rows (stage-6 memory): the dispatched buffers are
        sized for the static worst case, Rd = roundup(T*k, 32) + 31*E + 32 rows of H bf16 (T=2048: 22368 rows, 183 MB
        per die, live twice at the un-dispatch: tiled down output + its row-major copy), which OOMed the full model's
        2048-row prefill. Every routed row is computed independently (per-row matmuls on fixed program configs, the
        same k-order gather-add), so the split output is bit-identical (tools/memory_probe.py checks it)."""
        H, k = self.hidden, self.top_k
        dram = ttnn.DRAM_MEMORY_CONFIG
        ids32 = ttnn.typecast(ids, ttnn.uint32) if ids.dtype != ttnn.uint32 else ids  # exact; the kernel takes uint32
        parts = []
        for r0 in range(0, T, sub):
            r1 = min(T, r0 + sub)
            n = r1 - r0
            xs = ttnn.slice(x4, (0, 0, r0, 0), (1, 1, r1, H), memory_config=dram)
            ss = ttnn.slice(scores, (0, 0, r0, 0), (1, 1, r1, k), memory_config=dram)
            si = ttnn.slice(ids32, (0, 0, r0, 0), (1, 1, r1, k), memory_config=dram)
            m = self._dispatch.build(si, ss, n)
            parts.append(self._routed_sorted_built(xs, ss, si, n, m))
            for t in (xs, ss, si):
                ttnn.deallocate(t)
        if ids32 is not ids:
            ttnn.deallocate(ids32)
        y = ttnn.concat(parts, dim=2, memory_config=dram)
        for t in parts:
            ttnn.deallocate(t)
        return y

    def prepare_dispatch(self, lengths):
        """Create the sorted-dispatch constants of every row count (prefill chunk lengths, decode batch) at setup."""
        self._ensure_dispatch()
        sub = sorted_rows_cap()
        rows = set(int(t) for t in lengths)
        for t in list(rows):
            if sub and t > sub:  # the row blocks of a split chunk (_routed_sorted_split)
                rows |= {sub} | ({t % sub} if t % sub else set())
        self._dispatch.prepare(sorted(rows))

    def _ensure_dispatch(self):
        E, Nl, H, k = self.num_experts, self.inter_local, self.hidden, self.top_k
        opt = self.opt
        gs = opt.sorted_group_rows
        if getattr(self, "_dispatch", None) is None:
            skip = self._skip_pad_groups_rule()
            self._dispatch = SortedDispatch(self.device, E, k, group=gs, pad_group_id=0xFFFF if skip else 0)
            self._skip_pad_groups = skip
            self._cfg_sorted_gu = sparse_matmul_program_config(
                gs,
                2 * Nl,
                H,
                target_cores=opt.sorted_gate_up_cores,
                in0_block_w_cap=opt.sorted_gate_up_in0,
                tile_w=self.tile_gu.tile_shape[1],
            )
            self._cfg_sorted_dn = sparse_matmul_program_config(
                gs, H, Nl, target_cores=opt.sorted_down_cores, in0_block_w_cap=opt.sorted_down_in0
            )

    def _routed_sorted_built(self, x4, scores, ids, T, m):
        E, Nl, H, k = self.num_experts, self.inter_local, self.hidden, self.top_k
        opt, act = self.opt, self.prec.act_dtype
        gs = opt.sorted_group_rows
        Rd, G = m["Rd"], m["G"]
        grouped = bool(getattr(opt, "sorted_grouped", False))
        # the kernel gather needs the skip-aware grouped matmul (padding groups' tiles are not written)
        x_disp = gather_rows(
            x4,
            m["src_tok"],
            Rd,
            gs,
            group_ids=m["group_ids"] if (grouped and self._skip_pad_groups_rule()) else None,
            num_experts=E,
        )  # [1,G,gs,H]
        ttnn.deallocate(m["src_tok"])
        narrow = self.tile_gu.tile_shape[1] != 32
        if grouped:
            # indexed/gather mode: the id list drives the weight reads, `sparsity` is unused (kept for the op signature)
            sparse_kw = {"indices": m["group_ids"]}
            cfg_gu, cfg_dn = self._grouped_configs(gs, H, Nl)
            if getattr(self, "_cfg_grouped_down_untilize", False):
                cfg_dn = self._cfg_grouped_dn_rm  # row-major down projection straight into the gather-add un-dispatch
        else:
            sparse_kw = {"nnz": G}
            cfg_gu, cfg_dn = self._cfg_sorted_gu, self._cfg_sorted_dn
        out_gu = None if narrow else self._alloc_dev((1, G, gs, 2 * Nl), act)
        fused = ttnn.sparse_matmul(
            x_disp,
            self.up_gate,
            sparsity=m["group_mask"],
            **sparse_kw,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=cfg_gu,
            **({"output_tile": self.tile_gu} if narrow else {"optional_output_tensor": out_gu}),
            compute_kernel_config=self.routed_ckc,
            dtype=act,
        )  # compact [1,G,32,2Nl]
        ttnn.deallocate(x_disp)
        din = self._swiglu_clamped(fused, ttnn.DRAM_MEMORY_CONFIG)  # [1,G,32,Nl]
        ttnn.deallocate(fused)
        # routing weight on the (narrower) expert intermediate; the down projection is linear
        din = ttnn.multiply(ttnn.reshape(din, (1, 1, Rd, Nl)), m["weights"], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(m["weights"])
        din = ttnn.reshape(din, (1, G, gs, Nl))
        out_dn = self._alloc_dev(
            (1, G, gs, H),
            act,
            layout=(
                ttnn.ROW_MAJOR_LAYOUT
                if (grouped and getattr(self, "_cfg_grouped_down_untilize", False))
                else ttnn.TILE_LAYOUT
            ),
        )
        down = ttnn.sparse_matmul(
            din,
            self.down,
            sparsity=m["group_mask"],
            **sparse_kw,
            optional_output_tensor=out_dn,
            is_input_a_sparse=False,
            is_input_b_sparse=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=cfg_dn,
            compute_kernel_config=self.routed_ckc,
            dtype=act,
        )  # compact [1,G,32,H]
        ttnn.deallocate(din)
        ttnn.deallocate(m["group_mask"])
        ttnn.deallocate(m["group_ids"])
        y = undispatch_rows(down, None, m["undispatch"], T, k, H, undispatch_rm=m.get("undispatch_rm"))
        ttnn.deallocate(down)
        for key in ("undispatch", "undispatch_rm"):
            if m.get(key) is not None:
                ttnn.deallocate(m[key])
        return y

    def _skip_pad_groups_rule(self):
        """Whether the sorted dispatch marks its padding groups with the sentinel id 0xFFFF (>= E) and the grouped sparse
        matmul skips them. One rule for the dispatch AND the program configs: the MTP head's block built its grouped
        configs before its dispatch existed and cached skip_padding_groups=False against a sentinel-padded id list
        (the resident-in1 path then deadlocked on those groups, served warm-up hang 2026-10-01)."""
        from .moe_dispatch import DISPATCH_MODE as _dmode

        opt = self.opt
        return (
            bool(getattr(opt, "skip_pad_groups", False))
            and bool(getattr(opt, "sorted_grouped", False))
            and _dmode == "kernel"
        )

    def _grouped_configs(self, gs, H, Nl):
        """MatmulMultiCoreReuseProgramConfig pair (gate/up, down) for the group-parallel sparse matmul: whole groups per
        core (per_core_M = gs/32, per_core_N = all N tiles), K stepped in in0_block_w tiles, subblock 1 x w."""
        if getattr(self, "_cfg_grouped", None) is None:
            opt = self.opt
            grid = self.device.compute_with_storage_grid_size()

            def cfg(k, n_tiles, in0_cap, w_tile_bytes):
                k_tiles = k // 32
                ib = max(1, min(int(in0_cap), k_tiles))
                while k_tiles % ib:
                    ib -= 1
                # L1 budget of the grouped factory per core (its own 1.3 MB CB cap): double-buffered in0 [M, ib] blocks,
                # in1 either RESIDENT (the whole [K, N] expert block: weights read once per run of same-expert groups,
                # TT_SPARSE_GROUPED_RESIDENT_IN1, default on) or two streamed [ib, N] blocks, one bf16 output block and
                # the fp32 partials block. Shrink the K block until it fits (a too-large block is a program-build
                # TT_THROW; a resident block that does not fit makes the factory fall back to streaming).
                m_tiles = gs // 32
                budget = 1_300_000
                fixed = m_tiles * n_tiles * (2048 + 2048)  # bf16 output block + bf16 partials block
                resident = os.environ.get("TT_SPARSE_GROUPED_RESIDENT_IN1", "1") != "0"
                # the resident block must fit with the smallest in0 double buffer, else the factory streams in1
                resident = resident and 2 * m_tiles * 2048 + k_tiles * n_tiles * w_tile_bytes + fixed <= budget
                in1_bytes = (
                    (lambda ib_: k_tiles * n_tiles * w_tile_bytes)
                    if resident
                    else (lambda ib_: 2 * n_tiles * ib_ * w_tile_bytes)
                )
                while ib > 1 and 2 * m_tiles * ib * 2048 + in1_bytes(ib) + fixed > budget:
                    ib -= 1
                    while k_tiles % ib:
                        ib -= 1
                sw = max(1, min(int(opt.grouped_subblock_w), n_tiles))
                while n_tiles % sw:
                    sw -= 1
                if getattr(self.routed_ckc, "fp32_dest_acc_en", False):
                    while sw > 4:
                        sw -= 1
                pc = ttnn.MatmulMultiCoreReuseProgramConfig(
                    compute_with_storage_grid_size=ttnn.CoreCoord(grid.x, grid.y),
                    in0_block_w=ib,
                    out_subblock_h=1,
                    out_subblock_w=sw,
                    per_core_M=gs // 32,
                    per_core_N=n_tiles,
                    skip_padding_groups=self._skip_pad_groups_rule(),
                )
                return pc, bool(resident)  # the untilize_out path needs the resident compute kernel

            gu_tiles = -(-(2 * Nl) // self.tile_gu.tile_shape[1])
            tile_bytes = {ttnn.bfloat4_b: 576, ttnn.bfloat8_b: 1088}
            cfg_gu, _ = cfg(H, gu_tiles, opt.grouped_gate_up_in0, tile_bytes.get(self.up_gate.dtype, 2048))
            cfg_dn, dn_resident = cfg(Nl, H // 32, opt.grouped_down_in0, tile_bytes.get(self.down.dtype, 2048))
            # a SEPARATE config object for the expert-sorted path's down projection: the pairs path (decode widths
            # 2-16 and the MTP pass) shares cfg_dn and must keep TILE output (served c=1 pass 71.5 -> 86 ms when the
            # row-major flag leaked into it, 2026-10-01 21:30)
            cfg_dn_rm, _ = cfg(Nl, H // 32, opt.grouped_down_in0, tile_bytes.get(self.down.dtype, 2048))
            # QWEN35_DOWN_UNTILIZE_OUT (default on): the down projection writes ROW_MAJOR rows (the compute kernel's
            # pack-untilize path) straight into the gather-add un-dispatch - no separate untilize op (0.74 ms per MoE
            # layer at 2048 tokens, 0.25 ms at 16 decode rows; 2026-10-01). Needs the skip-aware path (padding groups'
            # rows are never written) and the bf16 activation dtype.
            if (
                os.environ.get("QWEN35_DOWN_UNTILIZE_OUT", "1") == "1"
                and hasattr(cfg_dn, "untilize_out")
                and dn_resident
                and self._skip_pad_groups_rule()
                and self.prec.act_dtype == ttnn.bfloat16
            ):
                cfg_dn_rm.untilize_out = True
            self._cfg_grouped = (cfg_gu, cfg_dn)
            self._cfg_grouped_dn_rm = cfg_dn_rm
            self._cfg_grouped_down_untilize = bool(cfg_dn_rm.untilize_out)
        return self._cfg_grouped

    def _route(self, x4):
        if self._gate_pc_decode is not None:
            self.gate.matmul_program_config = self._gate_pc_decode if _is_single_tile_rows(x4) else None
        return super()._route(x4)

    def _shared(self, x4):
        if self._shared_pc is None or not _is_single_tile_rows(x4):
            return super()._shared(x4)
        pc_packed, pc_down = self._shared_pc
        mc = ttnn.DRAM_MEMORY_CONFIG
        Sl = self.shared_inter // self.num_devices
        T = x4.shape[2]
        x_in = ttnn.to_memory_config(x4, ttnn.L1_MEMORY_CONFIG) if x4.is_sharded() else x4
        fused = ttnn.linear(
            x_in,
            self.shared_packed,
            compute_kernel_config=self.shared_ckc,
            program_config=pc_packed,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        if x_in is not x4:
            ttnn.deallocate(x_in)
        gate = ttnn.slice(fused, [0, 0, 0, 0], [1, 1, T, Sl], memory_config=mc)
        up = ttnn.slice(fused, [0, 0, 0, Sl], [1, 1, T, 2 * Sl], memory_config=mc)
        sg = (
            ttnn.slice(fused, [0, 0, 0, 2 * Sl], [1, 1, T, 2 * Sl + 1], memory_config=mc) if self.shared_gated else None
        )
        ttnn.deallocate(fused)
        gate, up = self._clamp_gate_up(gate, up, ttnn.L1_MEMORY_CONFIG)
        hidden = ttnn.multiply(
            up, gate, input_tensor_b_activations=[ttnn.UnaryOpType.SILU], memory_config=ttnn.L1_MEMORY_CONFIG
        )
        ttnn.deallocate(gate)
        ttnn.deallocate(up)
        out = ttnn.linear(
            hidden, self.shared_w2, compute_kernel_config=self.shared_ckc, program_config=pc_down, memory_config=mc
        )
        ttnn.deallocate(hidden)
        if sg is None:
            return out
        out = ttnn.multiply(out, sg, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID], memory_config=mc)
        ttnn.deallocate(sg)
        return out
