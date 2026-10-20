# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""MoE feed-forward block: router + routed experts (top-k) + shared expert.

GLM-4.5-Air semantics (HF ``Glm4MoeMoE``, ``args.router_kind == "sigmoid_bias"``, 128 experts, top-8)::

    s       = sigmoid(x @ W_gate^T)                       # fp32
    ids     = topk(s + e_score_correction_bias, 8)        # selection only (n_group = topk_group = 1)
    w       = s[ids] / sum(s[ids]) * routed_scaling_factor
    routed  = sum_i  w_i * down_i(silu(gate_i(x)) * up_i(x))
    shared  = down_s(silu(gate_s(x)) * up_s(x))           # no shared-expert gate
    out     = routed + shared

The Qwen3.5-MoE semantics this block was written for (``router_kind == "softmax"``, ``shared_expert_gated``) stay
available::

    probs   = softmax(x @ W_gate^T)            # over all 256 experts, fp32
    w, ids  = topk(probs, 8);  w = w / sum(w)   # renormalized routing weights
    shared  = sigmoid(x @ W_sg^T) * down_s(silu(gate_s(x)) * up_s(x))

Routed experts run through ``ttnn.sparse_matmul`` so only selected experts are computed:

* ``gather`` (decode, few tokens): ``indices=`` mode visits exactly the top-k expert ids of one
  token (compact output, no ``nnz`` contract, no per-slot scan).
* ``masked`` (prefill chunks, batched decode): tokens are grouped into tile-height chunks and the
  sparsity mask is the union of experts any token in the chunk selected (``nnz=None``: the count
  is data dependent and a wrong static ``nnz`` deadlocks the device).

Weights are packed ``[up | gate]`` on the output dim so ``ttnn.swiglu`` (``first * silu(second)``)
fuses the activation. Expert weights are laid out ``[1, E, H, 2N]`` / ``[1, E, N, H]``; on a TP mesh
the intermediate dim ``N`` is sharded (column-parallel up/gate, row-parallel down) and the block
returns per-device partial sums for the layer's reduce (see ``num_devices``).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, replace

import torch
from loguru import logger

import ttnn

_MOE_DEBUG = __import__("os").environ.get("GLM53_MOE_DEBUG") == "1"
_FAULT_ROUTED_SCALE = (
    float(os.environ["GLM53_FAULT_ROUTED_SCALE"]) if os.environ.get("GLM53_FAULT_ROUTED_SCALE") else None
)

TILE = 32
# functional router: top-k on the logits then softmax over the selected k (2 ops) instead of softmax -> top-k -> sum -> div
ROUTER_TOPK_FIRST = os.environ.get("QWEN35_ROUTER_TOPK_FIRST", "1") == "1"  # default since the perf pass (2026-09-29)
ROUTER_TOPK_PAD = int(os.environ.get("QWEN35_ROUTER_TOPK_PAD", "0") or 0)  # probe knob; 1024 measured SLOWER (rejected)
# Router top-k through Blackhole's large-k route: ttnn.topk with k > 64 (here 80) and stable=False takes the
# topk_large_indices composite (45 us on [32, 512]) instead of the single-core bitonic factory (77 us for k = 10 over
# 512 experts, the costliest op of the Qwen3-Coder-Next decode layer); the top `top_k` columns are sliced off.
# 0 = stock ttnn.topk. Results: same top-k set and sorted values as the stock op (bf16 ties may swap order).
ROUTER_TOPK_ROUTE_K = int(
    os.environ.get("QWEN35_ROUTER_TOPK_ROUTE_K", "80") or 0
)  # 48-layer token-out 27.39 -> 24.36 ms
# GLM router: weights of the selected experts by "gather" (ttnn.gather) or "onehot" (compare + multiply + row sum)
ROUTER_WEIGHTS = os.environ.get("QWEN35_ROUTER_WEIGHTS", "onehot")
ROUTER_L1 = (
    os.environ.get("QWEN35_ROUTER_L1", "0") == "1"
)  # L1-resident router intermediates for decode rows (experiment)


def _largest_divisor_at_most(n: int, cap: int) -> int:
    d = max(1, min(cap, n))
    while d > 1 and n % d:
        d -= 1
    return d


def sparse_matmul_program_config(
    m: int,
    n: int,
    k: int,
    *,
    target_cores: int,
    in0_block_w_cap: int,
    grid_y: int = 8,
    tile_w: int = TILE,
    dest_tiles: int = 4,
):
    """1D-mcast config for ``ttnn.sparse_matmul``: cores must divide Nt, ``Kt % in0_block_w == 0``."""
    n_tiles = max(1, math.ceil(n / tile_w))
    k_tiles = max(1, math.ceil(k / TILE))
    m_tiles = max(1, math.ceil(m / TILE))
    cores = _largest_divisor_at_most(n_tiles, max(1, target_cores))
    per_core_n = n_tiles // cores
    cy = _largest_divisor_at_most(cores, grid_y)
    cx = cores // cy
    # fp32 destination accumulation halves the DST register: an output subblock holds at most `dest_tiles` tiles
    sub_w = _largest_divisor_at_most(per_core_n, dest_tiles)
    sub_h = _largest_divisor_at_most(m_tiles, max(1, dest_tiles // sub_w))
    in0_block_w = _largest_divisor_at_most(k_tiles, max(1, in0_block_w_cap))
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(cx, cy),
        in0_block_w=in0_block_w,
        out_subblock_h=sub_h,
        out_subblock_w=sub_w,
        out_block_h=m_tiles,
        out_block_w=per_core_n,
        per_core_M=m_tiles,
        per_core_N=per_core_n,
        fuse_batch=False,
        fused_activation=None,
        mcast_in0=True,
    )


@dataclass
class MoEPrecision:
    """Datatype / fidelity policy for the MoE block (finalized by the datatype-sweep stage)."""

    routed_gate_up: object = ttnn.bfloat8_b
    routed_down: object = ttnn.bfloat8_b
    shared: object = ttnn.bfloat8_b
    router: object = ttnn.bfloat16
    routed_fidelity: object = ttnn.MathFidelity.HiFi2
    shared_fidelity: object = ttnn.MathFidelity.HiFi2
    router_fidelity: object = ttnn.MathFidelity.HiFi4
    act_dtype: object = ttnn.bfloat16

    @classmethod
    def from_args(cls, args, layer_num=None):
        """The block's policy: ``args.moe_precision`` (stage-3 config) with the datatype-sweep layer exceptions
        (``args.precision_layer_exceptions``: {layer_idx: {field: value}}, e.g. first/last layer routed experts in bfp8)
        applied for ``layer_num``."""
        p = getattr(args, "moe_precision", None)
        p = p if p is not None else cls()
        exc = {int(k): v for k, v in (getattr(args, "precision_layer_exceptions", None) or {}).items()}
        if layer_num is not None and int(layer_num) in exc:
            over = exc[int(layer_num)]
            unknown = [k for k in over if not hasattr(p, k)]
            if unknown:
                raise ValueError(f"precision_layer_exceptions[{layer_num}]: unknown MoEPrecision fields {unknown}")
            p = replace(p, **over)
        return p


class Qwen35MoE:
    """Routed + shared expert feed-forward for one decoder layer (``forward(x, mode=...)``)."""

    # Prefill chunk (tokens per sparse-matmul group). Tile height is the floor; larger chunks raise the
    # live-expert union per token. Tuned in the optimize stages.
    PREFILL_CHUNK = 32
    # Decode: tokens <= this take the per-token gather path; more tokens take one masked group.
    # QWEN35_MOE_GATHER_MAX_TOKENS overrides (stage-10 probe knob: 0 = always the masked path).
    GATHER_MAX_TOKENS = int(os.environ.get("QWEN35_MOE_GATHER_MAX_TOKENS", "4"))

    def __init__(self, mesh_device, args, state_dict, tensor_cache_path=None, tt_ccl=None, layer_num=None):
        self.device = mesh_device
        self.args = args
        self.tt_ccl = tt_ccl
        self.layer_num = layer_num
        self.num_devices = getattr(args, "num_devices", 1)
        self.hidden = args.dim
        self.num_experts = args.num_experts
        self.top_k = args.num_experts_per_tok
        self.inter = args.moe_intermediate_size
        self.shared_inter = args.shared_expert_intermediate_size
        self.prec = MoEPrecision.from_args(args, layer_num)
        tp = self.num_devices
        assert self.inter % (tp * TILE) == 0, f"moe_intermediate_size {self.inter} must split into tiles over TP={tp}"
        self.inter_local = self.inter // tp

        # tensor_cache_path: a tt/weight_cache.WeightCache (final device tensors; the state_dict values are read only on
        # a miss, so `t` below may be a thunk) or a directory for ttnn.as_tensor's host-side cache
        cache = tensor_cache_path
        wcache = cache if hasattr(cache, "get") and hasattr(cache, "store") else None
        if wcache is not None:
            cache = None

        def cache_name(name):
            return str(cache / f"mlp.{name}") if cache else None

        mapper = (lambda dim: ttnn.ShardTensorToMesh(mesh_device, dim=dim)) if tp > 1 else (lambda dim: None)

        def upload(t, dtype, name, dim=None, layout=ttnn.TILE_LAYOUT, tile=None):
            if wcache is not None and tile is None:
                return wcache.get(f"mlp.{name}", lambda: upload(t, dtype, name, dim, layout, tile=False))
            if tile is False:
                tile = None
            if callable(t):
                t = t()
            kw = {}
            mm = mapper(dim) if dim is not None else (ttnn.ReplicateTensorToMesh(mesh_device) if tp > 1 else None)
            if mm is not None:
                kw["mesh_mapper"] = mm
            if (
                tile is not None
            ):  # non-default weight tile (stage-5 probe): as_tensor has no tile= -> uncached from_torch
                return ttnn.from_torch(
                    t,
                    dtype=dtype,
                    layout=layout,
                    device=mesh_device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    tile=tile,
                    **kw,
                )
            return ttnn.as_tensor(
                t,
                dtype=dtype,
                layout=layout,
                device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                cache_file_name=cache_name(name),
                **kw,
            )

        # ---- router: W_gate [E, H] -> [H, E] ----------------------------------------------------
        self.router_kind = getattr(args, "router_kind", "softmax")
        self.routed_scaling = float(getattr(args, "routed_scaling_factor", 1.0))
        self.router_bias = None
        if self.router_kind == "sigmoid_bias":
            # The selection bias of GLM-4.5-Air sits at ~16-20 with a spread of ~0.02-0.05 (layer 2: 19.69 +- 0.023): in
            # bf16 (0.125 apart at 19.7) it would erase the selection. Selection is invariant to a constant shift, so the
            # per-layer mean is subtracted and the selection chain runs in fp32.
            def centered_bias():
                b = state_dict["gate.e_score_correction_bias"].to(torch.float64)
                return (b - b.mean()).to(torch.float32).reshape(1, 1, 1, -1)

            self.router_bias = upload(centered_bias, ttnn.float32, "gate.e_score_correction_bias.centered")
            self._expert_iota = ttnn.from_torch(  # [1,1,1,E] expert ids for the one-hot weight select
                torch.arange(self.num_experts, dtype=torch.float32).reshape(1, 1, 1, -1),
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                **({"mesh_mapper": ttnn.ReplicateTensorToMesh(mesh_device)} if tp > 1 else {}),
            )
        self.router_w = upload(lambda: state_dict["gate.weight"].T.contiguous(), self.prec.router, "gate.weight")
        # Decode router as a DRAM-sharded matmul (turn 6): the stock linear ran the [32, 3072] x [3072, 256] router on
        # 7-8 cores (one per N tile) at 71 us per layer at width 8 - the 1.5 MB bf16 weight streamed through 8 cores;
        # the DRAM-sharded decode kernel splits K over the DRAM-bank cores like every other decode projection (~10 us).
        self._router_sharded = (
            self.num_devices > 1 and os.environ.get("QWEN35_ROUTER_DRAM_SHARDED", "1") == "1"
        ) and getattr(args, "act_shard_hidden", None) is not None
        if self._router_sharded:
            from models.demos.blackhole.qwen36.tt import tp_common as tpc

            ttnn.deallocate(self.router_w)
            self.router_w = ttnn.as_tensor(
                state_dict["gate.weight"].T.contiguous(),
                dtype=self.prec.router,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=tpc.create_dram_sharded_mem_config(self.hidden, self.num_experts),
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
                cache_file_name=cache_name("gate.weight.dram_sharded"),
            )
            self.router_decode_progcfg = tpc.create_dram_sharded_matmul_program_config(
                TILE, self.hidden, self.num_experts
            )

        # ---- routed experts (HF fused: gate_up [E, 2N, H] = [gate; up] rows, down [E, H, N]) -------
        E, N, H = self.num_experts, self.inter, self.hidden
        Nl = self.inter_local

        def packed_up_gate():
            gate_up = state_dict["experts.gate_up_proj"]  # [E, 2N, H]
            assert tuple(gate_up.shape) == (E, 2 * N, H), gate_up.shape
            gate_t = gate_up[:, :N, :].transpose(1, 2)  # [E, H, N]
            up_t = gate_up[:, N:, :].transpose(1, 2)  # [E, H, N]
            # Pack per TP shard as [up_d | gate_d] so ttnn.swiglu (first * silu(second)) is exact per device.
            packed = torch.cat(
                [
                    torch.cat([up_t[:, :, d * Nl : (d + 1) * Nl], gate_t[:, :, d * Nl : (d + 1) * Nl]], dim=-1)
                    for d in range(tp)
                ],
                dim=-1,
            )  # [E, H, tp*2*Nl]
            return packed.unsqueeze(0).contiguous()

        def down_t():
            down = state_dict["experts.down_proj"]  # [E, H, N]
            assert tuple(down.shape) == (E, H, N), down.shape
            return down.transpose(1, 2).unsqueeze(0).contiguous()  # [1, E, N, H]

        # stage-5 probe: a 16-wide output tile on the gather-mode gate/up needs the weight stored with 16-wide tiles
        # (matmul: out_tile_shape[1] % in1_tile_w == 0).
        _gu_tile = getattr(args, "gather_gate_up_output_tile", None)
        self.up_gate = upload(
            packed_up_gate,
            self.prec.routed_gate_up,
            "experts.up_gate",
            dim=-1,
            tile=ttnn.Tile([32, _gu_tile[1]]) if _gu_tile else None,
        )
        # a 16-wide gate/up output feeds the down matmul as in0 with 16-wide K tiles -> the down weight needs 16-high tiles
        self.down = upload(
            down_t,
            self.prec.routed_down,
            "experts.down",
            dim=-2,
            tile=ttnn.Tile([_gu_tile[1], 32]) if _gu_tile else None,
        )

        # ---- shared expert (dense SwiGLU, TP column/row parallel) + sigmoid gate -----------------------
        # a subclass that packs the shared expert's gate / up (Qwen35MoEFused, FusionFlags.shared_expert_packed) frees
        # these right away: not built at all then
        packed_shared = getattr(getattr(self, "flags", None), "shared_expert_packed", False)
        self.shared_w1 = self.shared_w3 = self.shared_gate_w = None
        if not packed_shared:
            self.shared_w1 = upload(
                lambda: state_dict["shared_expert.gate_proj.weight"].T.contiguous(),
                self.prec.shared,
                "shared_expert.gate_proj",
                dim=-1,
            )
            self.shared_w3 = upload(
                lambda: state_dict["shared_expert.up_proj.weight"].T.contiguous(),
                self.prec.shared,
                "shared_expert.up_proj",
                dim=-1,
            )
        self.shared_w2 = upload(
            lambda: state_dict["shared_expert.down_proj.weight"].T.contiguous(),
            self.prec.shared,
            "shared_expert.down_proj",
            dim=-2,
        )
        self.shared_gated = bool(getattr(args, "shared_expert_gated", True))
        self.shared_gate_w = (
            upload(lambda: state_dict["shared_expert_gate.weight"].T.contiguous(), ttnn.bfloat16, "shared_expert_gate")
            if self.shared_gated and not packed_shared
            else None
        )  # [H, 1]

        # ---- constants ----------------------------------------------------------------------------------
        self.ones_mask = ttnn.from_torch(
            torch.ones(1, 1, 1, E, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            **({"mesh_mapper": ttnn.ReplicateTensorToMesh(mesh_device)} if tp > 1 else {}),
        )
        # GLM-5.3 SwiGLU clamp (HF swiglu_limit): gate <= L only, up in [-L, L], before the activation.
        self.swiglu_limit = getattr(args, "swiglu_limit", None)
        self._clamp_lo = None
        if self.swiglu_limit is not None:
            L = float(self.swiglu_limit)
            lo = torch.cat([torch.full((Nl,), -L), torch.full((Nl,), -3.0e38)]).reshape(1, 1, 1, 2 * Nl)
            self._clamp_lo = ttnn.from_torch(
                lo,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                **({"mesh_mapper": ttnn.ReplicateTensorToMesh(mesh_device)} if tp > 1 else {}),
            )
        self.tile = ttnn.Tile([TILE, TILE])
        self.tile_gu = self.tile  # gate/up gather output tile (Qwen35MoEOptimized may narrow it)
        self.routed_ckc = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=self.prec.routed_fidelity,
            math_approx_mode=False,
            # fp32 accumulation: a bf16 DST accumulator biases the K = 4096 expert projections UP by ~5 % (K = 2048: ~3 %)
            # on Blackhole, and the routed output by ~12-14 % after SwiGLU (tools/sparse_mm_probe.py, 2026-10-14). The
            # subblocks are capped at 4 tiles (half DST) in sparse_matmul_program_config, the condition of tt-metal #49068.
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        self.shared_ckc = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=self.prec.shared_fidelity,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        self.router_ckc = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=self.prec.router_fidelity,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        # Program configs. gather (M=1) fills the grid; masked groups size the grid from the expected union.
        union = int(E * (1.0 - (1.0 - 1.0 / E) ** (self.PREFILL_CHUNK * self.top_k)))
        self.cfg_gate_up_gather = sparse_matmul_program_config(1, 2 * Nl, H, target_cores=64, in0_block_w_cap=96)
        self.cfg_down_gather = sparse_matmul_program_config(1, H, Nl, target_cores=64, in0_block_w_cap=32)
        self.cfg_gate_up_masked = sparse_matmul_program_config(
            self.PREFILL_CHUNK, 2 * Nl, H, target_cores=max(8, min(32, union // 2)), in0_block_w_cap=64
        )
        self.cfg_down_masked = sparse_matmul_program_config(
            self.PREFILL_CHUNK, H, Nl, target_cores=max(8, min(32, union // 4)), in0_block_w_cap=32
        )
        logger.debug(f"Qwen35MoE layer {layer_num}: E={E} k={self.top_k} N={N} (local {Nl}) tp={tp} union~{union}")

    # ------------------------------------------------------------------------------------------------
    def forward(self, x, mode=None):
        """x: [B, T, H] (single device prefill/decode) or [1, 1, T, H]. Returns the same shape."""
        orig = list(x.shape)
        H = orig[-1]
        T = math.prod(orig[:-1])
        if self.num_devices > 1 and x.is_sharded():
            # TP: the ffn norm hands over an L1 width-sharded replicated activation; the router / sparse matmuls take
            # interleaved inputs. One conversion at the block entry.
            x = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG)
        x4 = x if (len(orig) == 4 and orig[2] == T) else ttnn.reshape(x, (1, 1, T, H))
        scores, ids = self._route(x4)
        if _MOE_DEBUG:  # diagnostics: keep the last routing decision on device (read after the pass)
            self.last_ids = ttnn.clone(ids)
        if mode == "decode" or (mode is None and T <= self.GATHER_MAX_TOKENS):
            if T <= self.GATHER_MAX_TOKENS:
                routed = self._routed_gather(x4, scores, ids, T)
            else:
                routed = self._routed_masked(x4, scores, ids, T, chunk=TILE)
        else:
            routed = self._routed_masked(x4, scores, ids, T, chunk=self.PREFILL_CHUNK)
        ttnn.deallocate(scores)
        ttnn.deallocate(ids)
        if _FAULT_ROUTED_SCALE is not None:  # test-only fault injection (stage-1 gate evidence): the review #1 bug
            routed = ttnn.multiply(routed, _FAULT_ROUTED_SCALE, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        shared = self._shared(x4)
        out = ttnn.add(routed, shared, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(routed)
        ttnn.deallocate(shared)
        if self.num_devices > 1:
            # TP on the expert intermediate dim: every device holds a partial [.., H] sum (routed + shared); the layer
            # contract wants the FRACTURED residual [.., H / tp], so reduce-scatter along the hidden dim (the same
            # collective the dense Qwen36MLP TP path issues). On a (1,4) mesh tt_all_reduce is a reduce-scatter.
            from models.tt_transformers.tt.ccl import tt_all_reduce

            out = tt_all_reduce(
                out,
                self.device,
                self.tt_ccl,
                cluster_axis=0,
                dim=3,
                # links / decode output memory: the multichip decoder's MCOpts (tt/multichip_decoder.py), if installed
                num_reduce_scatter_links=getattr(self.args, "ccl_num_links", lambda: None)(),
                topology=self.args.ccl_topology(),
                memory_config=getattr(self.args, "ccl_out_memory_config", lambda t: ttnn.DRAM_MEMORY_CONFIG)(out),
                dtype=ttnn.bfloat16,
            )
            orig = orig[:-1] + [out.shape[-1]]
        if list(out.shape) != orig:
            out = ttnn.reshape(out, orig)
        return out

    # ------------------------------------------------------------------------------------------------
    def _router_logits(self, x4, r_mc, dtype=ttnn.bfloat16):
        if getattr(self, "_router_sharded", False):
            from models.demos.blackhole.qwen36.tt import tp_common as tpc

            logits = tpc.sharded_decode_matmul(
                x4,
                self.router_w,
                self.router_ckc,
                self.router_decode_progcfg,
                self.args.act_shard_hidden,
                tpc.create_prefill_matmul_program_config,
                self.hidden,
                decode_out_memory_config=r_mc,
                out_dtype=dtype if dtype != ttnn.bfloat16 else None,
            )
            if dtype != logits.dtype:
                cast = ttnn.typecast(logits, dtype, memory_config=r_mc)
                ttnn.deallocate(logits)
                logits = cast
            return logits
        return ttnn.linear(x4, self.router_w, dtype=dtype, compute_kernel_config=self.router_ckc, memory_config=r_mc)

    def _route_sigmoid_bias(self, x4, r_mc):
        """GLM router (module docstring): fp32 sigmoid scores, top-k of score + centered bias, normalized gathered
        scores. Returns bf16 weights [1,1,T,k] and uint16 ids [1,1,T,k] like the softmax router."""
        logits = self._router_logits(x4, r_mc, dtype=ttnn.float32)
        sig = ttnn.sigmoid(logits, memory_config=r_mc)
        ttnn.deallocate(logits)
        choice = ttnn.add(sig, self.router_bias, memory_config=r_mc)
        # plain top-k: the large-k composite route (ROUTER_TOPK_ROUTE_K) takes bf16 only, an fp32 row with k = 80 would
        # run the stock single-core sort for 10x the k
        pad_w = getattr(self.args, "router_topk_pad", lambda: 0)()  # stage-5 A/B (tt/multichip_decoder.MCOpts)
        if pad_w and pad_w > choice.shape[-1]:  # padded columns at -1e30 never rank; the indices stay < E
            padded = ttnn.pad(choice, [(0, 0), (0, 0), (0, 0), (0, pad_w - choice.shape[-1])], -1e30)
            ttnn.deallocate(choice)
            choice = padded
        values, ids = ttnn.topk(choice, k=self.top_k, dim=-1, largest=True, sorted=True, memory_config=r_mc)
        ttnn.deallocate(values)
        ttnn.deallocate(choice)
        if ROUTER_WEIGHTS == "onehot" and sig.shape[-2] <= TILE:  # decode rows; prefill rows keep the gather
            # w[t, j] = sum_e sig[t, e] * (ids[t, j] == e): a broadcast compare + multiply + row sum on all cores instead
            # of ttnn.gather (one core, 31 us + a fill-pad for a [32, 128] fp32 row at decode, Tracy 2026-10-08)
            T = sig.shape[-2]
            idf = ttnn.typecast(ids, ttnn.float32, memory_config=r_mc)
            idc = ttnn.reshape(idf, (1, T, self.top_k, 1))
            ttnn.deallocate(idf)
            sig4 = ttnn.reshape(sig, (1, T, 1, self.num_experts))
            oh = ttnn.eq(idc, self._expert_iota, memory_config=r_mc)  # [1, T, k, E]
            ttnn.deallocate(idc)
            sel = ttnn.multiply(oh, sig4, memory_config=r_mc)
            ttnn.deallocate(oh)
            w = ttnn.reshape(ttnn.sum(sel, dim=-1, keepdim=True, memory_config=r_mc), (1, 1, T, self.top_k))
            ttnn.deallocate(sel)
        else:
            w = ttnn.gather(sig, dim=-1, index=ids, memory_config=r_mc)  # [1,1,T,k] fp32
        # force=False: frees only when no view of this buffer is alive (stage-6/7 rework lifetime guard)
        ttnn.deallocate(sig, force=False)
        if getattr(self.args, "norm_topk_prob", True):
            denom = ttnn.sum(w, dim=-1, keepdim=True, memory_config=r_mc)
            if self.routed_scaling != 1.0:
                denom = ttnn.multiply(denom, 1.0 / self.routed_scaling, memory_config=r_mc)
            wn = ttnn.div(w, denom, memory_config=r_mc)
            ttnn.deallocate(denom)
        else:
            wn = ttnn.multiply(w, self.routed_scaling, memory_config=r_mc)
        ttnn.deallocate(w)
        scores = ttnn.typecast(wn, ttnn.bfloat16, memory_config=r_mc)
        ttnn.deallocate(wn)
        if ids.dtype != ttnn.uint16:
            ids16 = ttnn.typecast(ids, ttnn.uint16, memory_config=r_mc)
            ttnn.deallocate(ids)
            ids = ids16
        return scores, ids

    def _route(self, x4):
        """Router: returns renormalized top-k scores [1,1,T,k] (bf16) and expert ids [1,1,T,k] (uint16)."""
        # decode rows (<= 32 tokens): keep the router chain L1-resident (perf pass, QWEN35_ROUTER_L1); prefill rows in DRAM
        r_mc = ttnn.L1_MEMORY_CONFIG if (ROUTER_L1 and x4.shape[-2] <= TILE) else ttnn.DRAM_MEMORY_CONFIG
        if self.router_kind == "sigmoid_bias":
            return self._route_sigmoid_bias(x4, r_mc)
        if getattr(self, "_router_sharded", False):
            from models.demos.blackhole.qwen36.tt import tp_common as tpc

            logits = tpc.sharded_decode_matmul(
                x4,
                self.router_w,
                self.router_ckc,
                self.router_decode_progcfg,
                self.args.act_shard_hidden,
                tpc.create_prefill_matmul_program_config,
                self.hidden,
                decode_out_memory_config=r_mc,
            )
        else:
            logits = ttnn.linear(
                x4,
                self.router_w,
                dtype=ttnn.bfloat16,
                compute_kernel_config=self.router_ckc,
                memory_config=r_mc,
            )
        if ROUTER_TOPK_FIRST:
            # softmax over all experts -> top-k -> renormalize == top-k of the logits -> softmax over the k selected
            # (the global partition function cancels in the renormalization): 2 ops instead of 4 (perf pass turn 1).
            if ROUTER_TOPK_PAD > logits.shape[-1]:
                # ttnn.topk runs single-core below a 1024-wide input (40 us on [32, 256] at width 1 - the costliest op of
                # the width-1 GDN layer after the sparse matmuls); padded to 1024 with -inf it takes the multi-core path
                padded = ttnn.pad(
                    logits,
                    [(0, 0), (0, 0), (0, 0), (0, ROUTER_TOPK_PAD - logits.shape[-1])],
                    value=-1e9,
                    memory_config=r_mc,
                )
                if padded is not logits:
                    ttnn.deallocate(logits)
                logits = padded
            if (
                ROUTER_TOPK_ROUTE_K > max(64, self.top_k)
                and logits.shape[-1] >= ROUTER_TOPK_ROUTE_K
                and logits.shape[-2] <= TILE  # decode rows only (prefill row counts not measured)
            ):
                v_all, i_all = ttnn.topk(
                    logits, k=ROUTER_TOPK_ROUTE_K, dim=-1, largest=True, sorted=True, stable=False, memory_config=r_mc
                )
                rows = v_all.shape[-2]
                values = ttnn.slice(v_all, [0, 0, 0, 0], [1, 1, rows, self.top_k], memory_config=r_mc)
                ids = ttnn.slice(i_all, [0, 0, 0, 0], [1, 1, rows, self.top_k], memory_config=r_mc)
                ttnn.deallocate(v_all)
                ttnn.deallocate(i_all)
            else:
                values, ids = ttnn.topk(logits, k=self.top_k, dim=-1, largest=True, sorted=True, memory_config=r_mc)
            ttnn.deallocate(logits)
            scores = ttnn.softmax(values, dim=-1, numeric_stable=True, memory_config=r_mc)
            ttnn.deallocate(values)
        else:
            probs = ttnn.softmax(logits, dim=-1, numeric_stable=True)
            ttnn.deallocate(logits)
            values, ids = ttnn.topk(probs, k=self.top_k, dim=-1, largest=True, sorted=True)
            ttnn.deallocate(probs)
            denom = ttnn.sum(values, dim=-1, keepdim=True)
            scores = ttnn.div(values, denom)
            ttnn.deallocate(values)
            ttnn.deallocate(denom)
        if ids.dtype != ttnn.uint16:
            ids = ttnn.typecast(ids, ttnn.uint16)
        return scores, ids

    def _alloc_dev(self, shape, dtype, zero=False, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG):
        """Device-side buffer allocation (no host write, trace-safe, nothing cached): ``ttnn.empty`` and, when the
        consumer needs zeros (the routing-matrix scatter target), a device ``ttnn.fill``. The sparse-matmul outputs of the
        expert-sorted path do not need zeros: every real group is written by the op and the padded groups are never
        read (``undispatch_rows`` gathers real rows only)."""
        t = ttnn.empty(list(shape), dtype, layout, self.device, memory_config)
        if zero:
            z = ttnn.fill(t, 0.0, memory_config=memory_config)
            ttnn.deallocate(t)
            return z
        return t

    def _routing_matrix(self, scores, ids, T):
        """Dense [1,1,T,E] routing weights (zero for unselected experts)."""
        zeros = self._alloc_dev((1, 1, T, self.num_experts), ttnn.bfloat16, zero=True)
        idx = ttnn.typecast(ids, ttnn.uint32) if ids.dtype != ttnn.uint32 else ids
        routing = ttnn.scatter(zeros, dim=-1, index=idx, src=scores)
        ttnn.deallocate(zeros)
        if idx is not ids:
            ttnn.deallocate(idx)
        return routing

    def _rows_view(self, t, shape):
        """Reinterpret the compact gather-mode sparse-matmul output (one tile-padded row per selected expert) as
        ``shape`` without moving data (``moe_gather_tail``: ``ttnn.view``, the last two dims are unchanged);
        otherwise the stock ``ttnn.reshape``."""
        if getattr(self.args, "moe_gather_tail", False):
            use_view = getattr(self, "_rows_view_ok", None)
            if use_view is None:  # decided on the first (eager) call, fixed before any trace capture
                try:
                    out = ttnn.view(t, shape)
                    self._rows_view_ok = True
                    return out
                except Exception as e:  # noqa: BLE001 - op support probe
                    logger.warning(
                        f"moe_gather_tail: ttnn.view {tuple(t.shape)} -> {shape} rejected ({str(e)[:100]}); reshape"
                    )
                    self._rows_view_ok = False
            elif use_view:
                return ttnn.view(t, shape)
        return ttnn.reshape(t, shape)

    def _clamp_packed(self, fused):
        """swiglu_limit on a packed [..., up | gate] projection: min(x, L) everywhere, max(x, -L) on the up half."""
        if self.swiglu_limit is None:
            return fused
        c = ttnn.minimum(fused, float(self.swiglu_limit))
        # force=False: frees only when no view of this buffer is alive (stage-6/7 rework lifetime guard)
        ttnn.deallocate(fused, force=False)
        out = ttnn.maximum(c, self._clamp_lo)
        ttnn.deallocate(c)
        return out

    def _swiglu_rows(self, fused, lead, half):
        """SwiGLU on the packed [1, lead, 1, 2*half] projection (M=1 rows: slices avoid tile padding)."""
        if tuple(fused.tile.tile_shape) != (32, 32):
            # narrow-output-tile probe: ttnn.slice supports 32x32 tiles only -> the fused swiglu op (gate/up halves)
            out = ttnn.swiglu(fused)
            ttnn.deallocate(fused)
            return out
        fused = self._clamp_packed(fused)
        up = ttnn.slice(fused, [0, 0, 0, 0], [1, lead, 1, half])
        gate = ttnn.slice(fused, [0, 0, 0, half], [1, lead, 1, 2 * half])
        ttnn.deallocate(fused)
        out = ttnn.multiply(up, gate, input_tensor_b_activations=[ttnn.UnaryOpType.SILU])
        ttnn.deallocate(gate)
        ttnn.deallocate(up)
        return out

    # Stage 10 note: the served width-1 (single live user) decode step is intermittently non-deterministic on the
    # 48-layer model (doc/optimized_vllm/work_log.md 05:50-07:40). It is NOT this shortcut: T = 1 through explicit copies,
    # through the T = 2 code with a duplicated token, with DRAM outputs or the plain reshape/sum tail, and with the
    # attention / GDN blocks padded to two rows all stayed non-deterministic, while two real users (width 2) are
    # bit-reproducible. The adapter therefore serves single users on the width-2 decode trace (QWEN35_DECODE_MIN_WIDTH).
    # The shortcut (stage-9 behaviour, the standalone B=1 generator's path) stays the default;
    # QWEN35_MOE_GATHER_T1_SHORTCUT=0 takes the copy path (probe knob).
    GATHER_T1_SHORTCUT = os.environ.get("QWEN35_MOE_GATHER_T1_SHORTCUT", "1") == "1"
    # stage-10 probe knobs for the gather path (instance attribute overrides the class default):
    #   out_dram: sparse-matmul outputs in DRAM instead of L1; plain_tail: ttnn.reshape + ttnn.sum instead of the
    #   moe_gather_tail view + fast_reduce_nc; dup2: a single token is processed as two identical rows (T = 2 path, row 0 kept)
    GATHER_KNOBS = {"out_dram": False, "plain_tail": False, "dup2": False}

    def _routed_gather(self, x4, scores, ids, T):
        k, Nl, H = self.top_k, self.inter_local, self.hidden
        knobs = self.GATHER_KNOBS
        if T == 1 and knobs.get("dup2"):
            # probe variant: duplicate the single token into a 2-row batch, run the T = 2 code, keep row 0
            x2 = ttnn.concat([x4, x4], dim=2)
            sc2 = ttnn.concat([scores, scores], dim=2)
            id2 = ttnn.concat([ids, ids], dim=2)
            out2 = self._routed_gather(x2, sc2, id2, 2)
            for t in (x2, sc2, id2):
                ttnn.deallocate(t)
            out1 = ttnn.slice(out2, [0, 0, 0, 0], [1, 1, 1, H])
            ttnn.deallocate(out2)
            return out1
        mm_mc = ttnn.DRAM_MEMORY_CONFIG if knobs.get("out_dram") else ttnn.L1_MEMORY_CONFIG
        plain_tail = knobs.get("plain_tail")
        ids_rm = ids if ids.layout == ttnn.ROW_MAJOR_LAYOUT else ttnn.to_layout(ids, ttnn.ROW_MAJOR_LAYOUT)
        outs = []
        shortcut = T == 1 and self.GATHER_T1_SHORTCUT
        for u in range(T):
            # T == 1 without the shortcut: a full-extent ttnn.slice returns its INPUT (no copy), so take explicit copies
            # (the activation into DRAM, where the T > 1 slices land)
            if shortcut:
                ids_u, x_u = ids_rm, x4
            elif T == 1:
                ids_u = ttnn.clone(ids_rm)
                x_u = ttnn.clone(x4, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            else:
                ids_u = ttnn.slice(ids_rm, [0, 0, u, 0], [1, 1, u + 1, k])
                x_u = ttnn.slice(x4, [0, 0, u, 0], [1, 1, u + 1, H])
            fused = ttnn.sparse_matmul(
                x_u,
                self.up_gate,
                sparsity=self.ones_mask,
                indices=ids_u,
                is_input_a_sparse=False,
                is_input_b_sparse=True,
                memory_config=mm_mc,
                output_tile=self.tile_gu,
                program_config=self.cfg_gate_up_gather,
                compute_kernel_config=self.routed_ckc,
                dtype=self.prec.act_dtype,
            )  # compact [1,1,1,k,1,2Nl]
            fused_rows = (
                ttnn.reshape(fused, (1, k, 1, 2 * Nl)) if plain_tail else self._rows_view(fused, (1, k, 1, 2 * Nl))
            )
            din = self._swiglu_rows(fused_rows, k, Nl)
            down = ttnn.sparse_matmul(
                din,
                self.down,
                sparsity=self.ones_mask,
                indices=ids_u,
                is_input_a_sparse=True,
                is_input_b_sparse=True,
                memory_config=mm_mc,
                output_tile=self.tile,
                program_config=self.cfg_down_gather,
                compute_kernel_config=self.routed_ckc,
                dtype=self.prec.act_dtype,
            )  # [1,k,1,H]
            ttnn.deallocate(din)
            if shortcut:
                sc_u = scores
            elif T == 1:
                sc_u = ttnn.clone(scores)
            else:
                sc_u = ttnn.slice(scores, [0, 0, u, 0], [1, 1, u + 1, k])
            sc_u = ttnn.reshape(sc_u, (1, k, 1, 1))
            down_rows = ttnn.reshape(down, (1, k, 1, H)) if plain_tail else self._rows_view(down, (1, k, 1, H))
            weighted = ttnn.multiply(down_rows, sc_u)
            # force=False: frees only when no view of this buffer is alive (stage-6/7 rework lifetime guard)
            ttnn.deallocate(down_rows, force=False)
            if getattr(self.args, "moe_gather_tail", False) and not plain_tail:
                # sum over the k expert rows with the plain NC reduction: ttnn.sum(dim=1) adds a fill_pad of the
                # tile padding around the same kernel; row 0 (the only logical row) is identical. The reduction
                # keeps the tile-padded row count, so trim to the one logical row when it does.
                acc_u = ttnn.experimental.fast_reduce_nc(weighted, dims=[1], compute_kernel_config=self.routed_ckc)
                if acc_u.shape[-2] != 1:
                    acc_t = ttnn.slice(acc_u, [0, 0, 0, 0], [1, 1, 1, H])
                    ttnn.deallocate(acc_u)
                    acc_u = acc_t
                outs.append(acc_u)
            else:
                outs.append(ttnn.sum(weighted, dim=1, keepdim=True, compute_kernel_config=self.routed_ckc))  # [1,1,1,H]
            ttnn.deallocate(weighted)
            if not shortcut:
                ttnn.deallocate(ids_u)
                ttnn.deallocate(x_u)
        if ids_rm is not ids:
            ttnn.deallocate(ids_rm)
        if shortcut:
            acc = outs[0]
        else:
            acc = ttnn.concat(outs, dim=2) if T > 1 else ttnn.clone(outs[0])
            for o in outs:
                ttnn.deallocate(o)
        return ttnn.reshape(acc, (1, 1, T, H))

    def _routed_masked(self, x4, scores, ids, T, chunk):
        E, Nl, H = self.num_experts, self.inter_local, self.hidden
        dram = ttnn.DRAM_MEMORY_CONFIG
        routing = self._routing_matrix(scores, ids, T)  # [1,1,T,E]
        pad = (-T) % chunk
        if pad:
            x4 = ttnn.pad(x4, [(0, 0), (0, 0), (0, pad), (0, 0)], value=0.0)
            routing = ttnn.pad(routing, [(0, 0), (0, 0), (0, pad), (0, 0)], value=0.0)
        Tp = T + pad
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
            # a full-extent ttnn.slice returns its input (no copy): a one-chunk call uses x4 itself and must not free it
            x_c = x4 if Tp == chunk else ttnn.slice(x4, [0, 0, s0, 0], [1, 1, s0 + chunk, H], memory_config=dram)
            r_c = ttnn.slice(routing, [0, 0, s0, 0], [1, 1, s0 + chunk, E], memory_config=dram)
            # union of experts selected by any token in the chunk (scores are non-negative)
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
            )  # [1,1,1,E,chunk,2Nl]
            if x_c is not x4:
                ttnn.deallocate(x_c)
            fused = self._clamp_packed(ttnn.reshape(fused, (1, E, chunk, 2 * Nl)))
            din = ttnn.swiglu(fused, memory_config=dram)
            ttnn.deallocate(fused)
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
            weights_c = ttnn.permute(r_c, (0, 3, 2, 1), memory_config=dram)  # [1,E,chunk,1]
            weighted = ttnn.multiply(ttnn.reshape(down, (1, E, chunk, H)), weights_c, memory_config=dram)
            ttnn.deallocate(down)
            ttnn.deallocate(weights_c)
            ttnn.deallocate(r_c)
            ttnn.deallocate(mask)
            outs.append(ttnn.experimental.fast_reduce_nc(weighted, dims=[1]))  # [1,1,chunk,H]
            ttnn.deallocate(weighted)
        ttnn.deallocate(routing)
        acc = outs[0] if len(outs) == 1 else ttnn.concat(outs, dim=2, memory_config=dram)
        if len(outs) > 1:
            for o in outs:
                ttnn.deallocate(o)
        acc = ttnn.reshape(acc, (1, 1, Tp, H))
        if pad:
            acc = ttnn.slice(acc, [0, 0, 0, 0], [1, 1, T, H], memory_config=dram)
        return acc

    def _shared(self, x4):
        mc = ttnn.DRAM_MEMORY_CONFIG
        if self.swiglu_limit is None:
            gate = ttnn.linear(
                x4, self.shared_w1, activation="silu", compute_kernel_config=self.shared_ckc, memory_config=mc
            )
            up = ttnn.linear(x4, self.shared_w3, compute_kernel_config=self.shared_ckc, memory_config=mc)
        else:
            L = float(self.swiglu_limit)
            g_raw = ttnn.linear(x4, self.shared_w1, compute_kernel_config=self.shared_ckc, memory_config=mc)
            gate = ttnn.silu(ttnn.minimum(g_raw, L), memory_config=mc)
            ttnn.deallocate(g_raw)
            u_raw = ttnn.linear(x4, self.shared_w3, compute_kernel_config=self.shared_ckc, memory_config=mc)
            up = ttnn.clamp(u_raw, min=-L, max=L, memory_config=mc)
            ttnn.deallocate(u_raw)
        hidden = ttnn.mul(gate, up, memory_config=mc)
        ttnn.deallocate(gate)
        ttnn.deallocate(up)
        out = ttnn.linear(hidden, self.shared_w2, compute_kernel_config=self.shared_ckc, memory_config=mc)
        ttnn.deallocate(hidden)
        if not self.shared_gated:
            return out
        g = ttnn.linear(x4, self.shared_gate_w, compute_kernel_config=self.shared_ckc, memory_config=mc)  # [1,1,T,1]
        g = ttnn.sigmoid(g)
        out = ttnn.multiply(out, g, memory_config=mc)
        ttnn.deallocate(g)
        return out
