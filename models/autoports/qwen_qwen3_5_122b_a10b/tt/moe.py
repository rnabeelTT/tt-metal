# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Qwen3.5-MoE feed-forward block: router + 256 routed experts (top-8) + gated shared expert.

HF semantics (``Qwen3_5MoeSparseMoeBlock``)::

    probs   = softmax(x @ W_gate^T)            # over all 256 experts, fp32
    w, ids  = topk(probs, 8);  w = w / sum(w)   # renormalized routing weights
    routed  = sum_i  w_i * down_i(silu(gate_i(x)) * up_i(x))
    shared  = sigmoid(x @ W_sg^T) * down_s(silu(gate_s(x)) * up_s(x))
    out     = routed + shared

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
from dataclasses import dataclass, replace

import torch
from loguru import logger

import ttnn

TILE = 32


def _largest_divisor_at_most(n: int, cap: int) -> int:
    d = max(1, min(cap, n))
    while d > 1 and n % d:
        d -= 1
    return d


def sparse_matmul_program_config(
    m: int, n: int, k: int, *, target_cores: int, in0_block_w_cap: int, grid_y: int = 8, tile_w: int = TILE
):
    """1D-mcast config for ``ttnn.sparse_matmul``: cores must divide Nt, ``Kt % in0_block_w == 0``."""
    n_tiles = max(1, math.ceil(n / tile_w))
    k_tiles = max(1, math.ceil(k / TILE))
    m_tiles = max(1, math.ceil(m / TILE))
    cores = _largest_divisor_at_most(n_tiles, max(1, target_cores))
    per_core_n = n_tiles // cores
    cy = _largest_divisor_at_most(cores, grid_y)
    cx = cores // cy
    sub_w = _largest_divisor_at_most(per_core_n, 8)
    sub_h = _largest_divisor_at_most(m_tiles, max(1, 8 // sub_w))
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
    GATHER_MAX_TOKENS = 4

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

        cache = tensor_cache_path

        def cache_name(name):
            return str(cache / f"mlp.{name}") if cache else None

        mapper = (lambda dim: ttnn.ShardTensorToMesh(mesh_device, dim=dim)) if tp > 1 else (lambda dim: None)

        def upload(t, dtype, name, dim=None, layout=ttnn.TILE_LAYOUT, tile=None):
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
        self.router_w = upload(state_dict["gate.weight"].T.contiguous(), self.prec.router, "gate.weight")

        # ---- routed experts (HF fused: gate_up [E, 2N, H] = [gate; up] rows, down [E, H, N]) -------
        gate_up = state_dict["experts.gate_up_proj"]  # [E, 2N, H]
        down = state_dict["experts.down_proj"]  # [E, H, N]
        E, N, H = self.num_experts, self.inter, self.hidden
        assert tuple(gate_up.shape) == (E, 2 * N, H) and tuple(down.shape) == (E, H, N)
        gate_t = gate_up[:, :N, :].transpose(1, 2)  # [E, H, N]
        up_t = gate_up[:, N:, :].transpose(1, 2)  # [E, H, N]
        # Pack per TP shard as [up_d | gate_d] so ttnn.swiglu (first * silu(second)) is exact per device.
        Nl = self.inter_local
        packed = torch.cat(
            [
                torch.cat([up_t[:, :, d * Nl : (d + 1) * Nl], gate_t[:, :, d * Nl : (d + 1) * Nl]], dim=-1)
                for d in range(tp)
            ],
            dim=-1,
        )  # [E, H, tp*2*Nl]
        # stage-5 probe: a 16-wide output tile on the gather-mode gate/up needs the weight stored with 16-wide tiles
        # (matmul: out_tile_shape[1] % in1_tile_w == 0).
        _gu_tile = getattr(args, "gather_gate_up_output_tile", None)
        self.up_gate = upload(
            packed.unsqueeze(0).contiguous(),
            self.prec.routed_gate_up,
            "experts.up_gate",
            dim=-1,
            tile=ttnn.Tile([32, _gu_tile[1]]) if _gu_tile else None,
        )
        down_t = down.transpose(1, 2).unsqueeze(0).contiguous()  # [1, E, N, H]
        # a 16-wide gate/up output feeds the down matmul as in0 with 16-wide K tiles -> the down weight needs 16-high tiles
        self.down = upload(
            down_t,
            self.prec.routed_down,
            "experts.down",
            dim=-2,
            tile=ttnn.Tile([_gu_tile[1], 32]) if _gu_tile else None,
        )
        del gate_t, up_t, packed, down_t

        # ---- shared expert (dense SwiGLU, TP column/row parallel) + sigmoid gate -----------------------
        self.shared_w1 = upload(
            state_dict["shared_expert.gate_proj.weight"].T.contiguous(),
            self.prec.shared,
            "shared_expert.gate_proj",
            dim=-1,
        )
        self.shared_w3 = upload(
            state_dict["shared_expert.up_proj.weight"].T.contiguous(), self.prec.shared, "shared_expert.up_proj", dim=-1
        )
        self.shared_w2 = upload(
            state_dict["shared_expert.down_proj.weight"].T.contiguous(),
            self.prec.shared,
            "shared_expert.down_proj",
            dim=-2,
        )
        self.shared_gate_w = upload(
            state_dict["shared_expert_gate.weight"].T.contiguous(), ttnn.bfloat16, "shared_expert_gate"
        )  # [H, 1]

        # ---- constants ----------------------------------------------------------------------------------
        self.ones_mask = ttnn.from_torch(
            torch.ones(1, 1, 1, E, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            **({"mesh_mapper": ttnn.ReplicateTensorToMesh(mesh_device)} if tp > 1 else {}),
        )
        self.tile = ttnn.Tile([TILE, TILE])
        self.tile_gu = self.tile  # gate/up gather output tile (Qwen35MoEOptimized may narrow it)
        self.routed_ckc = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=self.prec.routed_fidelity,
            math_approx_mode=False,
            fp32_dest_acc_en=False,  # tt-metal #49068: fp32 dest halves the matmul dest and corrupts expert output on BH
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
        if mode == "decode" or (mode is None and T <= self.GATHER_MAX_TOKENS):
            if T <= self.GATHER_MAX_TOKENS:
                routed = self._routed_gather(x4, scores, ids, T)
            else:
                routed = self._routed_masked(x4, scores, ids, T, chunk=TILE)
        else:
            routed = self._routed_masked(x4, scores, ids, T, chunk=self.PREFILL_CHUNK)
        ttnn.deallocate(scores)
        ttnn.deallocate(ids)
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
                topology=self.args.ccl_topology(),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                dtype=ttnn.bfloat16,
            )
            orig = orig[:-1] + [out.shape[-1]]
        if list(out.shape) != orig:
            out = ttnn.reshape(out, orig)
        return out

    # ------------------------------------------------------------------------------------------------
    def _route(self, x4):
        """Router: returns renormalized top-k scores [1,1,T,k] (bf16) and expert ids [1,1,T,k] (uint16)."""
        logits = ttnn.linear(
            x4,
            self.router_w,
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.router_ckc,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
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

    def _swiglu_rows(self, fused, lead, half):
        """SwiGLU on the packed [1, lead, 1, 2*half] projection (M=1 rows: slices avoid tile padding)."""
        if tuple(fused.tile.tile_shape) != (32, 32):
            # narrow-output-tile probe: ttnn.slice supports 32x32 tiles only -> the fused swiglu op (gate/up halves)
            out = ttnn.swiglu(fused)
            ttnn.deallocate(fused)
            return out
        up = ttnn.slice(fused, [0, 0, 0, 0], [1, lead, 1, half])
        gate = ttnn.slice(fused, [0, 0, 0, half], [1, lead, 1, 2 * half])
        ttnn.deallocate(fused)
        out = ttnn.multiply(up, gate, input_tensor_b_activations=[ttnn.UnaryOpType.SILU])
        ttnn.deallocate(gate)
        ttnn.deallocate(up)
        return out

    def _routed_gather(self, x4, scores, ids, T):
        k, Nl, H = self.top_k, self.inter_local, self.hidden
        ids_rm = ids if ids.layout == ttnn.ROW_MAJOR_LAYOUT else ttnn.to_layout(ids, ttnn.ROW_MAJOR_LAYOUT)
        outs = []
        for u in range(T):
            ids_u = ids_rm if T == 1 else ttnn.slice(ids_rm, [0, 0, u, 0], [1, 1, u + 1, k])
            x_u = x4 if T == 1 else ttnn.slice(x4, [0, 0, u, 0], [1, 1, u + 1, H])
            fused = ttnn.sparse_matmul(
                x_u,
                self.up_gate,
                sparsity=self.ones_mask,
                indices=ids_u,
                is_input_a_sparse=False,
                is_input_b_sparse=True,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                output_tile=self.tile_gu,
                program_config=self.cfg_gate_up_gather,
                compute_kernel_config=self.routed_ckc,
                dtype=self.prec.act_dtype,
            )  # compact [1,1,1,k,1,2Nl]
            din = self._swiglu_rows(self._rows_view(fused, (1, k, 1, 2 * Nl)), k, Nl)
            down = ttnn.sparse_matmul(
                din,
                self.down,
                sparsity=self.ones_mask,
                indices=ids_u,
                is_input_a_sparse=True,
                is_input_b_sparse=True,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                output_tile=self.tile,
                program_config=self.cfg_down_gather,
                compute_kernel_config=self.routed_ckc,
                dtype=self.prec.act_dtype,
            )  # [1,k,1,H]
            ttnn.deallocate(din)
            sc_u = scores if T == 1 else ttnn.slice(scores, [0, 0, u, 0], [1, 1, u + 1, k])
            sc_u = ttnn.reshape(sc_u, (1, k, 1, 1))
            down_rows = self._rows_view(down, (1, k, 1, H))
            weighted = ttnn.multiply(down_rows, sc_u)
            ttnn.deallocate(down_rows)
            if getattr(self.args, "moe_gather_tail", False):
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
            if T != 1:
                ttnn.deallocate(ids_u)
                ttnn.deallocate(x_u)
        if ids_rm is not ids:
            ttnn.deallocate(ids_rm)
        acc = outs[0] if T == 1 else ttnn.concat(outs, dim=2)
        if T != 1:
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
            x_c = ttnn.slice(x4, [0, 0, s0, 0], [1, 1, s0 + chunk, H], memory_config=dram)
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
            ttnn.deallocate(x_c)
            fused = ttnn.reshape(fused, (1, E, chunk, 2 * Nl))
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
        gate = ttnn.linear(
            x4, self.shared_w1, activation="silu", compute_kernel_config=self.shared_ckc, memory_config=mc
        )
        up = ttnn.linear(x4, self.shared_w3, compute_kernel_config=self.shared_ckc, memory_config=mc)
        hidden = ttnn.mul(gate, up, memory_config=mc)
        ttnn.deallocate(gate)
        ttnn.deallocate(up)
        out = ttnn.linear(hidden, self.shared_w2, compute_kernel_config=self.shared_ckc, memory_config=mc)
        ttnn.deallocate(hidden)
        g = ttnn.linear(x4, self.shared_gate_w, compute_kernel_config=self.shared_ckc, memory_config=mc)  # [1,1,T,1]
        g = ttnn.sigmoid(g)
        out = ttnn.multiply(out, g, memory_config=mc)
        ttnn.deallocate(g)
        return out
