# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Optimized decoder layer for Qwen/Qwen3.5-122B-A10B (stage 3 of the autoport).

Same public contract as :class:`~.fused_decoder.FusedDecoder`. Every optimization is a field of
:class:`OptimizationConfig` so candidates can be A/B measured on real weights with
``tools/sweep_optimized.py``; the defaults are the selected policy (see doc/optimized_decoder/README.md).

Knobs (single device):
* precision / fidelity per tensor group: routed expert gate/up and down, shared expert, attention and GDN
  projections, KV cache dtype (bfp8 fill cast lives in the shared kernel), activations;
* ``ttnn.sparse_matmul`` geometry per role (gather decode gate/up + down, masked prefill gate/up + down):
  core count and ``in0_block_w`` cap; prefill group (chunk) width;
* decode residual/norm layout: sharded RMSNorm program on a width-sharded L1 copy of the residual;
* paged SDPA-decode chunk sizes.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, replace

from loguru import logger

import ttnn
from models.autoports.qwen_qwen3_5_122b_a10b.tt.fused_decoder import FusedDecoder, FusionFlags, Qwen35MoEFused
from models.autoports.qwen_qwen3_5_122b_a10b.tt.layer import Qwen35MoeDecoderLayer
from models.autoports.qwen_qwen3_5_122b_a10b.tt.moe import MoEPrecision, sparse_matmul_program_config
from models.autoports.qwen_qwen3_5_122b_a10b.tt.moe_dispatch import SortedDispatch, gather_rows, undispatch_rows
from models.demos.blackhole.qwen36.tt import tp_common as tpc
from models.demos.blackhole.qwen36.utils.substate import substate


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
    # routed-expert prefill: "masked" (per-group expert union, stage 1/2) or "sorted" (expert-sorted dispatch: every 32-row
    # group is one expert; see tt/moe_dispatch.py); geometry of the two compact sparse matmuls
    prefill_mode: str = "sorted"
    sorted_min_tokens: int = 512  # below this many tokens the masked path wins (static per-expert padding dominates)
    sorted_group_rows: int = 32  # rows per expert block (multiple of 32): fewer weight re-reads vs more padding
    sorted_max_tokens: int = (
        4096  # largest prefill chunk routed to the sorted path (index math exact to 7196; checked to 4096)
    )
    sorted_gate_up_cores: int = 64
    sorted_gate_up_in0: int = 32
    sorted_down_cores: int = 64
    sorted_down_in0: int = 32
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
                    32, H, 2 * Sl + 1, opt.shared_decode_cores, fp32_acc=fp32, grid_w=grid_w
                ),
                tpc.create_matmul_1d_decode_progcfg(32, Sl, H, opt.shared_decode_cores, fp32_acc=fp32, grid_w=grid_w),
            )

    # ---- routed experts, sorted dispatch (prefill) ----------------------------------------------------------
    def _routed_masked(self, x4, scores, ids, T, chunk):
        if self.opt.prefill_mode != "sorted" or not (self.opt.sorted_min_tokens <= T <= self.opt.sorted_max_tokens):
            return super()._routed_masked(x4, scores, ids, T, chunk)
        return self._routed_sorted(x4, scores, ids, T)

    def _routed_sorted(self, x4, scores, ids, T):
        E, Nl, H, k = self.num_experts, self.inter_local, self.hidden, self.top_k
        opt, act = self.opt, self.prec.act_dtype
        gs = opt.sorted_group_rows
        if getattr(self, "_dispatch", None) is None:
            self._dispatch = SortedDispatch(self.device, E, k, group=gs)
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
        m = self._dispatch.build(ids, scores, T)
        Rd, G = m["Rd"], m["G"]
        x_disp = gather_rows(x4, m["src_tok"], Rd, gs)  # [1,G,gs,H]
        ttnn.deallocate(m["src_tok"])
        narrow = self.tile_gu.tile_shape[1] != 32
        out_gu = None if narrow else self._alloc_dev((1, G, gs, 2 * Nl), act)
        fused = ttnn.sparse_matmul(
            x_disp,
            self.up_gate,
            sparsity=m["group_mask"],
            nnz=G,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=self._cfg_sorted_gu,
            **({"output_tile": self.tile_gu} if narrow else {"optional_output_tensor": out_gu}),
            compute_kernel_config=self.routed_ckc,
            dtype=act,
        )  # compact [1,G,32,2Nl]
        ttnn.deallocate(x_disp)
        din = ttnn.swiglu(fused, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # [1,G,32,Nl]
        ttnn.deallocate(fused)
        # routing weight on the (narrower) expert intermediate; the down projection is linear
        din = ttnn.multiply(ttnn.reshape(din, (1, 1, Rd, Nl)), m["weights"], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(m["weights"])
        din = ttnn.reshape(din, (1, G, gs, Nl))
        out_dn = self._alloc_dev((1, G, gs, H), act)
        down = ttnn.sparse_matmul(
            din,
            self.down,
            sparsity=m["group_mask"],
            nnz=G,
            optional_output_tensor=out_dn,
            is_input_a_sparse=False,
            is_input_b_sparse=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=self._cfg_sorted_dn,
            compute_kernel_config=self.routed_ckc,
            dtype=act,
        )  # compact [1,G,32,H]
        ttnn.deallocate(din)
        ttnn.deallocate(m["group_mask"])
        y = undispatch_rows(down, None, m["undispatch"], T, k, H)
        ttnn.deallocate(down)
        ttnn.deallocate(m["undispatch"])
        return y

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
        sg = ttnn.slice(fused, [0, 0, 0, 2 * Sl], [1, 1, T, 2 * Sl + 1], memory_config=mc)
        ttnn.deallocate(fused)
        hidden = ttnn.multiply(
            up, gate, input_tensor_b_activations=[ttnn.UnaryOpType.SILU], memory_config=ttnn.L1_MEMORY_CONFIG
        )
        ttnn.deallocate(gate)
        ttnn.deallocate(up)
        out = ttnn.linear(
            hidden, self.shared_w2, compute_kernel_config=self.shared_ckc, program_config=pc_down, memory_config=mc
        )
        ttnn.deallocate(hidden)
        out = ttnn.multiply(out, sg, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID], memory_config=mc)
        ttnn.deallocate(sg)
        return out


class Qwen35MoeOptimizedDecoderLayer(Qwen35MoeDecoderLayer):
    def _build_feed_forward(self, mesh_device, args, state_dict, layer_num, tensor_cache_path, tt_ccl):
        if not args.is_moe_layer(layer_num):
            return super()._build_feed_forward(mesh_device, args, state_dict, layer_num, tensor_cache_path, tt_ccl)
        mlp_state = substate(state_dict, f"layers.{layer_num}.mlp")
        mlp_cache = (tensor_cache_path / f"layers.{layer_num}") if tensor_cache_path else None
        return Qwen35MoEOptimized(mesh_device, args, mlp_state, mlp_cache, tt_ccl=tt_ccl, layer_num=layer_num)


class OptimizedDecoder(FusedDecoder):
    """Decoder layer with the selected optimization policy. Same contract as FunctionalDecoder."""

    LAYER_CLS = Qwen35MoeOptimizedDecoderLayer

    @classmethod
    def from_state_dict(
        cls, state_dict, *, hf_config, layer_idx, mesh_device, opt: OptimizationConfig | None = None, **kwargs
    ):
        opt = opt or SELECTED

        def hook(args):
            args.opt = opt
            args.fusion_flags = opt.fusion
            args.attn_fuse_qkv = opt.fusion.attn_fused_qkv
            args.attn_weight_dtype = opt.attn_weight_dtype
            args.gdn_weight_dtype = opt.gdn_weight_dtype
            args.decode_norm_sharded = opt.decode_norm_sharded
            mode_attn_eff = opt.dense_decode_mode_attn or opt.dense_decode_mode
            assert not opt.attn_norm_out_sharded or (
                mode_attn_eff == "dram_sharded" and opt.fusion.attn_fused_qkv
            ), "attn_norm_out_sharded needs the DRAM-sharded decode hook and the fused qkv projection (sharded-input consumer)"
            args.attn_norm_out_sharded = opt.attn_norm_out_sharded
            args.gdn_state_dtype = ttnn.float32 if opt.gdn_state_fp32 else ttnn.bfloat16
            args.gdn_decode_high_precision = opt.gdn_state_fp32
            args.sdpa_decode_q_chunk = opt.sdpa_decode_q_chunk
            args.sdpa_decode_k_chunk = opt.sdpa_decode_k_chunk
            mode_attn = opt.dense_decode_mode_attn or opt.dense_decode_mode
            mode_gdn = opt.dense_decode_mode_gdn or opt.dense_decode_mode
            grid_w = mesh_device.compute_with_storage_grid_size().x
            H = args.dim
            if mode_attn != "default":
                args.attn_decode_matmul_fn = DenseDecodeMatmul(opt, grid_w, mode_attn)
                attn_qkv_n = args.n_heads * args.head_dim * 2 + 2 * args.n_kv_heads * args.head_dim
                args.attn_weight_memcfgs = dense_weight_memcfgs(
                    mode_attn, {"qkv": (H, attn_qkv_n), "o": (args.n_heads * args.head_dim, H)}, opt.dense_ds_roles
                )
            if mode_gdn != "default":
                args.gdn_decode_matmul_fn = DenseDecodeMatmul(opt, grid_w, mode_gdn)
                args.gdn_decode_matmul_fn.mega_qkv_n = args.linear_q_dim + args.linear_k_dim + args.linear_v_dim
                gdn_mega_n = (
                    args.linear_q_dim
                    + args.linear_k_dim
                    + args.linear_v_dim
                    + 2 * args.linear_num_value_heads
                    + args.linear_v_dim
                )
                args.gdn_weight_memcfgs = dense_weight_memcfgs(
                    mode_gdn, {"mega": (H, gdn_mega_n), "o": (args.linear_v_dim, H)}, opt.dense_ds_roles
                )

        kwargs.pop("fusion_flags", None)
        # FusedDecoder.from_state_dict installs its own hook; call FunctionalDecoder's directly with ours.
        from models.autoports.qwen_qwen3_5_122b_a10b.tt.functional_decoder import FunctionalDecoder

        dec = FunctionalDecoder.from_state_dict.__func__(
            cls,
            state_dict,
            hf_config=hf_config,
            layer_idx=layer_idx,
            mesh_device=mesh_device,
            _args_hook=hook,
            **kwargs,
        )
        dec.opt = opt
        return dec

    def allocate_kv_cache(self, *, max_batch_size, max_seq_len, block_size=64, dtype=None):
        return super().allocate_kv_cache(
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            block_size=block_size,
            dtype=dtype or self.opt.kv_dtype,
        )
