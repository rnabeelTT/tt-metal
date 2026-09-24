# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Multichip (tensor-parallel) decoder layer for Qwen/Qwen3.5-122B-A10B (stage 4 of the autoport).

One :class:`MultichipDecoder` is one HF ``Qwen3_5MoeDecoderLayer`` spread over a 1D mesh (the target is the
(1,4) Blackhole P300x2 mesh, FABRIC_1D, Ring). It keeps the stage-3 :class:`~.optimized_decoder.OptimizedDecoder`
public contract (``from_state_dict`` / ``allocate_kv_cache`` / ``allocate_gdn_state`` / ``reset_gdn_state`` /
``rope_decode_host`` / ``prefill_forward`` / ``decode_forward``) but drives the qwen36 TENSOR-PARALLEL token mixers
(``TPAttention``, ``TPGatedDeltaNet``, ``DistributedNorm``; ``Qwen36DecoderLayer`` builds them when
``args.num_devices > 1``) and the TP path of the MoE block (``Qwen35MoE`` with ``num_devices > 1``: expert
intermediate dim sharded, ``up_gate`` dim=-1 / ``down`` dim=-2, reduce-scatter of routed+shared at the exit).

Parallelisation (see doc/multichip_decoder/PLAN.md, "Strategy" / "Residual contract"):

* the residual stream is FRACTURED along the hidden dim: every device holds ``[1, 1, T|B, dim/tp]`` (768 columns at
  TP=4); the layer's norms all-gather it to the replicated full-dim mixer input; attention / GDN / MoE end with one
  reduce-scatter (``tt_all_reduce(dim=3)`` == ``reduce_scatter_minimal_async`` on a 1D mesh) so the residual adds
  stay fractured. 4 collectives per layer at decode (attn-norm AG, mixer RS, ffn-norm AG, MoE RS).
* attention: device ``d`` holds q heads ``8d..8d+7`` and KV head ``d // 2`` (``kv_replication``: TP=4 > 2 KV heads);
  the paged KV cache is one ``[blocks, 1, block_size, head_dim]`` tensor PER DEVICE (each device caches its local
  head; devices 0/1 hold head 0, 2/3 hold head 1). Page tables / positions / cos-sin are replicated.
* GDN: device ``d`` holds key heads ``4d..4d+3`` and value heads ``16d..16d+15``; the recurrent state
  ``[B, Nv/tp, Dk, Dv]`` and the conv history (``K`` taps ``[1, B, qkv_dim/tp]``) live inside ``TPGatedDeltaNet``
  (``_stable_state`` == fixed addresses, in-place updates -> trace-safe decode).
* routed experts: TP on the expert intermediate dim (``routed_mode="tp"``: per device ``up_gate [1,256,3072,512]``,
  ``down [1,256,256,3072]``; routing replicated). Expert parallelism (EP=4) stays a documented alternative
  (PLAN.md "Routed experts (b)").

Tensor contract (per device; the tests shard at entry with ``ShardTensorToMesh(dim=-1)`` and compose at exit with
``ConcatMeshToTensor(dim=-1)``):

``prefill_forward(hidden, *, start_pos=0, page_table=None, user_id=0, valid_len=None)``
    ``hidden``: ``[1, 1, T, dim/tp]`` bf16 TILE, fractured. Any logical ``T`` is accepted (padded to the GDN chunk
    of 128 internally, masked with ``valid_len`` for GDN, sliced back). ``page_table``: replicated int32
    ``[B, blocks]``; K/V for ``[start_pos, start_pos+T)`` are written through ``paged_fill_cache`` into the row of
    ``user_id``. ``start_pos > 0`` continues a single-user sequence (multiple of 128 and of the KV block size).
    Returns ``[1, 1, T, dim/tp]`` fractured.
``decode_forward(hidden, *, current_pos, page_table=None, cos=None, sin=None)``
    ``hidden``: ``[1, 1, B, dim/tp]`` fractured; ``current_pos``: replicated int32 ``[B]``; ``cos``/``sin``:
    replicated bf16 ``[1, B, 1, rope_dim]`` (rope_tp / HF split-halves format, see :meth:`rope_decode_host`);
    ``page_table``: replicated int32 ``[B, blocks]``. Returns ``[1, 1, B, dim/tp]`` fractured. Trace-safe.

No host/torch work happens inside a forward pass (the only accepted boundary is the qwen36 GDN kernels' host-built
``valid_len`` selector for NON-aligned prefill lengths, exactly as on the single-chip baseline).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path

import torch  # host-side only: rope_decode_host (setup helper), never inside a forward pass

import ttnn
from models.autoports.qwen_qwen3_5_122b_a10b.tt.functional_decoder import GDN_CHUNK, _text_config
from models.autoports.qwen_qwen3_5_122b_a10b.tt.model_config import Qwen35MoeModelArgs
from models.autoports.qwen_qwen3_5_122b_a10b.tt.optimized_decoder import (
    SELECTED,
    OptimizationConfig,
    OptimizedDecoder,
    Qwen35MoeOptimizedDecoderLayer,
)
from models.autoports.qwen_qwen3_5_122b_a10b.tt.weights import remap_hf_layer_state
from models.demos.blackhole.qwen36.tt import tp_common as tpc
from models.demos.blackhole.qwen36.tt.rope import Qwen36RoPESetup
from models.tt_transformers.tt.ccl import TT_CCL

# Bytes per element incl. the shared exponent (block floats), for the per-device contract numbers.
_BYTES_PER_ELEM = {ttnn.bfloat16: 2.0, ttnn.bfloat8_b: 1.0625, ttnn.bfloat4_b: 0.5625, ttnn.float32: 4.0}


@dataclass(frozen=True)
class MultichipConfig:
    """Stage-4 policy: what transfers from the stage-3 ``SELECTED`` policy and the TP-only knobs.

    Everything here is a field so ``tools``/tests can A/B it on the mesh (``with_``). Defaults are the plan's
    starting point (PLAN.md "Implementation steps" 4); values marked *verify* are re-measured on hardware.
    """

    # --- parallelisation -------------------------------------------------------------------------------------------
    mesh_shape: tuple = (1, 4)  # documented target; from_state_dict checks the opened mesh against it
    routed_mode: str = (
        "tp"  # "tp" = experts sharded on the intermediate dim (implemented); "ep" = documented alternative
    )
    # PREFILL norm mode. False (option A): the framework rule (dim <= 4096 on a 1D mesh) gathers BEFORE the norm, the
    # mixers take the plain column-parallel matmul on the replicated activation (AGMM fusion gated off). True (option
    # B): distributed prefill norm (fractured output, 768/die = 24 tiles) + fused all-gather+matmul in-projections.
    prefill_norm_distributed: bool = False
    # Channels per depthwise ttnn.conv1d call in the GDN prefill. The blocked implementation issues C / max_channels
    # calls ([q512|k512], [v1024], [v1024]); the 9B TP path (4096 channels per device) overflowed the conv1d CB in one
    # call. Here C = 3072 per device also works in one call (sweep `conv1d`, equal PCC/time); 1024 is the proven block.
    gdn_conv1d_max_channels: int = 1024
    # --- decode 1D matmul grids of the TP token mixers (None = the qwen36 _init_tp_config defaults tuned on 27B:
    # attn qkv 64, attn wo 33, GDN qkvzab 44, GDN out 33 cores). Re-sweep for K=3072 / N=4608, 2048/3072, 5152, 2048/3072.
    attn_qkv_decode_cores: int | None = None
    attn_wo_decode_cores: int | None = None
    gdn_qkvz_decode_cores: int | None = None
    gdn_out_decode_cores: int | None = None
    # shared expert decode matmuls ([3072,513] packed and [256,3072] down per device): 1D decode configs on 16 cores
    # (sweep `shared`: 16 cores -26 us per step on both layer kinds vs the ttnn auto program on 17 cores; the stage-3
    # 44-core config was sized for N=2049 / K=1024 and does not transfer to 17 / 8 tiles). None = auto.
    shared_decode_cores: int | None = (
        16  # 1D decode configs for the [3072,513] / [256,3072] shared expert (sweep `shared`: -26 us vs auto)
    )
    # --- sparse (routed expert) matmul geometry at the TP width Nl = 256 ---------------------------------------------
    # gate_up: N = 2*Nl = 512 -> 16 output tiles -> at most 16 cores with per_core_N = 1 (K = 3072 -> in0 32 of 96 tiles).
    # down: K = Nl = 256 -> 8 tiles -> in0_block_w 8; N = 3072 -> 96 tiles -> 48 cores (2 tiles each).
    gate_up_cores: int = 16
    gate_up_in0: int = 32
    down_cores: int = 48
    down_in0: int = 8
    # --- precision knobs taken from SELECTED where they transfer (verify on the mesh: PLAN.md "Risks") -------------
    kv_dtype: object = SELECTED.kv_dtype  # bfp8 paged KV (drives the bf8 SDPA / fill cast in TPAttention)
    gdn_state_dtype: object = ttnn.float32 if SELECTED.gdn_state_fp32 else ttnn.bfloat16  # TPGatedDeltaNet.reset_state
    gdn_decode_high_precision: bool = SELECTED.gdn_state_fp32  # fp32 recurrent decode step (TP default was True)
    sdpa_decode_q_chunk: int | None = SELECTED.sdpa_decode_q_chunk  # TP default was 0/0 (auto); SELECTED = 32/256
    sdpa_decode_k_chunk: int | None = SELECTED.sdpa_decode_k_chunk
    # the stage-3 policy every other MoE knob is inherited from (precision, fidelity, prefill mode, router pc, ...)
    base: OptimizationConfig = SELECTED

    def with_(self, **kw):
        return replace(self, **kw)

    def to_optimization_config(self) -> OptimizationConfig:
        """The ``OptimizationConfig`` the (shared) ``Qwen35MoEOptimized`` block reads on the mesh.

        Single-device-only knobs are disabled: the DRAM-sharded / 1D dense decode hooks (the TP mixers use their own
        ``proj_1d_decode`` matmuls and ignore ``attn_decode_matmul_fn``), the sharded decode norm and the sharded
        attention-norm output (``DistributedNorm`` already emits the width-sharded decode activation).
        """
        return self.base.with_(
            kv_dtype=self.kv_dtype,
            gdn_state_fp32=(self.gdn_state_dtype == ttnn.float32),
            sdpa_decode_q_chunk=self.sdpa_decode_q_chunk,
            sdpa_decode_k_chunk=self.sdpa_decode_k_chunk,
            gather_gate_up_cores=self.gate_up_cores,
            gather_gate_up_in0=self.gate_up_in0,
            gather_down_cores=self.down_cores,
            gather_down_in0=self.down_in0,
            sorted_gate_up_cores=self.gate_up_cores,
            sorted_gate_up_in0=self.gate_up_in0,
            sorted_down_cores=self.down_cores,
            sorted_down_in0=self.down_in0,
            masked_gate_up_cores=self.gate_up_cores,
            masked_down_cores=self.down_cores,
            masked_down_in0=self.down_in0,
            shared_decode_cores=self.shared_decode_cores,
            dense_decode_mode="default",
            dense_decode_mode_attn=None,
            dense_decode_mode_gdn=None,
            mega_split_ds=False,
            decode_norm_sharded=False,
            attn_norm_out_sharded=False,
        )

    def describe(self) -> dict:
        d = {}
        for k, v in self.__dict__.items():
            if isinstance(v, OptimizationConfig):
                d[k] = v.describe()
            elif isinstance(v, (int, bool, type(None), str, tuple)):
                d[k] = v
            else:
                d[k] = str(v).replace("DataType.", "")
        return d


# The stage-4 starting policy (PLAN.md step 4); re-selected by the stage-4 sweeps (doc/multichip_decoder/README.md).
MULTICHIP_SELECTED = MultichipConfig()


class MultichipDecoder(OptimizedDecoder):
    """Tensor-parallel decoder layer on a 1D mesh. Same public contract as :class:`OptimizedDecoder`; see the module
    docstring for the per-device tensor shapes."""

    LAYER_CLS = Qwen35MoeOptimizedDecoderLayer

    def __init__(self, *, args, layer, layer_idx, mesh_device, rope, tt_ccl, mc: MultichipConfig):
        super().__init__(args=args, layer=layer, layer_idx=layer_idx, mesh_device=mesh_device, rope=rope)
        self.tt_ccl = tt_ccl
        self.mc = mc
        self.num_devices = int(args.num_devices)
        self.hidden_local = args.dim // self.num_devices  # fractured residual width per device

    # ---------------------------------------------------------------------------------------------
    @classmethod
    def from_state_dict(
        cls,
        state_dict,
        *,
        hf_config,
        layer_idx,
        mesh_device,
        max_batch_size: int = 1,
        max_seq_len: int | None = None,
        tensor_cache_path: str | Path | None = None,
        mc: MultichipConfig | None = None,
        **kwargs,
    ):
        """Build the TP layer from HF-format weights of layer ``layer_idx`` on ``mesh_device`` (a 1D mesh, > 1 device).

        Mirrors :meth:`FunctionalDecoder.from_state_dict` but builds ONE ``TT_CCL`` per layer (semaphore pools for the
        collectives) and hands it to the layer; ``mc`` (default :data:`MULTICHIP_SELECTED`) is turned into the args
        the TP modules read (see :meth:`_args_hook`). ``tensor_cache_path`` should carry the mesh suffix
        (``Qwen36ModelArgs.weight_cache_path`` -> ``tensor_cache_bfp8_mesh1x4``): ``as_tensor`` reloads a cached file
        as-is, ignoring the mesh mapper, so single-device and mesh caches must never share a directory.
        """
        mc = mc or MULTICHIP_SELECTED
        if mc.routed_mode != "tp":
            raise NotImplementedError(
                f"routed_mode={mc.routed_mode!r}: only the intermediate-dim TP layout is implemented (EP is the documented "
                "alternative, PLAN.md 'Routed experts (b)')"
            )
        nd = mesh_device.get_num_devices()
        if nd <= 1:
            raise ValueError("MultichipDecoder needs a mesh with > 1 device; use OptimizedDecoder on a single die")
        if tuple(mesh_device.shape) != tuple(mc.mesh_shape):
            raise ValueError(f"mesh {tuple(mesh_device.shape)} != MultichipConfig.mesh_shape {tuple(mc.mesh_shape)}")
        kwargs.pop("opt", None)
        kwargs.pop("fusion_flags", None)
        kwargs.pop("moe_precision", None)
        kwargs.pop("_args_hook", None)
        opt = mc.to_optimization_config()

        text = _text_config(hf_config)
        if max_seq_len is None:
            max_seq_len = int(text.max_position_embeddings)
        args = Qwen35MoeModelArgs(mesh_device, max_batch_size=max_batch_size, max_seq_len=max_seq_len)
        cls._args_hook(args, mc, opt, mesh_device)
        if int(text.hidden_size) != args.dim or int(text.num_hidden_layers) != args.n_layers:
            raise ValueError(
                f"hf_config (hidden {text.hidden_size}, layers {text.num_hidden_layers}) does not match the "
                f"HF_MODEL checkpoint (hidden {args.dim}, layers {args.n_layers})"
            )
        if text.layer_types[layer_idx] not in ("linear_attention", "full_attention"):
            raise ValueError(f"unsupported layer_type {text.layer_types[layer_idx]!r}")
        internal = remap_hf_layer_state(
            state_dict, layer_idx, linear_q_dim=args.linear_q_dim, linear_k_dim=args.linear_k_dim
        )
        cache = Path(tensor_cache_path) if tensor_cache_path else None
        tt_ccl = TT_CCL(mesh_device)
        layer = cls.LAYER_CLS(mesh_device, args, internal, layer_idx, cache, tt_ccl=tt_ccl)
        rope = Qwen36RoPESetup(mesh_device, args) if layer.is_full_attention else None
        dec = cls(args=args, layer=layer, layer_idx=layer_idx, mesh_device=mesh_device, rope=rope, tt_ccl=tt_ccl, mc=mc)
        dec.opt = opt
        return dec

    @staticmethod
    def _args_hook(args, mc: MultichipConfig, opt: OptimizationConfig, mesh_device):
        """Install the policy on ``Qwen35MoeModelArgs`` (``num_devices``/per-device dims already come from the mesh via
        ``Qwen36ModelArgs._init_tp_config``)."""
        nd = mesh_device.get_num_devices()
        assert int(args.num_devices) == nd, (args.num_devices, nd)
        assert args.n_heads % nd == 0, f"n_heads {args.n_heads} not divisible by TP={nd}"
        assert nd % args.n_kv_heads == 0 or args.n_kv_heads % nd == 0, (args.n_kv_heads, nd)
        assert args.moe_intermediate_size % (nd * 32) == 0, (args.moe_intermediate_size, nd)
        assert args.shared_expert_intermediate_size % (nd * 32) == 0, (args.shared_expert_intermediate_size, nd)
        # stage-3 knobs that transfer (the MoE block reads args.opt; the fusion flags select the gate op / packed shared)
        args.mc = mc
        args.opt = opt
        args.fusion_flags = opt.fusion
        args.attn_fuse_qkv = opt.fusion.attn_fused_qkv  # single-device attention only; harmless on TP
        args.attn_weight_dtype = (
            opt.attn_weight_dtype
        )  # recorded; the TP loaders hardcode bfp8 mixer weights (= policy)
        args.gdn_weight_dtype = opt.gdn_weight_dtype
        # single-device-only layouts off (layer.forward gates them on num_devices == 1 anyway)
        args.decode_norm_sharded = False
        args.attn_norm_out_sharded = False
        # TP policy hooks consumed by the shared modules
        args.prefill_norm_distributed = mc.prefill_norm_distributed  # Qwen35MoeModelArgs.is_distributed_norm(PREFILL)
        args.gdn_conv1d_max_channels = mc.gdn_conv1d_max_channels  # load_gdn_weights_tp conv channel blocks
        args.gdn_state_dtype = mc.gdn_state_dtype  # TPGatedDeltaNet.reset_state
        args.gdn_decode_high_precision = mc.gdn_decode_high_precision  # recurrent_gated_delta_rule_decode_ttnn
        args.kv_dtype = mc.kv_dtype  # TPAttention._sdpa_bf8 (fill cast + bf8 chunked SDPA)
        args.sdpa_decode_q_chunk = mc.sdpa_decode_q_chunk  # TPAttention.forward_decode SDPAProgramConfig
        args.sdpa_decode_k_chunk = mc.sdpa_decode_k_chunk
        # optional re-sweep of the 1D decode grids of the token mixers (interleaved weights, proj_1d_decode)
        grid_w = getattr(args, "decode_grid_w", 8)
        if mc.attn_qkv_decode_cores:
            args.attn_qkv_decode_1d_progcfg = tpc.create_matmul_1d_decode_progcfg(
                1, args.dim, args.attn_qkv_fused_dim_tp, mc.attn_qkv_decode_cores, grid_w=grid_w
            )
        if mc.attn_wo_decode_cores:
            args.attn_wo_decode_1d_progcfg = tpc.create_matmul_1d_decode_progcfg(
                1, args.attn_out_dim_tp, args.dim, mc.attn_wo_decode_cores, grid_w=grid_w
            )
        if mc.gdn_qkvz_decode_cores:
            args.gdn_qkvz_decode_1d_progcfg = tpc.create_matmul_1d_decode_progcfg(
                1, args.dim, args.gdn_qkvzab_dim_tp, mc.gdn_qkvz_decode_cores, grid_w=grid_w
            )
        if mc.gdn_out_decode_cores:
            args.gdn_out_decode_1d_progcfg = tpc.create_matmul_1d_decode_progcfg(
                1, args.gdn_value_dim_tp, args.dim, mc.gdn_out_decode_cores, grid_w=grid_w
            )

    # ---------------------------------------------------------------------------------------------
    # Contract numbers (doc/context_contract.json "multichip_decoder_stage4")
    def per_device_contract(self) -> dict:
        a = self.args
        kv_b = _BYTES_PER_ELEM.get(self.mc.kv_dtype, 2.0)
        st_b = _BYTES_PER_ELEM.get(self.mc.gdn_state_dtype, 2.0)
        conv_b = 2.0
        return {
            "mesh": list(self.mesh_device.shape),
            "layer_idx": self.layer_idx,
            "layer_type": self.layer_type,
            "hidden_local": self.hidden_local,
            "kv_heads_per_device": int(a.n_local_kv_heads),
            "kv_replication": bool(getattr(a, "kv_replication", False)),
            "kv_bytes_per_token_per_device_per_layer": int(round(2 * a.n_local_kv_heads * a.head_dim * kv_b)),
            "gdn_state_bytes_per_user_per_device_per_layer": int(
                round(a.gdn_nv_tp * a.gdn_dk * a.gdn_dv * st_b + a.gdn_conv_kernel_size * a.gdn_qkv_dim_tp * conv_b)
            ),
            "routed_expert_local_intermediate": a.moe_intermediate_size // self.num_devices,
            "config": self.mc.describe(),
        }

    # ---------------------------------------------------------------------------------------------
    # State allocation (setup-time)
    def allocate_kv_cache(self, *, max_batch_size, max_seq_len, block_size=64, dtype=None):
        """Paged KV cache, ONE ``[num_blocks, n_local_kv_heads, block_size, head_dim]`` tensor per device (replicated
        shape; each device fills/reads its own local KV head). Bound to ``TPAttention.set_paged_kv_cache``."""
        assert self.layer.is_full_attention, "only full_attention layers hold a KV cache"
        dtype = dtype or self.mc.kv_dtype
        blocks_per_seq = math.ceil(max_seq_len / block_size)
        num_blocks = max_batch_size * blocks_per_seq
        shape = (num_blocks, self.args.n_local_kv_heads, block_size, self.args.head_dim)
        # ttnn.zeros with device=<mesh> materialises the same zero buffer on every device (replicated placement).
        k = ttnn.zeros(shape, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=self.mesh_device)
        v = ttnn.zeros(shape, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=self.mesh_device)
        self.layer.attention.set_paged_kv_cache(k, v)
        self.kv_cache = {
            "k": k,
            "v": v,
            "block_size": block_size,
            "num_blocks": num_blocks,
            "blocks_per_seq": blocks_per_seq,
            "per_device_shape": shape,
            "kv_heads_per_device": self.args.n_local_kv_heads,
        }
        return self.kv_cache

    def allocate_gdn_state(self, *, batch_size: int):
        """Batched GDN decode buffers INSIDE ``TPGatedDeltaNet`` (as ``Qwen36Model._allocate_kv_caches_tp`` does):
        ``rec_state [B, Nv/tp, Dk, Dv]`` (``mc.gdn_state_dtype``), ``conv_states`` = K taps ``[1, B, qkv_dim/tp]``,
        ``conv_carry [1, K-1, qkv_dim/tp]`` (chunk continuation) and the persistent zero sources. ``_stable_state``
        makes every later prefill/decode update them IN PLACE (addresses fixed -> trace-safe)."""
        assert not self.layer.is_full_attention, "only linear_attention layers hold GDN state"
        dn = self.layer.attention
        self._free_gdn_buffers(dn)
        dn.B = int(batch_size)
        dn.reset_state()
        dn._stable_state = True
        dn._pending = []
        self.gdn_state = {
            "recurrent": dn.rec_state,
            "conv": dn.conv_states,
            "carry": dn.conv_carry,
            "batch_size": int(batch_size),
        }
        return self.gdn_state

    @staticmethod
    def _free_gdn_buffers(dn):
        """Release a previous allocation (``reset_state`` allocates fresh buffers and would leak the old ones)."""
        for name in ("rec_state", "conv_carry", "_zero_conv0", "_zero_conv_carry", "_zero_rec"):
            t = getattr(dn, name, None)
            if t is not None:
                ttnn.deallocate(t)
                setattr(dn, name, None)
        for t in getattr(dn, "conv_states", None) or []:
            ttnn.deallocate(t)
        dn.conv_states = None

    def reset_gdn_state(self):
        """Zero the recurrent / conv / carry buffers in place (new sequence; addresses unchanged)."""
        assert self.gdn_state is not None, "call allocate_gdn_state(batch_size=...) first"
        dn = self.layer.attention
        dn.reset_state_inplace()
        dn._pending = []

    # ---------------------------------------------------------------------------------------------
    def rope_decode_host(self, position_ids: torch.Tensor):
        """Host bf16 cos/sin ``[1, B, 1, rope_dim]`` for decode positions (rope_tp format consumed by
        ``apply_partial_rope_decode``; replicate to the mesh / copy into persistent replicated buffers).

        NOTE: the single-chip baseline returns ``[B, 1, rope_dim]``; the TP attention wants the batch on dim 1.
        """
        assert self.rope is not None
        pos = position_ids.reshape(-1).to(torch.long)
        rd = self.args.rope_head_dim
        cos = self.rope.cos_cpu[pos].reshape(1, -1, 1, rd)
        sin = self.rope.sin_cpu[pos].reshape(1, -1, 1, rd)
        return cos, sin

    def _as_4d(self, x, rows_first: bool):
        """Accept the 4D contract shape or a 3D convenience shape and return (``[1,1,N,dim/tp]`` view, original shape).

        ``rows_first``: a 3D input is ``[1, T, dim/tp]`` (prefill) when True, ``[B, 1, dim/tp]`` (decode) when False.
        """
        orig = list(x.shape)
        if len(orig) == 4:
            assert orig[0] == 1 and orig[1] == 1, f"expected [1,1,N,{self.hidden_local}], got {orig}"
            x4 = x
        elif len(orig) == 3:
            n = orig[1] if rows_first else orig[0]
            assert (orig[0] if rows_first else orig[1]) == 1, f"3D input must be [1,T,dim/tp] or [B,1,dim/tp]: {orig}"
            x4 = ttnn.reshape(x, (1, 1, n, orig[2]))
        else:
            raise ValueError(f"unsupported input rank {len(orig)}: {orig}")
        assert x4.shape[-1] == self.hidden_local, (
            f"input must be FRACTURED on the hidden dim: expected {self.hidden_local} columns per device, got "
            f"{x4.shape[-1]} (shard the full-dim tensor with ShardTensorToMesh(dim=-1))"
        )
        return x4, orig

    # ---------------------------------------------------------------------------------------------
    def prefill_forward(self, hidden_states, *, start_pos: int = 0, page_table=None, user_id: int = 0, valid_len=None):
        x, orig = self._as_4d(hidden_states, rows_first=True)
        T = x.shape[2]
        T_logical = T if valid_len is None else int(valid_len)
        T_pad = -(-T // GDN_CHUNK) * GDN_CHUNK
        if T_pad != T:
            x = ttnn.pad(x, [(0, 0), (0, 0), (0, T_pad - T), (0, 0)], value=0.0)
        # Chunk-continuation contract (as the baseline): start_pos is a multiple of the GDN chunk and the KV block size;
        # the chunked SDPA derives its q/k chunk from the lowest set bit of chunk_start_idx.
        assert start_pos % GDN_CHUNK == 0, f"start_pos {start_pos} must be a multiple of {GDN_CHUNK}"
        if self.layer.is_full_attention:
            assert valid_len is None, "full_attention prefill takes the logical length from the tensor (no valid_len)"
            out = self._attention_prefill(x, T, T_pad, start_pos, page_table, user_id)
        else:
            out = self._gdn_prefill(x, T_logical if T_pad != T_logical else None, user_id, start_pos)
        if T_pad != T:
            ttnn.deallocate(x)
            out_full = out
            out = ttnn.slice(out_full, [0, 0, 0, 0], [1, 1, T, self.hidden_local])
            ttnn.deallocate(out_full)
        if len(orig) != 4:
            out = ttnn.reshape(out, orig)
        return out

    def _attention_prefill(self, x, T, T_pad, start_pos, page_table, user_id):
        """Paged TP attention prefill of one user: cos/sin ``[1,1,T_pad,rope_dim]`` sliced from the resident replicated
        tables, a ONE-ROW page table for ``user_id`` (``TPAttention.forward_prefill_paged`` fills ``batch_idx=0`` of the
        page table it is given, so the row selects the user) and the chunk page table starting at this chunk's block."""
        assert page_table is not None and self.kv_cache is not None, "full_attention prefill needs a paged KV cache"
        bs = self.kv_cache["block_size"]
        assert start_pos % bs == 0, f"start_pos {start_pos} must be a multiple of the KV block size {bs}"
        rd = self.args.rope_head_dim
        cos3 = ttnn.slice(self.rope.cos_device, [0, start_pos, 0], [1, start_pos + T_pad, rd])
        sin3 = ttnn.slice(self.rope.sin_device, [0, start_pos, 0], [1, start_pos + T_pad, rd])
        cos = ttnn.reshape(cos3, (1, 1, T_pad, rd))  # rope_tp prefill format [1,1,S,rope_dim]
        sin = ttnn.reshape(sin3, (1, 1, T_pad, rd))
        # K/V are written for the LOGICAL tokens' blocks only (pad rows inside the last block are harmless and are
        # overwritten by decode); never beyond the user's page-table row.
        n_blocks = -(-(start_pos + T) // bs)
        assert (
            n_blocks <= page_table.shape[1]
        ), f"{start_pos + T} tokens need {n_blocks} blocks > page table width {page_table.shape[1]}"
        pt_row = (
            page_table
            if page_table.shape[0] == 1
            else ttnn.slice(page_table, [user_id, 0], [user_id + 1, page_table.shape[1]])
        )
        # paged_fill_cache writes input tile i into chunk_page_table[i // block_size]: the chunk's table must START at
        # this chunk's first block (start_pos // bs). forward_prefill_paged slices K/V to the table's length, so only the
        # logical tokens' blocks are written; the chunked SDPA reads the (32-block padded) full row.
        chunk_pt = ttnn.slice(pt_row, [0, start_pos // bs], [1, n_blocks])
        out = self.layer.forward(
            x,
            cos=cos,
            sin=sin,
            mode="prefill",
            page_table=pt_row,
            chunk_page_table=chunk_pt,
            chunk_start_idx=start_pos,
        )
        for t in (cos, sin, chunk_pt):
            ttnn.deallocate(t)
        if pt_row is not page_table:
            ttnn.deallocate(pt_row)
        return out

    def _gdn_prefill(self, x, valid_len, user_id, start_pos):
        """Linear-attention prefill of ONE user into the batched TP state buffers.

        ``B == 1``: the ``_stable_state`` carry path of ``TPGatedDeltaNet.forward_prefill`` (``capture_state=True``): the
        chunk kernel starts from ``rec_state`` / ``conv_carry`` and copies the new state back IN PLACE. A fresh sequence
        zeroes them first (``reset_state_inplace``); a continuation (``start_pos > 0``) first rebuilds ``conv_carry``
        from the decode shift register ``conv_states[1..K-1]`` (decode advances only the taps; right after a prefill both
        are already in sync, so the copy is a value no-op there) -- the TP analogue of the baseline's fused-window rebuild.
        ``B > 1``: from-scratch per-user prefill (``gdn_collect=True`` -> ``forward_prefill_collect`` stashes
        ``(rec, conv_window)``), written into decode slot ``user_id`` with ``write_slot`` (the other rows are preserved).
        """
        dn = self.layer.attention
        st = self.gdn_state
        assert st is not None, "call allocate_gdn_state(batch_size=...) before prefill"
        B = st["batch_size"]
        assert 0 <= user_id < B, (user_id, B)
        if B == 1:
            if start_pos == 0:
                dn.reset_state_inplace()
            else:
                self._rebuild_conv_carry(dn)
            return self.layer.forward(x, mode="prefill", chunk_size=GDN_CHUNK, valid_len=valid_len)
        assert start_pos == 0, "chunked prefill continuation is supported for single-user GDN state (B == 1)"
        out = self.layer.forward(x, mode="prefill", chunk_size=GDN_CHUNK, valid_len=valid_len, gdn_collect=True)
        assert len(dn._pending) == 1, f"expected one collected per-user state, got {len(dn._pending)}"
        rec, conv_window = dn._pending.pop()  # rec [1, Nv/tp, Dk, Dv]; conv_window [1, K-1, qkv_dim/tp] (oldest first)
        taps = self._conv_taps_from_window(dn, conv_window)
        ttnn.deallocate(conv_window)
        dn.write_slot(user_id, rec, taps)  # consumes rec and taps
        return out

    @staticmethod
    def _rebuild_conv_carry(dn):
        """``conv_carry[0, j] = conv_states[j + 1][0, 0]`` for ``j < K-1`` (B == 1): the last K-1 conv inputs, oldest first."""
        window = ttnn.concat(list(dn.conv_states[1:]), dim=1)  # K-1 x [1,1,D] -> [1,K-1,D]
        if window.layout != dn.conv_carry.layout:
            window = ttnn.to_layout(window, dn.conv_carry.layout)
        ttnn.copy(window, dn.conv_carry)
        ttnn.deallocate(window)

    @staticmethod
    def _conv_taps_from_window(dn, conv_window):
        """``[1, K-1, D]`` last-inputs window -> the K ``[1, 1, D]`` shift-register taps ``write_slot`` wants:
        tap 0 = the shifted-out zero, tap m = window[m-1] (``assemble_batched_state`` convention)."""
        D = dn.qkv_dim_tp
        zero = ttnn.slice(dn._zero_conv0, (0, 0, 0), (1, 1, D))  # [1,B,D] zero source -> one row
        taps = [zero]
        for m in range(1, dn.K):
            taps.append(ttnn.reshape(ttnn.slice(conv_window, (0, m - 1, 0), (1, m, D)), (1, 1, D)))
        return taps

    # ---------------------------------------------------------------------------------------------
    def decode_forward(self, hidden_states, *, current_pos, page_table=None, cos=None, sin=None):
        x, orig = self._as_4d(hidden_states, rows_first=False)
        if self.layer.is_full_attention:
            assert page_table is not None and cos is not None and sin is not None
            out = self.layer.forward(x, cos, sin, mode="decode", position_tensor=current_pos, page_table=page_table)
        else:
            out = self.layer.forward(x, mode="decode")
        if len(orig) != 4:
            out = ttnn.reshape(out, orig)
        return out
