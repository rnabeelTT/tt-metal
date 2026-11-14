# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Inkling decoder layer on the qwen35 autoport framework (TP over a (1, N) mesh, fractured residual [1, 1, T, H/N]):

    a = attention(attention_norm(x))          tt/attention.InklingAttention -> reduce-scatter -> fractured
    h = x + attn_sconv(a)                      fp32 short conv on the die's hidden slice (depthwise: no collective)
    f = feed_forward(ffn_norm(h))              dense MLP (layers 0-1) or the Inkling MoE (tt/optimized_decoder.py)
    out = h + mlp_sconv(f)

Same contract as the qwen36 layer (``forward(x, cos, sin, mode, position_tensor, page_table, ...)``; cos / sin are
ignored: no RoPE), the norms and the feed-forward built by the shared code (so the hybrid overlay's MoE patches apply).
Implemented: one-row decode. Prefill / spec-verify rows come next.
"""

from __future__ import annotations

import torch

import ttnn
from models.autoports.minimaxai_minimax_m2_7.tt.layer import Qwen35MoeDecoderLayer as _M27Layer
from models.autoports.thinkingmachines_inkling_small.tt.attention import InklingAttention
from models.autoports.thinkingmachines_inkling_small.tt.sconv import ShortConv
from models.demos.blackhole.qwen36.utils.substate import substate
from models.tt_transformers.tt.common import Mode


class _HFConfigView:
    """The fields InklingAttention reads, from the framework args (args.inkling = the raw text config)."""

    def __init__(self, args):
        r = args.inkling
        L = args.n_layers_total if hasattr(args, "n_layers_total") else len(args.attention_type_list)
        local = set(r["local_layer_ids"])
        self.layer_types = ["hybrid_sliding" if i in local else "hybrid" for i in range(max(L, 64))]
        for k in ("hidden_size", "head_dim", "num_attention_heads", "num_key_value_heads", "swa_head_dim",
                  "swa_num_attention_heads", "swa_num_key_value_heads", "d_rel", "rel_extent", "sliding_window_size",
                  "log_scaling_n_floor", "log_scaling_alpha", "rms_norm_eps"):
            setattr(self, k, r[k])


class InklingDenseMLP:
    """Layers 0-1: SwiGLU (I 16384, global_scale folded into down_proj by the state dict), TP: gate/up column-parallel,
    down row-parallel + reduce-scatter (fractured output, like the MoE). bfp8 weights, HiFi4 / fp32 accumulation: two
    layers only, so precision over the shared Qwen36MLP's bfp4 gate/up."""

    def __init__(self, mesh, args, sd, tt_ccl, dtype=ttnn.bfloat8_b):
        self.mesh, self.args, self.tt_ccl = mesh, args, tt_ccl
        self.N = mesh.get_num_devices()
        col, row = ttnn.ShardTensorToMesh(mesh, dim=-1), ttnn.ShardTensorToMesh(mesh, dim=-2)
        up = lambda t, m: ttnn.as_tensor(t.to(torch.bfloat16).T.contiguous(), dtype=dtype, layout=ttnn.TILE_LAYOUT,  # noqa
                                         device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=m)
        self.w1 = up(sd["gate_proj.weight"], col)
        self.w3 = up(sd["up_proj.weight"], col)
        self.w2 = up(sd["down_proj.weight"], row)
        self.ckc = ttnn.init_device_compute_kernel_config(mesh.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
                                                          math_approx_mode=False, fp32_dest_acc_en=True,
                                                          packer_l1_acc=True)

    def forward(self, x, mode="decode"):
        if x.is_sharded():
            x = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG)
        mc = ttnn.L1_MEMORY_CONFIG if int(x.shape[-2]) <= 32 else ttnn.DRAM_MEMORY_CONFIG
        g = ttnn.linear(x, self.w1, compute_kernel_config=self.ckc, memory_config=mc)
        u = ttnn.linear(x, self.w3, compute_kernel_config=self.ckc, memory_config=mc)
        h = ttnn.multiply(g, u, input_tensor_a_activations=[ttnn.UnaryOpType.SILU], memory_config=mc)
        ttnn.deallocate(g)
        ttnn.deallocate(u)
        part = ttnn.linear(h, self.w2, compute_kernel_config=self.ckc, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(h)
        if self.N == 1:
            return part
        from models.tt_transformers.tt.ccl import tt_all_reduce

        return tt_all_reduce(part, self.mesh, self.tt_ccl, cluster_axis=0, dim=3, topology=self.args.ccl_topology(),
                             memory_config=ttnn.DRAM_MEMORY_CONFIG)


class InklingDecoderLayer(_M27Layer):
    def __init__(self, mesh_device, args, state_dict, layer_num, tensor_cache_path=None, tt_ccl=None):
        # not Qwen36DecoderLayer.__init__: it would build the qwen attention; the parts kept are rebuilt here
        self.layer_num = layer_num
        self.device = mesh_device
        self.args = args
        self.tt_ccl = tt_ccl
        self.num_devices = getattr(args, "num_devices", 1)
        self._tp_modules = True
        self.is_full_attention = True  # every layer is attention (the model loop passes position / page table)
        self._decode_replicated = False
        self._fuse_norm_agmm = False
        self._fuse_ff_agmm = False
        self._addnorm = None
        self.attention_norm = self._make_norm(mesh_device, args, state_dict, layer_num, "input_layernorm",
                                              tensor_cache_path, tt_ccl, "attention_norm")
        self.ffn_norm = self._make_norm(mesh_device, args, state_dict, layer_num, "post_attention_layernorm",
                                        tensor_cache_path, tt_ccl, "ff_norm")
        attn_sd = substate(state_dict, f"layers.{layer_num}.self_attn")
        self.attention = InklingAttention(
            mesh_device, _HFConfigView(args), {k: attn_sd[k] for k in attn_sd}, layer_num, tt_ccl=tt_ccl,
            block_size=int(getattr(args, "kv_block_size", 64)), max_seq_len=int(args.max_seq_len),
            weight_dtype=getattr(args, "attn_weight_dtype", None) or ttnn.bfloat8_b,
            kv_dtype=getattr(args, "inkling_kv_dtype", None) or getattr(args, "kv_dtype", None) or ttnn.bfloat8_b,
            own_global_cache=bool(getattr(args, "inkling_own_kv", True)),
        )
        H, N = args.dim, self.num_devices
        shard = ttnn.ShardTensorToMesh(mesh_device, dim=-1) if N > 1 else None
        self.attn_sconv = ShortConv(mesh_device, state_dict[f"layers.{layer_num}.attn_sconv.weight"].reshape(H, -1),
                                    shard, H // N)
        self.mlp_sconv = ShortConv(mesh_device, state_dict[f"layers.{layer_num}.mlp_sconv.weight"].reshape(H, -1),
                                   shard, H // N)
        self.feed_forward = self._build_feed_forward(mesh_device, args, state_dict, layer_num, tensor_cache_path,
                                                     tt_ccl)

    def _build_feed_forward(self, mesh_device, args, state_dict, layer_num, tensor_cache_path, tt_ccl):
        if not args.is_moe_layer(layer_num):  # dense SwiGLU (global_scale folded into down_proj by the state dict)
            return InklingDenseMLP(mesh_device, args, substate(state_dict, f"layers.{layer_num}.mlp"), tt_ccl)
        from models.autoports.thinkingmachines_inkling_small.tt.optimized_decoder import Qwen35MoEOptimized

        mlp_state = substate(state_dict, f"layers.{layer_num}.mlp")
        mlp_cache = (tensor_cache_path / f"layers.{layer_num}") if tensor_cache_path else None
        return Qwen35MoEOptimized(mesh_device, args, mlp_state, mlp_cache, tt_ccl=tt_ccl, layer_num=layer_num)

    def reset(self):
        self.attention.reset()
        self.attn_sconv.reset()
        self.mlp_sconv.reset()

    def forward(self, x, cos=None, sin=None, mode="decode", chunk_size=128, position_tensor=None, page_table=None,
                chunk_page_table=None, chunk_start_idx=None, chunk_start_idx_tensor=None, valid_len=None,
                gdn_collect=False, tau=None):
        if mode == "rows":
            return self.forward_rows(x, position_tensor, page_table, tau=tau)
        if mode != "decode":
            raise NotImplementedError("Inkling prefill: next step (tt/attn prefill kernel); use mode='rows' groups")
        norm_cfg = self.args.get_norm_config("attn", Mode.DECODE) if self.num_devices > 1 else None
        attn_in = self.attention_norm(x, mode=Mode.DECODE, norm_config=norm_cfg)
        if attn_in.is_sharded():
            attn_in = ttnn.sharded_to_interleaved(attn_in, ttnn.L1_MEMORY_CONFIG)
        a = self.attention.forward_decode(attn_in, position_tensor, page_table, tau=tau)
        ttnn.deallocate(attn_in)
        a2 = self.attn_sconv.decode(a)
        ttnn.deallocate(a)
        h = ttnn.add(x, a2)
        ttnn.deallocate(a2)
        ff_in = self.ffn_norm(h, mode=Mode.DECODE, norm_config=norm_cfg)
        f = self.feed_forward.forward(ff_in, mode="decode")
        ttnn.deallocate(ff_in)
        f2 = self.mlp_sconv.decode(f)
        ttnn.deallocate(f)
        out = ttnn.add(h, f2)
        ttnn.deallocate(h)
        ttnn.deallocate(f2)
        return out

    def forward_rows(self, x, positions, page_table, tau=None, commit_all=True):
        """R <= 8 rows at consecutive positions (spec verify / prefill row groups): x [1, 1, R, H/N] fractured,
        positions [R] int32. commit_all=False leaves the 4 conv states for commit_rows(n) (verify)."""
        R = int(x.shape[2])
        norm_cfg = self.args.get_norm_config("attn", Mode.DECODE) if self.num_devices > 1 else None
        attn_in = self.attention_norm(x, mode=Mode.DECODE, norm_config=norm_cfg)
        if attn_in.is_sharded():
            attn_in = ttnn.sharded_to_interleaved(attn_in, ttnn.L1_MEMORY_CONFIG)
        a = self.attention.forward_rows(attn_in, positions, page_table, tau=tau, commit_all=commit_all)
        ttnn.deallocate(attn_in)
        a2 = self.attn_sconv.rows(a, commit_all=commit_all)
        ttnn.deallocate(a)
        h = ttnn.add(x, a2)
        ttnn.deallocate(a2)
        ff_in = self.ffn_norm(h, mode=Mode.DECODE, norm_config=norm_cfg)
        f = self.feed_forward.forward(ff_in, mode="decode")
        ttnn.deallocate(ff_in)
        f2 = self.mlp_sconv.rows(f, commit_all=commit_all)
        ttnn.deallocate(f)
        out = ttnn.add(h, f2)
        ttnn.deallocate(h)
        ttnn.deallocate(f2)
        return out

    def commit_rows(self, n: int):
        """Spec verify: every conv state advances past the first n rows of the last forward_rows."""
        self.attention.commit_rows(n)
        self.attn_sconv.commit_rows(n)
        self.mlp_sconv.commit_rows(n)

    def forward_prefill(self, x, p0: int, page_table_host=None):
        """A prefill chunk (T rows, T % 64 == 0, p0 % block == 0): x [1, 1, T, H/N] fractured -> same. One attention
        launch for the chunk (tt/attention.forward_prefill), the MoE's prefill path, every conv state past all T rows."""
        mc = Mode.PREFILL
        norm_cfg = self.args.get_norm_config("attn", mc) if self.num_devices > 1 else None
        attn_in = self.attention_norm(x, mode=mc, norm_config=norm_cfg)
        if attn_in.is_sharded():
            attn_in = ttnn.sharded_to_interleaved(attn_in, ttnn.DRAM_MEMORY_CONFIG)
        a = self.attention.forward_prefill(attn_in, p0, page_table_host)
        ttnn.deallocate(attn_in)
        a2 = self.attn_sconv.rows(a, commit_all=True)
        ttnn.deallocate(a)
        h = ttnn.add(x, a2)
        ttnn.deallocate(a2)
        ff_cfg = self.args.get_norm_config("ff", mc) if self.num_devices > 1 else None
        ff_in = self.ffn_norm(h, mode=mc, norm_config=ff_cfg)
        if ff_in.is_sharded():
            ff_in = ttnn.sharded_to_interleaved(ff_in, ttnn.DRAM_MEMORY_CONFIG)
        f = self.feed_forward.forward(ff_in, mode="prefill")
        ttnn.deallocate(ff_in)
        f2 = self.mlp_sconv.rows(f, commit_all=True)
        ttnn.deallocate(f)
        out = ttnn.add(h, f2)
        ttnn.deallocate(h)
        ttnn.deallocate(f2)
        return out
