# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Multi-token-prediction (MTP) head of Qwen3.5-122B-A10B on the mesh (perf pass, MTP turn; plan: doc/perf_pass/mtp_plan.md).

    h0 = fc(cat(norm_e(embed(tok)), norm_h(hidden)))   hidden = the main model's FINAL-NORMED state of the extended token
    h1 = layer(h0, pos)                                one full-attention MoE decoder layer, its own paged KV cache
    logits = lm_head(norm(h1))                         embeddings / LM head shared with the main model

The layer is the autoport's optimized multichip decoder layer built as virtual layer index ``n_layers`` (the args'
``attention_type_list`` gains one ``full_attention`` entry; the checkpoint's per-expert MTP tensors are fused by
``StreamingStateDict.mtp_state``). Layouts follow the decode path: token ids in the model's ``[1,1,1,S]`` buffer,
fractured residual into the layer, replicated full-dim vectors into the fc and the LM head (vocab-sharded logits out).
"""
from __future__ import annotations

import copy
import os

import torch

import ttnn
from models.common.rmsnorm import RMSNorm
from models.tt_transformers.tt.common import Mode
from models.tt_transformers.tt.distributed_norm import DistributedNorm


class Qwen35MtpHead:
    def __init__(self, model, state_dict, tensor_cache_path=None, weight_dtype=ttnn.bfloat8_b):
        self.model = model
        self.device = model.mesh_device
        args = model.args
        self.args = args
        self.layer_idx = len(args.attention_type_list)
        args.attention_type_list = list(args.attention_type_list) + ["full_attention"]  # is_full_attention_layer(L)
        st = state_dict.mtp_state(fuse_experts=True)
        prefix = "mtp.layers.0."
        layer_state = {f"layers.{self.layer_idx}." + k[len(prefix) :]: v for k, v in st.items() if k.startswith(prefix)}
        layer_cls = type(model.layers[0])
        # The MTP layer is more sensitive to the shipped bfp4 token-mixer weights than the main attention layers (hidden PCC
        # vs the torch reference 0.95 with bfp4 mixers, 0.99 with bfp8; the routed experts' dtype made no difference), and
        # its attention projections are ~20 MB per die: build it on an args copy with bfp8 attention weights at HiFi2.
        layer_args = copy.copy(args)
        attn_dtype = os.environ.get("QWEN35_MTP_ATTN_DTYPE", "bfp8")
        if attn_dtype == "bfp8":
            from models.demos.blackhole.qwen36.tt import tp_common as tpc

            layer_args.attn_weight_dtype = ttnn.bfloat8_b
            layer_args.mixer_compute_cfg = tpc.COMPUTE_HIFI2
        self.layer = layer_cls(
            self.device, layer_args, layer_state, self.layer_idx, tensor_cache_path, tt_ccl=model.tt_ccl
        )
        assert self.layer.is_full_attention
        del layer_state
        tp = getattr(args, "num_devices", 1) > 1
        rep = ttnn.ReplicateTensorToMesh(self.device) if tp else None
        H = args.dim
        # fc [H, 2H] -> device [2H, H], output columns fractured across the mesh (the layer's residual layout)
        fc = st["mtp.fc.weight"].T.contiguous()
        self.fc_weight = ttnn.as_tensor(
            fc,
            dtype=weight_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(self.device, dim=-1) if tp else None,
            cache_file_name=(tensor_cache_path / "mtp.fc") if tensor_cache_path else None,
        )
        norm_state = {k: v for k, v in st.items() if k.endswith("norm.weight") or "_norm_" in k}

        def make_norm(key, distributed):
            norm = RMSNorm(
                device=self.device,
                dim=H,
                state_dict=norm_state,
                weight_key=key,
                state_dict_prefix="mtp.",
                weight_cache_path=tensor_cache_path,
                weight_dtype=ttnn.bfloat16,
                add_unit_offset=True,
                eps=args.norm_eps,
                **(
                    dict(is_distributed=args.is_distributed_norm, ccl_topology=args.ccl_topology(), tt_ccl=model.tt_ccl)
                    if tp and distributed
                    else {}
                ),
            )
            if tp and distributed:
                norm = DistributedNorm(norm, args, tt_ccl=model.tt_ccl, TG=args.is_galaxy)
            return norm

        # embedding output is fractured -> gather-then-norm; the final-normed hidden is already replicated -> plain norm
        self.pre_fc_norm_embedding = make_norm("pre_fc_norm_embedding", distributed=True)
        self.pre_fc_norm_hidden = make_norm("pre_fc_norm_hidden", distributed=False)
        self.norm = make_norm("norm", distributed=True)
        self._norm_cfg = None
        del st

    # ---- caches ---------------------------------------------------------------------------------------------------
    def allocate_kv_cache(self, kv_cache_shape, dtype=ttnn.bfloat16):
        """Paged KV cache of the MTP layer (same shape / dtype as the main attention layers, replicated per device)."""
        caches = []
        for _ in range(2):
            caches.append(
                ttnn.from_torch(
                    torch.zeros(kv_cache_shape, dtype=torch.bfloat16),
                    dtype=dtype,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(self.device)
                    if getattr(self.args, "num_devices", 1) > 1
                    else None,
                )
            )
        self.layer.attention.set_paged_kv_cache(*caches)
        self.kv_cache = caches
        return caches

    # ---- forward ----------------------------------------------------------------------------------------------------
    def mix(self, tokens, hidden):
        """tokens: the model's decode token buffer ([1,1,1,S] uint32) or a [1,B] row; hidden: replicated [1,1,B,H] bf16
        (the main model's final-normed hidden of the token being extended). Returns the layer input, fractured [1,1,B,H/TP].
        """
        B = hidden.shape[2]
        tok = self.model._decode_tokens_view(tokens, B)
        e = self.model.embd(tok)  # [1, B, H/TP] (TP) or [1, B, H]
        e = ttnn.reshape(e, (1, 1, B, e.shape[-1]))
        e = ttnn.to_memory_config(e, ttnn.DRAM_MEMORY_CONFIG)
        e_n = self.pre_fc_norm_embedding(e, mode=Mode.DECODE)  # replicated [1,1,B,H]
        ttnn.deallocate(e)
        h_n = self.pre_fc_norm_hidden(hidden, mode=Mode.DECODE)
        x = ttnn.concat([e_n, h_n], dim=-1)  # [1,1,B,2H] replicated
        ttnn.deallocate(e_n)
        ttnn.deallocate(h_n)
        out = ttnn.linear(x, self.fc_weight, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16)
        ttnn.deallocate(x)
        return out  # [1,1,B,H/TP]

    def decode_step(self, tokens, hidden, cur_pos, rot_idxs, page_table):
        """One MTP row per user (decode layout): returns (vocab-sharded logits [1,1,B,V/TP], normed MTP hidden [1,1,B,H])."""
        m = self.model
        cos, sin = m._rope_from_idxs(rot_idxs)
        x = self.mix(tokens, hidden)
        x = self.layer.forward(x, cos, sin, position_tensor=cur_pos, page_table=page_table, mode="decode")
        if getattr(self.args, "num_devices", 1) > 1:
            nc = dict(self.args.get_norm_config("lm_head", Mode.DECODE))
            nc["output_mem_config"] = ttnn.DRAM_MEMORY_CONFIG
            h = self.norm(x, mode=Mode.DECODE, norm_config=nc)
        else:
            h = self.norm(x, mode=Mode.DECODE)
        ttnn.deallocate(x)
        logits = ttnn.linear(h, m.lm_head_weight, compute_kernel_config=m._lm_head_ckc())
        return logits, h
