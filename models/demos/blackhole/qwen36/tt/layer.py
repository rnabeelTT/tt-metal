# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Hybrid TransformerBlock for Qwen3.5-9B.

Dispatches to either Gated DeltaNet (linear attention) or Gated Full Attention
based on the layer index. Both share the same RMSNorm + residual pattern and MLP.
"""

import math
import os

import ttnn
from models.common.rmsnorm import RMSNorm
from models.demos.blackhole.qwen36.tt.attention import AttentionConfig, Qwen36GatedAttention
from models.demos.blackhole.qwen36.tt.gdn import GDNConfig, Qwen36GatedDeltaNet
from models.demos.blackhole.qwen36.tt.mlp import Qwen36MLP
from models.demos.blackhole.qwen36.utils.substate import substate
from models.tt_transformers.tt.common import Mode


class Qwen36DecoderLayer:
    """Single transformer layer with hybrid attention dispatch.

    Pattern: x → attention_norm → attention → residual → ff_norm → MLP → residual
    Attention is either GatedAttention (full, with RoPE) or GatedDeltaNet (linear).
    """

    def __init__(self, mesh_device, args, state_dict, layer_num, tensor_cache_path=None, tt_ccl=None):
        self.layer_num = layer_num
        self.device = mesh_device
        self.args = args
        self.tt_ccl = tt_ccl
        self.num_devices = getattr(args, "num_devices", 1)
        # TP attention / GDN modules: on a multi-device mesh, or on one device when the args ask for them
        # (tp_modules_single_device: the TP modules at degree 1, CCLs are no-ops; used by the autoports' single-die serving)
        self._tp_modules = self.num_devices > 1 or bool(getattr(args, "tp_modules_single_device", False))
        self.is_full_attention = args.is_full_attention_layer(layer_num)
        # args.decode_replicated (TP): decode keeps the residual replicated, one all-reduce per boundary
        # (decode_boundary.py); built here so its buffers / semaphores exist before any trace capture
        self._decode_replicated = self.num_devices > 1 and bool(getattr(args, "decode_replicated", False))
        if self._decode_replicated:
            from models.demos.blackhole.qwen36.tt.decode_boundary import decode_boundary

            decode_boundary(args, mesh_device)
            if getattr(args, "fabric_boundary", False):
                # the fabric boundary's persistent receive buffers too: allocated lazily (first decode, between the
                # prefill warmup and its trace capture) they shifted the prefill MoE's L1 programs out of the cache
                from models.demos.blackhole.qwen36.tt.fabric_boundary.ops import fabric_boundary

                fabric_boundary(args, mesh_device)

        prefix = f"layers.{layer_num}"

        # Zero-centered RMSNorm (Qwen3.5): output = x_normed * (1 + weight). The
        # framework RMSNorm applies the +1 internally via add_unit_offset=True and
        # is mesh-aware (replicates the weight across a MeshDevice).
        #
        # Single device: plain RMSNorm on the full hidden state (validated path).
        # TP (27B on a (1,4) mesh): the residual stream is fractured along the
        # hidden dim, so each norm is wrapped in the framework DistributedNorm,
        # which all-gathers (PREFILL: distributed rmsnorm + gather; DECODE:
        # gather-then-norm) to hand the modules a replicated full-dim input —
        # exactly as models/demos/qwen35_27b does via the framework decoder.
        # Prefill fuses the norm all-gather into the in-proj matmul (all_gather_minimal_matmul_async):
        # GDN qkvzab and full-attn QKV. attention_norm then skips its post-norm AG (prefill only;
        # decode gathers pre-norm). Gates must match the module-side _fuse_agmm gates.
        # The fusion also needs the norm to leave its output fractured, which only the distributed-norm path does
        # (dim <= 4096 on a 1D mesh gathers BEFORE the norm instead) — tp_common.prefill_norm_agmm_enabled.
        from models.demos.blackhole.qwen36.tt import tp_common as tpc

        self._fuse_norm_agmm = tpc.prefill_norm_agmm_enabled(args) and (
            (not self.is_full_attention and getattr(args, "gdn_qkvz_weight_memcfg", None) is not None)
            or (self.is_full_attention and getattr(args, "attn_qkv_fused_weight_memcfg", None) is not None)
        )
        self.attention_norm = self._make_norm(
            mesh_device,
            args,
            state_dict,
            layer_num,
            "input_layernorm",
            tensor_cache_path,
            tt_ccl,
            "attention_norm",
            enable_all_gather=not self._fuse_norm_agmm,
        )
        # Prefill: ff_norm skips AG (fused into gate/up AGMM); decode gathers pre-norm so this is a no-op there.
        self._fuse_ff_agmm = tpc.mlp_gateup_agmm_enabled(self.num_devices, args)
        self.ffn_norm = self._make_norm(
            mesh_device,
            args,
            state_dict,
            layer_num,
            "post_attention_layernorm",
            tensor_cache_path,
            tt_ccl,
            "ff_norm",
            enable_all_gather=not self._fuse_ff_agmm,
        )

        if self._tp_modules:
            # Tensor-parallel modules (sharded weights from the raw substate).
            # Cache the sharded mesh weights to disk so re-runs skip the (slow,
            # single-threaded) reorder+shard of the full 27B.
            tp_cache = (tensor_cache_path / f"layers.{layer_num}" / "tp") if tensor_cache_path else None
            if self.is_full_attention:
                from models.demos.blackhole.qwen36.tt.attention.tp import TPAttention, load_attention_weights_tp

                tw = load_attention_weights_tp(
                    mesh_device, substate(state_dict, f"layers.{layer_num}.self_attn"), args, cache_dir=tp_cache
                )
                self.attention = TPAttention(mesh_device, args, tw, tt_ccl)
            else:
                from models.demos.blackhole.qwen36.tt.gdn.tp import TPGatedDeltaNet, load_gdn_weights_tp

                tw = load_gdn_weights_tp(
                    mesh_device, substate(state_dict, f"layers.{layer_num}.linear_attn"), args, cache_dir=tp_cache
                )
                self.attention = TPGatedDeltaNet(mesh_device, args, tw, tt_ccl)
        elif self.is_full_attention:
            attn_state = substate(state_dict, f"layers.{layer_num}.self_attn")
            attn_cache = (tensor_cache_path / f"layers.{layer_num}") if tensor_cache_path else None
            self.attention = Qwen36GatedAttention(mesh_device, AttentionConfig.from_args(args), attn_state, attn_cache)
        else:
            gdn_state = substate(state_dict, f"layers.{layer_num}.linear_attn")
            gdn_cache = (tensor_cache_path / f"layers.{layer_num}") if tensor_cache_path else None
            self.attention = Qwen36GatedDeltaNet(mesh_device, GDNConfig.from_args(args), gdn_state, gdn_cache)

        self.feed_forward = self._build_feed_forward(
            mesh_device, args, state_dict, layer_num, tensor_cache_path, tt_ccl
        )
        # Decode add + RMSNorm op (QWEN35_ADDNORM_DECODE=1, addnorm/ops.py) on the replicated-residual boundaries: the
        # attention norm and residual + all-reduce + ffn norm each in one program (sharded norm + add + the mixer's
        # sharded->interleaved copy before). Built here (gamma tensors) so nothing allocates during trace capture.
        self._addnorm = None
        if self._decode_replicated and os.environ.get("QWEN35_ADDNORM_DECODE", "0") == "1":
            self._addnorm_setup()

    def _addnorm_setup(self):
        from models.demos.blackhole.qwen36.tt.addnorm.ops import AddNorm
        from models.demos.blackhole.qwen36.tt.decode_boundary import CORES

        op = self.args.__dict__.get("_qwen36_addnorm")
        if op is None:
            op = AddNorm(self.device, self.args.dim, cores=CORES, eps=self.args.norm_eps)
            self.args._qwen36_addnorm = op
        self._addnorm = op
        self._g_attn = op.prepare(self.attention_norm.norm.weight)
        self._g_ffn = op.prepare(self.ffn_norm.norm.weight)

    def _build_feed_forward(self, mesh_device, args, state_dict, layer_num, tensor_cache_path, tt_ccl):
        """Build this layer's feed-forward block.

        Hook for checkpoints whose FF is not a dense SwiGLU (Qwen3.5-MoE autoports replace
        the MLP with a routed-expert block). The returned object must expose
        ``forward(x, mode=...)`` and keep the layer's residual-layout contract.
        """
        mlp_state = substate(state_dict, f"layers.{layer_num}.mlp")
        mlp_cache = (tensor_cache_path / f"layers.{layer_num}") if tensor_cache_path else None
        return Qwen36MLP(mesh_device, mlp_state, mlp_cache, args=args, tt_ccl=tt_ccl)

    def _make_norm(
        self,
        mesh_device,
        args,
        state_dict,
        layer_num,
        weight_key,
        tensor_cache_path,
        tt_ccl,
        ag_key,
        enable_all_gather=True,
    ):
        """Build the per-layer RMSNorm; wrap in DistributedNorm when TP>1.

        On a single device this returns the same plain RMSNorm the validated 9B
        path used. The DistributedNorm wrapper (TP>1) mirrors tt_transformers
        decoder.py and handles the fractured->replicated transition.
        """
        norm = RMSNorm(
            device=mesh_device,
            dim=args.dim,
            state_dict=state_dict,
            weight_key=weight_key,
            state_dict_prefix=f"layers.{layer_num}.",
            weight_cache_path=tensor_cache_path,
            weight_dtype=ttnn.bfloat16,
            add_unit_offset=True,
            eps=args.norm_eps,
            **(
                dict(is_distributed=args.is_distributed_norm, ccl_topology=args.ccl_topology(), tt_ccl=tt_ccl)
                if self.num_devices > 1
                else {}
            ),
        )
        if self.num_devices > 1:
            from models.tt_transformers.tt.distributed_norm import DistributedNorm

            return DistributedNorm(
                norm, args, tt_ccl=tt_ccl, TG=args.is_galaxy, ag_config_key=ag_key, enable_all_gather=enable_all_gather
            )
        return norm

    def _decode_norm(self, norm, x, norm_mode, norm_config, keep_sharded=False):
        if not getattr(self, "_sharded_decode_norm", False):
            return norm(x, mode=norm_mode, norm_config=norm_config)
        cfg = self.args.get_norm_config("attn", Mode.DECODE)
        # The sharded norm program is sized for one 32-row tile of users ([1,1,<=32,dim]). The single-device
        # batched decode layout [1,B,1,dim] pads every user to its own tile (physical height 32*B), which the
        # 32-row shard spec rejects; keep the interleaved norm there.
        shard_rows = cfg["sharded_output_config"].shard_spec.shape[0]
        phys = list(x.padded_shape)
        if phys[-2] * math.prod(phys[:-2]) != shard_rows:
            return norm(x, mode=norm_mode, norm_config=norm_config)
        x_sh = ttnn.to_memory_config(x, cfg["sharded_output_config"])
        out = norm(x_sh, mode=norm_mode, in_sharded=True, out_sharded=True, norm_config=cfg)
        ttnn.deallocate(x_sh)
        if keep_sharded:
            # the consumer (DRAM-sharded decode matmul hook) reads the width-sharded tensor directly
            return out
        out_il = ttnn.sharded_to_interleaved(out, ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(out)
        return out_il

    def forward(
        self,
        x,
        cos=None,
        sin=None,
        mode="decode",
        chunk_size=128,  # = GDN long_prefill_chunk_size; the only size the chunk-seq prefill kernel supports
        position_tensor=None,
        page_table=None,
        chunk_page_table=None,
        chunk_start_idx=None,
        chunk_start_idx_tensor=None,
        valid_len=None,
        gdn_collect=False,
    ):
        # the replicated residual is full width; a fractured [.., dim / tp] input (the MTP head / verify paths) keeps the
        # reduce-scatter contract
        if mode == "decode" and self._decode_replicated and int(x.shape[-1]) == self.args.dim:
            return self._forward_decode_replicated(x, cos, sin, position_tensor, page_table)
        _norm_mode = Mode.PREFILL if mode == "prefill" else Mode.DECODE
        if self.num_devices > 1:
            # TP: DistributedNorm uses the framework's per-norm memory configs.
            _attn_norm_config = self.args.get_norm_config("attn", _norm_mode)
            # PREFILL: distributed rmsnorm outputs in L1 so the fused in-proj AGMM gathers from L1, not DRAM.
            if _norm_mode == Mode.PREFILL:
                _attn_norm_config = {**_attn_norm_config, "distributed_output_mem_config": ttnn.L1_MEMORY_CONFIG}
            # DECODE ff_norm uses the attn_norm layout (act_shard_hidden, 32-core) so Qwen36MLP's input reshard is a no-op and the norm runs on 32 cores not 8; PREFILL keeps the framework ff config.
            if _norm_mode == Mode.DECODE:
                _ff_norm_config = self.args.get_norm_config("attn", _norm_mode)
            else:
                # ff_norm output stays DRAM: L1 keeps the full-width norm resident across the whole MLP,
                # clashing with each matmul's CBs (w1/w3/w2) for no gain. Verified dead end; keep DRAM.
                _ff_norm_config = self.args.get_norm_config("ff", _norm_mode)
        else:
            # In decode the norm output stays in L1 (as the old rms_norm_ttnn(memory_config=L1) did);
            # in prefill the framework RMSNorm returns interleaved DRAM (matches the old None default).
            _attn_norm_config = _ff_norm_config = (
                {"output_mem_config": ttnn.L1_MEMORY_CONFIG} if mode == "decode" else None
            )
        # Opt-in (args.decode_norm_sharded): single-device decode norms run the sharded RMSNorm program on a
        # width-sharded L1 copy of the residual instead of the one-core interleaved norm; output back to L1
        # interleaved for the (interleaved) token-mixer inputs.
        self._sharded_decode_norm = (
            self.num_devices == 1 and mode == "decode" and bool(getattr(self.args, "decode_norm_sharded", False))
        )
        attn_input = self._decode_norm(
            self.attention_norm,
            x,
            _norm_mode,
            _attn_norm_config,
            keep_sharded=bool(self.is_full_attention and getattr(self.args, "attn_norm_out_sharded", False)),
        )

        if self._tp_modules:
            # TP modules: input is the gathered (full-dim) norm output [1,1,B/S,dim];
            # output is fractured along dim=3. cos/sin are in rope_tp format.
            if self.is_full_attention:
                if mode == "prefill":
                    # Contract/vLLM path supplies a page_table → paged KV prefill; the
                    # demo path (no page_table) uses the internal concat caches.
                    if page_table is not None:
                        attn_output = self.attention.forward_prefill_paged(
                            attn_input,
                            cos,
                            sin,
                            page_table,
                            chunk_page_table=chunk_page_table,
                            chunk_start_idx=chunk_start_idx if chunk_start_idx is not None else 0,
                            chunk_start_idx_tensor=chunk_start_idx_tensor,
                        )
                    else:
                        attn_output = self.attention.forward_prefill(attn_input, cos, sin)
                else:
                    attn_output = self.attention.forward_decode(
                        attn_input, position_tensor, cos, sin, page_table=page_table
                    )
            else:
                # GDN carries its recurrent/conv state internally (capture_state on
                # prefill, read on decode); it has no paged KV, so page_table is N/A.
                if mode == "prefill":
                    if gdn_collect:
                        # Batched per-user prefill: stash this user's from-scratch state for
                        # assembly into row u of the batched buffers (finalize_pending later).
                        attn_output = self.attention.forward_prefill_collect(
                            attn_input, chunk_size=chunk_size, valid_len=valid_len
                        )
                    else:
                        attn_output = self.attention.forward_prefill(
                            attn_input, chunk_size=chunk_size, valid_len=valid_len, capture_state=True
                        )
                else:
                    attn_output = self.attention.forward_decode(attn_input)
        elif self.is_full_attention:
            attn_output = self.attention.forward(
                attn_input,
                cos,
                sin,
                position_tensor=position_tensor,
                page_table=page_table,
                chunk_page_table=chunk_page_table,
                chunk_start_idx=chunk_start_idx,
                chunk_start_idx_tensor=chunk_start_idx_tensor,
            )
        else:
            deltanet_mode = "chunk" if mode == "prefill" else "recurrent"
            attn_output = self.attention.forward(
                attn_input, mode=deltanet_mode, chunk_size=chunk_size, valid_len=valid_len
            )
        ttnn.deallocate(attn_input)

        h = ttnn.add(x, attn_output)
        ttnn.deallocate(attn_output)

        ff_input = self._decode_norm(self.ffn_norm, h, _norm_mode, _ff_norm_config)

        ff_output = self.feed_forward.forward(ff_input, mode=mode)
        ttnn.deallocate(ff_input)

        output = ttnn.add(h, ff_output)
        ttnn.deallocate(h)
        ttnn.deallocate(ff_output)

        return output

    def _forward_decode_replicated(self, x, cos, sin, position_tensor, page_table):
        """TP decode with the replicated residual (decode_boundary.py): x is [1, 1, B, dim] on every device in the
        boundary's residual layout; local norms, the token mixer and the feed-forward return per-device partial sums,
        and each boundary is residual + all_reduce(partial). Returns the residual in the same layout."""
        from models.demos.blackhole.qwen36.tt.decode_boundary import decode_boundary

        bnd = decode_boundary(self.args, self.device)
        if self._addnorm is not None:
            return self._forward_decode_addnorm(bnd, x, cos, sin, position_tensor, page_table)
        nc = bnd.norm_config
        attn_input = self.attention_norm.norm(x, mode=Mode.DECODE, in_sharded=True, out_sharded=True, norm_config=nc)
        if self.is_full_attention:
            part = self.attention.forward_decode(attn_input, position_tensor, cos, sin, page_table=page_table, partial=True)
        else:
            part = self.attention.forward_decode(attn_input, partial=True)
        ttnn.deallocate(attn_input)
        h = bnd.reduce_add(part, x, "mixer")
        ff_input = self.ffn_norm.norm(h, mode=Mode.DECODE, in_sharded=True, out_sharded=True, norm_config=nc)
        ff_part = self.feed_forward.forward(ff_input, mode="decode", partial=True)
        ttnn.deallocate(ff_input)
        out = bnd.reduce_add(ff_part, h, "mlp")
        ttnn.deallocate(h)
        return out

    def forward_decode_fabric(self, fb, res, x, g_next, cos, sin, position_tensor, page_table, last=False, proj=None,
                              next_consumer=None):
        """One-row decode on the fabric boundary (fabric_boundary/ops.py): res is the flat residual, x the normed row
        (this layer's input norm applied) as a [1, 1, 1, dim] L1 tile; the mixer and the MoE return L1 BF16 partial
        sums and each boundary is all-reduce + add + the next RMSNorm (g_next: the next layer's input norm, or the
        final norm when last: x then lands in DRAM for the LM head). Returns (res', x', None).
        proj: this layer's in-projection output when the previous boundary computed it (x is None then);
        next_consumer: the next layer's in-projection (its inproj_consumer), run inside this layer's last boundary:
        returns (res', None, its output)."""
        if proj is not None:
            if self.is_full_attention:
                part = self.attention.forward_decode_proj(proj, position_tensor, cos, sin, page_table=page_table)
            else:
                part = self.attention.forward_decode_proj(proj)
        else:
            if self.is_full_attention:
                part = self.attention.forward_decode(x, position_tensor, cos, sin, page_table=page_table, partial=True)
            else:
                part = self.attention.forward_decode(x, partial=True)
            ttnn.deallocate(x)
        # QWEN35_FB_ROUTER=1: the router matmul runs inside this boundary's program (its weight streams meanwhile)
        logits = None
        rows = int(part.shape[-2])
        if os.environ.get("QWEN35_FB_ROUTER", "0") == "1" and rows == 1 and hasattr(self.feed_forward, "router_consumer"):
            h, ff_in, logits = fb(res, part, self._fb_g_ffn, "mixer", consumer=self.feed_forward.router_consumer(fb),
                                  keep_x=True)
        else:
            h, ff_in = fb(res, part, self._fb_g_ffn, "mixer")
        ttnn.deallocate(res)
        ttnn.deallocate(part)
        ff_part = self.feed_forward.forward(ff_in, mode="decode", partial="fabric", logits=logits)
        ttnn.deallocate(ff_in)  # (the fused MoE frees the logits)
        if next_consumer is not None:
            res2, nxt = fb(h, ff_part, g_next, "mlp", consumer=next_consumer)
            ttnn.deallocate(h)
            return res2, None, nxt
        out_mc = ttnn.DRAM_MEMORY_CONFIG if last else ttnn.L1_MEMORY_CONFIG
        res2, x2 = fb(h, ff_part, g_next, "mlp", out_memory_config=out_mc)
        ttnn.deallocate(h)
        if rows > 1:  # the multi-row MoE partial is allocated per call (moe.py partial="fabric")
            ttnn.deallocate(ff_part)
        return res2, x2, None

    def _forward_decode_addnorm(self, bnd, x, cos, sin, position_tensor, page_table):
        """_forward_decode_replicated with the add + RMSNorm op (addnorm/ops.py): norm(x) -> mixer -> all-reduce ->
        (h, norm(h)) in one program -> MoE -> h + all-reduce."""
        _, attn_input = self._addnorm(x, None, self._g_attn)
        if self.is_full_attention:
            part = self.attention.forward_decode(attn_input, position_tensor, cos, sin, page_table=page_table, partial=True)
        else:
            part = self.attention.forward_decode(attn_input, partial=True)
        ttnn.deallocate(attn_input)
        red = bnd.all_reduce(part, "mixer")
        h, ff_input = self._addnorm(x, red, self._g_ffn)
        ttnn.deallocate(red)
        ff_part = self.feed_forward.forward(ff_input, mode="decode", partial=True)
        ttnn.deallocate(ff_input)
        out = bnd.reduce_add(ff_part, h, "mlp")
        ttnn.deallocate(h)
        return out
