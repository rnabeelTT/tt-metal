# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Functional TTNN decoder layer of GLM-5.3-Flash (HF `Glm5NextTextDecoderLayer`, transformers glm5_next).

Layer kinds (by index): KDA linear attention (l % 4 != 3) or DeepSeek sparse attention (l % 4 == 3, MLA + lightning
indexer); dense SwiGLU MLP (l < 3) or the 192-expert MoE (E192 checkpoint). Every layer is wrapped twice in mHC.

Contract (one decoder layer):
  * The residual is the packed 4-stream tensor X [1, 1, T, 4 * 4096] fp32, T = tokens (prefill) or users (decode).
  * prefill_forward(X, *, length, start, ...) runs one user's chunk; X rows >= length are padding.
  * decode_forward(X, ...) runs one token for each of B users (B = decode batch); recurrent state, conv history and
    the attention caches live in fixed-address device buffers owned by the layer (trace-safe).
  * Returns X' with the same shape.

Per sub-block (HF order):  h, post, comb = hc.pre(X);  y = block(rmsnorm(bf16(h)));  X = hc.post(y, X, post, comb)
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule

from .ckpt import LazyMap
from .kda import GlmKDA
from .mhc import HyperConnection, hc_config
from .mlp import DenseMLP
from .moe_opt import OptimizationConfig, Qwen35MoEOptimized

# MoE policy of the functional decoder: bfp8 experts (accuracy first; the datatype sweep chooses the shipped policy),
# every prefill chunk on the expert-sorted path (the masked path's sparse matmul is biased, see tt/moe_opt.py)
FUNCTIONAL_MOE = OptimizationConfig(
    routed_gate_up_dtype=ttnn.bfloat8_b,
    routed_down_dtype=ttnn.bfloat8_b,
    routed_fidelity=ttnn.MathFidelity.HiFi4,  # HiFi2 truncates the activation mantissa: routed output -3.3 % (probe)
    shared_fidelity=ttnn.MathFidelity.HiFi4,
    sorted_min_tokens=1,
)

DRAM = ttnn.DRAM_MEMORY_CONFIG


def layer_kind(tc: dict, layer_idx: int) -> tuple[str, str]:
    attn = "kda" if tc["layer_types"][layer_idx] == "linear_attention" else "dsa"
    mlp = "dense" if tc["mlp_layer_types"][layer_idx] == "dense" else "moe"
    return attn, mlp


def moe_args(tc: dict, tp: int, num_experts: int = 192, opt: OptimizationConfig = FUNCTIONAL_MOE):
    return SimpleNamespace(
        opt=opt,
        dim=tc["hidden_size"],
        num_experts=num_experts,
        num_experts_per_tok=tc["num_experts_per_tok"],
        moe_intermediate_size=tc["moe_intermediate_size"],
        shared_expert_intermediate_size=tc["moe_intermediate_size"] * tc["n_shared_experts"],
        router_kind="sigmoid_bias",
        routed_scaling_factor=tc["routed_scaling_factor"],
        shared_expert_gated=False,
        swiglu_limit=tc["swiglu_limit"],
        num_devices=tp,
        act_shard_hidden=None,
        moe_precision=None,
        precision_layer_exceptions=None,
        ccl_topology=lambda: ttnn.Topology.Linear,
    )


def moe_state_dict(layer_sd: dict[str, torch.Tensor], num_experts: int) -> LazyMap:
    """Checkpoint suffixes -> the Qwen35MoE names (HF-fused expert tensors, singular shared_expert). Lazy: a tensor
    (the 192-expert stacks above all) is read from layer_sd only when looked up (weight-cache misses)."""
    E = range(num_experts)
    return LazyMap(
        {
            "gate.weight": lambda: layer_sd["mlp.gate.weight"],
            "gate.e_score_correction_bias": lambda: layer_sd["mlp.gate.e_score_correction_bias"].float(),
            "experts.gate_up_proj": lambda: torch.stack(
                [
                    torch.cat(
                        [layer_sd[f"mlp.experts.{e}.gate_proj.weight"], layer_sd[f"mlp.experts.{e}.up_proj.weight"]]
                    )
                    for e in E
                ],
                dim=0,
            ),
            "experts.down_proj": lambda: torch.stack([layer_sd[f"mlp.experts.{e}.down_proj.weight"] for e in E], dim=0),
            "shared_expert.gate_proj.weight": lambda: layer_sd["mlp.shared_experts.gate_proj.weight"],
            "shared_expert.up_proj.weight": lambda: layer_sd["mlp.shared_experts.up_proj.weight"],
            "shared_expert.down_proj.weight": lambda: layer_sd["mlp.shared_experts.down_proj.weight"],
        }
    )


class FunctionalDecoder(LightweightModule):
    @classmethod
    def from_state_dict(cls, state_dict, *, hf_config, layer_idx, mesh_device, **kwargs):
        """state_dict: one layer's checkpoint-named tensors (suffixes after `model.language_model.layers.{i}.`);
        hf_config: the text_config dict of the checkpoint."""
        return cls(mesh_device, hf_config, state_dict, layer_idx, **kwargs)

    def __init__(
        self,
        mesh_device,
        tc: dict,
        sd: dict[str, torch.Tensor],
        layer_idx: int,
        *,
        max_batch: int = 32,
        prefill_lengths=(128, 512, 2048),
        dsa_kwargs: dict | None = None,
        weight_cache=None,
        tt_ccl=None,
        moe_opt=None,
        kda_dtypes=None,
        dsa_bmm_dtype=None,
    ):
        """weight_cache: tt/weight_cache.WeightCache of the model (None = no cache); kda_dtypes (input, output
        projection) and dsa_bmm_dtype (W_UK / W_UV): the precision policy's dtypes, applied at load (None = bf16)."""
        self.mesh = mesh_device
        self.tc = tc
        self.layer_idx = layer_idx
        self.tp = tuple(mesh_device.shape)[1]
        self.attn_kind, self.mlp_kind = layer_kind(tc, layer_idx)
        self.max_batch = max_batch
        self.prefill_lengths = tuple(prefill_lengths)
        self.hidden = tc["hidden_size"]
        self.eps = tc["rms_norm_eps"]
        hcfg = hc_config(tc)
        self.n_streams = hcfg.n
        self.attn_hc = HyperConnection(mesh_device, hcfg, sd["hc_attn_fn"], sd["hc_attn_base"], sd["hc_attn_scale"])
        self.ffn_hc = HyperConnection(mesh_device, hcfg, sd["hc_ffn_fn"], sd["hc_ffn_base"], sd["hc_ffn_scale"])
        rep = ttnn.ReplicateTensorToMesh(mesh_device)
        norm_w = lambda t: ttnn.from_torch(
            t.to(torch.bfloat16).reshape(1, 1, -1, 32),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            mesh_mapper=rep,
        )
        # fp32 accumulation: the collapsed stream is small (rms ~3e-3 early on, mean(x^2) ~ eps), so a bf16 sum of squares
        # shifts the normalized scale by percents (and the MoE routing with it)
        self.norm_ckc = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        self.input_norm_w = norm_w(sd["input_layernorm.weight"])
        self.post_norm_w = norm_w(sd["post_attention_layernorm.weight"])
        # per-component cache scopes (a source edit of one component's weight build re-keys only that component)
        wc = (
            (lambda kind, policy=None: weight_cache.sub(f"layer_{layer_idx}").component(kind, policy))
            if weight_cache is not None
            else None
        )

        if self.attn_kind == "kda":
            self.attn = GlmKDA(
                mesh_device,
                tc,
                sd,
                layer_idx,
                prefill_lengths=prefill_lengths,
                max_batch=max_batch,
                tt_ccl=tt_ccl,
                weight_cache=wc and wc("kda"),
                dtypes=kda_dtypes,
            )
        else:
            from .dsa import GlmDSA

            T = (dsa_kwargs or {}).get("max_context")
            if T is not None:  # every prefill chunk occupies a whole constructed length of the per-slot caches
                assert T % max(prefill_lengths) == 0, f"max_context {T} must be a multiple of {max(prefill_lengths)}"

            self.attn = GlmDSA(
                mesh_device,
                tc,
                sd,
                layer_idx,
                max_batch=max_batch,
                weight_cache=wc and wc("dsa"),
                bmm_dtype=dsa_bmm_dtype,
                **(dsa_kwargs or {}),
            )

        if self.mlp_kind == "dense":
            self.mlp = DenseMLP(
                mesh_device,
                lambda: sd["mlp.gate_proj.weight"],
                lambda: sd["mlp.up_proj.weight"],
                lambda: sd["mlp.down_proj.weight"],
                tc["swiglu_limit"],
                cache=wc and wc("mlp"),
            )
        else:
            num_experts = sd["mlp.gate.weight"].shape[0]
            args = moe_args(tc, self.tp, num_experts, opt=moe_opt or FUNCTIONAL_MOE)
            # a per-layer precision exception (MoEPrecision.from_args) keys its own weight-cache directory
            from .moe import MoEPrecision

            eff = MoEPrecision.from_args(args, layer_idx)
            exc = None if eff == MoEPrecision.from_args(args) else {k: str(v) for k, v in vars(eff).items()}
            self.mlp = Qwen35MoEOptimized(
                mesh_device,
                args,
                moe_state_dict(sd, num_experts),
                tensor_cache_path=wc and wc("moe", exc),
                tt_ccl=tt_ccl,
                layer_num=layer_idx,
            )
        self.decode_state = None

    # ------------------------------------------------------------------------------------------------- helpers
    def _norm(self, h: ttnn.Tensor, w) -> ttnn.Tensor:
        hb = ttnn.typecast(h, ttnn.bfloat16)
        ttnn.deallocate(h)
        out = ttnn.rms_norm(hb, epsilon=self.eps, weight=w, memory_config=DRAM, compute_kernel_config=self.norm_ckc)
        ttnn.deallocate(hb)
        return out

    def _ffn(self, X: ttnn.Tensor, mode: str) -> ttnn.Tensor:
        h, post, comb = self.ffn_hc.pre(X)
        x = self._norm(h, self.post_norm_w)
        if self.mlp_kind == "dense":
            z = self.mlp(x)
        else:
            z = self.mlp.forward(x, mode=mode)
        ttnn.deallocate(x)
        X2 = self.ffn_hc.post(z, X, post, comb)
        for t in (z, post, comb):
            ttnn.deallocate(t)
        return X2

    # ------------------------------------------------------------------------------------------------- prefill
    def prefill_inputs(self, n: int, *, start: int, length: int, slot: int = 0):
        """Device inputs of one prefill chunk (n = constructed chunk rows, start % 128 == 0, length <= n valid rows,
        decode slot of the user). Host -> device input construction happens here, never inside prefill_forward."""
        if self.attn_kind == "kda":
            return {"attn": self.attn.prefill_inputs(start, length)}
        return {"attn": self.attn.prefill_inputs(n, start, length, slot)}

    def prefill_forward(self, X: ttnn.Tensor, *, inputs, attn_state=None):
        """One user's chunk. X [1,1,n,4*hidden] fp32 with n a constructed prefill length; inputs = prefill_inputs().
        attn_state: KDA -> the KdaState carried from the previous chunk (None = fresh sequence); DSA -> unused (its
        caches are device state of the layer). Returns (X', new attention state)."""
        h, post, comb = self.attn_hc.pre(X)
        x = self._norm(h, self.input_norm_w)
        if self.attn_kind == "kda":
            state = attn_state if attn_state is not None else self.attn.new_prefill_state()
            y, new_state = self.attn.prefill_forward(x, state, inputs["attn"])
        else:
            y, new_state = self.attn.prefill_forward(x, inputs=inputs["attn"])
        ttnn.deallocate(x)
        X1 = self.attn_hc.post(y, X, post, comb)
        for t in (y, post, comb):
            ttnn.deallocate(t)
        X2 = self._ffn(X1, "prefill")
        ttnn.deallocate(X1)
        return X2, new_state

    # -------------------------------------------------------------------------------------------------- decode
    def allocate_decode_state(self, batch: int | None = None):
        """Allocate the layer's decode state (and every constant of its passes) before any trace capture. DSA layers
        need this before their first prefill (prefill writes the caches)."""
        self.decode_state = self.attn.allocate_decode_state(batch)
        if self.attn_kind == "dsa":
            self.attn.prepare_constants(self.prefill_lengths)
        if self.mlp_kind == "moe":  # sorted-dispatch constants of every prefill chunk length and the decode batch
            self.mlp.prepare_dispatch(list(self.prefill_lengths) + [batch or self.max_batch])
        return self.decode_state

    def commit_prefill(self, slot: int, prefill_state):
        """Hand one user's final prefill state to decode slot `slot` (KDA carries; DSA caches are already paged)."""
        if self.attn_kind == "kda":
            self.attn.write_slot(self.decode_state, slot, prefill_state)

    def decode_inputs(self, positions):
        """Fixed-address device buffers of the per-step decode inputs (allocate before any trace capture).
        positions[s] = position of slot s's token this step; -1 = inactive slot (its state is left unchanged)."""
        return self.attn.decode_inputs(positions)

    def update_decode_inputs(self, bufs, positions):
        """Refresh the decode input buffers in place (host -> device; between trace replays)."""
        self.attn.update_decode_inputs(bufs, positions)

    def set_page_table(self, page_table):
        """DSA layers: install the [S, <= max_context/128] block table of the paged latent / pool-key caches."""
        if self.attn_kind == "dsa":
            self.attn.set_page_table(page_table)

    def decode_forward(self, X: ttnn.Tensor, *, inputs) -> ttnn.Tensor:
        """X [1,1,B,4*hidden] fp32, one token per user -> X'; inputs = decode_inputs(...)."""
        h, post, comb = self.attn_hc.pre(X)
        x = self._norm(h, self.input_norm_w)
        if self.attn_kind == "kda":
            y = self.attn.decode_forward(x, self.decode_state, inputs=inputs)
        else:
            y = self.attn.decode_forward(x, inputs=inputs)
        ttnn.deallocate(x)
        X1 = self.attn_hc.post(y, X, post, comb)
        for t in (y, post, comb):
            ttnn.deallocate(t)
        X2 = self._ffn(X1, "decode")
        ttnn.deallocate(X1)
        return X2
