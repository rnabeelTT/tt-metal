# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""ModelArgs for Ornith-1.5-35B-A3B (HF ``model_type: qwen3_5_moe``).

Extends the Qwen3.5/3.6 Blackhole args with the MoE fields and the per-mesh precision
policy. The base class already derives everything hybrid-specific from the HF text
config — layer_types, GDN head dims, partial rotary factor, m-RoPE sections, vocab —
so this subclass only has to add what a mixture-of-experts model has and a dense one
does not.
"""
import os

from models.demos.blackhole.qwen36.tt.model_config import Qwen36ModelArgs

# Which precision each board can actually host.
#
# THE FUSED moe_compute PIPELINE IS NOT AVAILABLE ON THIS HARDWARE. TTMoEDecode targets
# Galaxy-class meshes: every shipped config is 8x4 / 16x4 / 16x8 (32-128 chips), and its
# own test skips with "FABRIC_1D only valid for (8,1) BH LB mesh". Measured on a QB2
# (2x p300c, 4 chips), a (1,4) or (4,1) mesh returns ~FLT_MAX garbage and a (2,2) mesh
# fails in reduce-scatter on an unsupported ND_SHARDED layout -- with random weights, so
# this is the pipeline and not the checkpoint. All three of our meshes therefore run the
# UNFUSED path: experts stored plainly, applied with a per-expert loop.
#
# That turns out to help precision. The fused path stores weights in a packed ring layout
# costing a measured ~1.9x (bfp4 0.53x, bfp8 1.01x, bf16 1.90x of raw bf16 bytes), which
# is what made bf16 impossible everywhere. Unpacked weights cost their true size, so each
# board can hold roughly one grade better:
#
#   routed experts = 64.4 GB raw bf16 (92.5% of 34.8B params); non-expert = 4.4 GB bf16
#   per chip: 34.2 GB DRAM (8 banks x 4.278 GB), ~30 GB usable
#
#   dtype   experts   1 chip           2 chips          4 chips
#   bfp4    18.1 GB   22.5 GB  OK      13.5 GB  OK       8.9 GB  OK
#   bfp8    34.2 GB   38.6 GB  --      21.5 GB  OK      12.9 GB  OK
#   bf16    64.4 GB   68.8 GB  --      36.6 GB  --      20.5 GB  OK
#
# So each mesh takes the best precision it can hold: bf16 on 4 chips, bfp8 on 2, bfp4 on
# one. The cost is throughput -- the unfused loop has no token dispatch, so it touches
# every expert a device owns rather than only the 8 a token selected. At batch 32 with
# 256 experts that is close to unavoidable anyway (32 tokens x top-8 = 256 selections
# spread over 256 experts, so most are hit); at batch 1 it is genuinely wasteful, which
# is a latency problem for a single user and is what the optimize stage targets.
#
# The router and shared expert stay bf16 on every mesh. Together they are 0.5% of the
# weights, so quantizing them frees nothing worth having, and the router decides which
# experts run at all -- an error there swaps an expert, which perturbs the output far
# more than a rounding error inside one.
# Routed experts are bfp4 on EVERY mesh, and that is a consequence of choosing data
# parallelism over expert parallelism (see _SparseExperts).
#
# Under expert parallelism the routed weights are split N ways, so 4 dies could afford
# bfloat16 (64 GB / 4) and 2 dies bfloat8_b (34 GB / 2). Replicated, every die needs a
# WHOLE copy against 30 GB of usable DRAM, and only bfp4 fits:
#
#   experts, 32.2B params   bfloat16  64.4 GB    bfloat8_b  34.2 GB    bfp4  17.1 GB
#
# What that buys is worth more than the precision: the gather path costs the same 79 us
# per expert pair at 64 or 256 experts per die, so splitting experts was never buying
# speed -- only an all-reduce per layer. Replicated, extra dies serve extra USERS, and
# aggregate throughput scales linearly instead of sub-linearly. bfp4 measures 0.968
# top-1 against the HF reference on P150, so the quality cost is small and already
# accepted on the single-chip profile.
#
# Set ORNITH_MOE_EP=1 to shard experts again; the higher-precision entries below are
# then the ones that apply, and they are kept here for that path.
MESH_PRECISION = {
    "P150": {
        "routed": "bfloat4_b",
        "kv": "bfloat8_b",
    },
    "P300": {
        "routed": "bfloat4_b",
        "kv": "bfloat8_b",
        "routed_ep": "bfloat8_b",
    },
    "P150x4": {
        "routed": "bfloat4_b",
        "kv": "bfloat8_b",
        "routed_ep": "bfloat16",
    },
}

VALID_ROUTED_DTYPES = ("bfloat4_b", "bfloat4_bfloat8", "bfloat8_b", "bfloat16")


class OrnithModelArgs(Qwen36ModelArgs):
    """Ornith-1.5-35B-A3B: the Qwen3.5 hybrid decoder with every MLP replaced by MoE."""

    def _set_params_from_dict(self, config, *args, **kwargs):
        """Supply the dense-MLP fields the base class requires but an MoE config lacks.

        tt_transformers' ModelArgs needs `intermediate_size` (or the
        ffn_dim_multiplier/multiple_of pair) to size the dense MLP, and a
        `qwen3_5_moe` text config has neither -- it has `moe_intermediate_size` and
        `shared_expert_intermediate_size`. Without this the base KeyErrors before any
        of this model's own code runs.

        Done HERE rather than by patching the checkpoint's config.json, because the
        packaged image fetches weights straight from the Hub: a patched local copy
        exists only on the machine that made it, so the container would hit the
        original config and fail at engine init. Every layer is MoE, so
        moe_intermediate_size is the honest per-expert width to report; nothing
        constructs a dense MLP.
        """
        text = config.get("text_config", config)
        if "intermediate_size" not in text and "moe_intermediate_size" in text:
            text["intermediate_size"] = text["moe_intermediate_size"]
        if "num_local_experts" not in text and "num_experts" in text:
            text["num_local_experts"] = text["num_experts"]
        return super()._set_params_from_dict(config, *args, **kwargs)

    def __init__(self, mesh_device=None, **kwargs):
        # ORNITH_WEIGHTS points at the local snapshot; HF_MODEL is what the base class
        # reads. Setting it here keeps every caller from having to know both names.
        weights = os.environ.get("ORNITH_WEIGHTS")
        if weights:
            os.environ.setdefault("HF_MODEL", weights)
        super().__init__(mesh_device, **kwargs)

        tc = self.hf_config.get_text_config()
        self.num_experts = tc.num_experts
        self.num_experts_per_tok = tc.num_experts_per_tok
        self.moe_intermediate_size = tc.moe_intermediate_size
        self.shared_expert_intermediate_size = tc.shared_expert_intermediate_size
        # Every layer in this family is MoE; there is no dense prefix to skip.
        self.first_k_dense_replace = 0
        # The checkpoint ships one MTP layer for speculative decode. It is not part of
        # the base forward pass and is excluded until MTP is scoped as its own goal.
        self.mtp_num_hidden_layers = getattr(tc, "mtp_num_hidden_layers", 0)

        policy = MESH_PRECISION.get(self.device_name)
        if policy is None:
            raise ValueError(
                f"No precision policy for mesh {self.device_name!r}. "
                f"Known meshes: {sorted(MESH_PRECISION)}. Add an entry (and a matching "
                f"configs/moe_*.yaml) before running on this board."
            )
        self.kv_cache_dtype = policy["kv"]

        # Stage 09 sweeps precision by overriding the policy per run. Validated here so
        # a typo fails at construction rather than as a confusing dtype error 40 layers
        # into weight upload.
        override = os.environ.get("ORNITH_ROUTED_DTYPE")
        if override:
            if override not in VALID_ROUTED_DTYPES:
                raise ValueError(f"ORNITH_ROUTED_DTYPE={override!r} is not one of {VALID_ROUTED_DTYPES}")
            self.routed_expert_dtype = override
        else:
            self.routed_expert_dtype = policy["routed"]

        # PARALLELISM: replicate attention/GDN, shard only the experts.
        #
        # qwen36's tensor-parallel path assumes the dense Qwen3.5 weight layout, and this
        # MoE checkpoint does not match it: q arrives with its output gate pre-fused as
        # one [8192, 2048] tensor, and GDN's qkv is stored separately from z/a/b where
        # the TP path wants one folded qkvzab. That surfaces as
        #   "all_gather_minimal_matmul_async inner dimensions must match, K=8192 K_w=2048"
        # and, once the in-projection fusions are disabled, as further mismatches in the
        # tuned prefill program configs and the sharded decode matmul.
        #
        # Rather than re-derive those fusions, note what tensor parallelism was buying:
        # attention + GDN are 1.03B of 34.8B parameters (~2 GB bf16), so sharding them
        # saves ~1.5 GB per chip on a 4-chip mesh. The experts are 92.5% of the weights
        # and are sharded regardless. So the attention and GDN weights are REPLICATED --
        # every module takes its validated single-device path -- and expert parallelism
        # alone carries the memory win:
        #
        #   4 chips: 16.1 GB experts/chip + 4.4 GB replicated = 20.5 GB/chip
        #
        # which is the same per-chip total the TP plan had. `mesh_devices` keeps the true
        # device count for the MoE, which does shard; `num_devices = 1` is what makes the
        # qwen36 modules replicate (an upload with no mesh mapper replicates across a
        # MeshDevice -- verified on this box).
        self.mesh_devices = mesh_device.get_num_devices() if mesh_device is not None else 1
        if self.mesh_devices > 1:
            self.num_devices = 1
            # cluster_shape has to agree. It is read INDEPENDENTLY of num_devices -- the
            # embedding shards its weight on the hidden dim via
            # ShardTensor2dMesh(dims=(None, 3), mesh_shape=cluster_shape) -- so leaving
            # it at the real mesh shape hands the first layer a hidden/N-wide residual
            # and the very first RMSNorm fails on a gamma/​input width mismatch. A (1, 1)
            # cluster shape makes that shard a no-op, which is exactly the replicated
            # behaviour this configuration wants.
            self.cluster_shape = [1, 1]
            # And is_multichip, which the base __init__ already computed as
            # `num_devices > 1` from the REAL device count before this override ran.
            # It is what DistributedNorm gates its all_gather on
            # (distributed_norm.py: `if self.args.is_multichip and not
            # is_distributed_norm(mode)`), so leaving it True makes a fully replicated
            # model reach for a collective it has no fabric for and die in
            # control_plane.cpp with "Trying to get un-initialized fabric context" --
            # in the first norm of the first layer, nowhere near this line.
            self.is_multichip = False

        # n_local_kv_heads is only assigned inside qwen36's tensor-parallel block, so it
        # does not exist on a single device -- and it does not exist here either, because
        # this configuration forces num_devices to 1 to replicate attention. Anything
        # sizing a paged KV cache needs it (`allocate_kv_caches` takes
        # [num_blocks, n_local_kv_heads, block, head_dim]), and its absence surfaces as a
        # bare AttributeError from the caller rather than from the model.
        #
        # This matters more than it looks: without a paged KV cache, the attention module
        # silently takes its CONCAT-PREFILL branch for single-token decode
        # (`use_paged_attention and T == 1` is false), so "decode" measures a one-token
        # prefill and trace capture fails inside the prefill path.
        if not hasattr(self, "n_local_kv_heads"):
            self.n_local_kv_heads = max(1, self.n_kv_heads // max(self.num_devices, 1))

    def weight_cache_path(self, dtype=None):
        """Qualify the tensor cache by routed-expert dtype.

        The base implementation buckets everything that is not bfloat8_b into
        `tensor_cache_bf16`, so a bfp4 run and a bf16 run would share a directory. Cached
        filenames do embed their dtype, so nothing is silently mis-read today -- but the
        datatype sweep re-runs the same model at four precisions back to back, and one
        directory holding four copies of every expert is both confusing and large enough
        to matter (the experts are 92.5% of the weights).
        """
        base = super().weight_cache_path(dtype)
        return base.parent / f"{base.name}_routed_{self.routed_expert_dtype}"
