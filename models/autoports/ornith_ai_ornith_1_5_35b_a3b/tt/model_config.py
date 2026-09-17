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
def multidie_parallel_enabled(mesh_devices: int) -> bool:
    """Should a multi-die mesh run TENSOR + EXPERT parallel rather than replicate?

    NO by default, on every mesh. Opt in with ORNITH_TP=1.

    This is the honest answer as of 2026-09-17, and it is deliberately NOT the answer the
    prefill numbers alone would suggest. Tensor parallelism on four dies is a large
    prefill and TTFT win at the MODEL level -- 706 tok/s at ISL 128 against 353
    replicated (2.0x), 1241 at ISL 4096 against 395 (3.1x, which passes
    qwen3-coder-30b-a3b's 1016), TTFT 360 -> 181 ms -- and it is the only route to
    on-device sampling, since qwen36 builds its SamplingGenerator only when the mesh is
    (1,4) or (1,8) AND vocab_size // num_devices <= 64K (replication forces
    num_devices = 1, so 248320 fails the bound; 248320 // 4 = 62080 passes).

    It is nevertheless off by default because of two measured facts:

    1. IT DOES NOT SERVE. A 4-die TP instance fails in the vLLM engine's warmup with

         Statically allocated circular buffers on core range [0-0 - 10-5] grow to
         1684544 B which is beyond max L1 size of 1572864 B

       and that byte count is IDENTICAL at max_model_len 4096 and 2048, with and without
       an l1_small_size reservation -- so it is a fixed circular-buffer configuration in
       the serving warmup path, not a context-length or chunking problem. The bench's TP
       prefill and decode both work, so whatever the warmup does differently is where the
       defect lives. Unfixed, and not guessed at.

    2. SINGLE-STREAM DECODE IS WORSE, not better: 8.79 tok/s eager on four dies against
       18.40 on one. The arithmetic points at collectives -- 114 ms/token versus 54 ms,
       and this port's TP does roughly four per layer (the DistributedNorm all-gather, the
       out-projection reduce-scatter, the MoE all-reduce, and mesh_partition) where
       ornith-1.0's replicated-residual design does two, which is how it reaches 43 tok/s
       on four dies. Matching that means adopting the replicated residual, not tuning.

    So the shipped multi-die deployment stays LANES -- one replica per die via
    tools/serve_lanes.sh, measured end to end through vLLM at 26 / 54 / 91 tok/s
    aggregate on 1 / 2 / 4 dies. Replication is what every profile in tt-model.yaml
    serves, and it is what the readiness evidence was produced on.

    ORNITH_TP=1 turns tensor + expert parallelism on for a mesh of two or more dies, which
    is how the numbers above were taken and how the remaining work should be continued.
    Note that two dies additionally hang in the expert-parallel all-reduce
    (moe.py::_reduce), caught twice with py-spy on a freshly reset device, so four dies is
    the only configuration where TP is known to run at all.
    """
    if mesh_devices <= 1:
        return False  # nothing to shard across, and a fabric here has no peer
    return os.environ.get("ORNITH_TP") == "1"


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
VALID_KV_DTYPES = ("bfloat8_b", "bfloat16")


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

        # NON-MoE MATH FIDELITY. qwen36's attention and GDN build their compute kernel
        # configs from QWEN36_MATH_FIDELITY, whose default stays `lofi` so the dense
        # Qwen3.5/3.6 models that share those modules are unaffected. Ornith wants hifi2,
        # so it asks for it HERE rather than by changing shared code.
        #
        # Measured on the full 40-layer model against the HF reference, P150, bfp4
        # experts, 2 reps each and bitwise reproducible (non-MoE fidelity / shared-expert
        # fidelity -> logit PCC p0/p1/p2, top-8 overlap):
        #
        #   lofi  / hifi4  0.96809 0.94212 0.93442   6/8 6/8 6/8   <- what shipped
        #   hifi2 / lofi   0.97174 0.97134 0.94334   6/8 8/8 6/8   <- this policy
        #   hifi2 / hifi2  0.96959 0.95724 0.92689   6/8 6/8 7/8
        #   lofi  / lofi   0.96757 0.96115 0.93623   8/8 7/8 8/8
        #   hifi4 / lofi   0.97238 0.96287 0.93214   7/8 7/8 7/8
        #
        # hifi2/lofi is the best PCC on ALL THREE prompts and the largest single gain
        # available (+0.029 on prompt 1, +0.009 on prompt 2). It works because the
        # non-MoE path DOES set fp32_dest_acc_en=True, so extra fidelity passes there
        # actually resolve; the shared expert does not, so extra passes there hurt (see
        # moe.py::_hifi).
        #
        # A REAL TRADE-OFF, recorded rather than hidden: lofi/lofi has better TOP-8 TOKEN
        # AGREEMENT (8/7/8 against this policy's 6/8/6), and test_model.py's own docstring
        # argues token agreement discriminates better than PCC. Neither config dominates.
        # hifi2/lofi is chosen because PCC is the criterion the stage gate actually fails,
        # and it is uniformly better there; top-1 is 3/3 and top-8 clears the 6/8 bar in
        # both. Set QWEN36_MATH_FIDELITY=lofi ORNITH_MOE_FIDELITY=lofi to take the other
        # side of that trade.
        #
        # NOTE none of this reaches the 0.95 bar on prompt 2, and more precision will not:
        # the binding constraint is discrete router expert-selection agreement with HF.
        # Changing any fidelity flips 11-16% of expert slots and alters the expert set for
        # 59-69% of token-decisions, concentrated in layers 23-32. Top-8-of-256 is
        # discontinuous, so bf16 noise flips near-ties and the model draws a DIFFERENT
        # expert -- a bounded-but-large change no matmul precision removes.
        os.environ.setdefault("QWEN36_MATH_FIDELITY", "hifi2")

        policy = MESH_PRECISION.get(self.device_name)
        if policy is None:
            raise ValueError(
                f"No precision policy for mesh {self.device_name!r}. "
                f"Known meshes: {sorted(MESH_PRECISION)}. Add an entry (and a matching "
                f"configs/moe_*.yaml) before running on this board."
            )
        # Only 10 of 40 layers hold a KV cache (20 KB/token), so raising its precision is
        # nearly free in memory here -- which makes it a dtype-sweep knob rather than a
        # fixed policy. ORNITH_KV_DTYPE overrides; the policy value is the default.
        kv_override = os.environ.get("ORNITH_KV_DTYPE")
        if kv_override:
            if kv_override not in VALID_KV_DTYPES:
                raise ValueError(f"ORNITH_KV_DTYPE={kv_override!r} is not one of {VALID_KV_DTYPES}")
            self.kv_cache_dtype = kv_override
        else:
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
        # CORRECTED 2026-09-17: the paragraph that used to stand here said the checkpoint
        # layout prevents tensor parallelism -- that q's pre-fused [8192, 2048] [q|gate]
        # and GDN's separate qkv/z/a/b keys were what produced
        #   "all_gather_minimal_matmul_async inner dimensions must match, K=8192 K_w=2048".
        # That was wrong on both counts. load_gdn_weights_tp wants qkv/z/a/b SEPARATE and
        # folds them itself, load_attention_weights_tp de-interleaves the fused [q|gate],
        # and the K=8192 assert had nothing to do with the weights: it was the prefill
        # norm-allgather/in-proj FUSION being enabled while `is_distributed_norm(PREFILL)`
        # is False at dim=2048, so DistributedNorm gathered BEFORE the norm and the fused
        # op tried to gather the already-replicated activation a second time
        # (see tp_common.prefill_norm_agmm_enabled). ORNITH_TP=1 now runs correct prefill.
        #
        # Replication remains the DEFAULT anyway, for the reason below -- and note what
        # tensor parallelism on its own buys, measured:
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
        # ORNITH_TP=1 keeps the REAL device count, i.e. tensor-parallel attention + GDN
        # instead of replication. Replication is the default because TP buys nothing on
        # single-stream decode (dispatch-bound: the same op count on smaller tensors,
        # plus collectives). The reason to want TP anyway is BATCH: the per-slot serving
        # machinery -- prefill_paged_slots, _write_gdn_slot, _remap_gdn_slots -- is
        # implemented against TPGatedDeltaNet and gated on num_devices > 1, so a
        # replicated model cannot reach max_num_seqs > 1 at all. Concurrency on a
        # replicated model comes from lanes instead (tools/serve_lanes.sh).
        #
        # PREFILL, measured on P150x4 (tools/bench_model.py --prefill-only, warmed):
        #   ISL      single chip     ORNITH_TP=1    ORNITH_TP=1 + ORNITH_MOE_EP=1
        #    128    351 tok/s       367  (1.04x)    684  (1.95x)
        #   4096    395 tok/s       478  (1.21x)   1272  (3.22x)
        #  16384    381 tok/s       468  (1.23x)   1198  (3.15x)
        # Tensor parallelism ALONE is worth ~1.2x, because it parallelises only the 3% of
        # the parameters that are not experts. The 3.2x needs EXPERT parallelism with it,
        # which is what the ornith-1.0 comparison (3.29-3.51x at TP=4/EP=4) was measuring.
        # Default ON for any multi-die mesh; see multidie_parallel_enabled above for why
        # (prefill 3.2x, and it is the precondition for on-device sampling). ORNITH_TP=0
        # opts back out to replication.
        self.tensor_parallel = multidie_parallel_enabled(self.mesh_devices)
        # Expert parallelism travels WITH tensor parallelism: TP alone is worth only
        # ~1.2x because it parallelises the 3% of parameters that are not experts, and
        # EP contributes the rest (2.60x at ISL 4096, 2.55x at 16384, measured with TP
        # held fixed). ornith-1.0 likewise runs TP=4 and EP=4 on the same four chips.
        # ORNITH_MOE_EP still overrides explicitly, either way.
        self.expert_parallel = self.tensor_parallel
        _ep = os.environ.get("ORNITH_MOE_EP")
        if _ep is not None:
            self.expert_parallel = _ep == "1"
        if self.mesh_devices > 1 and not self.tensor_parallel:
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
