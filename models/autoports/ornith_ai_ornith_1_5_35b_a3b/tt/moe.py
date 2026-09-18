# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The one seam where Ornith-1.5-35B-A3B differs from the qwen36 Blackhole port.

qwen36 gives every decoder layer a dense SwiGLU MLP. Ornith replaces all 40 of them
with a 256-expert top-8 MoE plus one always-on shared expert. Everything else in the
layer — Gated DeltaNet, gated full attention, partial/m-RoPE, zero-centred RMSNorm —
is unchanged, so this module is a drop-in replacement for `qwen36.tt.mlp` and nothing
else in the decoder needs to know an MoE is present.

Decode and prefill take deliberately different paths:

* **Decode** (one token per user) is latency-bound and sparse: 8 of 256 experts fire
  per token, so a dense pass would waste 32x the bandwidth. It uses the shared
  `TTMoEDecode` pipeline (all_to_all_dispatch -> moe_compute -> fast_reduce_nc_fused
  -> reduce-scatter), which moves tokens to the devices that own their experts.

* **Prefill** processes thousands of tokens at once. There, every expert is hit by
  *some* token, so gathering per-expert token groups and running one grouped matmul
  per expert is both denser and simpler than a dispatch round-trip. `_forward_prefill`
  loops the locally-owned experts and scatter-adds their outputs.

The routed-expert dtype is the single lever that decides which boards can host this
model at all (routed experts are 92.5% of the 34.8B parameters), so it is a parameter
here rather than a constant — see `MESH_PRECISION` in `model_config.py`.
"""
from __future__ import annotations

import math
import os

import torch
from loguru import logger

import ttnn
from models.common.modules.moe.tt_moe_gate import TTMoEGate
from models.common.modules.moe.tt_moe_gate_config import TTMoEGateConfig

# Each routed-expert recipe is an explicit (gate/up, down) dtype pair rather than a
# single dtype plus a rule. down_proj carries the block's residual-scale output and is
# the error-sensitive projection of the three, so the recipes that spend precision
# spend it there first -- and making that visible in the table beats hiding it in an
# `if dtype is bfp4` branch that silently collapses two recipes into one.
# ttnn.linear defaults to LoFi on Blackhole, which truncates the bf16 mantissa in the
# matmul and costs ~2% PCC here -- measured: bf16 activations AND bf16 weights only cost
# 0.9998 against an fp32 reference, so anything materially below that is arithmetic, not
# quantization. HiFi4 with fp32 accumulation closes it. The experts are the whole model's
# compute, so this is the single most consequential precision knob in the block.
# The mesh axis experts would shard along under expert parallelism. Every mesh this
# model runs on is (1, N), so the device axis is 1. This was previously read out of a
# per-mesh MoE YAML through TTMoEDecodeConfig; that config object existed for the fused
# TTMoEDecode pipeline, and once the sparse path replaced it the only two fields still
# being read were this and a token-block size that nothing blocked on.
CLUSTER_AXIS = 1


def _reshape(t, shape):
    """ttnn.reshape, skipped when the tensor already has that shape.

    A no-op reshape is not free: it dispatches a ReshapeViewDeviceOperation like any
    other program, and decode pays ~8.4 us per program whatever it does. Three of these
    per layer are no-ops at batch 1 (the gather output is already [1, k, 1, H], and the
    ids and activation already carry the target shape), which is 120 programs across the
    model for nothing.
    """
    return t if list(t.shape) == list(shape) else ttnn.reshape(t, shape)


def _expert_compute_config(device, fidelity=None):
    """Compute config for the expert matmuls. LoFi by default; ORNITH_EXPERT_FIDELITY overrides.

    LoFi, not HiFi4: bfp4 weights carry 4 mantissa bits, so HiFi4's extra passes
    have nothing to resolve. Measured in the Qwen3-Coder-30B-A3B reference as both
    faster and very slightly MORE accurate at bfp4.

    That inherited claim was TESTED here on the graded gate, because fidelity truncates
    the ACTIVATION as well as the weight and the activation is bf16, not bfp4. Measured on
    the full 40-layer model, P150, bfp4 routed experts, tests/test_model.py logit PCC
    p0/p1/p2 (see scratchpad/REPORT_l22.md):
        lofi (default)  0.97174 / 0.97134 / 0.94334   top-8 6/8 8/8 6/8
        hifi2           0.97174 / 0.97134 / 0.94334   BIT-IDENTICAL to lofi -- with a
                        bfp4 srcB the second fidelity phase has no mantissa bits left to
                        add, so HiFi2 degenerates to LoFi here.
        hifi4           0.97348 / 0.96993 / 0.92605   top-8 7/8 7/8 7/8 -- +0.0017 on
                        prompt 0 but -0.0173 on prompt 2, i.e. a loss where the gate binds.
    So the inherited claim holds on the graded metric and LoFi stays the default.

    fp32_dest_acc_en is OFF and must stay off. It halves the matmul dest from 8 tiles
    to 4, which corrupts expert output on Blackhole (tt-metal #49068). This port had it
    ON, inherited from a generic "high fidelity" config.
    """
    if fidelity is None:
        name = os.environ.get("ORNITH_EXPERT_FIDELITY", "lofi").strip().lower()
        if name not in _MOE_FIDELITIES:
            raise ValueError(f"ORNITH_EXPERT_FIDELITY={name!r} is not one of {sorted(_MOE_FIDELITIES)}")
        fidelity = _MOE_FIDELITIES[name]
    return ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=fidelity,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )


_MOE_FIDELITIES = {
    "lofi": ttnn.MathFidelity.LoFi,
    "hifi2": ttnn.MathFidelity.HiFi2,
    "hifi4": ttnn.MathFidelity.HiFi4,
}


def _hifi():
    """Compute config for the SHARED EXPERT's matmuls. Defaults to LoFi, measured.

    Despite the name this is not the router -- the router goes through TTMoEGate, which
    already defaults to HiFi2 WITH fp32 accumulation and is therefore already in the
    correct regime. This function covers the shared expert's gate projection and its
    matmuls, and nothing else.

    It used to be HiFi4 with fp32_dest_acc_en=False, which is the worst available
    combination at this K. Measured against an exact fp64 product of the same bf16
    inputs, with fp32_dest_acc_en=False higher fidelity is WORSE than LoFi past K~1024 --
    1.7x worse at K=2048 with a ~2.8% scale bias -- because the extra fidelity passes
    accumulate into a bf16 destination. With fp32_dest_acc_en=True fidelity behaves as
    intended (HiFi4 ~16x better than LoFi, error flat in K).

    Turning fp32 accumulation ON is not available here: it halves the matmul dest from 8
    tiles to 4 and corrupts expert output on Blackhole (tt-metal #49068). Measured on the
    full model, enabling it moved prompt 1 up (0.94212 -> 0.96125) and prompt 2 sharply
    DOWN (0.93442 -> 0.90373), which is that corruption. So the way out of the bad regime
    is to drop the fidelity rather than to add the accumulator.

    Selectable via ORNITH_MOE_FIDELITY.

    The knob exists because these are the model's ONLY high-fidelity matmuls, and on
    Blackhole high fidelity plus `matmul_block` is a suspect combination:
    `matmul_block` routes unconditionally through
    `matmul_block_math_dynamic_throttle` (matmul.h:246-266), which reads a
    firmware-written word and, when its low bit is set, reprograms the MOP for
    MM_THROTTLE_MAX=5 while passing MATH_FIDELITY through UNGUARDED -- and the LLK's own
    test skips exactly that combination, because "throttle levels 4/5 do not increment
    the fidelity phase per LLK; only correct for LoFi (single-phase)" and "for a
    high-fidelity (multi-phase) matmul this collapses the extra phases and yields ~half
    the result" (tt-llk/tests/python_tests/test_matmul_custom.py:216-228).

    Since the throttle word is power/thermal state, that would be intermittent and
    immune to synchronisation -- the signature in NONDETERMINISM.md. Setting this to
    lofi takes the model off the suspect path entirely, which makes it the discriminating
    experiment rather than a tuning option.
    """
    name = os.environ.get("ORNITH_MOE_FIDELITY", "lofi").strip().lower()
    if name not in _MOE_FIDELITIES:
        raise ValueError(f"ORNITH_MOE_FIDELITY={name!r} is not one of {sorted(_MOE_FIDELITIES)}")
    # fp32_dest_acc_en is OFF by default here, matching _expert_compute_config -- but for a
    # different and much weaker reason, which is why it is a knob.
    #
    # The experts must keep it off: it halves the matmul dest from 8 tiles to 4 and
    # corrupts expert output on Blackhole (tt-metal #49068). The router and shared expert
    # inherited that False without a recorded rationale, and there is a measured reason to
    # doubt it: with bf16 operands, extra fidelity passes accumulate into a bf16 dest, so
    # HiFi's additional passes can add rounding rather than remove it. Measured standalone
    # on a 512x2048x2048 bf16 matmul with fp32_dest_acc_en=False, higher fidelity was
    # WORSE against an fp32 torch reference:
    #     LoFi  mean_abs_err 0.835   HiFi2 1.432   HiFi4 1.455
    # which is the same inversion the full model shows (hifi4 buys nothing over hifi2).
    # This file's own notes say "HiFi4 with fp32 accumulation closes it", and every other
    # high-fidelity config in the stack -- qwen36's attention and GDN, and the GDN
    # delta-rule kernel -- sets fp32_dest_acc_en=True. _hifi() is the odd one out.
    #
    # ORNITH_MOE_FP32_ACC=1 turns it on. Treat a PCC collapse as #49068 biting the shared
    # expert and revert; treat an improvement as the inversion above being the cause.
    fp32_acc = os.environ.get("ORNITH_MOE_FP32_ACC") == "1"
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=_MOE_FIDELITIES[name],
        math_approx_mode=False,
        fp32_dest_acc_en=fp32_acc,
        packer_l1_acc=False,
    )


def _sparse_matmul_config(m: int, n: int, in0_block_w: int = 1):
    """Spread N over the largest usable slice of an 8x8 grid.

    `in0_block_w` left at 1 feeds the kernel ONE tile of the inner dimension per pass.
    Measured here at gate_up's decode shape: 489.7 us at 1, 73.4 at 64 -- 6.7x, the
    largest single lever in the expert path. Both upstream sparse_matmul callers
    (gemma4, gpt_oss) and the Qwen3-Coder reference ship the default 1.

    `out_subblock_w` must divide `out_block_w`, or the compute kernel bakes
    in1_num_subblocks = out_block_w // out_subblock_w == 0, never pushes cb_out, and
    the in1 writer parks forever on cb_out.wait_front() -- a silent device hang
    (see models/demos/gpt_oss/tt/experts/config.py). out_subblock_w=1 divides
    everything, so the pair below is always legal.
    """
    n_tiles = math.ceil(n / 32)
    best_cores, best_cx, best_cy = 1, 1, 1
    for num_cores in range(1, min(65, n_tiles + 1)):
        if n_tiles % num_cores:
            continue
        for cy in range(1, 9):
            if num_cores % cy == 0:
                cx = num_cores // cy
                if cx <= 8 and num_cores > best_cores:
                    best_cores, best_cx, best_cy = num_cores, cx, cy
                    break
    # CEILING, not floor: the kernel computes num_blocks_x = ceil(Nt / per_core_N) and
    # asserts it fits in the grid, so a floor here overflows whenever best_cores does
    # not divide n_tiles.
    per_core_n = -(-n_tiles // best_cores)
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(best_cx, best_cy),
        in0_block_w=in0_block_w,
        out_subblock_h=1,
        out_subblock_w=1,
        out_block_h=1,
        out_block_w=per_core_n,
        per_core_M=max(32, m) // 32,
        per_core_N=per_core_n,
        fuse_batch=False,
        fused_activation=None,
        mcast_in0=True,
    )


def _decode_matmul_config(k: int, n: int, max_cores: int = 64, in0_cap: int = 8):
    """Tuned 1D-mcast config for a skinny (M=1) decode projection.

    ttnn's heuristic picks in0_block_w=1 for these, which feeds the inner dimension one
    tile at a time. Measured on Blackhole, bfloat8_b weights, traced, against the
    heuristic:

        shared gate/up [2048 x 512]   28.97 -> 11.53 us   (16 cores, in0_block_w 8)
        shared down    [512 x 2048]   11.54 ->  6.07 us   (64 cores, in0_block_w 8)
        GDN o_proj     [4096 x 2048]  73.75 -> 32.61 us   (64 cores, in0_block_w 8)

    Returns None when no grid of at least 8 cores divides Nt; the caller then keeps the
    heuristic, which is the right answer for the shapes that land there (the fused GDN
    projection at Nt=386 already runs at 345 GB/s, and lm_head's heuristic beat every
    config in the sweep).
    """
    nt, kt = max(1, n // 32), max(1, k // 32)
    cores = min(max_cores, nt)
    while cores > 1 and nt % cores:
        cores -= 1
    if cores < 8:
        return None
    ibw = min(in0_cap, kt)
    while ibw > 1 and kt % ibw:
        ibw -= 1
    per_core_n = nt // cores
    cy = min(cores, 8)
    while cores % cy:
        cy -= 1
    sub_w = min(per_core_n, 8)
    while per_core_n % sub_w:
        sub_w -= 1
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(cores // cy, cy),
        in0_block_w=ibw,
        out_subblock_h=1,
        out_subblock_w=sub_w,
        out_block_h=1,
        out_block_w=sub_w,
        per_core_M=1,
        per_core_N=per_core_n,
        fuse_batch=False,
        fused_activation=None,
        mcast_in0=True,
    )


def _largest_divisor_at_most(n: int, cap: int) -> int:
    """Largest d <= cap with n % d == 0 (at least 1)."""
    d = max(1, min(cap, n))
    while d > 1 and n % d:
        d -= 1
    return d


# Core count for a routed sparse matmul.
#
# The Ornith-1.0 port sizes this from how many experts the call can ACTIVATE (its op does
# `active_experts` K-sweeps per output block) and measured 8 cores with a wide output
# block about 40% FASTER than the 32/64-core per_core_N=1 form at 8 active experts.
# That does not transfer to this port and the sweep says so plainly -- decode, P150,
# pure execute_trace, one knob driving both phases:
#
#     cores      8        16       32       64
#     decode   38.60 ms 35.44    34.28    34.21
#     TTFT    464.2 ms  376.9    359.4    373.1
#
# The reason is the path, not the tuning: Ornith-1.0 scans the sparsity, so its decode op
# really does sweep every expert slot and narrow-and-wide pays. This port takes the
# INDEXED/gather path, where the op visits only the top-8 and the output group axis is
# already compact, so there is no slot scan left to trade against -- what remains wants
# the widest grid it can fill. Decode therefore keeps the divisor-of-Nt form, and only
# PREFILL, whose ~162-expert union does scan, takes the active-expert rule (which lands
# it on 32 cores and is worth ~5% of TTFT over the previous 64).
SPARSE_CORES_PER_ACTIVE = {"gate_up": 2, "down": 4}
SPARSE_MIN_CORES = 8
SPARSE_MAX_CORES = 32
#: Decode fills the grid instead: the largest divisor of Nt at or below this.
SPARSE_DECODE_CORES = 64
# in0_block_w cap for the packed gate/up matmul, keyed on whether the call got more than
# the minimum core count. At 8 active experts a 32-tile inner block beats the whole tiled
# K; at a prefill group's ~162 the whole K wins by a wider margin. One value cannot serve
# both, and the bound that picks the core count picks this too.
SPARSE_GATE_UP_IN0_BLOCK_W = {False: 32, True: 64}


def _sparse_cores(role: str, active_experts: int | None) -> int:
    """Target cores. `active_experts=None` means decode: fill the grid."""
    override = os.environ.get("ORNITH_SPARSE_CORES")
    if override:
        return int(override)
    if active_experts is None:
        return SPARSE_DECODE_CORES
    per = SPARSE_CORES_PER_ACTIVE[role]
    return max(SPARSE_MIN_CORES, min(SPARSE_MAX_CORES, active_experts // per))


def _tuned_sparse_matmul_config(m, n, k, role, active_experts, grid_y=8):
    """Sparse-matmul program config sized from the ACTIVE expert bound.

    The sparse factory requires mcast_in0, `Kt % in0_block_w == 0`, and output blocks that
    exactly tile a rectangle of the grid -- so the core count must divide Nt exactly. Both
    the core count and in0_block_w are targets, reduced to the largest legal divisor.
    """
    n_tiles = max(1, math.ceil(n / 32))
    k_tiles = max(1, math.ceil(k / 32))
    m_tiles = max(1, math.ceil(m / 32))

    cores = _largest_divisor_at_most(n_tiles, max(1, _sparse_cores(role, active_experts)))
    per_core_n = n_tiles // cores
    # Fill one grid axis first (a 1 x cores column), widening only when cores exceeds it.
    cy = min(cores, grid_y)
    while cores % cy:
        cy -= 1
    cx = cores // cy

    block_w = per_core_n
    sub_w = _largest_divisor_at_most(block_w, 8)  # dst register file: h*w <= 8
    sub_h = _largest_divisor_at_most(m_tiles, max(1, 8 // sub_w))
    want = (
        SPARSE_GATE_UP_IN0_BLOCK_W[cores > SPARSE_MIN_CORES]
        if role == "gate_up" and active_experts is not None
        else k_tiles
    )
    in0_block_w = _largest_divisor_at_most(k_tiles, max(1, want))

    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(cx, cy),
        in0_block_w=in0_block_w,
        out_subblock_h=sub_h,
        out_subblock_w=sub_w,
        out_block_h=m_tiles,
        out_block_w=block_w,
        per_core_M=m_tiles,
        per_core_N=per_core_n,
        fuse_batch=False,
        fused_activation=None,
        mcast_in0=True,
    )


_RECIPES = {
    "bfloat4_b": (ttnn.bfloat4_b, ttnn.bfloat4_b),
    "bfloat4_bfloat8": (ttnn.bfloat4_b, ttnn.bfloat8_b),
    "bfloat8_b": (ttnn.bfloat8_b, ttnn.bfloat8_b),
    "bfloat16": (ttnn.bfloat16, ttnn.bfloat16),
}


def _split_fused_experts(state_dict, num_experts, hidden, inter):
    """Unpack this checkpoint's PRE-STACKED, PRE-FUSED expert tensors.

    Unlike the per-expert `experts.<i>.gate_proj.weight` layout most MoE checkpoints
    use, Ornith ships two tensors per layer covering all 256 experts at once:

        mlp.experts.gate_up_proj  [E, 2*inter, hidden]   gate and up fused on the out dim
        mlp.experts.down_proj     [E, hidden, inter]

    Both are nn.Linear-shaped ([out, in]) and are transposed here, because every
    consumer downstream computes ``x @ w``.

    Returns (w_gate, w_up, w_down), each [E, in, out].
    """
    gu = state_dict["experts.gate_up_proj"]
    dn = state_dict["experts.down_proj"]
    assert gu.shape == (num_experts, 2 * inter, hidden), f"gate_up_proj {tuple(gu.shape)}"
    assert dn.shape == (num_experts, hidden, inter), f"down_proj {tuple(dn.shape)}"

    gu = gu.to(torch.bfloat16).transpose(1, 2).contiguous()  # [E, hidden, 2*inter]
    # Gate occupies the first half of the fused out dim, up the second -- the
    # convention this architecture's reference implementation uses. test_moe_block.py
    # checks the halves are not swapped by comparing against the HF module directly,
    # because a swap still produces plausible-looking output.
    w_gate = gu[:, :, :inter].contiguous()
    w_up = gu[:, :, inter:].contiguous()
    w_down = dn.to(torch.bfloat16).transpose(1, 2).contiguous()  # [E, inter, hidden]
    return w_gate, w_up, w_down


class OrnithMoEBank:
    """Routed-expert weights for ALL layers, held by one `_SparseExperts`.

    Expert weights are stacked [L, E, H, N] and the layer is selected at forward time
    with `layer_id`, so one instance serves the whole stack rather than 40 copies of
    the same scratch buffers and expert map.

    The per-layer router and shared expert stay in `OrnithMoE`; only the routed experts
    (92.5% of the parameters) live here.
    """

    def __init__(self, mesh_device, args, layer_states: list[dict], tensor_cache_path=None, tt_ccl=None):
        self.mesh_device = mesh_device
        self.num_layers = len(layer_states)

        gates, ups, downs = [], [], []
        for st in layer_states:
            g, u, d = _split_fused_experts(st, args.num_experts, args.dim, args.moe_intermediate_size)
            gates.append(g)
            ups.append(u)
            downs.append(d)
        w0 = torch.stack(gates)
        w1 = torch.stack(ups)
        w2 = torch.stack(downs)

        self.experts_per_device = args.num_experts // max(mesh_device.get_num_devices(), 1)
        self.unfused = _SparseExperts(mesh_device, args, w0, w1, w2, CLUSTER_AXIS, tensor_cache_path, tt_ccl=tt_ccl)
        logger.debug(
            f"MoE bank: {self.num_layers} layers x {args.num_experts} experts "
            f"({args.routed_expert_dtype}) on {args.device_name}"
        )

    def forward(self, x, scores, indices, layer_id: int):
        return self.unfused.forward(x, scores, indices, layer_id)


class _SparseExperts:
    """Routed experts via `ttnn.sparse_matmul`: the experts a token did not pick are
    genuinely skipped, not computed and multiplied by zero. Replaces a dense loop over
    all 256 local experts that read 18.1 GB of bfp4 weights per token at batch 1 --
    62 GB/s effective, ~12% of peak, 3.4 tok/s.

    `in0_block_w` IS THE LEVER, and it is not the one the prior art points at. Measured
    on this box at Ornith's shapes (H=2048, inter=512, bfp4 weights, one die, gate_up
    at E=32, 20 iterations):

        in0_block_w      1        2        8       64
              gate_up  489.7us  257.9us  104.6us  73.4us

    6.7x, from how many tiles of the inner dimension the kernel takes per pass. Both
    upstream callers (models/demos/gemma4, models/demos/gpt_oss) and the Qwen3-Coder
    reference on this box leave it at 1. Expert COUNT barely matters next to it:
    32 -> 256 experts is 490 -> 544 us at blk=1, +11% for 8x the experts, because the
    kernel already skips inactive slots. That is why this model does not need expert
    parallelism to be viable on one die, and why the first port looked like it did.

    GATHER IS 10x THE SCAN AND FLAT IN EXPERT COUNT. Measured pair cost (gate_up +
    down, one die, tuned blk, 20 iterations):

        experts/die      64      128      256
          scan       259.8us  439.4us  799.8us      (nnz=None, per-slot validity scan)
          gather      79.5us   78.7us   80.2us      (indices=, visits top-8 only)

    Gather on ONE die beats the scan on FOUR (79.5 vs 259.8 us at 64 experts/die), so
    the MoE does not need expert parallelism to be viable here -- 40 layers at 80 us
    is a 316 tok/s ceiling from the expert path, which puts the bottleneck elsewhere.

    THE `nnz` CONTRACT IS A DEVICE HANG, not a validation error. `sparse_matmul` bakes
    `nnz` in as a compile-time arg: the in0-mcast receivers loop exactly `nnz` times
    while the sender multicasts once per *actual* non-zero sparsity entry, so any
    `nnz != count_nonzero(sparsity)` deadlocks in `noc_semaphore_wait` -- silently,
    unless the watcher is on (tt-metal #45943, #45052). Decode routing weights are a
    softmax over top-k whose small entries FLUSH TO ZERO in bfloat16, so the true
    count is frequently < top_k and no static value is correct. Decode passes
    `nnz=None` (inferred on device) or takes the gather path, which has no nnz at all.
    Prefill's mask is all-ones by construction, so there the count is exact.

    (Not to be confused with a cold kernel compile, which these shapes make expensive
    enough to look like a hang: the first sparse_matmul of a given shape on an empty
    TT_METAL_CACHE took over 7 minutes at 100% host CPU and 0.4 s once warm. Budget
    for it in timeouts rather than reaching for the watcher.)

    GATHER (`indices=`) is the better decode path and is absent from the reference,
    whose tt-metal predates it: hand the op the top-k expert ids as a uint16 stick and
    the kernels visit only those groups -- no 256-slot validity scan, no per-slot
    multicast, and no nnz contract to violate. The output's group axis is COMPACT
    (length top_k), so routing weights apply straight from the router's own top-k
    scores and the dense [T, 256] routing matrix is never built at all. The op's
    program cache re-dispatches a changed indices buffer, so this still traces.
    """

    # Prefill groups tokens into 32-row tiles: sparse_matmul folds the group dimension
    # into M, so a larger chunk grows num_blocks_y and can overflow the core grid.
    #
    # 32 was chosen on the EXPERT UNION alone -- per-token work is live_experts, which
    # rises 162 -> ~251 going from 32 to 128 tokens (see _prefill's docstring), measured
    # at short ISL where this loop runs a handful of times per layer. That cost function
    # leaves out per-iteration OVERHEAD, and tools/prefill_profile.py measures the loop at
    # 15 device ops per iteration of which only 2 are matmuls. The iteration count is
    # ISL/CHUNK per layer -- 4 at ISL 128, but 4096 at ISL 131072 -- so at long prompts
    # the omitted term dominates and the optimum can move. ORNITH_MOE_CHUNK sweeps it;
    # the default keeps the measured short-ISL behaviour exactly.
    # MEASURED, one die, best of 2, prefill wall clock (32 -> 64):
    #   ISL  2048   402.9 -> 428.9 tok/s  (+6.5%)   greedy token identical (198)
    #   ISL  8192   369   -> 380   tok/s  (+3.1%)   [4-layer run x10]
    #   ISL 16384   363   -> 374   tok/s  (+3.0%)   [4-layer run x10]
    # 128 was also swept and is WORSE than 64 (415.7 tok/s at ISL 2048): the union
    # penalty catches back up. The gain SHRINKS with length rather than holding, because
    # other length-dependent costs grow and dilute this loop's share -- so this is worth
    # ~3% at long prompts, not the ~6.5% the short-ISL number suggests. Quote the length.
    CHUNK = int(os.environ.get("ORNITH_MOE_CHUNK", "64"))

    #: The prefill/decode path threshold, deliberately NOT CHUNK. It is the sparse tile
    #: height, so raising CHUNK cannot push a 32..CHUNK-1-token prompt onto the decode
    #: path and silently change which kernel short prefills use.
    MIN_PREFILL_T = 32

    #: dtype of the routed-expert intermediates (the packed gate/up result and the down
    #: output). Ornith-1.0's precision policy ships BFP8 here. Traced op cost is flat up
    #: to ~256 KB and rises after, so halving these only pays where they are large.
    ACT_DTYPE = {"bfloat16": ttnn.bfloat16, "bfloat8_b": ttnn.bfloat8_b}[
        os.environ.get("ORNITH_EXPERT_ACT", "bfloat16")
    ]

    def __init__(self, mesh_device, args, w0, w1, w2, cluster_axis, cache=None, tt_ccl=None):
        self.mesh_device = mesh_device
        self.cluster_axis = cluster_axis
        self.args = args
        self.tt_ccl = tt_ccl
        # DATA PARALLEL, not expert parallel, on multi-chip. Expert parallelism exists to
        # cut per-die expert count, and with the gather path that buys nothing: the pair
        # costs 79 us at 64, 128 or 256 experts per die, because only the top-8 are ever
        # visited. All it leaves behind is an all-reduce per layer -- 40 collectives per
        # forward, which is also where both multi-chip meshes were failing ("Unsupported
        # fabric config", and previously an all-reduce deadlock).
        #
        # The whole model is ~18.2 GB (17.1 GB of bfp4 experts + ~1.1 GB of bf8
        # everything else) against 30 GB of usable DRAM per chip, so every die can hold
        # a complete copy, and one die per user (tools/serve_lanes.sh) is still the
        # highest-AGGREGATE deployment.
        #
        # But this is no longer the multi-die DEFAULT, and the paragraph above overstated
        # its case. "Expert parallelism buys nothing" was a DECODE measurement -- the
        # gather path costs 79 us per expert pair at 64, 128 or 256 experts/die because
        # only the top-8 are visited. Prefill routes a 4096-token chunk to essentially
        # all 256, so sharding the bank really does quarter that work: with TP held
        # fixed, EP is worth 2.60x at ISL 4096 and 2.55x at 16384.
        #
        # The decision now lives in ONE place for the whole model, so the layer config
        # and the expert bank cannot disagree about whether the mesh is sharded --
        # OrnithModelArgs.expert_parallel, set by model_config.multidie_parallel_enabled
        # (default on for any multi-die mesh, ORNITH_MOE_EP / ORNITH_TP=0 to override).
        # Reading the env var independently here is what let the two diverge.
        mesh_n = getattr(args, "mesh_devices", mesh_device.get_num_devices())
        ep = getattr(args, "expert_parallel", None)
        if ep is None:  # a bare Qwen36ModelArgs (tests build the block standalone)
            ep = os.environ.get("ORNITH_MOE_EP", "0") == "1" and mesh_n > 1
        self.expert_parallel = bool(ep)
        self.num_devices = mesh_n if self.expert_parallel else 1
        self.experts_per_device = args.num_experts // max(self.num_devices, 1)
        self.hidden_size = args.dim
        self.intermediate_size = args.moe_intermediate_size
        self.top_k = args.num_experts_per_tok
        gate_up_dt, down_dt = _RECIPES[args.routed_expert_dtype]
        nd, epd, H, N = self.num_devices, self.experts_per_device, self.hidden_size, self.intermediate_size

        # Gather needs one call per user (the ids are a single stick) and one id space
        # (no EP remap), so it is the single-die decode path. Expert-parallel meshes
        # take the sparsity scan, whose per-die count is data-dependent -- which is
        # exactly the case `nnz=None` exists for.
        self.use_gather = (nd == 1) and os.environ.get("ORNITH_MOE_GATHER", "1") == "1"

        # dim 0 = device: each device gets its own contiguous expert block, matching the
        # `sequential` expert map, and the expert axis stays unsharded so it can be
        # indexed uniformly.
        # No mapper == replicate across the MeshDevice. Under expert parallelism, shard
        # dim 0 so each die gets its own contiguous expert block (matching the
        # `sequential` expert map) while the expert axis itself stays unsharded and can
        # be indexed uniformly.
        shard = ttnn.ShardTensorToMesh(mesh_device, dim=0) if self.expert_parallel else None

        def upload(src, dtype, name, shape):
            per_layer = src[0]  # [E, ...]
            grouped = per_layer.reshape(nd, epd, *per_layer.shape[1:]).contiguous()
            t = ttnn.as_tensor(
                grouped,
                dtype=dtype,
                device=mesh_device,
                mesh_mapper=shard,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                # The expert-parallel and replicated uploads are DIFFERENT tensors (the
                # host side is reshaped to [nd, epd, ...] and sharded on dim 0), and
                # as_tensor ignores mesh_mapper on a cache hit -- so they must not share a
                # cache file. Without the suffix, flipping ORNITH_MOE_EP on a warm cache
                # reloads the other layout and dies in the reshape below on volume.
                cache_file_name=(
                    str(cache / f"moe.sparse.{name}{f'.ep{nd}' if self.expert_parallel else ''}") if cache else None
                ),
            )
            return ttnn.reshape(t, shape)

        # gate and up packed on the output dim: the program config parallelises only
        # over N, so N tiles cap the usable cores. Packing doubles gate/up's core count.
        # UP first, GATE second. ttnn.swiglu computes `first * silu(second)`, and the
        # activation belongs on the gate, so the pack order is what makes one fused op
        # replace slice + slice + silu + multiply. Packing gate-first and calling swiglu
        # would compute gate * silu(up): same shapes, same cost, silently wrong.
        up_gate = torch.cat([w1, w0], dim=-1)  # [1, E, H, 2*inter]
        self.up_gate_proj = upload(up_gate, gate_up_dt, "up_gate", (1, epd, H, 2 * N))
        # NOT chunked over the expert axis. An earlier port split this at 64 experts to
        # satisfy "num_blocks_total <= num_cores_available"; that check is computed from
        # the M and N TILE counts only (sparse_matmul_multicore_reuse_mcast_1d_optimized
        # .cpp: num_blocks_y = ceil(Mt/per_core_M), num_blocks_x = ceil(Nt/per_core_N)).
        # The expert axis is a batch loop and never enters it, so 256 experts are fine.
        self.down_proj = upload(w2, down_dt, "down", (1, epd, N, H))

        # Each device's own global expert ids, for building its slice of the routing.
        ids = torch.arange(nd * epd, dtype=torch.float32).reshape(nd, 1, 1, epd)
        self.expert_ids = ttnn.as_tensor(
            ids,
            dtype=ttnn.bfloat16,
            device=mesh_device,
            mesh_mapper=shard,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        # All-ones [1,1,1,E_local] ROW_MAJOR. Prefill uses it as its real mask (see
        # _prefill); gather requires a sparsity operand but never reads it.
        self.sparsity_base = ttnn.from_torch(
            torch.ones(1, 1, 1, epd, dtype=torch.bfloat16),
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.bfloat16,
            device=mesh_device,
        )
        self.compute_config = _expert_compute_config(mesh_device)
        # k is the inner dim: H for gate/up, `inter` for down.
        self.gate_up_cfg_decode = _tuned_sparse_matmul_config(1, 2 * N, H, "gate_up", None)
        self.down_cfg_decode = _tuned_sparse_matmul_config(1, H, N, "down", None)
        # A 32-token chunk draws 32 x top_k times from E experts; the expected distinct
        # union is what the prefill config should be sized for, not top_k.
        union = int(epd * (1.0 - (1.0 - 1.0 / epd) ** (self.CHUNK * self.top_k)))
        self.gate_up_cfg_prefill = _tuned_sparse_matmul_config(self.CHUNK, 2 * N, H, "gate_up", union)
        self.down_cfg_prefill = _tuned_sparse_matmul_config(self.CHUNK, H, N, "down", union)
        self.tile = ttnn.Tile([32, 32])
        logger.debug(
            f"sparse experts: {epd}/device x {nd} devices, {args.routed_expert_dtype}, "
            f"decode={'gather' if self.use_gather else 'scan'}"
        )

    def _routing_matrix(self, scores, indices, T):
        """(scores, indices)[T,1,1,k] -> dense per-(token, local expert) weights
        `[1, 1, T, E_local]`. Zero where this device does not own the chosen expert.
        Only the scan path needs this; gather reads the top-k scores directly."""
        epd = self.experts_per_device
        idx = ttnn.typecast(indices, ttnn.bfloat16)
        idx = ttnn.reshape(idx, (T, 1, idx.shape[-1], 1))  # [T,1,k,1]
        sc = ttnn.reshape(scores, (T, 1, scores.shape[-1], 1))
        hit = ttnn.eq(idx, self.expert_ids)  # [T,1,k,epd]
        wt = ttnn.sum(ttnn.multiply(hit, sc), dim=-2, keepdim=True)  # [T,1,1,epd]
        for t in (idx, sc, hit):
            ttnn.deallocate(t)
        return ttnn.reshape(wt, (1, 1, T, epd))

    def forward(self, x, scores, indices, layer_id: int):
        """x: [1,1,T,H] or [B,T,H]; scores/indices [T,1,1,k] -> x's rank."""
        orig = list(x.shape)
        H = orig[-1]
        # Same rule as _run_blocks: tokens are every leading dim multiplied together.
        T = math.prod(orig[:-1])
        x4 = x if len(orig) == 4 and orig[-2] == T else ttnn.reshape(x, (1, 1, T, H))
        decode = T == 1 or T < self.MIN_PREFILL_T
        if decode and self.use_gather:
            out = self._decode_gather(x4, scores, indices, T, H)
        else:
            routing = self._routing_matrix(scores, indices, T)
            out = self._decode_scan(x4, routing, T, H) if decode else self._prefill(x4, routing, T, H)
            ttnn.deallocate(routing)
        out = self._reduce(out, H)
        # Back to the caller's shape, not a canonical one: at batch > 1 decode the layer
        # passes [B, 1, H], and returning [1, B, H] made the shared-expert add fail with
        # a binary_ng shape error two calls later.
        return out if list(out.shape) == orig else ttnn.reshape(out, orig)

    def _swiglu(self, fused, lead, half, memory_config=None):
        """SwiGLU over the packed [.., 2*inter] projection, for DECODE (M=1).

        Three ops where `ttnn.swiglu` would be one, and that is deliberate: at M=1 the
        fused op PADS the row axis to a full tile, returning [1, k, 32, inter] instead
        of [1, k, 1, inter]. The values on the real row are right (max abs err 0.017 vs
        torch), but the shape then drives 32 rows through the down projection. Prefill,
        whose M is already 32, uses the fused op -- see _prefill.
        """
        # UP is the FIRST half and GATE the second -- the pack order the fused
        # ttnn.swiglu in _prefill requires (`first * silu(second)`). Slicing these the
        # other way round still runs and still produces plausible logits; that is what
        # test_gate_up_halves_are_not_swapped exists to catch.
        up = ttnn.slice(fused, [0, 0, 0, 0], [1, lead, 1, half], memory_config=memory_config)
        gate = ttnn.slice(fused, [0, 0, 0, half], [1, lead, 1, 2 * half], memory_config=memory_config)
        ttnn.deallocate(fused)
        # SiLU as an activation on the gate operand, not its own program. Both operands are
        # slices of the same `fused` tensor, so they carry the same dtype -- the pairing a
        # fused activation needs. One fewer program per layer, 40 per token.
        out = ttnn.multiply(up, gate, input_tensor_b_activations=[ttnn.UnaryOpType.SILU], memory_config=memory_config)
        ttnn.deallocate(gate)
        ttnn.deallocate(up)
        return out

    def _decode_gather(self, x4, scores, indices, T, H):
        """Indexed/gather decode: only the top-k experts are visited, per user.

        `indices` must be a UINT16 ROW_MAJOR tensor occupying a single stick, so a
        multi-user step issues one call per user rather than one call for the batch.
        That is not a regression: the sparsity scan bills per (user, expert) slot and
        was measured to scale strictly linearly with users anyway -- 544 us at 1 user,
        4241 at 8, 16957 at 32 (E=256, blk=1) -- because each slot is its own M=1
        matmul and nothing amortises across users.
        """
        k, N = self.top_k, self.intermediate_size
        l1 = ttnn.L1_MEMORY_CONFIG
        # Two ops per layer, 80 over the model, and both are usually no-ops -- decode is
        # op-count-bound (485 programs per token, ~15.6 us of launch overhead each
        # against 7.1 ms of total kernel time), so a skippable op is worth skipping.
        ids_u16 = indices if indices.dtype == ttnn.uint16 else ttnn.typecast(indices, ttnn.uint16)
        if ids_u16.layout != ttnn.ROW_MAJOR_LAYOUT:
            ids_u16 = ttnn.to_layout(ids_u16, ttnn.ROW_MAJOR_LAYOUT)
        outs = []
        for u in range(T):
            ids_u = ids_u16 if T == 1 else ttnn.slice(ids_u16, [u, 0, 0, 0], [u + 1, 1, 1, k])
            ids_u = _reshape(ids_u, (1, 1, 1, k))
            x_u = x4 if T == 1 else ttnn.slice(x4, [0, 0, u, 0], [1, 1, u + 1, H])
            x_u = _reshape(x_u, (1, 1, 1, H))

            fused = ttnn.sparse_matmul(
                x_u,
                self.up_gate_proj,
                sparsity=self.sparsity_base,
                indices=ids_u,
                is_input_a_sparse=False,
                is_input_b_sparse=True,
                memory_config=l1,
                output_tile=self.tile,
                program_config=self.gate_up_cfg_decode,
                compute_kernel_config=self.compute_config,
                dtype=self.ACT_DTYPE,
            )
            # [.., k, 1, 2*inter] compact -> [1, k, 1, 2*inter]
            din = self._swiglu(ttnn.reshape(fused, (1, k, 1, 2 * N)), k, N)

            # Compact-A gather: A is indexed by SLOT i, B by the id at indices[i].
            down = ttnn.sparse_matmul(
                din,
                self.down_proj,
                sparsity=self.sparsity_base,
                indices=ids_u,
                is_input_a_sparse=True,
                is_input_b_sparse=True,
                memory_config=l1,
                output_tile=self.tile,
                program_config=self.down_cfg_decode,
                compute_kernel_config=self.compute_config,
                dtype=self.ACT_DTYPE,
            )
            ttnn.deallocate(din)
            # Scale each expert by ITS OWN routing weight, then sum over the k of them.
            # Scaling here rather than on the expert INPUT is deliberate: SwiGLU is not
            # homogeneous, so scaling the input costs ~2% PCC uniformly; after the down
            # projection it is exact, because that projection is linear.
            #
            # Tried and reverted: folding the scale and the sum into one matmul,
            # [1,1,1,k] @ [1,1,k,H]. It is four programs as written (the multiply, then
            # FillPad + FastReduceNC for the reduce, then a slice) and would be two. Both
            # ways of getting `down` into [1,1,k,H] SEGFAULT -- reshape because that axis
            # move changes the PADDED shape ([1,k,32,H] vs [1,1,32,H]) rather than being a
            # view, and permute because the resulting matmul has K=8, below one tile.
            sc_u = ttnn.reshape(
                scores if T == 1 else ttnn.slice(scores, [u, 0, 0, 0], [u + 1, 1, 1, k]),
                (1, k, 1, 1),
            )
            st = ttnn.multiply(_reshape(down, (1, k, 1, H)), sc_u)
            ttnn.deallocate(down)
            # The expert COMBINE. ttnn.sum's default compute config is not the library
            # default: generic_reductions.cpp passes default_fp32_acc=TRUE and HiFi4, so
            # this reduction runs with a 32-BIT DEST while the prefill spelling below
            # (fast_reduce_nc) runs with a 16-bit one. That asymmetry is invisible at the
            # call site, and fp32_dest_acc_en halves the matmul dest from 8 tiles to 4,
            # which is the mechanism tt-metal #49068 says corrupts expert output on
            # Blackhole. Passing the expert config makes decode numerically identical to
            # prefill and costs nothing. ORNITH_DECODE_SUM_DEFAULT=1 restores the old
            # implicit config for A/B testing.
            _sum_cfg = None if os.environ.get("ORNITH_DECODE_SUM_DEFAULT") == "1" else self.compute_config
            outs.append(ttnn.sum(st, dim=1, keepdim=True, compute_kernel_config=_sum_cfg))  # [1,1,1,H]
            ttnn.deallocate(st)
            if T != 1:
                ttnn.deallocate(ids_u)
                ttnn.deallocate(x_u)
        if ids_u16 is not indices:
            ttnn.deallocate(ids_u16)
        acc = outs[0] if T == 1 else ttnn.concat(outs, dim=2)
        if T != 1:
            for o in outs:
                ttnn.deallocate(o)
        return ttnn.reshape(acc, (1, 1, T, H))

    def _decode_scan(self, x4, routing, T, H):
        """Sparsity-scan decode, for expert-parallel meshes where gather's single id
        space does not hold. `nnz=None` throughout: under EP the number of locally-live
        experts is data-dependent and differs per die, and a TTNN mesh op is SPMD, so
        there is no single correct static value -- and a wrong one hangs the device."""
        epd, N = self.experts_per_device, self.intermediate_size
        l1 = ttnn.L1_MEMORY_CONFIG
        # Tokens ride the BATCH dims, not M: sparse_matmul indexes sparsity by batch, so
        # this is what makes the mask per-token. Free here -- decode's M is genuinely 1.
        x_b = ttnn.reshape(x4, (1, T, 1, H))
        sp_gu = ttnn.to_layout(ttnn.reshape(routing, (1, T, 1, epd)), ttnn.ROW_MAJOR_LAYOUT)
        fused = ttnn.sparse_matmul(
            x_b,
            self.up_gate_proj,
            sparsity=sp_gu,
            nnz=None,
            memory_config=l1,
            output_tile=self.tile,
            program_config=self.gate_up_cfg_decode,
            compute_kernel_config=self.compute_config,
            dtype=ttnn.bfloat16,
        )
        ttnn.deallocate(sp_gu)
        din = self._swiglu(ttnn.reshape(fused, (1, T * epd, 1, 2 * N)), T * epd, N)

        # is_input_b_sparse=False: with a sparse activation the op takes batch_length_A
        # (= T*E_local); leaving b_sparse at its default true would give it just E and
        # silently reject T > 1.
        sp_dn = ttnn.to_layout(ttnn.reshape(routing, (1, 1, T, epd)), ttnn.ROW_MAJOR_LAYOUT)
        down = ttnn.sparse_matmul(
            ttnn.reshape(din, (T, epd, 1, N)),
            self.down_proj,
            sparsity=sp_dn,
            nnz=None,
            is_input_a_sparse=True,
            is_input_b_sparse=False,
            memory_config=l1,
            output_tile=self.tile,
            program_config=self.down_cfg_decode,
            compute_kernel_config=self.compute_config,
            dtype=ttnn.bfloat16,
        )
        ttnn.deallocate(din)
        ttnn.deallocate(sp_dn)
        states = ttnn.multiply(ttnn.reshape(down, (T, epd, H)), ttnn.reshape(routing, (T, epd, 1)))
        ttnn.deallocate(down)
        states = ttnn.unsqueeze_to_4D(ttnn.sum(states, dim=1))
        return ttnn.reshape(states, (1, 1, T, H))

    def _prefill(self, x4, routing, T, H):
        """Tile-granular sparsity, ONE 32-token chunk at a time.

        The mask is the routing REDUCED OVER THE CHUNK -- an expert is live if any of
        the 32 tokens picked it -- with `nnz=None`. The count is data-dependent (an
        expert no token in the group picked reduces to exactly 0) and a static `nnz`
        that misses count_nonzero HANGS THE DEVICE, so it must be inferred on device.
        An all-ones mask makes nnz exact by construction and was what shipped first;
        it also computes all 256 experts for every token, which measured 1.67 ms
        (gate_up) + 0.83 ms (down) per chunk per layer -- about 400 ms of a 640 ms
        TTFT at 128 tokens. 32 tokens x top-8 over 256 experts leaves ~63% of experts
        live, so the real mask is worth ~1.6x for a mask reduction per chunk.

        Why 32 and not larger or smaller: work per chunk is live_experts x CHUNK, so
        per token it is just live_experts, which RISES with chunk size as the union
        saturates (162 live at 32 tokens, ~251 at 128). Smaller chunks cut the union
        but the tile is 32 rows, so anything under 32 pays the padding back with
        interest (8 tokens -> 57 live x 32 rows = 228/token against 162). 32 is the
        tile-aligned optimum, which is also where the Qwen3-Coder reference landed.

        The chunk LOOP is not optional. Folding every group into one call makes the
        down projection's output [1, E, T, H] all at once -- 128 MB at just 128 tokens,
        which is an L1 OOM and grows linearly with the prompt. One chunk at a time
        bounds it at [1, E, 32, H] regardless of prompt length, and it goes to DRAM:
        these prefill intermediates are far too big for L1 even chunked.

        EVERY intermediate here is placed in DRAM EXPLICITLY. ttnn defaults to L1, and
        these are [1, 256, 32, *] tensors -- 8 MB for a gate/up half, 32 MB for the
        scaled expert outputs. Interleaved across 110 banks that is only ~300 KB per
        bank, so it fits and nothing complains; it just leaves the GDN prefill kernel
        (which wants ~1.36 MB of circular buffers in a 1.5 MB L1) no room, and the run
        dies several ops later with "statically allocated circular buffers clash with
        L1 buffers" at whatever prompt length first pushes it over. Measured before
        this was made explicit: ISL 128 fits, ISL 512 does not.
        """
        epd, N = self.experts_per_device, self.intermediate_size
        dram = ttnn.DRAM_MEMORY_CONFIG
        C = self.CHUNK
        pad = (-T) % C
        if pad:
            x4 = ttnn.pad(x4, [(0, 0), (0, 0), (0, pad), (0, 0)], value=0.0)
            routing = ttnn.pad(routing, [(0, 0), (0, 0), (0, pad), (0, 0)], value=0.0)
        Tp = T + pad

        outs = []
        for s0 in range(0, Tp, C):
            x_c = ttnn.slice(x4, [0, 0, s0, 0], [1, 1, s0 + C, H], memory_config=dram)
            r_c = ttnn.slice(routing, [0, 0, s0, 0], [1, 1, s0 + C, epd], memory_config=dram)
            # [1,1,1,E]: which experts any of these 32 tokens picked. Routing weights
            # are softmax outputs and non-negative, so the sum cannot cancel a live
            # expert back to zero.
            sp_c = ttnn.to_layout(
                ttnn.sum(r_c, dim=2, keepdim=True, memory_config=dram), ttnn.ROW_MAJOR_LAYOUT, memory_config=dram
            )
            fused = ttnn.sparse_matmul(
                x_c,
                self.up_gate_proj,
                sparsity=sp_c,
                nnz=None,
                memory_config=dram,
                output_tile=self.tile,
                program_config=self.gate_up_cfg_prefill,
                compute_kernel_config=self.compute_config,
                dtype=ttnn.bfloat16,
            )
            ttnn.deallocate(x_c)
            width = fused.shape[-1]
            fused = ttnn.reshape(fused, (1, epd, C, width))
            # Fused: prefill's M is CHUNK=32, already a whole tile, so swiglu returns
            # [1, E, 32, inter] unpadded. Replaces slice + slice + silu + multiply.
            din = ttnn.swiglu(fused, memory_config=dram)
            ttnn.deallocate(fused)

            down = ttnn.sparse_matmul(
                din,
                self.down_proj,
                sparsity=sp_c,
                nnz=None,
                memory_config=dram,
                output_tile=self.tile,
                program_config=self.down_cfg_prefill,
                is_input_a_sparse=True,
                compute_kernel_config=self.compute_config,
                dtype=ttnn.bfloat16,
            )
            ttnn.deallocate(din)
            # Scale each expert's contribution by its routing weight (zero for the 248
            # experts this token did not pick), then sum over the expert axis.
            st = ttnn.multiply(
                ttnn.reshape(down, (1, epd, C, H)),
                ttnn.permute(r_c, (0, 3, 2, 1), memory_config=dram),
                memory_config=dram,
            )
            ttnn.deallocate(down)
            ttnn.deallocate(r_c)
            ttnn.deallocate(sp_c)
            # The expert COMBINE. Stays `fast_reduce_nc`, and that is now MEASURED rather
            # than inherited.
            #
            # `deepseek_moe_fast_reduce_nc` looked like a free accuracy win: ornith-1.0
            # measured its accumulation at PCC 0.999999 against a float32 sum of 256
            # bfloat16 expert blocks where `fast_reduce_nc` gives 0.999409, and prompt 2
            # here sits at 0.94334 against a 0.95 bar. Summing 256 bfp4 expert
            # contributions is exactly that shape, so it was worth trying.
            #
            # IT IS WORSE END TO END. A/B against the cached HF reference, same build,
            # baseline reproduced to five decimals first (handoff/log_ab_reduce.txt):
            #
            #   prompt        fast_reduce_nc   deepseek_moe_fast_reduce_nc
            #   0                0.97174              0.96974
            #   1                0.97134              0.97016
            #   2                0.94334              0.93604
            #
            # Every prompt regressed and prompt 2 moved AWAY from the bar. Top-1 stayed
            # correct 3/3 either way, and top-8 overlap actually improved on prompts 0 and
            # 2 (6/8 -> 7/8), so the op is not broken -- it is a different accumulation
            # order whose op-level accuracy does not survive to the logits. A more accurate
            # reduce in isolation is not a more accurate model.
            outs.append(ttnn.experimental.fast_reduce_nc(st, dims=[1]))
            ttnn.deallocate(st)

        acc = outs[0] if len(outs) == 1 else ttnn.concat(outs, dim=2, memory_config=dram)
        if len(outs) > 1:
            for o in outs:
                ttnn.deallocate(o)
        states = ttnn.reshape(acc, (1, 1, Tp, H))
        return states if not pad else ttnn.slice(states, [0, 0, 0, 0], [1, 1, T, H], memory_config=dram)

    def _reduce(self, acc, H):
        if self.num_devices == 1:
            return acc  # replicated experts: every die already has the full sum
        # Partial sums: each device computed only its own experts. All-reduce keeps the
        # residual stream replicated, which is the layer's input/output contract.
        # TOPOLOGY AND LINK COUNT MATTER, and this passed Linear with neither.
        #
        # The four chips on a QB2 are a physical ring: p300_x2_mesh_graph_descriptor
        # declares `device_topology { dims: [2, 2] }` with `channels { count: 2 }`, every
        # chip degree 2, both directions 2 links wide. ornith-1.0 -- the same architecture
        # on the same host -- measured every spelling of this collective inside a captured
        # trace and records that `Topology.Ring` beats `Topology.Linear` "at every traced
        # shape, which is the physical ring being real", with num_links=2, and calls the
        # fabric config "load-bearing at decode". We were asking for a line on ring
        # hardware with one link.
        #
        # Ring is selected from the MESH, not from a flag, so it cannot drift from the
        # fabric the device was opened with (>=4 dies is where the descriptor declares a
        # cycle; 2 chips really are a line). ORNITH_CCL_TOPOLOGY overrides by name for
        # A/B measurement.
        _topo_name = os.environ.get("ORNITH_CCL_TOPOLOGY") or ("Ring" if self.num_devices >= 4 else "Linear")
        _topo = getattr(ttnn.Topology, _topo_name)
        _links = int(os.environ.get("ORNITH_CCL_NUM_LINKS", "2"))
        reduced = ttnn.all_reduce(acc, cluster_axis=self.cluster_axis, topology=_topo, num_links=_links)
        ttnn.deallocate(acc)
        assert reduced.shape[-1] == H, f"reduce returned {reduced.shape[-1]}, expected {H}"
        return reduced


class OrnithMoE:
    """MoE block for one decoder layer: router -> 8 routed experts + 1 gated shared expert.

    `bank` is the shared multi-layer routed-expert store. Passing None builds a
    single-layer bank, which is what the unit test wants; the full model builds one
    bank for all 40 layers and hands the same object to every layer.
    """

    def __init__(
        self,
        mesh_device,
        args,
        state_dict,
        layer_id: int,
        tensor_cache_path=None,
        bank: "OrnithMoEBank | None" = None,
        tt_ccl=None,
    ):
        """`bank` shares one expert store across all layers. Passing None builds a
        single-layer bank, which is what the unit tests want; the full model builds one
        bank for the whole stack and hands the same object to every layer."""
        self.mesh_device = mesh_device
        self.args = args
        self.layer_id = layer_id
        self.num_experts = args.num_experts
        self.top_k = args.num_experts_per_tok
        self.hidden_size = args.dim
        try:
            self.gate_up_dtype, self.down_dtype = _RECIPES[args.routed_expert_dtype]
        except KeyError:
            raise ValueError(
                f"unknown routed-expert recipe {args.routed_expert_dtype!r}; " f"known: {sorted(_RECIPES)}"
            ) from None

        # --- router -------------------------------------------------------
        # ONE TTMoEGate is shared by all 40 layers, with each layer swapping in its own
        # router weight. The gate preallocates per-core L1 buffers (bias, input indices,
        # output, output indices) whose size depends only on batch and expert count, not
        # on the layer -- so building one gate per layer holds 40 copies of them in L1.
        # Measured: ~6.4 KB of L1 per layer, which is fine at 8 layers and fatal by 16,
        # where the GDN seq op can no longer place its ~1.36 MB of circular buffers in a
        # 1.5 MB L1 ("circular buffers clash with L1 buffers"). Only the router weight is
        # per-layer, and it lives in DRAM, so sharing the gate costs nothing.
        # Kept bf16 unconditionally. It is 2048x256 per layer (0.06% of the model), and
        # it picks which experts run at all: a rounding error here swaps an expert, which
        # is a far larger output perturbation than a rounding error inside one.
        gate_w = state_dict["gate.weight"].to(torch.bfloat16).T.contiguous()
        shared = getattr(args, "_ornith_shared_gate", None)
        if shared is None:
            shared = TTMoEGate(
                mesh_device,
                TTMoEGateConfig(
                    select_experts_k=self.top_k,
                    hidden_size=self.hidden_size,
                    num_routed_experts=self.num_experts,
                    n_group=1,  # ungrouped generalized_moe_gate kernel
                    score_func="softmax",
                    softmax_position="post",  # softmax-over-selected == reference (no bias, Z cancels)
                    # Let ttnn pick the router matmul's program config. The derived
                    # 2D-mcast config is tuned for a decode-shaped gate and fails
                    # "num_blocks_y <= grid.y" once the token count passes ~256, which
                    # is every prefill longer than one GDN chunk. The router is a
                    # [T, 2048] x [2048, 256] matmul -- far too small for the tuning to
                    # be worth a hard failure at length.
                    gate_matmul_auto_program_config=False,
                ),
                torch_gate_weight=gate_w,
            )
            args._ornith_shared_gate = shared
        self.gate = shared
        # This layer's own router weight, in the layout and placement the gate expects.
        # Swapped into the shared gate immediately before each use; layers run
        # sequentially, so there is never more than one live at a time.
        self.tt_gate_weight = ttnn.from_torch(
            gate_w,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # --- routed experts (shared across layers via the bank) -------------
        self.bank = (
            bank
            if bank is not None
            else OrnithMoEBank(mesh_device, args, [state_dict], tensor_cache_path, tt_ccl=tt_ccl)
        )
        # A per-layer instance addresses its own slice of a shared bank; a bank built for
        # this layer alone only has slot 0.
        self.bank_slot = layer_id if bank is not None else 0

        # --- shared expert (always fires, so it never goes through dispatch) --
        sh0 = state_dict["shared_expert.gate_proj.weight"].to(torch.bfloat16).T.contiguous()
        sh1 = state_dict["shared_expert.up_proj.weight"].to(torch.bfloat16).T.contiguous()
        sh2 = state_dict["shared_expert.down_proj.weight"].to(torch.bfloat16).T.contiguous()

        # The shared expert is NOT simply added. This architecture gates it with a
        # learned per-token sigmoid (`shared_expert_gate`, a [1, hidden] projection):
        #     out = routed + sigmoid(x @ g) * shared(x)
        # The MoE YAML's `shared_expert_scale` is a fixed scalar and cannot express
        # this, so the gate is applied here, outside TTMoEDecode. Dropping it is a
        # silent accuracy bug -- the model still produces fluent text, it is just
        # wrong -- so test_moe_block.py asserts against the gated reference.
        self.shared_gate_w = ttnn.as_tensor(
            state_dict["shared_expert_gate.weight"].to(torch.bfloat16).T.contiguous(),
            dtype=ttnn.bfloat16,
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        self._shared = self._upload_shared_expert(sh0, sh1, sh2, tensor_cache_path)
        # Built once: the shapes are fixed by the config, and building a program config
        # per forward would put host work in the decode path.
        self._sh_pc = {
            "gate_up": _decode_matmul_config(self.hidden_size, args.shared_expert_intermediate_size),
            "down": _decode_matmul_config(args.shared_expert_intermediate_size, self.hidden_size),
        }

    def _upload_shared_expert(self, sh0, sh1, sh2, cache):
        """Upload ONLY the shared expert.

        An earlier version of this also uploaded a second copy of the 256 routed experts
        so prefill could loop over them in a [E, in, out] layout. That is 18-34 GB of
        duplicate weights on top of the bank's copy and would have put every board over
        its DRAM. Prefill now reuses the bank; the shared expert is the one thing that
        genuinely needs its own tensors, and at 0.13 GB replicated it is free.

        bf16 regardless of the routed dtype: the shared expert runs for EVERY token, so
        its error is systematic rather than averaged over a changing set of 8 experts, and
        at 0.25 GB replicated the extra precision costs nothing worth counting.
        """
        rep = ttnn.ReplicateTensorToMesh(self.mesh_device)
        nd = self.mesh_device.get_num_devices()

        def up(t, name, mapper=rep):
            return ttnn.as_tensor(
                t,
                dtype=ttnn.bfloat16,
                device=self.mesh_device,
                mesh_mapper=mapper,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                cache_file_name=(str(cache / f"moe.{name}.l{self.layer_id}") if cache else None),
            )

        # Fully replicated. The shared expert runs for every token and its output is
        # added to a replicated residual stream, so there is nothing to shard and no
        # collective to pay: 0.13 GB per chip buys a complete copy.
        return {"sh0": up(sh0, "sh0"), "sh1": up(sh1, "sh1"), "sh2": up(sh2, "sh2")}

    # ------------------------------------------------------------------
    def forward(self, x, mode: str = "decode"):
        if mode == "decode":
            return self._forward_decode(x)
        return self._forward_prefill(x)

    def _shared_gate(self, x, logit=False):
        """sigmoid(x @ shared_expert_gate) -- the per-token scale on the shared expert.

        `logit=True` returns the projection unsquashed, for a caller that folds the
        sigmoid into the multiply that consumes it.
        """
        proj = ttnn.linear(x, self.shared_gate_w, compute_kernel_config=_hifi(), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if logit:
            return proj
        out = ttnn.sigmoid(proj, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(proj)
        return out

    def _shared_expert(self, x):
        """Ungated shared-expert SwiGLU over x. The learned per-token sigmoid gate is
        applied by the caller, so this is reusable by both the decode and prefill paths.
        """
        w = self._shared
        ckc = _hifi()
        dram = ttnn.DRAM_MEMORY_CONFIG
        # Explicit program configs at M=1. These three are 2.5x, 2.5x and 1.9x off the
        # heuristic respectively -- see _decode_matmul_config. Only for a single-token
        # step: prefill's M is tall and the heuristic is not the loser there.
        decode = x.shape[-2] == 1
        # ...but ONLY when the activation is interleaved, which is the single-chip case
        # these configs were tuned for.
        #
        # Under tensor parallelism the layer's DistributedNorm hands the feed-forward an
        # L1-SHARDED activation (full width -- the norm all-gathers, so only the
        # placement differs, not the shape). These are
        # MatmulMultiCoreReuseMultiCast1DProgramConfigs with fuse_batch=False, and the op
        # forbids exactly that combination:
        #   "MatmulMultiCoreReuseMultiCast1DProgramConfig: fuse_batch must be enabled
        #    when input A is sharded" (matmul_device_operation.cpp:1944)
        # which is what made EVERY TP decode step die on layer 0's shared expert, and is
        # the real content of the "4-chip decode hangs" blocker.
        #
        # Falling back to ttnn's heuristic for a sharded in0 costs the 2.5x/1.9x these
        # configs win, and only on the TP path. Two alternatives were considered:
        #
        #   * Convert the activation to interleaved at the top of _forward_decode. Not
        #     taken: it adds an op per layer on every die, and the natural spelling also
        #     frees that copy at the end -- while _run_bank documents that ttnn.reshape
        #     hands the routed bank a VIEW over the same buffer ("freeing it frees x, and
        #     the next matmul fails with 'operand must be on device'"), so the lifetime is
        #     a trap. This was tried and a single-chip eager decode then hung; that hang is
        #     NOT offered as evidence against the approach, because the device had been
        #     left wedged by a SIGKILLed run and a separate 4-die hang in the same window
        #     turned out to be a stalled MeshCommandQueue weight upload with no relation
        #     to this code. The aliasing argument stands on its own; the hang is unattributed.
        #   * Assert fuse_batch=True on the existing configs. Not taken: they were tuned
        #     and measured at fuse_batch=False, so flipping it changes the very geometry
        #     the 2.5x was measured with, and nothing here re-measures it.
        #
        # A correct slow path beats a fast wrong one. Retuning a sharded-in0 config, or
        # keeping the activation interleaved without the aliasing hazard, is a follow-up.
        sharded_in = x.memory_config().is_sharded()
        pc0 = self._sh_pc["gate_up"] if decode and not sharded_in else None
        pc2 = self._sh_pc["down"] if decode and not sharded_in else None
        g = ttnn.linear(x, w["sh0"], compute_kernel_config=ckc, memory_config=dram, program_config=pc0)
        u = ttnn.linear(x, w["sh1"], compute_kernel_config=ckc, memory_config=dram, program_config=pc0)
        if decode:
            # SiLU folded onto the gate operand: one program instead of two, 40 per token
            # over the model. Both operands come off `ttnn.linear` with the same dtype, so
            # the fused form is faithful. Prefill keeps the separate spelling so its logit
            # PCC against the HF reference stays bit-stable -- a fused activation is the
            # same arithmetic but not the same rounding order, and prefill is not
            # dispatch-bound.
            act = ttnn.multiply(u, g, input_tensor_b_activations=[ttnn.UnaryOpType.SILU], memory_config=dram)
        else:
            act = ttnn.multiply(ttnn.silu(g), u, memory_config=dram)
        y = ttnn.linear(act, w["sh2"], compute_kernel_config=ckc, memory_config=dram, program_config=pc2)
        for t in (g, u, act):
            ttnn.deallocate(t)
        return y

    def _run_bank(self, x, n_tokens):
        """Feed one token block through the routed-expert bank.

        `TTMoEGate` takes `[1, 1, B, H]` in TILE layout; the sparse expert path takes the
        activation as-is. (The removed fused pipeline additionally wanted a ROW_MAJOR
        `[B, 1, 1, H]` copy, which is why this used to convert layouts twice.)
        """
        # TTMoEGate wants [1, 1, B, H]; the decoder layer hands the FF a 3D [B, T, H].
        # Reshape rather than push the convention onto the layer -- it is metadata-only,
        # both orderings being token-major with hidden last.
        gate_in = x if len(x.shape) == 4 else ttnn.reshape(x, (1, 1, n_tokens, self.hidden_size))
        self.gate.tt_gate_weight = self.tt_gate_weight  # this layer's router
        scores, indices = self.gate.forward(gate_in)
        # gate_in is NOT deallocated: ttnn.reshape returns a view over x's buffer, so
        # freeing it frees x and the next matmul fails with "operand must be on device".
        out = self.bank.forward(x, scores, indices, layer_id=self.bank_slot)
        ttnn.deallocate(scores)
        ttnn.deallocate(indices)
        return out

    def _forward_decode(self, x):
        """x: [1, 1, batch, hidden] -> [1, 1, batch, hidden].

        Routed experts go through the bank; the gated shared expert is added on top. Same
        decomposition as prefill, so the two paths can be diffed against each other.
        """
        dram = ttnn.DRAM_MEMORY_CONFIG
        # Everything here is explicitly DRAM and explicitly freed. ttnn.linear and the
        # eltwise ops default to L1, and the GDN seq op needs ~1.36 MB of the 1.5 MB L1
        # for its circular buffers -- so a couple of L1 tensors left behind per layer is
        # enough, forty layers in, to make the NEXT layer's attention fail to place its
        # buffers ("circular buffers clash with L1 buffers"), naming neither this module
        # nor the leak.
        se = self._shared_expert(x)
        # The gate's sigmoid rides on the multiply that consumes it -- one fewer program
        # per layer, 40 per token. Prefill keeps the separate sigmoid (see _shared_expert).
        sg = self._shared_gate(x, logit=True)
        shared = ttnn.multiply(se, sg, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID], memory_config=dram)
        ttnn.deallocate(se)
        ttnn.deallocate(sg)
        routed = self._run_blocks(x)
        out = ttnn.add(routed, shared, memory_config=dram)
        ttnn.deallocate(routed)
        ttnn.deallocate(shared)
        return out

    def _run_blocks(self, x):
        """Run x through the routed experts, whole sequence in one pass.

        This used to block the sequence into `batch_per_device` chunks, which only the
        fused TTMoEDecode pipeline needed -- its buffers were allocated for exactly that
        many tokens, so a shorter block had to be padded and a longer one split. The
        sparse path is a matmul loop with no such constraint, and blocking it only adds
        ops.
        """
        # Token count is the product of EVERY leading dim, not shape[-2]. The decoder
        # hands decode a [B, T, H] tensor, so at batch 1 shape[-2] is the token count by
        # coincidence and at batch 4 it is 1 while there are 4 tokens -- which showed up
        # as a reshape volume error inside the router, not here.
        n_tokens = math.prod(list(x.shape)[:-1])
        return self._run_bank(x, n_tokens)

    def _forward_prefill(self, x):
        """x: [1, 1, seq, hidden] -> [1, 1, seq, hidden].

        Identical to decode apart from the token count: the SAME expert weights are used,
        fed through the pipeline in blocks. Prefill deliberately does not get its own copy
        of the experts -- a second copy is 18-34 GB and does not fit on any of our boards.
        """
        dram = ttnn.DRAM_MEMORY_CONFIG
        # Everything here is explicitly DRAM and explicitly freed. ttnn.linear and the
        # eltwise ops default to L1, and the GDN seq op needs ~1.36 MB of the 1.5 MB L1
        # for its circular buffers -- so a couple of L1 tensors left behind per layer is
        # enough, forty layers in, to make the NEXT layer's attention fail to place its
        # buffers ("circular buffers clash with L1 buffers"), naming neither this module
        # nor the leak.
        se = self._shared_expert(x)
        sg = self._shared_gate(x)
        shared = ttnn.multiply(se, sg, memory_config=dram)
        ttnn.deallocate(se)
        ttnn.deallocate(sg)
        routed = self._run_blocks(x)
        out = ttnn.add(routed, shared, memory_config=dram)
        ttnn.deallocate(routed)
        ttnn.deallocate(shared)
        return out

    __call__ = forward
