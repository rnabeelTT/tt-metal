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


def _expert_compute_config(device, fidelity=ttnn.MathFidelity.LoFi):
    """Compute config for the expert matmuls.

    LoFi, not HiFi4: bfp4 weights carry 4 mantissa bits, so HiFi4's extra passes
    have nothing to resolve. Measured in the Qwen3-Coder-30B-A3B reference as both
    faster and very slightly MORE accurate at bfp4.

    fp32_dest_acc_en is OFF and must stay off. It halves the matmul dest from 8 tiles
    to 4, which corrupts expert output on Blackhole (tt-metal #49068). This port had it
    ON, inherited from a generic "high fidelity" config.
    """
    return ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=fidelity,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )


def _hifi():
    """HiFi4 for the small non-expert matmuls (router, shared expert)."""
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
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
    CHUNK = 32

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
        # a complete copy. Extra chips then serve extra USERS instead of splitting one:
        # no collectives, no fabric, and aggregate throughput scales linearly with dies
        # rather than sub-linearly. Set ORNITH_MOE_EP=1 to get the sharded path back.
        self.expert_parallel = os.environ.get("ORNITH_MOE_EP", "0") == "1"
        mesh_n = getattr(args, "mesh_devices", mesh_device.get_num_devices())
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
                cache_file_name=(str(cache / f"moe.sparse.{name}") if cache else None),
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
        decode = T == 1 or T < self.CHUNK
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

        Four ops where `ttnn.swiglu` would be one, and that is deliberate: at M=1 the
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
        out = ttnn.multiply(ttnn.silu(gate, memory_config=memory_config), up, memory_config=memory_config)
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
            outs.append(ttnn.sum(st, dim=1, keepdim=True))  # [1,1,1,H]
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
        reduced = ttnn.all_reduce(acc, cluster_axis=self.cluster_axis, topology=ttnn.Topology.Linear)
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

    def _shared_gate(self, x):
        """sigmoid(x @ shared_expert_gate) -- the per-token scale on the shared expert."""
        proj = ttnn.linear(x, self.shared_gate_w, compute_kernel_config=_hifi(), memory_config=ttnn.DRAM_MEMORY_CONFIG)
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
        pc0 = self._sh_pc["gate_up"] if decode else None
        pc2 = self._sh_pc["down"] if decode else None
        g = ttnn.linear(x, w["sh0"], compute_kernel_config=ckc, memory_config=dram, program_config=pc0)
        u = ttnn.linear(x, w["sh1"], compute_kernel_config=ckc, memory_config=dram, program_config=pc0)
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
        sg = self._shared_gate(x)
        shared = ttnn.multiply(se, sg, memory_config=dram)
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
