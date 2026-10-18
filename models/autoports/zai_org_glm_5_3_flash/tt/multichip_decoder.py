# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Multichip decoder layer of GLM-5.3-Flash (stage 4): the optimized decoder (tt/optimized_decoder.py) on a (1, tp) mesh
(p300x2: tp = 4, FABRIC_1D). Semantics, contracts, paging and state are those of tt/functional_decoder.py.

Residual contract (input and output of every layer): the 4-stream mHC state [1,1,T,4*C] fp32, stream-interleaved and
hidden-sharded. Device d holds columns [d*C/tp, (d+1)*C/tp) of every stream, so its tensor is [1,1,T,4*C/tp]. The
mHC ops (mhc_mix, mhc_pre, mhc_post) are column-local and run unchanged on that slice; only the mix projection and the
row sum of squares are partial sums (one small all-gather + sum).

Per sub-block (attention, FFN):
  1. mhc_mix partials [1,1,T,64] fp32 -> all-gather (dim 1) -> sum: mixes and sum(X^2) of the whole row.
  2. mhc_split_sinkhorn (replicated, tiny) -> mhc_pre -> h [T, C/tp] bf16.
  3. all-gather h -> [T, C] replicated -> RMSNorm: the input of the token mixer / FFN.
  4. Token mixer / FFN with TP over heads / intermediate (KDA 16 heads, DSA 16 query heads with the latent cache and the
     indexer replicated, MoE experts and shared expert over the intermediate, dense MLP over the intermediate). The
     row-parallel output projection gives partial sums -> reduce-scatter -> y [T, C/tp] (KDA prefill and the MoE do the
     reduce-scatter inside the module).
  5. mhc_post on the local columns.
Collectives per layer: 2 x (all-gather 64 fp32 columns + all-gather C bf16 + reduce-scatter C bf16).

Stage 5 (doc/optimized_multichip_decoder/README.md): MCOpts / DEFAULT_MC select ring collectives when the fabric routes
them, L1 collective outputs, width-sharded decode RMSNorms, the five replicated DSA projections of the normed input as
two packed matmuls (MultichipDSA), 1D decode configs for the packs, the DSA q_b / wq_b and the MoE router, the packed
weights shared with prefill, the DSA prefill matmuls on an explicit 2D config, and a padded decode batch for 1-slot KDA
layers (MultichipKDA). MCOpts() is the stage-4 path.
"""

from __future__ import annotations

import dataclasses
import math
import os
import types
from dataclasses import dataclass

import torch

import ttnn
from models.demos.blackhole.qwen36.tt import tp_common as tpc
from models.tt_transformers.tt.ccl import TT_CCL, tt_all_gather, tt_all_reduce

from .dsa import BLOCK, IDX_WIDTH, KPOOL, SENTINEL_F, TOPK_POOLS, GlmDSA
from .fused_decoder import FusedGlmKDA, FusedHyperConnection, mix_chunks
from .mhc import DRAM, W
from .mlp import DenseMLP
from .optimized_decoder import OptimizedDecoder

TOPOLOGY = ttnn.Topology.Linear  # stage-4 default; MCOpts.topology selects per decoder (stage 5)
L1 = ttnn.L1_MEMORY_CONFIG


@dataclass
class MCOpts:
    """Stage-5 multichip decode options (doc/optimized_multichip_decoder/README.md has the A/B measurements).
    Most options change decode rows only (one tile of users). Prefill changes too with `topology` / `num_links` (every
    collective), `ds_shared` (the packed weights serve prefill), `prefill_2d` / `packs_prefill_2d` / `prefill_2d_ffn`
    (explicit 2D prefill configs) and the MoE geometry the A/B tools set.

    GLM53_MC="key=value,..." overrides fields of DEFAULT_MC at construction (A/B runs; tools/mc_decode_ab.py); a
    construction with it set logs a warning, since it changes the served decoder."""

    topology: str = "linear"  # CCL topology: linear | ring | auto (ring when the fabric has the wraparound link)
    num_links: int = 0  # CCL links per collective (0 = all available)
    ccl_l1: bool = False  # decode all-gather / reduce-scatter outputs in L1 instead of DRAM
    norm_sharded: bool = False  # decode RMSNorms on width-sharded L1 over many cores (ttnn picks 1 core interleaved)
    dsa_pack: bool = False  # DSA decode: q_a | kv_a and indexer wk | gate | weights packed, DRAM-sharded matmuls
    router_ds: bool = False  # MoE decode router matmul on a DRAM-sharded (zero-padded 192 -> 256) weight
    mix_nc: bool = False  # mHC mix partial sums with fast_reduce_nc instead of ttnn.sum (transpose + reduce)
    # decode all-gathers write into preallocated (per call site) output buffers. Rejected: breaks the stacked model
    # (ttnn skips the all-gather start barrier with persistent outputs; see MeshCCL)
    ag_persistent: bool = False
    # ds_shared: the DRAM-sharded packed DSA / router weights also serve the prefill rows (2D program config) and the
    # separate interleaved weights are freed at build: no second weight copy (decided at construction)
    ds_shared: bool = False
    # pack_b_1d / router_1d: the indexer pack (wk | gate | weights, N = 288) / the router (N = 192) as interleaved weights
    # on a 1D decode config instead of DRAM-sharded (DRAM sharding pads N to 8 banks x 32: +1.8 / +0.5 MB per layer)
    pack_b_1d: bool = False
    router_1d: bool = False
    pack_a_1d: bool = False  # q_a | kv_a pack interleaved on a 1D decode config instead of DRAM-sharded
    dsa_1d: bool = False  # DSA q_b and indexer wq_b decode matmuls (input q_a-normed, K = 1536) on 1D configs
    prefill_2d: bool = False  # DSA prefill matmuls (q_b, wq_b, o_proj) on the explicit 2D config (in0_block_w > 1)
    packs_prefill_2d: bool = False  # ... and the two packed DSA projections (needs prefill_2d)
    # dsa_shard: the DSA latent pool block-sharded over the dies instead of replicated (tt/dsa_shard.py; long-context
    # capacity: 3,696 vs 11,968 B per token per die). Changes prefill and decode attention. Also GLM53_DSA_SHARD=1.
    dsa_shard: bool = False
    # per-role decode programs (stage-5 review #1 P1-2; tools/mc_matmul_sweep.py): "" = the built path, "1d[:ib<n>][:c<n>]"
    # = 1D multicast (in0 block / cores), "ds[:a<n>][:ib<n>][:r<n>]" = DRAM-sharded (activation cores / in0 block / readers
    # per bank; runtime A/B keeps a sharded copy, ds_shared moves the weight)
    pack_a_spec: str = ""  # q_a | kv_a pack decode program when pack_a_1d ("" = 1D, in0 blocks of 8)
    pack_b_spec: str = ""  # indexer pack decode program when pack_b_1d
    kda_in: str = ""
    kda_out: str = ""
    dsa_o: str = ""
    dsa_qb: str = ""
    dsa_wqb: str = ""
    shared_gu: str = ""
    shared_dn: str = ""
    mlp_gu: str = ""
    mlp_dn: str = ""
    # explicit 2D prefill configs for the dense MLP, the shared expert and the KDA output projection
    prefill_2d_ffn: bool = False
    # Sinkhorn permutation-sum fidelity (0 = the op default / environment; 3 = HiFi3, exact: the permutation is 0 / 1);
    # applied as MHC_SINKHORN_PERM_FIDELITY at construction unless the environment sets it
    sinkhorn_perm_fidelity: int = 0
    # Sinkhorn normalizations on the used DST faces only (n*n = 16 columns: face 0 at <= 16 tokens, faces 0 / 2 otherwise;
    # the padding of m zeroed once): applied as MHC_SINKHORN_FACES=1 at construction unless the environment sets it
    sinkhorn_faces: bool = False
    # DSA decode data movement at 1 slot (stage-5 review #2 P2-2; README "DSA attention data movement"):
    # dsa_q_mm: q_abs of all heads as ONE matmul of the head-masked, row-replicated q_b output [32 heads (16 zero), 4096]
    # with W_UK viewed as [H*256, 512] (instead of reshape + permute + per-head bmm + head pad + untilize): q_abs comes
    # out with heads as rows, padded to the 32 heads sparse_sdpa takes, one row-major view away from its input
    dsa_q_mm: bool = False
    # dsa_o_mm: the per-head W_UV products as ONE matmul of the 16 attention rows [16, 512] with W_UV laid out [512, H*256],
    # the off-head blocks masked to 0 and the rows summed (one nonzero per column: exact), instead of tilize of the
    # [16 heads x 1 row] output + per-head bmm + concat (transpose + reshape)
    # (1); 2 = the same products from W_UV viewed as [H*512, 256] (no relayout): the attention rows flattened to one row
    # [1, H*512], head-masked into H rows, one matmul -> [H, 256], row-major concat
    dsa_o_mm: int = 0
    # dsa_tail_shared: the 0-3 tail tokens of the sparse index list (8 one-tile element-wise ops on the step's positions,
    # ~34 µs) computed once per decode step by the model's first DSA layer and read by the other DSA layers from the
    # step's shared input dict ("_tail"); a layer without it computes its own (a per-step host input "tail", A/B only,
    # stands for a consumer layer)
    dsa_tail_shared: bool = False
    # kda_gnorm_op: the KDA decode o-norm + sigmoid gate as ttnn.experimental.kda.sigmoid_gated_rms_norm (head-first
    # [B*H, T, V] with T a whole tile: each (slot, head) row relaid to its own tile row and back) instead of rms_norm +
    # relayout + gate multiply (measurement of the review #1 rejection, review #2 P2-2)
    kda_gnorm_op: bool = False
    mix_chunks_decode: int = 0  # mhc_mix K chunks at decode rows (0 = mix_chunks(): 64 at TP4; fewer = a smaller sum)
    kda_conv_op: bool = (
        False  # KDA decode conv step as one fused op (ttnn.experimental.deepseek_prefill.kda_decode_conv)
    )
    # the fused conv op only up to this decode batch (0 = every batch): its per-row work grows with B, and at 32 slots
    # the unfused chain is faster (review #2: kda_moe 4408 vs 4355 µs, records/kda_conv_cmp.jsonl)
    kda_conv_max_batch: int = 0
    kda_conv_mixed: bool = False  # KDA decode conv: addcmul on the bf16 history rows directly (no fp32 typecasts)
    moe_grouped_in0: int = 0  # MoE grouped sparse matmuls: in0 K block (gate/up and down; 0 = the stage-3 value 4)
    router_topk_pad: int = 0  # MoE router top-k on a -1e30-padded width (256: power of two; 1024: multi-core top-k)

    @classmethod
    def from_env(cls, base: "MCOpts") -> "MCOpts":
        shard = os.environ.get("GLM53_DSA_SHARD", "").strip()
        if shard:
            base = dataclasses.replace(base, dsa_shard=shard.lower() in ("1", "true", "on", "yes"))
        spec = os.environ.get("GLM53_MC", "").strip()
        if spec:
            from loguru import logger

            logger.warning(f"GLM53_MC={spec!r} overrides the multichip decoder options (A/B runs only)")
        if not spec:
            return dataclasses.replace(base)
        kw = {}
        for item in spec.split(","):
            if not item.strip():
                continue
            k, v = item.split("=")
            k = k.strip()
            typ = type(getattr(base, k))
            kw[k] = (v.strip().lower() in ("1", "true", "on", "yes")) if typ is bool else typ(v.strip())
        return dataclasses.replace(base, **kw)


# Selected by the stage-5 A/B (doc/optimized_multichip_decoder/README.md). MCOpts() is the stage-4 path.
DEFAULT_MC = MCOpts(
    topology="auto",
    ccl_l1=True,
    norm_sharded=True,
    dsa_pack=True,
    pack_a_1d=True,
    pack_b_1d=True,
    dsa_1d=True,
    router_1d=True,
    ds_shared=True,
    prefill_2d=True,
    # stage-5 review #1 rework (README "Candidates and decisions", round 6)
    packs_prefill_2d=True,
    prefill_2d_ffn=True,
    kda_conv_op=True,
    sinkhorn_perm_fidelity=3,
    # stage-5 review #2: Sinkhorn normalizations on the used DST faces (bit-identical; -30 to -43 µs per layer)
    # stage-5 review #2 (full-model gate: teacher forcing 92 / 100 / 100 with these four, records/s5r2_tf_cand3.log):
    # Sinkhorn normalizations on the used DST faces (bit-identical), the 1-slot DSA q_abs as one matmul (bit-identical),
    # the sparse-index tail computed once per step (bit-identical), the fused KDA conv up to 16 slots
    sinkhorn_faces=True,
    dsa_q_mm=True,
    dsa_tail_shared=True,
    kda_conv_max_batch=16,
    mix_chunks_decode=16,
    # off: on the 45-layer model the persistent decode all-gather outputs (shared per call site across layers) broke
    # decode (1 slot: teacher forcing top-5 32 %; 8 slots: identical greedy runs differ), although 2 x 120 replays of
    # one layer were bit-identical. Stage 7 measured readiness with it off (top-5 100 %); see the stage-7 README.
    ag_persistent=False,
    moe_grouped_in0=8,
    kda_out="1d:c16:ib4",
    dsa_o="1d:c16:ib4",
    dsa_qb="1d:c16:ib4",
    dsa_wqb="1d:c16:ib4",
    pack_a_spec="1d:c32:ib8",
    shared_gu="1d:c32:ib16",
    shared_dn="1d:c16:ib4",
    mlp_gu="1d:c16:ib4",
    mlp_dn="1d:c16:ib4",
)


def _decode_rows(t) -> bool:
    """Decode rows = one tile of rows (users <= 32). Contract: every prefill chunk is a multiple of 128 rows
    (GlmDSA.prefill_inputs and the KDA prefill buckets assert it), so 32 padded rows only ever mean a decode step."""
    return int(t.padded_shape[-2]) == 32


def _ds_grid(K: int, N: int, max_cores: int = 32):
    """(cores, CoreGrid) of a DRAM-sharded decode matmul / a width-sharded activation: cores divide K and N tiles, a grid
    of rows <= 8 x cols <= 8 (the moe_opt.DenseDecodeMatmul._ds_configs rule)."""
    g = math.gcd(K // 32, N // 32)
    legal = [c for c in range(1, max_cores + 1) if g % c == 0 and any(c % r == 0 and c // r <= 8 for r in range(1, 9))]
    c = max(legal)
    rows = next(r for r in range(8, 0, -1) if c % r == 0 and c // r <= 8)
    return c, ttnn.CoreGrid(x=c // rows, y=rows)


def _width_shard(K: int, cores: int, grid) -> ttnn.MemoryConfig:
    return ttnn.create_sharded_memory_config(
        shape=(32, K // cores),
        core_grid=grid,
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def prefill_pc(x, K: int, N: int, in0_block_w: int | None = None):
    """Explicit 2D multicast config of a prefill matmul [m, K] x [K, N] (tt-perf-report flagged ttnn's auto config at
    in0_block_w = 1 on these rows). in0_block_w: override of the qwen36 helper's choice (must divide K tiles)."""
    m = int(x.padded_shape[-2]) * int(math.prod(list(x.padded_shape)[:-2]))
    pc = tpc.create_prefill_matmul_program_config(m, K, N)
    if in0_block_w:
        assert (K // 32) % in0_block_w == 0, (K, in0_block_w)
        pc.in0_block_w = in0_block_w
    return pc


class DSMatmul:
    """Decode matmul (one tile of rows) on a DRAM width-sharded weight [K, N] (N a multiple of 32 * 8 banks): every
    DRAM bank streams its own column block (the bandwidth-bound decode form; ttnn's default program puts an N <= 9 tile
    weight on N cores)."""

    def __init__(
        self,
        w_il,
        max_cores: int = 32,
        act_cores: int | None = None,
        in0_block_w: int | None = None,
        readers: int = 1,
        convert: bool = True,
    ):
        """act_cores: cores of the width-sharded activation (K tiles / act_cores = the in0 shard; in0_block_w must divide
        it, so fewer activation cores allow larger in0 blocks); readers: workers per DRAM bank (Blackhole multi-reader,
        legal when the bank's N tiles split evenly). The compute runs on the DRAM-bank reader cores (8 x readers)."""
        K, N = int(w_il.shape[-2]), int(w_il.shape[-1])
        assert N % (32 * tpc.DRAM_CORES) == 0, N
        self.l1_acc = True  # packer L1 accumulation of the K-block partials (acc_ckc); False = the caller's config
        self.K, self.N = K, N
        self.w = ttnn.to_memory_config(w_il, tpc.create_dram_sharded_mem_config(K, N)) if convert else w_il
        if act_cores is None:
            cores, grid = _ds_grid(K, N, max_cores)
        else:
            cores = act_cores
            rows = next(r for r in range(8, 0, -1) if cores % r == 0 and cores // r <= 8)
            grid = ttnn.CoreGrid(x=cores // rows, y=rows)
        shard = K // 32 // cores
        self.act = _width_shard(K, cores, grid)
        self.pc = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
            in0_block_w=in0_block_w or tpc._find_largest_divisor(shard),
            per_core_M=1,
            per_core_N=max(1, N // 32 // cores),
            fused_activation=None,
        )
        if readers > 1:
            assert (N // 32 // tpc.DRAM_CORES) % readers == 0, (N, readers)
            self.pc.num_workers_per_dram_bank = readers
        self.geometry = {"act_cores": cores, "in0_block_w": self.pc.in0_block_w, "readers": readers}

    _acc_ckc = {}

    @classmethod
    def acc_ckc(cls, ckc):
        """The caller's compute config with packer L1 accumulation: the K-block partials then accumulate in fp32 L1
        (the default spill / reload of partials through the FPU source registers truncates them to TF32: ~10x the
        error of the interleaved matmul on the fp32-output router, unit check of 2026-10-17)."""
        key = id(ckc)
        if key not in cls._acc_ckc:
            cls._acc_ckc[key] = (
                ckc,
                ttnn.WormholeComputeKernelConfig(
                    math_fidelity=ckc.math_fidelity,
                    math_approx_mode=ckc.math_approx_mode,
                    fp32_dest_acc_en=ckc.fp32_dest_acc_en,
                    packer_l1_acc=True,
                ),
            )
        return cls._acc_ckc[key][1]

    def prefill(self, x, ckc, dtype=None):
        """Rows > 32 on the same DRAM-sharded weight: 2D multicast program config (models/demos/blackhole/qwen36)."""
        pc = prefill_pc(x, self.K, self.N)
        kw = {"dtype": dtype} if dtype is not None else {}
        return ttnn.linear(x, self.w, compute_kernel_config=ckc, program_config=pc, memory_config=DRAM, **kw)

    def __call__(self, x, ckc, dtype=None, out_mc=DRAM):
        same = x.is_sharded() and x.memory_config() == self.act
        xs = x if same else ttnn.to_memory_config(x, self.act)
        kw = {"dtype": dtype} if dtype is not None else {}
        out = ttnn.linear(
            xs,
            self.w,
            compute_kernel_config=self.acc_ckc(ckc) if self.l1_acc else ckc,
            program_config=self.pc,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            **kw,
        )
        if not same:
            ttnn.deallocate(xs)
        o = ttnn.to_memory_config(out, out_mc)
        ttnn.deallocate(out)
        return o


class IL1DMatmul:
    """DSMatmul's interface on an INTERLEAVED weight [K, N] (no DRAM-bank padding of N): decode rows on a 1D multicast
    config with one output tile per core and in0 blocks of 8 tiles (ttnn's default put the [4096, 128] indexer
    projections on 4 cores at 49 us); prefill rows on ttnn's default program."""

    def __init__(self, w_il, grid_w: int, max_cores: int = 64, in0_block_w: int | None = None, prefill_2d=False):
        K, N = int(w_il.shape[-2]), int(w_il.shape[-1])
        self.K, self.N, self.w = K, N, w_il
        nt = N // 32
        cores = max(c for c in range(1, min(nt, max_cores) + 1) if nt % c == 0)  # whole output tiles per core
        ib = in0_block_w or max(d for d in (8, 4, 2, 1) if (K // 32) % d == 0)
        self.pc = tpc.create_matmul_1d_decode_progcfg(32, K, N, cores, fp32_acc=True, grid_w=grid_w, in0_block_w=ib)
        self.prefill_2d = prefill_2d
        self.geometry = {"cores": cores, "in0_block_w": ib}

    def prefill(self, x, ckc, dtype=None):
        kw = {"dtype": dtype} if dtype is not None else {}
        if self.prefill_2d:
            kw["program_config"] = prefill_pc(x, self.K, self.N)
        return ttnn.linear(x, self.w, compute_kernel_config=ckc, memory_config=DRAM, **kw)

    def __call__(self, x, ckc, dtype=None, out_mc=DRAM):
        xi = ttnn.to_memory_config(x, DRAM) if x.is_sharded() else x
        kw = {"dtype": dtype} if dtype is not None else {}
        out = ttnn.linear(
            xi, self.w, compute_kernel_config=DSMatmul.acc_ckc(ckc), program_config=self.pc, memory_config=out_mc, **kw
        )
        if xi is not x:
            ttnn.deallocate(xi)
        return out


def parse_role(spec: str) -> tuple[str, dict]:
    """ "ds:a8:ib16:r3" -> ("ds", {"a": 8, "ib": 16, "r": 3}); "" -> ("", {})."""
    if not spec:
        return "", {}
    parts = spec.split(":")
    kv = {}
    for p_ in parts[1:]:
        k = p_.rstrip("0123456789")
        kv[k] = int(p_[len(k) :])
    return parts[0], kv


def make_role_mm(spec: str, w, grid_w: int, sharded: bool = False):
    """Decode matmul object for a role spec on weight w ([K, N] interleaved, or already DRAM-sharded when `sharded`)."""
    kind, kv = parse_role(spec)
    if kind == "1d":
        return IL1DMatmul(w, grid_w, max_cores=kv.get("c", 64), in0_block_w=kv.get("ib"))
    if kind == "ds":
        if sharded:
            return DSMatmul(w, act_cores=kv.get("a"), in0_block_w=kv.get("ib"), readers=kv.get("r", 1), convert=False)
        N = int(w.shape[-1])
        wp = _pad_cols(w, tpc._roundup(N, 32 * tpc.DRAM_CORES))
        mm = DSMatmul(wp, act_cores=kv.get("a"), in0_block_w=kv.get("ib"), readers=kv.get("r", 1))
        if wp is not w:
            ttnn.deallocate(wp)
        mm.n_logical = N
        return mm
    raise ValueError(spec)


class RoleMatmuls:
    """Per-module cache of decode matmul objects: (role, spec, weight id) -> object (runtime A/B builds each once)."""

    def __init__(self, grid_w: int):
        self.grid_w, self.cache = grid_w, {}

    def get(self, role: str, spec: str, w):
        key = (role, spec, id(w))
        if key not in self.cache:
            self.cache[key] = make_role_mm(spec, w, self.grid_w, sharded=w.is_sharded())
        return self.cache[key]

    def run(self, role, spec, x, w, ckc, out_mc=DRAM, dtype=None):
        mm = self.get(role, spec, w)
        out = mm(x, ckc, dtype=dtype, out_mc=out_mc)
        n = int(w.shape[-1]) if not w.is_sharded() else getattr(mm, "n_logical", int(w.shape[-1]))
        if int(out.shape[-1]) != n:  # DRAM-bank padding columns
            o2 = ttnn.slice(out, (0, 0, 0, 0), (1, 1, int(out.shape[-2]), n), memory_config=out_mc)
            ttnn.deallocate(out)
            out = o2
        return out


class RoleDecodeMM:
    """Wraps a module's decode_mm hook (moe_opt.DenseDecodeMatmul, role "o"): decode rows of `role` with a spec go
    through RoleMatmuls; everything else to the wrapped hook."""

    def __init__(self, inner, mc: "MCOpts", spec_attr: str, grid_w: int):
        self.inner, self.mc, self.spec_attr = inner, mc, spec_attr
        self.mms = RoleMatmuls(grid_w)

    def __call__(self, x, w, role, ckc, mc):
        spec = getattr(self.mc, self.spec_attr)
        if role == "o" and spec and _decode_rows(x):
            return self.mms.run("o", spec, x, w, ckc, out_mc=mc if mc is not None else DRAM)
        return self.inner(x, w, role, ckc, mc) if self.inner is not None else None


def _pad_cols(w, N: int):
    """[.., K, n] TILE -> [.., K, N] with zero columns (device concat)."""
    n = int(w.shape[-1])
    if n == N:
        return w
    z = ttnn.from_torch(
        torch.zeros(int(w.shape[-2]), N - n),
        dtype=w.dtype,
        layout=ttnn.TILE_LAYOUT,
        device=w.device(),
        mesh_mapper=ttnn.ReplicateTensorToMesh(w.device()),
    )
    out = ttnn.concat([w, z], dim=-1)
    ttnn.deallocate(z)
    return out


def sharded_rms_norm(x, w, eps, ckc, *, max_cores=32, keep_sharded=False, out_mc=DRAM, bias=None):
    """RMSNorm (LayerNorm with bias) of one tile of rows on width-sharded L1: the interleaved ttnn norm splits by tile
    rows, so a 32-row decode activation runs on ONE core (65 us for 4096 columns). Returns the sharded output when
    keep_sharded (its memory config is the width shard of `_ds_grid(K, K)`), else `out_mc` interleaved."""
    K = int(x.shape[-1])
    cores, grid = _ds_grid(K, K, max_cores)
    mc = _width_shard(K, cores, grid)
    bw = K // 32 // cores
    pc = ttnn.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=(grid.x, grid.y),
        subblock_w=max(d for d in (4, 3, 2, 1) if bw % d == 0),
        block_h=1,
        block_w=bw,
        inplace=False,
    )
    xs = x if (x.is_sharded() and x.memory_config() == mc) else ttnn.to_memory_config(x, mc)
    fn = ttnn.layer_norm if bias is not None else ttnn.rms_norm
    kw = {"bias": bias} if bias is not None else {}
    out = fn(xs, epsilon=eps, weight=w, program_config=pc, memory_config=mc, compute_kernel_config=ckc, **kw)
    if xs is not x:
        ttnn.deallocate(xs)
    if keep_sharded:
        return out
    o = ttnn.to_memory_config(out, out_mc)
    ttnn.deallocate(out)
    return o


class MeshCCL:
    """The collectives of one decoder layer on the (1, tp) mesh (the model's shared TT_CCL semaphores), configured by
    MCOpts: topology, links, decode output memory."""

    def __init__(self, mesh, tt_ccl, mc: MCOpts):
        self.mesh, self.tt_ccl, self.mc = mesh, tt_ccl, mc
        self._topo = {}
        # ag_persistent output buffers, one per (call site, shape, dtype), shared by every layer on this TT_CCL.
        # REJECTED (off in DEFAULT_MC): with a persistent output ttnn skips the all-gather's start barrier
        # (all_gather_async_default_program_factory.cpp: "use synchronize barrier semaphore" = barrier_semaphore &&
        # !using_persistent_buffers), so nothing orders a remote write into a buffer after its consumer on the
        # receiving die. 1-layer replays (2 x 120, 200) were bit-identical, but the 45-layer model was not repeatable,
        # eager as well as traced (stage 7: s7_det_reduced_s8.log; teacher forcing top-5 32 %). Root cause not
        # established. Any CCL / buffer option must pass the stacked repeatability probe (README "Stacked gate").
        self._bufs = tt_ccl.__dict__.setdefault("_glm53_ag_bufs", {}) if tt_ccl is not None else {}

    def topology(self):
        t = self.mc.topology
        if t not in self._topo:
            if t == "linear":
                topo = ttnn.Topology.Linear
            elif t == "ring":
                topo = ttnn.Topology.Ring
            else:  # auto: Ring when the fabric the process got routes the wraparound (both p300 board links up)
                probe = ttnn.from_torch(
                    torch.zeros(1, 1, 32, 32),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.mesh,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
                )
                topo = ttnn.get_usable_topology(probe, ttnn.Topology.Ring)
                ttnn.deallocate(probe)
            self._topo[t] = topo
        return self._topo[t]

    def is_persistent(self, t) -> bool:
        return any(t is b for b in self._bufs.values())

    def release(self, t):
        """Deallocate a collective output unless it is a persistent buffer."""
        if not self.is_persistent(t):
            ttnn.deallocate(t)

    def links(self):
        return self.mc.num_links or None

    def all_gather(self, x, dim: int, dtype, decode: bool, site: str | None = None):
        if decode and self.mc.ag_persistent and site is not None and x.dtype == dtype:
            shape = list(x.shape)
            shape[dim] *= tuple(self.mesh.shape)[1]
            key = (site, tuple(shape), dtype)
            buf = self._bufs.get(key)
            if buf is None:
                # DRAM, allocated on device (no host transfer inside a pass): persistent L1 buffers sit at the top of
                # L1 for the life of the model and clashed with the static CBs of later prefill programs (suite3)
                buf = self._bufs[key] = ttnn.zeros(
                    shape, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=self.mesh, memory_config=DRAM
                )
            xi = x if not x.is_sharded() else ttnn.to_memory_config(x, DRAM)
            ttnn.experimental.all_gather_async(  # writes buf; the caller gets buf itself (release() keeps it)
                xi,
                persistent_output_buffer=buf,
                dim=dim,
                multi_device_global_semaphore=self.tt_ccl.get_and_cycle_ag_semaphore_handles(),
                num_links=self.links() or self.tt_ccl.get_num_links(),
                topology=self.topology(),
                memory_config=buf.memory_config(),
                barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(),
                chunks_per_sync=10,
                num_workers_per_link=2,
                num_buffers_per_channel=2,
            )
            ttnn.deallocate(xi)
            return buf
        return tt_all_gather(
            x,
            self.mesh,
            self.tt_ccl,
            cluster_axis=None,
            dim=dim,
            num_links=self.links(),
            memory_config=L1 if (decode and self.mc.ccl_l1) else None,
            topology=self.topology(),
            dtype=dtype,
        )

    def reduce_scatter(self, y, decode: bool):
        return tt_all_reduce(
            y,
            self.mesh,
            self.tt_ccl,
            cluster_axis=0,
            dim=3,
            num_reduce_scatter_links=self.links(),
            topology=self.topology(),
            memory_config=L1 if (decode and self.mc.ccl_l1) else DRAM,
            dtype=y.dtype,
        )


class MultichipHyperConnection(FusedHyperConnection):
    """FusedHyperConnection on the stream-interleaved slice: fn_T rows permuted / sharded to match, mix partials summed
    over the devices."""

    def __init__(self, base: FusedHyperConnection, fn: torch.Tensor, tp: int, ccl: MeshCCL):
        self.__dict__.update(base.__dict__)
        self.tp, self.ccl = tp, ccl
        self.site = None  # ag_persistent call-site name (set by MultichipDecoder)
        mesh = self.fn_T.device()
        n = self.n
        fn = fn.to(torch.bfloat16).float()  # HF: fn bf16 (applied in fp32), as HyperConnection
        K = fn.shape[1]  # n * C
        fn_T = torch.zeros(K, W)
        fn_T[:, : fn.shape[0]] = fn.t()
        # rows (stream i, column c) -> device-major (device, stream, local column): device d gets its slice's rows
        fn_T = fn_T.reshape(n, tp, K // n // tp, W).transpose(0, 1).reshape(1, 1, K, W)
        ttnn.deallocate(self.fn_T)
        self.fn_T = ttnn.from_torch(
            fn_T,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=-2),
        )
        self.K_global = K

    def _sum1(self, t, decode: bool):
        """Sum over dim 1 (keepdim) of an fp32 [1, C, T, 64]."""
        if decode and self.ccl.mc.mix_nc:
            return ttnn.experimental.fast_reduce_nc(
                t, dims=[1], output=None, compute_kernel_config=self.ckc, memory_config=DRAM
            )
        return ttnn.sum(t, dim=1, keepdim=True, memory_config=DRAM)

    def _mixes_split_k(self, x):
        T, K = x.shape[-2], x.shape[-1]
        decode = _decode_rows(x)
        chunks = mix_chunks(-(-T // 32) * 32, K // 32)
        if decode and self.ccl.mc.mix_chunks_decode and (K // 32) % self.ccl.mc.mix_chunks_decode == 0:
            chunks = self.ccl.mc.mix_chunks_decode
        part = ttnn.experimental.deepseek_prefill.mhc_mix(x, self.fn_T, chunks)
        local = self._sum1(part, decode)  # [1,1,T,64] this device's columns
        ttnn.deallocate(part)
        gathered = self.ccl.all_gather(local, dim=1, dtype=ttnn.float32, decode=decode, site=self.site)  # [1,tp,T,64]
        tot = self._sum1(gathered, decode)
        self.ccl.release(gathered)
        mixes = ttnn.slice(tot, (0, 0, 0, 0), (1, 1, T, 32))
        gram = ttnn.slice(tot, (0, 0, 0, 32), (1, 1, T, 64))
        ttnn.deallocate(tot)
        ss = ttnn.sum(ttnn.where(self._mask(T), gram, 0.0, memory_config=DRAM), dim=-1, keepdim=True)
        ttnn.deallocate(gram)
        scale = ttnn.rsqrt(ttnn.add(ttnn.multiply(ss, 1.0 / self.K_global), self.cfg.norm_eps), memory_config=DRAM)
        ttnn.deallocate(ss)
        out = ttnn.multiply(mixes, scale, memory_config=DRAM)
        ttnn.deallocate(mixes)
        ttnn.deallocate(scale)
        return out


class _ReducedOutput:
    """Delegating wrapper of a token mixer / dense MLP whose output projection is row-parallel: reduce-scatters a full
    width (partial-sum) output to the hidden-sharded y; an output that is already hidden-sharded passes through."""

    def __init__(self, inner, reduce):
        self.__dict__["_inner"] = inner
        self.__dict__["_reduce"] = reduce

    def __getattr__(self, k):
        return getattr(self._inner, k)

    def __setattr__(self, k, v):
        setattr(self._inner, k, v)

    def __call__(self, *a, **kw):
        return self._reduce(self._inner(*a, **kw))

    def prefill_forward(self, *a, **kw):
        y, state = self._inner.prefill_forward(*a, **kw)
        return self._reduce(y), state

    def decode_forward(self, *a, **kw):
        return self._reduce(self._inner.decode_forward(*a, **kw))


class MultichipDSA(GlmDSA):
    """GlmDSA with the stage-5 decode options (MultichipDecoder swaps the class of its DSA module; state unchanged).

    dsa_pack: the five replicated projections of the normed input x ([4096, 1536] q_a and [4096, 512] kv_a, bfp8;
    indexer [4096, 128] wk, [4096, 128] pool gate, [4096, 32] head weights, bf16) run as two packed matmuls instead of
    five: q_a | kv_a on a DRAM-sharded bfp8 [4096, 2048] (or the interleaved packed weight on a 1D config,
    dsa_pack_1d), wk | gate | weights on a DRAM-sharded bf16 [4096, 512] (zero-padded) with fp32 output. `_mm` hands
    out the column slices in the order decode_forward asks for them; pack_flush() frees the unused ones per step.
    The prefill rows keep the separate weights."""

    def mc_init(self, mc: MCOpts):
        self.mc = mc
        self._pack = None
        self._pack_out = {}

    def _build_pack(self):
        a = ttnn.concat([self.w_q_a, self.w_kv_a], dim=-1)  # [4096, 2048] bfp8
        grid_w = self.mesh.compute_with_storage_grid_size().x
        if self.mc.pack_a_1d:
            A = make_role_mm(self.mc.pack_a_spec, a, grid_w) if self.mc.pack_a_spec else IL1DMatmul(a, grid_w)
        else:
            A = DSMatmul(a)
            ttnn.deallocate(a)
        b = ttnn.concat([self.w_ik, self.w_ig, self.w_iw], dim=-1)  # [4096, 288] bf16
        if self.mc.pack_b_1d:
            gw = self.mesh.compute_with_storage_grid_size().x
            B = make_role_mm(self.mc.pack_b_spec, b, gw) if self.mc.pack_b_spec else IL1DMatmul(b, gw)
        else:
            bp = _pad_cols(b, tpc._roundup(int(b.shape[-1]), 32 * tpc.DRAM_CORES))
            B = DSMatmul(bp)
            ttnn.deallocate(b)
            ttnn.deallocate(bp)
        self._pack = (A, B)
        self._pack_b_1d = self.mc.pack_b_1d
        self._pack_a_1d = self.mc.pack_a_1d
        self._pack_specs = (self.mc.pack_a_spec, self.mc.pack_b_spec)
        self._pack_ids = {
            id(self.w_q_a): ("A", 0),
            id(self.w_kv_a): ("A", 1),
            id(self.w_ik): ("B", 0),
            id(self.w_ig): ("B", 1),
            id(self.w_iw): ("B", 2),
        }

    def pack_flush(self):
        for grp, (_, parts) in list(self._pack_out.items()):
            for t in parts:
                if t is not None:
                    ttnn.deallocate(t)
        self._pack_out = {}

    def free_separate(self):
        """ds_shared: the packed DRAM-sharded weights serve prefill too; free the five separate weights."""
        if self._pack is None:
            self._build_pack()
        for n in ("w_q_a", "w_kv_a", "w_ik", "w_ig", "w_iw"):
            ttnn.deallocate(getattr(self, n))
        self._shared = True

    def _mm(self, x, w, **kw):
        shared = getattr(self, "_shared", False)
        decode = _decode_rows(x)
        if (self.mc.dsa_pack and decode) or shared:
            if self._pack is None or (
                not shared
                and (self._pack_b_1d, self._pack_a_1d, self._pack_specs)
                != (self.mc.pack_b_1d, self.mc.pack_a_1d, (self.mc.pack_a_spec, self.mc.pack_b_spec))
            ):
                self._build_pack()
            key = self._pack_ids.get(id(w))
            if key is not None:
                grp, i = key
                st = self._pack_out.get(grp)
                if st is None or st[0] is not x:
                    if st is not None:
                        for t in st[1]:
                            if t is not None:
                                ttnn.deallocate(t)
                    S = x.shape[-2]
                    A, B = self._pack
                    for mm in (A, B):  # packs_prefill_2d: the packed prefill matmuls on the explicit 2D config too
                        mm.prefill_2d = self.mc.prefill_2d and self.mc.packs_prefill_2d
                    if grp == "A":
                        out = A(x, self.ckc) if decode else A.prefill(x, self.ckc)  # [1,1,S,2048] bf16
                        q = int(self.w_q_a.shape[-1])
                        parts = [
                            ttnn.slice(out, (0, 0, 0, 0), (1, 1, S, q), memory_config=DRAM),
                            ttnn.slice(out, (0, 0, 0, q), (1, 1, S, q + self.R), memory_config=DRAM),
                        ]
                    else:
                        out = B(x, self.ckc, dtype=ttnn.float32) if decode else B.prefill(x, self.ckc, ttnn.float32)
                        c0 = int(self.w_ik.shape[-1])
                        c1 = c0 + int(self.w_ig.shape[-1])
                        c2 = c1 + int(self.w_iw.shape[-1])
                        ik = ttnn.slice(out, (0, 0, 0, 0), (1, 1, S, c0), memory_config=DRAM)
                        parts = [
                            ttnn.typecast(ik, ttnn.bfloat16),
                            ttnn.slice(out, (0, 0, 0, c0), (1, 1, S, c1), memory_config=DRAM),
                            ttnn.slice(out, (0, 0, 0, c1), (1, 1, S, c2), memory_config=DRAM),
                        ]
                        ttnn.deallocate(ik)
                    ttnn.deallocate(out)
                    st = [x, parts]
                    self._pack_out[grp] = st
                t = st[1][i]
                st[1][i] = None
                want = kw.get("dtype") or ttnn.bfloat16  # _mm's default output: bf16
                if t.dtype != want:
                    t2 = ttnn.typecast(t, want)
                    ttnn.deallocate(t)
                    t = t2
                return t
        if decode and (w is self.w_q_b or w is self.w_iq):
            spec = self.mc.dsa_qb if w is self.w_q_b else self.mc.dsa_wqb
            spec = spec or ("1d" if self.mc.dsa_1d else "")
            if spec:
                mms = self.__dict__.setdefault("_role_mms", RoleMatmuls(self.mesh.compute_with_storage_grid_size().x))
                if x.is_sharded():
                    x = ttnn.to_memory_config(x, DRAM)
                return mms.run("qb" if w is self.w_q_b else "wqb", spec, x, w, self.ckc, dtype=kw.get("dtype"))
        if x.is_sharded():
            x = ttnn.to_memory_config(x, DRAM)
        if not decode and self.mc.prefill_2d and w.layout == ttnn.TILE_LAYOUT and not w.is_sharded():
            m = int(x.padded_shape[-2]) * int(math.prod(list(x.padded_shape)[:-2]))
            pc = tpc.create_prefill_matmul_program_config(m, int(w.shape[-2]), int(w.shape[-1]))
            return ttnn.linear(x, w, compute_kernel_config=self.ckc, memory_config=DRAM, program_config=pc, **kw)
        return super()._mm(x, w, **kw)

    def _rms(self, x, w):
        if self.mc.norm_sharded and _decode_rows(x):
            return sharded_rms_norm(x, w, self.eps, self.ckc)
        return super()._rms(x, w)

    # ---------------------------------------------------------------- 1-slot decode data movement (dsa_q_mm / dsa_o_mm)
    def _hm(self, x) -> bool:
        return self.S == 1 and _decode_rows(x) and not getattr(self, "dsa_shard", False)

    def _hm_consts(self):
        c = self.__dict__.get("_hm_c")
        if c is None:
            H, Dn, Dv, R = self.H, self.Dn, self.Dv, self.R
            Hq = H + (-H % 32)
            qmask = torch.zeros(1, 1, Hq, H * Dn)
            omask = torch.zeros(1, 1, H, H * Dv)
            for h in range(H):
                qmask[0, 0, h, h * Dn : (h + 1) * Dn] = 1.0
                omask[0, 0, h, h * Dv : (h + 1) * Dv] = 1.0
            grid_w = self.mesh.compute_with_storage_grid_size().x
            # in0 blocks of 8 tiles = one head's K (256) for q / half the latent (512) for o, as the per-head bmm: the
            # same partial sums per output (the other heads' blocks add exact zeros)
            ib = Dn // 32 if Dn // 32 <= 8 else 8
            c = types.SimpleNamespace(
                Hq=Hq,
                qmask=self._dev(qmask, ttnn.bfloat16),
                omask=self._dev(omask, ttnn.bfloat16) if self.mc.dsa_o_mm == 1 else None,
                wuk=ttnn.reshape(self.w_uk, (1, 1, H * Dn, R)),  # [1,H,256,512] tiles in the same order: a view
                wuv=None,
                pc_q=tpc.create_matmul_1d_decode_progcfg(
                    32, H * Dn, R, min(R // 32, 64), fp32_acc=True, grid_w=grid_w, in0_block_w=ib
                ),
                pc_o=None,
            )
            nt = H * Dv // 32
            c.pc_o = tpc.create_matmul_1d_decode_progcfg(
                32, R, H * Dv, max(d for d in range(1, 65) if nt % d == 0), fp32_acc=True, grid_w=grid_w, in0_block_w=8
            )
            self._hm_c = c
        if self.mc.dsa_o_mm == 2 and getattr(c, "wuv_v", None) is None:
            H, R, Dv = self.H, self.R, self.Dv
            om = torch.zeros(1, 1, H, H * R)
            for h in range(H):
                om[0, 0, h, h * R : (h + 1) * R] = 1.0
            c.omask2 = self._dev(om, ttnn.bfloat16)
            c.wuv_v = ttnn.reshape(self.w_uv_t, (1, 1, H * R, Dv))  # [1,H,512,256] tiles in order: a view
            grid_w = self.mesh.compute_with_storage_grid_size().x
            c.pc_o2 = tpc.create_matmul_1d_decode_progcfg(
                32, H * R, Dv, Dv // 32, fp32_acc=True, grid_w=grid_w, in0_block_w=8
            )
        if self.mc.dsa_o_mm == 1 and c.omask is None:
            H, Dv = self.H, self.Dv
            om = torch.zeros(1, 1, H, H * Dv)
            for h in range(H):
                om[0, 0, h, h * Dv : (h + 1) * Dv] = 1.0
            c.omask = self._dev(om, ttnn.bfloat16)
        if self.mc.dsa_o_mm == 1 and c.wuv is None:
            # W_UV as [512, H*256] (head h in columns h*256..): one-time relayout of [1,H,512,256] (+ H*512*256 weights)
            p = ttnn.permute(self.w_uv_t, (0, 2, 1, 3), memory_config=DRAM)
            c.wuv = ttnn.reshape(p, (1, 1, self.R, self.H * self.Dv))
            if os.environ.get("GLM53_DSA_WUV_CHECK") == "1":  # diagnostic: the relayout against a host relayout
                from loguru import logger

                ref = ttnn.to_torch(ttnn.get_device_tensors(self.w_uv_t)[0]).float()  # [1,H,512,256]
                got = ttnn.to_torch(ttnn.get_device_tensors(c.wuv)[0]).float().reshape(self.R, self.H, self.Dv)
                d = (got - ref[0].permute(1, 0, 2)).abs().max()
                logger.warning(f"DSA W_UV relayout ({self.w_uv_t.dtype} -> {c.wuv.dtype}): max |diff| {float(d):.3e}")
        return c

    def _q_abs(self, x, rows: int):
        if not (self.mc.dsa_q_mm and self._hm(x)):
            return super()._q_abs(x, rows)
        c = self._hm_consts()
        qr = self._rms(self._mm(x, self.w_q_a), self.q_a_norm)
        q = self._mm(qr, self.w_q_b)  # [1,1,1,H*256]
        qm = ttnn.multiply(c.qmask, q, memory_config=DRAM)  # row h = q on head h's columns (the row broadcast), 0 else
        ttnn.deallocate(q)
        qa = ttnn.matmul(qm, c.wuk, compute_kernel_config=self.ckc, program_config=c.pc_q, memory_config=DRAM)
        ttnn.deallocate(qm)
        # [1,1,Hq,512] tiles -> row-major [1, Hq, 1, 512]: the sparse_sdpa query (zero heads H..Hq-1 included)
        q_rm = ttnn.reshape(ttnn.to_layout(qa, ttnn.ROW_MAJOR_LAYOUT, memory_config=DRAM), (1, c.Hq, 1, self.R))
        ttnn.deallocate(qa)
        return q_rm, qr

    def _sparse_attend(self, q_abs, index_f32, cache_batch_idx=None, latent_rows=None):
        rm_q = q_abs.layout == ttnn.ROW_MAJOR_LAYOUT
        o_rm = self.mc.dsa_o_mm and cache_batch_idx is None and self.S == 1 and not getattr(self, "dsa_shard", False)
        if not (rm_q or o_rm):
            return super()._sparse_attend(q_abs, index_f32, cache_batch_idx, latent_rows)
        idx = ttnn.to_layout(ttnn.typecast(index_f32, ttnn.uint32), ttnn.ROW_MAJOR_LAYOUT)
        if rm_q:
            q_rm = q_abs
        else:
            pad = -int(q_abs.shape[1]) % 32
            q_p = ttnn.pad(q_abs, [(0, 0), (0, pad), (0, 0), (0, 0)], 0.0) if pad else q_abs
            q_rm = ttnn.to_layout(q_p, ttnn.ROW_MAJOR_LAYOUT)
            if pad:
                ttnn.deallocate(q_p)
        o = ttnn.transformer.sparse_sdpa(
            q_rm,
            self.latent,
            idx,
            self.R,
            kv_format=(
                ttnn.SparseKVFormat.FP8_E4M3 if self.latent.dtype == ttnn.fp8_e4m3 else ttnn.SparseKVFormat.BF16
            ),
            scale=self.scale,
            k_chunk_size=32,
            compute_kernel_config=self.sdpa_ckc,
            cache_batch_idx=cache_batch_idx,
            page_table=self.page_table,
            page_block_size=BLOCK,
        )
        ttnn.deallocate(idx)
        ttnn.deallocate(q_rm)
        o_h = ttnn.slice(o, (0, 0, 0, 0), (1, self.H, o.shape[2], o.shape[3]))  # row-major [1, H, 1, 512]
        ttnn.deallocate(o)
        if o_rm:
            return o_h  # _out takes the row-major rows
        out = ttnn.to_layout(o_h, ttnn.TILE_LAYOUT)
        ttnn.deallocate(o_h)
        return out

    def _out(self, o_lat, rows: int):
        if o_lat.layout != ttnn.ROW_MAJOR_LAYOUT:
            return super()._out(o_lat, rows)
        c = self._hm_consts()
        if self.mc.dsa_o_mm == 2:
            H, R, Dv = self.H, self.R, self.Dv
            flat = ttnn.to_layout(ttnn.reshape(o_lat, (1, 1, 1, H * R)), ttnn.TILE_LAYOUT, memory_config=DRAM)
            inm = ttnn.multiply(c.omask2, flat, memory_config=DRAM)  # row h = head h's 512 values in block h
            ttnn.deallocate(flat)
            oh = ttnn.matmul(inm, c.wuv_v, compute_kernel_config=self.ckc, program_config=c.pc_o2, memory_config=DRAM)
            ttnn.deallocate(inm)
            rm = ttnn.reshape(ttnn.to_layout(oh, ttnn.ROW_MAJOR_LAYOUT, memory_config=DRAM), (1, 1, 1, H * Dv))
            ttnn.deallocate(oh)
            o = ttnn.to_layout(rm, ttnn.TILE_LAYOUT, memory_config=DRAM)
            ttnn.deallocate(rm)
            out = self.decode_mm(o, self.w_o, "o", self.ckc, DRAM) if self.decode_mm is not None else None
            out = out if out is not None else self._mm(o, self.w_o)
            ttnn.deallocate(o)
            return out
        ot = ttnn.to_layout(ttnn.reshape(o_lat, (1, 1, self.H, self.R)), ttnn.TILE_LAYOUT, memory_config=DRAM)
        full = ttnn.matmul(ot, c.wuv, compute_kernel_config=self.ckc, program_config=c.pc_o, memory_config=DRAM)
        ttnn.deallocate(ot)
        masked = ttnn.multiply(full, c.omask, memory_config=DRAM)  # row h keeps head h's 256 columns
        ttnn.deallocate(full)
        # one nonzero per column: the concat, exact at HiFi4 (ttnn.sum's default fidelity truncates the value's
        # mantissa: PCC 0.9992 in the A/B, records/dsa_dm_ab.jsonl)
        o = ttnn.sum(masked, dim=-2, keepdim=True, memory_config=DRAM, compute_kernel_config=self.ckc)
        ttnn.deallocate(masked)
        out = self.decode_mm(o, self.w_o, "o", self.ckc, DRAM) if self.decode_mm is not None else None
        out = out if out is not None else self._mm(o, self.w_o)
        ttnn.deallocate(o)
        return out

    # ---------------------------------------------------------------------- sparse index tail (dsa_tail_shared)
    def _decode_host_inputs(self, positions):
        d = super()._decode_host_inputs(positions)
        # A/B only (GLM53_AB_HOST_TAIL=1, with dsa_tail_shared): the tail as a host-written per-step input, standing for a
        # layer that reads the first DSA layer's tail. Never in a model: DeviceDecodeInputs refreshes only its own fields
        if (
            os.environ.get("GLM53_AB_HOST_TAIL") == "1"
            and getattr(self, "mc", None) is not None
            and self.mc.dsa_tail_shared
        ):
            S = self.S
            pos = torch.tensor([max(p, 0) for p in positions], dtype=torch.float32).reshape(S, 1)
            tail_cols = IDX_WIDTH - TOPK_POOLS * KPOOL
            tok = torch.floor((pos + 1) / KPOOL) * KPOOL + torch.arange(tail_cols).float().reshape(1, tail_cols)
            tail = torch.where(tok <= pos, tok, torch.full_like(tok, SENTINEL_F))
            d["tail"] = (tail.reshape(1, 1, S, tail_cols), ttnn.float32, ttnn.TILE_LAYOUT)
        return d

    def _tail(self, pos_dev):
        """Tail tokens 4*floor((t+1)/4) .. t (0-3 of them), then sentinels: fp32 [1,1,rows,IDX_WIDTH - 2048]."""
        tail_cols = IDX_WIDTH - TOPK_POOLS * KPOOL
        j = self._const(("tail_j", tail_cols))
        start = ttnn.multiply(ttnn.floor(ttnn.multiply(ttnn.add(pos_dev, 1.0), 1.0 / KPOOL)), float(KPOOL))
        tail_tok = ttnn.add(start, j)
        tail_ok = ttnn.le(tail_tok, pos_dev)
        tail = ttnn.where(tail_ok, tail_tok, SENTINEL_F)
        for t in (tail_tok, tail_ok, start):
            ttnn.deallocate(t)
        return tail

    def _sparse_rows(self, top_pools, pos_dev, rows: int):
        inp = self.__dict__.get("_step_inputs")
        if inp is None:
            return super()._sparse_rows(top_pools, pos_dev, rows)
        own = None
        tail = inp.get("tail")  # per-step host input (A/B: a consumer layer)
        if tail is None:
            if getattr(self, "_tail_producer", False):
                # this step's tail, for the later DSA layers, in ONE fixed buffer per input dict (never freed: every
                # captured trace of the model reads the address it was captured with; a fresh tensor per call freed the
                # buffer an earlier trace still read, teacher forcing 10 / 48 / 100, records/s5r2_tf_cand.log)
                t = self._tail(pos_dev)
                buf = inp.get("_tail")
                if buf is None or tuple(buf.shape) != tuple(t.shape):
                    buf = inp["_tail"] = ttnn.clone(t)  # the first (eager) step: allocate, and compile the copy below
                ttnn.copy(t, buf)
                ttnn.deallocate(t)
                tail = buf
            elif "_tail" in inp:
                tail = inp["_tail"]
            else:
                tail = own = self._tail(pos_dev)
        offs = self._const("pool_offs")
        p = ttnn.typecast(top_pools, ttnn.float32)
        p4 = ttnn.repeat_interleave(p, KPOOL, dim=3)
        ttnn.deallocate(p)
        tok = ttnn.add(ttnn.multiply(p4, float(KPOOL)), offs)
        ttnn.deallocate(p4)
        out = ttnn.concat([tok, tail], dim=-1)
        ttnn.deallocate(tok)
        if own is not None:
            ttnn.deallocate(own)
        return out

    def decode_forward(self, x, state=None, *, inputs):
        self._step_inputs = inputs if self.mc.dsa_tail_shared else None
        try:
            return super().decode_forward(x, state, inputs=inputs)
        finally:
            self._step_inputs = None


def _pad_rows(t, rows: int):
    """[1,1,r,C] TILE -> [1,1,rows,C] with zero rows (inside the tile: ttnn.pad; a row-major round trip if refused)."""
    r = int(t.shape[-2])
    try:
        return ttnn.pad(t, [(0, 0), (0, 0), (0, rows - r), (0, 0)], 0.0)
    except RuntimeError as e:  # a pad that refuses non-tile-aligned tile padding: row-major round trip
        from loguru import logger

        logger.warning(f"_pad_rows: in-tile pad refused ({str(e)[:100]}); row-major round trip")
        rm = ttnn.to_layout(t, ttnn.ROW_MAJOR_LAYOUT)
        p = ttnn.pad(rm, [(0, 0), (0, 0), (0, rows - r), (0, 0)], 0.0)
        ttnn.deallocate(rm)
        out = ttnn.to_layout(p, ttnn.TILE_LAYOUT)
        ttnn.deallocate(p)
        return out


class MultichipKDA(FusedGlmKDA):
    """FusedGlmKDA whose decode batch is padded so that batch x local heads fills whole tile rows: the recurrence op
    (ttnn.experimental.kda.recurrent_decode) needs B * H % 32 == 0, and TP4 leaves H = 16 local heads, so a 1-slot
    decode (B = 1) is padded with one inactive dummy slot (state unchanged, output row dropped)."""

    def _pad_batch(self, B: int) -> int:
        step = 32 // math.gcd(32, self.H)
        return -(-B // step) * step

    def allocate_decode_state(self, batch: int | None = None):
        B = batch or self.max_batch
        return super().allocate_decode_state(self._pad_batch(B))

    # the per-slot inputs are allocated with the padded slot count (extra slots inactive): only x needs a pad per step
    # (a [B, 1] pad runs on one core at ~10 us)
    def decode_inputs(self, positions):
        return super().decode_inputs(list(positions) + [-1] * (self._pad_batch(len(positions)) - len(positions)))

    def update_decode_inputs(self, bufs, positions):
        super().update_decode_inputs(bufs, list(positions) + [-1] * (self._pad_batch(len(positions)) - len(positions)))

    def _gated_norm(self, o, ogate, B, fold):
        if not (hasattr(self, "mc") and self.mc.kda_gnorm_op):
            return super()._gated_norm(o, ogate, B, fold)
        H, V = self.H, self.V
        w = self.__dict__.get("_gnorm_w")
        if w is None:  # the o_norm weight as the op's bf16 [V] (the checkpoint weight is bf16: exact)
            w = self._gnorm_w = ttnn.typecast(ttnn.reshape(self.o_norm_w, (V,)), ttnn.bfloat16)
        # the op needs a whole tile of T: each (slot, head) row becomes its own tile row, padded to 32 zero rows
        o1 = ttnn.reshape(o, (B * H, 1, V))
        ttnn.deallocate(o)
        o3 = ttnn.pad(o1, [(0, 0), (0, 31), (0, 0)], 0.0)
        ttnn.deallocate(o1)
        g = ogate if ogate.dtype == ttnn.bfloat16 else ttnn.typecast(ogate, ttnn.bfloat16)
        g1 = ttnn.reshape(g, (B, 1, H * V))
        g3 = ttnn.pad(g1, [(0, 0), (0, 31), (0, 0)], 0.0)
        ttnn.deallocate(g1)
        out = ttnn.experimental.kda.sigmoid_gated_rms_norm(
            o3,
            g3,
            w,
            H,
            epsilon=self.cfg.norm_eps,
            memory_config=DRAM,
            compute_kernel_config=self.ckc,
            output_dtype=ttnn.bfloat16,
        )  # [B, 32, H*V]
        for t in (o3, g3, ogate) if g is ogate else (o3, g3, g, ogate):
            ttnn.deallocate(t)
        row = ttnn.slice(out, (0, 0, 0), (B, 1, H * V))
        ttnn.deallocate(out)
        return ttnn.reshape(row, (1, 1, B, H * V))

    def _conv_step(self, proj, dstate, inputs, B):
        if (
            not (hasattr(self, "mc") and self.mc.kda_conv_op)
            or not self.flags.kda_decode_fold
            or (self.mc.kda_conv_max_batch and B > self.mc.kda_conv_max_batch)
        ):
            return super()._conv_step(proj, dstate, inputs, B)
        # one op: conv (fp32) + SiLU + idle-slot select, history shifted in place (tests/test_kda_decode_conv.py)
        return ttnn.experimental.deepseek_prefill.kda_decode_conv(
            proj, *dstate.conv, *self.taps32, inputs["active"], self.W
        )

    def _conv_fold(self, cur, dstate):
        if not (hasattr(self, "mc") and self.mc.kda_conv_mixed):
            return super()._conv_fold(cur, dstate)
        acc = ttnn.multiply(cur, self.taps32[3], dtype=ttnn.float32, memory_config=DRAM)
        for j in range(3):  # bf16 history x fp32 tap into the fp32 accumulator (exact widening inside the op)
            nxt = ttnn.addcmul(acc, dstate.conv[j], self.taps32[j], memory_config=DRAM)
            ttnn.deallocate(acc)
            acc = nxt
        return acc

    def _in_proj(self, x, w, cc):
        spec = self.mc.kda_in if hasattr(self, "mc") else ""
        if spec or w.is_sharded():
            mms = self.__dict__.setdefault("_role_mms", RoleMatmuls(self.mesh.compute_with_storage_grid_size().x))
            return mms.run("in", spec or self._shared_in_spec, x, w, cc)
        return super()._in_proj(x, w, cc)

    def prefill_forward(self, x, state, bounds):
        """prefill_2d_ffn: the output projection of the prefill layer on the explicit 2D config (ttnn's / the tuned
        config otherwise)."""
        n = x.shape[-2]
        layer = self._prefill.get(n)
        if layer is not None and hasattr(self, "mc"):
            if "_out_pc0" not in layer.__dict__:
                layer._out_pc0 = layer.output_projection_program_config
            w = self.weights.output_projection
            layer.output_projection_program_config = (
                prefill_pc(x, int(w.shape[-2]), int(w.shape[-1])) if self.mc.prefill_2d_ffn else layer._out_pc0
            )
        return super().prefill_forward(x, state, bounds)

    def decode_forward(self, x, dstate, *, inputs):
        B = x.shape[-2]
        Bp = int(dstate.recurrent.shape[0])
        if Bp == B:
            return super().decode_forward(x, dstate, inputs=inputs)
        pad = lambda t: t if int(t.shape[-2]) == Bp else _pad_rows(t, Bp)  # padded slots: zero rows (inactive)
        xp = pad(x)
        ip = {k: pad(v) for k, v in inputs.items()}
        y = super().decode_forward(xp, dstate, inputs=ip)
        # no deallocate of xp / ip: a pad inside the tile padding is a view of the caller's buffer (freeing it freed
        # the decode input buffers: segfault at the next host copy, 2026-10-17 05:07)
        out = ttnn.slice(y, (0, 0, 0, 0), (1, 1, B, int(y.shape[-1])), memory_config=DRAM)
        ttnn.deallocate(y)
        return out


class MultichipMLP(DenseMLP):
    """DenseMLP (layers 0-2, packed gate | up) with the stage-5 role programs: decode rows on mlp_gu / mlp_dn specs,
    prefill rows on the explicit 2D config (prefill_2d_ffn). MultichipDecoder swaps the class; weights unchanged."""

    def forward(self, x):
        mc = getattr(self, "mc", None)
        decode = _decode_rows(x)
        if (
            self.wgu is None
            or mc is None
            or not ((decode and (mc.mlp_gu or mc.mlp_dn)) or (not decode and mc.prefill_2d_ffn))
        ):
            return super().forward(x)
        mms = self.__dict__.setdefault("_role_mms", RoleMatmuls(x.device().compute_with_storage_grid_size().x))

        def mm(role, spec, a, w):
            if decode and spec:
                return mms.run(role, spec, a, w, self.ckc)
            pc = prefill_pc(a, int(w.shape[-2]), int(w.shape[-1])) if not decode else None
            return ttnn.linear(a, w, compute_kernel_config=self.ckc, memory_config=DRAM, program_config=pc)

        gu = mm("gu", mc.mlp_gu, x, self.wgu)
        shp = list(gu.shape)
        il = shp[-1] // 2
        g = ttnn.slice(gu, [0] * len(shp), shp[:-1] + [il], memory_config=DRAM)
        u = ttnn.slice(gu, [0] * (len(shp) - 1) + [il], shp, memory_config=DRAM)
        ttnn.deallocate(gu)
        h = ttnn.multiply(
            u, g, input_tensor_a_activations=self.act_up, input_tensor_b_activations=self.act_gate, memory_config=DRAM
        )
        ttnn.deallocate(g)
        ttnn.deallocate(u)
        out = mm("dn", mc.mlp_dn, h, self.wd)
        ttnn.deallocate(h)
        return out


def _install_shared(moe, mc: MCOpts):
    """MoE shared expert (packed gate | up, down; ungated in GLM) with the stage-5 role programs: decode rows on
    shared_gu / shared_dn specs (the others keep moe_opt's 1D decode config), prefill rows on the explicit 2D config
    (prefill_2d_ffn)."""
    orig = moe._shared

    def _shared(self, x4):
        decode = _decode_rows(x4)
        use = (decode and (mc.shared_gu or mc.shared_dn)) or (not decode and mc.prefill_2d_ffn)
        if not use or self.shared_gated or getattr(self, "shared_packed", None) is None:
            return orig(x4)
        mms = self.__dict__.setdefault("_role_mms", RoleMatmuls(self.device.compute_with_storage_grid_size().x))
        pcs = getattr(self, "_shared_pc", None) or (None, None)
        dram = ttnn.DRAM_MEMORY_CONFIG
        Sl = self.shared_inter // self.num_devices
        T = x4.shape[2]

        def mm(role, spec, a, w, pc_decode):
            if decode and spec:
                return mms.run(role, spec, a, w, self.shared_ckc)
            pc = prefill_pc(a, int(w.shape[-2]), int(w.shape[-1])) if not decode else pc_decode
            a_in = ttnn.to_memory_config(a, ttnn.L1_MEMORY_CONFIG) if a.is_sharded() else a
            return ttnn.linear(a_in, w, compute_kernel_config=self.shared_ckc, program_config=pc, memory_config=dram)

        fused = mm("sgu", mc.shared_gu, x4, self.shared_packed, pcs[0])
        gate = ttnn.slice(fused, [0, 0, 0, 0], [1, 1, T, Sl], memory_config=dram)
        up = ttnn.slice(fused, [0, 0, 0, Sl], [1, 1, T, 2 * Sl], memory_config=dram)
        ttnn.deallocate(fused)
        gate, up = self._clamp_gate_up(gate, up, dram)
        hidden = ttnn.multiply(up, gate, input_tensor_b_activations=[ttnn.UnaryOpType.SILU], memory_config=dram)
        ttnn.deallocate(gate)
        ttnn.deallocate(up)
        out = mm("sdn", mc.shared_dn, hidden, self.shared_w2, pcs[1])
        ttnn.deallocate(hidden)
        return out

    moe._shared = types.MethodType(_shared, moe)


def _install_router_ds(moe, mc: MCOpts):
    """MoE decode router logits [T, 4096] x [4096, 192] on a 1D config (router_1d, the interleaved weight, 6 cores with
    in0 blocks of 8) or on a DRAM-sharded weight zero-padded to 256 columns (router_ds) instead of ttnn's 6-core default
    program (49 us). Prefill rows keep the interleaved weight unless ds_shared moved them to the DRAM-sharded one."""
    orig = moe._router_logits
    state = {}

    def build(self):
        E = int(self.router_w.shape[-1])
        state["E"] = E
        key = "1d" if mc.router_1d else "ds"
        if key not in state:
            if mc.router_1d:  # the interleaved weight itself (shared with prefill; nothing to free)
                state[key] = IL1DMatmul(self.router_w, self.device.compute_with_storage_grid_size().x)
            else:
                wp = _pad_cols(self.router_w, tpc._roundup(E, 32 * tpc.DRAM_CORES))
                state[key] = DSMatmul(wp)
                if wp is not self.router_w:
                    ttnn.deallocate(wp)
        state["mm"] = state[key]

    def _router_logits(self, x4, r_mc, dtype=ttnn.bfloat16):
        decode = _decode_rows(x4)
        shared = state.get("shared", False)
        if not (((mc.router_ds or mc.router_1d) and decode) or shared) or getattr(self, "_router_sharded", False):
            return orig(x4, r_mc, dtype)
        if not shared:
            build(self)  # (cached per mode)
        ds, E = state["mm"], state["E"]
        T = x4.shape[-2]
        if decode:
            out = ds(x4, self.router_ckc, dtype=dtype, out_mc=r_mc)
        else:
            out = ds.prefill(x4, self.router_ckc, dtype=dtype)
        if int(out.shape[-1]) == E:
            return out
        lo = ttnn.slice(out, (0, 0, 0, 0), (1, 1, T, E), memory_config=r_mc)
        ttnn.deallocate(out)
        return lo

    moe._router_logits = types.MethodType(_router_logits, moe)
    if mc.ds_shared and mc.router_ds and not mc.router_1d:  # the DRAM-sharded weight serves prefill: free the other
        build(moe)
        ttnn.deallocate(moe.router_w)
        state["shared"] = True


class MultichipDecoder(OptimizedDecoder):
    @classmethod
    def from_state_dict(cls, state_dict, *, hf_config, layer_idx, mesh_device, mc: MCOpts | None = None, **kw):
        tp = tuple(mesh_device.shape)[1]
        tt_ccl = kw.get("tt_ccl") or (TT_CCL(mesh_device) if tp > 1 else None)
        kw["tt_ccl"] = tt_ccl
        # stage-8 datatype-sweep candidate (GLM53_PRECISION, tt/precision.py; unset = the shipped policy, no change)
        from .precision import PrecisionOverrides

        prec, last = PrecisionOverrides.from_env(), hf_config["num_hidden_layers"] - 1
        kw["opt"] = prec.layer_opt(kw.get("opt"), layer_idx, 0, last)
        kw["weight_cache"] = prec.layer_cache(kw.get("weight_cache"), layer_idx)
        dec = super().from_state_dict(
            state_dict, hf_config=hf_config, layer_idx=layer_idx, mesh_device=mesh_device, **kw
        )
        dec.tt_ccl = tt_ccl
        dec.mc = MCOpts.from_env(mc or DEFAULT_MC)
        dec.ccl = MeshCCL(mesh_device, tt_ccl, dec.mc) if tp > 1 else None
        if dec.ccl is not None:
            dec.ccl.topology()  # resolve "auto" now (its probe tensor is a host write: never inside a trace capture)
        if dec.attn_kind == "dsa":
            dec.attn.__class__ = MultichipDSA
            dec.attn.mc_init(dec.mc)
            # the model's first DSA layer computes the step's shared sparse-index tail (dsa_tail_shared)
            kinds = hf_config["layer_types"]
            dec.attn._tail_producer = layer_idx == next(i for i, k in enumerate(kinds) if k != "linear_attention")
            if dec.mc.ds_shared and dec.mc.dsa_pack:
                dec.attn.free_separate()
            if dec.mc.dsa_shard and tp > 1:
                from .dsa_shard import install_dsa_shard

                install_dsa_shard(dec.attn, dec.ccl)
        elif isinstance(dec.attn, FusedGlmKDA):
            dec.attn.__class__ = MultichipKDA
            dec.attn.mc = dec.mc
        grid_w = mesh_device.compute_with_storage_grid_size().x
        dec.attn.decode_mm = RoleDecodeMM(
            getattr(dec.attn, "decode_mm", None), dec.mc, "dsa_o" if dec.attn_kind == "dsa" else "kda_out", grid_w
        )
        if dec.mlp_kind == "moe":
            _install_router_ds(dec.mlp, dec.mc)
            _install_shared(dec.mlp, dec.mc)
            if dec.mc.moe_grouped_in0:  # TP4 geometry sweep (README "Sparse matmul geometry")
                dec.mlp.opt = dataclasses.replace(
                    dec.mlp.opt, grouped_gate_up_in0=dec.mc.moe_grouped_in0, grouped_down_in0=dec.mc.moe_grouped_in0
                )
                dec.mlp._cfg_grouped = None
        elif isinstance(dec.mlp, DenseMLP):
            dec.mlp.__class__ = MultichipMLP
            dec.mlp.mc = dec.mc
        if dec.mc.sinkhorn_perm_fidelity and "MHC_SINKHORN_PERM_FIDELITY" not in os.environ:
            os.environ["MHC_SINKHORN_PERM_FIDELITY"] = str(dec.mc.sinkhorn_perm_fidelity)
        if dec.mc.sinkhorn_faces and "MHC_SINKHORN_FACES" not in os.environ:
            os.environ["MHC_SINKHORN_FACES"] = "1"
        if tp > 1:
            dec.attn_hc = MultichipHyperConnection(dec.attn_hc, state_dict["hc_attn_fn"], tp, dec.ccl)
            dec.ffn_hc = MultichipHyperConnection(dec.ffn_hc, state_dict["hc_ffn_fn"], tp, dec.ccl)
            dec.attn_hc.site, dec.ffn_hc.site = "mix_attn", "mix_ffn"
            dec.attn = _ReducedOutput(dec.attn, dec._reduce)
            if dec.mlp_kind == "dense":
                dec.mlp = _ReducedOutput(dec.mlp, dec._reduce)
            else:  # the MoE reduce-scatters inside the module (tt/moe.py)
                dec.mlp.args.ccl_topology = dec.ccl.topology
                dec.mlp.args.ccl_num_links = dec.ccl.links
                dec.mlp.args.router_topk_pad = lambda: dec.mc.router_topk_pad
                dec.mlp.args.ccl_out_memory_config = lambda t: L1 if (dec.mc.ccl_l1 and _decode_rows(t)) else DRAM
        prec.apply_layer(dec, layer_idx, 0, last)
        return dec

    def _reduce(self, y):
        if self.tp == 1 or y.shape[-1] != self.hidden:
            return y
        return self.ccl.reduce_scatter(y, decode=_decode_rows(y))

    def _norm(self, h, w):
        decode = _decode_rows(h)
        if self.tp > 1:  # h [T, C/tp] -> replicated [T, C]
            h = self.ccl.all_gather(
                h, dim=3, dtype=h.dtype, decode=decode, site="h_attn" if w is self.input_norm_w else "h_ffn"
            )
        if decode and self.mc.norm_sharded and h.dtype == ttnn.bfloat16:
            # the DSA packed projections take the width-sharded norm output as is (same shard as their activation)
            keep = self.attn_kind == "dsa" and self.mc.dsa_pack and not self.mc.pack_a_1d and w is self.input_norm_w
            out = sharded_rms_norm(h, w, self.eps, self.norm_ckc, keep_sharded=keep)
            if self.ccl is not None:
                self.ccl.release(h)
            else:  # TP = 1 (one die; review #1 P1-5 measurement): h is the norm input of this step, as FusedDecoder._norm
                ttnn.deallocate(h)
            return out
        if self.ccl is not None and self.ccl.is_persistent(h):  # FusedDecoder._norm would free the buffer
            return ttnn.rms_norm(h, epsilon=self.eps, weight=w, memory_config=DRAM, compute_kernel_config=self.norm_ckc)
        return super()._norm(h, w)

    def decode_forward(self, X, *, inputs):
        out = super().decode_forward(X, inputs=inputs)
        if self.attn_kind == "dsa":
            self.attn.pack_flush()
        return out

    def prefill_forward(self, X, *, inputs, attn_state=None):
        out = super().prefill_forward(X, inputs=inputs, attn_state=attn_state)
        if self.attn_kind == "dsa":
            self.attn.pack_flush()
        return out
