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

from .dsa import GlmDSA
from .fused_decoder import FusedGlmKDA, FusedHyperConnection, mix_chunks
from .mhc import DRAM, W
from .optimized_decoder import OptimizedDecoder

TOPOLOGY = ttnn.Topology.Linear  # stage-4 default; MCOpts.topology selects per decoder (stage 5)
L1 = ttnn.L1_MEMORY_CONFIG


@dataclass
class MCOpts:
    """Stage-5 multichip decode options (doc/optimized_multichip_decoder/README.md has the A/B measurements).
    Every option only changes decode rows (one tile of users); prefill keeps the stage-4 path.

    GLM53_MC="key=value,..." overrides fields of DEFAULT_MC at construction (A/B runs; tools/mc_decode_ab.py)."""

    topology: str = "linear"  # CCL topology: linear | ring | auto (ring when the fabric has the wraparound link)
    num_links: int = 0  # CCL links per collective (0 = all available)
    ccl_l1: bool = False  # decode all-gather / reduce-scatter outputs in L1 instead of DRAM
    norm_sharded: bool = False  # decode RMSNorms on width-sharded L1 over many cores (ttnn picks 1 core interleaved)
    dsa_pack: bool = False  # DSA decode: q_a | kv_a and indexer wk | gate | weights packed, DRAM-sharded matmuls
    router_ds: bool = False  # MoE decode router matmul on a DRAM-sharded (zero-padded 192 -> 256) weight
    mix_nc: bool = False  # mHC mix partial sums with fast_reduce_nc instead of ttnn.sum (transpose + reduce)
    ag_persistent: bool = False  # decode all-gathers write into preallocated (per call site) output buffers
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

    @classmethod
    def from_env(cls, base: "MCOpts") -> "MCOpts":
        spec = os.environ.get("GLM53_MC", "").strip()
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
)


def _decode_rows(t) -> bool:
    """One tile of rows (decode: users <= 32); prefill chunks are >= 128 rows."""
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


class DSMatmul:
    """Decode matmul (one tile of rows) on a DRAM width-sharded weight [K, N] (N a multiple of 32 * 8 banks): every
    DRAM bank streams its own column block (the bandwidth-bound decode form; ttnn's default program puts an N <= 9 tile
    weight on N cores)."""

    def __init__(self, w_il, max_cores: int = 32):
        K, N = int(w_il.shape[-2]), int(w_il.shape[-1])
        assert N % (32 * tpc.DRAM_CORES) == 0, N
        self.l1_acc = True  # packer L1 accumulation of the K-block partials (acc_ckc); False = the caller's config
        self.K, self.N = K, N
        self.w = ttnn.to_memory_config(w_il, tpc.create_dram_sharded_mem_config(K, N))
        cores, grid = _ds_grid(K, N, max_cores)
        self.act = _width_shard(K, cores, grid)
        self.pc = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
            in0_block_w=tpc._find_largest_divisor(K // 32 // cores),
            per_core_M=1,
            per_core_N=N // 32 // cores,
            fused_activation=None,
        )

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
        m = int(x.padded_shape[-2]) * int(math.prod(list(x.padded_shape)[:-2]))
        pc = tpc.create_prefill_matmul_program_config(m, self.K, self.N)
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

    def __init__(self, w_il, grid_w: int, max_cores: int = 64):
        K, N = int(w_il.shape[-2]), int(w_il.shape[-1])
        self.K, self.N, self.w = K, N, w_il
        nt = N // 32
        cores = max(c for c in range(1, min(nt, max_cores) + 1) if nt % c == 0)  # whole output tiles per core
        ib = max(d for d in (8, 4, 2, 1) if (K // 32) % d == 0)
        self.pc = tpc.create_matmul_1d_decode_progcfg(32, K, N, cores, fp32_acc=True, grid_w=grid_w, in0_block_w=ib)

    def prefill(self, x, ckc, dtype=None):
        kw = {"dtype": dtype} if dtype is not None else {}
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
        self._bufs = {}  # ag_persistent output buffers

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
                buf = self._bufs[key] = ttnn.from_torch(
                    torch.zeros(shape),
                    dtype=dtype,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.mesh,
                    memory_config=L1 if self.mc.ccl_l1 else DRAM,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
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
        part = ttnn.experimental.deepseek_prefill.mhc_mix(x, self.fn_T, mix_chunks(-(-T // 32) * 32, K // 32))
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
            A = IL1DMatmul(a, grid_w)
        else:
            A = DSMatmul(a)
            ttnn.deallocate(a)
        b = ttnn.concat([self.w_ik, self.w_ig, self.w_iw], dim=-1)  # [4096, 288] bf16
        if self.mc.pack_b_1d:
            B = IL1DMatmul(b, self.mesh.compute_with_storage_grid_size().x)
        else:
            bp = _pad_cols(b, tpc._roundup(int(b.shape[-1]), 32 * tpc.DRAM_CORES))
            B = DSMatmul(bp)
            ttnn.deallocate(b)
            ttnn.deallocate(bp)
        self._pack = (A, B)
        self._pack_b_1d = self.mc.pack_b_1d
        self._pack_a_1d = self.mc.pack_a_1d
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
                not shared and (self._pack_b_1d, self._pack_a_1d) != (self.mc.pack_b_1d, self.mc.pack_a_1d)
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
        if decode and self.mc.dsa_1d and (w is self.w_q_b or w is self.w_iq):
            cache = self.__dict__.setdefault("_mm1d", {})
            mm = cache.get(id(w))
            if mm is None:  # the interleaved weight itself, on a 1D decode config
                mm = cache[id(w)] = IL1DMatmul(w, self.mesh.compute_with_storage_grid_size().x)
            return mm(x, self.ckc, dtype=kw.get("dtype"))
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


def _pad_rows(t, rows: int):
    """[1,1,r,C] TILE -> [1,1,rows,C] with zero rows (inside the tile: ttnn.pad; a row-major round trip if refused)."""
    r = int(t.shape[-2])
    try:
        return ttnn.pad(t, [(0, 0), (0, 0), (0, rows - r), (0, 0)], 0.0)
    except Exception:  # noqa: BLE001 - older pad: tile-layout padding must be tile aligned
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
            if dec.mc.ds_shared and dec.mc.dsa_pack:
                dec.attn.free_separate()
        elif isinstance(dec.attn, FusedGlmKDA):
            dec.attn.__class__ = MultichipKDA
        if dec.mlp_kind == "moe":
            _install_router_ds(dec.mlp, dec.mc)
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
                dec.mlp.args.ccl_out_memory_config = lambda t: L1 if (dec.mc.ccl_l1 and _decode_rows(t)) else DRAM
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
            self.ccl.release(h)
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
