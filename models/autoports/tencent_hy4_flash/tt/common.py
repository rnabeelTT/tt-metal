# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Mesh helpers shared by the Hy4 modules: uploads (replicated / sharded), compute-kernel configs, the all-reduce.

Layout contract of the port (TP over a (1, N) mesh, N = 4 on a QB2): the residual (the 4 hyper-connection streams) and
every sublayer input are REPLICATED on all dies; column-parallel weights shard their output dim, row-parallel weights
their input dim, and each sublayer ends in one all-reduce that leaves its output replicated.
"""

from __future__ import annotations

import os

import torch
import ttnn

TILE = 32


def num_devices(mesh) -> int:
    return int(mesh.get_num_devices())


def rep(mesh):
    return ttnn.ReplicateTensorToMesh(mesh) if num_devices(mesh) > 1 else None


def shard(mesh, dim):
    return ttnn.ShardTensorToMesh(mesh, dim=dim) if num_devices(mesh) > 1 else None


def _kw(mapper):
    return {"mesh_mapper": mapper} if mapper is not None else {}


def upload(mesh, t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mapper="rep", mem=None, cache=None):
    """torch -> device. mapper: "rep", ("shard", dim), or a ttnn mapper. cache: optional tensor-cache file name."""
    if mapper == "rep":
        mapper = rep(mesh)
    elif isinstance(mapper, tuple):
        mapper = shard(mesh, mapper[1])
    mem = mem or ttnn.DRAM_MEMORY_CONFIG
    if cache is not None:
        return ttnn.as_tensor(
            t.contiguous(), dtype=dtype, layout=layout, device=mesh, memory_config=mem, cache_file_name=cache, **_kw(mapper)
        )
    return ttnn.from_torch(t.contiguous(), dtype=dtype, layout=layout, device=mesh, memory_config=mem, **_kw(mapper))


def host(t, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, mesh=None):
    """A host-side (unplaced) mesh tensor, replicated: for copy_host_to_device_tensor into persistent inputs."""
    return ttnn.from_torch(t.contiguous(), dtype=dtype, layout=layout, **_kw(rep(mesh) if mesh is not None else None))


def to_torch0(t) -> torch.Tensor:
    """Device 0's copy of a (replicated) tensor."""
    return ttnn.to_torch(ttnn.get_device_tensors(t)[0])


def to_torch_cat(t, dim) -> torch.Tensor:
    """Concatenate every die's shard along dim (sharded tensors)."""
    return torch.cat([ttnn.to_torch(x) for x in ttnn.get_device_tensors(t)], dim=dim)


def ckc(mesh, fidelity=ttnn.MathFidelity.HiFi4, fp32=True, approx=False, l1_acc=False):
    return ttnn.init_device_compute_kernel_config(
        mesh.arch(), math_fidelity=fidelity, math_approx_mode=approx, fp32_dest_acc_en=fp32, packer_l1_acc=l1_acc
    )


class AllReduce:
    """Sum of a per-die partial over the (1, N) mesh, result replicated on every die.

    HY4_ALLREDUCE=composite (default): ttnn.all_reduce (fabric). Kept behind one class so the collective can be
    swapped (fused reduce-scatter + all-gather, or the CCL handles of the serving framework) without touching modules.
    """

    def __init__(self, mesh):
        self.mesh = mesh
        self.n = num_devices(mesh)
        self.topology = ttnn.Topology.Linear
        self.mode = os.environ.get("HY4_ALLREDUCE", "composite")

    def __call__(self, x, mem=None):
        if self.n == 1:
            return x
        mem = mem or ttnn.DRAM_MEMORY_CONFIG
        return ttnn.all_reduce(x, cluster_axis=1, topology=self.topology, memory_config=mem)


class AllGather:
    def __init__(self, mesh):
        self.mesh = mesh
        self.n = num_devices(mesh)

    def __call__(self, x, dim, mem=None):
        if self.n == 1:
            return x
        return ttnn.all_gather(
            x, dim=dim, cluster_axis=1, topology=ttnn.Topology.Linear, memory_config=mem or ttnn.DRAM_MEMORY_CONFIG
        )


def rms_scale_fp32(x_list, eps, n_total, mem=None):
    """rsqrt(mean over all of x_list's last dims of x^2 + eps) in fp32: [.., T, 1]. x_list tensors are fp32."""
    acc = None
    for x in x_list:
        s = ttnn.sum(ttnn.multiply(x, x, memory_config=mem), dim=-1, keepdim=True, memory_config=mem)
        acc = s if acc is None else ttnn.add(acc, s, memory_config=mem)
    m = ttnn.multiply(acc, 1.0 / float(n_total), memory_config=mem)
    return ttnn.rsqrt(ttnn.add(m, eps, memory_config=mem), memory_config=mem)


class RMSNorm:
    """HF HYV4FlashRMSNorm: y = w * bf16(x_f32 * rsqrt(mean(x^2) + eps)). Replicated weight, bf16 in/out."""

    def __init__(self, mesh, weight: torch.Tensor, eps: float):
        self.mesh, self.eps = mesh, float(eps)
        self.w = upload(mesh, weight.to(torch.bfloat16).reshape(1, 1, 1, -1), ttnn.bfloat16)
        self.ckc = ckc(mesh)
        W = int(weight.numel())
        self.W = W
        self.dec = None
        wt = W // TILE
        if os.environ.get("HY4_NORM_SHARDED", "1") == "1" and W % TILE == 0 and W >= 1024:
            # decode rows: width-sharded multi-core norm (the interleaved op runs one core per tile row = 1 core)
            nc = next(c for c in (16, 8, 4, 2, 1) if wt % c == 0)
            grid = (nc, 1) if nc <= 8 else (8, nc // 8)
            self.dec_mem = ttnn.create_sharded_memory_config((TILE, W // nc), ttnn.CoreGrid(x=grid[0], y=grid[1]),
                                                             ttnn.ShardStrategy.WIDTH, ttnn.ShardOrientation.ROW_MAJOR,
                                                             use_height_and_width_as_shard_shape=True)
            bw = wt // nc
            self.dec_pc = ttnn.LayerNormShardedMultiCoreProgramConfig(
                compute_with_storage_grid_size=grid, subblock_w=next(b for b in (4, 3, 2, 1) if bw % b == 0),
                block_h=1, block_w=bw, inplace=False)
            self.w_rm = upload(mesh, weight.to(torch.bfloat16).reshape(1, 1, wt, TILE), ttnn.bfloat16,
                               ttnn.ROW_MAJOR_LAYOUT)
            self.dec = True

    def __call__(self, x, mem=None):
        if self.dec is not None and int(x.shape[-2]) == TILE and int(x.shape[-1]) == self.W and x.dtype == ttnn.bfloat16:
            xs = ttnn.to_memory_config(x, self.dec_mem)
            y = ttnn.rms_norm(xs, epsilon=self.eps, weight=self.w_rm, program_config=self.dec_pc,
                              compute_kernel_config=self.ckc, memory_config=self.dec_mem)
            ttnn.deallocate(xs)
            out = ttnn.sharded_to_interleaved(y, mem or ttnn.L1_MEMORY_CONFIG)
            ttnn.deallocate(y)
            return out
        return ttnn.rms_norm(x, epsilon=self.eps, weight=self.w, compute_kernel_config=self.ckc, memory_config=mem)



def stream_enabled(mesh) -> bool:
    """HY4_STREAM (default 1): decode linears on per-bank streamed weight copies (StreamLinear); prefill unchanged."""
    return os.environ.get("HY4_STREAM", "1") == "1" and num_devices(mesh) > 1


class StreamLinear:
    """A decode linear [1, 1, 32, K] x [K, N] on a decode-only copy of the weight streamed per DRAM bank (qwen36
    decode_linear: 6 reader cores per bank, column-major packets; ~2-3x the interleaved matmul at one tile row, LM head
    ~495 GB/s). bf16 weights give bit-identical results to the interleaved matmul (FP32 DST); all 32 rows are computed
    (the padded rows of a pass stay defined).

    parts: one torch [K, N] per die (die d's weight: replicated parts and its column shards concatenated, i.e. several
    linears of the same input fused into one). Prefill keeps the interleaved weights (the copies cost DRAM)."""

    _OPS = {}

    def __init__(self, mesh, parts, dtype, fidelity=None, cache=None, readers=6, out_dtype=None):
        """fidelity (default by weight dtype, HY4_STREAM_FID_BFP8 overrides bfp8): bf16 HiFi4, bfp8 HiFi2 (the weight's
        7 mantissa bits in full; at HiFi4 the bfp8 streams were compute bound: o_proj 77 vs 32 us), bfp4 LoFi."""
        from models.demos.blackhole.qwen36.tt.decode_linear.ops import DecodeLinear

        if fidelity is None:
            fids = {"HiFi4": ttnn.MathFidelity.HiFi4, "HiFi2": ttnn.MathFidelity.HiFi2, "LoFi": ttnn.MathFidelity.LoFi}
            fidelity = {ttnn.bfloat16: ttnn.MathFidelity.HiFi4,
                        ttnn.bfloat8_b: fids[os.environ.get("HY4_STREAM_FID_BFP8", "HiFi2")],
                        ttnn.bfloat4_b: ttnn.MathFidelity.LoFi}[dtype]

        K, N = int(parts[0].shape[-2]), int(parts[0].shape[-1])
        key = (id(mesh), K, N, dtype, fidelity, readers)
        op = StreamLinear._OPS.get(key)
        if op is None:
            op = StreamLinear._OPS[key] = DecodeLinear(mesh, K, N, dtype, readers=readers, fidelity=fidelity)
        self.op, self.K, self.N = op, K, N
        self.out_dtype = out_dtype or ttnn.bfloat16
        cname = f"{cache}.stream{readers}" if cache else None
        self.w = op.weight([p.reshape(K, N).to(torch.bfloat16) for p in parts], cache_file_name=cname)

    def __call__(self, x, mem=None):
        R = DECODE_ROWS[0]
        if R <= 16 and os.environ.get("HY4_STREAM_REAL_ROWS", "1") == "1":
            # read only the pass's real rows of the activation (64 B per tile per row instead of the whole 2 KB tile into
            # every reader core: the compute waited ~28 us for it); the output keeps 32 rows, padded rows zeroed
            return self.op(x, self.w, R, mem or ttnn.L1_MEMORY_CONFIG, out_dtype=self.out_dtype, out_rows=32)
        return self.op(x, self.w, 32, mem or ttnn.L1_MEMORY_CONFIG, out_dtype=self.out_dtype)


DECODE_ROWS = [32]  # the real rows (0 .. R-1) of the decode pass in flight (set by Hy4Model.step / the drafter)


class real_rows:
    """with real_rows(R): the decode linears read rows 0 .. R-1 only."""

    def __init__(self, R):
        self.R = int(R)

    def __enter__(self):
        self.prev = DECODE_ROWS[0]
        DECODE_ROWS[0] = self.R

    def __exit__(self, *a):
        DECODE_ROWS[0] = self.prev


def col_split(mesh, cols, out_mem=None):
    """Slices [c0, c1) of the last dim of a fused StreamLinear output: [(c0, c1), ...] -> callable(y) -> list."""
    def f(y):
        return [ttnn.slice(y, (0, 0, 0, a), (1, 1, int(y.shape[-2]), b), memory_config=out_mem) for a, b in cols]
    return f


class Boundary:
    """Decode layer boundary: all-reduce of the per-die partial + the sublayer's closing RMSNorm as ONE fabric op
    (qwen36 fabric_boundary, after gpt-oss's DecodeBoundary: packed partial rows multicast over the fabric, summed in
    slot order on every die, FP32 norm statistics) instead of reduce-scatter + all-gather + a sharded norm (~36 us ->
    ~8 us). Hy4 has no residual add at this point (the hyper-connection post follows), so the residual is a persistent
    zero vector. rows <= 8 (the real rows of the pass); returns x [1, 1, rows, H] L1 (rows' normed values).
    HY4_FABRIC_BOUNDARY=1 (default)."""

    _B = {}

    def __init__(self, mesh, hidden, eps):
        from models.demos.blackhole.qwen36.tt.fabric_boundary.ops import FabricBoundary

        key = (id(mesh), hidden)
        st = Boundary._B.get(key)
        if st is None:
            # the receive buffers are sharded L1 (rows x ring x H bf16 per site) and an L1 address range is reserved
            # on every core: 8 rows (2 x 192 KB + the zero vectors) crowded the streamed linears' circular buffers off
            # L1. Size for the plain decode row (HY4_BOUNDARY_ROWS, default 1); wider passes take the all-reduce.
            fb = FabricBoundary(mesh, hidden, eps, max_rows=max(Boundary.row_set()))
            zero = {}
            for r in sorted(Boundary.row_set()):
                pages = r * fb.tiles
                zero[r] = ttnn.from_torch(torch.zeros(fb._flat_shape(pages)), dtype=ttnn.bfloat16,
                                          layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=fb._flat_config(pages),
                                          mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
            st = Boundary._B[key] = (fb, zero)
        self.fb, self.zero = st
        self.gammas = {}

    @staticmethod
    def enabled(mesh, rows=1):
        return (os.environ.get("HY4_FABRIC_BOUNDARY", "1") == "1" and num_devices(mesh) > 1
                and rows in Boundary.row_set())

    @staticmethod
    def row_set():
        """The row counts the boundary serves (HY4_BOUNDARY_ROWS, comma separated; default 1): e.g. 1,6 for plain
        decode and DFly2 verify rows. The receive buffers are sized for the largest."""
        rows = os.environ.get("HY4_BOUNDARY_ROWS")
        if rows is None:  # plain decode + the DFly2 verify pass (k + 1 rows; 6 rows do not fit the boundary's CBs)
            k = int(os.environ.get("HY4_SPEC_K", "4"))
            rows = "1" if not 0 < k < 5 else f"1,{k + 1}"
        return {int(v) for v in rows.split(",")}

    def __call__(self, part, rows, norm, site):
        """part [1, 1, 32, H] L1 BF16 per-die partial (rows 0..rows-1 real) -> normed [1, 1, rows, H]."""
        g = self.gammas.get(id(norm))
        if g is None:
            g = self.gammas[id(norm)] = self.fb.gamma(norm.w)
        p = part if int(part.shape[-2]) == rows else ttnn.slice(part, (0, 0, 0, 0), (1, 1, rows, int(part.shape[-1])),
                                                                 memory_config=ttnn.L1_MEMORY_CONFIG)
        res, x = self.fb(self.zero[rows], p, g, site=site)
        ttnn.deallocate(res)
        if p is not part:
            ttnn.deallocate(p)
        return x
