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
    """Concatenate every die's shard along dim (sharded tensors). One composer read (same values): 8 logits rows fp32
    0.87 -> 0.31 ms, 1 row 0.15 -> 0.07 ms vs per-die to_torch + cat (2026-11-20)."""
    dev = t.device() if t.storage_type() == ttnn.StorageType.DEVICE else None
    if dev is not None and os.environ.get("HY4_D2H_COMPOSER", "1") == "1" and dev.get_num_devices() > 1:
        return ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(dev, dim=dim))
    return torch.cat([ttnn.to_torch(x) for x in ttnn.get_device_tensors(t)], dim=dim)


def ckc(mesh, fidelity=ttnn.MathFidelity.HiFi4, fp32=True, approx=False, l1_acc=False):
    return ttnn.init_device_compute_kernel_config(
        mesh.arch(), math_fidelity=fidelity, math_approx_mode=approx, fp32_dest_acc_en=fp32, packer_l1_acc=l1_acc
    )


_WFID = {}


def wfid(base, w):
    """The compute config of a matmul against weight w: HiFi2 for a bfp8 weight (HY4_BFP8_HIFI2, default 1; covers
    the weight's 7 mantissa bits, as the decode streams), else base."""
    if w.dtype != ttnn.bfloat8_b or os.environ.get("HY4_BFP8_HIFI2", "1") != "1":
        return base
    c = _WFID.get(id(base))
    if c is None:
        c = _WFID[id(base)] = (base, ttnn.init_device_compute_kernel_config(
            ttnn.get_device_tensors(w)[0].device().arch(), math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False))
    return c[1]


_PF_CFG = {}


def _largest_div(n, cap):
    return max(d for d in range(1, cap + 1) if n % d == 0)


_W_TILE = {ttnn.bfloat16: 2048, ttnn.bfloat8_b: 1088, ttnn.bfloat4_b: 576}
MM_L1_BUDGET = 1_440_000  # CB bytes per core of the 2D-multicast factory (bench 2026-11-23: 1.436 MB ran, 1.52 failed)
_L1_ROOM = {}


def l1_cb_room(device, refresh: bool = False) -> int:
    """Bytes per core a program's static CBs can take: from the L1 allocator base (where the CB region starts) up to the
    lowest allocated L1 buffer (L1 buffers are placed top-down). The served engine holds L1 residents after warm-up
    (fabric / decode buffers, trace intermediates) that the engine-only benches never had: a CB plan sized for the empty
    L1 failed there (`Statically allocated dataflow buffers ... clash with L1 buffers`, 2026-11-23). Cached per device;
    Hy4Model.forward refreshes it once per prefill chunk."""
    k = id(device)
    v = _L1_ROOM.get(k)
    if v is None or refresh:
        # the block table's addresses are offsets from the allocator base (an L1 buffer at 1507328 lists at 1394752 over
        # the base 112576 on Blackhole): the lowest allocated offset is the room itself
        mv = ttnn.get_memory_view(device, ttnn.BufferType.L1)
        v = _L1_ROOM[k] = min([int(b["address"]) for b in mv.block_table if b.get("allocated") == "yes"],
                              default=int(mv.total_bytes_per_bank))
    return v


def _full_grid_cfg(device, M, K, N, wdt, budget=MM_L1_BUDGET):
    """HY4_PF_MCAST_FULL (default 1, FP32 DST only): the whole worker grid (11 x 10 on Blackhole: per-core M / N rounded
    up, the last row / column of cores partly idle) and the largest K block (<= 16 tiles) whose CBs fit L1, instead of
    the <= 8 x 11 divisor grid with K blocks of <= 8. With FP32 DST the K sum runs tile by tile in the same order
    whatever the blocking (partials spill and reload in fp32): bit-identical outputs (mm_bench, every prefill
    projection: 1.2-1.8x faster)."""
    g = device.compute_with_storage_grid_size()
    Mt, Kt, Nt = M // TILE, K // TILE, N // TILE
    gy, gx = min(g.y, Mt), min(g.x, Nt)
    pm, pn = -(-Mt // gy), -(-Nt // gx)
    gy, gx = -(-Mt // pm), -(-Nt // pn)
    wt = _W_TILE.get(wdt)
    if wt is None or gx * gy < 32:
        return False
    sw = _largest_div(pn, 4)
    sh = _largest_div(pm, 4 // sw)
    for ib in sorted((d for d in range(1, 17) if Kt % d == 0), reverse=True):
        l1 = 2 * pm * ib * 2048 + 2 * pn * ib * wt + pm * pn * (2048 + 4096)
        if l1 <= budget:
            return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=(gx, gy), in0_block_w=ib, out_subblock_h=sh, out_subblock_w=sw,
                per_core_M=pm, per_core_N=pn, transpose_mcast=False, fused_activation=None, fuse_batch=True)
    return False


def mm(x, w, ck, mem=None, **kw):
    """ttnn.matmul for a prefill-sized [1, 1, M, K] x [1, 1, K, N] (M >= HY4_PF_MCAST_MIN rows, default 256): an explicit
    2D-multicast program config (<= 8 x 11 cores, K blocks of up to 8 tiles, FP32 DST subblocks of <= 4 tiles): the
    default config ran these at ~40 TFLOPS (3072 -> 4096 bfp8 at 2048 rows: 1197 -> 532 us). Else the plain matmul."""
    M, K, N = int(x.shape[-2]), int(x.shape[-1]), int(w.shape[-1])
    ok = (os.environ.get("HY4_PF_MCAST", "1") == "1" and M >= int(os.environ.get("HY4_PF_MCAST_MIN", "256"))
          and len(x.shape) == 4 and int(x.shape[0]) == 1 and int(x.shape[1]) == 1 and int(w.shape[1]) == 1
          and M % TILE == 0 and K % TILE == 0 and N % TILE == 0)
    if ok:
        fp32 = bool(getattr(ck, "fp32_dest_acc_en", True))
        full = (os.environ.get("HY4_PF_MCAST_FULL", "1") == "1" and fp32
                and not bool(getattr(ck, "packer_l1_acc", False)))
        # the CBs must fit below the L1 buffers allocated now (l1_cb_room): the budget is part of the key
        budget = min(MM_L1_BUDGET, l1_cb_room(x.device()))
        key = (M, K, N, fp32, full, w.dtype, budget)
        cfg = _PF_CFG.get(key, False)
        if cfg is False and full:
            cfg = _full_grid_cfg(x.device(), M, K, N, w.dtype, budget)
            _PF_CFG[key] = cfg
        if cfg is False:
            Mt, Kt, Nt = M // TILE, K // TILE, N // TILE
            gy, gx = _largest_div(Mt, 8), _largest_div(Nt, 11)
            pm, pn = Mt // gy, Nt // gx
            cfg = None
            ib = _largest_div(Kt, 8)
            wt = _W_TILE.get(w.dtype, 2048)
            cbs = 2 * pm * ib * 2048 + 2 * pn * ib * wt + pm * pn * (2048 + (4096 if fp32 else 0))
            if gy >= 4 and gx >= 4 and pm <= 8 and pn <= 16 and cbs <= budget:
                lim = 4 if fp32 else 8
                sw = _largest_div(pn, lim)
                sh = _largest_div(pm, lim // sw)
                cfg = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                    compute_with_storage_grid_size=(gx, gy), in0_block_w=ib, out_subblock_h=sh,
                    out_subblock_w=sw, per_core_M=pm, per_core_N=pn, transpose_mcast=False, fused_activation=None,
                    fuse_batch=True)
            _PF_CFG[key] = cfg
        if cfg is not None:
            kw["program_config"] = cfg
    return ttnn.matmul(x, w, compute_kernel_config=ck, memory_config=mem, **kw)


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

    def __call__(self, x, mem=None, shard_out=False):
        """shard_out: return the decode rows' output width-sharded (self.dec_mem; consumers that read it by page, e.g.
        the streamed linears) instead of converting it back to interleaved. An input already in self.dec_mem (e.g. the
        fused HC's in_mem=) skips the conversion program."""
        if self.dec is not None and int(x.shape[-2]) == TILE and int(x.shape[-1]) == self.W and x.dtype == ttnn.bfloat16:
            pre = x.memory_config() == self.dec_mem
            xs = x if pre else ttnn.to_memory_config(x, self.dec_mem)
            y = ttnn.rms_norm(xs, epsilon=self.eps, weight=self.w_rm, program_config=self.dec_pc,
                              compute_kernel_config=self.ckc, memory_config=self.dec_mem)
            if not pre:
                ttnn.deallocate(xs)
            if shard_out:
                return y
            out = ttnn.sharded_to_interleaved(y, mem or ttnn.L1_MEMORY_CONFIG)
            ttnn.deallocate(y)
            return out
        return ttnn.rms_norm(x, epsilon=self.eps, weight=self.w, compute_kernel_config=self.ckc, memory_config=mem)



def stream_enabled(mesh) -> bool:
    """HY4_STREAM (default 1): decode linears on per-bank streamed weight copies (StreamLinear); prefill unchanged."""
    return os.environ.get("HY4_STREAM", "1") == "1" and num_devices(mesh) > 1


def _xy(crs):
    return {(c.x, c.y) for c in ttnn.corerange_to_cores(crs, row_wise=True)}


PF_PENDING = {}  # id(StreamLinear) -> DecodeLinear: the weight columns an earlier program prefetched (HY4_PF)


def pf_enabled() -> bool:
    """HY4_PF=1 (default 0): the fused HC prefetches the next streamed linears' weights (the attention's xa, the MoE's
    dec_in_s (+ router with HY4_PAIR)) into their reader cores' L1 while it runs (cross-op weight prefetch; same values).
    Measured a net loss (hy4-perf round 5): the HC is latency bound and slows more under the prefetch's DRAM / NOC load
    (31 -> 52 us) than the prefetched linear gains (29 -> 19 us: x read + compute bound). Kept for a host that tolerates
    the load."""
    return os.environ.get("HY4_PF", "0") == "1"


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
        self.readers, self.dtype, self.fidelity, self.mesh = readers, dtype, fidelity, mesh
        self.out_dtype = out_dtype or ttnn.bfloat16
        cname = f"{cache}.stream{readers}" if cache else None
        self.w = op.weight([p.reshape(K, N).to(torch.bfloat16) for p in parts], cache_file_name=cname)

    def prefetch_parts(self, cb_index, op):
        """(kernels, cbs) for an earlier program (HY4_PF, e.g. the fused HC) to prefetch this linear's weight columns
        (as many per core as its CB holds: all of them on a pf_plan placement) into the reader cores' L1 of `op` (a
        placement off the host's cores); this linear's next program on that op then skips reading them."""
        PF_PENDING[id(self)] = op
        return op.prefetch_parts(self.w, cb_index)

    def parts(self, x, mem=None, splits=None, heads=None, rows_out=32, op=None):
        """(kernels, cbs, io, out) of this linear's program (op: a placement; default the prefetched one when an earlier
        program prefetched this linear (PF_PENDING), else the plain one)."""
        R = DECODE_ROWS[0]
        pend = PF_PENDING.pop(id(self), None) if heads is None else None
        op = op or pend or self._op(heads)
        pf = pend is op
        if splits is not None:
            splits = [(s[0], s[1], s[2] if len(s) > 2 else None, s[3] if len(s) > 3 else None) for s in splits]
        if R <= 16 and os.environ.get("HY4_STREAM_REAL_ROWS", "1") == "1":
            # read only the pass's real rows of the activation (64 B per tile per row instead of the whole 2 KB tile into
            # every reader core: the compute waited ~28 us for it); the output keeps 32 rows, padded rows zeroed
            return op._parts(x, self.w, R, mem or ttnn.L1_MEMORY_CONFIG, out_dtype=self.out_dtype,
                             out_rows=max(rows_out, R), splits=splits, heads=heads, pf=pf)
        return op._parts(x, self.w, 32, mem or ttnn.L1_MEMORY_CONFIG, out_dtype=self.out_dtype, splits=splits,
                         heads=heads, pf=pf)

    def __call__(self, x, mem=None, splits=None, heads=None, rows_out=32):
        """splits [(c0, c1[, rows]), ...]: the output as column parts (one tensor each, rows default 32; HY4_DL_SPLIT).
        rows_out: the output's rows when the pass reads its real rows only (e.g. the real rows for a fabric boundary,
        which then needs no slice; rows past them undefined)."""
        kernels, cbs, io, out = self.parts(x, mem, splits, heads, rows_out)
        _pf_ran(kernels)
        ttnn.generic_op(io, ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))
        return out

    def alt_op(self, avoid, readers=None):
        """This linear's DecodeLinear with `readers` cores per bank (default the build's) placed off the cores `avoid`
        ((x, y) set): prefetched / paired placements (pf_plan). Same weight copy (the per-bank layout does not depend on
        the placement: exact)."""
        from models.demos.blackhole.qwen36.tt.decode_linear.ops import DecodeLinear

        readers = readers or self.readers
        key = (id(self.mesh), self.K, self.N, self.dtype, self.fidelity, readers, tuple(sorted(avoid)))
        op = StreamLinear._OPS.get(key)
        if op is None:
            op = StreamLinear._OPS[key] = DecodeLinear(self.mesh, self.K, self.N, self.dtype, readers=readers,
                                                       fidelity=self.fidelity, exclude=tuple(sorted(avoid)))
        return op

    def _op(self, heads):
        """HY4_DL_READERS (A/B; default: the build's readers): reader cores per bank. The per-bank weight copy does not
        depend on it (each core streams whole columns: exact)."""
        r = int(os.environ.get("HY4_DL_READERS", "0"))
        if not r or r == self.readers or heads is not None:
            return self.op
        from models.demos.blackhole.qwen36.tt.decode_linear.ops import DecodeLinear

        key = (id(self.mesh), self.K, self.N, self.dtype, self.fidelity, r)
        op = StreamLinear._OPS.get(key)
        if op is None:
            op = StreamLinear._OPS[key] = DecodeLinear(self.mesh, self.K, self.N, self.dtype, readers=r,
                                                       fidelity=self.fidelity)
        return op


HC_CORES = {(x, y) for x in range(8) for y in range(2)}  # the fused HC's / sublayer norms' cores (kept off: HY4_PF)
_PLANS = {}


def _pf_ran(kernels):
    """A streamed-linear program ran: prefetches into its cores' L1 (its CB range) are gone."""
    if PF_PENDING:
        used = set()
        for k in kernels:
            used |= _xy(k.core_ranges)
        for key, op in list(PF_PENDING.items()):
            if _xy(op.cores) & used:
                del PF_PENDING[key]


def pf_plan(sls):
    """[(StreamLinear, DecodeLinear)]: placements for prefetched (HY4_PF) / paired (stream_pair) linears, each off the
    HC's cores and the others', the widest first: one column per core where the grid has room (a prefetched linear is
    compute / activation bound: more cores, less time), else fewer readers (the build's, or fewer when a bank's
    neighbourhood runs out of cores)."""
    key = tuple(id(s) for s in sls)
    if key not in _PLANS:
        g = sls[0].mesh.compute_with_storage_grid_size()
        free = g.x * g.y - len(HC_CORES)
        avoid, plan = set(HC_CORES), {}
        order = sorted(range(len(sls)), key=lambda i: -sls[i].op.C)
        for oi, i in enumerate(order):
            sl = sls[i]
            C = sl.op.C
            rest = sum(min(sls[j].op.C, 4) for j in order[oi + 1:])  # room for the later ones (>= 4 readers a bank)
            op = None
            for r in sorted({min(C, r) for r in (C, -(-C // 2), -(-C // 3), sl.readers, -(-C // 4), -(-C // 5))},
                            reverse=True):
                if (r + rest) * sl.op.banks > free - len(avoid) + len(HC_CORES):
                    continue
                try:  # place_along_rows asserts when a bank's neighbourhood has no room left
                    op = sl.alt_op(avoid, readers=r)
                    break
                except AssertionError:
                    continue
            assert op is not None, f"pf_plan: no placement for a [{sl.K}, {sl.N}] linear"
            avoid |= {(c.x, c.y) for c, _, _, _ in op.assign}
            plan[i] = (sl, op)
        _PLANS[key] = [plan[i] for i in range(len(sls))]
    return _PLANS[key]


def stream_pair(a, a_kw, b, b_kw):
    """Two streamed linears (e.g. of the same input) as ONE program on disjoint core groups (HY4_PAIR; pf_plan
    placements): one dispatch and fixed cost instead of two, both weights streaming (or, prefetched, computing) at once.
    a_kw / b_kw: their __call__ kwargs (x, mem, splits, rows_out). -> (out_a, out_b); same values as two calls."""
    (_, op_a), (_, op_b) = pf_plan([a, b])
    ka, ca, ia, oa = a.parts(op=op_a, **a_kw)
    kb, cb, ib, ob = b.parts(op=op_b, **b_kw)
    _pf_ran(ka + kb)
    io = list(ia)
    for t in ib:
        if not any(t is u for u in io):
            io.append(t)
    ttnn.generic_op(io, ttnn.ProgramDescriptor(kernels=ka + kb, semaphores=[], cbs=ca + cb))
    return oa, ob


def pair_enabled() -> bool:
    """HY4_PAIR (default 1): same-input streamed linear pairs as one program (the MoE's dec_in_s + router linear; with
    HY4_PAIR_G (default 1) the attention's xa + output gate g): one dispatch, both weights streaming at once (exact;
    round 5: 1u 21.13 -> 20.90, 8u 36.37 -> 35.82 ms engine)."""
    return os.environ.get("HY4_PAIR", "1") == "1"


def boundary_rows_out(mesh, R) -> int:
    """HY4_BND_ROWS (default 1): a partial sum headed for the fabric boundary is written with the pass's R real rows
    only (the boundary then takes it without a slice program; same values). Else 32."""
    if R is not None and os.environ.get("HY4_BND_ROWS", "1") == "1" and Boundary.enabled(mesh, R):
        return int(R)
    return 32


def norm_shard() -> bool:
    """HY4_NORM_SHARD=1: decode sublayer norms take their input width-sharded from the fused HC and hand their output
    sharded to the streamed linears (no to_memory_config / sharded_to_interleaved programs)."""
    return os.environ.get("HY4_NORM_SHARD", "1") == "1"


def dl_split() -> bool:
    """HY4_DL_SPLIT=1: fused decode linears write their column parts as separate tensors (no slice programs)."""
    return os.environ.get("HY4_DL_SPLIT", "1") == "1"


# batched speculative decoding (tt/engine_batch.py): HY4_BSPEC "u:K,..." = K drafts per verify pass at u live sequences
# (other u: plain batched steps). Every verify row count is exact (tt/engram.py _row_sum: rows past 8 bit-identical to
# the same rows in passes of <= 8; b8 0.8650 / 0.2229 and long3k 0.8263 / 0.3009 at R 12 / 15 / 16 / 32). Default: the
# best K per u by device time (engine, 96 tokens a user, greedy / t0.9 p0.9; tok/s = tokens per step / (draft + verify
# pass + accept + a 3.5 ms host floor), plain: pass + 2 ms; 2026-11-23, MoE r2): u2 K = 3; u3 K = 1 (plain 106 -> 132;
# K2 136 within noise); u4 K = 2 (135 -> 179; K1 170, K3 173-177); u5 K = 2 (160 -> 207 / 204; K1 192); u6-u8 K = 1
# (186 -> 222 / 216, 210 -> 250 / 242, 232 -> 280 / 269; K2 crosses 16 rows: 50-54 ms verify passes, 193-246). The
# verify pass's host part (Engram rows of u (K + 1) rows) is not in that: on a host whose Engram tables page in from
# disk (no RAM copy) it measured 4-99 ms and the spec steps of 5-8 users lost to plain; HY4_BSPEC overrides.
BSPEC_DEFAULT = "2:3,3:1,4:2,5:2,6:1,7:1,8:1"


def bspec_policy(k_max: int = 5) -> dict:
    """{u: K} of HY4_BSPEC (u >= 2, K <= k_max, u (K + 1) <= 32); empty when off ("0") or HY4_SPEC_K=0."""
    if os.environ.get("HY4_SPEC_K", "4") == "0":
        return {}
    out = {}
    for x in os.environ.get("HY4_BSPEC", BSPEC_DEFAULT).split(","):
        if ":" in x:
            u, K = (int(v) for v in x.split(":"))
            K = min(K, int(k_max))
            if u >= 2 and K > 0 and u * (K + 1) <= 32:
                out[u] = K
    return out


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
            # HY4_FB_FIDELITY (default HiFi3): the boundary norm's FPU fidelity. HiFi2 put its error vs FP32 at 1.2e-2,
            # 2x the all-reduce + rms_norm path's; HiFi3 5.4-6.5e-3 (= all-reduce + rms_norm) at rows 1, 2, 4
            fb = FabricBoundary(mesh, hidden, eps, max_rows=max(Boundary.row_set()),
                                math_fidelity=os.environ.get("HY4_FB_FIDELITY", "HiFi3"))
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
        # default 1: e.g. 1,5 (the k = 4 verify width) took the 5-row verify pass 36.72 -> 35.36 ms device, but its
        # receive buffers keep ~330 KB of L1 reserved on every core while serving, and the prefill programs' circular
        # buffers (flash SDPA, the grouped expert matmuls) then clash with them
        return {int(v) for v in os.environ.get("HY4_BOUNDARY_ROWS", "1").split(",")}

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
