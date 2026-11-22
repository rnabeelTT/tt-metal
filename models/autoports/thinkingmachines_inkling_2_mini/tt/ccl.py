# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Collectives for the replicated-residual TP layout on a 1-D mesh (1, N): the stock ttnn.all_reduce over the mesh
axis (every die gets the full sum). Not models/tt_transformers' tt_all_reduce: on a mesh with a size-1 dim that one
only reduce-scatters (the fractured-residual layout of the other ports) and returns its input unreduced for
cluster_axis=1.

Decode rows (<= 32, I2M_AR_ASYNC=1 default): ttnn.experimental.all_reduce_async on persistent width-sharded scratch
(the qwen36 DecodeBoundary op, bf16 partials): 25.9 -> 20.1 us per all-reduce incl. the reshards (tools/bench_ar.py,
4 dies). One scratch buffer + global semaphore per site ("mixer": attention output, "mlp": MoE output); the two sites
strictly alternate, and every all-reduce waits for every device's partial, so a site's buffer is never overwritten
before every device consumed it."""

import os

import ttnn

ASYNC = os.environ.get("I2M_AR_ASYNC", "1") == "1"
# I2M_AR_PACK: <= 8 live rows packed into K = 3 / 9 / 18 tiles around the all-reduce (tt/kernels/arpack.cpp; traced,
# 4 dies: 9.6 vs 20.1 us at 1 row, 12.3 vs 20.1 at 4, 16.1 vs 20.1 at 8, none at 16: tests/test_arpack_device.py)
PACK = os.environ.get("I2M_AR_PACK", "1") == "1"
# I2M_AR_LINKS: fabric links per packed all-reduce (default 2 on 4 dies: B = 8 28.3 -> 28.0 ms, bit-identical; 1 else)
LINKS = int(os.environ.get("I2M_AR_LINKS", "0"))
PACK_SITES = os.environ.get("I2M_AR_PACK_SITES", "mixer,mlp").split(",")
PACK_K = ((1, 3, (3, 1)), (4, 9, (9, 1)), (8, 18, (9, 2)))  # (rows <=, tiles, shard grid)
CORES = (9, 2)  # 18 cores: H 2304 = 72 tiles -> 4 per core
_STATE = {}


def _decode_ar(mesh, H):
    key = (id(mesh), H)
    st = _STATE.get(key)
    if st is None:
        N = mesh.get_num_devices()
        n = CORES[0] * CORES[1]
        mk = lambda w: ttnn.create_sharded_memory_config(  # noqa: E731
            shape=(32, w // n), core_grid=ttnn.CoreGrid(x=CORES[0], y=CORES[1]), strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR, use_height_and_width_as_shard_shape=True)
        grid = mesh.compute_with_storage_grid_size()
        sem_cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
        mc = mk(H)
        st = _STATE[key] = {
            "mc": mc,
            "sites": {s: (ttnn.empty([1, 1, 32, H * N], ttnn.bfloat16, ttnn.TILE_LAYOUT, mesh, mk(H * N)),
                          ttnn.create_global_semaphore(mesh, sem_cores, 0)) for s in ("mixer", "mlp")},
        }
    return st


class _Packed:
    """One packed width: the shard memory config, a scratch + global semaphore per site, the pack / unpack programs."""

    def __init__(self, mesh, K, grid):
        N = mesh.get_num_devices()
        self.mesh, self.K, self.cx, self.cy = mesh, K, grid[0], grid[1]
        n = grid[0] * grid[1]
        self.KJ = K // n
        mk = lambda w: ttnn.create_sharded_memory_config(  # noqa: E731
            shape=(32, w // n), core_grid=ttnn.CoreGrid(x=grid[0], y=grid[1]), strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR, use_height_and_width_as_shard_shape=True)
        g = mesh.compute_with_storage_grid_size()
        sem_cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(g.x - 1, g.y - 1))})
        self.mc = mk(32 * K)
        self.sites = {s: (ttnn.empty([1, 1, 32, 32 * K * N], ttnn.bfloat16, ttnn.TILE_LAYOUT, mesh, mk(32 * K * N)),
                          ttnn.create_global_semaphore(mesh, sem_cores, 0)) for s in ("mixer", "mlp")}
        self.crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid[0] - 1, grid[1] - 1))})

    def _move(self, pack, t, shard, T):
        import hashlib

        from .bconv import KDIR

        Ht = int(t.shape[-1]) // 32
        ct = [T, self.KJ, 2 * T * Ht, self.cx] + ttnn.TensorAccessorArgs(t).get_compile_time_args()
        kd = ttnn.KernelDescriptor(kernel_source=str(KDIR / "arpack.cpp"), core_ranges=self.crs, compile_time_args=ct,
                                   defines=[("ARP_PACK", "1")] if pack else [],
                                   config=ttnn.DataMovementConfigDescriptor(
                                       processor=ttnn.DataMovementProcessor.RISCV_1 if pack else
                                       ttnn.DataMovementProcessor.RISCV_0,
                                       noc=ttnn.NOC.NOC_1 if pack else ttnn.NOC.NOC_0))
        kd.common_runtime_args = [t.buffer_address(), shard.buffer_address()]
        cbs = []
        if pack:  # (the staging tile: tt/kernels/arpack.cpp)
            from .bconv import _cb

            prefix = (512 + 32 * T + 63) // 64 * 64
            cbs = [_cb(self.crs, 0, ttnn.bfloat16, prefix * (self.KJ * 64 // (2 * T) + 2), 1)]
        prog = ttnn.ProgramDescriptor(kernels=[kd], semaphores=[], cbs=cbs)
        key = ("i2m_arpack", pack, T, self.K, Ht, repr(t.memory_config()))
        prog.custom_program_hash = int.from_bytes(hashlib.sha1(repr(key).encode()).digest()[:8], "little") & ((1 << 63) - 1)
        ttnn.generic_op([t, shard], prog)

    def unpack(self, r, T, H):
        out = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, T, H]), ttnn.bfloat16, ttnn.TILE_LAYOUT, self.mesh,
                                             ttnn.L1_MEMORY_CONFIG)
        self._move(False, out, r, T)
        return out

    def __call__(self, x, site, rows=None, packed_out=False):
        T, H = rows or int(x.shape[-2]), int(x.shape[-1])
        p = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, 32, 32 * self.K]), ttnn.bfloat16, ttnn.TILE_LAYOUT,
                                           self.mesh, self.mc)
        self._move(True, x, p, T)
        scratch, sem = self.sites[site]
        r = ttnn.experimental.all_reduce_async(p, scratch, cluster_axis=1, mesh_device=self.mesh,
                                               multi_device_global_semaphore=sem, memory_config=self.mc,
                                               dtype=ttnn.bfloat16, topology=ttnn.Topology.Linear,
                                               num_links=LINKS or (2 if self.mesh.get_num_devices() == 4 else 1))
        ttnn.deallocate(p)
        if packed_out:  # (the consumer reads the packed rows: tt/post.py, tt/pre.py)
            return PackedAR(r, T, H, self)
        out = self.unpack(r, T, H)
        ttnn.deallocate(r)
        return out


class PackedAR:
    """A packed all-reduce result: T rows of [1, 1, T, H] bf16 as chunk c = (tile, row, face) at byte 32 c of t."""

    def __init__(self, t, T, H, op):
        self.t, self.T, self.H, self.op = t, T, H, op
        self.shape = ttnn.Shape([1, 1, T, H])

    def unpack(self):
        return self.op.unpack(self.t, self.T, self.H)


def unpacked(x):
    """A tensor from an all-reduce result (unpacks a PackedAR)."""
    return x.unpack() if isinstance(x, PackedAR) else x


def _packed(mesh, T):
    for rows, K, grid in PACK_K:
        if T <= rows:
            key = (id(mesh), "pack", K)
            if key not in _STATE:
                _STATE[key] = _Packed(mesh, K, grid)
            return _STATE[key]
    return None


def prepare(mesh, H):
    """Allocate the decode all-reduce scratch (before any trace capture)."""
    if ASYNC and mesh.get_num_devices() > 1 and (H // 32) % (CORES[0] * CORES[1]) == 0:
        _decode_ar(mesh, H)
    if PACK and mesh.get_num_devices() > 1:
        for rows, _, _ in PACK_K:
            _packed(mesh, rows)


def packed_rows(T):
    """True when a T-row decode all-reduce takes the packed path."""
    return PACK and T <= PACK_K[-1][0]


def all_reduce(x, mesh, tt_ccl=None, memory_config=ttnn.DRAM_MEMORY_CONFIG, site=None, rows=None, packed_out=False):
    """rows: only the first `rows` rows of x are live (the packed path reads just those; -> [1, 1, rows, H])."""
    if mesh.get_num_devices() == 1:
        return x
    H = int(x.shape[-1])
    if rows is not None and site in PACK_SITES and packed_rows(rows) and x.layout == ttnn.TILE_LAYOUT \
            and x.dtype == ttnn.bfloat16 and not x.is_sharded():
        return _packed(mesh, rows)(x, site, rows, packed_out=packed_out)
    if (PACK and site in PACK_SITES and int(x.shape[-2]) <= PACK_K[-1][0] and len(x.shape) == 4 and int(x.shape[1]) == 1
            and x.dtype == ttnn.bfloat16 and x.layout == ttnn.TILE_LAYOUT and not x.is_sharded()):
        return _packed(mesh, int(x.shape[-2]))(x, site, packed_out=packed_out)
    if (ASYNC and site is not None and int(x.shape[-2]) <= 32 and len(x.shape) == 4 and int(x.shape[1]) == 1
            and (H // 32) % (CORES[0] * CORES[1]) == 0 and x.dtype == ttnn.bfloat16):
        st = _decode_ar(mesh, H)
        scratch, sem = st["sites"][site]
        p = ttnn.to_memory_config(x, st["mc"])
        r = ttnn.experimental.all_reduce_async(p, scratch, cluster_axis=1, mesh_device=mesh,
                                               multi_device_global_semaphore=sem, memory_config=st["mc"],
                                               dtype=ttnn.bfloat16, topology=ttnn.Topology.Linear, num_links=1)
        ttnn.deallocate(p)
        out = ttnn.to_memory_config(r, ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(r)
        return out
    return ttnn.all_reduce(x, cluster_axis=1, topology=ttnn.Topology.Linear, memory_config=memory_config)
