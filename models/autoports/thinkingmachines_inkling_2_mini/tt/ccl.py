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


def prepare(mesh, H):
    """Allocate the decode all-reduce scratch (before any trace capture)."""
    if ASYNC and mesh.get_num_devices() > 1 and (H // 32) % (CORES[0] * CORES[1]) == 0:
        _decode_ar(mesh, H)


def all_reduce(x, mesh, tt_ccl=None, memory_config=ttnn.DRAM_MEMORY_CONFIG, site=None):
    if mesh.get_num_devices() == 1:
        return x
    H = int(x.shape[-1])
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
