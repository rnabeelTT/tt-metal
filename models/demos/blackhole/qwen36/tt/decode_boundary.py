# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Replicated-residual decode boundaries for the qwen36 TP layer (args.decode_replicated, fusion step 3).

The default TP decode contract keeps the residual fractured along the hidden dim: each norm all-gathers it and each
mixer / MLP ends with a reduce-scatter (4 collectives per layer plus layout copies). Here the decode residual is
replicated on every device in one width-sharded L1 layout (`residual_memory_config`), so a boundary is

    residual' = residual + all_reduce(partial)          (ttnn.experimental.all_reduce_async, BFP8 partial in)
    normed    = RMSNorm(residual')                      (local, sharded program on the same cores)

and the mixer / MLP return their per-device partial sums (no reduce-scatter). 2 collectives per layer, no gathers.
Measured on the (1, 4) mesh (fusion/tools/mbench_ccl.py): reduce-scatter + all-gather ~25-51 us per boundary vs
9.0 us for the all-reduce on 16 cores (13.8 us on the 32-core norm layout, hence 16 cores here).

One persistent scratch buffer and global semaphore per site ("mixer", "mlp"), shared by every layer: the two sites
alternate and each all-reduce is an all-device synchronization (the gpt-oss decode all-reduce relies on the same).
"""

from __future__ import annotations

import ttnn

CORES = (8, 2)  # residual / norm / all-reduce shard grid (x, y)


def residual_memory_config(hidden, cores=CORES):
    n = cores[0] * cores[1]
    return ttnn.create_sharded_memory_config(
        shape=(ttnn.TILE_SIZE, hidden // n),
        core_grid=ttnn.CoreGrid(x=cores[0], y=cores[1]),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def norm_config(hidden, cores=CORES):
    """norm_config for models.common.rmsnorm.RMSNorm: sharded program in and out on the residual layout."""
    mc = residual_memory_config(hidden, cores)
    block_w = hidden // (cores[0] * cores[1]) // ttnn.TILE_SIZE
    pc = ttnn.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(cores[0], cores[1]),
        subblock_w=max(d for d in range(1, 5) if block_w % d == 0),
        block_h=1,
        block_w=block_w,
        inplace=False,
    )
    return {"sharded_program_config": pc, "sharded_output_config": mc, "output_mem_config": mc}


class DecodeBoundary:
    def __init__(self, mesh_device, hidden, topology):
        self.mesh_device = mesh_device
        self.hidden = hidden
        self.topology = topology
        self.num_devices = mesh_device.get_num_devices()
        self.mc = residual_memory_config(hidden)
        self.norm_config = norm_config(hidden)
        n = CORES[0] * CORES[1]
        scratch_mc = ttnn.create_sharded_memory_config(
            shape=(ttnn.TILE_SIZE, hidden * self.num_devices // n),
            core_grid=ttnn.CoreGrid(x=CORES[0], y=CORES[1]),
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        grid = mesh_device.compute_with_storage_grid_size()
        sem_cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
        self.sites = {
            site: (
                ttnn.empty([1, 1, ttnn.TILE_SIZE, hidden * self.num_devices], ttnn.bfloat8_b, ttnn.TILE_LAYOUT,
                           mesh_device, scratch_mc),
                ttnn.create_global_semaphore(mesh_device, sem_cores, 0),
            )
            for site in ("mixer", "mlp")
        }

    def gather(self, x_frac, tt_ccl):
        """The fractured [1, 1, B, dim / tp] (the embedding rows, the per-layer test contract) -> the replicated residual."""
        return ttnn.experimental.all_gather_async(
            x_frac,
            persistent_output_buffer=None,
            dim=3,
            multi_device_global_semaphore=tt_ccl.get_and_cycle_ag_semaphore_handles(),
            num_links=1,
            topology=self.topology,
            memory_config=self.mc,
            barrier_semaphore=tt_ccl.get_and_cycle_barrier_semaphore_handle(),
            chunks_per_sync=10,
            num_workers_per_link=2,
            num_buffers_per_channel=2,
        )

    def fracture(self, x, memory_config=ttnn.DRAM_MEMORY_CONFIG):
        """The replicated residual -> this device's [1, 1, B, dim / tp] slice (the fractured contract)."""
        frac = ttnn.mesh_partition(x, dim=3, cluster_axis=1)
        out = ttnn.to_memory_config(frac, memory_config)
        if out is not frac:
            ttnn.deallocate(frac)
        return out

    def to_residual(self, x):
        """Any replicated [1, 1, B, H] tensor -> the residual layout (BF16)."""
        return ttnn.to_memory_config(x, self.mc, dtype=ttnn.bfloat16)

    def to_partial(self, x):
        """A [1, 1, B, H] partial sum -> the all-reduce input layout (BFP8, residual shards)."""
        return ttnn.to_memory_config(x, self.mc, dtype=ttnn.bfloat8_b)

    def reduce_add(self, partial, residual, site):
        """residual + sum over devices of partial ([1, 1, B, H] per-device partial sums); frees partial."""
        p = self.to_partial(partial)
        if p.buffer_address() != partial.buffer_address():  # already in the layout: no copy was made
            ttnn.deallocate(partial)
        scratch, sem = self.sites[site]
        red = ttnn.experimental.all_reduce_async(
            p,
            scratch,
            cluster_axis=1,
            mesh_device=self.mesh_device,
            multi_device_global_semaphore=sem,
            memory_config=self.mc,
            dtype=ttnn.bfloat16,
            topology=self.topology,
            num_links=1,
        )
        ttnn.deallocate(p)
        out = ttnn.add(residual, red, memory_config=self.mc)
        ttnn.deallocate(red)
        return out


def decode_boundary(args, mesh_device):
    """The model-wide DecodeBoundary (one per args object)."""
    b = args.__dict__.get("_decode_boundary")
    if b is None:
        b = DecodeBoundary(mesh_device, args.dim, args.ccl_topology())
        args._decode_boundary = b
    return b
