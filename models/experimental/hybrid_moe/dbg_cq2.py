# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Does a second command queue cost worker cores on this mesh? Grid with 1 vs 2 CQs (usage: python dbg_cq2.py)."""
import ttnn

for ncq in (1, 2):
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    m = ttnn.open_mesh_device(
        ttnn.MeshShape(1, 4), l1_small_size=24576, trace_region_size=10_000_000, num_command_queues=ncq
    )
    print(f"CQS {ncq}: compute grid {m.compute_with_storage_grid_size()}", flush=True)
    ttnn.close_mesh_device(m)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
