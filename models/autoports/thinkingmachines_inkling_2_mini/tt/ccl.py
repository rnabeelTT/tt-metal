# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Collectives for the replicated-residual TP layout on a 1-D mesh (1, N): the stock ttnn.all_reduce over the mesh
axis (every die gets the full sum). Not models/tt_transformers' tt_all_reduce: on a mesh with a size-1 dim that one
only reduce-scatters (the fractured-residual layout of the other ports) and returns its input unreduced for
cluster_axis=1."""

import ttnn


def all_reduce(x, mesh, tt_ccl=None, memory_config=ttnn.DRAM_MEMORY_CONFIG):
    if mesh.get_num_devices() == 1:
        return x
    return ttnn.all_reduce(x, cluster_axis=1, topology=ttnn.Topology.Linear, memory_config=memory_config)
