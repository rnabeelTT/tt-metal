# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""DRAM per die after building the model with SLOTS x CTX (bfp8 experts, bf16 KV) and one decoder + trace per width:
prints the allocator statistics (die 0). env: MESH (1), SLOTS (8), CTX (8192)"""

import os

import ttnn
from models.autoports.thinkingmachines_inkling_2_mini.tt.model import Inkling2Model


def stats(mesh, tag):
    d = mesh.get_devices()[0] if hasattr(mesh, "get_devices") else mesh
    try:
        s = ttnn.get_memory_view(d, ttnn.BufferType.DRAM)
        print(f"{tag}: DRAM allocated {s.total_bytes_allocated_per_bank * s.num_banks / 2**30:.2f} GiB, free "
              f"{s.total_bytes_free_per_bank * s.num_banks / 2**30:.2f} GiB, largest free block "
              f"{s.largest_contiguous_bytes_free_per_bank * s.num_banks / 2**30:.2f} GiB", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"{tag}: memory view unavailable ({e})", flush=True)


def main():
    N = int(os.environ.get("MESH", "1"))
    if N > 1:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, N), trace_region_size=300 << 20)
    stats(mesh, "empty")
    S, C = int(os.environ.get("SLOTS", "8")), int(os.environ.get("CTX", "8192"))
    m = Inkling2Model(mesh, max_seq_len=C + 64, expert_dtype=ttnn.bfloat8_b, kv_dtype=ttnn.bfloat16, prefill_chunk=256,
                      slots=S)
    stats(mesh, f"model {S} slots x {C}")
    ttnn.close_mesh_device(mesh)
    print("== probe done")


if __name__ == "__main__":
    main()
