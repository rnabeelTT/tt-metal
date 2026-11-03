# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Cost of the hybrid host step's small transfers on the (1,4) mesh: readback of one die's tensor, host->device copy
into a persistent replicated tensor, and packed variants. Prints ms per call."""
import json
import time

import torch
import ttnn

from capture_routing import MESH_PARAMS


def t(fn, n=50):
    fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return round((time.perf_counter() - t0) / n * 1e3, 3)


def main():
    ttnn.set_fabric_config(MESH_PARAMS["fabric_config"])
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), l1_small_size=MESH_PARAMS["l1_small_size"],
                                 trace_region_size=MESH_PARAMS["trace_region_size"])
    rep = ttnn.ReplicateTensorToMesh(mesh)
    dram = ttnn.DRAM_MEMORY_CONFIG
    res = {}
    for name, shape, tdt, dt, lay in [
        ("ids_u16_rm", [1, 1, 32, 10], torch.int32, ttnn.uint16, ttnn.ROW_MAJOR_LAYOUT),
        ("sc_bf16_tile", [1, 1, 32, 10], torch.bfloat16, ttnn.bfloat16, ttnn.TILE_LAYOUT),
        ("x_bf16_tile", [1, 1, 32, 4096], torch.bfloat16, ttnn.bfloat16, ttnn.TILE_LAYOUT),
        ("x_bf16_rm1", [1, 1, 1, 4096], torch.bfloat16, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT),
        ("packed_f32_rm", [1, 1, 1, 4096 + 64], torch.float32, ttnn.float32, ttnn.ROW_MAJOR_LAYOUT),
    ]:
        h = torch.zeros(shape, dtype=tdt)
        d = ttnn.from_torch(h, dtype=dt, layout=lay, device=mesh, mesh_mapper=rep, memory_config=dram)
        d0 = ttnn.get_device_tensors(d)[0]
        r = {}
        r["get_device_tensors"] = t(lambda: ttnn.get_device_tensors(d)[0])
        r["to_torch_die0"] = t(lambda: ttnn.to_torch(ttnn.get_device_tensors(d)[0]))
        r["to_torch_die0_cached"] = t(lambda: ttnn.to_torch(d0))
        r["from_device_die0"] = t(lambda: ttnn.from_device(d0))
        hh = ttnn.from_torch(h, dtype=dt, layout=lay, mesh_mapper=rep)
        r["from_torch_host"] = t(lambda: ttnn.from_torch(h, dtype=dt, layout=lay, mesh_mapper=rep))
        r["copy_h2d_rep"] = t(lambda: ttnn.copy_host_to_device_tensor(hh, d))
        r["copy_h2d_rep_sync"] = t(lambda: (ttnn.copy_host_to_device_tensor(hh, d), ttnn.synchronize_device(mesh)))
        r["sync_idle"] = t(lambda: ttnn.synchronize_device(mesh))
        res[name] = r
        print(name, json.dumps(r), flush=True)
    ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
