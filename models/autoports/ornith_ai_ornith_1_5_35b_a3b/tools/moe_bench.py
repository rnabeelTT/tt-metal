#!/usr/bin/env python3
"""Op count + traced cost for ONE MoE block, without loading the model.

The full-model loop is ~5 minutes per measurement, which is too slow to iterate on op
count. One block with random weights reproduces the decode graph exactly (same shapes,
same dtypes, same program configs) and runs in well under a minute, so this is the loop
for cutting ops in the expert path; the full bench only has to confirm the result.
"""
from __future__ import annotations

import argparse
import collections
import time
import types

import torch

import ttnn
from models.autoports.ornith_ai_ornith_1_5_35b_a3b.tt.moe import OrnithMoE


def fake_args(mesh, experts=256, k=8, dim=2048, inter=512):
    a = types.SimpleNamespace()
    a.num_experts, a.num_experts_per_tok = experts, k
    a.dim, a.moe_intermediate_size = dim, inter
    a.shared_expert_intermediate_size = inter
    a.routed_expert_dtype = "bfloat4_b"
    a.mesh_devices = mesh.get_num_devices()
    a.num_devices = 1
    a.device_name = "P150"
    a.mesh_device = mesh
    return a


def state_for(a):
    E, H, I = a.num_experts, a.dim, a.moe_intermediate_size
    sd = {
        "experts.gate_up_proj": torch.randn(E, 2 * I, H, dtype=torch.bfloat16) * 0.02,
        "experts.down_proj": torch.randn(E, H, I, dtype=torch.bfloat16) * 0.02,
        "gate.weight": torch.randn(E, H, dtype=torch.bfloat16) * 0.02,
        "shared_expert.gate_proj.weight": torch.randn(I, H, dtype=torch.bfloat16) * 0.02,
        "shared_expert.up_proj.weight": torch.randn(I, H, dtype=torch.bfloat16) * 0.02,
        "shared_expert.down_proj.weight": torch.randn(H, I, dtype=torch.bfloat16) * 0.02,
        "shared_expert_gate.weight": torch.randn(1, H, dtype=torch.bfloat16) * 0.02,
    }
    return sd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--show", type=int, default=0, help="print the first N device ops in order")
    a = ap.parse_args()

    mesh = ttnn.open_mesh_device(
        ttnn.MeshShape(1, 1), l1_small_size=24576, num_command_queues=1, trace_region_size=512 * 1024 * 1024
    )
    args = fake_args(mesh)
    moe = OrnithMoE(mesh, args, state_for(args), layer_id=0)

    x = ttnn.from_torch(
        torch.randn(1, 1, 1, args.dim, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh
    )
    moe.forward(x, mode="decode")
    ttnn.synchronize_device(mesh)

    ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
    out = moe.forward(x, mode="decode")
    ttnn.synchronize_device(mesh)
    g = ttnn.graph.end_graph_capture()
    ops = [
        n.get("params", {}).get("name", "?")
        for n in g
        if n.get("node_type") == "function_end" and n.get("params", {}).get("name", "").endswith("DeviceOperation")
    ]
    ttnn.deallocate(out)

    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
    for _ in range(a.iters):
        o = moe.forward(x, mode="decode")
    ttnn.end_trace_capture(mesh, tid, cq_id=0)
    ttnn.synchronize_device(mesh)
    IT = 20
    t0 = time.perf_counter()
    for _ in range(IT):
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh)
    us = (time.perf_counter() - t0) / IT / a.iters * 1e6

    print(f"MOE decode: {len(ops)} device ops, {us:.1f} us/call, " f"x40 layers = {us * 40 / 1000:.2f} ms")
    for n, c in collections.Counter(o.replace("DeviceOperation", "") for o in ops).most_common():
        print(f"   {c:3d}  {n}")
    if a.show:
        print("   order: " + " ".join(o.replace("DeviceOperation", "") for o in ops[: a.show]))
    ttnn.release_trace(mesh, tid)
    ttnn.close_mesh_device(mesh)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
