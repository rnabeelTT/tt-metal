# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Where the MTP drafter's time goes: device time per op of one depth-1 draft step (InklingMtp.step + the device argmax),
named from a ttnn graph capture as tools/layer_prof.py does, plus the wall time of that step replayed as a trace.

usage (hybrid env as tools/eval_hybrid.sh, TOOL=draft_prof.py, EXTRA="-e TT_METAL_DEVICE_PROFILER=1 ..."):
       draft_prof.py [reps=20]          without the profiler env only the traced wall time is printed
"""

import collections
import os
import sys
import time

import torch

if os.environ.get("HYB_ENABLE") == "1":
    sys.path.insert(0, os.environ.get("HYB_POC_DIR", "/opt/tt-metal/models/experimental/hybrid_moe"))
    os.environ.setdefault("HYB_PORT", "thinkingmachines_inkling_small")
    import hybrid

    hybrid.install()

import ttnn  # noqa: E402

from models.autoports.thinkingmachines_inkling_small.tt.inkling_mtp import InklingMtp  # noqa: E402
from models.autoports.thinkingmachines_inkling_small.tt.model import create_model  # noqa: E402

# the device-profile helpers of tools/layer_prof.py (importing it would install the hybrid overlay twice)
CSV = os.path.join(
    os.environ.get("TT_METAL_PROFILER_DIR", "/opt/tt-metal/generated/profiler"), ".logs", "profile_log_device.csv"
)


def programs(lines_from):
    """{run host id: duration us} of die 0's programs in CSV lines past lines_from (FW zones over all cores)."""
    t0, t1, order = {}, {}, []
    with open(CSV) as f:
        for i, row in enumerate(f):
            if i < max(lines_from, 2):
                continue
            r = row.split(",")
            if len(r) < 12 or r[0].strip() != "0" or not r[10].strip().endswith("-FW"):
                continue
            rid, t, ph = int(r[7]), int(r[5]), r[11].strip()
            if rid not in t0:
                order.append(rid)
                t0[rid], t1[rid] = t, t
            if ph == "ZONE_START":
                t0[rid] = min(t0[rid], t)
            else:
                t1[rid] = max(t1[rid], t)
    return [(rid, (t1[rid] - t0[rid]) / 1350.0) for rid in order]


def graph_ops(g):
    ops, stack = [], []
    for node in g:
        nt, prm = node.get("node_type"), node.get("params", {})
        if nt == "function_start":
            name = prm.get("name", "")
            stack.append(name)
            if "DeviceOperation" in name or name.endswith("device_operation"):
                ops.append(stack[0] + " > " + name)
        elif nt == "function_end" and stack:
            stack.pop()
    return ops


def main():
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    prof = os.environ.get("TT_METAL_DEVICE_PROFILER") == "1"
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), l1_small_size=24576, trace_region_size=64_000_000)
    ops = []
    try:
        args, model = create_model(mesh, max_seq_len=8192)
        mtp = InklingMtp(model, 2)
        rep = ttnn.ReplicateTensorToMesh(mesh)
        nblk = model.layers[0].attention.max_blocks
        pt = ttnn.from_torch(
            torch.arange(nblk, dtype=torch.int32).reshape(1, -1),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh,
            mesh_mapper=rep,
        )
        tok = ttnn.from_torch(
            torch.tensor([[1000]], dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh,
            mesh_mapper=rep,
        )
        pos = ttnn.from_torch(
            torch.tensor([300], dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh,
            mesh_mapper=rep,
        )
        H = int(args.dim)
        prev = ttnn.from_torch(
            torch.randn(1, 1, 1, H) * 0.5, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep
        )
        V = int(args.unpadded_vocab_size)

        def one():
            lg, full = mtp.step(1, prev, tok, pos, pt)
            if lg.layout != ttnn.ROW_MAJOR_LAYOUT:
                lg = ttnn.to_layout(lg, ttnn.ROW_MAJOR_LAYOUT)
            v = ttnn.slice(lg, (0, 0, 0, 0), (1, 1, 1, V))
            return ttnn.argmax(v, dim=-1), full

        with mtp.own_ccl():
            one(), one()
            ttnn.synchronize_device(mesh)
            if prof:
                ttnn.ReadDeviceProfiler(mesh)
                ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
                try:
                    one()
                finally:
                    ops = graph_ops(ttnn.graph.end_graph_capture())
                ttnn.synchronize_device(mesh)
                ttnn.ReadDeviceProfiler(mesh)
            else:
                tid = ttnn.begin_trace_capture(mesh, cq_id=0)
                out = one()
                ttnn.end_trace_capture(mesh, tid, cq_id=0)
                for _ in range(3):
                    ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
                t0 = time.perf_counter()
                for _ in range(reps):
                    ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
                ttnn.synchronize_device(mesh)
                dt = (time.perf_counter() - t0) / reps
                print(f"== traced depth-1 draft step (incl. argmax): {1e3 * dt:.2f} ms per step ({reps} replays)")
                ttnn.release_trace(mesh, tid)
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
    if not prof:
        return
    progs = programs(0)
    seg = progs[-len(ops) :]
    tot = sum(d for _, d in seg)
    print(f"== depth-1 draft step: {len(ops)} device ops, {tot:.0f} us device (die 0, sum of programs)")
    kinds = collections.defaultdict(float)
    for name, (_, d) in zip(ops, seg):
        print(f"   {d:8.1f} us  {name}")
        kinds[name.split(" > ")[-1]] += d
    print("== by device op:")
    for k, v in sorted(kinds.items(), key=lambda kv: -kv[1])[:20]:
        print(f"   {v:9.1f} us  {k}")


if __name__ == "__main__":
    main()
