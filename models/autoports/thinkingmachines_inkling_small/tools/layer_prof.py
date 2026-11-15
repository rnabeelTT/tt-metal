# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device time per op of one verify pass (R rows, eager) through two layers, named: the ops come from a ttnn graph
capture, their device durations from the device profiler (TT_METAL_DEVICE_PROFILER=1; die 0, per program: first
FW start to last FW end over all cores). Prints each layer's ops with us and a total per op kind.

usage (hybrid env as tools/eval_hybrid.sh, TOOL=layer_prof.py, EXTRA="-e TT_METAL_DEVICE_PROFILER=1 ..."):
       layer_prof.py [R=4] [layers=10,11]
"""

import collections
import csv
import json
import os
import sys

import torch

if os.environ.get("HYB_ENABLE") == "1":
    sys.path.insert(0, os.environ.get("HYB_POC_DIR", "/opt/tt-metal/models/experimental/hybrid_moe"))
    os.environ.setdefault("HYB_PORT", "thinkingmachines_inkling_small")
    import hybrid

    hybrid.install()

import ttnn  # noqa: E402
from models.autoports.thinkingmachines_inkling_small.tt.model import create_model  # noqa: E402

CSV = os.path.join(os.environ.get("TT_METAL_PROFILER_DIR", "/opt/tt-metal/generated/profiler"), ".logs", "profile_log_device.csv")


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
    """The device profiler's CSV is complete only after the mesh closes: the third pass captures the op graph of every
    layer, and after the close its programs (the CSV's last ones on die 0) are split by each layer's op count."""
    R = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    layers = [int(v) for v in (sys.argv[2] if len(sys.argv) > 2 else "10,11").split(",")]
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), l1_small_size=24576, trace_region_size=64_000_000)
    counts = []
    try:
        args, model = create_model(mesh, max_seq_len=8192)
        rep = ttnn.ReplicateTensorToMesh(mesh)
        nblk = model.layers[0].attention.max_blocks
        pt = ttnn.from_torch(torch.arange(nblk, dtype=torch.int32).reshape(1, -1), dtype=ttnn.int32,
                             layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, mesh_mapper=rep)
        ids = [100 + i for i in range(R)]
        tok = ttnn.from_torch(torch.tensor(ids, dtype=torch.int32).reshape(1, R), dtype=ttnn.uint32,
                              layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, mesh_mapper=rep)
        pos = ttnn.from_torch(torch.arange(256, 256 + R, dtype=torch.int32), dtype=ttnn.int32,
                              layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, mesh_mapper=rep)
        idx = torch.full((R, R), -1, dtype=torch.int32)
        idx[torch.arange(R), torch.arange(R)] = torch.arange(256, 256 + R, dtype=torch.int32)
        kvi = ttnn.from_torch(idx, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, mesh_mapper=rep)
        x0 = ttnn.reshape(model.embd(tok), (1, 1, R, -1))

        def run(capture):
            x = ttnn.clone(x0)
            for i, layer in enumerate(model.layers):
                if capture:
                    ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
                try:
                    x = layer.forward_rows(x, pos, pt, commit_all=False, kv_idx=kvi)
                finally:
                    if capture:
                        counts.append(graph_ops(ttnn.graph.end_graph_capture()))
                if i % 6 == 5:  # drain the device buffers before they fill (they hold a few thousand zones per core)
                    ttnn.synchronize_device(mesh)
                    ttnn.ReadDeviceProfiler(mesh)
            ttnn.synchronize_device(mesh)
            ttnn.ReadDeviceProfiler(mesh)

        model.reset_sequence_state()
        run(False)
        run(False)
        run(True)
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
    progs = programs(0)
    n = sum(len(c) for c in counts)
    print(f"{len(progs)} programs on die 0; the profiled pass has {n} graph device ops")
    tail = progs[-n:]
    kinds = collections.defaultdict(float)
    per_layer = []
    o = 0
    for L, ops in enumerate(counts):
        seg = tail[o : o + len(ops)]
        o += len(ops)
        per_layer.append(sum(d for _, d in seg))
        if L in layers:
            print(f"== layer {L}: {len(ops)} ops, {per_layer[-1]:.0f} us device")
            for name, (_, d) in zip(ops, seg):
                print(f"   {d:8.1f} us  {name}")
        for name, (_, d) in zip(ops, seg):
            kinds[name.split(" > ")[0]] += d
    print("== device us per layer:", " ".join(f"{v:.0f}" for v in per_layer), f"| total {sum(per_layer):.0f}")
    print("== by top-level op (all layers):")
    for k, v in sorted(kinds.items(), key=lambda kv: -kv[1])[:30]:
        print(f"   {v:9.1f} us  {k}")


if __name__ == "__main__":
    main()
