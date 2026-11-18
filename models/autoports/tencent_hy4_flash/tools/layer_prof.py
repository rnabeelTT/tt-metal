# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device time per op of one decode step (the trace-safe step path, eager), per layer, named (the Inkling tool's
method: op names from a ttnn graph capture, durations from the device profiler, die 0, first FW start to last FW end
over all cores of each program).

    TT_METAL_DEVICE_PROFILER=1 python -m models.autoports.tencent_hy4_flash.tools.layer_prof [--layers 0,1,2,3] [--show 1,2]
"""

from __future__ import annotations

import argparse
import collections
import os

import torch
import ttnn

CSV = os.path.join(os.environ.get("TT_METAL_PROFILER_DIR", os.path.join(os.environ.get("TT_METAL_HOME", "."),
                                                                         "generated", "profiler")),
                   ".logs", "profile_log_device.csv")


def programs():
    t0, t1, order = {}, {}, []
    with open(CSV) as f:
        for i, row in enumerate(f):
            if i < 2:
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="0,1,2,3")
    ap.add_argument("--show", default="1,2")
    ap.add_argument("--prompt-len", type=int, default=100)
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--no-engram", action="store_true")
    ap.add_argument("--trace", action="store_true", help="profile a trace replay of the whole step (op names from a "
                    "graph capture of the eager pass)")
    args = ap.parse_args()
    from models.autoports.tencent_hy4_flash.tools.generate import build, open_mesh
    from models.autoports.tencent_hy4_flash.tt.decode import Hy4Decoder

    show = [int(v) for v in args.show.split(",")]
    mesh = open_mesh()
    names = []
    try:
        model = build(mesh, args)
        ids = torch.randint(1000, 100000, (args.prompt_len + 4,))
        model.reset()
        model.forward(ids[: args.prompt_len])
        dec = Hy4Decoder(model, trace=False)
        dec.start(args.prompt_len)
        for i in range(2):
            dec.run([int(ids[args.prompt_len + i])])
        ttnn.synchronize_device(mesh)
        ttnn.ReadDeviceProfiler(mesh)
        if args.trace:
            h, sparse, bucket = dec._host_inputs([int(ids[args.prompt_len + 2])], dec.pos)
            dec._copy_inputs(h)
            dec.si.sparse, dec.si.bucket = sparse, bucket
            ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
            lg = model.step(dec.si, 1)
            names.append(("step", graph_ops(ttnn.graph.end_graph_capture())))
            ttnn.deallocate(lg)
            tid = ttnn.begin_trace_capture(mesh, cq_id=0)
            lg = model.step(dec.si, 1)
            ttnn.end_trace_capture(mesh, tid, cq_id=0)
            for _ in range(2):
                ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
            ttnn.synchronize_device(mesh)
            ttnn.ReadDeviceProfiler(mesh)
            args.show = "step"
            show = []
        # the profiled step: the same body as model.step, captured per layer
        h, sparse, bucket = dec._host_inputs([int(ids[args.prompt_len + 2])], dec.pos)
        dec._copy_inputs(h)
        dec.si.sparse, dec.si.bucket = sparse, bucket
        si, cfg = dec.si, model.cfg
        if args.trace:
            raise SystemExit(0)
        ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
        x = ttnn.embedding(si.tok, model.emb, layout=ttnn.TILE_LAYOUT)
        x = ttnn.reshape(x, (1, 1, 32, cfg.hidden))
        X = ttnn.repeat(x, ttnn.Shape((1, cfg.hc_mult, 1, 1)))
        names.append(("embed", graph_ops(ttnn.graph.end_graph_capture())))
        for layer in model.layers:
            ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
            X = layer.step(X, si, 1)
            names.append((f"layer {layer.li}", graph_ops(ttnn.graph.end_graph_capture())))
            ttnn.synchronize_device(mesh)
            ttnn.ReadDeviceProfiler(mesh)
        ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
        hh, _ = model.head_hc.pre(X)
        lg = ttnn.matmul(model.norm(hh), model.lm, dtype=ttnn.float32, compute_kernel_config=model.ckc)
        names.append(("head", graph_ops(ttnn.graph.end_graph_capture())))
        ttnn.synchronize_device(mesh)
        ttnn.ReadDeviceProfiler(mesh)
    except SystemExit:
        pass
    finally:
        ttnn.close_mesh_device(mesh)
    progs = programs()
    n = sum(len(o) for _, o in names)
    tail = progs[-n:]
    print(f"{len(progs)} programs on die 0; the profiled step has {n} device ops")
    kinds = collections.defaultdict(float)
    counts = collections.defaultdict(int)
    o = 0
    totals = []
    for label, ops in names:
        seg = tail[o : o + len(ops)]
        o += len(ops)
        tot = sum(d for _, d in seg)
        totals.append((label, len(ops), tot))
        if label in [f"layer {L}" for L in show] or (label == "step" and os.environ.get("HY4_PROF_LIST") == "1"):
            print(f"== {label}: {len(ops)} ops, {tot:.0f} us")
            for name, (_, d) in zip(ops, seg):
                print(f"   {d:8.1f} us  {name}")
        for name, (_, d) in zip(ops, seg):
            k = name.split(" > ")[-1]
            kinds[k] += d
            counts[k] += 1
    print("== per block:", " | ".join(f"{l}: {c} ops {t:.0f} us" for l, c, t in totals))
    print("== by device op:")
    for k, v in sorted(kinds.items(), key=lambda kv: -kv[1])[:30]:
        print(f"   {v:9.1f} us  {counts[k]:5d}x  {k}")


if __name__ == "__main__":
    main()
