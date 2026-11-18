# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device time per op of one prefill chunk (eager), per layer, named: graph-capture op names + device profiler
durations (die 0), as tools/layer_prof.py does for decode.

    TT_METAL_DEVICE_PROFILER=1 python -m models.autoports.tencent_hy4_flash.tools.prefill_prof --layers 0,1,2 --T 2048
"""

from __future__ import annotations

import argparse
import collections

import torch
import ttnn

from models.autoports.tencent_hy4_flash.tools.layer_prof import graph_ops, programs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="0,1,2")
    ap.add_argument("--show", default="1")
    ap.add_argument("--T", type=int, default=2048)
    ap.add_argument("--start", type=int, default=0, help="a first chunk of this many tokens runs before (unprofiled)")
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--no-engram", action="store_true")
    args = ap.parse_args()
    from models.autoports.tencent_hy4_flash.tools.generate import build, open_mesh

    mesh = open_mesh()
    names = []
    try:
        model = build(mesh, args)
        cfg = model.cfg
        ids = torch.randint(1000, 100000, (args.start + args.T,))
        for _ in range(2):  # compile + warm
            model.reset()
            if args.start:
                model.forward(ids[: args.start])
            model.forward(ids[args.start :])
        model.reset()
        if args.start:
            model.forward(ids[: args.start])
        ttnn.synchronize_device(mesh)
        ttnn.ReadDeviceProfiler(mesh)
        # the profiled chunk: model.forward's body with a graph capture per layer
        T, p0 = args.T, model.pos
        Tp = -(-T // 32) * 32
        tok = torch.zeros(1, Tp, dtype=torch.int32)
        tok[0, :T] = ids[args.start :].to(torch.int32)
        from models.autoports.tencent_hy4_flash.tt.common import upload

        ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
        tok_t = upload(mesh, tok, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        x = ttnn.reshape(ttnn.embedding(tok_t, model.emb, layout=ttnn.TILE_LAYOUT), (1, 1, Tp, cfg.hidden))
        X = ttnn.repeat(x, ttnn.Shape((1, cfg.hc_mult, 1, 1)))
        names.append(("embed", graph_ops(ttnn.graph.end_graph_capture())))
        pos = torch.arange(p0, p0 + Tp)
        cos_sin = {k: model.rope.tables(pos, cfg.rope_theta[k]) for k in ("sparse", "swa")}
        eng = model._engram_rows(ids[args.start :], p0, Tp)
        for layer in model.layers:
            ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
            X = layer.forward(X, p0, T, cos_sin, eng.get(layer.li), None, eng.get("keep"))
            names.append((f"layer {layer.li}", graph_ops(ttnn.graph.end_graph_capture())))
            ttnn.synchronize_device(mesh)
            ttnn.ReadDeviceProfiler(mesh)
    finally:
        ttnn.close_mesh_device(mesh)
    progs = programs()
    n = sum(len(o) for _, o in names)
    tail = progs[-n:]
    show = [f"layer {s}" for s in args.show.split(",")]
    kinds, counts, o, totals = collections.defaultdict(float), collections.defaultdict(int), 0, []
    for label, ops in names:
        seg = tail[o : o + len(ops)]
        o += len(ops)
        totals.append((label, len(ops), sum(d for _, d in seg)))
        if label in show:
            print(f"== {label}: {len(ops)} ops, {totals[-1][2]:.0f} us")
            for name, (_, d) in zip(ops, seg):
                print(f"   {d:9.1f} us  {name}")
        for name, (_, d) in zip(ops, seg):
            kinds[name.split(" > ")[0]] += d
            counts[name.split(" > ")[0]] += 1
    print("== per block:", " | ".join(f"{l}: {c} ops {t / 1e3:.1f} ms" for l, c, t in totals))
    print("== by python op:")
    for k, v in sorted(kinds.items(), key=lambda kv: -kv[1])[:30]:
        print(f"   {v / 1e3:9.2f} ms  {counts[k]:5d}x  {k}")


if __name__ == "__main__":
    main()
