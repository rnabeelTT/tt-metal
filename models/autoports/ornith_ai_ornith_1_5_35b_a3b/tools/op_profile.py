#!/usr/bin/env python3
"""Per-OP-NAME cost for one decode step, via ttnn graph capture.

The device profiler records per-dispatch timings with no module or op identity, and
ablation inside a captured trace deadlocks. Graph capture carries the op names, so this
is the one view that says which ttnn op to attack rather than which module.

There is no source attribution here: graph capture on this build records only `name` in
each node's params (node keys: arguments, connections, counter, duration_ns,
input_tensors, node_type, params, stacking_level), and neither
ttnn.graph.enable_python_stack_traces() nor CONFIG.enable_graph_python_stack_traces
populates a traceback -- so which Python line emitted a program has to be read off the
ORDERED sequence (--sequence) against the code, not looked up.

Build with --spec gdn to isolate the Gated DeltaNet layers, which the layer sweep showed
are 74.6% of decode (905 us/layer x 30 against 764 us x 10 for full attention).
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib

import torch

import ttnn
from models.autoports.ornith_ai_ornith_1_5_35b_a3b.tt.model import OrnithModel
from models.autoports.ornith_ai_ornith_1_5_35b_a3b.tt.model_config import OrnithModelArgs
from models.autoports.ornith_ai_ornith_1_5_35b_a3b.tt.weight_mapping import load_ornith_state_dict
from models.demos.blackhole.qwen36.tt.model_config import GDN_CONV1D_L1_SMALL_SIZE

MESH_SHAPES = {"P150": (1, 1), "P300": (1, 2), "P150x4": (1, 4)}
FULL_ATTN = lambda i: (i % 4) == 3


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", default="P150", choices=sorted(MESH_SHAPES))
    ap.add_argument("--weights", required=True)
    ap.add_argument("--spec", default="gdn", choices=("mixed", "gdn", "full"))
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument(
        "--sequence", action="store_true", help="print the ordered DEVICE-op sequence instead of the histogram"
    )
    ap.add_argument(
        "--slow-dispatch",
        action="store_true",
        help="disable enable_fast_runtime_mode so capture records each sub-graph",
    )
    a = ap.parse_args()

    if a.spec == "mixed":
        idx = list(range(a.n))
    elif a.spec == "gdn":
        idx = [i for i in range(40) if not FULL_ATTN(i)][: a.n]
    else:
        idx = [i for i in range(40) if FULL_ATTN(i)][: a.n]

    rows, cols = MESH_SHAPES[a.mesh]
    mesh = ttnn.open_mesh_device(
        ttnn.MeshShape(rows, cols),
        l1_small_size=GDN_CONV1D_L1_SMALL_SIZE,
        num_command_queues=2,
        trace_region_size=1024 * 1024 * 1024,
    )
    weights = pathlib.Path(a.weights)
    cfg = json.loads((weights / "config.json").read_text())["text_config"]
    args = OrnithModelArgs(mesh, max_batch_size=1, max_seq_len=256)
    args.layer_indices = idx
    cache = pathlib.Path(args.weight_cache_path(args.routed_expert_dtype))
    model = OrnithModel(mesh, args, load_ornith_state_dict(weights, cfg), tensor_cache_path=cache)

    BLOCK = 64
    num_blocks = ((max(64, (256 + BLOCK - 1) // BLOCK) + 7) // 8) * 8
    model.allocate_kv_caches([num_blocks, args.n_local_kv_heads, BLOCK, args.head_dim], ttnn.bfloat16, batch_size=1)
    model.prefill(torch.randint(1000, 40000, (1, 64)))
    ttnn.synchronize_device(mesh)

    tok = torch.full((1, 1), 1234, dtype=torch.long)
    pos = lambda p: torch.full((1,), p, dtype=torch.long)
    model.decode(tok, pos(64))  # warm the program cache
    ttnn.synchronize_device(mesh)

    ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
    model.decode(tok, pos(65))
    ttnn.synchronize_device(mesh)
    captured = ttnn.graph.end_graph_capture()

    # extract_operation_durations() keys a dict by op name, so repeated calls overwrite
    # each other and 30 layers collapse to one sample. Walk the raw graph instead and sum.
    per_op = collections.Counter()
    counts = collections.Counter()
    kinds = collections.Counter()
    for node in captured:
        kinds[node.get("node_type", "?")] += 1
        if node.get("node_type") == "function_end":
            name = node.get("params", {}).get("name", "unknown")
            per_op[name] += float(node.get("duration_ns", 0)) / 1e3  # -> us
            counts[name] += 1

    if not per_op:
        print(f"no function_end nodes; node types seen: {dict(kinds)}")
        ttnn.close_mesh_device(mesh)
        return 1

    if a.sequence:
        # Device ops only: the *DeviceOperation entries are the ones that cost a program
        # launch (~8.4 us each in the traced model, independent of tensor size). Python
        # wrappers and view-only reshapes are free and would only hide the pattern.
        seq = [
            n.get("params", {}).get("name", "?")
            for n in captured
            if n.get("node_type") == "function_end" and n.get("params", {}).get("name", "").endswith("DeviceOperation")
        ]
        print(
            f"\n=== ordered device-op sequence, {a.spec} x{len(idx)} layers "
            f"({len(seq)} ops, {len(seq)/len(idx):.0f}/layer) ==="
        )
        for i, n in enumerate(seq):
            print(f"{i:4d}  {n}")
        ttnn.close_mesh_device(mesh)
        return 0

    total = sum(per_op.values())
    nl = len(idx)
    print(f"\n=== op profile: {a.spec} x{nl} layers, {a.mesh} ===")
    print(f"{'op':<42} {'us/step':>9} {'calls':>6} {'us/call':>8} {'/layer':>8}  share")
    for name, us in per_op.most_common(a.top):
        print(
            f"{name[:42]:<42} {us:9.1f} {counts[name]:6d} {us/counts[name]:8.2f} " f"{us/nl:8.2f} {us/total*100:5.1f}%"
        )
    print(f"{'TOTAL':<42} {total:9.1f} {sum(counts.values()):6d} {'':8} {total/nl:8.2f}")

    ttnn.close_mesh_device(mesh)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
