# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Count the device ops of one traced decode step per layer kind on the (1,4) mesh, for a MultichipConfig A/B
(stage 7: dispatch is ~1 us per op on the mesh, so the op count is the lever). Uses ttnn graph capture around one
eager decode step (the trace replays exactly these ops). Prints ``OPCOUNT <name> <layer> <n_ops> <json counter>``.

    python -m models.autoports.qwen_qwen3_5_122b_a10b.tools.decode_op_count --layers 0,3 --configs ops_base,ops_all
"""
import argparse
import collections
import json

import torch

import ttnn
from models.autoports.qwen_qwen3_5_122b_a10b.tests.test_functional_decoder import _hidden
from models.autoports.qwen_qwen3_5_122b_a10b.tests.test_multichip_decoder import MESH_SHAPE, _MeshHarness
from models.autoports.qwen_qwen3_5_122b_a10b.tools.sweep_multichip import families
from models.autoports.qwen_qwen3_5_122b_a10b.tt.multichip_decoder import MULTICHIP_SELECTED
from models.demos.blackhole.qwen36.tt.model_config import GDN_CONV1D_L1_SMALL_SIZE


def _device_ops(captured):
    """Device-operation names in call order from a captured graph (the ``ttnn::prim::`` / ``*DeviceOperation``
    nodes; composite python-level wrappers are excluded)."""
    names = ttnn.graph.extract_calltrace(captured)
    ops = [n for n in names if "DeviceOperation" in n or "::prim::" in n]
    return ops


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="0,3")
    ap.add_argument("--configs", default="ops_base,ops_all")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    fam = {}
    for f in families(MULTICHIP_SELECTED).values():
        fam.update(dict(f))
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(
        ttnn.MeshShape(*MESH_SHAPE), l1_small_size=GDN_CONV1D_L1_SMALL_SIZE, trace_region_size=64_000_000
    )
    rows = []
    try:
        for layer_idx in [int(s) for s in a.layers.split(",")]:
            for name in a.configs.split(","):
                mc = fam[name]
                h = _MeshHarness(mesh, layer_idx, batch=1, max_seq_len=4096, real=True, hf=False, mc=mc)
                hidden = _hidden(1, 128, h.H, seed=2000)
                h.tt_prefill(hidden)
                x, kw = h.decode_inputs(_hidden(1, 1, h.H, seed=2002), torch.tensor([128]))
                out = h.tt.decode_forward(x, **kw)  # compile pass (also what the trace capture would record)
                ttnn.deallocate(out)
                ttnn.synchronize_device(mesh)
                ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
                out = h.tt.decode_forward(x, **kw)
                ttnn.synchronize_device(mesh)
                captured = ttnn.graph.end_graph_capture()
                ttnn.deallocate(out)
                ops = _device_ops(captured)
                cnt = collections.Counter(o.split("::")[-1] for o in ops)
                row = {
                    "config": name,
                    "layer": layer_idx,
                    "kind": h.kind,
                    "n_ops": len(ops),
                    "ops": dict(cnt),
                    "seq": ops,
                }
                rows.append(row)
                print(
                    f"OPCOUNT {name} layer{layer_idx} {h.kind} n_ops={len(ops)} {json.dumps(dict(cnt.most_common()))}",
                    flush=True,
                )
                del h
        if a.out:
            with open(a.out, "w") as f:
                json.dump(rows, f, indent=1)
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()
