# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The ordered device-op list of one eager one-row decode step (ttnn graph capture), to name the programs of a device
profile: the device profiler's CSV carries program ids and timings but no op names. Writes the captured graph to
$OPS_OUT (default /cache/op_graph.json) and prints each layer's device-op names in order.

usage (in the image, hybrid env as tools/eval_hybrid.sh, e.g. TOOL=op_list.py): .../tools/op_list.py
"""

import json
import os
import sys

import torch

if os.environ.get("HYB_ENABLE") == "1":
    sys.path.insert(0, os.environ.get("HYB_POC_DIR", "/opt/tt-metal/models/experimental/hybrid_moe"))
    os.environ.setdefault("HYB_PORT", "thinkingmachines_inkling_small")
    os.environ.setdefault("HYB_FUSED", "0")
    import hybrid

    hybrid.install()

import ttnn  # noqa: E402
from models.autoports.thinkingmachines_inkling_small.tt.model import create_model  # noqa: E402

CAPTURE_LAYERS = (2, 10)
PROMPT = [{"role": "user", "content": "Explain in three short paragraphs how a hash map handles collisions."}]


def main():
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), l1_small_size=24576, trace_region_size=64_000_000)
    try:
        args, model = create_model(mesh, max_seq_len=8192)
        tok = args.tokenizer
        rep = ttnn.ReplicateTensorToMesh(mesh)
        nblk = model.layers[0].attention.max_blocks
        pt_host = torch.arange(nblk, dtype=torch.int32)
        pt = ttnn.from_torch(pt_host.reshape(1, -1), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh,
                             mesh_mapper=rep)

        csv = "/opt/tt-metal/generated/profiler/.logs/profile_log_device.csv"

        def flush():
            """With TT_METAL_DEVICE_PROFILER=1: drain the device profiler, -> the CSV's line count (else 0), so a
            layer's programs are the CSV lines between two flushes, in the graph's device-op order."""
            if os.environ.get("TT_METAL_DEVICE_PROFILER") != "1":
                return 0
            ttnn.synchronize_device(mesh)
            ttnn.ReadDeviceProfiler(mesh)
            return sum(1 for _ in open(csv)) if os.path.exists(csv) else 0

        def rows(ids, p0, capture=False):
            n = len(ids)
            x = model.embd(ttnn.from_torch(torch.tensor(ids, dtype=torch.int32).reshape(1, n), dtype=ttnn.uint32,
                                           layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, mesh_mapper=rep))
            x = ttnn.reshape(x, (1, 1, n, x.shape[-1]))
            pos = ttnn.from_torch(torch.arange(p0, p0 + n, dtype=torch.int32), dtype=ttnn.int32,
                                  layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, mesh_mapper=rep)
            marks = []
            for i, layer in enumerate(model.layers):
                if capture and i in CAPTURE_LAYERS:
                    a = flush()
                    ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
                    try:
                        x = layer.forward_rows(x, pos, pt, commit_all=False)
                    finally:
                        graphs[i] = ttnn.graph.end_graph_capture()
                    lines[i] = (a, flush())
                else:
                    x = layer.forward_rows(x, pos, pt, commit_all=False)
            lg = model._lm_head(model._final_norm_decode(x))
            ttnn.to_torch(ttnn.get_device_tensors(lg)[0])
            return marks

        graphs, lines = {}, {}
        model.reset_sequence_state()
        prompt = tok.encode(tok.apply_chat_template(PROMPT, tokenize=False, add_generation_prompt=True,
                                                    reasoning_effort="none"), add_special_tokens=False)
        pre = (len(prompt) - 1) // 64 * 64
        if pre:
            model.prefill_tokens(torch.tensor(prompt[:pre]), pt_host, logits_rows="last")
        p = pre
        rows([prompt[p]], p)  # warm (compiles)
        flush()
        rows([prompt[p]], p, capture=True)
        json.dump(lines, open(os.environ.get("OPS_LINES", "/cache/op_lines.json"), "w"))
        print("profiler CSV line ranges per captured layer:", lines)
        out = os.environ.get("OPS_OUT", "/cache/op_graph.json")
        json.dump({str(k): v for k, v in graphs.items()}, open(out, "w"))
        # device ops in order, with the enclosing top-level ttnn op and layer
        stack, seq = [], []
        for layer, graph in sorted(graphs.items()):
          for node in graph:
            nt, prm = node.get("node_type"), node.get("params", {})
            if nt == "function_start":
                name = prm.get("name", "")
                stack.append(name)
                if "DeviceOperation" in name or name.endswith("device_operation"):
                    seq.append((layer, stack[:2] + [name]))
            elif nt == "function_end" and stack:
                stack.pop()
        print(f"device ops: {len(seq)}")
        for L in (2, 10):
            ops = [s for l, s in seq if l == L]
            print(f"layer {L}: {len(ops)} device ops")
            for s in ops:
                print("   ", " > ".join(s))
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()
