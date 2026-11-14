# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device test: the chunked prefill path (tt/layer.forward_prefill: one attention launch per chunk, the MoE prefill
path) equals the rows path (groups of 8, validated against HF), on real weights of a reduced model (layers 0, 2, 5:
dense/sliding, MoE/sliding, MoE/global). Compares the final hidden state of every row for: one chunk of T rows and
chunks of 128 (cross-chunk conv states and KV). Also times both.

usage (in the image): tools/run_dev.sh python models/autoports/thinkingmachines_inkling_small/tests/test_prefill_device.py [T]
"""

import json
import sys
import time

import torch

import ttnn
from models.autoports.thinkingmachines_inkling_small.tt.model import create_model

LAYERS = [0, 2, 5]


def pcc(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def main():
    T = int(sys.argv[1]) if len(sys.argv) > 1 else 640
    texts = "/opt/tt-metal/models/experimental/hybrid_moe/runs/inkling/texts_held.json"
    ids = torch.tensor(json.load(open(texts))["code_python_file"][:T])
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), l1_small_size=24576, trace_region_size=64_000_000)
    ok = False
    try:
        args, model = create_model(mesh, max_seq_len=4096, layer_indices=LAYERS)
        rep = ttnn.ReplicateTensorToMesh(mesh)
        nblk = model.layers[0].attention.max_blocks
        pt_host = torch.arange(nblk, dtype=torch.int32)
        pt = ttnn.from_torch(pt_host.reshape(1, -1), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh,
                             mesh_mapper=rep)
        comp = ttnn.ConcatMeshToTensor(mesh, dim=-1)

        def embed(tok_ids):
            n = len(tok_ids)
            tt = ttnn.from_torch(tok_ids.to(torch.int32).reshape(1, n), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT,
                                 device=mesh, mesh_mapper=rep)
            x = model.embd(tt)
            return ttnn.reshape(x, (1, 1, n, x.shape[-1]))

        # rows path (run twice: the first compiles)
        for g in range(0, 16, 8):
            x = embed(ids[g : g + 8])
            pos = ttnn.from_torch(torch.arange(g, g + 8, dtype=torch.int32), dtype=ttnn.int32,
                                  layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, mesh_mapper=rep)
            for layer in model.layers:
                x = layer.forward(x, mode="rows", position_tensor=pos, page_table=pt)
        model.reset_sequence_state()
        t0 = time.time()
        rows_out = []
        for g in range(0, T, 8):
            x = embed(ids[g : g + 8])
            pos = ttnn.from_torch(torch.arange(g, g + 8, dtype=torch.int32), dtype=ttnn.int32,
                                  layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, mesh_mapper=rep)
            for layer in model.layers:
                x = layer.forward(x, mode="rows", position_tensor=pos, page_table=pt)
            rows_out.append(ttnn.to_torch(x, mesh_composer=comp).float().reshape(8, -1))
        t_rows = time.time() - t0
        ref = torch.cat(rows_out)

        def prefill(chunk):
            model.reset_sequence_state()
            outs = []
            t0 = time.time()
            for c0 in range(0, T, chunk):
                x = embed(ids[c0 : c0 + chunk])
                for layer in model.layers:
                    x = layer.forward_prefill(x, c0, pt_host)
                outs.append(ttnn.to_torch(x, mesh_composer=comp).float().reshape(chunk, -1))
            return torch.cat(outs), time.time() - t0

        res = []
        for chunk in (T, 128):
            prefill(chunk)  # warm-up: JIT compile of this chunk size's programs
            out, dt = prefill(chunk)
            pr = [pcc(out[t], ref[t]) for t in range(T)]
            print(f"  rows below 0.999: {sum(p < 0.999 for p in pr)} of {T}")
            res.append((chunk, min(pr), sum(pr) / T, dt))
            print(f"prefill chunk {chunk}: per-row pcc vs rows worst {min(pr):.5f} mean {sum(pr) / T:.5f}; "
                  f"{dt:.2f}s vs rows {t_rows:.2f}s ({T} tokens, {len(LAYERS)} layers)", flush=True)
        ok = all(w > 0.999 for _, w, _, _ in res)
        print("OK" if ok else "FAIL")
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
