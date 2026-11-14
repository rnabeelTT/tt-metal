# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Where does the device model drift from the reference? Runs the full model (hybrid overlay when HYB_ENABLE=1) in
prefill over the gold text (ref_inkling.py REF_GOLD=1: the residual entering every layer for sequence 0) and, after
every layer, compares the device residual with the reference's:

  cum   PCC of the device residual vs the gold residual (cumulative: the device ran on its own inputs)
  step  PCC of this layer's contribution, device (out - in, on the device's own input) vs gold (gold out - gold in)

A layer whose step PCC is low while its input was still close injects error; a smooth cum decline is compounding.

usage (in the image, hybrid env as tools/eval_hybrid.sh): TOOL=drift.py tools/eval_hybrid.sh     env: DRIFT_T (960)
"""

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


def pcc_rows(a, b):
    a = a.double() - a.double().mean(-1, keepdim=True)
    b = b.double() - b.double().mean(-1, keepdim=True)
    return (a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1) + 1e-30)


def main():
    gold = torch.load("/cache/ref/held/gold.pt")
    T = int(os.environ.get("DRIFT_T", "960")) // 64 * 64
    ids = gold["ids"][:T]
    xs = [x[:T].float() for x in gold["x_in"]]  # residual entering layer L
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), l1_small_size=24576, trace_region_size=64_000_000)
    try:
        nl = os.environ.get("DRIFT_LAYERS")  # a reduced model: layers 0 .. DRIFT_LAYERS-1 (all experts fit on device)
        args, model = create_model(mesh, max_seq_len=4096, layer_indices=list(range(int(nl))) if nl else None)
        model.reset_sequence_state()
        pt_host = torch.arange(model.layers[0].attention.max_blocks, dtype=torch.int32)
        rep = ttnn.ReplicateTensorToMesh(mesh)
        comp = ttnn.ConcatMeshToTensor(mesh, dim=-1)
        tok = ttnn.from_torch(ids.to(torch.int32).reshape(1, T), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT,
                              device=mesh, mesh_mapper=rep)
        e = model.embd(tok)  # hidden-sharded across the dies (the fractured residual)
        x = ttnn.reshape(e, (1, 1, T, e.shape[-1]))
        prev = ttnn.to_torch(x, mesh_composer=comp).float().reshape(T, -1)
        c0 = pcc_rows(prev, xs[0])
        print(f"embed: cum pcc mean {c0.mean():.5f} worst {c0.min():.5f}", flush=True)
        n = len(model.layers)
        for L, layer in enumerate(model.layers):
            x = layer.forward_prefill(x, 0, pt_host)
            cur = ttnn.to_torch(x, mesh_composer=comp).float().reshape(T, -1)
            if L + 1 < n:
                gout = xs[L + 1]
                cum = pcc_rows(cur, gout)
                step = pcc_rows(cur - prev, gout - xs[L])
                kind = ("S" if args.is_sliding_layer(L) else "G") + ("d" if L < 2 else "m")
                print(f"layer {L:2d} [{kind}] cum mean {cum.mean():.5f} p01 {cum.quantile(0.01):.5f} | "
                      f"step mean {step.mean():.5f} p01 {step.quantile(0.01):.5f}", flush=True)
            prev = cur
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()
