# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device test of one Inkling decoder layer (tt/layer.py via tt/model.create_model with layer_indices=[L]) on the real
weights, one-row decode token by token from position 0, against the HF fixture of that layer
(models/experimental/hybrid_moe/fixtures_inkling.py: x_in -> x_out). Reports the PCC of the output and, stricter, of
the layer's contribution (out - x_in vs x_out - x_in: the residual dominates the plain output PCC).

usage (in the image): tools/run_dev.sh python models/autoports/thinkingmachines_inkling_small/tests/test_layer_device.py \
        /cache/fixtures/layer0.pt 0 [T]
   or with the reference's gold residuals (ref_inkling.py REF_GOLD=1):  .../test_layer_device.py /cache/ref_held/gold.pt 17
"""

import os
import sys

import torch

import ttnn
from models.autoports.thinkingmachines_inkling_small.tt.model import create_model


def pcc(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def main():
    fx_path, li = sys.argv[1], int(sys.argv[2])
    T_max = int(sys.argv[3]) if len(sys.argv) > 3 else 10**9
    fx = torch.load(fx_path)
    if isinstance(fx.get("x_in"), list):  # ref_inkling REF_GOLD=1 format: the residual entering every layer
        xs = fx["x_in"] + [fx["x_out"]]
        x_in, x_out = xs[li].float(), xs[li + 1].float()
        if li + 1 == len(fx["x_in"]):
            raise SystemExit("the last layer's gold output is post-final-norm; test layers < n_layers - 1")
    else:  # fixtures_inkling.py per-layer format
        x_in, x_out = fx["x_in"].float(), fx["x_out"].float()
    T = min(x_in.shape[0], T_max)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), l1_small_size=24576, trace_region_size=64_000_000)
    ok = False
    try:
        args, model = create_model(mesh, max_seq_len=max(2048, T), layer_indices=[li])
        layer = model.layers[0]
        layer.reset()
        rep = ttnn.ReplicateTensorToMesh(mesh)
        nblk = layer.attention.max_blocks
        pt = ttnn.from_torch(torch.arange(nblk, dtype=torch.int32).reshape(1, -1), dtype=ttnn.int32,
                             layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, mesh_mapper=rep)
        p_out, p_delta = [], []
        if os.environ.get("LAYER_MODE") == "prefill":  # the whole T rows as one prefill chunk (T % 64 == 0)
            T = T // 64 * 64
            x = ttnn.from_torch(x_in[:T].reshape(1, 1, T, -1).to(torch.bfloat16), dtype=ttnn.bfloat16,
                                layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=-1))
            out = layer.forward_prefill(x, 0, torch.arange(nblk, dtype=torch.int32))
            got = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=-1)).float().reshape(T, -1)
            for t in range(T):
                p_out.append(pcc(got[t], x_out[t]))
                p_delta.append(pcc(got[t] - x_in[t], x_out[t] - x_in[t]))
        for t in range(T if os.environ.get("LAYER_MODE") != "prefill" else 0):
            x = ttnn.from_torch(x_in[t].reshape(1, 1, 1, -1).to(torch.bfloat16), dtype=ttnn.bfloat16,
                                layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=-1))
            cur = ttnn.from_torch(torch.tensor([t], dtype=torch.int32), dtype=ttnn.int32,
                                  layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, mesh_mapper=rep)
            out = layer.forward(x, mode="decode", position_tensor=cur, page_table=pt)
            got = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=-1)).float().reshape(-1)
            p_out.append(pcc(got, x_out[t]))
            p_delta.append(pcc(got - x_in[t], x_out[t] - x_in[t]))
            for z in (x, cur, out):
                ttnn.deallocate(z)
            if t < 3 or t % 128 == 0 or t == T - 1:
                print(f"t={t:5d} pcc out {p_out[-1]:.5f} delta {p_delta[-1]:.5f}", flush=True)
        mean = lambda v: sum(v) / len(v)  # noqa: E731
        if os.environ.get("LAYER_MODE") == "prefill":
            worst = sorted(range(len(p_delta)), key=lambda t: p_delta[t])[:5]
            for t in worst:
                dr, dd = x_out[t] - x_in[t], got[t] - x_in[t]
                print(f"  worst t={t}: delta pcc {p_delta[t]:.4f} |x_in| {x_in[t].norm():.0f} |delta ref| {dr.norm():.0f} "
                      f"|delta dev| {dd.norm():.0f} max|x_in| {x_in[t].abs().max():.0f} argmax {int(x_in[t].abs().argmax())}")
        print(f"layer {li}: {T} tokens; out pcc worst {min(p_out):.5f} mean {mean(p_out):.5f}; "
              f"delta pcc worst {min(p_delta):.5f} mean {mean(p_delta):.5f}")
        ok = min(p_delta) > 0.98 and mean(p_delta) > 0.995
        print("OK" if ok else "FAIL")
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
