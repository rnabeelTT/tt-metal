# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Per-layer MoE expert-precision sensitivity: for every MoE layer and every precision policy, the MoE block alone
(tt/optimized_decoder.Qwen35MoEOptimized, prefill path: all T rows in one call) on the real weights against the HF
fixture (fixtures_inkling.py ... moe_all: mlp_in -> mlp_out). Writes {layer: {policy: {pcc_mean, pcc_p01, pcc_worst}}}
to drive the per-layer precision assignment (which layers get bfp8 experts within the device budget).

usage (in the image): tools/run_dev.sh python .../tools/sweep_moe_precision.py <fixture dir> <out.json> [layers a-b]
env: SWEEP_POLICIES (comma list of doc/precision/*.json stems; default routed_bfp4,routed_bfp8)
"""

import gc
import json
import os
import sys

import torch

import ttnn

POL_DIR = "/opt/tt-metal/models/autoports/thinkingmachines_inkling_small/doc/precision"


def pcc_rows(a, b):
    a = a.double() - a.double().mean(-1, keepdim=True)
    b = b.double() - b.double().mean(-1, keepdim=True)
    return (a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1) + 1e-30)


def main():
    fx_dir, out_path = sys.argv[1], sys.argv[2]
    a, b = (sys.argv[3].split("-") if len(sys.argv) > 3 else ("2", "41"))
    layers = list(range(int(a), int(b) + 1))
    pols = os.environ.get("SWEEP_POLICIES", "routed_bfp4,routed_bfp8").split(",")
    res = json.load(open(out_path)) if os.path.exists(out_path) else {}
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), l1_small_size=24576, trace_region_size=64_000_000)
    try:
        from models.autoports.thinkingmachines_inkling_small.tt.model import create_model

        for li in layers:
            fx = torch.load(os.path.join(fx_dir, f"layer{li}.pt"))
            x_in, want = fx["mlp_in"].float(), fx["mlp_out"].float()
            T = x_in.shape[0] // 32 * 32
            for pol in pols:
                if res.get(str(li), {}).get(pol):
                    continue
                os.environ["QWEN35_PRECISION_CONFIG"] = f"{POL_DIR}/{pol}.json"
                args, model = create_model(mesh, max_seq_len=2048, layer_indices=[li])
                moe = model.layers[0].feed_forward
                x = ttnn.from_torch(x_in[:T].reshape(1, 1, T, -1).to(torch.bfloat16), dtype=ttnn.bfloat16,
                                    layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
                out = moe.forward(x, mode="prefill")
                got = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=-1)).float().reshape(T, -1)
                p = pcc_rows(got, want[:T])
                r = {"pcc_mean": p.mean().item(), "pcc_p01": p.quantile(0.01).item(), "pcc_worst": p.min().item()}
                res.setdefault(str(li), {})[pol] = r
                print(f"layer {li:2d} {pol:16s} mean {r['pcc_mean']:.5f} p01 {r['pcc_p01']:.5f} worst {r['pcc_worst']:.5f}",
                      flush=True)
                json.dump(res, open(out_path, "w"), indent=1)
                ttnn.deallocate(x)
                ttnn.deallocate(out)
                del model, moe
                gc.collect()
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()
