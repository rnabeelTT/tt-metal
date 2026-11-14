# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device test of one Inkling MoE block (tt/optimized_decoder.Qwen35MoEOptimized, router_kind inkling) on the real
weights against the HF fixture (fixtures_inkling.py: mlp_in -> mlp_out, topk_ids). Per token: PCC of the MoE output,
and whether the device's 6 routed picks equal HF's (a routing flip on a near tie moves a whole expert's worth of
output). The precision policy is the run's (QWEN35_PRECISION_CONFIG=<json> to ablate, e.g. bfp8 experts).

usage (in the image): tools/run_dev.sh python models/autoports/thinkingmachines_inkling_small/tests/test_moe_device.py \
        /cache/fixtures/layer5.pt 5 [T]
"""

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
    x_in, want, ids_hf = fx["mlp_in"].float(), fx["mlp_out"].float(), fx["topk_ids"].long()
    T = min(x_in.shape[0], T_max)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), l1_small_size=24576, trace_region_size=64_000_000)
    try:
        args, model = create_model(mesh, max_seq_len=2048, layer_indices=[li])
        moe = model.layers[0].feed_forward
        rep = ttnn.ReplicateTensorToMesh(mesh)
        p_all, flips, p_flip, p_ok = [], 0, [], []
        for t in range(T):
            x = ttnn.from_torch(x_in[t].reshape(1, 1, 1, -1).to(torch.bfloat16), dtype=ttnn.bfloat16,
                                layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep)
            sc, ids = moe._route(x)
            dev_ids = ttnn.to_torch(ttnn.get_device_tensors(ids)[0]).reshape(-1)[:8].long()
            out = moe.forward(x, mode="decode")
            got = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=-1)).float().reshape(-1)
            p = pcc(got, want[t])
            same = set(dev_ids[:6].tolist()) == set(ids_hf[t].tolist())
            flips += not same
            (p_ok if same else p_flip).append(p)
            p_all.append(p)
            for z in (x, sc, ids, out):
                ttnn.deallocate(z)
        m = lambda v: sum(v) / max(len(v), 1)  # noqa: E731
        print(f"layer {li} MoE: {T} tokens; pcc worst {min(p_all):.5f} mean {m(p_all):.5f}; routing flips {flips} "
              f"({flips / T:.1%}); pcc same-routing mean {m(p_ok):.5f} worst {min(p_ok or [1]):.5f}; "
              f"flipped-routing mean {m(p_flip):.5f}")
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()
