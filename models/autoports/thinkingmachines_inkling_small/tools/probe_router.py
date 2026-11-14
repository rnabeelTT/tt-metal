# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Which step of the device router disagrees with the fp32 reference router (logits, sigmoid, top-k)?
usage (in the image): tools/run_dev.sh python .../tools/probe_router.py /cache/fixtures/layer5.pt 5"""

import sys

import torch

import ttnn
from models.autoports.thinkingmachines_inkling_small.tt.model import create_model


def main():
    fx = torch.load(sys.argv[1])
    li = int(sys.argv[2])
    x_in, ids_hf = fx["mlp_in"], fx["topk_ids"].long()
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), l1_small_size=24576)
    try:
        args, model = create_model(mesh, max_seq_len=2048, layer_indices=[li])
        moe = model.layers[0].feed_forward
        sd = args.load_state_dict()
        W = sd[f"layers.{li}.mlp.gate.weight"].to(torch.bfloat16).float()
        bias = sd[f"layers.{li}.mlp.gate.e_score_correction_bias"].float()
        rep = ttnn.ReplicateTensorToMesh(mesh)
        T = 256
        st = {"logit_maxerr": 0.0, "sig_maxerr": 0.0, "flip_dev": 0, "flip_hostsel_on_devlogits": 0, "flip_ref": 0}
        for t in range(T):
            xb = x_in[t].to(torch.bfloat16)
            x = ttnn.from_torch(xb.reshape(1, 1, 1, -1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh,
                                mesh_mapper=rep)
            lg = moe._router_logits(x, ttnn.L1_MEMORY_CONFIG, dtype=ttnn.float32)
            sg = ttnn.sigmoid(lg)
            d_lg = ttnn.to_torch(ttnn.get_device_tensors(lg)[0]).reshape(-1)[:258].float()
            d_sg = ttnn.to_torch(ttnn.get_device_tensors(sg)[0]).reshape(-1)[:258].float()
            h_lg = xb.float() @ W.T
            st["logit_maxerr"] = max(st["logit_maxerr"], (d_lg - h_lg).abs().max().item())
            st["sig_maxerr"] = max(st["sig_maxerr"], (d_sg - h_lg.sigmoid()).abs().max().item())
            ref = set(torch.topk(h_lg[:256].sigmoid() + bias, 6).indices.tolist())
            st["flip_ref"] += ref != set(ids_hf[t].tolist())  # host fp32 vs the fixture (should be 0)
            st["flip_hostsel_on_devlogits"] += set(torch.topk(d_lg[:256].sigmoid() + bias, 6).indices.tolist()) != ref
            _, ids = moe._route(x)
            dev = set(ttnn.to_torch(ttnn.get_device_tensors(ids)[0]).reshape(-1)[:6].long().tolist())
            st["flip_dev"] += dev != ref
            for z in (x, lg, sg, ids):
                ttnn.deallocate(z)
        print({k: (round(v, 6) if isinstance(v, float) else v) for k, v in st.items()}, f"over {T} tokens")
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()
