# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Debug the fused HC PRE: HY4_HCF_DBG=1 (w := mix sum), 2 (ss), 3 (sum S) vs torch."""
import os

import torch
import ttnn


def main():
    from models.autoports.tencent_hy4_flash.tt.hc import HyperConnection
    from models.autoports.tencent_hy4_flash.tt.hc_fused import HCFusedOp

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4))
    try:
        rep = ttnn.ReplicateTensorToMesh(mesh)
        torch.manual_seed(0)
        hc, H = 4, 3072
        fn = torch.randn(8, hc * H) * 0.02
        base, scale = torch.randn(8) * 0.5, torch.tensor([0.7, 1.3])
        X = (torch.randn(1, hc, 32, H) * 2).to(torch.bfloat16)
        Xt = ttnn.from_torch(X, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep,
                             memory_config=ttnn.L1_MEMORY_CONFIG)
        h = HyperConnection(mesh, fn, base, scale, hc, H, 1e-5, 1e-6, 2.0)
        op = HCFusedOp.get(mesh, H)
        _, inp, w = op(Xt, None, None, h)
        got = ttnn.to_torch(ttnn.get_device_tensors(w)[0]).float().reshape(32, 32)
        flat = X.float().permute(2, 0, 1, 3).reshape(32, hc * H)  # [rows, (s, h)]
        mix = flat @ fn.T
        ss = flat.pow(2).sum(-1)
        dbg = os.environ.get("HY4_HCF_DBG", "0")
        if dbg == "1":
            print("mix: got", got[0, :8].tolist(), "\n     ref", mix[0, :8].tolist())
            print("max err", float((got[:, :8] - mix).abs().max()), "ref max", float(mix.abs().max()))
        elif dbg == "2":
            print("ss: got col0", got[:4, 0].tolist(), "col5", got[:4, 5].tolist(), "\n    ref", ss[:4].tolist())
        elif dbg == "3":
            print("S diag", torch.diagonal(got)[:4].tolist(), "ref", ss[:4].tolist(), "S[0,:4]", got[0, :4].tolist())
        else:
            r = torch.rsqrt(ss / (hc * H) + 1e-5)
            mixn = mix * r[:, None]
            sc = torch.tensor([0.7] * 4 + [1.3] * 4)
            a = torch.tensor([1.0] * 4 + [2.0] * 4)
            wr = a * torch.sigmoid(mixn * sc + base) + 1e-6
            print("w got", got[0, :8].tolist(), "\n  ref", wr[0, :8].tolist(), "\nmax err", float((got[:, :8] - wr).abs().max()))
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
