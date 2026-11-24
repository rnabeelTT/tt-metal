# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""tt/lane_sel vs a torch reference of the batched selection (synthetic scores)."""

import numpy as np
import torch
import ttnn

from models.autoports.tencent_hy4_flash.tt.lane_sel import LaneSelOp


def main():
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4))
    try:
        rep = ttnn.ReplicateTensorToMesh(mesh)
        rm = lambda t, dt: ttnn.from_torch(t, dtype=dt, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, mesh_mapper=rep)  # noqa
        back = lambda t: ttnn.to_torch(ttnn.get_device_tensors(t)[0])  # noqa
        torch.manual_seed(0)
        L, gcap, R, tw, r, n_sel = 4, 4096, 4, 4, 4, 511
        W = L * gcap + 64
        sc = torch.randn(1, 1, 32, W).to(torch.bfloat16)
        meta = np.zeros((32, 16), dtype=np.uint32)
        vis = [725, 723, 1500, 600]
        bases = [100000, 200000, 300000, 400000]
        for s in range(R):
            meta[s] = 0
            meta[s, :4] = [s * gcap, vis[s], bases[s], 1]
        dense = np.full((32, n_sel * r + tw), 0xFFFFFFFF, dtype=np.uint32)
        dense[:, 0] = 7
        tail = np.arange(32 * tw, dtype=np.uint32).reshape(32, tw) + 5
        op = LaneSelOp.get(mesh)
        sct = rm(sc, ttnn.bfloat16)
        mt = rm(torch.from_numpy(meta.view(np.int32)).reshape(1, 1, 32, 16), ttnn.uint32)
        comp = op.compact(sct, mt, gcap)
        c = back(comp).reshape(32, gcap)
        ok = True
        for s in range(32):
            v = int(meta[s, 1]) if meta[s, 3] else 0
            want = torch.full((gcap,), float("-inf"), dtype=torch.bfloat16)
            if v:
                want[:v] = sc[0, 0, s, int(meta[s, 0]): int(meta[s, 0]) + v]
            if not torch.equal(c[s], want):
                ok = False
                bad = (c[s] != want).nonzero()
                print(f"LANESEL compact row {s} mismatch at {bad[:5].flatten().tolist()} (of {bad.numel()})")
        print(f"LANESEL compact exact {ok}")
        vm = rm(torch.tensor([max(vis)], dtype=torch.int32).reshape(1, 1, 1, 1), ttnn.uint32)
        ids = ttnn.experimental.topk_large_indices(comp, k=512, valid_length_tensor=vm)
        print(f"LANESEL ids {ids.shape} {ids.dtype} {ids.layout}")
        it = back(ids).reshape(32, -1).to(torch.int64)
        idx = op.expand(ids, mt, rm(torch.from_numpy(tail.view(np.int32)).reshape(1, 1, 32, tw), ttnn.uint32),
                        rm(torch.from_numpy(dense.view(np.int32)).reshape(1, 1, 32, -1), ttnn.uint32), r=r, n_sel=n_sel)
        x = back(idx).reshape(32, -1).numpy().view(np.uint32)
        ok = True
        for s in range(32):
            if meta[s, 3]:
                top = it[s, :n_sel]
                vals = c[s].float()[top]
                ref_top = torch.topk(c[s].float()[: vis[s]], 512).values[:n_sel]
                if not torch.equal(torch.sort(vals).values, torch.sort(ref_top).values):
                    print(f"LANESEL topk row {s} differs from torch topk")
                want = np.concatenate([(np.repeat(top.numpy() * r, r) + np.tile(np.arange(r), n_sel) + bases[s]).astype(np.uint32), tail[s]])
            else:
                want = dense[s]
            if not np.array_equal(x[s], want):
                ok = False
                bad = np.nonzero(x[s] != want)[0]
                print(f"LANESEL expand row {s} mismatch at {bad[:5].tolist()} (of {bad.size}): {x[s][bad[:3]]} vs {want[bad[:3]]}")
        print(f"LANESEL expand exact {ok}")
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
