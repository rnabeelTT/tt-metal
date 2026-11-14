# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device test of tt/attention.InklingAttention (one-row decode, TP over the (1, 4) mesh) on the real weights of one layer
against the HF fixture (models/experimental/hybrid_moe/fixtures_inkling.py: attn_in -> attn_out for the first T tokens
of a held-out text). Decodes token by token from position 0 and compares every step's output (PCC per token, worst and
mean), so the sconv state, the KV write / ring, the window and the relative bias are all exercised; T > 512 crosses the
sliding window, T > 1024 the global band.

usage (in the image): tools/run_dev.sh python models/autoports/thinkingmachines_inkling_small/tests/test_attention_device.py \
        /cache/fixtures/layer5.pt 5
"""

import sys

import torch

import ttnn
from models.autoports.thinkingmachines_inkling_small.tt.attention import InklingAttention
from models.autoports.thinkingmachines_inkling_small.tt.weights import InklingCheckpoint
from models.tt_transformers.tt.ccl import TT_CCL


def pcc(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def main():
    fx_path, li = sys.argv[1], int(sys.argv[2])
    T_max = int(sys.argv[3]) if len(sys.argv) > 3 else 10**9
    fx = torch.load(fx_path)
    x_all, want = fx["attn_in"], fx["attn_out"]
    T = min(x_all.shape[0], T_max)
    ck = InklingCheckpoint()
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), l1_small_size=24576, trace_region_size=64_000_000)
    try:
        tt_ccl = TT_CCL(mesh)
        att = InklingAttention(mesh, ck.cfg, ck.attn(li), li, tt_ccl=tt_ccl, max_seq_len=max(T, 2048))
        pt = ttnn.from_torch(
            torch.arange(att.max_blocks, dtype=torch.int32).reshape(1, -1), dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        rep = ttnn.ReplicateTensorToMesh(mesh)
        pccs = []
        for t in range(T):
            x = ttnn.from_torch(x_all[t].reshape(1, 1, 1, -1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                device=mesh, mesh_mapper=rep)
            cur = ttnn.from_torch(torch.tensor([t], dtype=torch.int32), dtype=ttnn.int32,
                                  layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, mesh_mapper=rep)
            out = att.forward_decode(x, cur, pt)  # [1, 1, 1, H/4] per die, fractured
            got = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=-1)).float().reshape(-1)
            pccs.append(pcc(got, want[t]))
            for z in (x, cur, out):
                ttnn.deallocate(z)
            if t < 4 or t % 128 == 0 or t == T - 1:
                print(f"t={t:5d} pcc {pccs[-1]:.5f}", flush=True)
        worst = min(pccs)
        mean = sum(pccs) / len(pccs)
        print(f"layer {li} ({att.W and 'sliding' or 'global'}): {T} tokens, pcc worst {worst:.5f} mean {mean:.5f}")
        ok = worst > 0.99 and mean > 0.998
        print("OK" if ok else "FAIL")
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
