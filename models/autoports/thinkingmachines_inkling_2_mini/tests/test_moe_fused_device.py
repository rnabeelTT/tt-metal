# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The fused decode MoE (tt/moe_fused.py, ACT_INKLING kernels) vs the sparse_matmul MoE (tt/moe.py, verified against
transformers) on real layer inputs (the golden fixture's post-attention-layernorm rows), M = 1, 4, 8 rows, layers 0
(NVFP4) and 23 (MXFP8). Prints pcc and the device time of each path (eager, synced).
usage: tools/run_dev.sh python .../tests/test_moe_fused_device.py      env MESH (1), EXPERT_DT (bfp8)"""

import os
import time

import torch

import ttnn
from models.autoports.thinkingmachines_inkling_2_mini.tt.moe import MoE
from models.autoports.thinkingmachines_inkling_2_mini.tt.moe_fused import FusedExperts
from models.autoports.thinkingmachines_inkling_2_mini.tt.weights import Checkpoint, load_config


def pcc(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def main():
    N = int(os.environ.get("MESH", "1"))
    if N > 1:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, N))
    dt = {"bfp8": ttnn.bfloat8_b, "bfp4": ttnn.bfloat4_b}[os.environ.get("EXPERT_DT", "bfp8")]
    cfg, ck = load_config(), Checkpoint()
    g = torch.load("/cache/fixtures/golden_t256.pt")
    xs = g["internals"]["layers.0.post_attention_layernorm"][0][0]  # [T, H]
    rep = ttnn.ReplicateTensorToMesh(mesh) if N > 1 else None
    d0 = (lambda t: ttnn.get_device_tensors(t)[0]) if N > 1 else (lambda t: t)
    worst = 1.0
    for li in (0, 23):
        moe = MoE(mesh, cfg, ck, li, expert_dtype=dt, cache_dir="/cache/wcache")
        fe = FusedExperts(moe, ck, li, dtype=dt, cache_dir="/cache/wcache")
        rows = [int(v) for v in os.environ.get("ROWS", "1,4,8").split(",")]
        fe.build_rows([r for r in rows if r <= fe.rows_max])
        for M in rows:
            x = ttnn.from_torch(xs[:M].reshape(1, 1, M, -1).to(torch.bfloat16), dtype=ttnn.bfloat16,
                                layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep)
            # reference: the prefill path (sorted dispatch + grouped sparse matmuls), all-reduced
            ref = moe._sorted(x, M) if M >= 32 or os.environ.get("REF_SORTED") else moe.forward(x)
            if N > 1 and (M >= 32 or os.environ.get("REF_SORTED")):
                from models.autoports.thinkingmachines_inkling_2_mini.tt.ccl import all_reduce

                ref = all_reduce(ref, mesh)
            ref = ttnn.to_torch(d0(ref)).float().reshape(M, -1)
            got = fe.forward(x)
            if N > 1:
                from models.autoports.thinkingmachines_inkling_2_mini.tt.ccl import all_reduce

                got = all_reduce(got, mesh)
            got = ttnn.to_torch(d0(got)).float().reshape(M, -1)
            p = pcc(got, ref)
            worst = min(worst, p)
            t = {}
            for name, fn in (("sparse", lambda: moe.forward(x)), ("fused", lambda: fe.forward(x))):
                fn()
                ttnn.synchronize_device(mesh)
                t0 = time.time()
                for _ in range(10):
                    fn()
                ttnn.synchronize_device(mesh)
                t[name] = (time.time() - t0) / 10 * 1e6
            print(f"layer {li} M={M}: fused vs sparse pcc {p:.5f}, max abs diff {(got - ref).abs().max():.4f}  "
                  f"time sparse {t['sparse']:.0f} us, fused {t['fused']:.0f} us", flush=True)
    ttnn.close_mesh_device(mesh)
    print(f"{'PASS' if worst > 0.999 else 'FAIL'}: worst pcc {worst:.5f}")


if __name__ == "__main__":
    main()
