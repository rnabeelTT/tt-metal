# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""tt/gmm.py (grouped expert matmul) vs torch: the prefill MoE's gate|up
(K 2304, N 768) and down (K 384, N 2304) shapes, bfp8 weights, groups sorted by expert (repeats + a padding group)."""

import time

import torch

import ttnn
from models.autoports.minimaxai_minimax_m2_7.tt.moe import sparse_matmul_program_config
from models.autoports.thinkingmachines_inkling_2_mini.tt.gmm import GroupedMM


def main():
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    E, gs = 32, 32
    ckc = ttnn.init_device_compute_kernel_config(mesh.arch(), math_fidelity=ttnn.MathFidelity.HiFi2,
                                                 math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False)
    gmm = GroupedMM(mesh)
    ok = True
    torch.manual_seed(0)
    for K, N in ((2304, 768), (384, 2304)):
        experts = sorted(torch.randint(0, E, (300,)).tolist())
        ids = torch.tensor(experts + [E])  # (a padding group)
        G = len(ids)
        a = torch.randn(1, G, gs, K)
        w = torch.randn(1, E, K, N) * 0.02
        dram = ttnn.DRAM_MEMORY_CONFIG
        ta = ttnn.from_torch(a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=dram)
        tw = ttnn.from_torch(w, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=dram)
        tid = ttnn.from_torch(ids.reshape(1, G).to(torch.int32), dtype=ttnn.uint16, layout=ttnn.ROW_MAJOR_LAYOUT,
                              device=mesh, memory_config=dram)
        out = ttnn.allocate_tensor_on_device(ttnn.Shape([1, G, gs, N]), ttnn.bfloat16, ttnn.TILE_LAYOUT, mesh, dram)
        print(f"K={K} N={N}: gmm launch", flush=True)
        gmm(ta, tw, tid, out, E)
        ttnn.synchronize_device(mesh)
        print("  gmm done", flush=True)
        got = ttnn.to_torch(out).float()
        print("  out read", flush=True)
        wq = ttnn.to_torch(tw).float()  # (the bfp8-rounded weights; slow for many experts)
        print("  weights read", flush=True)
        ref = torch.stack([a[0, g].bfloat16().float() @ wq[0, int(ids[g])] for g in range(G - 1)])
        rel = ((got[0, : G - 1] - ref).abs().max() / ref.abs().max()).item()
        print(f"  ref done rel {rel:.2e}", flush=True)
        fn = lambda: gmm(ta, tw, tid, out, E)  # noqa: E731
        fn()
        ttnn.synchronize_device(mesh)
        t0 = time.perf_counter()
        for _ in range(10):
            fn()
        ttnn.synchronize_device(mesh)
        ms = (time.perf_counter() - t0) / 10 * 1e3
        ok &= rel < 1e-2
        print(f"K={K} N={N} G={G}: rel err vs torch {rel:.2e}; gmm {ms:.2f} ms", flush=True)
    ttnn.close_mesh_device(mesh)
    print("PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
