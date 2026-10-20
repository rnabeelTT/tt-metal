# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage-3 re-review #2 P2-2a, step 3: standalone repro of the batched MatmulMultiCoreReuseProgramConfig wrong output
(no model code). One process per case (--case), fresh random tensors:
  [1, 64, M, K] bf16 x [1, 64, K, N] bfp8, full 11 x 10 grid, in0_block_w = min(Kt, 8), out_subblock 1 x w, per_core_N = Nt
  calls: x1 (first call, program compiled), x2 (same shapes, new tensors: program-cache hit), x1 again, then x2 with the
  program cache cleared before the call. Each output against torch fp32 of the device operands.
Cases: uk (K 256, N 512, the DSA W_UK shape) and uv (K 512, N 256, W_UV), M 2048 rows, per_core_M 8 / 2; and uk at M 128
(per_core_M 4 = Mt, the shape that works in the decoder).

--oob (review #3, discriminating): no prior default call; the reuse matmul writes into a NaN-prefilled output
(optional_output_tensor) followed in DRAM by a canary tensor of 7.0; reports the output tiles never written (still NaN),
the canary elements overwritten (out-of-bounds writes), and the correct / wrong heads.

usage: python -m models.autoports.zai_org_glm_5_3_flash.tools.s3r2_bmm_repro3 --case uk --pm 8 --m 2048 [--oob]"""

import argparse

import torch

import ttnn
from models.autoports.zai_org_glm_5_3_flash.tests.harness import open_mesh, pcc

DRAM = ttnn.DRAM_MEMORY_CONFIG


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="uk")
    ap.add_argument("--pm", type=int, default=8)
    ap.add_argument("--m", type=int, default=2048)
    ap.add_argument("--ib", type=int, default=0, help="in0_block_w override (0 = min(Kt, 8))")
    ap.add_argument("--swap", action="store_true", help="x2's values first (order / address vs data dependence)")
    ap.add_argument("--oob", action="store_true", help="NaN-prefilled output + canary, no prior default call")
    a = ap.parse_args()
    K, N = (256, 512) if a.case == "uk" else (512, 256)
    H, M = 64, a.m
    mesh = open_mesh()
    try:
        grid = mesh.compute_with_storage_grid_size()
        ckc = ttnn.init_device_compute_kernel_config(
            mesh.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        Kt, Nt = K // 32, N // 32
        ib = a.ib or min(Kt, 8)
        pc = ttnn.MatmulMultiCoreReuseProgramConfig(
            compute_with_storage_grid_size=(grid.x, grid.y),
            in0_block_w=ib,
            out_subblock_h=1,
            out_subblock_w=max(d for d in (4, 2, 1) if Nt % d == 0),
            per_core_M=a.pm,
            per_core_N=Nt,
        )
        print(
            f"case {a.case}: x [1,{H},{M},{K}] bf16, w [1,{H},{K},{N}] bfp8, grid {grid.x}x{grid.y}, {pc}", flush=True
        )
        g = torch.Generator().manual_seed(0)
        w = (torch.randn(1, H, K, N, generator=g) / K**0.5).bfloat16()
        tw = ttnn.from_torch(w, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=DRAM)
        wq = ttnn.to_torch(tw).float()
        xs = [torch.randn(1, H, M, K, generator=g).bfloat16() for _ in range(2)]
        if a.swap:
            xs = xs[::-1]
        txs = [
            ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=DRAM)
            for x in xs
        ]
        refs = [x.float() @ wq for x in xs]

        def run(i, tag, default=False):
            kw = {} if default else {"program_config": pc}
            o = ttnn.to_torch(ttnn.matmul(txs[i], tw, compute_kernel_config=ckc, memory_config=DRAM, **kw)).float()
            per_head = torch.stack([torch.tensor(pcc(o[0, h], refs[i][0, h])) for h in range(H)])
            bad = (per_head < 0.999).nonzero().flatten().tolist()
            stale = [h for h in bad if pcc(o[0, h], refs[1 - i][0, h]) > 0.999]
            print(
                f"  {tag:34}: PCC {pcc(o, refs[i]):.6f}  heads wrong {len(bad)} {bad[:12]}  of which equal to the OTHER "
                f"input's product (stale input) {len(stale)}",
                flush=True,
            )

        if a.oob:
            bpc = -(-(M // 32) // a.pm)  # M blocks; blocks per core over the grid
            print(f"  M blocks {bpc} x H {H} = {bpc * H} blocks on {grid.x * grid.y} cores", flush=True)
            nan = torch.full((1, H, M, N), float("nan")).bfloat16()
            out = ttnn.from_torch(nan, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=DRAM)
            canary = ttnn.from_torch(
                torch.full((1, H, M, N), 7.0).bfloat16(),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                memory_config=DRAM,
            )
            ttnn.matmul(
                txs[0], tw, compute_kernel_config=ckc, memory_config=DRAM, program_config=pc, optional_output_tensor=out
            )
            o = ttnn.to_torch(out).float()
            c = ttnn.to_torch(canary).float()
            tiles = o.reshape(1, H, M // 32, 32, N // 32, 32).isnan().any(-1).any(-2)  # [1, H, Mt, Nt]
            per_head = [pcc(torch.nan_to_num(o[0, h]), refs[0][0, h]) for h in range(H)]
            bad = [h for h in range(H) if per_head[h] < 0.999]
            print(
                f"  oob: output tiles never written {int(tiles.sum())} of {tiles.numel()} (heads {sorted(set(tiles.nonzero()[:, 1].tolist()))[:16]}); "
                f"canary elements overwritten {int((c != 7.0).sum())} of {c.numel()}; heads wrong {len(bad)} {bad[:12]}",
                flush=True,
            )
            return
        run(0, "default config x1", default=True)
        run(0, "reuse x1 (compile)")
        run(1, "reuse x2 (program-cache hit)")
        run(0, "reuse x1 again (cache hit)")
        mesh.disable_and_clear_program_cache()
        mesh.enable_program_cache()
        run(1, "reuse x2 (cache cleared first)")
        run(0, "reuse x1 (cache hit after x2)")
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
