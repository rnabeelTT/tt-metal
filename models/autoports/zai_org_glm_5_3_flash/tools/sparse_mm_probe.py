# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Diagnostic: ttnn.sparse_matmul gather mode (the routed-expert path) vs torch at GLM-5.3 shapes, per in0_block_w."""

import torch

import ttnn
from models.autoports.zai_org_glm_5_3_flash.tests.harness import open_mesh, pcc
from models.autoports.zai_org_glm_5_3_flash.tt.moe import sparse_matmul_program_config

E, H, N, K = 16, 4096, 2048, 8


def main():
    torch.manual_seed(0)
    x = (torch.randn(1, 1, 1, H) * 0.4).bfloat16()
    wgu = (torch.randn(1, E, H, 2 * N) * 0.02).bfloat16()
    wd = (torch.randn(1, E, N, H) * 0.02).bfloat16()
    ids = torch.randperm(E)[:K].reshape(1, 1, 1, K)
    mesh = open_mesh()
    try:
        tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)
        twgu = ttnn.from_torch(wgu, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)
        twd = ttnn.from_torch(wd, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)
        tids = ttnn.from_torch(ids.to(torch.int32), dtype=ttnn.uint16, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh)
        ones = ttnn.from_torch(torch.ones(1, 1, 1, E), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh)
        ckc = ttnn.init_device_compute_kernel_config(
            mesh.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )
        ref_gu = torch.stack([x.float()[0, 0, 0] @ wgu[0, int(e)].float() for e in ids.flatten()])  # [K, 2N]
        h = torch.randn(1, K, 1, N).bfloat16() * 0.1
        th = ttnn.from_torch(h, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)
        ref_d = torch.stack([h.float()[0, j, 0] @ wd[0, int(e)].float() for j, e in enumerate(ids.flatten())])
        for fid in (ttnn.MathFidelity.HiFi2, ttnn.MathFidelity.HiFi4):
            for fp32, l1acc in ((False, False), (True, False), (False, True), (True, True)):
                ckc = ttnn.init_device_compute_kernel_config(
                    mesh.arch(), math_fidelity=fid, math_approx_mode=False, fp32_dest_acc_en=fp32, packer_l1_acc=l1acc
                )
                try:
                    cfg = sparse_matmul_program_config(1, 2 * N, H, target_cores=64, in0_block_w_cap=96)
                    out = ttnn.sparse_matmul(
                        tx,
                        twgu,
                        sparsity=ones,
                        indices=tids,
                        is_input_a_sparse=False,
                        is_input_b_sparse=True,
                        program_config=cfg,
                        compute_kernel_config=ckc,
                        dtype=ttnn.bfloat16,
                    )
                    got = ttnn.to_torch(out).float().reshape(-1)[: K * 2 * N].reshape(K, 2 * N)
                    cfg2 = sparse_matmul_program_config(1, H, N, target_cores=64, in0_block_w_cap=32)
                    out2 = ttnn.sparse_matmul(
                        th,
                        twd,
                        sparsity=ones,
                        indices=tids,
                        is_input_a_sparse=True,
                        is_input_b_sparse=True,
                        program_config=cfg2,
                        compute_kernel_config=ckc,
                        dtype=ttnn.bfloat16,
                    )
                    got2 = ttnn.to_torch(out2).float().reshape(-1)[: K * H].reshape(K, H)
                    print(
                        f"{str(fid):24s} fp32_dest={fp32!s:5} l1_acc={l1acc!s:5} gate_up PCC {pcc(got, ref_gu):.6f} "
                        f"ratio {(got.norm() / ref_gu.norm()).item():.4f} | down PCC {pcc(got2, ref_d):.6f} "
                        f"ratio {(got2.norm() / ref_d.norm()).item():.4f}"
                    )
                except Exception as ex:  # noqa: BLE001
                    print(f"{fid} fp32={fp32} l1acc={l1acc}: {str(ex)[:120]}")
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__" and __import__("os").environ.get("PROBE_MASKED") != "1":
    main()


def masked_main():
    """Masked mode (prefill chunks): 32 token rows, sparsity = union of the chunk's experts."""
    torch.manual_seed(0)
    R = 32
    x = (torch.randn(1, 1, R, H) * 0.4).bfloat16()
    wgu = (torch.randn(1, E, H, 2 * N) * 0.02).bfloat16()
    mask = torch.zeros(1, 1, 1, E)
    mask[0, 0, 0, torch.randperm(E)[:10]] = 1.0
    mesh = open_mesh()
    try:
        tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)
        twgu = ttnn.from_torch(wgu, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)
        tm = ttnn.from_torch(mask, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh)
        live = [int(e) for e in mask.flatten().nonzero()]
        ref = torch.stack([x.float()[0, 0] @ wgu[0, e].float() for e in live])  # [L, R, 2N]
        for fp32 in (False, True):
            ckc = ttnn.init_device_compute_kernel_config(
                mesh.arch(),
                math_fidelity=ttnn.MathFidelity.HiFi2,
                math_approx_mode=False,
                fp32_dest_acc_en=fp32,
                packer_l1_acc=False,
            )
            for cores in (8, 16, 32, 64):
                for dt in (1, 2, 4):
                    cfg = sparse_matmul_program_config(
                        R, 2 * N, H, target_cores=cores, in0_block_w_cap=16, dest_tiles=dt
                    )
                    try:
                        out = ttnn.sparse_matmul(
                            tx,
                            twgu,
                            sparsity=tm,
                            nnz=None,
                            program_config=cfg,
                            compute_kernel_config=ckc,
                            dtype=ttnn.bfloat16,
                        )
                    except Exception as ex:  # noqa: BLE001
                        print(f"masked fp32_dest={fp32} cores={cores} dt={dt}: {str(ex)[:100]}")
                        continue
                    got = ttnn.to_torch(out).float().reshape(E, R, 2 * N)[live]
                    print(
                        f"masked fp32_dest={fp32!s:5} cores={cores:2d} sub_w={cfg.out_subblock_w} per_core_N={cfg.per_core_N}: "
                        f"PCC {pcc(got, ref):.6f} ratio {(got.norm() / ref.norm()).item():.4f}"
                    )
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__" and __import__("os").environ.get("PROBE_MASKED") == "1":
    masked_main()
