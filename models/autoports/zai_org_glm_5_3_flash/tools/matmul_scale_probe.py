# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Diagnostic (stage-1 review #2 P2): output scale of the dense matmuls the decoder uses, vs torch on the SAME (read-back)
device weights: layer-0 dense-MLP gate projection (bfp8) and KDA input projection (bf16), 200 rows, fp32 DST HiFi4, with
ttnn's default program config and with an explicit 2D-multicast config (the factory with the fp32-partials fix)."""

import torch

import ttnn
from models.autoports.zai_org_glm_5_3_flash.tests.harness import open_mesh, pcc
from models.autoports.zai_org_glm_5_3_flash.tt.ckpt import CheckpointReader


def main():
    rd = CheckpointReader()
    sd = rd.layer_state(0)
    mesh = open_mesh()
    try:
        ckc = ttnn.init_device_compute_kernel_config(
            mesh.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=False
        )
        g = torch.Generator().manual_seed(0)
        x = (torch.randn(1, 1, 224, 4096, generator=g) * 0.3).bfloat16()
        tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)
        for name, w, dt in (
            ("mlp gate (bfp8)", sd["mlp.gate_proj.weight"], ttnn.bfloat8_b),
            ("kda in_proj q (bf16)", sd["self_attn.q_proj.weight"], ttnn.bfloat16),
        ):
            tw = ttnn.from_torch(
                w.float().T.contiguous().reshape(1, 1, *w.T.shape), dtype=dt, layout=ttnn.TILE_LAYOUT, device=mesh
            )
            wq = ttnn.to_torch(tw).float()[0, 0]
            ref = x.float()[0, 0] @ wq
            N = wq.shape[-1]
            pcs = {
                "default": None,
                "2d mcast": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                    compute_with_storage_grid_size=(10, 7),
                    in0_block_w=4,
                    out_subblock_h=1,
                    out_subblock_w=4,
                    per_core_M=1,
                    per_core_N=-(-N // 32 // 10),
                    transpose_mcast=False,
                    fused_activation=None,
                ),
            }
            for pname, pc in pcs.items():
                try:
                    y = ttnn.to_torch(
                        ttnn.linear(tx, tw, compute_kernel_config=ckc, program_config=pc, dtype=ttnn.float32)
                    ).float()[0, 0]
                    print(f"{name} [{pname}]: PCC {pcc(y, ref):.6f} rms ratio {(y.norm() / ref.norm()).item():.5f}")
                except Exception as e:
                    print(f"{name} [{pname}]: error {str(e).splitlines()[0][:150]}")
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
