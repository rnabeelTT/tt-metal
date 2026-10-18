# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""DSA decode W_UV products at 1 slot (stage-5 review #2, `MCOpts.dsa_o_mm`): the per-head bmm of GlmDSA._bmm (full-grid
reuse config) vs one [16, 512] x [512, 16*256] matmul (head-masked, rows summed) and vs the stacked-view form, each
against an fp64 product of the same (dequantized bfp8) weights. One die, TP4 per-die shapes (16 heads).

  python -m models.autoports.zai_org_glm_5_3_flash.tools.dsa_out_probe
"""

from __future__ import annotations

import json

import torch

import ttnn
from models.demos.blackhole.qwen36.tt import tp_common as tpc

H, R, DV = 16, 512, 256


def main():
    dev = ttnn.open_device(device_id=0)
    try:
        g = torch.Generator().manual_seed(0)
        ckc = ttnn.init_device_compute_kernel_config(
            dev.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        up = lambda t, dt, lay=ttnn.TILE_LAYOUT: ttnn.from_torch(t, dtype=dt, layout=lay, device=dev)
        w = up(torch.randn(1, H, R, DV, generator=g) / R**0.5, ttnn.bfloat8_b)
        wd = ttnn.to_torch(w).double()[0]  # [H, R, DV] dequantized
        o = torch.randn(1, H, 1, R, generator=g).bfloat16()
        ref = torch.einsum("hr,hrd->hd", o[0, :, 0].double(), wd).reshape(1, H * DV)
        grid = dev.compute_with_storage_grid_size()
        res = {}
        # base: per-head bmm, GlmDSA._bmm reuse config
        pc = ttnn.MatmulMultiCoreReuseProgramConfig(
            compute_with_storage_grid_size=(grid.x, grid.y),
            in0_block_w=8,
            out_subblock_h=1,
            out_subblock_w=4,
            per_core_M=1,
            per_core_N=DV // 32,
        )
        for name, kw in (("bmm_reuse", {"program_config": pc}), ("bmm_default", {})):
            y = ttnn.matmul(up(o, ttnn.bfloat16), w, compute_kernel_config=ckc, **kw)
            res[name] = ttnn.to_torch(y).double()[0, :, 0].reshape(1, H * DV)
        # omm 1: one matmul with W_UV laid out [512, H*256], masked, rows summed
        wcat = ttnn.reshape(ttnn.permute(w, (0, 2, 1, 3)), (1, 1, R, H * DV))
        mask = torch.zeros(1, 1, H, H * DV)
        for h in range(H):
            mask[0, 0, h, h * DV : (h + 1) * DV] = 1
        pco = tpc.create_matmul_1d_decode_progcfg(32, R, H * DV, 64, fp32_acc=True, grid_w=grid.x, in0_block_w=8)
        full = ttnn.matmul(
            up(o.reshape(1, 1, H, R), ttnn.bfloat16), wcat, compute_kernel_config=ckc, program_config=pco
        )
        res["omm_full_diag"] = (ttnn.to_torch(full).double()[0, 0] * mask[0, 0].double()).sum(0, keepdim=True)
        s = ttnn.sum(ttnn.multiply(full, up(mask, ttnn.bfloat16)), dim=-2, keepdim=True, compute_kernel_config=ckc)
        res["omm1"] = ttnn.to_torch(s).double().reshape(1, H * DV)
        full_d = ttnn.matmul(up(o.reshape(1, 1, H, R), ttnn.bfloat16), wcat, compute_kernel_config=ckc)
        res["omm_default_pc_diag"] = (ttnn.to_torch(full_d).double()[0, 0] * mask[0, 0].double()).sum(0, keepdim=True)
        out = {}
        for k, v in res.items():
            out[k] = {
                "max_rel_vs_fp64": float((v - ref).abs().max() / ref.abs().max()),
                "rel_l2_vs_fp64": float((v - ref).norm() / ref.norm()),
                "equal_bmm_reuse": bool(torch.equal(v, res["bmm_reuse"])),
                "max_rel_vs_bmm_reuse": float((v - res["bmm_reuse"]).abs().max() / ref.abs().max()),
            }
        print("DSAO", json.dumps(out))
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
