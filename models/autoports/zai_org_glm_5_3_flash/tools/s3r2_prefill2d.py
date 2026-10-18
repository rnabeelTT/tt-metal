# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage-3 re-review #2 P2-2b: adapted explicit 2D prefill configs for the dense gate|up [2048, 4096] x [4096, 24576]
and the KDA input projection [2048, 4096] x [4096, 32960] (bfp8 weights, bf16 activations, fp32 accumulation), one die,
synthetic weights at the real shapes. The review-1 probe rejected both at the first L1 error (create_prefill_matmul_
program_config, in0 4 on 11 x 10: circular buffers 3.85 / 5.09 MB). Here: MatmulMultiCoreReuseMultiCastProgramConfig
over grids (11, 10) / (8, 10) / (11, 8), in0_block_w 4 / 2 / 1, and out_block_w splits of per_core_N (the per-core output
in several blocks: the output / partials CBs hold one block). Host-timed (sync, best config n=5); PCC vs the ttnn
default of the same fidelity; a failing variant prints its exact error (CB bytes).

usage: python -m models.autoports.zai_org_glm_5_3_flash.tools.s3r2_prefill2d"""

import math
import time

import torch

import ttnn
from models.autoports.zai_org_glm_5_3_flash.tests.harness import open_mesh, pcc

DRAM = ttnn.DRAM_MEMORY_CONFIG
FID = {"HiFi4": ttnn.MathFidelity.HiFi4, "HiFi2": ttnn.MathFidelity.HiFi2}


def ckc_of(mesh, fid, l1_acc):
    return ttnn.init_device_compute_kernel_config(
        mesh.arch(), math_fidelity=FID[fid], math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=l1_acc
    )


def timed(mesh, fn, n=5):
    ttnn.deallocate(fn())
    ttnn.synchronize_device(mesh)
    t = time.perf_counter()
    for _ in range(n):
        ttnn.deallocate(fn())
    ttnn.synchronize_device(mesh)
    return (time.perf_counter() - t) / n * 1e6


def err(e):
    return " | ".join(l.strip() for l in str(e).splitlines() if l.strip() and "backtrace" not in l.lower())[:300]


def divisors_desc(n):
    return [d for d in range(n, 0, -1) if n % d == 0]


def main():
    mesh = open_mesh()
    try:
        for role, K, N in (("mlp_gu", 4096, 24576), ("kda_in", 4096, 32960)):
            g = torch.Generator().manual_seed(5)
            x = torch.randn(1, 1, 2048, K, generator=g).bfloat16()
            w = (torch.randn(1, 1, K, N, generator=g) / K**0.5).bfloat16()
            tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)
            tw = ttnn.from_torch(w, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=DRAM)
            Kt, Nt, Mt = K // 32, -(-N // 32), 2048 // 32
            good = []  # (us, tag, pc) of the HiFi4 sweep
            for fid, l1 in (("HiFi4", False), ("HiFi4", True), ("HiFi2", False)):
                ckc = ckc_of(mesh, fid, l1)
                tag0 = f"{role} {fid}{' l1acc' if l1 else ''}"
                ref = ttnn.to_torch(ttnn.linear(tx, tw, compute_kernel_config=ckc, memory_config=DRAM)).float()
                us = timed(mesh, lambda: ttnn.linear(tx, tw, compute_kernel_config=ckc, memory_config=DRAM))
                print(f"{tag0} default: {us:9.1f} us", flush=True)
                if fid == "HiFi4" and not l1:
                    cands = []
                    for gx, gy in ((11, 10), (8, 10), (11, 8)):
                        pm, pn = math.ceil(Mt / gy), math.ceil(Nt / gx)
                        obws = sorted({d for d in divisors_desc(pn) if d >= 4 or d == pn})[::-1][:4]
                        for ib in (4, 2, 1):
                            for obw in obws:
                                sw = max(d for d in (4, 2, 1) if obw % d == 0)
                                for obh in sorted({pm, max(d for d in divisors_desc(pm) if d <= max(1, pm // 2))}):
                                    tag = f"2D {gx}x{gy} pm {pm} pn {pn} in0 {ib} out_block {obh}x{obw} sub 1x{sw}"
                                    cands.append((tag, (gx, gy, ib, sw, obh, obw, pm, pn)))
                else:
                    cands = [(t, a) for _, t, a in sorted(good)[:3]]
                for tag, (gx, gy, ib, sw, obh, obw, pm, pn) in cands:
                    try:
                        pc = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                            compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy),
                            in0_block_w=ib,
                            out_subblock_h=1,
                            out_subblock_w=sw,
                            out_block_h=obh,
                            out_block_w=obw,
                            per_core_M=pm,
                            per_core_N=pn,
                            transpose_mcast=False,
                            fused_activation=None,
                            fuse_batch=True,
                        )
                        f = lambda pc=pc: ttnn.linear(
                            tx, tw, compute_kernel_config=ckc, program_config=pc, memory_config=DRAM
                        )
                        o = ttnn.to_torch(f()).float()
                        us2 = timed(mesh, f)
                        p = pcc(o, ref)
                        print(f"{tag0} {tag}: {us2:9.1f} us  PCC vs default {p:.6f}", flush=True)
                        if fid == "HiFi4" and not l1 and p > 0.9999:
                            good.append((us2, tag, (gx, gy, ib, sw, obh, obw, pm, pn)))
                    except Exception as e:  # noqa: BLE001
                        print(f"{tag0} {tag}: ERROR {err(e)}", flush=True)
                print(f"{tag0} BEST: {sorted(good)[:1] if good else None}", flush=True)
            ttnn.deallocate(tw)
            ttnn.deallocate(tx)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
