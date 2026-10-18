# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage-3 review #1 probe (P2-3): DSA prefill at 2048 rows, one die, synthetic tensors at the real shapes.

  --case bmm  the absorbed per-head matmuls q_abs = q @ W_UK [1,64,2048,256] x [1,64,256,512] and
              o = o_lat @ W_UV^T [1,64,2048,512] x [1,64,512,256] (bfp8 weights, HiFi4 as GlmDSA): ttnn default vs
              MatmulMultiCoreReuseProgramConfig with per_core_M = --pm (2 / 4 / 8; per_core_M = Mt hung the device in
              stage 3), full N per core. Run one per_core_M per process (a hang costs one timeout).
  --case sdpa ttnn.transformer.flash_mla_prefill(q [1,64,2048,512], latent [1,1,2048,512], head_dim_v 512, causal)
              without a program config (as tt/dsa.py) vs explicit SDPAProgramConfig q / k chunk sizes.
Host-timed (synchronized), PCC vs torch (bmm) or vs the default op (sdpa)."""

import argparse
import time

import torch

import ttnn
from models.autoports.zai_org_glm_5_3_flash.tests.harness import open_mesh, pcc


def _time(mesh, fn, n=10):
    ttnn.deallocate(fn())
    ttnn.synchronize_device(mesh)
    t = time.perf_counter()
    for _ in range(n):
        ttnn.deallocate(fn())
    ttnn.synchronize_device(mesh)
    return (time.perf_counter() - t) / n * 1e6


def err(e):
    return " | ".join(l.strip() for l in str(e).splitlines() if l.strip())[:300]


def bmm(mesh, grid, ckc, pms):
    g = torch.Generator().manual_seed(0)
    rows = 2048
    for name, K, N in (("q_abs", 256, 512), ("o_uv", 512, 256)):
        x = torch.randn(1, 64, rows, K, generator=g).bfloat16()
        w = (torch.randn(1, 64, K, N, generator=g) / K**0.5).bfloat16()
        tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)
        tw = ttnn.from_torch(w, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=mesh)
        ref = x.float() @ ttnn.to_torch(tw).float()
        Kt, Nt = K // 32, N // 32
        pcs = {"default": None}
        for pm in pms:
            pcs[f"reuse M{pm} N{Nt} in0 {min(Kt, 8)}"] = ttnn.MatmulMultiCoreReuseProgramConfig(
                compute_with_storage_grid_size=(grid.x, grid.y),
                in0_block_w=min(Kt, 8),
                out_subblock_h=1,
                out_subblock_w=max(d for d in (4, 2, 1) if Nt % d == 0),
                per_core_M=pm,
                per_core_N=Nt,
            )
        for tag, pc in pcs.items():
            f = lambda pc=pc: ttnn.matmul(
                tx, tw, program_config=pc, compute_kernel_config=ckc, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            print(f"{name} rows {rows} bfp8 {tag}: start", flush=True)
            try:
                out = ttnn.to_torch(f()).float()
                print(
                    f"{name} rows {rows} bfp8 {tag:24}: {_time(mesh, f):9.1f} us  PCC {pcc(out, ref):.6f}", flush=True
                )
            except Exception as e:  # noqa: BLE001
                print(f"{name} rows {rows} bfp8 {tag:24}: ERROR {err(e)}", flush=True)
        ttnn.deallocate(tx)
        ttnn.deallocate(tw)


def sdpa(mesh, grid, ckc):
    g = torch.Generator().manual_seed(1)
    n, H, D = 2048, 64, 512
    q = (torch.randn(1, H, n, D, generator=g) * 0.5).bfloat16()
    k = (torch.randn(1, 1, n, D, generator=g) * 0.5).bfloat16()
    tq = ttnn.from_torch(q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)
    tk = ttnn.from_torch(k, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)
    cands = {"default (no program config)": None}
    for qc in (32, 64, 128, 256):
        for kc in (64, 128, 256, 512):
            cands[f"q{qc} k{kc}"] = ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=(grid.x, grid.y), q_chunk_size=qc, k_chunk_size=kc, exp_approx_mode=False
            )
    ref = None
    for tag, pc in cands.items():
        f = lambda pc=pc: ttnn.transformer.flash_mla_prefill(
            tq,
            tk,
            D,
            is_causal=True,
            scale=256**-0.5,
            program_config=pc,
            compute_kernel_config=ckc,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        try:
            out = ttnn.to_torch(f()).float()
            if ref is None:
                ref = out
            print(
                f"flash_mla_prefill 2048 {tag:28}: {_time(mesh, f, 5):9.1f} us  PCC vs default {pcc(out, ref):.6f}",
                flush=True,
            )
        except Exception as e:  # noqa: BLE001
            print(f"flash_mla_prefill 2048 {tag:28}: ERROR {err(e)}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", choices=("bmm", "sdpa"), required=True)
    ap.add_argument("--pm", default="4")
    a = ap.parse_args()
    mesh = open_mesh()
    grid = mesh.compute_with_storage_grid_size()
    ckc = ttnn.init_device_compute_kernel_config(
        mesh.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )
    try:
        if a.case == "bmm":
            bmm(mesh, grid, ckc, [int(p) for p in a.pm.split(",")])
        else:
            sdpa(mesh, grid, ckc)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
