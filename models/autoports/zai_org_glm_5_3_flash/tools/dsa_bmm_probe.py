# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage-3 probe: the DSA absorbed per-head decode matmuls (q_abs = q @ W_UK [1,64,rows,256] x [1,64,256,512] and
o = o_lat @ W_UV^T [1,64,rows,512] x [1,64,512,256]) with ttnn's default program config vs explicit
MatmulMultiCoreReuseProgramConfig grids, bf16 vs bfp8 weights. Host-timed (synchronized, 50 calls), PCC vs torch."""

import time

import torch

import ttnn
from models.autoports.zai_org_glm_5_3_flash.tests.harness import open_mesh, pcc


def _time(fn, n=50):
    fn()
    ttnn.synchronize_device(_time.mesh)
    t = time.perf_counter()
    for _ in range(n):
        o = fn()
        ttnn.deallocate(o)
    ttnn.synchronize_device(_time.mesh)
    return (time.perf_counter() - t) / n * 1e6


def main():
    mesh = open_mesh()
    _time.mesh = mesh
    grid = mesh.compute_with_storage_grid_size()
    print("grid", grid.x, grid.y)
    ckc = ttnn.init_device_compute_kernel_config(
        mesh.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True
    )
    g = torch.Generator().manual_seed(0)
    try:
        for name, K, N in (("q_abs", 256, 512), ("o_uv", 512, 256)):
            for rows in (32, 1024):
                x = torch.randn(1, 64, rows, K, generator=g).bfloat16()
                w = (torch.randn(1, 64, K, N, generator=g) / K**0.5).bfloat16()
                tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)
                Mt, Kt, Nt = rows // 32, K // 32, N // 32
                for dt in (ttnn.bfloat16, ttnn.bfloat8_b):
                    tw = ttnn.from_torch(w, dtype=dt, layout=ttnn.TILE_LAYOUT, device=mesh)
                    ref = x.float() @ ttnn.to_torch(tw).float()
                    pcs = {"default": None}
                    for pm in sorted({1, min(Mt, 4), Mt}):
                        for pn in (Nt, Nt // 2, Nt // 4):
                            sw = max(d for d in (4, 2, 1) if pn % d == 0)
                            pcs[f"reuse M{pm} N{pn}"] = ttnn.MatmulMultiCoreReuseProgramConfig(
                                compute_with_storage_grid_size=(grid.x, grid.y),
                                in0_block_w=min(Kt, 8),
                                out_subblock_h=1,
                                out_subblock_w=sw,
                                per_core_M=pm,
                                per_core_N=pn,
                            )
                    for pn_name, pc in pcs.items():
                        try:
                            f = lambda: ttnn.matmul(
                                tx,
                                tw,
                                program_config=pc,
                                compute_kernel_config=ckc,
                                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                            )
                            out = ttnn.to_torch(f()).float()
                            print(
                                f"{name} rows {rows:5} {str(dt):22} {pn_name:16}: {_time(f):8.1f} us  PCC {pcc(out, ref):.6f}"
                            )
                        except Exception as e:  # noqa: BLE001
                            print(
                                f"{name} rows {rows:5} {str(dt):22} {pn_name:16}: error {str(e).splitlines()[0][:110]}"
                            )
                    ttnn.deallocate(tw)
                ttnn.deallocate(tx)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
