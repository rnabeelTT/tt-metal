# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage-3 probe: program config x fidelity x weight dtype sweep of the dominant dense decode matmuls (32 activation
rows = one tile of users): KDA input projection 4096 x 32960, KDA output projection 8192 x 4096, shared expert / MLP
4096 x 4096 and 2048 x 4096. Candidates: ttnn default, explicit 1D mcast_in0 grids (interleaved weight) and the
DRAM-sharded decode kernel (width-sharded weight, 1 / 2 readers per bank). Host-timed (synchronized, 50 calls); PCC vs
torch on the read-back device weight."""

import time

import torch

import ttnn
from models.autoports.zai_org_glm_5_3_flash.tests.harness import open_mesh, pcc
from models.demos.blackhole.qwen36.tt import tp_common as tpc

SHAPES = {"kda_in": (4096, 32960), "kda_out": (8192, 4096), "sq_4096": (4096, 4096), "down_2048": (2048, 4096)}
FID = {"HiFi4": ttnn.MathFidelity.HiFi4, "HiFi2": ttnn.MathFidelity.HiFi2, "LoFi": ttnn.MathFidelity.LoFi}


def _time(mesh, fn, n=50):
    ttnn.deallocate(fn())
    ttnn.synchronize_device(mesh)
    t = time.perf_counter()
    for _ in range(n):
        ttnn.deallocate(fn())
    ttnn.synchronize_device(mesh)
    return (time.perf_counter() - t) / n * 1e6


def main():
    mesh = open_mesh()
    grid = mesh.compute_with_storage_grid_size()
    max_cores = grid.x * grid.y
    print("grid", grid.x, grid.y)
    g = torch.Generator().manual_seed(0)
    try:
        for role, (K, N) in SHAPES.items():
            x = torch.randn(1, 1, 32, K, generator=g).bfloat16()
            w = (torch.randn(1, 1, K, N, generator=g) / K**0.5).bfloat16()
            tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)
            for dt in (ttnn.bfloat8_b, ttnn.bfloat4_b):
                tw = ttnn.from_torch(w, dtype=dt, layout=ttnn.TILE_LAYOUT, device=mesh)
                ref = x.float() @ ttnn.to_torch(tw).float()
                tw_ds = None
                try:
                    tw_ds = ttnn.to_memory_config(tw, tpc.create_dram_sharded_mem_config(K, N))
                except Exception as e:  # noqa: BLE001
                    print(f"{role} {dt}: DS weight error {str(e).splitlines()[0][:100]}")
                for fname, fid in FID.items():
                    ckc = ttnn.init_device_compute_kernel_config(
                        mesh.arch(), math_fidelity=fid, fp32_dest_acc_en=True, packer_l1_acc=True
                    )
                    cands = {"default": lambda: ttnn.linear(tx, tw, compute_kernel_config=ckc)}
                    for cores in (32, 64, max_cores):
                        pc = tpc.create_matmul_1d_decode_progcfg(32, K, N, cores, fp32_acc=True, grid_w=grid.x)
                        cands[f"1d c{cores}"] = lambda pc=pc: ttnn.linear(
                            tx, tw, compute_kernel_config=ckc, program_config=pc
                        )
                    if tw_ds is not None:
                        for readers in (1, 2):
                            try:
                                pc = tpc.create_dram_sharded_matmul_program_config(32, K, N, readers=readers)
                                k_tiles = K // 32
                                rows, cols = tpc._find_grid(k_tiles)
                                act = ttnn.create_sharded_memory_config(
                                    shape=(32, K // (rows * cols)),
                                    core_grid=ttnn.CoreGrid(x=cols, y=rows),
                                    strategy=ttnn.ShardStrategy.WIDTH,
                                    orientation=ttnn.ShardOrientation.ROW_MAJOR,
                                    use_height_and_width_as_shard_shape=True,
                                )
                                tx_sh = ttnn.to_memory_config(tx, act)

                                def ds(pc=pc, tx_sh=tx_sh):
                                    o = ttnn.linear(
                                        tx_sh,
                                        tw_ds,
                                        compute_kernel_config=ckc,
                                        program_config=pc,
                                        memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
                                    )
                                    return o

                                cands[f"dram_sharded r{readers}"] = ds
                            except Exception as e:  # noqa: BLE001
                                print(f"{role} {dt} DS r{readers}: setup error {str(e).splitlines()[0][:100]}")
                    for cname, fn in cands.items():
                        try:
                            out = ttnn.to_torch(fn()).float()[..., :N]
                            us = _time(mesh, fn)
                            gbs = K * N * (1.0625 if dt == ttnn.bfloat8_b else 0.5625) / us / 1e3
                            print(
                                f"{role:9} {str(dt)[9:]:10} {fname:5} {cname:16}: {us:7.1f} us {gbs:6.0f} GB/s  PCC {pcc(out, ref):.6f}"
                            )
                        except Exception as e:  # noqa: BLE001
                            print(
                                f"{role:9} {str(dt)[9:]:10} {fname:5} {cname:16}: error {str(e).splitlines()[0][:110]}"
                            )
                ttnn.deallocate(tw)
                if tw_ds is not None:
                    ttnn.deallocate(tw_ds)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
