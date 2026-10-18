# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage-3 review #1 probe (P1-4, P2-3): DRAM-sharded retries and prefill program-config / fidelity candidates of the
largest dense matmuls, one die, synthetic weights at the real shapes (timing and op contracts; PCC against the
reference variant of the same weight).

  kda_in   KDA input projection  [32 | 2048, 4096] x [4096, 32960]
  kda_out  KDA output projection [32 | 2048, 8192] x [8192, 4096]
  dsa_o    DSA o_proj            [32 | 2048, 16384] x [16384, 4096]
  mlp_gu   dense MLP gate|up     [2048, 4096] x [4096, 24576]   (prefill only)
  mlp_dn   dense MLP down        [2048, 12288] x [12288, 4096]  (prefill only)

Sections (--only): kda_in_ds, out_ds, prefill_fid. Host-timed (synchronized, n calls after a warm-up). Every candidate
runs in try/except: an op-contract or L1 error is printed as the result of that candidate.

usage: python -m models.autoports.zai_org_glm_5_3_flash.tools.s3r_probe_dense [--only kda_in_ds,out_ds,prefill_fid]"""

import argparse
import time

import torch

import ttnn
from models.autoports.zai_org_glm_5_3_flash.tests.harness import open_mesh, pcc
from models.demos.blackhole.qwen36.tt import tp_common as tpc

FID = {"HiFi4": ttnn.MathFidelity.HiFi4, "HiFi2": ttnn.MathFidelity.HiFi2, "LoFi": ttnn.MathFidelity.LoFi}
DRAM = ttnn.DRAM_MEMORY_CONFIG
BYTES = {ttnn.bfloat8_b: 1088 / 1024, ttnn.bfloat4_b: 576 / 1024}


def _time(mesh, fn, n=50):
    ttnn.deallocate(fn())
    ttnn.synchronize_device(mesh)
    t = time.perf_counter()
    for _ in range(n):
        ttnn.deallocate(fn())
    ttnn.synchronize_device(mesh)
    return (time.perf_counter() - t) / n * 1e6


def ckc_of(mesh, fid, l1_acc=True):
    return ttnn.init_device_compute_kernel_config(
        mesh.arch(), math_fidelity=FID[fid], math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=l1_acc
    )


def report(mesh, tag, fn, ref, N, wbytes, n=50):
    try:
        out = ttnn.to_torch(fn()).float()[..., :N]
        us = _time(mesh, fn, n)
        p = pcc(out, ref) if ref is not None else float("nan")
        print(f"{tag:58}: {us:9.1f} us {wbytes / us / 1e3:6.0f} GB/s  PCC {p:.6f}", flush=True)
        return out, us
    except Exception as e:  # noqa: BLE001
        msg = " | ".join(l.strip() for l in str(e).splitlines() if l.strip() and "backtrace" not in l.lower())[:400]
        print(f"{tag:58}: ERROR {msg}", flush=True)
        return None, None


def ds_pc(k_tiles, n_tiles, cores, in0, readers):
    pc = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
        in0_block_w=in0, per_core_M=1, per_core_N=n_tiles // cores, fused_activation=None
    )
    if readers > 1:
        pc.num_workers_per_dram_bank = readers
    return pc


def act_shard(K, cores):
    rows = next(r for r in range(8, 0, -1) if cores % r == 0 and cores // r <= 8)
    return ttnn.create_sharded_memory_config(
        shape=(32, K // cores),
        core_grid=ttnn.CoreGrid(x=cores // rows, y=rows),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def ds_linear(tx, tw_ds, pc, act, ckc):
    xs = ttnn.to_memory_config(tx, act)
    o = ttnn.linear(
        xs, tw_ds, compute_kernel_config=ckc, program_config=pc, memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG
    )
    ttnn.deallocate(xs)
    o2 = ttnn.to_memory_config(o, DRAM)
    ttnn.deallocate(o)
    return o2


def kda_in_ds(mesh, grid):
    """KDA in-projection, decode rows: 1D baselines, DRAM-sharded at the natural N (1032 tiles = 8 banks x 129: gcd with
    K 128 tiles allows 8 compute cores), DRAM-sharded on an N padded to 1056 tiles (8 x 132, 32 cores), and an N split
    (first 1024 tiles DRAM-sharded + last 6 tiles 1D, concatenated)."""
    K, N = 4096, 32960
    g = torch.Generator().manual_seed(1)
    x = torch.randn(1, 1, 32, K, generator=g).bfloat16()
    w = (torch.randn(1, 1, K, N, generator=g) / K**0.5).bfloat16()
    tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)
    for dt, fids in ((ttnn.bfloat8_b, ("HiFi4", "HiFi2")), (ttnn.bfloat4_b, ("HiFi4", "LoFi"))):
        tw = ttnn.from_torch(w, dtype=dt, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=DRAM)
        wq = ttnn.to_torch(tw).float()
        ref = x.float() @ wq
        wb = K * N * BYTES[dt]
        dname = str(dt).split(".")[-1]
        for fid in fids:
            ckc = ckc_of(mesh, fid)
            report(
                mesh,
                f"kda_in {dname} {fid} default (interleaved)",
                lambda: ttnn.linear(tx, tw, compute_kernel_config=ckc),
                ref,
                N,
                wb,
            )
            for cores in (64, grid.x * grid.y):
                pc1 = tpc.create_matmul_1d_decode_progcfg(32, K, N, cores, fp32_acc=True, grid_w=grid.x)
                report(
                    mesh,
                    f"kda_in {dname} {fid} 1d c{cores} in0 {pc1.in0_block_w}",
                    lambda pc1=pc1: ttnn.linear(tx, tw, compute_kernel_config=ckc, program_config=pc1),
                    ref,
                    N,
                    wb,
                )
            # (a) natural N: weight padded by the DRAM-sharded memory config to 1032 tiles
            try:
                tw_ds = ttnn.to_memory_config(tw, tpc.create_dram_sharded_mem_config(K, N))
            except Exception as e:  # noqa: BLE001
                print(f"kda_in {dname}: DS weight (natural N) ERROR {str(e).splitlines()[0][:200]}")
                tw_ds = None
            if tw_ds is not None:
                nt = tpc._roundup(N, 32 * tpc.DRAM_CORES) // 32  # 1032
                for cores in (8,):
                    for in0 in (8, 4, 2, 1):
                        for readers in (1, 3):
                            pc = ds_pc(128, nt, cores, in0, readers)
                            act = act_shard(K, cores)
                            report(
                                mesh,
                                f"kda_in {dname} {fid} DS natural c{cores} in0 {in0} r{readers}",
                                lambda pc=pc, act=act: ds_linear(tx, tw_ds, pc, act, ckc),
                                ref,
                                N,
                                wb,
                            )
                ttnn.deallocate(tw_ds)
            # (b) N padded to 1056 tiles (8 banks x 132): 32 compute cores
            Np = 1056 * 32
            wp = torch.nn.functional.pad(wq, (0, Np - N)).bfloat16()
            try:
                tw_p = ttnn.from_torch(
                    wp,
                    dtype=dt,
                    layout=ttnn.TILE_LAYOUT,
                    device=mesh,
                    memory_config=tpc.create_dram_sharded_mem_config(K, Np),
                )
            except Exception as e:  # noqa: BLE001
                print(f"kda_in {dname}: DS weight (padded N) ERROR {str(e).splitlines()[0][:200]}")
                tw_p = None
            if tw_p is not None:
                for cores in (32,):
                    for in0 in (4, 2, 1):
                        for readers in (1, 2, 3):
                            pc = ds_pc(128, 1056, cores, in0, readers)
                            act = act_shard(K, cores)
                            report(
                                mesh,
                                f"kda_in {dname} {fid} DS padN1056 c{cores} in0 {in0} r{readers}",
                                lambda pc=pc, act=act: ds_linear(tx, tw_p, pc, act, ckc),
                                ref,
                                N,
                                wb * 1056 / 1030,
                            )
                ttnn.deallocate(tw_p)
            # (c) N split: 32768 columns DRAM-sharded (128 tiles / bank) + 192 columns 1D, concatenated
            n1 = 32768
            tw1 = ttnn.from_torch(
                wq[..., :n1].bfloat16(),
                dtype=dt,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                memory_config=tpc.create_dram_sharded_mem_config(K, n1),
            )
            tw2 = ttnn.from_torch(
                wq[..., n1:].bfloat16(), dtype=dt, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=DRAM
            )
            pc2 = tpc.create_matmul_1d_decode_progcfg(32, K, N - n1, 6, fp32_acc=True, grid_w=grid.x)
            for cores in (32, 64):
                for in0 in (4, 2, 1):
                    if (128 // cores) % in0:
                        continue
                    for readers in (1, 2):
                        pc = ds_pc(128, 1024, cores, in0, readers)
                        act = act_shard(K, cores)

                        def split(pc=pc, act=act):
                            a = ds_linear(tx, tw1, pc, act, ckc)
                            b = ttnn.linear(tx, tw2, compute_kernel_config=ckc, program_config=pc2, memory_config=DRAM)
                            o = ttnn.concat([a, b], dim=-1, memory_config=DRAM)
                            ttnn.deallocate(a)
                            ttnn.deallocate(b)
                            return o

                        report(
                            mesh,
                            f"kda_in {dname} {fid} split DS c{cores} in0 {in0} r{readers} + 1D",
                            lambda split=split: split(),
                            ref,
                            N,
                            wb,
                        )
                        report(
                            mesh,
                            f"kda_in {dname} {fid}   (DS part only)",
                            lambda pc=pc, act=act: ds_linear(tx, tw1, pc, act, ckc),
                            ref[..., :n1],
                            n1,
                            wb * n1 / N,
                        )
            ttnn.deallocate(tw1)
            ttnn.deallocate(tw2)
        ttnn.deallocate(tw)


def out_ds(mesh, grid):
    """KDA out / DSA o_proj: decode DRAM-sharded (cores 32 / 64, readers 1 / 2 / 4) vs 1D, and prefill 2048 rows on the
    SAME DRAM width-sharded weight (2D multicast config, tt_transformers / DSMatmul.prefill pattern) vs the interleaved
    weight (ttnn default and explicit 2D)."""
    for role, K, N in (("kda_out", 8192, 4096), ("dsa_o", 16384, 4096)):
        g = torch.Generator().manual_seed(2)
        x = torch.randn(1, 1, 32, K, generator=g).bfloat16()
        xp = torch.randn(1, 1, 2048, K, generator=g).bfloat16()
        w = (torch.randn(1, 1, K, N, generator=g) / K**0.5).bfloat16()
        tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)
        txp = ttnn.from_torch(xp, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)
        kt, nt = K // 32, N // 32
        for dt, fids in ((ttnn.bfloat8_b, ("HiFi4", "LoFi")), (ttnn.bfloat4_b, ("HiFi4", "LoFi"))):
            dname = str(dt).split(".")[-1]
            tw = ttnn.from_torch(w, dtype=dt, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=DRAM)
            tw_ds = ttnn.to_memory_config(tw, tpc.create_dram_sharded_mem_config(K, N))
            wq = ttnn.to_torch(tw).float()
            ref = x.float() @ wq
            wb = K * N * BYTES[dt]
            for fid in fids:
                ckc = ckc_of(mesh, fid)
                for cores in (64, 96, grid.x * grid.y):
                    pc1 = tpc.create_matmul_1d_decode_progcfg(32, K, N, cores, fp32_acc=True, grid_w=grid.x)
                    report(
                        mesh,
                        f"{role} {dname} {fid} decode 1d c{cores}",
                        lambda pc1=pc1: ttnn.linear(tx, tw, compute_kernel_config=ckc, program_config=pc1),
                        ref,
                        N,
                        wb,
                    )
                for cores in (32, 64):
                    for readers in (1, 2, 4):
                        in0 = tpc._find_largest_divisor(kt // cores)
                        pc = ds_pc(kt, nt, cores, in0, readers)
                        act = act_shard(K, cores)
                        report(
                            mesh,
                            f"{role} {dname} {fid} decode DS c{cores} in0 {in0} r{readers}",
                            lambda pc=pc, act=act: ds_linear(tx, tw_ds, pc, act, ckc),
                            ref,
                            N,
                            wb,
                        )
            # prefill 2048 rows at HiFi4 (the KDA / DSA prefill fidelity), and LoFi for bfp4
            ckc = ckc_of(mesh, "HiFi4", l1_acc=False)
            out_il, _ = report(
                mesh,
                f"{role} {dname} HiFi4 prefill2048 interleaved default",
                lambda: ttnn.linear(txp, tw, compute_kernel_config=ckc, memory_config=DRAM),
                None,
                N,
                wb,
                n=10,
            )
            pc2d = tpc.create_prefill_matmul_program_config(2048, K, N)
            refp = out_il
            for tag, ww in (("interleaved 2D", tw), ("DRAM-sharded 2D", tw_ds)):
                o, _ = report(
                    mesh,
                    f"{role} {dname} HiFi4 prefill2048 {tag}",
                    lambda ww=ww: ttnn.linear(
                        txp, ww, compute_kernel_config=ckc, program_config=pc2d, memory_config=DRAM
                    ),
                    refp,
                    N,
                    wb,
                    n=10,
                )
            for gs in ((8, 8), (11, 10), (8, 10)):
                try:
                    pcg = tpc.create_prefill_matmul_program_config(2048, K, N, grid_size=gs)
                except Exception as e:  # noqa: BLE001
                    print(f"{role} pc grid {gs}: {e}")
                    continue
                report(
                    mesh,
                    f"{role} {dname} HiFi4 prefill2048 DRAM-sharded 2D grid{gs}",
                    lambda pcg=pcg: ttnn.linear(
                        txp, tw_ds, compute_kernel_config=ckc, program_config=pcg, memory_config=DRAM
                    ),
                    refp,
                    N,
                    wb,
                    n=10,
                )
            ttnn.deallocate(tw)
            ttnn.deallocate(tw_ds)
        ttnn.deallocate(tx)
        ttnn.deallocate(txp)


def prefill_fid(mesh, grid):
    """Large prefill matmuls (2048 rows, bfp8 weights): HiFi4 (current) vs HiFi2 / LoFi, ttnn default vs explicit 2D
    config; PCC against HiFi4 default."""
    for role, K, N in (
        ("kda_in", 4096, 32960),
        ("mlp_gu", 4096, 24576),
        ("mlp_dn", 12288, 4096),
        ("dsa_o", 16384, 4096),
    ):
        g = torch.Generator().manual_seed(3)
        xp = torch.randn(1, 1, 2048, K, generator=g).bfloat16()
        w = (torch.randn(1, 1, K, N, generator=g) / K**0.5).bfloat16()
        txp = ttnn.from_torch(xp, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)
        tw = ttnn.from_torch(w, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=DRAM)
        wb = K * N * BYTES[ttnn.bfloat8_b]
        ref = None
        for fid in ("HiFi4", "HiFi2", "LoFi"):
            ckc = ckc_of(mesh, fid, l1_acc=False)
            o, _ = report(
                mesh,
                f"{role} bfp8 {fid} prefill2048 default",
                lambda: ttnn.linear(txp, tw, compute_kernel_config=ckc, memory_config=DRAM),
                ref,
                N,
                wb,
                n=5,
            )
            if ref is None:
                ref = o
            pc2d = tpc.create_prefill_matmul_program_config(2048, K, N, grid_size=(grid.x, min(grid.y, 10)))
            report(
                mesh,
                f"{role} bfp8 {fid} prefill2048 2D {grid.x}x{min(grid.y, 10)} in0 {pc2d.in0_block_w}",
                lambda pc2d=pc2d: ttnn.linear(
                    txp, tw, compute_kernel_config=ckc, program_config=pc2d, memory_config=DRAM
                ),
                ref,
                N,
                wb,
                n=5,
            )
        ttnn.deallocate(tw)
        ttnn.deallocate(txp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="kda_in_ds,out_ds,prefill_fid")
    a = ap.parse_args()
    mesh = open_mesh()
    grid = mesh.compute_with_storage_grid_size()
    print("grid", grid.x, grid.y, flush=True)
    try:
        for sec in a.only.split(","):
            print(f"===== {sec}", flush=True)
            {"kda_in_ds": kda_in_ds, "out_ds": out_ds, "prefill_fid": prefill_fid}[sec](mesh, grid)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
