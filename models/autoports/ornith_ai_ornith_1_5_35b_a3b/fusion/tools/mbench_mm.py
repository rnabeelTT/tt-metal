"""Op microbenchmark of the decode projections at one row, one chip, Ornith TP4 per-device shapes (random weights):
the shipped 1D mcast configs (interleaved weight, L1-interleaved activation) vs DRAM-sharded (weight width-sharded over
the 8 banks, activation width-sharded in L1, 1..3 reader cores per bank).

Each variant is traced as REPS back-to-back calls and replayed; compare variants with each other.

    python mbench_mm.py [rows]
"""

import sys
import time

import torch

import ttnn
from models.demos.blackhole.qwen36.tt import tp_common as tpc

REPS, ITERS = 50, 10
# name, K, N, weight dtype, 1D cores, 1D grid_w (None = 8)
SHAPES = [
    ("gdn in-proj", 2048, 3136, ttnn.bfloat8_b, 44, 11),
    ("attn qkv", 2048, 2560, ttnn.bfloat8_b, 64, None),
    ("out-proj / wo", 1024, 2048, ttnn.bfloat8_b, 33, 11),
    ("router", 2048, 288, ttnn.bfloat16, 8, 11),
]


def ckc(dev):
    return ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )


def bench(dev, label, fn, mb):
    out = fn()  # compile
    ttnn.deallocate(out)
    tid = ttnn.begin_trace_capture(dev, cq_id=0)
    for _ in range(REPS):
        ttnn.deallocate(fn())
    ttnn.end_trace_capture(dev, tid, cq_id=0)
    ttnn.execute_trace(dev, tid, cq_id=0, blocking=True)
    t0 = time.time()
    for _ in range(ITERS):
        ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(dev)
    us = (time.time() - t0) / (ITERS * REPS) * 1e6
    ttnn.release_trace(dev, tid)
    print(f"  {label:52s} {us:7.1f} us/call  {mb / us * 1e3:6.0f} GB/s", flush=True)


def act_sharded(k, cores):
    cols = min(cores, 8)
    rows = cores // cols
    return ttnn.create_sharded_memory_config(
        shape=(32, k // cores), core_grid=ttnn.CoreGrid(x=cols, y=rows), strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR, use_height_and_width_as_shard_shape=True,
    )


def main():
    rows = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    dev = ttnn.open_device(device_id=0, trace_region_size=64 << 20)
    c = ckc(dev)
    try:
        for name, K, N, wdt, cores1d, gw in SHAPES:
            tb = {ttnn.bfloat8_b: 1088, ttnn.bfloat16: 2048}[wdt]
            mb = K // 32 * (-(-N // 32)) * tb / 1e6
            print(f"{name}: [{rows}, {K}] x [{K}, {N}] {wdt} ({mb:.2f} MB)", flush=True)
            w = torch.randn(K, N) * 0.02
            x_t = torch.randn(1, 1, 32, K)
            x_il = ttnn.from_torch(x_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                                   memory_config=ttnn.L1_MEMORY_CONFIG)
            w_il = ttnn.from_torch(w, dtype=wdt, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            kw = {"grid_w": gw} if gw else {}
            pc = tpc.create_matmul_1d_decode_progcfg(32, K, N, cores1d, **kw)
            bench(dev, f"1d mcast {cores1d} cores (shipped)",
                  lambda: ttnn.linear(x_il, w_il, compute_kernel_config=c, program_config=pc,
                                      memory_config=ttnn.L1_MEMORY_CONFIG), mb)
            for alt in (16, 32, 48, 64, 80):
                if alt == cores1d:
                    continue
                pc_a = tpc.create_matmul_1d_decode_progcfg(32, K, N, alt, grid_w=11)
                try:
                    bench(dev, f"1d mcast {alt} cores (grid_w 11)",
                          lambda: ttnn.linear(x_il, w_il, compute_kernel_config=c, program_config=pc_a,
                                              memory_config=ttnn.L1_MEMORY_CONFIG), mb)
                except Exception as e:
                    print(f"  1d {alt}: {str(e).splitlines()[0][:80]}")
            w_ds = ttnn.from_torch(w, dtype=wdt, layout=ttnn.TILE_LAYOUT, device=dev,
                                   memory_config=tpc.create_dram_sharded_mem_config(K, N))
            for acores in (16, 32):
                if (K // 32) % acores:
                    continue
                x_sh = ttnn.to_memory_config(x_il, act_sharded(K, acores))
                for readers in (1, 2, 3):
                    try:
                        pc_d = tpc.create_dram_sharded_matmul_program_config(32, K, N, num_cores=acores, readers=readers)
                        if readers > 1 and getattr(pc_d, "num_workers_per_dram_bank", 1) != readers:
                            continue
                        width = ttnn.create_sharded_memory_config(
                            shape=(32, pc_d.per_core_N * 32), core_grid=ttnn.CoreGrid(x=min(acores, 8), y=acores // min(acores, 8)),
                            strategy=ttnn.ShardStrategy.WIDTH, orientation=ttnn.ShardOrientation.ROW_MAJOR,
                            use_height_and_width_as_shard_shape=True,
                        )
                        bench(dev, f"dram-sharded act {acores} cores, {readers} reader(s)/bank",
                              lambda: ttnn.linear(x_sh, w_ds, compute_kernel_config=c, program_config=pc_d,
                                                  memory_config=width), mb)
                    except Exception as e:
                        print(f"  dram-sharded {acores}/{readers}: {str(e).splitlines()[0][:100]}")
                ttnn.deallocate(x_sh)
            for t in (x_il, w_il, w_ds):
                ttnn.deallocate(t)
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
