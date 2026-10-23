"""Op microbenchmark of MoE decode variants at one token, one chip, Ornith TP4 per-device shapes (random weights).

Each variant is traced as REPS back-to-back calls and replayed; the time per call includes the trace's per-op
dispatch, so compare variants with each other, not with the layer profile.

    python mbench_moe.py [router] [gate_up]
"""

import sys
import time

import torch

import ttnn
from models.autoports.ornith_ai_ornith_1_5_35b_a3b.tt.moe import sparse_matmul_program_config
from models.demos.blackhole.qwen36.tt import tp_common as tpc

H, E, K, NL = 2048, 256, 8, 128  # hidden, experts, top-k, expert intermediate per device (TP4)
REPS, ITERS = 50, 10


def ckc(dev, fid, fp32=True):
    return ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=fid, math_approx_mode=False, fp32_dest_acc_en=fp32, packer_l1_acc=False
    )


def bench(dev, label, fn):
    out = fn()  # compile
    tid = ttnn.begin_trace_capture(dev, cq_id=0)
    for _ in range(REPS):
        out = fn()
    ttnn.end_trace_capture(dev, tid, cq_id=0)
    ttnn.execute_trace(dev, tid, cq_id=0, blocking=True)
    t0 = time.time()
    for _ in range(ITERS):
        ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(dev)
    us = (time.time() - t0) / (ITERS * REPS) * 1e6
    ttnn.release_trace(dev, tid)
    print(f"  {label:58s} {us:7.1f} us/call")


def router(dev):
    print("router [32, 2048] x [2048, 256] bf16")
    w = torch.randn(H, E) * 0.02
    x = ttnn.from_torch(torch.randn(1, 1, 32, H), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                        memory_config=ttnn.L1_MEMORY_CONFIG)
    w_il = ttnn.from_torch(w, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    w_ds = ttnn.from_torch(w, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                           memory_config=tpc.create_dram_sharded_mem_config(H, E))
    act = tpc.create_activation_shard_config(H)
    grid_w = dev.compute_with_storage_grid_size().x
    for fname, fid in (("HiFi4", ttnn.MathFidelity.HiFi4), ("HiFi2", ttnn.MathFidelity.HiFi2)):
        c = ckc(dev, fid)
        pc_ds = tpc.create_dram_sharded_matmul_program_config(32, H, E)
        bench(dev, f"dram-sharded (shipped path, reshard in/out) {fname}",
              lambda: tpc.sharded_decode_matmul(x, w_ds, c, pc_ds, act, None, H, decode_out_memory_config=ttnn.L1_MEMORY_CONFIG))
        for cores in (8,):
            pc = tpc.create_matmul_1d_decode_progcfg(32, H, E, cores, fp32_acc=True, grid_w=grid_w)
            bench(dev, f"1d mcast {cores} cores interleaved {fname}",
                  lambda: ttnn.linear(x, w_il, compute_kernel_config=c, program_config=pc, memory_config=ttnn.L1_MEMORY_CONFIG))
            for ib in (4, 16):
                pc2 = tpc.create_matmul_1d_decode_progcfg(32, H, E, cores, fp32_acc=True, grid_w=grid_w, in0_block_w=ib)
                bench(dev, f"1d mcast {cores} cores interleaved in0_block_w {ib} {fname}",
                      lambda: ttnn.linear(x, w_il, compute_kernel_config=c, program_config=pc2, memory_config=ttnn.L1_MEMORY_CONFIG))
        bench(dev, f"ttnn auto interleaved {fname}",
              lambda: ttnn.linear(x, w_il, compute_kernel_config=c, memory_config=ttnn.L1_MEMORY_CONFIG))


def gate_up(dev):
    print("routed gate|up sparse_matmul, 8 of 256 experts, [2048, 256] bfp8 each")
    x = ttnn.from_torch(torch.randn(1, 1, 1, H), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                        memory_config=ttnn.L1_MEMORY_CONFIG)
    wt = torch.randn(1, E, H, 2 * NL) * 0.02
    ones = ttnn.from_torch(torch.ones(1, 1, 1, E), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
    ids = ttnn.from_torch(torch.randperm(E)[:K].reshape(1, 1, 1, K).int(), dtype=ttnn.uint16,
                          layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, memory_config=ttnn.L1_MEMORY_CONFIG)
    c = ckc(dev, ttnn.MathFidelity.HiFi2, fp32=False)
    for tile_w in (32, 16):
        w = ttnn.from_torch(wt, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=dev,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG, tile=ttnn.Tile([32, tile_w]))
        out_tile = ttnn.Tile([32, tile_w])
        for cores, in0 in ((16, 32), (16, 64), (16, 16), (32, 32)):
            pc = sparse_matmul_program_config(1, 2 * NL, H, target_cores=cores, in0_block_w_cap=in0, tile_w=tile_w)
            ncores = pc.compute_with_storage_grid_size.x * pc.compute_with_storage_grid_size.y
            bench(dev, f"tile_w {tile_w} cores {ncores} in0_block_w {pc.in0_block_w}",
                  lambda: ttnn.sparse_matmul(x, w, sparsity=ones, indices=ids, is_input_a_sparse=False,
                                             is_input_b_sparse=True, memory_config=ttnn.L1_MEMORY_CONFIG,
                                             output_tile=out_tile, program_config=pc, compute_kernel_config=c,
                                             dtype=ttnn.bfloat16))
        ttnn.deallocate(w)


def main():
    what = sys.argv[1:] or ["router", "gate_up"]
    dev = ttnn.open_device(device_id=0, trace_region_size=64 << 20)
    try:
        for w in what:
            globals()[w](dev)
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
