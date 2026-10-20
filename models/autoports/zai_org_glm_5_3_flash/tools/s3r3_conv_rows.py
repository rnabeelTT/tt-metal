# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage-3 review #3 P3: the KDA decode conv step on one die (W = 24576, P = 32960) at decode batches 1 / 8 / 16 / 32:
the fused op ttnn.experimental.deepseek_prefill.kda_decode_conv vs the fused decoder's 12-op chain (slice, multiply,
3 x typecast + addcmul, SiLU, select, 3 in-place shift selects; tt/fused_decoder.FusedGlmKDA._conv_step with
kda_decode_fold). Each variant captured as a trace and replayed 50 times (device bound); us per call. Picks
kda_conv_op_max_rows.

usage: python -m models.autoports.zai_org_glm_5_3_flash.tools.s3r3_conv_rows"""

import time

import torch

import ttnn
from models.autoports.zai_org_glm_5_3_flash.tests.harness import open_mesh

DRAM = ttnn.DRAM_MEMORY_CONFIG
W, P = 24576, 32960


def traced(mesh, fn, n=50):
    fn()
    ttnn.synchronize_device(mesh)
    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
    fn()
    ttnn.end_trace_capture(mesh, tid, cq_id=0)
    ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
    t = time.perf_counter()
    for _ in range(n):
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh)
    us = (time.perf_counter() - t) / n * 1e6
    ttnn.release_trace(mesh, tid)
    return us


def main():
    mesh = open_mesh()
    try:
        g = torch.Generator().manual_seed(0)
        dev = lambda t, dt: ttnn.from_torch(t, dtype=dt, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=DRAM)
        taps = [dev(torch.randn(1, 1, 1, W, generator=g) * 0.5, ttnn.float32) for _ in range(4)]
        for B in (1, 8, 16, 32):
            proj = dev(torch.randn(1, 1, B, P, generator=g), ttnn.bfloat16)
            hist = [dev(torch.randn(1, 1, B, W, generator=g), ttnn.bfloat16) for _ in range(3)]
            act = torch.ones(1, 1, B, 1)
            active, active_bf = dev(act, ttnn.float32), dev(act, ttnn.bfloat16)

            def op():
                q = ttnn.experimental.deepseek_prefill.kda_decode_conv(proj, *hist, *taps, active, W)
                ttnn.deallocate(q)

            def chain():
                cur = ttnn.slice(proj, (0, 0, 0, 0), (1, 1, B, W), memory_config=DRAM)
                acc = ttnn.multiply(cur, taps[3], dtype=ttnn.float32, memory_config=DRAM)
                for j in range(3):
                    h32 = ttnn.typecast(hist[j], ttnn.float32)
                    nxt = ttnn.addcmul(acc, h32, taps[j], memory_config=DRAM)
                    ttnn.deallocate(h32)
                    ttnn.deallocate(acc)
                    acc = nxt
                qkv = ttnn.silu(acc, memory_config=DRAM)
                q = ttnn.where(active, qkv, 0.0, memory_config=DRAM)
                ttnn.deallocate(qkv)
                ttnn.deallocate(acc)
                h0, h1, h2 = hist
                ttnn.where(active_bf, h1, h0, output_tensor=h0)
                ttnn.where(active_bf, h2, h1, output_tensor=h1)
                ttnn.where(active_bf, cur, h2, output_tensor=h2)
                ttnn.deallocate(cur)
                ttnn.deallocate(q)

            t_op, t_chain = traced(mesh, op), traced(mesh, chain)
            print(
                f"B={B:2d}: kda_decode_conv {t_op:7.1f} us, 12-op chain {t_chain:7.1f} us -> "
                f"{'op' if t_op < t_chain else 'chain'}",
                flush=True,
            )
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
