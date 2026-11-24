# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""tt/moe.py _stream_defaults: the qwen36 fused MoE decode with Hy4's streaming knobs (QWEN35_MOE_ACT_BLK,
QWEN35_MOE_DN_CHUNK, QWEN35_MOE_GU_FARW, QWEN35_MOE_GUR_READERS, QWEN35_MOE_GU1B*, QWEN35_MOE_GUDN) vs the qwen36 defaults, on one chip at Hy4's per-die
shapes (latent 1536, expert width 384, 512 experts, top-16, bfp4, routed only, "keys" router): the output must be
bit-identical at 1, 4, 8, 16, 24 and 32 rows (random weights, random distinct routing per row). Prints the per-call time
of each (trace replay). usage: python -m models.autoports.tencent_hy4_flash.tests.test_moe_stream"""

import os
import time

import torch
import ttnn

from models.demos.blackhole.qwen36.tt.moe_decode import experts as X
from models.demos.blackhole.qwen36.tt.moe_decode import union as U

H, I, E, K = 1536, 384, 512, 16
DT = ttnn.bfloat4_b
OFF = {"QWEN35_MOE_ACT_BLK": "0", "QWEN35_MOE_DN_CHUNK": "8", "QWEN35_MOE_GU_FARW": "0", "QWEN35_MOE_GUR_READERS": "",
       "QWEN35_MOE_GU1B": "0", "QWEN35_MOE_GUDN": "0"}
ON = {"QWEN35_MOE_ACT_BLK": "17", "QWEN35_MOE_DN_CHUNK": "17:2", "QWEN35_MOE_GU_FARW": "17", "QWEN35_MOE_GUR_READERS": "17:8",
      "QWEN35_MOE_GU1B": "2-16", "QWEN35_MOE_GU1B_READERS": "2:2,5:3,9:4", "QWEN35_MOE_GU1B_ONROW": "2:2,5:3,9:4",
      "QWEN35_MOE_GUDN": "17"}
ALL = {"QWEN35_MOE_ACT_BLK": "1", "QWEN35_MOE_DN_CHUNK": "2", "QWEN35_MOE_GU_FARW": "1", "QWEN35_MOE_GUR_READERS": "1:8",
       "QWEN35_MOE_GU1B": "1", "QWEN35_MOE_GU1B_READERS": "3", "QWEN35_MOE_GU1B_ONROW": "2", "QWEN35_MOE_GUDN": "1"}


def logits(R, g):
    """[keys | s] FP32 [1, 1, R, 2E]: 16 random experts per row ranked first."""
    lg = torch.zeros(1, 1, R, 2 * E)
    for r in range(R):
        keys = torch.rand(E, generator=g) * 0.5
        ids = torch.randperm(E, generator=g)[:K]
        keys[ids] = 2.0 - 0.01 * torch.arange(K, dtype=torch.float32)
        lg[0, 0, r, :E] = keys
        lg[0, 0, r, E:] = torch.rand(E, generator=g) * 0.2 + 0.01
    return lg


def main():
    dev = ttnn.open_device(device_id=0, l1_small_size=24576, trace_region_size=64 << 20)
    os.environ["MOE_SWIGLU_LIMIT"] = "7.0"
    ok = True
    try:
        g = torch.Generator().manual_seed(0)
        dram, l1 = ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG
        up = lambda t: ttnn.from_torch(t, dtype=DT, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=dram)  # noqa
        ug = up(torch.randn(1, E, H, 2 * I, generator=g) * 0.03)
        dn = up(torch.randn(1, E, I, H, generator=g) * 0.03)
        bias = U.upload_selection_bias(dev, torch.zeros(E))
        for R in (1, 2, 4, 8, 16, 24, 32):
            lg = ttnn.from_torch(logits(R, g), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=l1)
            x = ttnn.from_torch(torch.randn(1, 1, R, H, generator=g), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                device=dev, memory_config=dram)
            outs, times = {}, {}
            for name, env in (("off", OFF), ("on", ON), ("all", ALL)):
                os.environ.update(env)
                if R == 1:
                    b = X.MoEDecodeBuffers(dev, H, I, K)
                    gu = X.ExpertGateUp(dev, H, I, E, K, DT, readers=4, router="keys", shared=False)
                    dop = X.ExpertDown(dev, H, I, E, K, DT, shared=False)
                    fn = lambda: (gu(x, lg, ug, ug, b.act, b.route, bias=bias), dop(b.act, b.route, dn, dn, b.out))  # noqa
                else:
                    act = U.act_buffer(dev, I, E, K, R)
                    b = U.RowsBuffers(dev, H, I, E, K, R, DT, act=act)
                    route = U.RouteRows(dev, E, K, R, router="keys")
                    if U.gudn_rows() <= R:  # gate|up + down in one program
                        op = U.ExpertGateUpDownRows(dev, H, I, E, K, DT, R, shared=False)
                        fn = lambda: (route(lg, b.table, bias=bias),  # noqa
                                      op(x, b.table, b.identity, ug, ug, act, dn, dn, b.out))
                    else:
                        gu = U.ExpertGateUpRows(dev, H, I, E, K, DT, R, shared=False)
                        dop = U.ExpertDownRows(dev, H, I, E, K, DT, R, shared=False)
                        fn = lambda: (route(lg, b.table, bias=bias), gu(x, b.table, b.identity, ug, ug, act),  # noqa
                                      dop(act, b.table, dn, dn, b.out))
                fn()
                ttnn.synchronize_device(dev)
                outs[name] = ttnn.to_torch(b.out).reshape(-1, H)[:R].clone()
                tid = ttnn.begin_trace_capture(dev, cq_id=0)
                for _ in range(20):
                    fn()
                ttnn.end_trace_capture(dev, tid, cq_id=0)
                ttnn.execute_trace(dev, tid, cq_id=0, blocking=True)
                t0 = time.time()
                ttnn.execute_trace(dev, tid, cq_id=0, blocking=True)
                times[name] = (time.time() - t0) / 20 * 1e6
                ttnn.release_trace(dev, tid)
                if R > 1:
                    ttnn.deallocate(act)
            same = {n: bool(torch.equal(outs[n], outs["off"])) for n in ("on", "all")}
            finite = bool(torch.isfinite(outs["off"]).all()) and float(outs["off"].abs().max()) > 0
            ok &= all(same.values()) and finite
            print(f"R={R:2d}: on/all bit-identical to off {same} (finite, nonzero {finite}); us per call off "
                  f"{times['off']:.1f} on {times['on']:.1f} all {times['all']:.1f}", flush=True)
    finally:
        ttnn.close_device(dev)
    print("TEST_MOE_STREAM", "PASS" if ok else "FAIL", flush=True)


if __name__ == "__main__":
    main()
