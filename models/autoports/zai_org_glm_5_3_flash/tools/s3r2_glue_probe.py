# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage-3 re-review #2 P2-3 probes (one die):
  down   routed MoE down projection (layer 4 real weights, policy opt) with per_core_N = 16 in the grouped sparse config
         (stage-2 hand-off: a resident [Nl, 16 tiles] expert block lets the down matmul write row-major rows
         (untilize_out) instead of the separate 6240 x 4096 untilize) at 32 decode rows and 2048 prefill rows; time and
         PCC vs the default (per_core_N = all 128 N tiles, separate untilize)
  rec    ttnn.experimental.kda.recurrent_decode (64 heads, K = V = 128, fp32 state) at B = 1 / 2 / 4 / 8 / 16 / 32:
         device-time scaling (traced, mean of 50 replays)
  glue   the 32-user glue ops of the final Tracy: routing reshape [1,32,32(8),32(1)] -> [1,1,32,32(8)] (fp32 TILE, 1 core,
         35.6 us) and the DSA indexer query reshape [1,1,32,4096] -> [1,1,1024,128] (bf16 TILE, 42.1 us): ttnn.reshape vs a
         row-major round trip / view alternatives (bit-exact check)

usage: python -m models.autoports.zai_org_glm_5_3_flash.tools.s3r2_glue_probe [--only down,rec,glue]"""

import argparse
import time

import torch

import ttnn
from models.autoports.zai_org_glm_5_3_flash.tests.harness import open_mesh

DRAM = ttnn.DRAM_MEMORY_CONFIG


def traced_us(mesh, fn, n=50):
    o = fn()
    ttnn.synchronize_device(mesh)
    ttnn.deallocate(o)
    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
    out = fn()
    ttnn.end_trace_capture(mesh, tid, cq_id=0)
    ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
    t = time.perf_counter()
    for _ in range(n):
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh)
    us = (time.perf_counter() - t) / n * 1e6
    y = ttnn.to_torch(out).float()
    ttnn.release_trace(mesh, tid)
    ttnn.deallocate(out)
    return y, us


def err(e):
    return " | ".join(l.strip() for l in str(e).splitlines() if l.strip() and "backtrace" not in l.lower())[:300]


def down(mesh):
    from models.autoports.zai_org_glm_5_3_flash.tools.moe_scale_probe import stream_rows
    from models.autoports.zai_org_glm_5_3_flash.tools.s3r_probe_sparse import Sweep, run_eager, run_traced
    from models.autoports.zai_org_glm_5_3_flash.tt.ckpt import CheckpointReader
    from models.autoports.zai_org_glm_5_3_flash.tt.functional_decoder import moe_args, moe_state_dict
    from models.autoports.zai_org_glm_5_3_flash.tt.moe_opt import Qwen35MoEOptimized
    from models.autoports.zai_org_glm_5_3_flash.tt.optimized_decoder import CANDIDATES

    rd = CheckpointReader()
    sd = rd.layer_state(4)
    x_all = stream_rows(sd, 4, 2048)
    opt = CANDIDATES["opt"].moe
    moe = Qwen35MoEOptimized(
        mesh,
        moe_args(rd.text_config, 1, opt=opt),
        moe_state_dict(sd, 192),
        tensor_cache_path=None,
        tt_ccl=None,
        layer_num=4,
    )
    moe.fused_swiglu_act = True
    moe.prepare_dispatch([32, 2048])
    H, Nl = moe.hidden, moe.inter_local

    def set_cfg(pcn):
        moe._cfg_grouped = None
        cfg_gu, cfg_dn = moe._grouped_configs(32, H, Nl)
        if pcn is None:
            return (
                f"default: dn per_core_N {cfg_dn.per_core_N} in0 {cfg_dn.in0_block_w} sw {cfg_dn.out_subblock_w} "
                f"untilize_out {moe._cfg_grouped_down_untilize}"
            )
        for c in (cfg_dn, moe._cfg_grouped_dn_rm):
            c.per_core_N = pcn
            c.out_subblock_w = max(d for d in (4, 2, 1) if pcn % d == 0)
            c.in0_block_w = 8
        if hasattr(moe._cfg_grouped_dn_rm, "untilize_out"):
            moe._cfg_grouped_dn_rm.untilize_out = True
            moe._cfg_grouped_down_untilize = True
        return f"dn per_core_N {pcn} in0 8 sw {cfg_dn.out_subblock_w} untilize_out True"

    for rows, label, runner, mode in ((32, "B=32", run_traced, "decode"), (2048, "pf2048", run_eager, "prefill")):
        tx = ttnn.from_torch(
            x_all[:rows].reshape(1, 1, rows, H), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh
        )
        s = Sweep(mesh, rows, label)
        for pcn in (None, 16, 32):
            desc = set_cfg(pcn)
            s.run(desc, lambda: runner(mesh, moe, tx, mode))
        moe._cfg_grouped = None
        ttnn.deallocate(tx)


def rec(mesh):
    H, K, V = 64, 128, 128
    g = torch.Generator().manual_seed(0)
    for B in (1, 2, 4, 8, 16, 32):
        S = ttnn.from_torch(
            torch.randn(B, H, K, V, generator=g) * 0.05, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh
        )
        up = lambda t: ttnn.from_torch(
            t.reshape(1, 1, *t.shape), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh
        )
        qkv = up(torch.randn(B, 2 * H * K + H * V, generator=g))
        dec = up(torch.exp(-5 * torch.sigmoid(torch.randn(B, H * K, generator=g))))
        beta = up(torch.sigmoid(torch.randn(B, H, generator=g)))
        try:
            _, us = traced_us(mesh, lambda: ttnn.experimental.kda.recurrent_decode(S, qkv, dec, beta, K**-0.5))
            mb = 2 * B * H * K * V * 4 / 1e6
            print(
                f"recurrent_decode B={B:2d}: {us:8.1f} us/call (host-timed trace)  state r+w {mb:6.1f} MB -> {mb / us * 1e3:6.1f} GB/s",
                flush=True,
            )
        except Exception as e:  # noqa: BLE001
            print(f"recurrent_decode B={B}: ERROR {err(e)}", flush=True)
        for t in (S, qkv, dec, beta):
            ttnn.deallocate(t)


def glue(mesh):
    g = torch.Generator().manual_seed(0)
    cases = {
        "routing [1,32,8,1] fp32 -> [1,1,32,8]": (torch.randn(1, 32, 8, 1, generator=g), ttnn.float32, (1, 1, 32, 8)),
        "indexer q [1,1,32,4096] bf16 -> [1,1,1024,128]": (
            torch.randn(1, 1, 32, 4096, generator=g),
            ttnn.bfloat16,
            (1, 1, 1024, 128),
        ),
    }
    for name, (t, dt, shape) in cases.items():
        tx = ttnn.from_torch(t, dtype=dt, layout=ttnn.TILE_LAYOUT, device=mesh)
        ref = (t.bfloat16() if dt == ttnn.bfloat16 else t).reshape(shape).float()  # the device holds bf16 values
        variants = {
            "ttnn.reshape (TILE)": lambda: ttnn.reshape(tx, shape),
            "to RM -> reshape -> TILE": lambda: ttnn.to_layout(
                ttnn.reshape(ttnn.to_layout(tx, ttnn.ROW_MAJOR_LAYOUT, memory_config=DRAM), shape),
                ttnn.TILE_LAYOUT,
                memory_config=DRAM,
            ),
            "ttnn.view (TILE)": lambda: ttnn.view(tx, shape),
        }
        if name.startswith("indexer"):
            # head split op on the [32 rows, 32 heads x 128] layout: [1, 32 heads, 32 rows, 128] (the indexer score op would
            # need the head-major order instead of [1024 = rows x heads] rows)
            variants[
                "nlp_create_qkv_heads (head-major [1,32,32,128])"
            ] = lambda: ttnn.experimental.nlp_create_qkv_heads(
                tx, num_heads=32, num_kv_heads=0, transpose_k_heads=False, memory_config=DRAM
            )[
                0
            ]
        for vn, fn in variants.items():
            try:
                y, us = traced_us(mesh, fn)
                ok = (
                    y.numel() == ref.numel() and torch.equal(y.reshape(ref.shape), ref)
                    if "head-major" not in vn
                    else "n/a (layout)"
                )
                print(f"{name:48} {vn:48}: {us:8.1f} us  exact {ok}", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"{name:48} {vn:48}: ERROR {err(e)}", flush=True)
        ttnn.deallocate(tx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="down,rec,glue")
    a = ap.parse_args()
    mesh = open_mesh()
    try:
        for sec in a.only.split(","):
            print(f"===== {sec}", flush=True)
            {"down": down, "rec": rec, "glue": glue}[sec](mesh)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
