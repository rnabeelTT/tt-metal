# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage-3 review #1 probe (P1-4): geometry sweep of the routed-expert sparse matmuls under the selected precision
(policy `opt`: bfp4 experts, LoFi) on the real layer-4 weights, one die.

One Qwen35MoEOptimized build (layer 4, tt/optimized_decoder.CANDIDATES["opt"].moe, fused SwiGLU activation as in the
fused / optimized decoder); every variant swaps the program configs of the built module, so all variants read the
same weights:
  B=1  (1 row, decode): the gather path (cfg_gate_up_gather / cfg_down_gather: cores x in0_block_w cap), and the
       pairs path (pairs_min_tokens = 1, grouped configs)
  B=32 (32 rows, decode): the expert-sorted path with the group-parallel (grouped) sparse matmuls: in0_block_w of
       gate/up and down (2 / 4 / 8 / 16, forced past the module's L1 budget rule) and the output subblock width;
       and the pairs path (QWEN35_DECODE_SORTED_MIN_ROWS = 0)
  prefill 2048 rows: the same grouped configs, the row split (GLM53_MOE_SORTED_ROWS 1024 / 0)
  (fp32 destination accumulation caps the output subblock at 4 tiles: subblock widths 1 / 2 / 4 only)
Decode: the whole MoE forward captured as a trace, mean of 30 replays (device bound). Prefill: synchronized host time
of one forward, best of 3. Every variant's output is compared with the default variant's (PCC, max |diff|).
Input rows: real-stream statistics (tools/moe_scale_probe.stream_rows).

usage: python -m models.autoports.zai_org_glm_5_3_flash.tools.s3r_probe_sparse [--only b1,b32,prefill]"""

import argparse
import dataclasses
import os
import time

import ttnn
from models.autoports.zai_org_glm_5_3_flash.tests.harness import open_mesh, pcc
from models.autoports.zai_org_glm_5_3_flash.tools.moe_scale_probe import stream_rows
from models.autoports.zai_org_glm_5_3_flash.tt.ckpt import CheckpointReader
from models.autoports.zai_org_glm_5_3_flash.tt.functional_decoder import moe_args, moe_state_dict
from models.autoports.zai_org_glm_5_3_flash.tt.moe import sparse_matmul_program_config
from models.autoports.zai_org_glm_5_3_flash.tt.moe_opt import Qwen35MoEOptimized
from models.autoports.zai_org_glm_5_3_flash.tt.optimized_decoder import CANDIDATES


def run_traced(mesh, moe, tx, mode, n=30):
    o = moe.forward(tx, mode=mode)
    ttnn.synchronize_device(mesh)
    ttnn.deallocate(o)
    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
    out = moe.forward(tx, mode=mode)
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


def run_eager(mesh, moe, tx, mode, n=3):
    best = 1e30
    y = None
    for _ in range(n):
        ttnn.synchronize_device(mesh)
        t = time.perf_counter()
        o = moe.forward(tx, mode=mode)
        ttnn.synchronize_device(mesh)
        best = min(best, (time.perf_counter() - t) * 1e6)
        y = ttnn.to_torch(o).float()
        ttnn.deallocate(o)
    return y, best


class Sweep:
    def __init__(self, mesh, rows, label):
        self.mesh, self.rows, self.label = mesh, rows, label
        self.ref = None
        self.ref_us = None

    def run(self, name, fn):
        try:
            y, us = fn()
            if self.ref is None:
                self.ref, self.ref_us = y, us
            d = (y - self.ref).abs().max().item()
            print(
                f"{self.label:8} {name:52}: {us:9.1f} us ({us - self.ref_us:+8.1f} vs default)  "
                f"PCC vs default {pcc(y, self.ref):.6f} max|d| {d:.2e}",
                flush=True,
            )
        except Exception as e:  # noqa: BLE001
            msg = " | ".join(l.strip() for l in str(e).splitlines() if l.strip())[:300]
            print(f"{self.label:8} {name:52}: ERROR {msg}", flush=True)


def grouped(moe, gu_in0=None, dn_in0=None, sw=None):
    """Rebuild the module's grouped configs from its policy, then force in0_block_w / subblock (past the L1 rule)."""
    moe._cfg_grouped = None
    cfg_gu, cfg_dn = moe._grouped_configs(32, moe.hidden, moe.inter_local)
    cfgs_dn = (cfg_dn, moe._cfg_grouped_dn_rm)
    if gu_in0:
        cfg_gu.in0_block_w = gu_in0
    if dn_in0:
        for c in cfgs_dn:
            c.in0_block_w = dn_in0
    if sw:
        for c in (cfg_gu,) + cfgs_dn:
            c.out_subblock_w = sw
    return (
        f"gu in0 {cfg_gu.in0_block_w} sw {cfg_gu.out_subblock_w} / dn in0 {cfg_dn.in0_block_w} "
        f"sw {cfg_dn.out_subblock_w}{' untilize' if moe._cfg_grouped_down_untilize else ''}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="b1,b32,prefill")
    ap.add_argument("--layer", type=int, default=4)
    a = ap.parse_args()
    rd = CheckpointReader()
    sd = rd.layer_state(a.layer)
    x_all = stream_rows(sd, a.layer, 2048)
    mesh = open_mesh()
    try:
        opt = CANDIDATES["opt"].moe
        print(f"policy: {opt.routed_gate_up_dtype} / {opt.routed_down_dtype} {opt.routed_fidelity}", flush=True)
        margs = moe_args(rd.text_config, 1, opt=opt)
        moe = Qwen35MoEOptimized(
            mesh, margs, moe_state_dict(sd, 192), tensor_cache_path=None, tt_ccl=None, layer_num=a.layer
        )
        moe.fused_swiglu_act = True
        moe.prepare_dispatch([1, 32, 2048])
        H, Nl = moe.hidden, moe.inter_local
        base_opt = moe.opt
        dev = lambda r: ttnn.from_torch(
            x_all[:r].reshape(1, 1, r, H), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh
        )

        if "b1" in a.only:
            tx = dev(1)
            s = Sweep(mesh, 1, "B=1")
            g0, d0 = moe.cfg_gate_up_gather, moe.cfg_down_gather
            print(
                f"default gather: gu grid {g0.compute_with_storage_grid_size} in0 {g0.in0_block_w} per_core_N {g0.per_core_N}; "
                f"dn grid {d0.compute_with_storage_grid_size} in0 {d0.in0_block_w} per_core_N {d0.per_core_N}",
                flush=True,
            )
            s.run("gather default (gu c64 in0 32, dn c64 in0 32)", lambda: run_traced(mesh, moe, tx, "decode"))
            for cores, cap in ((64, 16), (64, 64), (64, 128), (32, 32), (32, 64), (32, 128), (128, 32), (128, 64)):
                moe.cfg_gate_up_gather = sparse_matmul_program_config(
                    1, 2 * Nl, H, target_cores=cores, in0_block_w_cap=cap
                )
                c = moe.cfg_gate_up_gather
                s.run(
                    f"gather gu c{c.compute_with_storage_grid_size.x * c.compute_with_storage_grid_size.y} in0 {c.in0_block_w}",
                    lambda: run_traced(mesh, moe, tx, "decode"),
                )
            moe.cfg_gate_up_gather = g0
            for cores, cap in ((64, 16), (64, 64), (32, 32), (32, 64), (128, 32), (128, 64)):
                moe.cfg_down_gather = sparse_matmul_program_config(1, H, Nl, target_cores=cores, in0_block_w_cap=cap)
                c = moe.cfg_down_gather
                s.run(
                    f"gather dn c{c.compute_with_storage_grid_size.x * c.compute_with_storage_grid_size.y} in0 {c.in0_block_w}",
                    lambda: run_traced(mesh, moe, tx, "decode"),
                )
            moe.cfg_down_gather = d0
            moe.opt = dataclasses.replace(base_opt, pairs_min_tokens=1)
            for kw in ({}, {"gu_in0": 8, "dn_in0": 8}, {"gu_in0": 2, "dn_in0": 2}):
                desc = grouped(moe, **kw)
                s.run(f"pairs path, {desc}", lambda: run_traced(mesh, moe, tx, "decode"))
            moe.opt = base_opt
            moe._cfg_grouped = None
            ttnn.deallocate(tx)

        if "b32" in a.only:
            tx = dev(32)
            s = Sweep(mesh, 32, "B=32")
            desc = grouped(moe)
            s.run(f"sorted grouped default ({desc})", lambda: run_traced(mesh, moe, tx, "decode"))
            for kw in (
                {"gu_in0": 2},
                {"gu_in0": 8},
                {"gu_in0": 16},
                {"dn_in0": 2},
                {"dn_in0": 8},
                {"dn_in0": 16},
                {"gu_in0": 8, "dn_in0": 8},
                {"sw": 1},
                {"sw": 2},
            ):
                desc = grouped(moe, **kw)
                s.run(f"sorted grouped {desc}", lambda: run_traced(mesh, moe, tx, "decode"))
            moe._cfg_grouped = None
            os.environ["QWEN35_DECODE_SORTED_MIN_ROWS"] = "0"
            desc = grouped(moe)
            s.run(f"pairs path (256 groups), {desc}", lambda: run_traced(mesh, moe, tx, "decode"))
            os.environ.pop("QWEN35_DECODE_SORTED_MIN_ROWS")
            moe._cfg_grouped = None
            ttnn.deallocate(tx)

        if "prefill" in a.only:
            tx = dev(2048)
            s = Sweep(mesh, 2048, "pf2048")
            desc = grouped(moe)
            s.run(f"sorted grouped default ({desc}), rows split 1024", lambda: run_eager(mesh, moe, tx, "prefill"))
            os.environ["GLM53_MOE_SORTED_ROWS"] = "0"
            s.run("  rows split off (whole chunk)", lambda: run_eager(mesh, moe, tx, "prefill"))
            os.environ.pop("GLM53_MOE_SORTED_ROWS")
            for kw in (
                {"gu_in0": 2},
                {"gu_in0": 8},
                {"dn_in0": 2},
                {"dn_in0": 8},
                {"gu_in0": 8, "dn_in0": 8},
                {"sw": 2},
            ):
                desc = grouped(moe, **kw)
                s.run(f"sorted grouped {desc}", lambda: run_eager(mesh, moe, tx, "prefill"))
            moe._cfg_grouped = None
            ttnn.deallocate(tx)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
