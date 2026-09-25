# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage-4 candidate sweep on the (1,4) mesh (real weights): PCC vs HF + warmed prefill wall + traced decode (3 replays)
for `MultichipConfig` variants.

    MESH_DEVICE=P300x2 python models/autoports/qwen_qwen3_5_122b_a10b/tools/sweep_multichip.py --family dense_1d --layers 0,3 --out sweep.jsonl
"""

import argparse
import json
import os
import time

import torch

import ttnn
from models.autoports.qwen_qwen3_5_122b_a10b.tests.test_functional_decoder import _hidden
from models.autoports.qwen_qwen3_5_122b_a10b.tests.test_multichip_decoder import MESH_SHAPE, _MeshHarness
from models.autoports.qwen_qwen3_5_122b_a10b.tt.multichip_decoder import MULTICHIP_SELECTED, MultichipConfig
from models.common.utility_functions import comp_pcc
from models.demos.blackhole.qwen36.tt.model_config import GDN_CONV1D_L1_SMALL_SIZE

os.environ.setdefault("HF_MODEL", "Qwen/Qwen3.5-122B-A10B")


def families(base: MultichipConfig):
    _ALL7 = base.with_(
        gdn_gqa_matmul=True,
        gdn_ab_tile_pad=True,
        gdn_state_inplace=True,
        moe_gate_fast=True,
        moe_router_l1=True,
        moe_gather_tail=True,
    )
    return {
        "baseline": [("mc_selected", base)],
        "dense_1d": [
            (f"attn_qkv{a}_wo{b}", base.with_(attn_qkv_decode_cores=a, attn_wo_decode_cores=b))
            for (a, b) in [(None, None), (32, 32), (64, 64), (88, 33), (44, 44)]
        ],
        "gdn_1d": [
            (f"gdn_qkvz{a}_out{b}", base.with_(gdn_qkvz_decode_cores=a, gdn_out_decode_cores=b))
            for (a, b) in [(None, None), (32, 32), (64, 64), (88, 33), (44, 44)]
        ],
        "norm_mode": [
            ("norm_A", base.with_(prefill_norm_distributed=False)),
            ("norm_B", base.with_(prefill_norm_distributed=True)),
        ],
        "gate_up": [
            (f"gu{c}_in{i}", base.with_(gate_up_cores=c, gate_up_in0=i))
            for (c, i) in [(16, 32), (16, 96), (8, 32), (16, 8)]
        ],
        "down": [
            (f"dn{c}_in{i}", base.with_(down_cores=c, down_in0=i)) for (c, i) in [(48, 8), (96, 8), (32, 8), (24, 8)]
        ],
        "shared": [(f"shared{c}", base.with_(shared_decode_cores=c)) for c in (None, 8, 16, 17)],
        "gdn_state": [
            ("state_bf16", base.with_(gdn_state_dtype=ttnn.bfloat16, gdn_decode_high_precision=False)),
            ("state_fp32", base.with_(gdn_state_dtype=ttnn.float32, gdn_decode_high_precision=True)),
        ],
        "sdpa": [
            ("sdpa_0_0", base.with_(sdpa_decode_q_chunk=0, sdpa_decode_k_chunk=0)),
            ("sdpa_32_256", base.with_(sdpa_decode_q_chunk=32, sdpa_decode_k_chunk=256)),
        ],
        "conv1d": [(f"conv_blk{c}", base.with_(gdn_conv1d_max_channels=c)) for c in (1024, 1536, 3072)],
        # --- stage 5 families ---
        "mixer_ds": [(f"mixer_{m}", base.with_(mixer_decode_mode=m)) for m in ("1d", "ds_in", "ds_all")],
        # review-2: Blackhole multi-reader DRAM-sharded candidates (2 and 3 readers per bank), in-projection only with the
        # tuned 1D out-projection kept (ds_in_out1d) and all projections DRAM-sharded (ds_all)
        "out_dtype": [("out_bf16", base), ("out_bfp8_rs", base.with_(out_proj_decode_dtype="bfp8"))],
        "norm_ag_persistent": [("norm_ag_alloc", base), ("norm_ag_persistent", base.with_(norm_ag_persistent=True))],
        # --- stage 7 families: decode op-count reductions (each knob A/B'd against the stage-5 default) ---
        "decode_ops": [
            ("ops_base", base),
            ("ops_gdn_gqa", base.with_(gdn_gqa_matmul=True)),
            ("ops_gdn_abpad", base.with_(gdn_ab_tile_pad=True)),
            ("ops_gdn_inplace", base.with_(gdn_state_inplace=True)),
            ("ops_gdn_all", base.with_(gdn_gqa_matmul=True, gdn_ab_tile_pad=True, gdn_state_inplace=True)),
            ("ops_moe_gate", base.with_(moe_gate_fast=True)),
            ("ops_moe_gate_l1", base.with_(moe_gate_fast=True, moe_router_l1=True)),
            ("ops_moe_tail", base.with_(moe_gather_tail=True)),
            (
                "ops_all",
                base.with_(
                    gdn_gqa_matmul=True,
                    gdn_ab_tile_pad=True,
                    gdn_state_inplace=True,
                    moe_gate_fast=True,
                    moe_router_l1=True,
                    moe_gather_tail=True,
                ),
            ),
        ],
        "decode_ops_r2": [  # round 2: bfp4 expansion matrix, v head rows, view fallback
            ("ops_base", base),
            ("ops_gdn_gqa", base.with_(gdn_gqa_matmul=True)),
            ("ops_gdn_all", base.with_(gdn_gqa_matmul=True, gdn_ab_tile_pad=True, gdn_state_inplace=True)),
            ("ops_moe_tail", base.with_(moe_gather_tail=True)),
            (
                "ops_all",
                base.with_(
                    gdn_gqa_matmul=True,
                    gdn_ab_tile_pad=True,
                    gdn_state_inplace=True,
                    moe_gate_fast=True,
                    moe_router_l1=True,
                    moe_gather_tail=True,
                ),
            ),
        ],
        "decode_ops_r3": [
            ("ops_base", base),
            ("ops_moe_tail", base.with_(moe_gather_tail=True)),
            (
                "ops_all",
                base.with_(
                    gdn_gqa_matmul=True,
                    gdn_ab_tile_pad=True,
                    gdn_state_inplace=True,
                    moe_gate_fast=True,
                    moe_router_l1=True,
                    moe_gather_tail=True,
                ),
            ),
        ],
        "decode_ops_r4": [
            ("ops_base", base),
            ("ops_all", _ALL7),
            ("ops_all_normfold", _ALL7.with_(gdn_norm_fold=True)),
            ("ops_all_convl1", _ALL7.with_(gdn_conv_l1=True)),
            ("ops_all_normfold_convl1", _ALL7.with_(gdn_norm_fold=True, gdn_conv_l1=True)),
        ],
        "prefill_sorted": [  # prefill MoE path at short buckets: masked 64-token chunks (default) vs expert-sorted dispatch
            ("sorted_min512", base),
            ("sorted_min128", base.with_(sorted_min_tokens=128)),
        ],
        "decode_ops_moe_r2": [
            ("ops_base", base),
            ("ops_moe_tail", base.with_(moe_gather_tail=True)),
            ("ops_moe_all", base.with_(moe_gate_fast=True, moe_router_l1=True, moe_gather_tail=True)),
        ],
        "decode_ops_moe": [
            ("ops_base", base),
            ("ops_moe_gate", base.with_(moe_gate_fast=True)),
            ("ops_moe_gate_l1", base.with_(moe_gate_fast=True, moe_router_l1=True)),
            ("ops_moe_tail", base.with_(moe_gather_tail=True)),
            ("ops_moe_all", base.with_(moe_gate_fast=True, moe_router_l1=True, moe_gather_tail=True)),
        ],
        "mixer_ds_readers": [("mixer_1d", base)]
        + [
            (f"mixer_ds_in_out1d_r{r}", base.with_(mixer_decode_mode="ds_in", mixer_ds_out_1d=True, mixer_ds_readers=r))
            for r in (1, 2, 3)
        ]
        + [(f"mixer_ds_all_r{r}", base.with_(mixer_decode_mode="ds_all", mixer_ds_readers=r)) for r in (2, 3)],
        "mmrs": [("mmrs_off", base), ("mmrs_on", base.with_(fuse_out_mmrs_decode=True))],
        "mixer_fidelity": [("mixer_hifi2", base), ("mixer_lofi", base.with_(mixer_fidelity="lofi"))],
        "mixer_in0": [
            (f"attn_in0_{a}_gdn_in0_{g}", base.with_(attn_qkv_in0=a, gdn_qkvz_in0=g))
            for (a, g) in [(None, None), (12, 12), (16, 16), (24, 24), (32, 32), (48, 48)]
        ],
        "gu_tile": [
            ("gu_tile32x32_c16", base),
            ("gu_tile32x16_c32", base.with_(gate_up_output_tile=(32, 16), gate_up_cores=32)),
        ],
        "dense_1d_long": [
            (f"attn_qkv{a}_wo{b}", base.with_(attn_qkv_decode_cores=a, attn_wo_decode_cores=b))
            for (a, b) in [(None, None), (88, 33), (44, 44), (None, None), (88, 33), (44, 44)]
        ],
    }


def _pcc(a, b):
    ok, pcc = comp_pcc(a.float(), b.float(), 0.0)
    return float(str(pcc).split()[-1]) if isinstance(pcc, str) else float(pcc)


def measure(mesh, layer_idx, mc, name, prefill_len=128, decode_ctx=128, hf=True, max_seq_len=4096, replays=3):
    row = {"name": name, "layer_idx": layer_idx, "config": mc.describe()}
    t0 = time.time()
    h = _MeshHarness(mesh, layer_idx, batch=1, max_seq_len=max_seq_len, real=True, hf=hf, mc=mc)
    row["kind"] = h.kind
    row["build_s"] = round(time.time() - t0, 1)
    hidden = _hidden(1, prefill_len, h.H, seed=2000)
    if hf:
        row["pcc_prefill"] = _pcc(h.hf_prefill(hidden), h.tt_prefill(hidden))
        dec = _hidden(1, 1, h.H, seed=2001)
        pos = torch.tensor([prefill_len])
        row["pcc_decode"] = _pcc(h.hf_decode(dec, pos), h.tt_decode(dec, pos))
    if h.kind == "linear_attention":
        h.tt.reset_gdn_state()
    h.tt_prefill(hidden)  # warm
    if h.kind == "linear_attention":
        h.tt.reset_gdn_state()
    ttnn.synchronize_device(mesh)
    t0 = time.time()
    h.tt_prefill(hidden)
    ttnn.synchronize_device(mesh)
    row["prefill_ms"] = round(1e3 * (time.time() - t0), 2)
    if decode_ctx > prefill_len and h.kind == "full_attention":
        start = prefill_len
        while start < decode_ctx:
            n = min(2048, decode_ctx - start)
            h.tt_prefill(_hidden(1, n, h.H, seed=3000 + start), start=start)
            start += n
    x, kw = h.decode_inputs(_hidden(1, 1, h.H, seed=2002), torch.tensor([decode_ctx]))
    out = h.tt.decode_forward(x, **kw)
    ttnn.deallocate(out)
    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
    out = h.tt.decode_forward(x, **kw)
    ttnn.end_trace_capture(mesh, tid, cq_id=0)
    ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
    ttnn.synchronize_device(mesh)
    t0 = time.time()
    for _ in range(replays):
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh)
    row["decode_ms"] = round(1e3 * (time.time() - t0) / replays, 3)
    row["replays"] = replays
    ttnn.release_trace(mesh, tid)
    return row


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True)
    ap.add_argument("--layers", default="0,3")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-hf", action="store_true")
    ap.add_argument("--decode-ctx", type=int, default=128)
    ap.add_argument("--prefill-len", type=int, default=128)
    ap.add_argument("--replays", type=int, default=3)
    a = ap.parse_args()
    cands = families(MULTICHIP_SELECTED)[a.family]
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(
        ttnn.MeshShape(*MESH_SHAPE),
        l1_small_size=GDN_CONV1D_L1_SMALL_SIZE,
        trace_region_size=64_000_000,
        num_command_queues=1,
    )
    try:
        with open(a.out, "a") as f:
            for layer_idx in [int(s) for s in a.layers.split(",")]:
                for name, mc in cands:
                    try:
                        row = measure(
                            mesh,
                            layer_idx,
                            mc,
                            name,
                            prefill_len=a.prefill_len,
                            hf=not a.no_hf,
                            decode_ctx=a.decode_ctx,
                            max_seq_len=max(4096, a.decode_ctx + 256, a.prefill_len + 256),
                            replays=a.replays,
                        )
                    except Exception as e:
                        import traceback

                        frames = [
                            f"{fr.filename.split('/models/')[-1]}:{fr.lineno} {fr.name}"
                            for fr in traceback.extract_tb(e.__traceback__)
                            if "/models/" in fr.filename
                        ][-6:]
                        row = {
                            "name": name,
                            "layer_idx": layer_idx,
                            "error": str(e)[:400],
                            "frames": frames,
                            "config": mc.describe(),
                        }
                        print("FAILED", name, layer_idx, str(e)[:200])
                    print(json.dumps({k: v for k, v in row.items() if k != "config"}))
                    f.write(json.dumps(row) + "\n")
                    f.flush()
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
