# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage-3 candidate sweep on real weights: PCC vs HF + warmed prefill wall + traced decode (3 replays).

    python models/autoports/qwen_qwen3_5_122b_a10b/tools/sweep_optimized.py --family precision --layers 0,3 --out sweep.jsonl

Families enumerate OptimizationConfig variants relative to a base; every row records the full config, PCCs and timings.
"""
import argparse
import json
import os
import time

import torch

import ttnn
from models.autoports.qwen_qwen3_5_122b_a10b.tests import reference as ref
from models.autoports.qwen_qwen3_5_122b_a10b.tests.test_functional_decoder import _Harness, _hidden
from models.autoports.qwen_qwen3_5_122b_a10b.tt.optimized_decoder import SELECTED, OptimizationConfig, OptimizedDecoder
from models.common.utility_functions import comp_pcc

os.environ.setdefault("HF_MODEL", "Qwen/Qwen3.5-122B-A10B")
BF4, BF8, BF16 = ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.bfloat16
LOFI, HIFI2 = ttnn.MathFidelity.LoFi, ttnn.MathFidelity.HiFi2


def families(base: OptimizationConfig):
    return {
        "baseline_stage2": [
            (
                "stage2_bfp8_hifi2",
                base.with_(
                    routed_gate_up_dtype=BF8, routed_down_dtype=BF8, routed_fidelity=HIFI2, shared_fidelity=HIFI2
                ),
            )
        ],
        "precision": [
            ("routed_bfp8_hifi2", base.with_(routed_gate_up_dtype=BF8, routed_down_dtype=BF8, routed_fidelity=HIFI2)),
            ("routed_bfp8_lofi", base.with_(routed_gate_up_dtype=BF8, routed_down_dtype=BF8, routed_fidelity=LOFI)),
            (
                "routed_bfp4gu_bfp8down_lofi",
                base.with_(routed_gate_up_dtype=BF4, routed_down_dtype=BF8, routed_fidelity=LOFI),
            ),
            ("routed_bfp4_lofi", base.with_(routed_gate_up_dtype=BF4, routed_down_dtype=BF4, routed_fidelity=LOFI)),
            ("routed_bfp4_hifi2", base.with_(routed_gate_up_dtype=BF4, routed_down_dtype=BF4, routed_fidelity=HIFI2)),
        ],
        "shared": [
            ("shared_bfp8_hifi2", base.with_(shared_dtype=BF8, shared_fidelity=HIFI2)),
            ("shared_bfp8_lofi", base.with_(shared_dtype=BF8, shared_fidelity=LOFI)),
            ("shared_bfp4_lofi", base.with_(shared_dtype=BF4, shared_fidelity=LOFI)),
        ],
        "mixer_weights": [
            ("attn_bfp8", base.with_(attn_weight_dtype=BF8)),
            ("attn_bfp4", base.with_(attn_weight_dtype=BF4)),
            ("gdn_bfp8", base.with_(gdn_weight_dtype=BF8)),
            ("gdn_bfp4", base.with_(gdn_weight_dtype=BF4)),
        ],
        "kv": [("kv_bf16", base.with_(kv_dtype=BF16)), ("kv_bfp8", base.with_(kv_dtype=BF8))],
        "gather": [
            (
                f"gather_gu{c}_in{i}_dn{dc}_in{di}",
                base.with_(gather_gate_up_cores=c, gather_gate_up_in0=i, gather_down_cores=dc, gather_down_in0=di),
            )
            for (c, i, dc, di) in [
                (64, 96, 64, 32),
                (64, 32, 64, 32),
                (64, 8, 64, 8),
                (32, 96, 32, 32),
                (110, 96, 96, 32),
                (64, 48, 64, 16),
            ]
        ],
        "masked": [
            (
                f"masked_c{ch}_gu{c}_in{i}_dn{dc}",
                base.with_(masked_chunk=ch, masked_gate_up_cores=c, masked_gate_up_in0=i, masked_down_cores=dc),
            )
            for (ch, c, i, dc) in [
                (32, 32, 64, 32),
                (32, 64, 64, 32),
                (32, 16, 96, 16),
                (64, 32, 64, 32),
                (64, 64, 96, 64),
                (128, 64, 96, 64),
            ]
        ],
        "masked2": [
            (
                f"masked2_c{ch}_gu{c}_in{i}_dn{dc}_in{di}",
                base.with_(
                    masked_chunk=ch,
                    masked_gate_up_cores=c,
                    masked_gate_up_in0=i,
                    masked_down_cores=dc,
                    masked_down_in0=di,
                ),
            )
            for (ch, c, i, dc, di) in [
                (64, 32, 96, 32, 32),
                (64, 32, 32, 32, 16),
                (64, 16, 96, 16, 32),
                (128, 32, 32, 32, 32),
                (128, 16, 64, 16, 32),
                (128, 32, 16, 32, 16),
            ]
        ],
        "norm": [
            ("norm_interleaved", base.with_(decode_norm_sharded=False)),
            ("norm_sharded", base.with_(decode_norm_sharded=True)),
        ],
        "sdpa": [
            ("sdpa_default_32_64", base.with_(sdpa_decode_q_chunk=None, sdpa_decode_k_chunk=None)),
            ("sdpa_32_128", base.with_(sdpa_decode_q_chunk=32, sdpa_decode_k_chunk=128)),
            ("sdpa_32_256", base.with_(sdpa_decode_q_chunk=32, sdpa_decode_k_chunk=256)),
            ("sdpa_0_0", base.with_(sdpa_decode_q_chunk=0, sdpa_decode_k_chunk=0)),
        ],
        "gather_max": [(f"gather_max{g}", base.with_(gather_max_tokens=g)) for g in (1, 4, 8, 32)],
        "router": [(f"router_{m}", base.with_(router_decode_pc=m)) for m in ("ttnn", "derived", "1d8")],
        "shared_pc": [(f"shared_pc_{c}", base.with_(shared_decode_cores=c)) for c in (None, 16, 32, 44, 64, 88)],
        "dense_1d": [
            (
                f"dense_1d_in{ci}_out{co}",
                base.with_(dense_decode_mode="1d", dense_decode_cores_in=ci, dense_decode_cores_out=co),
            )
            for (ci, co) in [(64, 32), (88, 32), (44, 32), (64, 48), (64, 96), (88, 88)]
        ]
        + [
            (
                f"dense_1d_in64_out32_bw{bw}",
                base.with_(
                    dense_decode_mode="1d", dense_decode_cores_in=64, dense_decode_cores_out=32, dense_decode_in0=bw
                ),
            )
            for bw in (16, 32)
        ],
        "dense_ds": [
            (f"dense_ds_r{r}", base.with_(dense_decode_mode="dram_sharded", dense_dram_readers=r)) for r in (1, 2, 3)
        ],
        "prefill_mode": [
            ("prefill_masked", base.with_(prefill_mode="masked")),
            ("prefill_sorted", base.with_(prefill_mode="sorted", sorted_min_tokens=1)),
        ],
        "sorted_geom": [
            (
                f"sorted_gu{c}_in{i}_dn{dc}_in{di}",
                base.with_(
                    prefill_mode="sorted",
                    sorted_min_tokens=1,
                    sorted_gate_up_cores=c,
                    sorted_gate_up_in0=i,
                    sorted_down_cores=dc,
                    sorted_down_in0=di,
                ),
            )
            for (c, i, dc, di) in [
                (64, 32, 64, 32),
                (64, 96, 64, 32),
                (32, 64, 32, 32),
                (64, 8, 64, 8),
                (96, 32, 96, 32),
            ]
        ],
        "sorted_group": [
            (f"sorted_g{g}", base.with_(prefill_mode="sorted", sorted_min_tokens=1, sorted_group_rows=g))
            for g in (32, 64, 128)
        ],
        "gdn_state": [
            ("gdn_state_bf16", base.with_(gdn_state_fp32=False)),
            ("gdn_state_fp32", base.with_(gdn_state_fp32=True)),
        ],
        "dense_ds2": [
            ("ds2_attn_all_r1", base.with_(dense_decode_mode_attn="dram_sharded", dense_dram_readers=1)),
            (
                "ds2_attn_o_r1",
                base.with_(dense_decode_mode_attn="dram_sharded", dense_dram_readers=1, dense_ds_roles=("o",)),
            ),
            (
                "ds2_gdn_o_r1",
                base.with_(dense_decode_mode_gdn="dram_sharded", dense_dram_readers=1, dense_ds_roles=("o",)),
            ),
            (
                "ds2_gdn_mega_r3",
                base.with_(dense_decode_mode_gdn="dram_sharded", dense_dram_readers=3, dense_ds_roles=("mega",)),
            ),
        ],
        "mega_split": [(f"mega_split_r{r}", base.with_(mega_split_ds=True, mega_split_readers=r)) for r in (1, 2, 3)],
        "ds_cores": [(f"ds_cores_{c}", base.with_(dense_ds_cores=c)) for c in (32, 16, 8)],
        "kv_final": [
            ("kvf_bf16", base.with_(kv_dtype=BF16)),
            ("kvf_bfp8_k64", base.with_(kv_dtype=BF8)),
            ("kvf_bfp8_k256", base.with_(kv_dtype=BF8, sdpa_decode_q_chunk=32, sdpa_decode_k_chunk=256)),
        ],
        "norm_ds": [
            ("norm_ds_off", base.with_(attn_norm_out_sharded=False)),
            ("norm_ds_on", base.with_(attn_norm_out_sharded=True)),
        ],
        "gdn_dtype2": [
            ("gdn2_bfp8", base.with_(gdn_weight_dtype=BF8)),
            ("gdn2_bfp4", base.with_(gdn_weight_dtype=BF4)),
        ],
        "sdpa_long": [
            ("sdpa_long_default_32_64", base.with_(sdpa_decode_q_chunk=None, sdpa_decode_k_chunk=None)),
            ("sdpa_long_32_128", base.with_(sdpa_decode_q_chunk=32, sdpa_decode_k_chunk=128)),
            ("sdpa_long_32_256", base.with_(sdpa_decode_q_chunk=32, sdpa_decode_k_chunk=256)),
            ("sdpa_long_32_512", base.with_(sdpa_decode_q_chunk=32, sdpa_decode_k_chunk=512)),
            ("sdpa_long_kv_bfp8_32_64", base.with_(kv_dtype=BF8)),
            ("sdpa_long_kv_bfp8_32_256", base.with_(kv_dtype=BF8, sdpa_decode_q_chunk=32, sdpa_decode_k_chunk=256)),
        ],
    }


def measure(device, layer_idx, opt, name, prefill_len=128, decode_batch=1, decode_ctx=128, hf=True, max_seq_len=4096):
    cfg, text = ref.text_config()
    row = {"name": name, "layer_idx": layer_idx, "kind": text.layer_types[layer_idx], "config": opt.describe()}
    t0 = time.time()
    h = _Harness(
        device,
        layer_idx,
        batch=decode_batch,
        max_seq_len=max_seq_len,
        real=True,
        hf=hf,
        decoder_cls=OptimizedDecoder,
        decoder_kwargs={"opt": opt},
    )
    row["build_s"] = round(time.time() - t0, 1)
    ff = h.tt.layer.feed_forward
    row["materialized"] = {
        "up_gate": str(ff.up_gate.dtype),
        "down": str(ff.down.dtype),
        "shared": str((ff.shared_packed if getattr(ff, "shared_packed", None) is not None else ff.shared_w1).dtype),
        "shared_w2": str(ff.shared_w2.dtype),
        "routed_fidelity": str(ff.routed_ckc.math_fidelity),
        "shared_fidelity": str(ff.shared_ckc.math_fidelity),
    }
    if h.kind == "full_attention":
        w = h.tt.layer.attention.weights
        row["materialized"]["attn_proj"] = str((w.qkv_proj if w.qkv_proj is not None else w.q_proj).dtype)
        row["materialized"]["kv_cache"] = str(h.tt.kv_cache["k"].dtype)
    else:
        row["materialized"]["gdn_qkv"] = str(h.tt.layer.attention.weights.qkv_proj_weight.dtype)
    hidden = _hidden(1, prefill_len, h.H, seed=2000)
    # PCC (prefill + decode) vs HF
    if hf:
        ref_pre = h.hf_prefill(hidden)
        tt_pre = h.tt_prefill(hidden)
        row["pcc_prefill"] = (
            float(comp_pcc(ref_pre.float(), tt_pre.float(), 0.0)[1].split()[-1])
            if isinstance(comp_pcc(ref_pre.float(), tt_pre.float(), 0.0)[1], str)
            else float(comp_pcc(ref_pre.float(), tt_pre.float(), 0.0)[1])
        )
        dec = _hidden(1, 1, h.H, seed=2001)
        pos = torch.tensor([prefill_len])
        r = h.hf_decode(dec, pos)
        g = h.tt_decode(dec, pos)
        pcc = comp_pcc(r.float(), g.float(), 0.0)[1]
        row["pcc_decode"] = float(str(pcc).split()[-1])
    else:
        h.tt_prefill(hidden)
    # warmed prefill wall
    if h.kind == "linear_attention":
        h.tt.reset_gdn_state()
    ttnn.synchronize_device(device)
    t0 = time.time()
    h.tt_prefill(hidden)
    ttnn.synchronize_device(device)
    row["prefill_ms"] = round(1e3 * (time.time() - t0), 2)
    if decode_ctx > prefill_len and h.kind == "full_attention":
        # fill the cache up to decode_ctx in 2048-token chunks so SDPA-decode reads a long context
        if h.kind == "linear_attention":
            h.tt.reset_gdn_state()
        start = 0
        while start < decode_ctx:
            n = min(2048, decode_ctx - start)
            h.tt_prefill(_hidden(1, n, h.H, seed=3000 + start), start=start)
            start += n
    # traced decode (3 replays)
    x, kw = h.decode_inputs(_hidden(decode_batch, 1, h.H, seed=2002), torch.full((decode_batch,), decode_ctx))
    out = h.tt.decode_forward(x, **kw)
    ttnn.deallocate(out)
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    out = h.tt.decode_forward(x, **kw)
    ttnn.end_trace_capture(device, tid, cq_id=0)
    ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
    ttnn.synchronize_device(device)
    t0 = time.time()
    for _ in range(3):
        ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(device)
    row["decode_ms"] = round(1e3 * (time.time() - t0) / 3, 3)
    ttnn.release_trace(device, tid)
    if h.kind == "full_attention":
        ttnn.deallocate(h.tt.kv_cache["k"])
        ttnn.deallocate(h.tt.kv_cache["v"])
    return row


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True)
    ap.add_argument("--layers", default="0,3")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-hf", action="store_true")
    ap.add_argument("--base", default="selected", choices=["selected", "stage2"])
    ap.add_argument("--decode-batch", type=int, default=1)
    ap.add_argument("--decode-ctx", type=int, default=128)
    ap.add_argument("--prefill-len", type=int, default=128)
    a = ap.parse_args()
    base = (
        SELECTED
        if a.base == "selected"
        else SELECTED.with_(
            routed_gate_up_dtype=BF8, routed_down_dtype=BF8, routed_fidelity=HIFI2, shared_fidelity=HIFI2
        )
    )
    cands = families(base)[a.family]
    dev = ttnn.open_mesh_device(
        ttnn.MeshShape(1, 1), l1_small_size=24576, trace_region_size=64_000_000, num_command_queues=1
    )
    try:
        with open(a.out, "a") as f:
            for layer_idx in [int(s) for s in a.layers.split(",")]:
                for name, opt in cands:
                    try:
                        row = measure(
                            dev,
                            layer_idx,
                            opt,
                            name,
                            prefill_len=a.prefill_len,
                            hf=not a.no_hf,
                            decode_batch=a.decode_batch,
                            decode_ctx=a.decode_ctx,
                            max_seq_len=max(4096, a.decode_ctx + 256, a.prefill_len + 256),
                        )
                    except Exception as e:  # record the failure and continue with other candidates
                        row = {"name": name, "layer_idx": layer_idx, "error": str(e)[:400], "config": opt.describe()}
                        print("FAILED", name, layer_idx, str(e)[:200])
                    print(json.dumps({k: v for k, v in row.items() if k != "config"}))
                    f.write(json.dumps(row) + "\n")
                    f.flush()
    finally:
        ttnn.close_mesh_device(dev)
