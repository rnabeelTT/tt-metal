# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Bit-level determinism probe of the optimized layer: repeat prefill (fresh state) and decode with identical inputs and count
bit-different outputs per policy variant. Used to localize the stress-test flake (which op family is non-deterministic).

    python models/autoports/qwen_qwen3_5_122b_a10b/tools/check_determinism.py [layer_idx] [repeats]
"""
import os
import sys

import torch

import ttnn
from models.autoports.qwen_qwen3_5_122b_a10b.tests.test_functional_decoder import _Harness, _hidden
from models.autoports.qwen_qwen3_5_122b_a10b.tt.functional_decoder import FunctionalDecoder
from models.autoports.qwen_qwen3_5_122b_a10b.tt.fused_decoder import FusedDecoder, FusionFlags
from models.autoports.qwen_qwen3_5_122b_a10b.tt.optimized_decoder import SELECTED, OptimizedDecoder

os.environ.setdefault("HF_MODEL", "Qwen/Qwen3.5-122B-A10B")


def probe(dev, layer_idx, name, opt, repeats, T=37, decoder_cls=OptimizedDecoder, batch=1):
    kwargs = {"opt": opt} if decoder_cls is OptimizedDecoder else {}
    h = _Harness(
        dev,
        layer_idx,
        batch=batch,
        max_seq_len=2048,
        real=True,
        hf=False,
        decoder_cls=decoder_cls,
        decoder_kwargs=kwargs,
    )
    hidden = _hidden(batch, T, h.H, seed=3100)
    dec_in = _hidden(batch, 1, h.H, seed=3101)
    outs, decs = [], []
    for _ in range(repeats):
        if h.kind == "linear_attention":
            h.tt.reset_gdn_state()
        outs.append(torch.cat([h.tt_prefill(hidden[u : u + 1], user_id=u) for u in range(batch)], dim=0))
        decs.append(h.tt_decode(dec_in, torch.full((batch,), T)))

    def report(label, xs):
        base = xs[0]
        diffs = [int((x != base).sum()) for x in xs[1:]]
        mx = max(((x.float() - base.float()).abs().max().item() for x in xs[1:]), default=0.0)
        print(
            f"{name} {h.kind} {label}: runs differing from run 0: {sum(d > 0 for d in diffs)}/{len(diffs)}, elements {diffs}, max abs {mx:.3e}"
        )
        return sum(d > 0 for d in diffs)

    return report("prefill", outs) + report("decode", decs)


def main():
    layer_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    repeats = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    dev = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=24576, trace_region_size=0, num_command_queues=1)
    bad = 0
    try:
        which = os.environ.get("DET_VARIANTS", "decode")
        if which == "selected":
            variants = [("selected", SELECTED, OptimizedDecoder)]
        elif which == "b32":
            # batched decode: 32 users = one full 32-row tile through the gate op
            for name, opt in (("selected_b32", SELECTED),):
                bad += probe(dev, layer_idx, name, opt, repeats, decoder_cls=OptimizedDecoder, batch=32)
            variants = []
        elif which == "focus":
            variants = [
                ("selected", SELECTED, OptimizedDecoder),
                (
                    "selected_router_functional",
                    SELECTED.with_(fusion=FusionFlags(router_gate_op=False)),
                    OptimizedDecoder,
                ),
                ("selected_sorted_at_37", SELECTED.with_(sorted_min_tokens=1), OptimizedDecoder),
                ("fused", None, FusedDecoder),
            ]
        elif which == "prefill":
            # prefill-path suspects: stage-1 functional graph, stage-2 fused graph, functional router in the optimized graph,
            # expert-sorted dispatch (exact nnz) instead of the masked union (runtime nnz inference)
            variants = [
                ("functional", None, FunctionalDecoder),
                ("fused", None, FusedDecoder),
                (
                    "selected_router_functional",
                    SELECTED.with_(fusion=FusionFlags(router_gate_op=False)),
                    OptimizedDecoder,
                ),
                ("selected_sorted_at_37", SELECTED.with_(sorted_min_tokens=1), OptimizedDecoder),
                ("selected", SELECTED, OptimizedDecoder),
            ]
        else:
            variants = [
                ("selected", SELECTED, OptimizedDecoder),
                ("dense_1d", SELECTED.with_(dense_decode_mode="1d"), OptimizedDecoder),
                ("dense_default", SELECTED.with_(dense_decode_mode="default"), OptimizedDecoder),
                ("norm_interleaved", SELECTED.with_(decode_norm_sharded=False), OptimizedDecoder),
                (
                    "router_ttnn_shared_none",
                    SELECTED.with_(router_decode_pc="ttnn", shared_decode_cores=None),
                    OptimizedDecoder,
                ),
            ]
        for name, opt, cls in variants:
            bad += probe(dev, layer_idx, name, opt, repeats, decoder_cls=cls)
    finally:
        ttnn.close_mesh_device(dev)
    print("RESULT", "DETERMINISTIC" if bad == 0 else f"NONDETERMINISTIC ({bad} differing runs)")


if __name__ == "__main__":
    main()
