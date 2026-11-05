# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Where a prefill segment's time goes: a reduced model (one GDN + one attention layer by default) prefilled through the
served bucket path, every sub-block timed with a device sync around it (host walls; prefill ops are large, so dispatch
is a small share). Reports per layer kind and per piece, scaled to the full model's layer counts.

    python -m models.autoports.ornith_ai_ornith_1_5_35b_a3b.tools.prefill_prof --lens 2048,8192 [--layers 0,3]
Works for every Qwen3.5-family autoport (MODEL_DIR / the module path of the port given by --port).
"""
from __future__ import annotations

import argparse
import importlib
import time
from collections import defaultdict

import torch

import ttnn

ACC = defaultdict(float)
CNT = defaultdict(int)
SYNC = [None]


def wrap(obj, name, key):
    orig = getattr(obj, name)

    def f(*a, **k):
        SYNC[0]()
        t0 = time.perf_counter()
        out = orig(*a, **k)
        SYNC[0]()
        kk = key() if callable(key) else key
        ACC[kk] += time.perf_counter() - t0
        CNT[kk] += 1
        return out

    setattr(obj, name, f)
    return orig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="ornith_ai_ornith_1_5_35b_a3b")
    ap.add_argument("--lens", default="2048,8192")
    ap.add_argument("--layers", default="0,3")
    ap.add_argument("--reps", type=int, default=2)
    args = ap.parse_args()
    base = f"models.autoports.{args.port}"
    gen_mod = importlib.import_module(f"{base}.tt.generator")
    model_dir = f"models/autoports/{args.port}"
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 4), l1_small_size=24576, trace_region_size=256 << 20)
    SYNC[0] = lambda: ttnn.synchronize_device(mesh)
    layer_indices = [int(x) for x in args.layers.split(",")]
    gen = gen_mod.build_generator(model_dir, mesh, max_seq_len=max(int(x) for x in args.lens.split(",")) + 256,
                                  warm_decode_trace=False, warm_prefill_trace=False, layer_indices=layer_indices)
    m = gen.model
    full_layers = int(getattr(gen.args, "num_hidden_layers_full", 0) or 0)
    # ---- timers: layer level
    for layer in m.layers:
        kind = "attn" if layer.is_full_attention else "gdn"
        wrap(layer, "forward", lambda kind=kind: f"{kind}.layer")
        wrap(layer.feed_forward, "forward", lambda kind=kind: f"{kind}.moe")
        if not layer.is_full_attention:
            dn = layer.attention
            wrap(dn, "_project_qkvzab", lambda: "gdn.mixer.in_proj")
    # ---- MoE phases (sorted path)
    od = importlib.import_module(f"{base}.tt.optimized_decoder")
    md = importlib.import_module(f"{base}.tt.moe_dispatch")
    wrap(md.SortedDispatch, "build", lambda: "moe.dispatch_build")
    for name in ("gather_rows", "undispatch_rows"):
        if hasattr(od, name):
            wrap(od, name, lambda name=name: f"moe.{name}")
    for layer in m.layers:
        ff = layer.feed_forward
        for nm in ("_route", "_shared"):
            if hasattr(ff, nm):
                wrap(ff, nm, lambda nm=nm: f"moe{nm}")
    for fn, key in ((ttnn, "sparse_matmul"), (ttnn, "swiglu")):
        wrap(fn, key, lambda key=key: f"moe.{key}")
    # ---- GDN chunk scan + conv
    gdn_tp = importlib.import_module("models.demos.blackhole.qwen36.tt.gdn.tp")
    for nm in ("_causal_conv1d_fir",):
        if hasattr(gdn_tp, nm):
            wrap(gdn_tp, nm, lambda nm=nm: f"gdn.mixer.conv")
    fc = importlib.import_module("models.demos.blackhole.qwen36.tt.gdn.fused_chunk")
    wrap(fc, "chunk_gated_delta_rule_fused_adapter", lambda: "gdn.mixer.chunk_scan")
    gdn_tp.chunk_gated_delta_rule_fused_adapter = fc.chunk_gated_delta_rule_fused_adapter
    tpc = importlib.import_module("models.demos.blackhole.qwen36.tt.tp_common")
    if hasattr(tpc, "matmul_reduce_scatter_prefill"):
        wrap(tpc, "matmul_reduce_scatter_prefill", lambda: "*.out_proj_mm_rs")
    ccl = importlib.import_module("models.tt_transformers.tt.ccl")
    wrap(ccl, "tt_all_reduce", lambda: "*.all_reduce")
    wrap(ccl, "tt_all_gather", lambda: "*.all_gather")
    for mod in (gdn_tp, importlib.import_module("models.demos.blackhole.qwen36.tt.attention.tp"),
                importlib.import_module(f"{base}.tt.moe")):
        for nm in ("tt_all_reduce", "tt_all_gather"):
            if hasattr(mod, nm):
                setattr(mod, nm, getattr(ccl, nm))
    pt = gen.page_table[:1]
    for L in [int(x) for x in args.lens.split(",")]:
        ids = torch.tensor([[1000 + (i * 7) % 9973 for i in range(L)]])
        for rep in range(args.reps + 1):
            ACC.clear()
            CNT.clear()
            SYNC[0]()
            t0 = time.perf_counter()
            gen.prefill_forward(ids, page_table=pt, prompt_lens=[L])
            SYNC[0]()
            wall = time.perf_counter() - t0
            if rep == 0:
                continue
            n_gdn = sum(1 for l in m.layers if not l.is_full_attention)
            n_att = len(m.layers) - n_gdn
            print(f"PREFILL_PROF len {L} rep {rep}: wall {wall * 1e3:.1f} ms for {n_gdn} gdn + {n_att} attn layers",
                  flush=True)
            for k in sorted(ACC):
                per = n_gdn if k.startswith(("gdn", "moe")) else (n_att if k.startswith("attn") else 1)
                print(f"  {k:28s} {ACC[k] * 1e3:9.2f} ms  ({CNT[k]} calls)", flush=True)
    gen.teardown()
    ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
