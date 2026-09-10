#!/usr/bin/env python3
"""Attribute one decode step's DEVICE time to modules.

Wraps each decoder-layer child on the INSTANCE (LightweightModule.__call__ delegates to
self.forward, so instance patching catches every call) and brackets it with
synchronize_device. The absolute total comes out higher than a traced step -- every
bracket is a full pipeline drain -- but the SPLIT is what this is for: it says which
module to optimize, which the device profiler could not, because its per-op records
carry no module identity.
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import time

import torch

import ttnn
from models.autoports.ornith_ai_ornith_1_5_35b_a3b.tt.model import OrnithModel
from models.autoports.ornith_ai_ornith_1_5_35b_a3b.tt.model_config import OrnithModelArgs
from models.autoports.ornith_ai_ornith_1_5_35b_a3b.tt.weight_mapping import load_ornith_state_dict
from models.demos.blackhole.qwen36.tt.model_config import GDN_CONV1D_L1_SMALL_SIZE

MESH_SHAPES = {"P150": (1, 1), "P300": (1, 2), "P150x4": (1, 4)}

acc = collections.Counter()
calls = collections.Counter()


def timed(name, fn, mesh):
    def wrapper(*a, **k):
        ttnn.synchronize_device(mesh)
        t0 = time.perf_counter()
        out = fn(*a, **k)
        ttnn.synchronize_device(mesh)
        acc[name] += time.perf_counter() - t0
        calls[name] += 1
        return out

    return wrapper


def instrument(model, mesh):
    """Patch every decoder-layer child, tagged by layer TYPE so the 30 linear-attention
    layers and the 10 full-attention layers are reported apart."""
    for i, layer in enumerate(model.layers):
        kind = "attn_full" if getattr(layer, "is_full_attention", False) else "attn_gdn"
        for attr, tag in (("attention_norm", "norm"), ("ff_norm", "norm")):
            mod = getattr(layer, attr, None)
            if mod is not None:
                mod.forward = timed(tag, mod.forward, mesh)
        ff = getattr(layer, "feed_forward", None)
        if ff is not None:
            ff.forward = timed("moe", ff.forward, mesh)
        att = getattr(layer, "attention", None)
        if att is not None:
            for m in [x for x in dir(att) if x.startswith("forward")]:
                f = getattr(att, m, None)
                if callable(f):
                    setattr(att, m, timed(kind, f, mesh))
    if hasattr(model, "_lm_head"):
        model._lm_head = timed("lm_head", model._lm_head, mesh)


ORIG: dict = {}


def _snapshot(model):
    """Remember every forward we may replace, so ablations can be undone between runs."""
    for i, layer in enumerate(model.layers):
        att = getattr(layer, "attention", None)
        ff = getattr(layer, "feed_forward", None)
        if att is not None:
            ORIG[("att", i)] = att.forward
        if ff is not None:
            ORIG[("ff", i)] = ff.forward
    ORIG[("lm_head", 0)] = model._lm_head


def _restore(model):
    for i, layer in enumerate(model.layers):
        if ("att", i) in ORIG:
            layer.attention.forward = ORIG[("att", i)]
        if ("ff", i) in ORIG:
            layer.feed_forward.forward = ORIG[("ff", i)]
    model._lm_head = ORIG[("lm_head", 0)]


def _stub(orig):
    """Replace a module with something shape-correct and nearly free.

    The real forward runs ONCE to learn the output shape, then every later call returns
    a clone of a cached zero tensor. Clone rather than the tensor itself because callers
    deallocate what they are handed; a clone is one cheap op, which is charged to the
    ablated configuration and so slightly UNDER-states what the module costs.
    """
    box = {}

    def stub(*a, **k):
        if "z" not in box:
            out = orig(*a, **k)
            box["z"] = ttnn.mul(out, 0.0)
            return out
        return ttnn.clone(box["z"])

    return stub


def ablate(model, which):
    _restore(model)
    n = 0
    for i, layer in enumerate(model.layers):
        full = bool(getattr(layer, "is_full_attention", False))
        if which == "attn_full" and full:
            layer.attention.forward = _stub(ORIG[("att", i)])
            n += 1
        elif which == "attn_gdn" and not full:
            layer.attention.forward = _stub(ORIG[("att", i)])
            n += 1
        elif which == "moe":
            layer.feed_forward.forward = _stub(ORIG[("ff", i)])
            n += 1
    if which == "lm_head":
        model._lm_head = _stub(ORIG[("lm_head", 0)])
        n = 1
    return n


def traced_step_ms(model, args, mesh, steps):
    """Pure device time for one decode step: capture, then replay with no host work."""
    from models.demos.blackhole.qwen36.tt.generator_interface import prime_decode_trace
    from models.tt_transformers.tt.generator import Generator

    tok = torch.full((1, 1), 1234, dtype=torch.long)
    pos = lambda p: torch.full((1,), p, dtype=torch.long)
    gen = Generator([model], [model.args], mesh)
    prime_decode_trace(gen, model, tok, pos(64), None)
    for _ in range(2):
        gen.decode_forward(tok, pos(64), page_table=None, kv_cache=None, enable_trace=True, read_from_device=False)
    ttnn.synchronize_device(mesh)
    tid = list(gen.trace_ids_decode[False].values())[0]
    ttnn.synchronize_device(mesh)
    t0 = time.perf_counter()
    for _ in range(steps):
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh)
    return (time.perf_counter() - t0) / steps * 1e3


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", default="P150", choices=sorted(MESH_SHAPES))
    ap.add_argument("--weights", required=True)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--mode", default="eager", choices=("eager", "ablate"))
    a = ap.parse_args()

    rows, cols = MESH_SHAPES[a.mesh]
    mesh = ttnn.open_mesh_device(
        ttnn.MeshShape(rows, cols),
        l1_small_size=GDN_CONV1D_L1_SMALL_SIZE,
        num_command_queues=2,
        trace_region_size=1024 * 1024 * 1024,
    )
    weights = pathlib.Path(a.weights)
    cfg = json.loads((weights / "config.json").read_text())["text_config"]
    args = OrnithModelArgs(mesh, max_batch_size=1, max_seq_len=256)
    cache = pathlib.Path(args.weight_cache_path(args.routed_expert_dtype))
    cache.mkdir(parents=True, exist_ok=True)
    model = OrnithModel(mesh, args, load_ornith_state_dict(weights, cfg), tensor_cache_path=cache)

    ids = torch.randint(1000, 40000, (1, 64))
    model.prefill(ids)
    ttnn.synchronize_device(mesh)

    tok = torch.full((1, 1), 1234, dtype=torch.long)
    pos = lambda p: torch.full((1,), p, dtype=torch.long)
    model.decode(tok, pos(64))  # warm / compile before instrumenting
    ttnn.synchronize_device(mesh)

    if a.mode == "ablate":
        _snapshot(model)
        base = traced_step_ms(model, args, mesh, max(a.steps, 16))
        print(f"\n=== TRACED decode attribution by ablation, {a.mesh} ===")
        print(f"{'ablated':<12} {'ms/token':>9} {'saved':>8} {'per call':>9}  share")
        print(f"{'(baseline)':<12} {base:9.2f}")
        for which in ("attn_full", "attn_gdn", "moe", "lm_head"):
            try:
                n = ablate(model, which)
                ms = traced_step_ms(model, args, mesh, max(a.steps, 16))
                saved = base - ms
                print(
                    f"{which:<12} {ms:9.2f} {saved:8.2f} {saved/max(n,1)*1e3:9.1f}us " f"{saved/base*100:5.1f}%",
                    flush=True,
                )
            except Exception as e:
                print(f"{which:<12}  FAILED: {type(e).__name__}: {str(e).splitlines()[0][:80]}", flush=True)
        _restore(model)
        ttnn.close_mesh_device(mesh)
        return 0

    instrument(model, mesh)
    t0 = time.perf_counter()
    for i in range(a.steps):
        model.decode(tok, pos(65 + i))
    ttnn.synchronize_device(mesh)
    wall = (time.perf_counter() - t0) / a.steps

    print(f"\n=== decode attribution, {a.mesh}, {a.steps} steps ===")
    print(f"{'module':<12} {'ms/token':>9} {'calls/tok':>10} {'us/call':>9}  share")
    total = sum(acc.values()) / a.steps
    for name, tot in sorted(acc.items(), key=lambda kv: -kv[1]):
        per = tot / a.steps
        n = calls[name] / a.steps
        print(f"{name:<12} {per*1e3:9.2f} {n:10.0f} {per/max(n,1)*1e6:9.1f}  {per/total*100:5.1f}%")
    print(f"{'MEASURED':<12} {total*1e3:9.2f}   (instrumented modules)")
    print(f"{'WALL':<12} {wall*1e3:9.2f}   (whole eager step, incl. sync overhead + rest)")
    print(f"{'UNATTRIB':<12} {(wall-total)*1e3:9.2f}   (embedding, router glue, dispatch, sync tax)")
    ttnn.close_mesh_device(mesh)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
