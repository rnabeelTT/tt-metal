#!/usr/bin/env python3
"""Cost per LAYER TYPE for traced decode, by building partial models.

Ablating a module inside a captured trace deadlocks the device, and eager timing measures
a different code path (model.decode takes no page table, so full attention falls back to
its concat branch). Building a real model from a SUBSET of layers avoids both: every
graph is a graph the model actually runs, and `layer_types` lets the subset be all-GDN or
all-full-attention, which separates the two.

    3 GDN : 1 full attention, repeating -- layer 3, 7, 11 ... are the full-attention ones.

Run several sizes and fit: slope = per-layer cost, intercept = embedding + final norm +
lm_head + whatever else does not scale with depth.
"""
from __future__ import annotations

import argparse
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
FULL_ATTN = lambda i: (i % 4) == 3


def pick(spec: str, n: int) -> list[int]:
    if spec == "mixed":
        return list(range(n))
    if spec == "gdn":
        return [i for i in range(40) if not FULL_ATTN(i)][:n]
    if spec == "full":
        return [i for i in range(40) if FULL_ATTN(i)][:n]
    raise SystemExit(f"unknown spec {spec}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", default="P150", choices=sorted(MESH_SHAPES))
    ap.add_argument("--weights", required=True)
    ap.add_argument("--spec", default="mixed", choices=("mixed", "gdn", "full"))
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--steps", type=int, default=32)
    a = ap.parse_args()

    idx = pick(a.spec, a.n)
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
    args.layer_indices = idx
    cache = pathlib.Path(args.weight_cache_path(args.routed_expert_dtype))
    model = OrnithModel(mesh, args, load_ornith_state_dict(weights, cfg), tensor_cache_path=cache)

    from models.demos.blackhole.qwen36.tt.generator_interface import prime_decode_trace
    from models.tt_transformers.tt.generator import Generator

    # Paged KV cache + identity page table. NOT optional: attention only takes its decode
    # branch when `use_paged_attention and T == 1`, so without a page table a single token
    # falls into the CONCAT PREFILL branch, which is both wrong for decode and untraceable
    # -- capture dies with "Cannot load new binaries during trace capture" from inside the
    # prefill path rather than saying the cache was never allocated.
    BLOCK = 64
    num_blocks = ((max(64, (256 + BLOCK - 1) // BLOCK) + 7) // 8) * 8
    model.allocate_kv_caches([num_blocks, args.n_local_kv_heads, BLOCK, args.head_dim], ttnn.bfloat16, batch_size=1)
    page_table = torch.arange(num_blocks, dtype=torch.int32).reshape(1, num_blocks)

    ids = torch.randint(1000, 40000, (1, 64))
    model.prefill(ids)
    ttnn.synchronize_device(mesh)
    tok = torch.full((1, 1), 1234, dtype=torch.long)
    pos = lambda p: torch.full((1,), p, dtype=torch.long)
    gen = Generator([model], [model.args], mesh)
    prime_decode_trace(gen, model, tok, pos(64), page_table)
    for _ in range(2):
        gen.decode_forward(
            tok, pos(64), page_table=page_table, kv_cache=None, enable_trace=True, read_from_device=False
        )
    ttnn.synchronize_device(mesh)

    tid = list(gen.trace_ids_decode[False].values())[0]
    t0 = time.perf_counter()
    for _ in range(a.steps):
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh)
    ms = (time.perf_counter() - t0) / a.steps * 1e3

    n_full = sum(1 for i in idx if FULL_ATTN(i))
    print(f"SWEEP {a.mesh} spec={a.spec} n={len(idx)} gdn={len(idx)-n_full} full={n_full} ms={ms:.3f}", flush=True)
    ttnn.close_mesh_device(mesh)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
