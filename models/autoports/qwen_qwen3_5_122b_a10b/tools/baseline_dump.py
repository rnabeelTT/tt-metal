# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Single-chip baseline dump for the stage-4 multichip tests.

Opens ONE Blackhole die (a (1,1) mesh, like ``tools/sweep_optimized.py``), builds the stage-3 ``OptimizedDecoder``
(``SELECTED`` policy) for one layer per kind (L0 = linear_attention, L3 = full_attention) with synthetic and/or real
weights, runs the fixed-seed cases the mesh tests replay (``tests/test_multichip_decoder.py``), and saves everything
needed to compare the (1,4) ``MultichipDecoder`` against the single-chip graph WITHOUT re-running it in the mesh session:

* per prefill length T in :data:`SEQ_LENS`: input hidden states, prefill output, a decode step at position T
  (input + output), the paged K/V content by position (attention; read back through the page table, compared per
  512-position window by the tests) and the GDN state (recurrent + fused conv window) after the prefill and after the decode;
* the chunked long-prompt case (:data:`CHUNK_LENS` = 2048 + 2048 + 333 as ``start_pos`` chunks): per-chunk outputs, the
  K/V content for all 4429 positions, the GDN state, and the decode at position 4429;
* the page table and positions used, the policy (``SELECTED.describe()``) and the git SHA.

Usage (device must be free; ~a few minutes per layer with real weights)::

    python models/autoports/qwen_qwen3_5_122b_a10b/tools/baseline_dump.py \\
        --out models/autoports/qwen_qwen3_5_122b_a10b/doc/multichip_decoder/baseline [--layers 0,3] \\
        [--weights synthetic,real] [--seq-lens 5,37,129,333,2048] [--skip-chunked]

The mesh tests find the dump through ``QWEN35_BASELINE_DUMP_DIR`` (default: the ``--out`` default above).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import torch

import ttnn
from models.autoports.qwen_qwen3_5_122b_a10b.tests import reference as ref
from models.autoports.qwen_qwen3_5_122b_a10b.tests.test_functional_decoder import (
    BLOCK_SIZE,
    _Harness,
    _hidden,
    _tt_kv_by_position,
)
from models.autoports.qwen_qwen3_5_122b_a10b.tt.optimized_decoder import SELECTED, OptimizedDecoder

os.environ.setdefault("HF_MODEL", "Qwen/Qwen3.5-122B-A10B")

# ---- the fixed cases (single source of truth for the tool AND tests/test_multichip_decoder.py) ------------------------
SEQ_LENS = (5, 37, 129, 333, 2048)
CHUNK_LENS = (2048, 2048, 333)
MAX_SEQ_LEN = 8192  # covers the chunked 4429 + decode and keeps ONE page table per layer for every case
DEFAULT_OUT = Path(__file__).resolve().parents[1] / "doc" / "multichip_decoder" / "baseline"
LAYERS = (0, 3)  # one layer per kind: linear_attention (L0), full_attention (L3)


def prefill_seed(T: int) -> int:
    """Hidden-state seed of the length-T prefill case (matches the stage-1..3 tests' ``100 + T`` convention)."""
    return 100 + T


def decode_seed(T: int) -> int:
    return 200 + T


CHUNKED_SEED, CHUNKED_DECODE_SEED = 800, 801  # as test_*_chunked_prefill_matches_hf


def dump_path(out_dir: str | Path, layer_idx: int, weights: str) -> Path:
    return Path(out_dir) / f"baseline_L{layer_idx}_{weights}.pt"


def layer_state(text, layer_idx: int, weights: str):
    """``weights``: "synthetic" (tests/weight_stats.json statistics, seed = layer_idx) or "real" (checkpoint layer)."""
    if weights == "real":
        return ref.real_layer_state(layer_idx)
    return ref.synthetic_layer_state(text, layer_idx, seed=layer_idx)


# ---- state readers -------------------------------------------------------------------------------------------------
def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent, text=True).strip()
    except Exception:  # pragma: no cover
        return "unknown"


def _kv_snapshot(h: _Harness, num_positions: int) -> dict:
    """Paged K/V of user 0 by position -> {"k": [n_kv, num_positions, D] fp32, "v": ...} (bfp8 cache upcast on read)."""
    k, v = _tt_kv_by_position(h, num_positions)
    return {"k": k.float().clone(), "v": v.float().clone()}


def _gdn_snapshot(h: _Harness) -> dict:
    """GDN state buffers: recurrent ``[B, Nv, Dk, Dv]`` and the fused conv window ``[B, K-1, q+k+v]`` (oldest first)."""
    st = h.tt.gdn_state
    return {
        "recurrent": ttnn.to_torch(st["recurrent"]).float().clone(),
        "conv": ttnn.to_torch(st["conv"]).float().clone(),
    }


def _state_snapshot(h: _Harness, num_positions: int) -> dict:
    return _kv_snapshot(h, num_positions) if h.kind == "full_attention" else _gdn_snapshot(h)


# ---- cases ---------------------------------------------------------------------------------------------------------
def run_prefill_case(h: _Harness, T: int) -> dict:
    """Prefill T tokens from scratch (user 0, start_pos 0), snapshot the state, decode one token at position T."""
    if h.kind == "linear_attention":
        h.tt.reset_gdn_state()
    hidden = _hidden(1, T, h.H, seed=prefill_seed(T))
    t0 = time.time()
    out = h.tt_prefill(hidden)
    ttnn.synchronize_device(h.device)
    prefill_s = time.time() - t0
    case = {
        "T": T,
        "hidden": hidden,
        "out": out.clone(),
        "state_after_prefill": _state_snapshot(h, T),
        "prefill_wall_s": prefill_s,
    }
    dec = _hidden(1, 1, h.H, seed=decode_seed(T))
    pos = torch.tensor([T])
    t0 = time.time()
    dec_out = h.tt_decode(dec, pos)
    ttnn.synchronize_device(h.device)
    case.update(
        {
            "decode_hidden": dec,
            "decode_pos": int(pos.item()),
            "decode_out": dec_out.clone(),
            "state_after_decode": _state_snapshot(h, T + 1),
            "decode_wall_s": time.time() - t0,
        }
    )
    return case


def run_chunked_case(h: _Harness, lens=CHUNK_LENS) -> dict:
    """Prefill sum(lens) tokens as start_pos chunks (state carried), snapshot, decode at position sum(lens)."""
    total = sum(lens)
    if h.kind == "linear_attention":
        h.tt.reset_gdn_state()
    hidden = _hidden(1, total, h.H, seed=CHUNKED_SEED)
    outs, start = [], 0
    for n in lens:
        outs.append(h.tt_prefill(hidden[:, start : start + n], start=start).clone())
        start += n
    case = {
        "lens": list(lens),
        "hidden": hidden,
        "chunk_outs": outs,
        "out": torch.cat(outs, dim=1),
        "state_after_prefill": _state_snapshot(h, total),
    }
    dec = _hidden(1, 1, h.H, seed=CHUNKED_DECODE_SEED)
    pos = torch.tensor([total])
    case.update(
        {
            "decode_hidden": dec,
            "decode_pos": total,
            "decode_out": h.tt_decode(dec, pos).clone(),
            "state_after_decode": _state_snapshot(h, total + 1),
        }
    )
    return case


def build_baseline(device, layer_idx: int, weights: str, state=None) -> _Harness:
    """Stage-3 OptimizedDecoder (SELECTED) for ``layer_idx`` on one die; no HF layer (the tests hold the HF reference)."""
    cfg, text = ref.text_config()
    if state is None:
        state = layer_state(text, layer_idx, weights)
    return _Harness(
        device,
        layer_idx,
        batch=1,
        max_seq_len=MAX_SEQ_LEN,
        hf=False,
        state=state,
        decoder_cls=OptimizedDecoder,
        decoder_kwargs={"opt": SELECTED},
    )


def dump_layer(device, layer_idx: int, weights: str, seq_lens=SEQ_LENS, chunked=True) -> dict:
    h = build_baseline(device, layer_idx, weights)
    dump = {
        "meta": {
            "model": os.environ["HF_MODEL"],
            "layer_idx": layer_idx,
            "kind": h.kind,
            "weights": weights,
            "weights_seed": layer_idx if weights == "synthetic" else None,
            "hidden_size": h.H,
            "block_size": BLOCK_SIZE,
            "max_seq_len": MAX_SEQ_LEN,
            "opt": SELECTED.describe(),
            "kv_dtype": str(h.tt.kv_cache["k"].dtype) if h.kind == "full_attention" else None,
            "git_sha": _git_sha(),
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "seq_lens": list(seq_lens),
            "chunk_lens": list(CHUNK_LENS) if chunked else None,
        },
        "page_table": h.page_table.clone() if h.page_table is not None else None,
        "prefill": {},
    }
    for T in seq_lens:
        case = run_prefill_case(h, T)
        dump["prefill"][T] = case
        print(
            json.dumps(
                {
                    "layer": layer_idx,
                    "weights": weights,
                    "T": T,
                    "prefill_ms": round(1e3 * case["prefill_wall_s"], 1),
                    "decode_ms": round(1e3 * case["decode_wall_s"], 2),
                    "out_std": round(float(case["out"].float().std()), 4),
                }
            )
        )
    if chunked:
        dump["chunked"] = run_chunked_case(h)
        print(json.dumps({"layer": layer_idx, "weights": weights, "chunked": list(CHUNK_LENS), "done": True}))
    if h.kind == "full_attention":
        ttnn.deallocate(h.tt.kv_cache["k"])
        ttnn.deallocate(h.tt.kv_cache["v"])
    return dump


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--layers", default=",".join(str(l) for l in LAYERS))
    ap.add_argument("--weights", default="synthetic,real", help="comma list of synthetic|real")
    ap.add_argument("--seq-lens", default=",".join(str(t) for t in SEQ_LENS))
    ap.add_argument("--skip-chunked", action="store_true")
    a = ap.parse_args()
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    layers = [int(s) for s in a.layers.split(",")]
    weights = [w.strip() for w in a.weights.split(",") if w.strip()]
    seq_lens = tuple(int(s) for s in a.seq_lens.split(","))
    for w in weights:
        assert w in ("synthetic", "real"), w
    # One die, same device knobs as the single-chip tests (l1_small for the GDN conv1d; trace region for traced decode).
    dev = ttnn.open_mesh_device(
        ttnn.MeshShape(1, 1), l1_small_size=24576, trace_region_size=64_000_000, num_command_queues=1
    )
    try:
        for layer_idx in layers:
            for w in weights:
                t0 = time.time()
                dump = dump_layer(dev, layer_idx, w, seq_lens=seq_lens, chunked=not a.skip_chunked)
                path = dump_path(out_dir, layer_idx, w)
                torch.save(dump, path)
                print(f"saved {path} ({path.stat().st_size / 1e6:.1f} MB, {time.time() - t0:.0f}s)")
    finally:
        ttnn.close_mesh_device(dev)


if __name__ == "__main__":
    main()
