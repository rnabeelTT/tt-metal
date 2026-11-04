# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Traced DFlash loop (tt/dflash_traced.py: device drafter + MTP verify pass / commit) vs plain greedy on the full model:
tokens equal to greedy, tokens per round, per-program walls, ms per emitted token.

    python -m models.autoports.qwen_qwen3_5_122b_a10b.tools.dflash_traced_run --new 128 [--prompts code,math] [--bs 16]
"""
from __future__ import annotations

import argparse
import time

import torch
from loguru import logger

import ttnn
from models.autoports.qwen_qwen3_5_122b_a10b.tools.dflash_acceptance import PROMPTS, drafter_dir, prefill_features
from models.autoports.qwen_qwen3_5_122b_a10b.tt.dflash_device import DFlashDevice
from models.autoports.qwen_qwen3_5_122b_a10b.tt.dflash_traced import DFlashTraced
from models.autoports.qwen_qwen3_5_122b_a10b.tt.generator import build_generator
from models.autoports.qwen_qwen3_5_122b_a10b.tt.mtp_sampling import SpecSampler
from models.autoports.qwen_qwen3_5_122b_a10b.tt.weights import resolve_checkpoint_dir

MODEL_DIR = "models/autoports/qwen_qwen3_5_122b_a10b"


def load_prompt_ctx(gen, runner, ids):
    """Eager prefill (context layers read back) -> the drafter's context over the prompt's last window. Returns
    (tok0, P)."""
    dr = runner.dr
    tok0, feats = prefill_features(gen, ids, dr.layer_ids, gen.args.dim)
    P = len(ids)
    dr.reset()
    BW = runner.BW
    first = max(0, P - dr.window)
    for s in range(first, P, BW):
        rows = feats[s : min(P, s + BW)]
        pad = torch.zeros(BW, rows.shape[1])
        pad[: rows.shape[0]] = rows
        host = ttnn.from_torch(pad.reshape(1, 1, BW, -1).to(torch.bfloat16), dtype=ttnn.bfloat16,
                               layout=ttnn.TILE_LAYOUT)
        ttnn.copy_host_to_device_tensor(host, runner.feats)
        runner.ctx(list(range(s, s + rows.shape[0])))
    ttnn.synchronize_device(gen.model.mesh_device)
    return tok0, P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default="code,math,prose,chat")
    ap.add_argument("--new", type=int, default=128)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--thinking", type=int, default=1)
    ap.add_argument("--no-greedy", action="store_true")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--top-p", type=float, default=0.95)
    args = ap.parse_args()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(resolve_checkpoint_dir()))
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 4), l1_small_size=24576, trace_region_size=512 << 20)
    gen = build_generator(MODEL_DIR, mesh, max_seq_len=8192, warm_decode_trace=False, warm_prefill_trace=False)
    drafter = DFlashDevice(gen.model, drafter_dir())
    runner = DFlashTraced(gen, drafter, args.bs)
    first_ids = None
    summary = {}
    for name in args.prompts.split(","):
        text = tok.apply_chat_template([{"role": "user", "content": PROMPTS[name]}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=bool(args.thinking))
        ids = tok(text, add_special_tokens=False)["input_ids"]
        greedy = None
        if not args.no_greedy and args.temperature <= 0:
            greedy = gen.generate(ids, args.new, enable_trace=False, on_device_sampling=False)
        if first_ids is None:
            # warm (compiles; advances state / KV with dummy rows) and capture on the first prompt's state
            load_prompt_ctx(gen, runner, ids)
            t0 = time.time()
            runner.warm()
            logger.info(f"warm {time.time() - t0:.1f}s")
            load_prompt_ctx(gen, runner, ids)
            t0 = time.time()
            runner.capture()
            logger.info(f"capture {time.time() - t0:.1f}s")
            first_ids = ids
        tok0, P = load_prompt_ctx(gen, runner, ids)
        for v in runner.timing.values():
            v.clear()
        sampler = SpecSampler(args.temperature, args.top_k, args.top_p, seed=1234) if args.temperature > 0 else None
        t0 = time.perf_counter()
        out, lens = runner.generate(tok0, P, args.new, sampler=sampler)
        wall = time.perf_counter() - t0
        n_emit = sum(lens)
        tau = n_emit / len(lens)
        agree = None if greedy is None else sum(int(a == b) for a, b in zip(greedy, out))
        first = None if greedy is None else next((i for i, (a, b) in enumerate(zip(greedy, out)) if a != b), None)
        tm = runner.timing_summary()
        summary[name] = (tau, wall / n_emit * 1e3)
        print(
            f"DFLASH_TRACED {name}: bs {args.bs} T {args.temperature} rounds {len(lens)} tau {tau:.2f} "
            f"ms/token {wall / n_emit * 1e3:.2f} ms/round {wall / len(lens) * 1e3:.2f} | greedy agree {agree}/{args.new} "
            f"first diff {first} | walls(ms) "
            + " ".join(f"{k} {v[1]:.2f}" for k, v in tm.items() if v[0]),
            flush=True,
        )
        print(f"  dflash: {tok.decode(out)!r}", flush=True)
        if greedy is not None and first is not None:
            print(f"  greedy: {tok.decode(greedy)!r}", flush=True)
    taus = [v[0] for v in summary.values()]
    mss = [v[1] for v in summary.values()]
    print(f"DFLASH_TRACED mean tau {sum(taus) / len(taus):.2f} mean ms/token {sum(mss) / len(mss):.2f}", flush=True)
    runner.release()
    gen.teardown()
    ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
