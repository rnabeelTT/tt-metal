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
    if getattr(runner, "fused", False):
        runner.set_base(None)  # the prefill wrote the bound state
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


def prof_pieces(mesh, runner):
    """Each trace replayed alone (device time + one dispatch), and the drafter's pieces traced on their own."""
    reps = {}
    commit = f"commit{runner.BW - 1}" if runner.fused else "commit"  # fused: the phase-end materialize
    for nm in ("pass", "cands_pass", commit, "ctx", "draft", "cands_draft"):
        ttnn.synchronize_device(mesh)
        t1 = time.perf_counter()
        for _ in range(5):
            ttnn.execute_trace(mesh, runner.traces[nm], cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh)
        reps[nm] = (time.perf_counter() - t1) / 5 * 1e3
    print(f"DFLASH_TRACE_TIMES bs {runner.bs} " + " ".join(f"{k} {v:.2f}" for k, v in reps.items()), flush=True)
    dr = runner.dr
    x_in = dr.embed_body()
    pieces = {
        "embed": lambda: dr.embed_body(),
        "layer0": lambda: dr.layer_body(dr.layers[0], ttnn.clone(x_in)),
        "head": lambda: dr.head_body(ttnn.clone(x_in)),
    }
    pt = {}
    for nm, fn in pieces.items():
        outs = fn()
        for t in outs if isinstance(outs, tuple) else (outs,):
            ttnn.deallocate(t)
        ttnn.synchronize_device(mesh)
        tid = ttnn.begin_trace_capture(mesh, cq_id=0)
        outs = fn()
        ttnn.end_trace_capture(mesh, tid, cq_id=0)
        ttnn.synchronize_device(mesh)
        t1 = time.perf_counter()
        for _ in range(5):
            ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh)
        pt[nm] = (time.perf_counter() - t1) / 5 * 1e3
        ttnn.release_trace(mesh, tid)
    ttnn.deallocate(x_in)
    print("DFLASH_DRAFT_PIECES " + " ".join(f"{k} {v:.2f}" for k, v in pt.items()), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default="code,math,prose,chat")
    ap.add_argument("--new", type=int, default=128)
    ap.add_argument("--bs", default="16", help="block sizes, comma separated (one runner each)")
    ap.add_argument("--thinking", type=int, default=1)
    ap.add_argument("--no-greedy", action="store_true")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--prof", type=int, default=1, help="trace timings of the pieces")
    args = ap.parse_args()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(resolve_checkpoint_dir()))
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 4), l1_small_size=24576, trace_region_size=768 << 20)
    gen = build_generator(MODEL_DIR, mesh, max_seq_len=8192, warm_decode_trace=False, warm_prefill_trace=False)
    drafter = DFlashDevice(gen.model, drafter_dir())
    prompts = {}
    for name in args.prompts.split(","):
        text = tok.apply_chat_template([{"role": "user", "content": PROMPTS[name]}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=bool(args.thinking))
        prompts[name] = tok(text, add_special_tokens=False)["input_ids"]
    greedy = {}
    if not args.no_greedy and args.temperature <= 0:
        for name, ids in prompts.items():
            greedy[name] = gen.generate(ids, args.new, enable_trace=False, on_device_sampling=False)
    first = next(iter(prompts.values()))
    # every runner's programs compile before ANY capture (a program first compiled after a capture can have its
    # binaries overwritten by a later replay, tt-metal #48536)
    runners = [DFlashTraced(gen, drafter, int(b)) for b in args.bs.split(",")]
    for runner in runners:
        load_prompt_ctx(gen, runner, first)
        t0 = time.time()
        runner.warm()
        logger.info(f"bs {runner.bs}: warm {time.time() - t0:.1f}s")
    if args.prof:  # the drafter piece programs too
        x_in = drafter.embed_body()
        for o in (drafter.layer_body(drafter.layers[0], ttnn.clone(x_in)),) + tuple(drafter.head_body(ttnn.clone(x_in))):
            ttnn.deallocate(o)
        ttnn.deallocate(x_in)
    for runner in runners:
        load_prompt_ctx(gen, runner, first)
        runner.capture()
    if args.prof:
        for runner in runners:
            prof_pieces(mesh, runner)
    for runner in runners:
        summary = {}
        for name, ids in prompts.items():
            tok0, P = load_prompt_ctx(gen, runner, ids)
            for v in runner.timing.values():
                v.clear()
            sampler = SpecSampler(args.temperature, args.top_k, args.top_p, seed=1234) if args.temperature > 0 else None
            t0 = time.perf_counter()
            out, lens = runner.generate(tok0, P, args.new, sampler=sampler)
            wall = time.perf_counter() - t0
            n_emit = sum(lens)
            tau = n_emit / len(lens)
            g = greedy.get(name)
            agree = None if g is None else sum(int(a == b) for a, b in zip(g, out))
            fd = None if g is None else next((i for i, (a, b) in enumerate(zip(g, out)) if a != b), None)
            tm = runner.timing_summary()
            summary[name] = (tau, wall / n_emit * 1e3)
            print(
                f"DFLASH_TRACED {name}: bs {runner.bs} T {args.temperature} rounds {len(lens)} tau {tau:.2f} "
                f"ms/token {wall / n_emit * 1e3:.2f} ms/round {wall / len(lens) * 1e3:.2f} | greedy agree "
                f"{agree}/{args.new} first diff {fd} | walls(ms) "
                + " ".join(f"{k} {v[1]:.2f}" for k, v in tm.items() if v[0]),
                flush=True,
            )
            print(f"  dflash: {tok.decode(out)!r}", flush=True)
        taus = [v[0] for v in summary.values()]
        mss = [v[1] for v in summary.values()]
        print(f"DFLASH_TRACED bs {runner.bs} mean tau {sum(taus) / len(taus):.2f} mean ms/token "
              f"{sum(mss) / len(mss):.2f}", flush=True)
    for runner in runners:
        runner.release()
    gen.teardown()
    ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
