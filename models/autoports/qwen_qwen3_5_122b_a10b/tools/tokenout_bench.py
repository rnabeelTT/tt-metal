# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Token-out decode benchmark for the (1,4) Qwen3.5-122B-A10B generator (stage 7 evidence).

Measures, per decode step, on a warmed generator:
  blocking   - the readiness path: nonblocking replays + a blocking token readback each step
  deferred   - the serving contract: ``decode_step_async`` (submit step i+1, then read step i)
  replay     - model trace + sampler trace replays only, no readback (device floor incl. dispatch)
  model      - model trace replay only
  sampler    - sampler trace replay only (same logits each step)
and checks that blocking and deferred produce identical tokens. Writes a JSON report.

    python -m models.autoports.qwen_qwen3_5_122b_a10b.tools.tokenout_bench --layers all --steps 64 --out doc/.../x.json
"""
import argparse
import json
import os
import time
from pathlib import Path

import torch

import ttnn
from models.autoports.qwen_qwen3_5_122b_a10b.tools.readiness import _open_mesh
from models.autoports.qwen_qwen3_5_122b_a10b.tt.generator import GREEDY, build_generator

MODEL_DIR = "models/autoports/qwen_qwen3_5_122b_a10b"
PROMPT = "The capital of France is Paris, the capital of Germany is Berlin, the capital of Italy is"


def _timed(mesh, fn, n):
    ttnn.synchronize_device(mesh)
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    ttnn.synchronize_device(mesh)
    return 1e3 * (time.perf_counter() - t0) / n


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="all", help="'all' or comma-separated layer indices")
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--batch", type=int, default=1, help="max_batch_size of the generator (rows decoded)")
    ap.add_argument("--max-seq-len", type=int, default=4096)
    ap.add_argument("--trace-region", type=int, default=int(os.environ.get("QWEN35_TRACE_REGION", 512 * 1024 * 1024)))
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    layers = None if a.layers == "all" else [int(s) for s in a.layers.split(",")]
    mesh = _open_mesh(a.trace_region)
    try:
        gen = build_generator(MODEL_DIR, mesh, layer_indices=layers, max_seq_len=a.max_seq_len, max_batch_size=a.batch)
        B, N = a.batch, a.steps
        prompt = gen.tokenizer.encode(PROMPT)
        P = len(prompt)
        toks = torch.tensor(prompt).reshape(1, -1).repeat(B, 1)

        def start():
            gen.reset()
            lg = gen.prefill_forward(toks, page_table=gen.page_table[:B], prompt_lens=[P] * B)
            t = torch.argmax(lg[:, -1], dim=-1).reshape(B, 1)
            return gen.decode_forward(t, torch.full((B,), P), enable_trace=True, reload_inputs=True).reshape(B, 1)

        # blocking readback (readiness runner path)
        t = start()
        out_a = []
        ttnn.synchronize_device(mesh)
        t0 = time.perf_counter()
        for i in range(N):
            t = gen.decode_forward(t, torch.full((B,), P + 1 + i), enable_trace=True, reload_inputs=False).reshape(B, 1)
            out_a.append(t.clone())
        ttnn.synchronize_device(mesh)
        ms_blocking = 1e3 * (time.perf_counter() - t0) / N

        # deferred readback (serving contract): submit step i (replays + an enqueued nonblocking read), THEN resolve
        # step i-1's read. With on-device token feedback the host tokens are stale by one step and ignored by the
        # replay (reload_inputs=False), exactly as vLLM's async scheduling drives the model.
        t = start()
        out_b = []
        prev = None
        ttnn.synchronize_device(mesh)
        t0 = time.perf_counter()
        for i in range(N):
            pending = gen.decode_step_async(t, torch.full((B,), P + 1 + i), enable_trace=True, reload_inputs=False)
            if prev is not None:
                t = prev.result().reshape(B, 1)
                out_b.append(t.clone())
            prev = pending
        out_b.append(prev.result().reshape(B, 1))
        ttnn.synchronize_device(mesh)
        ms_deferred = 1e3 * (time.perf_counter() - t0) / N

        tt = gen._tt
        tids = tt.trace_ids_decode[True]
        logits = tt.trace_output_decode[True]

        def replay_model():
            for tid in tids.values():
                ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)

        def replay_sampler():
            tt.sample_decode_on_device(logits, GREEDY, enable_trace=True)

        def replay_both():
            replay_model()
            replay_sampler()

        ms_replay = _timed(mesh, replay_both, N)
        ms_model = _timed(mesh, replay_model, N)
        ms_sampler = _timed(mesh, replay_sampler, N)
        equal = all(torch.equal(x, y) for x, y in zip(out_a, out_b))
        first_diff = next((i for i, (x, y) in enumerate(zip(out_a, out_b)) if not torch.equal(x, y)), None)
        # run-to-run determinism of the stock path itself (same prompt, fresh reset): a divergence here is a
        # near-tie / device-order effect, not the readback contract
        t = start()
        out_c = []
        for i in range(N):
            t = gen.decode_forward(t, torch.full((B,), P + 1 + i), enable_trace=True, reload_inputs=False).reshape(B, 1)
            out_c.append(t.clone())
        first_diff_rerun = next((i for i, (x, y) in enumerate(zip(out_a, out_c)) if not torch.equal(x, y)), None)
        rep = {
            "layers": a.layers,
            "batch": B,
            "steps": N,
            "prompt_tokens": P,
            "ms_per_step": {
                "blocking_readback": round(ms_blocking, 3),
                "deferred_readback": round(ms_deferred, 3),
                "replay_model_plus_sampler": round(ms_replay, 3),
                "replay_model_only": round(ms_model, 3),
                "replay_sampler_only": round(ms_sampler, 3),
            },
            "tokens_per_s_per_user": {
                "blocking_readback": round(1e3 / ms_blocking, 2),
                "deferred_readback": round(1e3 / ms_deferred, 2),
            },
            "tokens_identical_blocking_vs_deferred": equal,
            "first_divergence_blocking_vs_deferred": first_diff,
            "first_divergence_blocking_rerun": first_diff_rerun,
            "tokens_blocking": [int(x[0, 0]) for x in out_a],
            "tokens_deferred": [int(x[0, 0]) for x in out_b],
            "tokens_blocking_rerun": [int(x[0, 0]) for x in out_c],
            "first_tokens": [int(x[0, 0]) for x in out_a[:8]],
            "text": gen.tokenizer.decode([int(x[0, 0]) for x in out_a]),
            "text_deferred": gen.tokenizer.decode([int(x[0, 0]) for x in out_b]),
        }
        print("TOKENOUT " + json.dumps(rep), flush=True)
        if a.out:
            a.out.parent.mkdir(parents=True, exist_ok=True)
            a.out.write_text(json.dumps(rep, indent=2) + "\n")
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()
