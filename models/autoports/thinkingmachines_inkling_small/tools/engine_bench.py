# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Dev benchmark of the serving engine (tt/inkling_serve.py) without vLLM: the spec_bench prompts, greedy, per k.
Prints tokens/step, tok/s, TTFT and the texts; with SPEC_REF=<spec_bench SPEC_OUT json> checks each text against the
harness's k=0 text. Dev signal only (served numbers come from the vLLM server).

usage (hybrid env as tools/eval_hybrid.sh, TOOL=engine_bench.py): engine_bench.py [K list, e.g. 1,2,3] [new tokens]
env: ENGINE_CTX (max_seq_len, default 8192), ENGINE_WARM (untimed warm-up generations, default 1), SPEC_TRACE_REGION
"""

import json
import os
import sys
import time

if os.environ.get("HYB_ENABLE") == "1":
    sys.path.insert(0, os.environ.get("HYB_POC_DIR", "/opt/tt-metal/models/experimental/hybrid_moe"))
    os.environ.setdefault("HYB_PORT", "thinkingmachines_inkling_small")
    import hybrid

    hybrid.install()

import ttnn  # noqa: E402

if os.environ.get("INKLING_WATCHDOG", "600") != "0":  # `docker kill -s USR2 <container>`: every thread's Python stack
    import faulthandler
    import signal

    faulthandler.register(signal.SIGUSR2, all_threads=True)

from models.autoports.thinkingmachines_inkling_small.tt.inkling_serve import InklingSpecServing  # noqa: E402
from models.autoports.thinkingmachines_inkling_small.tt.model import create_model  # noqa: E402

PROMPTS = [
    [
        {
            "role": "user",
            "content": "Write a Python function that returns the n-th Fibonacci number iteratively, with a "
            "docstring and two doctests.",
        }
    ],
    [
        {
            "role": "user",
            "content": "Natalia sold clips to 48 of her friends in April, and then she sold half as many "
            "clips in May. How many clips did Natalia sell altogether in April and May? Explain "
            "step by step.",
        }
    ],
    [{"role": "user", "content": "Explain in three short paragraphs how a hash map handles collisions."}],
]

if os.environ.get("ENGINE_PROMPTS") == "heldout":  # the served bench's prompts (serve/bench.py: prompts_v2.HELDOUT)
    sys.path.insert(0, os.environ.get("HYB_POC_DIR", "/opt/tt-metal/models/experimental/hybrid_moe"))
    from prompts_v2 import HELDOUT  # noqa: E402

    PROMPTS = [[{"role": "user", "content": p}] for p in HELDOUT]


def main():
    ks = [int(v) for v in (sys.argv[1] if len(sys.argv) > 1 else "1").split(",")]
    NEW = int(sys.argv[2]) if len(sys.argv) > 2 else 128
    ref = json.load(open(os.environ["SPEC_REF"]))["texts"]["k0"] if os.environ.get("SPEC_REF") else None
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(
        ttnn.MeshShape(1, 4),
        l1_small_size=24576,
        trace_region_size=int(os.environ.get("SPEC_TRACE_REGION", "400000000")),
    )
    try:
        args, model = create_model(mesh, max_seq_len=int(os.environ.get("ENGINE_CTX", "8192")))
        tok = args.tokenizer
        prompts = [
            tok.encode(
                tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True, reasoning_effort="none"),
                add_special_tokens=False,
            )
            for m in PROMPTS
        ]
        for K in ks:  # one engine per k (each captures its own traces; run one k per process for clean numbers)
            eng = InklingSpecServing(model, args, mesh, K, trace=os.environ.get("INKLING_TRACE", "1") == "1")
            eng.warm()
            eng.capture()
            for _ in range(int(os.environ.get("ENGINE_WARM", "1"))):
                eng.generate(prompts[0], 32)
            tot_n = tot_t = tot_steps = 0
            ttfts = []
            hyb = getattr(sys.modules.get("hybrid.patch"), "HYB", None)
            st0 = dict(hyb.st) if hyb is not None else {}
            v0 = eng.stats["t_verify"], eng.stats["t_draft"], eng.stats["steps"]
            for i, ids in enumerate(prompts):
                s0 = eng.stats["steps"]
                t0 = time.time()
                out = [eng.prefill(ids)]
                t1 = time.time()
                while len(out) < NEW and out[-1] not in eng.eos:
                    out += eng.step()
                t2 = time.time()
                out = out[:NEW]
                ttfts.append(t1 - t0)
                tot_n += len(out) - 1
                tot_t += t2 - t1
                tot_steps += eng.stats["steps"] - s0
                text = tok.decode(out)
                same = "" if ref is None else f" | equals harness k0: {ref[i][:len(text)] == text[:len(ref[i])]}"
                print(
                    f"  k={K} prompt {i}: {len(ids)} prompt tokens, TTFT {t1 - t0:.3f}s{same}\n    {text[:120]!r}",
                    flush=True,
                )
            print(
                f"k={K}: {tot_n / max(tot_steps, 1):.2f} tokens/step, {tot_n / tot_t:.2f} tok/s, TTFT mean "
                f"{sum(ttfts) / len(ttfts):.3f}s | verify {1e3 * eng.stats['t_verify'] / max(eng.stats['steps'], 1):.1f}"
                f" ms, draft {1e3 * eng.stats['t_draft'] / max(eng.stats['steps'], 1):.1f} ms per step",
                flush=True,
            )
            if hyb is not None:  # the hybrid overlay's host timers / counters over the timed steps, per step
                n = max(eng.stats["steps"] - v0[2], 1)
                d = {k: hyb.st[k] - st0.get(k, 0) for k in hyb.st if isinstance(hyb.st[k], (int, float))}
                print(
                    "  per step: verify %.1f ms, draft %.1f ms | hybrid ms: "
                    % (1e3 * (eng.stats["t_verify"] - v0[0]) / n, 1e3 * (eng.stats["t_draft"] - v0[1]) / n)
                    + " ".join(f"{k} {1e3 * v / n:.1f}" for k, v in sorted(d.items()) if k.startswith("t_") and v)
                    + " | counts: "
                    + " ".join(f"{k} {v / n:.1f}" for k, v in sorted(d.items()) if not k.startswith("t_") and v),
                    flush=True,
                )
            if len(ks) > 1:
                break  # traces of a second engine would need the first one's released: one k per process
    finally:
        _quiesce(mesh)
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def _drop_sockets(hyb):
    import gc

    if hyb is not None and getattr(hyb, "socks", None) is not None:
        hyb.socks = None
        gc.collect()


def _quiesce(mesh):
    """Before closing the mesh: the async fetch's queued CQ1 copies flushed and both queues drained, the pinned regions
    released (their destructors talk to the mesh). Each step logs, so a hang at exit names the step it stopped in."""
    hyb = getattr(sys.modules.get("hybrid.patch"), "HYB", None)
    f = getattr(hyb, "fetch", None) if hyb is not None else None
    for name, fn in (
        ("pinx flush", lambda: f.px.flush() if f is not None and f.px is not None else None),
        ("synchronize", lambda: ttnn.synchronize_device(mesh)),
        ("fetch release", lambda: f.release() if f is not None else None),
        ("sockets release", lambda: _drop_sockets(hyb)),
    ):
        t0 = time.time()
        print(f"[exit] {name} ...", flush=True)
        try:
            fn()
        except Exception as e:  # keep closing
            print(f"[exit] {name} failed: {e}", flush=True)
        print(f"[exit] {name} done ({time.time() - t0:.1f} s)", flush=True)
    print("[exit] close_mesh_device ...", flush=True)


if __name__ == "__main__":
    main()
