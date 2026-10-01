# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Traced MTP loop (tt/mtp_traced.py) vs plain greedy on the full model: tokens per pass, per-program device walls."""
import argparse
import time

from loguru import logger

import ttnn
from models.autoports.qwen_qwen3_5_122b_a10b.tools.mtp_acceptance import PROMPTS
from models.autoports.qwen_qwen3_5_122b_a10b.tt.generator import build_generator
from models.autoports.qwen_qwen3_5_122b_a10b.tt.mtp_traced import MtpTraced
from models.autoports.qwen_qwen3_5_122b_a10b.tt.mtp_verify import prefill_with_head
from models.autoports.qwen_qwen3_5_122b_a10b.tt.weights import resolve_checkpoint_dir

MODEL_DIR = "models/autoports/qwen_qwen3_5_122b_a10b"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="prose")
    ap.add_argument(
        "--k", type=int, default=3
    )  # 4 rows = the served decode width; k = 4 (8 rows) costs +17 ms per pass
    ap.add_argument("--new", type=int, default=64)
    ap.add_argument("--layers", default=None)
    ap.add_argument("--profile", type=int, default=0, help="Tracy mode: replay each program N times between signposts")
    ap.add_argument(
        "--temperature", type=float, default=0.0, help="> 0: also time a sampled loop (exact speculative sampling)"
    )
    ap.add_argument("--top-k", type=int, default=20, help="sampled loop top_k (checkpoint generation default 20)")
    ap.add_argument("--top-p", type=float, default=0.95, help="sampled loop top_p (checkpoint generation default 0.95)")
    args = ap.parse_args()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(resolve_checkpoint_dir()))
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(1, 4), l1_small_size=24576, trace_region_size=512 * 1024 * 1024
    )
    layer_indices = [int(s) for s in args.layers.split(",")] if args.layers else None

    def flush():
        # Tracy: the device profiler buffers hold a few hundred ops; read them out at every stage boundary
        if args.profile:
            ttnn.synchronize_device(mesh)
            ttnn.ReadDeviceProfiler(mesh)

    gen = build_generator(
        MODEL_DIR,
        mesh,
        max_seq_len=4096,
        warm_decode_trace=False,
        warm_prefill_trace=False,
        warm_prefill=not args.profile,  # profile: fewer warm-up ops (programs compile at first use, before any capture)
        layer_indices=layer_indices,
        mtp_head=True,
    )
    flush()
    head = gen.mtp_head
    ids = tok(PROMPTS[args.prompt], add_special_tokens=False)["input_ids"]
    t0 = time.time()
    greedy = gen.generate(ids, args.new, enable_trace=False, on_device_sampling=False)
    t_greedy = time.time() - t0
    logger.info(f"greedy {args.new} tokens in {t_greedy:.1f}s (eager): {tok.decode(greedy)!r}")

    # prefill + head prompt rows (eager), then the traced step: warm (compiles) -> capture -> loop
    tok0, hid_last, P = prefill_with_head(gen, head, ids)
    flush()
    runner = MtpTraced(gen, head, args.k)
    ttnn.copy(hid_last, runner.d_hid)
    ttnn.deallocate(hid_last)
    t0 = time.time()
    runner.warm(flush=flush)
    logger.info(f"warm (eager compile of the 4 programs) {time.time() - t0:.1f}s")
    # the warm run advanced the state / KV with dummy rows: redo the prefill so the loop starts from the prompt state
    tok0, hid_last, P = prefill_with_head(gen, head, ids)
    flush()
    ttnn.copy(hid_last, runner.d_hid)
    ttnn.deallocate(hid_last)
    t0 = time.time()
    runner.capture(flush=flush)
    logger.info(f"captured 4 traces in {time.time() - t0:.1f}s")
    # the captures also ran the programs once: prefill again for the measured loop
    tok0, hid_last, P = prefill_with_head(gen, head, ids)
    flush()
    ttnn.copy(hid_last, runner.d_hid)
    ttnn.deallocate(hid_last)
    for v in runner.timing.values():
        v.clear()
    if args.profile:
        from tracy import signpost

        n = args.profile
        rows = [tok0] + [tok0 + 1 + j for j in range(args.k)]
        for name, fn in (
            ("PASS", lambda: runner.verify(rows, P)),
            ("COMMIT", lambda: runner.commit(3)),
            ("DRAFT", lambda: runner.draft(tok0, P)),
            ("ROWS", lambda: runner.head_rows(rows[1:], P)),
        ):
            fn()  # one un-signposted replay first
            flush()
            signpost(f"MTP_{name}_MESH")
            for _ in range(n):
                fn()
                flush()
            signpost(f"MTP_{name}_MESH_END")
        print(f"MTP_PROFILE done: {n} replays per program; walls {runner.timing_summary()}", flush=True)
        flush()
        runner.release()
        ttnn.close_mesh_device(mesh)
        return
    t0 = time.time()
    mtp, passes, hist = runner.generate(tok0, P, args.new, log=logger.info)
    t_mtp = time.time() - t0
    agree = sum(int(a == b) for a, b in zip(greedy, mtp))
    first_diff = next((i for i, (a, b) in enumerate(zip(greedy, mtp)) if a != b), None)
    tm = runner.timing_summary()
    per_pass_ms = sum(n * ms for key, (n, ms) in tm.items() if key in ("pass", "commit", "draft", "rows")) / passes
    print(
        f"MTP_TRACED {args.prompt}: k={args.k} N={args.new} passes {passes} -> {args.new / passes:.2f} tokens per pass "
        f"(hist {hist}); equal to greedy {agree}/{args.new} (first difference {first_diff}); "
        f"loop wall {t_mtp:.2f}s = {t_mtp / passes * 1e3:.1f} ms per pass -> {args.new / t_mtp:.1f} tok/s; "
        f"program walls (count, ms): {tm}; summed per pass {per_pass_ms:.1f} ms",
        flush=True,
    )
    print(f"  greedy: {tok.decode(greedy)!r}\n  mtp   : {tok.decode(mtp)!r}", flush=True)
    if args.temperature > 0:
        from models.autoports.qwen_qwen3_5_122b_a10b.tt.mtp_sampling import SpecSampler

        sampler = SpecSampler(args.temperature, args.top_k, args.top_p, 7)
        tok0, hid_last, P = prefill_with_head(gen, head, ids)
        ttnn.copy(hid_last, runner.d_hid)
        ttnn.deallocate(hid_last)
        for v in runner.timing.values():
            v.clear()
        out, pos, passes_s, hist_s = [tok0], P, 0, [0] * (args.k + 1)
        t0 = time.time()
        while len(out) < args.new:
            drafts, qs = runner.draft(out[-1], pos, sampler=sampler)
            rows = [out[-1]] + drafts
            preds, target = runner.verify(rows, pos, want_logits=True, sampler=sampler)
            a, nxt = sampler.accept(target, drafts, qs)
            if a < args.k:
                runner.commit(a + 1)
            if a > 0:
                runner.head_rows(rows[1 : a + 1], pos)
            runner._select_hidden(a)
            out.extend(rows[1 : a + 1])
            out.append(nxt)
            pos += a + 1
            passes_s += 1
            hist_s[a] += 1
        t_s = time.time() - t0
        tm = runner.timing_summary()
        print(
            f"MTP_TRACED_SAMPLED {args.prompt}: T={args.temperature} top_k={args.top_k} top_p={args.top_p} "
            f"(sparse {sampler.sparse_ok()}) k={args.k} N={args.new} passes {passes_s} -> "
            f"{len(out) / passes_s:.2f} tokens per pass (hist {hist_s}); loop wall {t_s:.2f}s = "
            f"{t_s / passes_s * 1e3:.1f} ms per pass; program walls (count, ms): {tm}; text {tok.decode(out[:args.new])!r}",
            flush=True,
        )
    runner.release()
    ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
