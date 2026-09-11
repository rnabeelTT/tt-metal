#!/usr/bin/env python3
"""Concurrency sweep against the running OpenAI-compatible server.

A single-request demo measures the model; this measures the SERVER. The numbers that
differ are the ones that matter for a deployment decision: how throughput scales with
concurrent users, and how much per-request latency is given up to get it. On an MoE
this is not academic — routing means concurrent tokens hit different experts, so
batching wins more here than on a dense model of the same active size.
"""
from __future__ import annotations

import argparse
import json
import statistics
import threading
import time

import requests

PROMPT = (
    "Explain, step by step, why a mixture-of-experts model with 256 experts "
    "and top-8 routing has a different memory-bandwidth profile than a dense "
    "model with the same number of active parameters."
)


def one_request(base, model, out, idx, max_tokens):
    t0 = time.perf_counter()
    first = None
    ntok = 0
    try:
        with requests.post(
            f"{base}/v1/completions",
            json={
                "model": model,
                "prompt": PROMPT,
                "max_tokens": max_tokens,
                "temperature": 0.0,
                "stream": True,
            },
            stream=True,
            timeout=600,
        ) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line or not line.startswith(b"data: "):
                    continue
                if line == b"data: [DONE]":
                    break
                if first is None:
                    first = time.perf_counter()
                ntok += 1
    except Exception as e:  # a failed request is data, not a crash
        out[idx] = {"error": str(e)}
        return
    t1 = time.perf_counter()
    out[idx] = {
        "ttft_ms": (first - t0) * 1e3 if first else None,
        "total_s": t1 - t0,
        "tokens": ntok,
        # Decode rate excludes prefill, so it is comparable across ISLs.
        "decode_tok_s": (ntok - 1) / (t1 - first) if first and t1 > first and ntok > 1 else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    # Several bases = several LANES. This model is replicated, one complete copy per
    # die, so aggregate throughput comes from running a replica per die and spreading
    # requests over them -- not from raising max_num_seqs, which the GDN external-state
    # assert pins to 1. Requests are dealt round-robin, so concurrency C over L lanes
    # puts C/L streams on each. One base behaves exactly as before.
    ap.add_argument("--base", nargs="+", default=["http://127.0.0.1:8000"])
    ap.add_argument("--model", default="ornith-1.5-35b-a3b")
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--concurrency", type=int, nargs="+", default=[1, 8, 32])
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    report = {"mesh": a.mesh, "model": a.model, "lanes": len(a.base), "bases": a.base, "levels": []}
    if len(a.base) > 1:
        print(f"{len(a.base)} lanes: {' '.join(a.base)}")
    for c in a.concurrency:
        results = [None] * c
        threads = [
            threading.Thread(
                target=one_request,
                args=(a.base[i % len(a.base)], a.model, results, i, a.max_tokens),
            )
            for i in range(c)
        ]
        t0 = time.perf_counter()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        wall = time.perf_counter() - t0

        good = [r for r in results if r and "error" not in r]
        errs = [r for r in results if r and "error" in r]
        if not good:
            report["levels"].append(
                {"concurrency": c, "failed": len(errs), "first_error": errs[0]["error"] if errs else "no result"}
            )
            print(f"  concurrency {c:3d}: ALL FAILED")
            continue
        ttfts = [r["ttft_ms"] for r in good if r["ttft_ms"]]
        rates = [r["decode_tok_s"] for r in good if r["decode_tok_s"]]
        total = sum(r["tokens"] for r in good)
        level = {
            "concurrency": c,
            "ok": len(good),
            "failed": len(errs),
            "ttft_ms_median": statistics.median(ttfts) if ttfts else None,
            "ttft_ms_p99": max(ttfts) if ttfts else None,
            "per_stream_tok_s_median": statistics.median(rates) if rates else None,
            "aggregate_tok_s": total / wall,
        }
        report["levels"].append(level)
        print(
            f"  concurrency {c:3d}: ttft(med) {level['ttft_ms_median']:.0f} ms  "
            f"per-stream {level['per_stream_tok_s_median']:.1f} tok/s  "
            f"aggregate {level['aggregate_tok_s']:.1f} tok/s" + (f"  [{len(errs)} failed]" if errs else "")
        )

    with open(a.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
