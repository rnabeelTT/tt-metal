# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Outer chunk width for the qwen36 hybrid stack's chunked (long-prompt) prefill.

A prompt longer than one masked bucket is prefilled in OUTER CHUNKS: one all-layer chunk
forward is captured as a trace and replayed per chunk, so the matmul shapes a long prefill
runs are set by this width and NOT by the prompt length. A 131,072-token prompt at the
historical 2048 is 64 identical replays.

That makes the width the one real shape knob for long prompts, because M for every
projection in the layer IS the chunk width. It had been the bare constant
``_PREFILL_WARMUP_CHUNK = 2048`` in qwen36_vllm.py plus a ``chunk_size=2048`` default on
``capture_prefill_trace_chunked``, with no recorded rationale and no way to sweep it.

Raising it trades compute density against three hard limits, all of which bite before
arithmetic does:

* **L1 circular buffers.** They scale with the chunk's row count. Measured on the
  single-shot TP path, which is the same shape question: a 8192-row in-projection asks for
  2,727,424 B and a 32768-row one for 10,139,136 B, against a 1,572,864 B L1. So ~4096 is
  the plausible ceiling and 8192 is expected to fail outright.
* **The 4 GiB trace limit**, which the chunk-outer design exists to stay under.
* **GDN + MoE intermediates**, which are per-chunk and grow with it.

``QWEN36_PREFILL_CHUNK`` sweeps it. The default stays 2048 -- the historical value -- so
nothing moves until a measurement earns it. Must be a multiple of 128 (the GDN sub-chunk),
which ``capture_prefill_trace_chunked`` also asserts.
"""
import os

DEFAULT_PREFILL_CHUNK = 2048


def prefill_chunk(default: int = DEFAULT_PREFILL_CHUNK) -> int:
    """Resolve ``QWEN36_PREFILL_CHUNK``, falling back to ``default``."""
    raw = os.environ.get("QWEN36_PREFILL_CHUNK")
    if raw is None or not raw.strip():
        return default
    try:
        val = int(raw)
    except ValueError:
        raise ValueError(f"QWEN36_PREFILL_CHUNK={raw!r} is not an integer")
    if val <= 0 or val % 128:
        # Not cosmetic: a non-128 width mis-shapes the GDN sub-chunk mask, and
        # capture_prefill_trace_chunked asserts it too.
        raise ValueError(f"QWEN36_PREFILL_CHUNK={val} must be a positive multiple of 128")
    return val
