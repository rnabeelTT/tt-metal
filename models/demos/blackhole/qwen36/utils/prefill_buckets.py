# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Width buckets for the qwen36 hybrid stack's masked short-prefill programs.

Masked prefill rounds a prompt UP to a fixed bucket so the compile set stays bounded, and
every bucket must be a 128-multiple because 128 is the GDN sub-chunk. The cost of a
prefill is then set by the BUCKET, not by the prompt: measured on the served container at
a near-constant **2.32-2.48 ms per bucket-token** from 128 all the way to 2560.

That makes TTFT a staircase, and with the historical list -- (128, 256, 512, 1024, 2048),
powers of two below 2048 -- a prompt one token past a boundary pays for the whole next
bucket:

    prompt   bucket   latency
      1000     1024    2540 ms
      1030     2048    4753 ms   <- 1.9x for 30 more tokens
       500      512    1308 ms
       520     1024    2388 ms   <- 1.8x for 20 more tokens

Above 2048 ``_mask_bucket_for`` already falls back to the next 128-multiple, so the waste
there is at most 127 tokens and vanishes proportionally with length. The coarse powers of
two BELOW 2048 are where it bites, and they are what ``dense`` fixes: with every
128-multiple available, the worst case is 127 padded tokens (~310 ms) at any length.

The trade is warmup, not memory: ``warmup_prefill_masked_buckets`` compiles two programs
per bucket (one masked, one full) and MUST compile them all before the decode trace is
parked, because a post-park compile clobbers the trace (#48536). So more buckets means a
longer startup, which is why every width is not simply always on.

``QWEN36_PREFILL_BUCKETS`` takes a comma-separated list of 128-multiples, or the keyword
``dense`` for every 128-multiple up to 2048. Default ``dense``: the 1.8-1.9x cliff hits
ordinary prompt lengths, and a bounded startup cost is the better trade. Set
``QWEN36_PREFILL_BUCKETS=128,256,512,1024,2048`` for the historical behaviour.
"""
import os

_LEGACY = (128, 256, 512, 1024, 2048)
_DENSE = tuple(range(128, 2048 + 1, 128))


def prefill_buckets(default: str = "dense") -> tuple:
    """Resolve ``QWEN36_PREFILL_BUCKETS`` into a sorted tuple of 128-multiples."""
    raw = os.environ.get("QWEN36_PREFILL_BUCKETS", default).strip().lower()
    if raw == "dense":
        return _DENSE
    if raw == "legacy":
        return _LEGACY
    try:
        vals = sorted({int(p) for p in raw.split(",") if p.strip()})
    except ValueError:
        raise ValueError(
            f"QWEN36_PREFILL_BUCKETS={raw!r} must be 'dense', 'legacy', or a comma-separated " f"list of 128-multiples"
        )
    if not vals:
        raise ValueError("QWEN36_PREFILL_BUCKETS resolved to an empty bucket list")
    bad = [v for v in vals if v <= 0 or v % 128]
    if bad:
        # Not cosmetic: a non-128 bucket mis-shapes the GDN sub-chunk mask.
        raise ValueError(f"QWEN36_PREFILL_BUCKETS: {bad} are not positive 128-multiples")
    return tuple(vals)
