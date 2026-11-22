# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Prompt-lookup ("suffix") drafts for the DFlash step: the continuation of the longest earlier occurrence of the
context's last tokens, and the per-request choice between it and the DFlash drafter's block.

The lookup is host-only (a vectorized scan of the context ids, ~0.1 ms at 128K tokens) and needs no weights. Copy-heavy
outputs (edits of pasted code, renames, quotes) repeat long spans of the prompt, where the earlier occurrence predicts
several tokens the drafter misses.

Which proposal a pass verifies is decided before either draft is looked at, from the match length and a per-request
head-to-head tally (Chooser): when the two proposals first differ at row j and the verify reaches row j, the target's
token there names the winner; a match-length bucket uses the suffix once it has won more often than the drafter. The DFlash drafter still runs every pass (its context cache takes every accepted row),
so a pass costs the same either way (plus the ~0.1 ms lookup).
"""

from __future__ import annotations

import os

import numpy as np

MIN_N = int(os.environ.get("DFLASH_SUFFIX_MIN", "3"))  # shortest match used
MAX_N = 32  # longer matches count as this long
# per match-length bucket: prior (suffix wins, DFlash wins) of the head-to-head tally. A request starts on the DFlash
# drafts in every bucket, and a bucket switches to the suffix once its suffix wins exceed its DFlash wins: the tally
# keeps counting while the suffix is unused (the verify's rows decide where the two proposals first differ). Chosen on
# 13 logged workloads (code edits, quotes, explanations, chat): no workload below its DFlash-only tokens per pass.
_BUCKETS = ((3, 3), (4, 4), (5, 7), (8, MAX_N))
_PRIORS = ((0.0, 2.0), (0.0, 2.0), (0.0, 1.0), (0.0, 1.0))


def _bucket(n):
    for b, (lo, hi) in enumerate(_BUCKETS):
        if lo <= n <= hi:
            return b
    return None


class SuffixIndex:
    """The request's context ids (prompt + accepted tokens) and the longest-earlier-match lookup."""

    def __init__(self, ids):
        n = len(ids)
        self.buf = np.empty(max(1024, 2 * n), dtype=np.int64)
        self.buf[:n] = np.asarray(ids, dtype=np.int64)
        self.n = n

    def extend(self, ids):
        m = len(ids)
        if self.n + m > self.buf.shape[0]:
            nb = np.empty(2 * (self.n + m), dtype=np.int64)
            nb[: self.n] = self.buf[: self.n]
            self.buf = nb
        self.buf[self.n : self.n + m] = ids
        self.n += m

    def lookup(self, k):
        """(match length, the up-to-k tokens that followed the longest earlier occurrence of the context's last
        tokens); the most recent one among equally long matches. (0, []) when no match of MIN_N tokens exists."""
        n, c = self.n, self.buf
        if n < MIN_N + 1:
            return 0, []
        # occurrences of the last token whose continuation is known: end index e (ctx[e] == ctx[n-1]), e < n - 1
        e = np.flatnonzero(c[: n - 1] == c[n - 1])
        L = 1
        while e.size and L < MAX_N:
            ok = e >= L
            e2 = e[ok]
            same = c[e2 - L] == c[n - 1 - L]
            if not same.any():
                break
            e = e2[same]
            L += 1
        if L < MIN_N or not e.size:
            return 0, []
        s = int(e[-1]) + 1  # the most recent of the longest matches
        return L, c[s : min(s + k, n)].tolist()


class Chooser:
    """Per-request head-to-head tally of suffix vs DFlash proposals, per match-length bucket."""

    def __init__(self):
        self.w = [list(p) for p in _PRIORS]
        self.used = self.considered = 0

    def pick(self, n):
        b = _bucket(n)
        if b is None:
            return False
        self.considered += 1
        s, d = self.w[b]
        return s > d

    def record(self, n, sfx, dfl, used_sfx, accepted, nxt):
        """sfx / dfl: the two proposals (same length; used_sfx: the pass verified sfx); `accepted` rows of it were
        accepted and `nxt` is the token the pass emitted after them. The first row where they differ decides the bucket's tally when
        the verify reached it."""
        b = _bucket(n)
        if b is None:
            return
        for j, (a, d) in enumerate(zip(sfx, dfl)):
            if a != d:
                if accepted < j:
                    return  # rejected before the difference: no information
                tgt = nxt if accepted == j else (sfx if used_sfx else dfl)[j]
                if tgt == a:
                    self.w[b][0] += 1.0
                elif tgt == d:
                    self.w[b][1] += 1.0
                return
