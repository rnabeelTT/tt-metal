# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Host tests of DFlash sampling (dflash_serve): (1) the accept rule (accept_sparse: argmax drafts as point masses, or drafts
sampled from the drafter's rows) emits tokens distributed exactly as plain sampling from the target: Monte Carlo over a small vocab with row-dependent targets (row
i's distribution depends on whether the drafts before it were accepted, as in a real verify), compared with the exact
joint distribution of the first two emitted tokens; (2) the device-candidate distribution (cand_probs) equals
SpecSampler.probs on the full logits, and declines (None) when a chunk may hide part of the top-k.
usage: ./run_in_image.sh test_dflash_sampling.py   (no device needed)"""
import itertools
import sys

import torch

sys.path.insert(0, ".")
from dflash_serve import accept_sparse, cand_probs, sparse  # noqa: E402
from models.autoports.minimaxai_minimax_m2_7.tt.mtp_sampling import SpecSampler  # noqa: E402

V, N = 6, 200_000
g = torch.Generator().manual_seed(0)
# target logits: row 0 = p(.|ctx); row 1 = p(.|ctx, d0) (only meaningful when d0 is emitted)
L0 = torch.randn(V, generator=g) * 2
L1 = {t: torch.randn(V, generator=g) * 2 for t in range(V)}


def check(t, k, p, drafts):
    smp = SpecSampler(t, k, p, seed=1)
    p0 = smp.probs(L0)
    p1 = {x: smp.probs(L1[x]) for x in range(V)}
    exact = {(a, b): float(p0[a] * p1[a][b]) for a, b in itertools.product(range(V), range(V))}
    counts = {}
    rows = [sparse(p0), sparse(p1[drafts[0]]), sparse(smp.probs(torch.zeros(V)))]
    for _ in range(N):
        a, nxt = accept_sparse(smp, lambda i: rows[i], drafts)
        if a == 0:  # emitted nxt (a sample from row 0): the second token comes from plain sampling of row 1 | nxt
            first, second = nxt, smp.sample(p1[nxt])
        else:  # emitted d0, then row 1's token (accepted d1 or its replacement)
            first, second = drafts[0], (drafts[1] if a >= 2 else nxt)
        counts[(first, second)] = counts.get((first, second), 0) + 1
    tv = 0.5 * sum(abs(counts.get(k_, 0) / N - v) for k_, v in exact.items())
    print(f"t={t} top_k={k} top_p={p} drafts={drafts}: TV {tv:.4f}", flush=True)
    return tv


worst = 0.0
for t, k, p in [(1.0, 0, 1.0), (1.0, 4, 0.95), (0.6, 0, 0.8)]:
    for drafts in ([int(L0.argmax()), 2], [5, 0]):
        worst = max(worst, check(t, k, p, drafts))

# sampled drafts: d_j ~ q_j (the drafter's rows, independent of each other's tokens), accepted with min(1, p / q)
Q = [torch.randn(V, generator=g) * 2 for _ in range(2)]


def check_sampled(t, k, p):
    smp = SpecSampler(t, k, p, seed=2)
    p0 = smp.probs(L0)
    p1 = {x: smp.probs(L1[x]) for x in range(V)}
    qs = [sparse(smp.probs(q)) for q in Q]
    qd = [smp.probs(q) for q in Q]
    exact = {(a, b): float(p0[a] * p1[a][b]) for a, b in itertools.product(range(V), range(V))}
    counts = {}
    for _ in range(N):
        drafts = [smp.sample(q) for q in qd]
        rows = [sparse(p0), sparse(p1[drafts[0]]), sparse(smp.probs(torch.zeros(V)))]
        a, nxt = accept_sparse(smp, lambda i: rows[i], drafts, qs)
        first = nxt if a == 0 else drafts[0]
        second = smp.sample(p1[nxt]) if a == 0 else (drafts[1] if a >= 2 else nxt)
        counts[(first, second)] = counts.get((first, second), 0) + 1
    tv = 0.5 * sum(abs(counts.get(k_, 0) / N - v) for k_, v in exact.items())
    print(f"sampled drafts t={t} top_k={k} top_p={p}: TV {tv:.4f}", flush=True)
    return tv


for t, k, p in [(1.0, 0, 1.0), (1.0, 4, 0.95)]:
    worst = max(worst, check_sampled(t, k, p))
# 36 cells x 200K samples: Monte Carlo TV is ~0.005; a biased rule (e.g. greedy acceptance) is > 0.1
if worst > 0.012:
    raise SystemExit(f"FAIL: sampled DFlash accept rule deviates from the target (TV {worst:.4f})")
print(f"accept rule: worst TV {worst:.4f}")

# (2) candidates: a 4-die x 4-chunk layout of top-8 per chunk over a 4 x 64-column vocab (the device: top-32 of 16384)
D, C, KC, W = 4, 4, 8, 64
VV = D * W
for trial in range(200):
    lg = torch.randn(VV, generator=g) * 3
    if trial % 4 == 0:  # concentrate the top of the distribution in one chunk: the check must decline
        lg[: KC + 4] += 20
    vals, ids, mins = [], [], []
    for d in range(D):
        for c in range(C):
            lo = d * W + c * (W // C)
            v, i = torch.topk(lg[lo : lo + W // C], KC)
            vals.append(v)
            ids.append(i + lo)
            mins.append(v.min())
    vals, ids, mins = torch.cat(vals), torch.cat(ids), torch.stack(mins)
    for t, k, p in [(1.0, 10, 0.95), (0.7, 4, 1.0), (1.0, 10, 0.5)]:
        smp = SpecSampler(t, k, p, seed=0)
        got = cand_probs(smp, vals, ids, mins)
        if got is not None:
            dense = torch.zeros(VV)
            dense[got[0]] = got[1]
            got = dense
        top_k_vals = torch.topk(lg, k).values
        complete = bool(torch.isin(top_k_vals, vals).all())
        if got is None:
            if trial % 4 and complete and not bool((mins >= torch.topk(vals, k).values[-1]).any()):
                raise SystemExit(f"FAIL: cand_probs declined a complete candidate set (trial {trial})")
            continue
        if not complete:
            raise SystemExit(f"FAIL: cand_probs accepted candidates missing part of the top-{k} (trial {trial})")
        ref = smp.probs(lg)
        if not torch.allclose(got, ref, atol=1e-6):
            raise SystemExit(
                f"FAIL: cand_probs != SpecSampler.probs (trial {trial}, max diff {float((got - ref).abs().max())})"
            )
print("candidates: exact where accepted, declined where a chunk may hide the top-k")
if worst > 0.012:
    raise SystemExit(f"FAIL: sampled DFlash accept rule deviates from the target (TV {worst:.4f})")
print("PASS")
