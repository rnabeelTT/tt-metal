# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Host tests of DFlash sampling (dflash_serve): (1) the accept rule (accept_sparse: argmax drafts as point masses, or drafts
sampled from the drafter's rows) emits tokens distributed exactly as plain sampling from the target: Monte Carlo over a small vocab with row-dependent targets (row
i's distribution depends on whether the drafts before it were accepted, as in a real verify), compared with the exact
joint distribution of the first two emitted tokens; (2) the device-candidate distribution (cand_probs) equals
SpecSampler.probs on the full logits, and declines (None) when a chunk may hide part of the top-k; (3) penalties: a row's
distribution with presence / frequency / repetition penalties (DFlashServing._dist, from the candidates or the full row,
greedy or sampled, the history = emitted tokens + the drafts before the row) equals the request's processing of vLLM's
penalized logits (an independent dense implementation of vllm apply_penalties).
usage: ./run_in_image.sh test_dflash_sampling.py   (no device needed)"""
import itertools
import sys

import torch

sys.path.insert(0, ".")
import collections  # noqa: E402

from dflash_serve import DFlashServing, accept_sparse, cand_probs, sparse  # noqa: E402
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


# (3) penalties
def vllm_penalties(logits, prompt, output, pres, freq, rep):
    """vllm/model_executor/layers/utils.py apply_penalties for one row (dense masks and bin counts)."""
    n = logits.numel()
    prompt_mask = torch.zeros(n, dtype=torch.bool)
    prompt_mask[torch.tensor(prompt, dtype=torch.long)] = True
    counts = torch.bincount(torch.tensor(output, dtype=torch.long), minlength=n).float()
    output_mask = counts > 0
    penalties = torch.where(prompt_mask | output_mask, torch.tensor(rep), torch.tensor(1.0))
    scaling = torch.where(logits > 0, 1.0 / penalties, penalties)
    out = logits * scaling
    out = out - freq * counts
    return out - pres * output_mask.float()


srv = object.__new__(DFlashServing)
n_cand = n_full = 0
for trial in range(300):
    lg = torch.randn(VV, generator=g) * 3
    prompt = torch.randint(0, VV, (int(torch.randint(1, 40, (1,), generator=g)),), generator=g).tolist()
    emitted = torch.randint(0, VV, (int(torch.randint(1, 30, (1,), generator=g)),), generator=g).tolist()
    drafts = torch.randint(0, VV, (trial % 4,), generator=g).tolist()
    if trial % 3 == 0:  # make the penalties matter: the history holds the row's top tokens
        emitted += torch.topk(lg, 6).indices.tolist()
    vals, ids, mins = [], [], []
    for d in range(D):
        for c in range(C):
            lo = d * W + c * (W // C)
            v, i = torch.topk(lg[lo : lo + W // C], KC)
            vals.append(v)
            ids.append(i + lo)
            mins.append(v.min())
    vals, ids, mins = torch.cat(vals), torch.cat(ids), torch.stack(mins)
    for pen in [(0.5, 0.3, 1.3), (0.0, 0.0, 2.0), (1.5, 0.0, 1.0), (0.0, 0.0, 0.8), (-0.5, 0.0, 1.0)]:
        for t, k, p in [(0.0, 0, 1.0), (1.0, 10, 0.95), (0.7, 4, 1.0)]:
            smp = SpecSampler(t, k, p, seed=0)
            srv._pen, srv._pen_prompt, srv._P, srv._ctx_ids = pen, None, len(prompt), prompt + emitted[1:]
            srv._out_counts, srv.stats = collections.Counter(emitted), {}
            got = srv._dist(smp, vals, ids, mins, drafts, lambda: lg.clone())
            used_full = srv.stats.get("full_rows", 0) == 1
            n_full += used_full
            n_cand += not used_full
            if pen[2] < 1.0 or pen[0] < 0.0:
                if not used_full:
                    raise SystemExit(f"FAIL: penalties that raise logits used the candidates (trial {trial}, {pen})")
            ref = vllm_penalties(lg, prompt, emitted + drafts, *pen)
            if smp.greedy:
                want = torch.zeros(VV)
                want[int(torch.argmax(ref))] = 1.0
            else:
                want = smp.probs(ref)
            dense = torch.zeros(VV)
            dense[got[0]] = got[1]
            if not torch.allclose(dense, want, atol=1e-6):
                raise SystemExit(
                    f"FAIL: penalized row != vLLM's penalties (trial {trial}, pen {pen}, t {t} k {k} p {p}, "
                    f"{'full row' if used_full else 'candidates'}, max diff {float((dense - want).abs().max())})"
                )
if n_cand == 0 or n_full == 0:
    raise SystemExit(f"FAIL: penalty test did not exercise both paths (candidates {n_cand}, full rows {n_full})")
print(f"penalties: equal to vLLM's apply_penalties ({n_cand} rows from candidates, {n_full} from the full row)")
if worst > 0.012:
    raise SystemExit(f"FAIL: sampled DFlash accept rule deviates from the target (TV {worst:.4f})")
print("PASS")
