# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only checks of the engine's sampler (tt/sampling.py): penalties match vLLM's formulas, and a speculative
step with penalties (drafts from a different distribution q) emits the penalized target distribution exactly."""
import torch

from models.autoports.thinkingmachines_inkling_small.tt.sampling import InklingSampler


def ref_penalize(logits, prompt, out, pres, freq, rep):
    x = logits.clone().float()
    counts = (
        torch.bincount(torch.tensor(out, dtype=torch.long), minlength=x.numel()).float() if out else torch.zeros_like(x)
    )
    seen = counts > 0
    if prompt:
        seen[torch.tensor(prompt)] = True
    x = torch.where(seen, torch.where(x > 0, x / rep, x * rep), x)
    return x - freq * counts - pres * (counts > 0).float()


def test_penalize_matches_formula():
    torch.manual_seed(0)
    V = 64
    s = InklingSampler(1.0, presence=0.5, frequency=0.3, repetition=1.2, prompt_ids=[1, 2, 3], vocab=V)
    s.observe([3, 4, 4])
    L = torch.randn(V)
    got = s.penalize(L, [5])
    want = ref_penalize(L, [1, 2, 3], [3, 4, 4, 5], 0.5, 0.3, 1.2)
    assert torch.allclose(got, want, atol=1e-6)


def test_unpenalized_is_identity():
    s = InklingSampler(0.7, vocab=10)
    L = torch.randn(10)
    assert s.penalize(L) is L and not s.penalized


def test_spec_step_distribution_exact():
    """Two-token outcome (next token after one draft) distribution of the spec step == plain penalized sampling."""
    torch.manual_seed(1)
    V, N = 6, 60000
    L = torch.randn(2, V) * 2  # target rows: after x, after x d_1
    Q = torch.randn(V) * 2  # drafter logits for d_1
    kw = dict(presence=0.8, frequency=0.4, repetition=1.3, prompt_ids=[0], vocab=V)
    # plain: t1 ~ p(L0 | out=[]), t2 ~ p(L1 | out=[t1])
    s = InklingSampler(1.0, seed=3, **kw)
    p0 = s.probs(s.penalize(L[0]))
    plain = torch.zeros(V, V)
    for a in range(V):
        plain[a] = p0[a] * s.probs(s.penalize(L[1], [a]))
    plain_first = plain.sum(1)
    # spec: d ~ q (penalized drafter), accept rule -> first emitted token distribution must equal p0
    s = InklingSampler(1.0, seed=4, **kw)
    first = torch.zeros(V)
    for _ in range(N):
        q = s.probs(s.penalize(Q))
        d = s.sample(q)
        n, nxt = s.verify(L, [d], [q])
        first[d if n == 1 else nxt] += 1
    first /= N
    assert (first - plain_first).abs().max() < 0.01, (first, plain_first)


def test_greedy_penalized_argmax():
    V = 8
    L = torch.tensor([[0.0, 5.0, 4.9, 0, 0, 0, 0, 0], [0.0, 5.0, 4.9, 0, 0, 0, 0, 0]])
    s = InklingSampler(0.0, repetition=1.5, prompt_ids=[], vocab=V)
    s.observe([1])  # token 1 seen: 5 / 1.5 < 4.9 -> 2 wins
    n, nxt = s.verify(L, [2], None)
    assert n == 1 and nxt == 1  # draft 2 accepted; row 1 has 1 and 2 seen: 5 / 1.5 > 4.9 / 1.5 -> 1
