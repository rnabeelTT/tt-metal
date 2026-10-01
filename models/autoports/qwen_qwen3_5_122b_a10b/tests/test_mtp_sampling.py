# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only: the speculative accept rule emits tokens distributed exactly like plain sampling from the target."""
import torch

from models.autoports.qwen_qwen3_5_122b_a10b.tt.mtp_sampling import SpecSampler


def _kl(p, q):
    m = p > 0
    return float((p[m] * (p[m] / q[m].clamp_min(1e-9)).log()).sum())


def test_speculative_sampling_matches_target_distribution():
    torch.manual_seed(0)
    V, k, trials = 50, 3, 40000
    target_logits = torch.randn(k + 1, V) * 2.0  # one target row per candidate position
    draft_logits = target_logits + torch.randn(k + 1, V) * 1.5  # a biased draft model
    s = SpecSampler(temperature=0.8, top_k=0, top_p=1.0, seed=123)
    p0 = s.probs(target_logits[0])
    counts = torch.zeros(V)
    for _ in range(trials):
        # drafts sampled from the draft distributions (position-wise, as the head would)
        qs = [s.probs(draft_logits[i]) for i in range(k)]
        drafts = [s.sample(q) for q in qs]
        a, nxt = s.accept(target_logits, drafts, qs)
        first = drafts[0] if a >= 1 else nxt  # the first emitted token of the pass
        counts[first] += 1
    emp = counts / counts.sum()
    kl = _kl(p0, emp)
    print(f"SPEC_SAMPLING first-token KL(target || empirical) = {kl:.5f} over {trials} trials")
    assert kl < 0.01, kl


def test_speculative_sampling_greedy_limit():
    s = SpecSampler(temperature=0.0)
    assert s.greedy
    s2 = SpecSampler(temperature=0.7, top_k=1)
    assert s2.greedy


def test_top_p_top_k_processing():
    s = SpecSampler(temperature=1.0, top_k=3, top_p=0.9, seed=1)
    p = s.probs(torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0]))
    assert int((p > 0).sum()) <= 3 and abs(float(p.sum()) - 1.0) < 1e-5
