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


def _dense_reference(logits, temperature, top_k, top_p):
    x = logits.float() / temperature
    if 0 < top_k < x.numel():
        kth = torch.topk(x, top_k).values[-1]
        x = torch.where(x < kth, torch.full_like(x, float("-inf")), x)
    p = torch.softmax(x, dim=-1)
    if top_p < 1.0:
        sp, idx = torch.sort(p, descending=True)
        cum = torch.cumsum(sp, dim=-1)
        keep = cum - sp <= top_p
        masked = torch.zeros_like(p)
        masked[idx[keep]] = sp[keep]
        p = masked / masked.sum()
    return p


def test_probs_fast_paths_match_dense_reference():
    """The survivor-only / nucleus-prefix top-p paths equal the straightforward full-row processing."""
    torch.manual_seed(1)
    V = 8192
    for top_k, top_p in [(20, 0.95), (32, 1.0), (0, 0.9), (0, 0.999), (50, 0.8), (0, 1.0)]:
        s = SpecSampler(temperature=0.7, top_k=top_k, top_p=top_p, seed=3)
        for scale in (1.0, 6.0):  # flat (nucleus wide) and peaked rows
            lg = torch.randn(V) * scale  # float32: no ties at the k-th value (the two paths break ties differently)
            got, ref = s.probs(lg), _dense_reference(lg, 0.7, top_k, top_p)
            assert torch.allclose(got, ref, atol=1e-6), (top_k, top_p, scale, (got - ref).abs().max())


def test_probs_topk_candidates_match_dense():
    """The candidate-only distribution (per-shard device top-k union) equals the dense one whenever top_k <= 32."""
    torch.manual_seed(2)
    V, TP, per_shard = 8192, 4, 64
    W = V // TP
    for top_k, top_p in [(20, 0.95), (32, 1.0), (1, 1.0), (5, 0.5)]:
        s = SpecSampler(temperature=0.6, top_k=top_k, top_p=top_p, seed=3)
        assert s.sparse_ok()
        lg = torch.randn(V) * 3
        vals, ids = [], []
        for sh in range(TP):  # what the device delivers: each shard's largest values with global ids
            v, i = torch.topk(lg[sh * W : (sh + 1) * W].float(), per_shard)
            vals.append(v)
            ids.append(i + sh * W)
        got = s.probs_topk(torch.cat(vals), torch.cat(ids), V)
        ref = s.probs(lg)
        assert torch.allclose(got, ref, atol=1e-6), (top_k, top_p, (got - ref).abs().max())
    assert not SpecSampler(0.7, top_k=0, top_p=0.9).sparse_ok()
    assert not SpecSampler(0.7, top_k=33).sparse_ok()


def test_accept_takes_callable_target():
    torch.manual_seed(4)
    V, k = 64, 3
    tl = torch.randn(k + 1, V) * 2
    dl = tl + torch.randn(k + 1, V)
    s1, s2 = SpecSampler(0.8, top_k=20, top_p=0.95, seed=9), SpecSampler(0.8, top_k=20, top_p=0.95, seed=9)
    qs = [s1.probs(dl[i]) for i in range(k)]
    drafts = [s1.sample(q) for q in qs]
    for _ in range(k):  # keep the two RNG streams aligned
        s2.probs(dl[0])
        s2.sample(qs[0])
    a1 = s1.accept(tl, drafts, qs)
    a2 = s2.accept(lambda i: s2.probs(tl[i]), drafts, qs)
    assert a1 == a2, (a1, a2)
