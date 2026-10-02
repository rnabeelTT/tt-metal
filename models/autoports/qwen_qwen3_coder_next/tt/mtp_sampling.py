# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Exact speculative sampling for the MTP step (host side, single user).

    Drafts are SAMPLED from the head's distribution q_j (the request's temperature / top-k / top-p applied to the head
    logits), the verify pass gives the target distribution p_i per row (same processing), and the accept rule
        accept d_i with probability min(1, p_i(d_i) / q_i(d_i)); on the first rejection emit a sample from
        norm(max(0, p_i - q_i)) and stop; if every draft is accepted emit a sample from p_k
    makes the emitted stream exactly distributed as plain sampling from p (Leviathan et al. 2023, Chen et al. 2023).
    Temperature 0 (or top_k == 1) is plain greedy: argmax drafts and exact-match acceptance, no logits read-back.
    Penalties (presence / frequency / repetition) are not applied in the MTP step (documented limitation).
"""
from __future__ import annotations

import torch


class SpecSampler:
    def __init__(self, temperature: float, top_k: int = 0, top_p: float = 1.0, seed=None):
        self.temperature = float(temperature)
        self.top_k = int(top_k or 0)
        self.top_p = float(top_p if top_p is not None else 1.0)
        self.greedy = self.temperature <= 0.0 or self.top_k == 1
        self.gen = torch.Generator()
        if seed is not None and int(seed) >= 0:
            self.gen.manual_seed(int(seed))
        else:
            self.gen.seed()

    # the per-shard candidate count the device top-k delivers (tt/mtp_traced.py reads 2 x 32 per shard): a request
    # whose top_k is at most this many is exactly reproducible from the candidates alone (its top_k survivors are
    # inside every shard's top-32, hence inside the union)
    SPARSE_TOP_K = 32

    def sparse_ok(self) -> bool:
        """True when the request's distribution is determined by the device top-k candidates (0 < top_k <= 32)."""
        return 0 < self.top_k <= self.SPARSE_TOP_K

    def probs(self, logits: torch.Tensor) -> torch.Tensor:
        """Request-processed distribution over the vocab from a logits row ([V], any float dtype). top_k keeps exactly
        k survivors (ties at the k-th value broken by torch.topk, as the device top-k does; vLLM's host sampler keeps
        the ties) - any fixed processing is a valid target for the accept rule as long as drafts and targets use it."""
        x = logits.float() * (1.0 / self.temperature)
        V = x.numel()
        if 0 < self.top_k < V:
            # top-k, then softmax / top-p over the k survivors only (the dense top-p sorted the whole 248K row: ~2 ms)
            vals, idx = torch.topk(x, self.top_k)
            return self._scatter(self._nucleus(torch.softmax(vals, dim=-1)), idx, V)
        p = torch.softmax(x, dim=-1)
        if self.top_p < 1.0:
            # the nucleus is among the largest probabilities: try the top 1024, then 16384 (~1 ms); the full sort
            # (~10 ms on 248K) only when those do not reach top_p
            for n in (1024, 16384):
                n = min(n, V)
                vals, idx = torch.topk(p, n)
                if n == V or float(vals.sum()) >= self.top_p:
                    return self._scatter(self._nucleus_unnormalized(vals), idx, V)  # topk values come sorted desc.
            sp, idx = torch.sort(p, descending=True)
            return self._scatter(self._nucleus_unnormalized(sp), idx, V)
        return p

    def probs_topk(self, vals: torch.Tensor, idx: torch.Tensor, V: int) -> torch.Tensor:
        """The same distribution from candidate logits only: `vals` [n] (any float dtype) at vocab ids `idx` [n], a
        superset of the row's top_k (requires sparse_ok()). Returns the dense [V] distribution."""
        assert self.sparse_ok(), (self.top_k, self.SPARSE_TOP_K)
        x = vals.float() * (1.0 / self.temperature)
        k = min(self.top_k, x.numel())
        v, i = torch.topk(x, k)
        return self._scatter(self._nucleus(torch.softmax(v, dim=-1)), idx.long()[i], V)

    def _nucleus(self, p_sorted_any: torch.Tensor) -> torch.Tensor:
        """top-p over a (small) normalized distribution given in any order; returns it renormalized."""
        if self.top_p >= 1.0:
            return p_sorted_any
        sp, order = torch.sort(p_sorted_any, descending=True)
        cum = torch.cumsum(sp, dim=-1)
        keep = cum - sp <= self.top_p  # keep every token up to the one that crosses top_p
        out = torch.zeros_like(p_sorted_any)
        out[order[keep]] = sp[keep]
        return out / out.sum()

    def _nucleus_unnormalized(self, sp: torch.Tensor) -> torch.Tensor:
        """top-p over probabilities already sorted descending (a prefix of the full row: their cumulative sums are the
        row's); returns the kept prefix renormalized, zeros elsewhere."""
        cum = torch.cumsum(sp, dim=-1)
        keep = cum - sp <= self.top_p
        out = torch.where(keep, sp, torch.zeros_like(sp))
        return out / out.sum()

    @staticmethod
    def _scatter(p_small: torch.Tensor, idx: torch.Tensor, V: int) -> torch.Tensor:
        out = torch.zeros(V, dtype=torch.float32)
        out[idx.long()] = p_small
        return out

    def sample(self, p: torch.Tensor) -> int:
        # inverse-CDF draw with one uniform (torch.multinomial over a 248K vocab costs ~1 ms per draw)
        u = float(torch.rand(1, generator=self.gen))
        cdf = torch.cumsum(p, dim=-1)
        i = int(torch.searchsorted(cdf, torch.tensor([u * float(cdf[-1])])).item())
        return min(i, p.numel() - 1)

    def accept(self, p_rows, drafts: list[int], q_rows: list[torch.Tensor]):
        """p_rows: [R, V] target logits rows (row i = after token i of the candidate rows) or a callable i -> the dense
        target distribution of row i (the device top-k candidate path); drafts d_1..d_k with their draft distributions
        q_rows. Returns (accepted count a, next token). Target rows are processed lazily: row i only when the loop
        reaches it."""
        k = len(drafts)
        ps = [None] * (k + 1)

        def P(i):
            if ps[i] is None:
                ps[i] = p_rows(i) if callable(p_rows) else self.probs(p_rows[i])
            return ps[i]

        for i, d in enumerate(drafts):
            pi = P(i)
            p_d, q_d = float(pi[d]), float(q_rows[i][d])
            u = float(torch.rand(1, generator=self.gen))
            if q_d <= 0.0 or u >= min(1.0, p_d / q_d):
                resid = torch.clamp(pi - q_rows[i], min=0.0)
                tot = float(resid.sum())
                nxt = self.sample(resid / tot) if tot > 0 else self.sample(pi)
                return i, nxt
        return k, self.sample(P(k))
