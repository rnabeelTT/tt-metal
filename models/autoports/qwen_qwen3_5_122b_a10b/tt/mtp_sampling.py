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

    def probs(self, logits: torch.Tensor) -> torch.Tensor:
        """Request-processed distribution over the vocab from a logits row ([V], any float dtype)."""
        x = logits.float() * (1.0 / self.temperature)
        if 0 < self.top_k < x.numel():
            kth = torch.topk(x, self.top_k).values[-1]
            x = torch.where(x < kth, torch.full_like(x, float("-inf")), x)
        p = torch.softmax(x, dim=-1)
        if self.top_p < 1.0:
            sp, idx = torch.sort(p, descending=True)
            cum = torch.cumsum(sp, dim=-1)
            keep = cum - sp <= self.top_p  # keep every token up to the one that crosses top_p
            masked = torch.zeros_like(p)
            masked[idx[keep]] = sp[keep]
            p = masked / masked.sum()
        return p

    def sample(self, p: torch.Tensor) -> int:
        # inverse-CDF draw with one uniform (torch.multinomial over a 248K vocab costs ~1 ms per draw)
        u = float(torch.rand(1, generator=self.gen))
        cdf = torch.cumsum(p, dim=-1)
        i = int(torch.searchsorted(cdf, torch.tensor([u * float(cdf[-1])])).item())
        return min(i, p.numel() - 1)

    def accept(self, p_rows: torch.Tensor, drafts: list[int], q_rows: list[torch.Tensor]):
        """p_rows [R, V] target logits rows (row i = after token i of the candidate rows), drafts d_1..d_k with their draft
        distributions q_rows. Returns (accepted count a, next token). Target rows are processed lazily: row i only when
        the loop reaches it."""
        k = len(drafts)
        ps = [None] * (k + 1)

        def P(i):
            if ps[i] is None:
                ps[i] = self.probs(p_rows[i])
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
