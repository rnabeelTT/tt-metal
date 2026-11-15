# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The engine's sampler: M2.7's exact speculative sampler (tt/mtp_sampling.py of minimaxai_minimax_m2_7) plus vLLM's
penalties, applied on the host to the verify pass's logits rows.

Penalties (vLLM semantics, before temperature / top-k / top-p, greedy included):
    repetition r   every token seen in the prompt or the output so far: logit / r when positive, logit * r otherwise
    frequency f    logit - f * (count of the token in the output so far)
    presence  p    logit - p * (1 if the token is in the output so far)
Row i of a verify pass [x, d_1 .. d_k] continues the output with d_1 .. d_i, so its target distribution counts those
drafts too: the accept rule then emits exactly the penalized sampling distribution, token for token. The drafter's
distribution q is penalized the same way (only the acceptance rate depends on it).
"""
from __future__ import annotations

import torch

from models.autoports.minimaxai_minimax_m2_7.tt.mtp_sampling import SpecSampler


class InklingSampler(SpecSampler):
    def __init__(
        self,
        temperature: float,
        top_k: int = 0,
        top_p: float = 1.0,
        seed=None,
        presence: float = 0.0,
        frequency: float = 0.0,
        repetition: float = 1.0,
        prompt_ids=None,
        vocab: int = 0,
    ):
        super().__init__(temperature, top_k, top_p, seed)
        self.presence = float(presence or 0.0)
        self.frequency = float(frequency or 0.0)
        self.repetition = float(repetition if repetition else 1.0)
        self.penalized = abs(self.presence) > 1e-6 or abs(self.frequency) > 1e-6 or abs(self.repetition - 1.0) > 1e-6
        self.V = int(vocab)
        self.counts = None
        self.seen = None
        if self.penalized:
            self.counts = torch.zeros(self.V, dtype=torch.float32)
            self.seen = torch.zeros(self.V, dtype=torch.bool)
            if prompt_ids is not None and len(prompt_ids):
                ids = torch.as_tensor(list(prompt_ids), dtype=torch.long)
                self.seen[ids[(ids >= 0) & (ids < self.V)]] = True

    def observe(self, toks):
        """Tokens appended to the output (emitted, or adopted from the host)."""
        if not self.penalized:
            return
        for t in toks:
            t = int(t)
            if 0 <= t < self.V:
                self.counts[t] += 1.0
                self.seen[t] = True

    def penalize(self, logits: torch.Tensor, extra=()) -> torch.Tensor:
        """Logits row [V'] (V' >= V; padded ids untouched) with the penalties of the output so far + `extra`."""
        if not self.penalized:
            return logits
        x = logits.float().clone()
        V = min(self.V, x.numel())
        counts, seen = self.counts[:V], self.seen[:V]
        if extra:
            counts, seen = counts.clone(), seen.clone()
            for t in extra:
                t = int(t)
                if 0 <= t < V:
                    counts[t] += 1.0
                    seen[t] = True
        v = x[:V]
        if abs(self.repetition - 1.0) > 1e-6:
            r = self.repetition
            v = torch.where(seen, torch.where(v > 0, v / r, v * r), v)
        out_mask = counts > 0
        if self.frequency:
            v = v - self.frequency * counts
        if self.presence:
            v = v - self.presence * out_mask.float()
        x[:V] = v
        return x

    def pick(self, logits: torch.Tensor, extra=()):
        """One token from a logits row (greedy argmax or a sample of the processed distribution) and its
        distribution (None when greedy)."""
        x = self.penalize(logits, extra)
        if self.greedy:
            return int(x.argmax()), None
        p = self.probs(x)
        return self.sample(p), p

    def verify(self, L: torch.Tensor, drafts: list[int], qs):
        """(accepted drafts n, next token) of a verify pass with target logits rows L [R, V]."""
        if self.greedy and not self.penalized:
            am = L.argmax(-1).tolist()
            n = 0
            while n < len(drafts) and drafts[n] == am[n]:
                n += 1
            return n, int(am[n])
        if self.greedy:
            n = 0
            while n < len(drafts):
                row = self.penalize(L[n], drafts[:n]) if self.penalized else L[n]
                if int(row.argmax()) != drafts[n]:
                    break
                n += 1
            row = self.penalize(L[n], drafts[:n]) if self.penalized else L[n]
            return n, int(row.argmax())
        if not self.penalized:
            return self.accept(L, drafts, qs)
        return self.accept(lambda i: self.probs(self.penalize(L[i], drafts[:i])), drafts, qs)
