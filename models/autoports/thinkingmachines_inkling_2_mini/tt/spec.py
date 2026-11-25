# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Greedy speculative decoding of one sequence: the DSpark drafter (tt/drafter.py) proposes d tokens, one verify pass
(BatchedDecoder verify=d+1, tt/kernels/seqshift.hpp) scores the anchor and the drafts, the longest agreeing prefix is
accepted plus the target's own next token. Every emitted token is the target's argmax given the emitted prefix (the
verify rows are the one-row sequential decode, bitwise).

A pass = two trace replays: the drafter's (commit the previous pass's accepted rows from the verify pass's taps output,
draft the next d tokens), then the verify pass's (with the device top-K of its rows)."""

from __future__ import annotations

import torch

import ttnn

from .batched import BatchedDecoder, TracedDecoder
from .sampling import DeviceTopK


class SpecGreedy:
    def __init__(self, model, full: BatchedDecoder, drafter, d: int = 3):
        """drafter: a DSparkDrafter built before any trace capture (it warms its eager programs)."""
        self.m, self.full, self.dr, self.d, self.R = model, full, drafter, d, d + 1
        self.ver = BatchedDecoder(model, self.R, shared=full, verify=self.R)
        self.dec = TracedDecoder(self.ver)
        self.topk = DeviceTopK(model.mesh, model.Vs, model.V, sharded=model.head_shard)
        self.captured = False
        self.passes = self.accepted = 0

    def _argmax(self, lg, rows):
        """Greedy tokens of the first `rows` logits rows (any mesh: the device top-K, merged on the host)."""
        return self._pick(*self.topk(lg, rows), rows)

    @staticmethod
    def _pick(idx, vals, rows):
        return [int(idx[r][int(torch.argmax(vals[r]))]) for r in range(rows)]

    def warm(self, ids: torch.Tensor):
        """Compile every program of a pass (eager), then capture the verify trace and the drafter's (whose taps buffer
        is the verify trace's taps output). Programs first compiled after a capture can be overwritten by its replays:
        the caller compiles its own (other widths, sampling) before this."""
        R = self.R
        self.prefill(ids)
        self.ver.set_inputs(torch.zeros(R, dtype=torch.long), torch.full((R,), -1))
        self._argmax(self.ver.step(), R)
        self.dec.capture(post=lambda lg: self.topk.device(lg, R))
        self.dr.capture(self.ver.taps, self.d)
        self.captured = True

    def prefill(self, ids: torch.Tensor) -> int:
        """Prompt ids [P] into slot 0 -> the first generated token."""
        m, P = self.m, int(ids.numel())
        pad = torch.cat([ids, torch.zeros((-P) % 64, dtype=ids.dtype)])
        m.reset()
        taps = []
        lg = m.forward(pad, 0, valid=P, slot=0, taps=taps, last_only=True, last_row=P - 1)
        self.full.load_slot(0)
        self.ver.set_accept([0])
        self.dr.reset()
        for c0 in range(0, P, 512):  # (the drafter commits <= 512 rows per call)
            c1 = min(P, c0 + 512)
            T = (c1 - c0 + 63) // 64 * 64
            self.dr.commit(ttnn.slice(taps[0], [0, 0, c0, 0], [1, 4, c0 + T, m.H]), c0, valid=c1 - c0)
        self.n, self.anchor, self.pending = P, self._argmax(lg, 1)[0], None
        return self.anchor

    def step(self) -> list[int]:
        """One draft + verify pass -> the accepted tokens (1 .. d + 1)."""
        R, n = self.R, self.n
        if self.pending is None:  # (right after the prefill: its commit is done, nothing to commit)
            _, drafts = self.dr.draft(self.anchor, n, d=self.d)
        else:
            self.dr.set_inputs(self.anchor, *self.pending)
            self.dr.replay()
            drafts = [int(t) for t in torch.as_tensor(self.dr.tokens_host()).reshape(-1)[: self.d]]
        self.dec.set_inputs(torch.tensor([self.anchor] + drafts), torch.arange(n, n + R))
        self.dec.step()
        y = self._pick(*self.topk.host(self.dec.extra), R)
        k = 0
        while k < min(self.d, self.ver.rows_ok - 1) and drafts[k] == y[k]:
            k += 1
        self.ver.set_accept([k])
        self.pending = (n, k + 1)  # (the next drafter replay commits rows 0 .. k of this pass's taps)
        self.n, self.anchor = n + k + 1, y[k]
        self.passes, self.accepted = self.passes + 1, self.accepted + k
        return drafts[:k] + [y[k]]
