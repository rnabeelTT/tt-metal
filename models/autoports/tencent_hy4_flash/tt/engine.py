# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Single-user generation engine for Hy4-flash (day 0: eager, no drafter).

The vLLM adapter (tt/generator_vllm.py) drives it through the same contract as the Inkling engine: prefill(ids,
sampling) -> first token; step(tok, pos, ...) -> [next token]; step_logits(tok, pos) -> logits (host-sampled requests);
last_logits after a prefill. Sampling (temperature / top-k / top-p / seed) runs on the host over the full fp32 logits.

The model's caches are position-indexed, so a new prompt that starts with the previous sequence's tokens could resume
(prefix reuse); day 0 restarts every request (model.reset()).
"""

from __future__ import annotations

import os
import time

import torch
from loguru import logger


class HostSampler:
    def __init__(self, sampling):
        t, k, p, seed = sampling if sampling is not None else (0.0, 0, 1.0, None)
        self.t, self.k, self.p = float(t), int(k or 0), float(p if p is not None else 1.0)
        self.gen = torch.Generator()
        if seed is not None:
            self.gen.manual_seed(int(seed))
        else:
            self.gen.seed()

    def __call__(self, logits: torch.Tensor) -> int:
        if self.t <= 0.0:
            return int(torch.argmax(logits))
        x = logits.float() / self.t
        if self.k > 0 and self.k < x.numel():
            kth = torch.topk(x, self.k).values[-1]
            x = torch.where(x < kth, torch.full_like(x, float("-inf")), x)
        if self.p < 1.0:
            sx, si = torch.sort(x, descending=True)
            cp = sx.softmax(-1).cumsum(-1)
            drop = cp - sx.softmax(-1) > self.p
            sx[drop] = float("-inf")
            x = torch.full_like(x, float("-inf")).scatter(0, si, sx)
        return int(torch.multinomial(x.softmax(-1), 1, generator=self.gen))


class Hy4Engine:
    multi = False
    adaptive = False
    in_trace = False

    def __init__(self, model, k: int = 0, trace: bool = True):
        from models.autoports.tencent_hy4_flash.tt.decode import Hy4Decoder

        self.model, self.cfg = model, model.cfg
        self.dec = Hy4Decoder(model, trace=trace)
        self.k = int(k)
        assert self.k == 0, "the DFly2 drafter is not wired yet (HY4_SPEC_K=0)"
        self.trace = bool(trace)
        self.V = int(self.cfg.vocab)
        self.eos = {int(self.cfg.eos)}
        self.chunk = int(model.chunk)
        self.adopt_next = False
        self.last_logits = None
        self._tok = self._pos = None
        self._smp = HostSampler(None)
        self.stats = {"steps": 0, "emitted": 0, "t_step": 0.0, "t_prefill": 0.0, "prefills": 0, "prefill_tokens": 0}

    # ------------------------------------------------------------------ warm-up
    def warm(self, prompt_len=None):
        """Compile the programs of a prefill chunk and a decode step (eager: just run them once)."""
        t0 = time.time()
        self.model.reset()
        n = min(int(prompt_len or 64), self.chunk)
        self.model.forward(torch.full((n,), int(self.cfg.bos), dtype=torch.long))
        self.dec.start(n)
        self.dec.run([int(self.cfg.bos)])
        self.dec.run([int(self.cfg.bos)])  # (traced: the first pass compiles + captures, the second replays)
        self.model.reset()
        logger.info(f"Hy4 engine warm-up {time.time() - t0:.1f} s")

    def capture(self):
        """Decode traces are captured on first use per (rows, regime, context bucket) by tt/decode.Hy4Decoder."""

    # ------------------------------------------------------------------ prefill / decode
    def prefill(self, prompt_ids, sampling=None) -> int:
        t0 = time.time()
        ids = torch.as_tensor(prompt_ids, dtype=torch.long).reshape(-1)
        self.model.reset()
        lg = None
        for c0 in range(0, ids.numel(), self.chunk):
            lg = self.model.forward(ids[c0 : c0 + self.chunk])
        self.last_logits = lg
        self.dec.start(int(ids.numel()))
        self._smp = HostSampler(sampling)
        tok = self._smp(lg)
        self._tok, self._pos = tok, int(ids.numel())
        dt = time.time() - t0
        self.stats["t_prefill"] += dt
        self.stats["prefills"] += 1
        self.stats["prefill_tokens"] += int(ids.numel())
        logger.info(f"Hy4 prefill {ids.numel()} tokens in {dt:.2f} s")
        return tok

    def _decode(self, tok: int) -> torch.Tensor:
        t0 = time.time()
        lg = self.dec.run([int(tok)])
        self.stats["t_step"] += time.time() - t0
        self.stats["steps"] += 1
        return lg

    def step(self, tok=None, pos=None, page_table_row=None, host_authoritative=False, sampling=None) -> list[int]:
        if self._pos is None:
            raise RuntimeError("Hy4 step before any prefill")
        adopt, self.adopt_next = self.adopt_next, False
        if (host_authoritative or adopt) and tok is not None and int(tok) != self._tok:
            if int(pos) != self._pos:
                raise RuntimeError(f"step input (tok {tok}, pos {pos}) vs engine (tok {self._tok}, pos {self._pos})")
            self._tok = int(tok)
        lg = self._decode(self._tok)
        nxt = self._smp(lg)
        self._tok, self._pos = nxt, self._pos + 1
        self.stats["emitted"] += 1
        return [nxt]

    def step_logits(self, tok: int, pos: int) -> torch.Tensor:
        """Host-sampled step: feed `tok` at `pos` (must continue the sequence), return the next-token logits."""
        if int(pos) != self._pos:
            raise RuntimeError(f"step_logits at pos {pos}, engine at {self._pos}")
        lg = self._decode(int(tok))
        self._pos += 1
        self._tok = None
        self.adopt_next = True
        return lg
