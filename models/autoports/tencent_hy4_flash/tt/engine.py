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


# HY4_FAST_DRAW=1 (default): sampled tokens by inverse CDF instead of torch.multinomial (same distribution; a seeded
# request draws a different, equally distributed sequence than before)
_FAST_DRAW = os.environ.get("HY4_FAST_DRAW", "1") == "1"


class HostSampler:
    def __init__(self, sampling):
        t, k, p, seed = sampling if sampling is not None else (0.0, 0, 1.0, None)
        self.t, self.k, self.p = float(t), int(k or 0), float(p if p is not None else 1.0)
        self.gen = torch.Generator()
        if seed is not None:
            self.gen.manual_seed(int(seed))
        else:
            self.gen.seed()

    @property
    def greedy(self) -> bool:
        return self.t <= 0.0

    def probs(self, logits: torch.Tensor) -> torch.Tensor:
        """The sampling distribution (temperature, top-k, top-p) over the vocabulary."""
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
        return x.softmax(-1)

    def draw(self, p: torch.Tensor) -> int:
        if _FAST_DRAW:
            return self._icdf(p.float(), inplace=False)
        return int(torch.multinomial(p, 1, generator=self.gen))

    def _icdf(self, w: torch.Tensor, inplace: bool) -> int:
        """One draw from the distribution proportional to the non-negative weights w (inverse CDF: cumsum + one uniform
        + binary search). Same distribution as torch.multinomial, ~20x cheaper on a 120K vocabulary (multinomial: 1.6 ms
        per row single-threaded; 8 sampled rows were 14 ms per batched step)."""
        c = w.cumsum_(0) if inplace else w.cumsum(0)
        u = float(torch.rand(1, generator=self.gen, dtype=torch.float64)) * float(c[-1])
        i = int(torch.searchsorted(c, torch.tensor([u], dtype=c.dtype), right=True))
        n = c.numel() - 1
        i = min(i, n)
        while i > 0 and w is not c and float(w[i]) == 0.0:  # never land on a zero-weight (filtered) entry
            i -= 1
        return i

    def uniform(self) -> float:
        return float(torch.rand(1, generator=self.gen))

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
        if _FAST_DRAW:
            return self._icdf((x - x.max()).exp_(), inplace=False)
        return int(torch.multinomial(x.softmax(-1), 1, generator=self.gen))


class Hy4Engine:
    multi = False
    adaptive = False
    in_trace = False

    def __init__(self, model, k: int = 0, trace: bool = True):
        from models.autoports.tencent_hy4_flash.tt.decode import Hy4Decoder, Hy4SpecDecoder

        self.model, self.cfg = model, model.cfg
        self.k = int(k)
        # k > 0: DFly2 speculative decoding (5 drafts per verify pass: the drafter's block width - 1)
        self.dec = Hy4SpecDecoder(model, trace=trace, k=self.k) if self.k > 0 else Hy4Decoder(model, trace=trace)
        if self.k > 0:
            self.k = self.dec.k
        self._n = None  # spec: the last pass's last accepted row (None: no pass since the prefill -> a 1-row pass)
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
        g = torch.Generator().manual_seed(0)
        # every prefill bucket first (tt/model.py PF_BUCKETS: a prompt's chunks run one of these program sets), so no
        # program is compiled after the decode / draft traces are captured (which would release and re-capture them)
        if os.environ.get("HY4_PF_BUCKETS", "1") == "1":
            for b in self.model.PF_BUCKETS:
                if b <= self.chunk and b < self.model.max_ctx:
                    self.prefill(torch.randint(1000, 100000, (b - 1,), generator=g).tolist())
                    self.model.warm_head_blocks(b)
            # later chunks (p0 >= the first chunk) run other programs: the sparse layers' selecting regime (its score
            # width bucket grows with the context: 1024 groups to 4K tokens, 2048 to 8K, ...) and the SWA flash over the
            # window. HY4_WARM_LONG=auto (default): every (score bucket, row bucket) the context allows
            # (tt/model.py warm_long; before: m <= 2 only, so a longer prompt compiled at serve time and released the
            # traces); a number m: prompts whose last chunk (every bucket) starts at 2048 * m' for m' <= m
            wl = os.environ.get("HY4_WARM_LONG", "auto")
            if wl == "auto":
                self.model.warm_long(lambda n: torch.randint(1000, 100000, (n,), generator=g).tolist())
            m_max = 0 if wl == "auto" else int(wl)
            for m in range(1, m_max + 1):
                for b in self.model.PF_BUCKETS:
                    L = self.chunk * m + b - 1
                    if b <= self.chunk and L + 64 < self.model.max_ctx:
                        self.prefill(torch.randint(1000, 100000, (L,), generator=g).tolist())
                        self.model.warm_head_blocks(b)
        n = min(int(prompt_len or 64), self.chunk)
        ids = torch.randint(1000, 100000, (n,), generator=g).tolist()
        self.prefill(ids)
        # plain: the first pass compiles + captures, the next replay; spec (k > 0): the 1-row pass, then drafts and
        # verify rows (every accepted-row draft trace is captured on first use)
        for _ in range(int(os.environ.get("HY4_WARM_STEPS", "8"))):
            self.step()
        if os.environ.get("HY4_PF_BUCKETS", "1") == "1" and os.environ.get("HY4_WARM_LONG", "auto") != "0":
            # decode past the dense regime (context > 2048: the indexer selects, other programs and traces)
            L = self.chunk + 1000
            if L + 64 < self.model.max_ctx:
                self.prefill(torch.randint(1000, 100000, (L,), generator=g).tolist())
                for _ in range(int(os.environ.get("HY4_WARM_STEPS", "8"))):
                    self.step()
                # every larger score bucket (forced: a longer context's passes), plain and spec (a 1-row pass first)
                for b in self.dec.buckets[1:]:
                    self.dec.min_bucket = b
                    self._n = None
                    for _ in range(int(os.environ.get("HY4_WARM_STEPS", "8"))):
                        self.step()
                self.dec.min_bucket = 0
        self.model.reset()
        self._pos = None
        self.stats = {k: 0 if isinstance(v, int) else 0.0 for k, v in self.stats.items()}
        logger.info(f"Hy4 engine warm-up {time.time() - t0:.1f} s")

    def capture(self):
        """Decode traces are captured on first use per (rows, regime, context bucket) by tt/decode.Hy4Decoder."""

    # ------------------------------------------------------------------ prefill / decode
    def prefill(self, prompt_ids, sampling=None, mm=None) -> int:
        """mm: image inputs of the prompt (Hy4Model.set_images kwargs, tt/vision.py Hy4Vision.prepare)."""
        t0 = time.time()
        ids = torch.as_tensor(prompt_ids, dtype=torch.long).reshape(-1)
        self.model.reset()
        if mm is not None:
            self.model.set_images(**mm)
        lg = self.model.prefill(ids)  # the dense-regime chunk first, then the model's chunk (tt/model.py prefill)
        self.model.prefill_caches_release()  # (cached device constants must not outlive it into a decode replay)
        self.last_logits = lg
        self.dec.start(int(ids.numel()))
        self._smp = HostSampler(sampling)
        tok = self._smp(lg)
        self._tok, self._pos = tok, int(ids.numel())
        if hasattr(self.dec, "set_history"):
            self.dec.set_history(ids.tolist())
        self._n = None
        dt = time.time() - t0
        self.stats["t_prefill"] += dt
        self.stats["prefills"] += 1
        self.stats["prefill_tokens"] += int(ids.numel())
        logger.info(f"Hy4 prefill {ids.numel()} tokens in {dt:.2f} s")
        return tok

    def _decode(self, tok: int) -> torch.Tensor:
        t0 = time.time()
        lg = self.dec.run([int(tok)], p0=self._pos)  # (explicit: a verify pass leaves the decoder past rejected rows)
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
        if self.k > 0 and self._n is not None:
            return self._spec_step()
        lg = self._decode(self._tok)
        nxt = self._smp(lg)
        self._tok, self._pos = nxt, self._pos + 1
        self.stats["emitted"] += 1
        self._n = 0 if self.k > 0 else None  # (the drafter reads this 1-row pass's row 0)
        return [nxt]

    def _spec_step(self) -> list[int]:
        """One verify pass: rows [tok, d_1 .. d_k] at positions pos .. pos + k (drafts from DFly2 on the last pass's
        target state); greedy: the matching prefix + the target's next token; sampled: exact speculative sampling with
        deterministic drafts (accept d_j with probability p_j(d_j); on a rejection draw from p_j with d_j removed;
        after k acceptances draw the bonus from p_k)."""
        t0 = time.time()
        drafts = self.dec.draft(self._tok, self._pos, self._n)[: self.k]
        t1 = time.time()
        rows = [self._tok] + drafts
        lgs = self.dec.run(rows, logits_rows="all", p0=self._pos)  # [k + 1, V]
        self.stats["t_draft"] = self.stats.get("t_draft", 0.0) + (t1 - t0)
        self.stats["t_step"] += time.time() - t0
        self.stats["steps"] += 1
        emit = []
        if self._smp.greedy:
            picks = lgs.argmax(-1).tolist()
            acc = 0
            while acc < len(drafts) and picks[acc] == drafts[acc]:
                acc += 1
            emit = drafts[:acc] + [int(picks[acc])]
        else:
            acc = 0
            while acc < len(drafts):
                p = self._smp.probs(lgs[acc])
                d = drafts[acc]
                if self._smp.uniform() < float(p[d]):
                    acc += 1
                    continue
                p[d] = 0.0
                emit = drafts[:acc] + [self._smp.draw(p / p.sum())]
                break
            if not emit:
                emit = drafts + [self._smp.draw(self._smp.probs(lgs[len(drafts)]))]
        self._n = acc
        self._pos += len(emit)
        self._tok = emit[-1]
        self.stats["emitted"] += len(emit)
        self.stats["accepted"] = self.stats.get("accepted", 0) + acc
        return emit

    def step_logits(self, tok: int, pos: int) -> torch.Tensor:
        """Host-sampled step: feed `tok` at `pos` (must continue the sequence), return the next-token logits."""
        if int(pos) != self._pos:
            raise RuntimeError(f"step_logits at pos {pos}, engine at {self._pos}")
        lg = self._decode(int(tok))
        self._pos += 1
        self._tok = None
        self.adopt_next = True
        self._n = 0 if self.k > 0 else None  # a 1-row pass: the drafter can continue from its row 0
        return lg
