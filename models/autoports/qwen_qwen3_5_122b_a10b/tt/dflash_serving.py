# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""DFlash serving: speculative decoding with the DFlash 2 drafter for a lone request (QWEN35_DFLASH=1).

    The adaptive MTP server's plumbing (tt/mtp_serving.MtpServing: GDN scratch binding, enter / leave / suspend, the
    variable-width output shim) with the DFlash runner (tt/dflash_traced.DFlashTraced) in place of the MTP head:

    prefill : inside every traced bucket program the drafter's context layers are captured (model._dflash_layer_hook)
              and the drafter writes the K / V of the segment's rows into its rings (model._mtp_bucket_hook); the host
              stages the segment's positions (model._mtp_stage_hook). The rings hold ONE request's context: the last
              request prefilled ("owner"). Prefix-cache hits run only the uncached suffix, so the drafter sees fewer
              context rows for such a request (draft quality only).
    decode  : exactly one live request whose slot owns the rings -> one DFlash round per step (1 .. bs tokens); any other
              case -> the plain batched decode (the rings are not advanced there: a request decoded plainly for a while
              drafts from an older context when it is alone again).
    guard   : a phase whose running acceptance falls below QWEN35_DFLASH_MIN_TAU tokens per round (after
              QWEN35_DFLASH_GUARD_ROUNDS rounds) hands its request to the plain decode for good (the request's own text
              is not predictable enough for drafting to pay).

Output is unchanged by construction: greedy requests emit the target's argmax at every position; sampled requests use
the exact speculative accept (tt/mtp_sampling.SpecSampler) with the selector's distribution as the draft distribution.
"""
from __future__ import annotations

import glob
import os
import time

import torch
from loguru import logger

import ttnn
from models.autoports.qwen_qwen3_5_122b_a10b.tt.mtp_rows import rebuild_conv_carry
from models.autoports.qwen_qwen3_5_122b_a10b.tt.mtp_serving import MtpServing

_DRAM = ttnn.DRAM_MEMORY_CONFIG


DRAFTER_REPO = "incoai/Ornith-1.5-122B-DFlash2"


def drafter_dir():
    """The drafter checkpoint: QWEN35_DFLASH_DRAFTER (a directory or a repo id), else the tt-model bundle's auxiliary
    weights entry for the drafter repo (TT_AUXILIARY_WEIGHTS=repo@sha,...: prefetched into the host HF cache), else the
    default repo from the HF cache / hub."""
    d = os.environ.get("QWEN35_DFLASH_DRAFTER")
    if d and os.path.isdir(d):
        return d
    repo, rev = d or DRAFTER_REPO, None
    for entry in filter(None, os.environ.get("TT_AUXILIARY_WEIGHTS", "").split(",")):
        r, _, sha = entry.partition("@")
        if r == repo:
            rev = sha or None
    hub = os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub")
    base = os.path.join(hub, "models--" + repo.replace("/", "--"), "snapshots")
    if rev and os.path.isdir(os.path.join(base, rev)):
        return os.path.join(base, rev)
    snaps = glob.glob(os.path.join(base, "*"))
    if snaps:
        return snaps[0]
    from huggingface_hub import snapshot_download

    return snapshot_download(repo, revision=rev)


class DFlashServing(MtpServing):
    def __init__(self, gen, bs: int):
        # MtpServing.__init__ builds the MTP head: set the fields by hand instead
        self.gen, self.k = gen, int(bs) - 1
        self.bs = int(bs)
        self.B = int(gen.max_batch_size)
        self.adaptive = True
        self.multi = False
        self.per_slot = self.B > 1
        self._phase_slot = None
        self._scratch_prev = None
        self._suspended = False
        self._force_reload = False
        self.model = gen.model
        self.mesh = self.model.mesh_device
        self.H = int(gen.args.dim)
        self.head = None
        self.runner = None
        self._pos = None
        self._tok = None
        self._stats = None
        self._prefill_queue = []
        self._cur = None  # (slot, prompt) of the request whose bucket segments are running
        self._owner = None  # slot whose context the drafter rings hold
        self._off = set()  # slots the acceptance guard handed to the plain decode (until a new request takes the slot)
        self.in_trace = True
        # acceptance guard: after QWEN35_DFLASH_GUARD_ROUNDS rounds, a request whose rounds produced fewer tokens than
        # the plain decode would have in the same wall time (QWEN35_DFLASH_PLAIN_MS per token, x QWEN35_DFLASH_MARGIN)
        # goes to the plain decode for the rest of the request; QWEN35_DFLASH_MIN_TAU (tokens / round) is a floor
        self.min_tau = float(os.environ.get("QWEN35_DFLASH_MIN_TAU", "1.5"))
        self.plain_ms = float(os.environ.get("QWEN35_DFLASH_PLAIN_MS", "7.6"))
        self.margin = float(os.environ.get("QWEN35_DFLASH_MARGIN", "1.0"))
        self.guard_rounds = int(os.environ.get("QWEN35_DFLASH_GUARD_ROUNDS", "6"))
        from models.autoports.qwen_qwen3_5_122b_a10b.tt.dflash_device import DFlashDevice

        self.drafter = DFlashDevice(self.model, drafter_dir())
        self._caps = {}
        self._bucket_feats = {}
        logger.info(f"DFlash serving: block {self.bs}, slots {self.B}, guard {self.min_tau} tok/round")

    # ------------------------------------------------------------------ MtpServing surface
    @property
    def no_head_slots(self):
        """Slots that must take the plain decode when alone: every slot but the rings' owner, and guarded ones."""
        return {s for s in range(self.B) if s != self._owner} | self._off

    @no_head_slots.setter
    def no_head_slots(self, v):  # MtpServing.remap assigns it; ownership is remapped in remap() below
        pass

    def allocate_kv(self, kv_shape, dtype):
        pass

    def _ensure_runner(self):
        if self.runner is not None:
            return
        from models.autoports.qwen_qwen3_5_122b_a10b.tt.dflash_traced import DFlashTraced

        prev = self.model._bind_gdn_mtp_scratch()
        try:
            self.runner = DFlashTraced(self.gen, self.drafter, self.bs)
        finally:
            self.model._unbind_gdn_mtp_scratch(prev)
        m = self.model
        for b in list(m._PREFILL_MASK_BUCKETS) + [self.bs]:
            self.drafter.ctx_stage(b)  # persistent staging inputs before any warm-up / capture
        m._dflash_layer_hook = self._layer_hook
        m._mtp_bucket_hook = self._bucket_ctx
        m._mtp_stage_hook = self._bucket_stage

    def warm(self):
        self._ensure_runner()
        with self._bound():
            self.runner.warm()
        self.warm_adaptive()
        ttnn.synchronize_device(self.mesh)

    def warm_prefill_path(self):
        """Every bucket program WITH the drafter's context hooks, at its full and a masked length, before any capture
        (the generator's own prefill warm-up ran before the hooks were installed)."""
        self._ensure_runner()
        pt = self.gen.page_table[:1]
        prev = self.model._bind_gdn_prefill_scratch() if self.per_slot else None
        try:
            for b in self.model._PREFILL_MASK_BUCKETS:
                for n in (b, max(2, b - 1)):
                    ids = [1000 + (i % 977) for i in range(n)]
                    self.begin_prefill(ids, pt[0], slot=0)
                    self.on_request(0, 0, n)
                    try:
                        if self.per_slot:
                            self.model._reset_gdn_state_for_new_sequence()
                        self.gen._prefill_single(
                            torch.tensor(ids, dtype=torch.int32).reshape(1, -1), pt, want_sharded=True
                        )
                    finally:
                        self.abort_prefill()
        finally:
            if prev is not None:
                self.model._unbind_gdn_prefill_scratch(prev)
        if not self.per_slot:
            self.model._reset_gdn_state_for_new_sequence()
        self._owner = None
        self.drafter.reset()
        ttnn.synchronize_device(self.mesh)

    def capture(self):
        self._ensure_runner()
        with self._bound():
            self.runner.capture()
        logger.info("DFlash traces captured (pass, commit, ctx, draft)")

    def _pick_row(self, slot):
        pass

    def _put_row(self, slot, row):
        pass

    def remap(self, remap):
        idx = [int(remap[i]) for i in range(self.B)]
        if all(idx[i] == i for i in range(self.B)):
            return
        assert self._phase_slot is None, "leave the DFlash phase before a slot remap"
        inv = {j: i for i, j in enumerate(idx)}
        self._owner = inv.get(self._owner) if self._owner is not None else None
        self._off = {inv[s] for s in self._off if s in inv}

    def set_page_table(self, row):
        r = self.runner
        row = torch.as_tensor(row).to(torch.int32).reshape(1, -1)
        last = getattr(self, "_last_pt_row", None)
        if last is not None and last.shape == row.shape and bool(torch.equal(last, row)):
            return
        self._last_pt_row = row.clone()
        for buf in (r.p_pt,):
            n = buf.shape[0]
            ttnn.copy_host_to_device_tensor(
                ttnn.from_torch(row.repeat(n, 1), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT), buf
            )

    def enter(self, slot, tok, pos, page_table_row):
        assert self._phase_slot is None
        t0 = time.perf_counter()
        m = self.model
        self._scratch_prev = m._bind_gdn_mtp_scratch()
        for (dn, rec, conv, _carry, _z), (_dn, _Bb, rec_b, conv_b, *_rest) in zip(m._gdn_mtp_scratch, self._scratch_prev):
            src = dn._slice_along(rec_b, 0, slot, slot + 1)
            ttnn.copy(src, rec)
            ttnn.deallocate(src)
            for j in range(dn.K):
                src = dn._slice_along(conv_b[j], 1, slot, slot + 1)
                ttnn.copy(src, conv[j])
                ttnn.deallocate(src)
            rebuild_conv_carry(dn)
        self._phase_slot = int(slot)
        self.runner.set_base(None)  # the slot's state was just copied into the bound scratch
        self.runner.rope_delta = self.model.slot_rope_delta(slot)
        self.set_page_table(page_table_row)
        self._tok, self._pos = int(tok), int(pos)
        self._sampler = None
        self._log_stats("previous phase")
        self._stats = {"passes": 0, "tokens": 0, "hist": [0] * (self.bs + 1), "wall": 0.0}
        logger.info(f"DFlash phase enter slot {slot} at pos {pos}: {(time.perf_counter() - t0) * 1e3:.1f} ms")

    def leave(self):
        """Phase end with the state written back: first the verify passes' state (snapshot / history of the last accepted
        row) into the MTP scratch the write-back reads."""
        if self._phase_slot is not None:
            if self._suspended:
                prev = self.model._bind_gdn_mtp_scratch()
                try:
                    self.runner.materialize()
                finally:
                    self.model._unbind_gdn_mtp_scratch(prev)
            else:
                self.runner.materialize()
        return super().leave()

    def drop(self):
        if self.runner is not None:
            self.runner.set_base(None)
        return super().drop()

    def step(self, tok: int, pos: int, page_table_row=None, host_authoritative=False, sampling=None) -> list[int]:
        t0 = time.perf_counter()
        assert self.in_phase and not self._suspended, "DFlash: step() outside a phase"
        if page_table_row is not None:
            self.set_page_table(page_table_row)
        if host_authoritative and self._pos is not None and (pos != self._pos or tok != self._tok):
            logger.warning(f"DFlash step input (tok {tok}, pos {pos}) differs from the model's ({self._tok}, {self._pos})")
        tok = self._tok if self._tok is not None else tok
        pos = self._pos if self._pos is not None else pos
        sampler = self._sampler_for(sampling)
        out = self.runner.round(tok, pos, sampler)
        self._pos = pos + len(out)
        self._tok = out[-1]
        st = self._stats
        st["passes"] += 1
        st["tokens"] += len(out)
        st["hist"][len(out)] += 1
        st["wall"] += time.perf_counter() - t0
        if st["passes"] % 64 == 0:
            self._log_stats("running")
        plain_equiv = st["wall"] * 1e3 / self.plain_ms * self.margin  # tokens the plain decode makes in that time
        if st["passes"] >= self.guard_rounds and (
            st["tokens"] < self.min_tau * st["passes"] or st["tokens"] < plain_equiv
        ):
            # not paying: the rest of this request runs the plain decode (the next decode step leaves the phase)
            self._off.add(self._phase_slot)
            self._log_stats("guard: plain decode for the rest of the request")
        return out

    def _log_stats(self, why):
        st = self._stats
        if st and st["passes"]:
            ms = st["wall"] / st["passes"] * 1e3
            logger.info(
                f"DFLASH_STATS ({why}): {st['passes']} rounds, {st['tokens']} tokens = "
                f"{st['tokens'] / st['passes']:.2f} per round, hist {st['hist']}, {ms:.1f} ms per round -> "
                f"{st['tokens'] / st['wall']:.1f} tok/s (device steps only)"
            )

    # ------------------------------------------------------------------ prefill side
    def begin_prefill(self, prompt_ids, page_table_row, slot: int = 0):
        self._prefill_queue.append((int(slot), len(prompt_ids)))
        self._off.discard(int(slot))

    def on_request(self, slot, start, end):
        """generator.prefill_serving: request `slot` runs its segments [start, end) next (start > 0: a prefix-cache hit
        or a chunk continuation)."""
        slot = int(slot)
        for i, (s, _n) in enumerate(self._prefill_queue):
            if s == slot:
                self._prefill_queue.pop(i)
                break
        if start == 0 or self._owner != slot:
            self.drafter.reset()  # a new context (a resumed request without its earlier rows keeps what it gets)
        self._cur = slot
        self._owner = slot

    def note_batched(self, slot, hid_row, page_table_row):
        """A request admitted by the batched short-prompt pass: no context rows for it (no DFlash phase)."""
        slot = int(slot)
        for i, (s, _n) in enumerate(self._prefill_queue):
            if s == slot:
                self._prefill_queue.pop(i)
                break
        if self._owner == slot:
            self._owner = None

    def _bucket_stage(self, bi, bucket, cs, actual_len):
        """Host side, after a bucket's inputs are staged: the segment's positions for the drafter's context rows."""
        n = int(actual_len)
        pos = [cs + j for j in range(n)]
        self.drafter.stage_ctx(pos, bucket)

    def _layer_hook(self, bi, bucket, li, x):
        if li in self.drafter.layer_ids:
            self._caps[li] = self.drafter._full(ttnn.clone(x, memory_config=_DRAM))

    def _bucket_ctx(self, bi, bucket, cs, x):
        caps = [self._caps.pop(li) for li in self.drafter.layer_ids]
        feats = ttnn.concat(caps, dim=3, memory_config=_DRAM)  # [1,1,bucket,8H]
        for c in caps:
            ttnn.deallocate(c)
        self.drafter.ctx_body(feats)
        ttnn.deallocate(feats)

    def abort_prefill(self):
        self._prefill_queue = []
        self._cur = None

    def end_prefill(self, tok0, P, slots=None):
        self._prefill_queue = []
        self._cur = None
        if self._phase_slot is not None and self._phase_slot in [int(x) for x in (slots or [])]:
            self.drop()
