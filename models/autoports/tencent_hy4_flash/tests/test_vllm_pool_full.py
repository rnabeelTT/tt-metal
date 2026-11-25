# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU test of the vLLM adapter's shared-pool lifecycle (tt/generator_vllm.py), against a fake engine (2026-11-25):
release_request finishes the row's engine slot; a prompt the pool cannot take ends with EOS (prefill and every later
decode step) instead of raising; a dropped intermediate part is retried from its next part; condense keeps the marks.
python -m pytest models/autoports/tencent_hy4_flash/tests/test_vllm_pool_full.py (or run it as a module)"""
import sys
import types

import torch

if "ttnn" not in sys.modules:  # (tt/pool.py imports ttnn; only its PoolFull is used here)
    try:
        import ttnn  # noqa: F401
    except Exception:
        sys.modules["ttnn"] = types.ModuleType("ttnn")

from models.autoports.tencent_hy4_flash.tt import generator_vllm as G  # noqa: E402
from models.autoports.tencent_hy4_flash.tt.pool import PoolFull  # noqa: E402

V, EOS = 16, 7


class FakeEngine:
    multi = True

    def __init__(self, full=()):
        self.V, self.eos, self.seqs, self.cp, self.idle = V, {EOS}, {}, {}, set()
        self.stats, self.last_logits, self.finished, self.full = {}, {}, [], set(full)
        self.full_dec, self.stepped = set(), []
        self.on_evict = None

    def alloc(self, taken=(), ids=None, part=False):
        return next(s for s in range(8) if s not in self.seqs and s not in self.cp and s not in taken)

    def finish(self, slot):
        self.finished.append(slot)
        self.seqs.pop(slot, None)
        self.cp.pop(slot, None)

    def _first(self, slot, n):
        if slot in self.full:
            raise PoolFull(f"Hy4 pool full: slot {slot} needs {n} tokens")
        self.seqs[slot] = {"tok": 3, "adopt": False, "smp": None}
        self.last_logits[slot] = torch.zeros(V)
        return 3

    def prefill_many(self, reqs):
        if any(s in self.full for s, _, _ in reqs):
            raise PoolFull("together")
        return [self._first(s, len(ids)) for s, ids, _ in reqs]

    def prefill(self, slot, ids, smp, mm=None):
        return self._first(slot, len(ids))

    def prefill_part(self, slot, ids, final, sampling=None):
        if slot in self.full:
            raise PoolFull(f"Hy4 pool full: slot {slot} part")
        if not final:
            self.cp.setdefault(slot, {"ids": list(ids)})["ids"] = list(ids)
            return None
        self.cp.pop(slot, None)
        return self._first(slot, len(ids))

    def _grow(self, reqs):  # (as _ensure_step: before any device work, the first slot that cannot grow raises)
        for s in reqs:
            if s in self.full_dec:
                raise PoolFull(f"Hy4 pool full: slot {s} needs more", slot=s)
        self.stepped.append(sorted(reqs))

    def step(self, reqs):
        self._grow(reqs)
        return {s: [5] for s in reqs}

    def step_logits(self, reqs):
        self._grow(reqs)
        return {s: torch.zeros(V) for s in reqs}


def adapter(eng):
    a = G.HYV4FlashForConditionalGeneration.__new__(G.HYV4FlashForConditionalGeneration)
    G.HYV4FlashForConditionalGeneration.__init__(a, None, eng, None, 4096)
    return a


def toks(n, rows=1, base=1):
    return torch.arange(base, base + n).repeat(rows, 1)


def test_release_request_finishes_slot():
    eng = FakeEngine()
    a = adapter(eng)
    out = a._prefill_multi(toks(8, 2), [8, 8], sp=object(), rows=[0, 1])
    assert out.reshape(-1).tolist() == [3, 3] and a._row_slot == {0: 0, 1: 1}
    a.release_request(1)  # vLLM preempted / finished row 1
    assert 1 not in eng.seqs and a._row_slot == {0: 0} and eng.finished[-1] == 1
    a.release_request(5)  # (a row we never saw: nothing)
    assert a._row_slot == {0: 0}


def test_reject_whole_prompt_then_eos_decode():
    eng = FakeEngine(full={1})
    a = adapter(eng)
    out = a._prefill_multi(toks(8, 2), [8, 8], sp=object(), rows=[0, 1])
    assert out.reshape(-1).tolist() == [3, EOS]  # row 0 prefilled one at a time, row 1 rejected
    assert a._rejected == {1} and 1 not in a._row_slot and eng.stats["rejected"] == 1
    a._decode_steps = 1
    o = a._decode_multi(torch.tensor([3, 4]), torch.tensor([8, 8]), sp=object(), kwargs={})
    assert o[0, 0] == 5 and o[1, 0] == EOS
    lg = a._decode_multi(torch.tensor([3, 4]), torch.tensor([8, 8]), sp=None, kwargs={})
    assert int(lg[1, 0].argmax()) == EOS and torch.isinf(lg[1, 0]).sum() == V - 1
    a._remap([1, 0])  # condense swaps the rows: the mark follows
    assert a._rejected == {0} and a._row_slot == {1: 0}
    a.release_request(0)
    assert a._rejected == set()


def test_reject_host_sampling_logits():
    eng = FakeEngine(full={0})
    a = adapter(eng)
    lg = a._prefill_multi(toks(8, 1), [8], sp=None, rows=[0])
    assert int(lg[0, 0].argmax()) == EOS and a._rejected == {0}


def test_intermediate_part_deferred_then_final():
    eng = FakeEngine()
    a = adapter(eng)
    a._hy4_inter = None
    tk = toks(4096)
    # first part of row 2's prompt: slot 0
    a._prefill_parts(tk, [2048], sp=object(), rows=[2], mms=[None], start=[0], inter=[True])
    assert a._row_slot == {2: 0} and 0 in eng.cp
    eng.full = {0}  # its next part does not fit: dropped, not fatal
    out = a._prefill_parts(tk, [3072], sp=object(), rows=[2], mms=[None], start=[2048], inter=[True])
    assert out.reshape(-1).tolist() == [0] and 0 not in eng.cp and a._row_slot == {2: 0}
    eng.full = set()  # the final part fits: the prompt continues in its slot
    out = a._prefill_parts(tk, [4096], sp=object(), rows=[2], mms=[None], start=[3072], inter=[False])
    assert out.reshape(-1).tolist() == [3] and 0 in eng.seqs and not a._rejected


def test_final_part_rejected():
    eng = FakeEngine(full={0})
    a = adapter(eng)
    out = a._prefill_parts(toks(64), [64], sp=object(), rows=[3], mms=[None], start=[0], inter=[False])
    assert out.reshape(-1).tolist() == [EOS] and a._rejected == {3}


def test_decode_guard_ends_only_that_sequence():
    eng = FakeEngine()
    a = adapter(eng)
    a._prefill_multi(toks(8, 3), [8, 8, 8], sp=object(), rows=[0, 1, 2])
    a._decode_steps = 1
    eng.full_dec = {1}  # row 1's sequence cannot grow for this step
    o = a._decode_multi(torch.tensor([3, 3, 3]), torch.tensor([8, 8, 8]), sp=object(), kwargs={})
    assert o[:, 0].tolist() == [5, EOS, 5] and eng.stepped[-1] == [0, 2]
    assert a._rejected == {1} and 1 not in eng.seqs and eng.stats["rejected"] == 1
    eng.full_dec = set()  # next step: row 1 stays EOS without touching the engine, the others decode
    o = a._decode_multi(torch.tensor([3, 4, 3]), torch.tensor([9, 9, 9]), sp=object(), kwargs={})
    assert o[:, 0].tolist() == [5, EOS, 5] and eng.stepped[-1] == [0, 2]
    eng.full_dec = {0, 2}  # host-sampled path, every sequence out of room
    lg = a._decode_multi(torch.tensor([3, 4, 3]), torch.tensor([10, 10, 10]), sp=None, kwargs={})
    assert [int(lg[i, 0].argmax()) for i in range(3)] == [EOS] * 3 and a._rejected == {0, 1, 2}


if __name__ == "__main__":  # (no pytest needed: python -m models.autoports.tencent_hy4_flash.tests.test_vllm_pool_full)
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
