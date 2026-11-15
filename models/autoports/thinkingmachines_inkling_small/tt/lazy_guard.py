# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Prefill's lazily built device caches vs. the decode traces.

A device tensor allocated after a trace capture can land in that trace's freed intermediates, and every replay then
overwrites it. The MoE prefill keeps such tensors across calls: the expert-sorted dispatch objects
(``_dispatch``, ``_dispatch_two`` per staging size, and the per-row-count constants each one keeps in ``_const``) and
the sparse matmuls' ``_pairs_masks``. One built during a prefill after the captures (the first streamed long prompt
builds the two-set dispatch of the staging path; a row count not seen in the warm-up builds new constants) was
overwritten by the decode replays that followed, and the next prefill read it: the streamed-prefill hang
(HANDOFF_INKLING.md, 2026-11-15).

``LazyCacheGuard.snapshot()`` (at the end of the captures) records what exists; ``purge()`` (after every prefill)
frees what was built since, so a post-capture entry never outlives the prefill that made it. Entries from before the
captures are left alone (the warm-up's eager passes allocated them before any trace existed).
"""
from __future__ import annotations

import ttnn

_KEYS = ("_dispatch", "_dispatch_two", "_pairs_masks")


def _free(t):
    if isinstance(t, ttnn.Tensor):
        try:
            ttnn.deallocate(t)
        except Exception:  # noqa: BLE001 - already freed / host tensor
            pass


def _free_dispatch(d):
    _free(getattr(d, "arange_E", None))
    for c in list(getattr(d, "_const", {}).values()):
        for t in c.values():
            _free(t)
    getattr(d, "_const", {}).clear()


class LazyCacheGuard:
    def __init__(self, modules, args_list):
        self.modules = [m for m in modules if m is not None]
        self.args_list = [a for a in args_list if a is not None]
        self.snap = None
        self.purged = 0

    def _containers(self):
        """Every dict that can hold the caches: each MoE module's own attributes and its attribute sets swapped in for
        the staging / dynamic expert calls; the model args (the shared pairs masks)."""
        for m in self.modules:
            yield m.__dict__
            for name in ("_stream_attrs", "_dyn_attrs", "_static_attrs"):
                a = getattr(m, name, None)
                if isinstance(a, dict):
                    yield a
        for a in self.args_list:
            yield a.__dict__

    def _dispatches(self, c):
        d = c.get("_dispatch")
        if d is not None:
            yield ("_dispatch", None), d
        for k, v in (c.get("_dispatch_two") or {}).items():
            yield ("_dispatch_two", k), v

    def snapshot(self):
        objs, consts, masks = set(), set(), set()
        for c in self._containers():
            for _, d in self._dispatches(c):
                objs.add(id(d))
                consts.update((id(d), T) for T in getattr(d, "_const", {}))
            for key in c.get("_pairs_masks") or {}:
                masks.add((id(c), key))
        self.snap = (objs, consts, masks)

    def purge(self):
        """Free every cache entry built since snapshot(); returns how many."""
        if self.snap is None:
            return 0
        objs, consts, masks = self.snap
        n = 0
        for c in self._containers():
            for (key, sub), d in list(self._dispatches(c)):
                if id(d) not in objs:
                    _free_dispatch(d)
                    if sub is None:
                        c[key] = None
                    else:
                        del c[key][sub]
                    n += 1
                    continue
                for T in [T for T in getattr(d, "_const", {}) if (id(d), T) not in consts]:
                    for t in d._const.pop(T).values():
                        _free(t)
                    n += 1
            pm = c.get("_pairs_masks")
            if pm:
                for key in [k for k in pm if (id(c), k) not in masks]:
                    _free(pm.pop(key))
                    n += 1
        self.purged += n
        return n
