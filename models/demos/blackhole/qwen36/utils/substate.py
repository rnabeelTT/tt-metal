# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Helpers for slicing nested state dicts."""
from __future__ import annotations

import itertools
from collections.abc import Mapping
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


class LazyState(Mapping):
    """A read-only state dict whose values load on first access (``getter(full key)``, then kept): a layer's view of a
    streaming checkpoint, so tensors a build never reads (e.g. routed experts supplied from a device-image cache) are
    never loaded. ``substate`` of a LazyState is again lazy."""

    def __init__(self, getter, keys, prefix=""):
        self._get, self._prefix = getter, prefix
        self._keys = list(keys)
        self._cache = {}

    def __getitem__(self, key):
        if key not in self._cache:
            if key not in self._keyset():
                raise KeyError(key)
            self._cache[key] = self._get(self._prefix + key)
        return self._cache[key]

    def _keyset(self):
        ks = self.__dict__.get("_ks")
        if ks is None:
            ks = self._ks = set(self._keys)
        return ks

    def __iter__(self):
        return iter(self._keys)

    def __len__(self):
        return len(self._keys)

    def __contains__(self, key):
        return key in self._keyset()

    def sub(self, key):
        prefix = f"{key}."
        n = len(prefix)
        return LazyState(self._get, [k[n:] for k in self._keys if k.startswith(prefix)], self._prefix + prefix)


def substate(state: dict[str, "torch.Tensor"], key: str) -> dict[str, "torch.Tensor"]:
    """Return the sub-dict of entries whose keys start with `key.`, with that prefix removed (lazy for a LazyState)."""
    if isinstance(state, LazyState):
        return state.sub(key)
    prefix = f"{key}."
    prefix_len = len(prefix)
    return {k[prefix_len:]: v for k, v in state.items() if k.startswith(prefix)}


def has_substate(state: dict[str, "torch.Tensor"], key: str) -> bool:
    """True if any key starts with `key.`."""
    prefix = f"{key}."
    return any(k.startswith(prefix) for k in state)


def indexed_substates(state: dict[str, "torch.Tensor"], key: str) -> list[dict[str, "torch.Tensor"]]:
    """Extract a list of indexed sub-states (e.g. `key.0`, `key.1`, ...)."""
    result = []
    for i in itertools.count():
        s = substate(state, f"{key}.{i}")
        if not s:
            return result
        result.append(s)
    return []
