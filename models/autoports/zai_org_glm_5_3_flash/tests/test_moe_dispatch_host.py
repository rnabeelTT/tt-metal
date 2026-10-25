# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only tests of the sorted-dispatch row bound (tt/moe_dispatch.py dispatch_rows, stage 12 tight bound): the
row layout of ttnn.experimental.moe_sorted_dispatch (experts ascending, each padded to whole groups, then the trailing
padding group) must fit the bound for every routing, and the bound must equal the E-based one wherever T*k >= E."""

import importlib

import pytest
import torch

from models.autoports.zai_org_glm_5_3_flash.tt import moe_dispatch as M

E, K, GS = 192, 8, 32  # E192 routed experts, top-8, 32-row groups


def _layout_rows(ids: torch.Tensor, gs: int) -> int:
    """Rows the kernel layout needs: every used expert's rows padded to whole groups + the trailing padding group."""
    counts = torch.bincount(ids.flatten(), minlength=E)
    return int(((counts + gs - 1) // gs * gs).sum()) + gs


@pytest.fixture
def tight(monkeypatch):
    monkeypatch.setenv("GLM53_MOE_TIGHT_DISPATCH", "1")
    yield importlib.reload(M)
    monkeypatch.delenv("GLM53_MOE_TIGHT_DISPATCH")
    importlib.reload(M)


@pytest.mark.parametrize("T", [1, 2, 3, 4, 8, 16, 23, 24, 32])
def test_tight_bound_fits_every_routing(tight, T):
    """Worst cases (k distinct experts per row, all rows different -> min(E, T*k) used experts with one row each) and
    random routings all fit."""
    Rd = tight.dispatch_rows(T, K, E, GS)
    assert Rd % GS == 0
    spread = torch.arange(T * K).reshape(T, K) % E  # as many distinct experts as possible
    assert _layout_rows(spread, GS) <= Rd
    gen = torch.Generator().manual_seed(T)
    for _ in range(200):
        ids = torch.stack([torch.randperm(E, generator=gen)[:K] for _ in range(T)])
        assert _layout_rows(ids, GS) <= Rd


@pytest.mark.parametrize("T", [24, 32, 512, 2048])
def test_tight_bound_unchanged_at_full_width(tight, monkeypatch, T):
    """T*k >= E (the 32-slot S bucket, every prefill chunk): the same bound as before stage 12."""
    Rd = tight.dispatch_rows(T, K, E, GS)
    monkeypatch.setenv("GLM53_MOE_TIGHT_DISPATCH", "0")
    loose = importlib.reload(M)
    assert loose.dispatch_rows(T, K, E, GS) == Rd


def test_tight_bound_small_buckets(tight):
    assert [tight.dispatch_rows(t, K, E, GS) for t in (1, 2, 4, 8, 16)] == [320, 576, 1056, 2080, 4128]
