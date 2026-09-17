# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Where a multi-die mesh's parallelism decision is made, and that it is made once.

No device: this is the policy, not the kernels. It exists because the decision used to be
taken in two places from the same environment variable -- `model_config` for tensor
parallelism and `moe.py` for expert parallelism -- which let the layer config and the
expert bank disagree about whether the mesh was sharded.
"""
from __future__ import annotations

import pytest

from models.autoports.ornith_ai_ornith_1_5_35b_a3b.tt.model_config import MESH_PRECISION, multidie_parallel_enabled


@pytest.fixture
def clean_env(monkeypatch):
    for var in ("ORNITH_TP", "ORNITH_MOE_EP"):
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


def test_multi_die_replicates_by_default(clean_env):
    """Replication is the default on EVERY mesh, and that is deliberate.

    Tensor parallelism is a 2.0-3.1x prefill win on four dies and the only route to
    on-device sampling, but it does not SERVE -- a 4-die TP instance dies in the vLLM
    warmup with a fixed circular-buffer overflow (1684544 B against 1572864 B of L1,
    identical at max_model_len 4096 and 2048) -- and its single-stream decode is worse
    than one die's (8.79 vs 18.40 tok/s eager, collective-bound). So the shipped
    multi-die deployment is lanes, and every tt-model.yaml profile serves replicated.

    This test is the guard: flipping the default without fixing the warmup overflow
    would ship profiles that cannot start.
    """
    for dies in (1, 2, 3, 4, 8):
        assert multidie_parallel_enabled(dies) is False, dies


def test_opt_in_enables_two_or_more_dies(clean_env):
    """ORNITH_TP=1 is how every TP measurement was taken and how the work continues."""
    clean_env.setenv("ORNITH_TP", "1")
    assert multidie_parallel_enabled(4) is True
    # Two dies are reachable on purpose: they hang in the expert-parallel all-reduce
    # (moe.py::_reduce, caught twice with py-spy on a freshly reset device), and a gate
    # that refused them would make that defect unreproducible.
    assert multidie_parallel_enabled(2) is True
    # Never one die: no peer, and DistributedNorm would reach for a collective with no
    # fabric ("Trying to get un-initialized fabric context" from control_plane.cpp).
    assert multidie_parallel_enabled(1) is False


def test_explicit_off_is_honoured(clean_env):
    clean_env.setenv("ORNITH_TP", "0")
    assert multidie_parallel_enabled(4) is False


def test_mesh_precision_covers_every_shipped_profile():
    """tt-model.yaml's own verify block asserts exactly this set, so keep them in step."""
    assert set(MESH_PRECISION) == {"P150", "P300", "P150x4"}
