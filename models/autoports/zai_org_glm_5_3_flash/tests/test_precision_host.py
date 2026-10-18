# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only checks of the stage-8 precision policy plumbing (tt/precision.py); no device.

- GLM53_PRECISION unset loads doc/datatype_sweep/selected_precision_config.json (the model default);
  "none" / "shipped" give the stage-7 policy (no override).
- The default and the explicit GLM53_PRECISION=<selected file> resolve the same policy AND the same weight-cache
  scopes / file names of the re-quantized tensors (so a cache written by one path is hit by the other).
"""

from __future__ import annotations

import json

import pytest

from models.autoports.zai_org_glm_5_3_flash.tt import precision as P

FIELDS = ("id", "requant", "fidelity", "fidelity_layers", "moe_layer_exceptions", "ccl_h_dtype", "kv_latent_dtype")


class _FakeCache:
    """WeightCache stand-in: records the scopes and names the policy asks for (has() = False: always a miss)."""

    def __init__(self, log, scope="top"):
        self.log, self.scope = log, scope

    def sub(self, scope):
        return _FakeCache(self.log, scope)

    def has(self, *names):
        self.log.append(("has", self.scope, names))
        return False

    def file(self, name):
        return f"{self.scope}/{name}.tensorbin"

    def store(self, name, t):
        self.log.append(("store", self.scope, name))
        return t


def _load(monkeypatch, value):
    P._INSTANCES.clear()
    if value is None:
        monkeypatch.delenv(P.ENV, raising=False)
    else:
        monkeypatch.setenv(P.ENV, value)
    return P.PrecisionOverrides.from_env()


def _cache_names(o, monkeypatch):
    """Scopes / names of every re-quantized tensor of the policy (requant stubbed: identity)."""
    log = []
    monkeypatch.setattr(P, "requant", lambda t, dt: t)
    for scope, i in (("top", -1), ("layer_4", 4), ("layer_3", 3)):
        o._scope(_FakeCache(log), scope)
        for g in o.requant:
            dt = o.requant_dtype(g, i if i >= 0 else 0, 0, 44)
            if dt is not None and (g == "lm_head") == (scope == "top"):
                o._rq(object(), dt, g)
    return log


def test_default_is_selected_config(monkeypatch):
    sel = json.loads(P.SELECTED_CONFIG.read_text())
    o = _load(monkeypatch, None)
    assert o.source == str(P.SELECTED_CONFIG)
    for f in FIELDS:
        if f in sel:
            assert getattr(o, f) == sel[f], f
    assert not o.is_shipped


@pytest.mark.parametrize("value", ["none", "shipped", "NONE"])
def test_none_is_stage7_policy(monkeypatch, value):
    o = _load(monkeypatch, value)
    assert o.is_shipped and o.id == "shipped"


def test_default_and_explicit_share_cache_keys(monkeypatch):
    d = _load(monkeypatch, None)
    log_d = _cache_names(d, monkeypatch)
    e = _load(monkeypatch, str(P.SELECTED_CONFIG))
    log_e = _cache_names(e, monkeypatch)
    assert [getattr(d, f) for f in FIELDS] == [getattr(e, f) for f in FIELDS]
    assert log_d and log_d == log_e
    assert all(s.split("/")[-1] == f"prec-{P._rq_version()}" for _, s, _ in log_d)
