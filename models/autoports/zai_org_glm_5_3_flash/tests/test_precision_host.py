# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only checks of the stage-8 precision policy plumbing (tt/precision.py); no device.

- GLM53_PRECISION unset loads doc/datatype_sweep/selected_precision_config.json (the model default);
  "none" / "shipped" / "off" / "0" give the stage-7 policy (no override).
- The default and the explicit GLM53_PRECISION=<selected file> resolve the same policy AND the same weight-cache
  scopes / file names of the re-quantized tensors (so a cache written by one path is hit by the other), with the real
  tt/weight_cache.WeightCache path layout.
- The re-quantized tensor's cache name changes with the source tensor's memory config, padded shape, source dtype,
  target dtype and the owning component digest.
- A layer with a weight-dtype exception caches no re-quantized tensor.
- With the block-sharded DSA cache the fp8 latent request falls back to bf16 (no latent_dtype set).
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import ttnn
from models.autoports.zai_org_glm_5_3_flash.tt import precision as P
from models.autoports.zai_org_glm_5_3_flash.tt.weight_cache import WeightCache

FIELDS = ("id", "requant", "fidelity", "fidelity_layers", "moe_layer_exceptions", "ccl_h_dtype", "kv_latent_dtype")


class _Src:
    """Device-tensor stand-in with the attributes rq_name reads."""

    def __init__(self, mc="DRAM", shape=(1, 1, 4096, 8256), dtype=ttnn.bfloat8_b):
        self._mc, self.padded_shape, self.dtype = mc, shape, dtype

    def memory_config(self):
        return self._mc


class _RecCache(WeightCache):
    """The real WeightCache path layout (root / scope / name.tensorbin) with has / store recorded, no files."""

    def __init__(self, root, log, scope="top"):
        super().__init__(root, None, scope)
        self.log = log

    def sub(self, scope):
        return _RecCache(self.root, self.log, scope)

    def has(self, *names):
        return False

    def store(self, name, t):
        self.log.append(str(self.file(name).relative_to(self.root)))
        return t


def _load(monkeypatch, value):
    P._INSTANCES.clear()
    if value is None:
        monkeypatch.delenv(P.ENV, raising=False)
    else:
        monkeypatch.setenv(P.ENV, value)
    return P.PrecisionOverrides.from_env()


def _cache_files(o, monkeypatch, root):
    """Cache files of every re-quantized tensor of the policy for layers 3 (DSA), 4 (KDA) and the LM head."""
    log = []
    monkeypatch.setattr(P, "requant", lambda t, dt: t)
    kinds = {"kda_in": "kda", "kda_out": "kda", "dsa_proj": "dsa", "shared_expert": "moe", "dense_mlp": "mlp"}
    for scope, i in (("top", None), ("layer_4", 4), ("layer_3", 3)):
        o._scope(_RecCache(root, log), scope)
        for g in o.requant:
            if (g == "lm_head") != (scope == "top"):
                continue
            dt = o.requant_dtype(g, 0 if i is None else i, 0, 44)
            if dt is not None:
                o._rq(_Src(), dt, g, "top" if g == "lm_head" else kinds[g])
    return log


def test_default_is_selected_config(monkeypatch):
    sel = json.loads(P.SELECTED_CONFIG.read_text())
    o = _load(monkeypatch, None)
    assert o.source == str(P.SELECTED_CONFIG)
    for f in FIELDS:
        if f in sel:
            assert getattr(o, f) == sel[f], f
    assert not o.is_shipped


@pytest.mark.parametrize("value", ["none", "shipped", "NONE", "off", "0"])
def test_none_is_stage7_policy(monkeypatch, value):
    o = _load(monkeypatch, value)
    assert o.is_shipped and o.id == "shipped"


def test_default_and_explicit_share_cache_files(monkeypatch, tmp_path):
    d = _load(monkeypatch, None)
    files_d = _cache_files(d, monkeypatch, tmp_path)
    e = _load(monkeypatch, str(P.SELECTED_CONFIG))
    files_e = _cache_files(e, monkeypatch, tmp_path)
    assert [getattr(d, f) for f in FIELDS] == [getattr(e, f) for f in FIELDS]
    assert files_d and files_d == files_e
    v = P._rq_version()
    for f in files_d:
        parts = f.split("/")
        assert parts[-2] == f"prec-{v}" and parts[0] in ("top", "layer_3", "layer_4") and f.endswith(".tensorbin"), f


def test_cache_name_tracks_source_layout():
    base = P.rq_name("kda_in", _Src(), ttnn.bfloat4_b, "kda-abc")
    assert base != P.rq_name("kda_in", _Src(mc="L1_WIDTH_SHARDED"), ttnn.bfloat4_b, "kda-abc")
    assert base != P.rq_name("kda_in", _Src(shape=(1, 1, 4096, 8288)), ttnn.bfloat4_b, "kda-abc")
    assert base != P.rq_name("kda_in", _Src(dtype=ttnn.bfloat16), ttnn.bfloat4_b, "kda-abc")
    assert base != P.rq_name("kda_in", _Src(), ttnn.bfloat8_b, "kda-abc")
    assert base != P.rq_name("kda_in", _Src(), ttnn.bfloat4_b, "kda-def")
    assert base == P.rq_name("kda_in", _Src(), ttnn.bfloat4_b, "kda-abc")


def test_exception_layer_not_cached(monkeypatch, tmp_path):
    o = P.PrecisionOverrides(id="x", moe_layer_exceptions={"44": {"routed_down_dtype": "bfp8"}})
    assert o.layer_cache(object(), 44) is None and o.layer_cache("c", 4) == "c"
    log = []
    dec = SimpleNamespace(attn_kind="kda", mlp_kind="none", mesh=None, attn=SimpleNamespace(weights=None), mlp=None)
    monkeypatch.setattr(o, "_apply_kda", lambda *a: None)
    o.apply_layer(dec, 44, 0, 44, cache=_RecCache(tmp_path, log))
    assert o._cache is None
    o.apply_layer(dec, 4, 0, 44, cache=_RecCache(tmp_path, log))
    assert o._cache is not None


def test_dsa_shard_takes_fp8_latent():
    """The block-sharded cache (tt/dsa_shard.py) takes the fp8 latent too (576-wide fp8 rows), as the replicated one."""
    o = P.PrecisionOverrides(id="kv", kv_latent_dtype="fp8_e4m3")
    attn = SimpleNamespace(dsa_shard=True, latent=None)
    o._apply_dsa(attn, None, 3, 0, 44)
    assert attn.latent_dtype == ttnn.fp8_e4m3
    assert o.applied[-1] == (3, "kv_latent", "fp8_e4m3 (dsa_shard)")
    attn2 = SimpleNamespace(latent=None)
    o._apply_dsa(attn2, None, 7, 0, 44)
    assert attn2.latent_dtype == ttnn.fp8_e4m3
    assert o.applied[-1] == (7, "kv_latent", "fp8_e4m3")
