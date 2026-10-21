# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only tests of the stage-10 decode buckets (tt/dsa_buckets.py) and the prefill pool-key views (tt/dsa.py):
bucket rules, the sharded-layout gate, the MoE compaction index math, the view geometry and release()."""

import types

import pytest

import ttnn
from models.autoports.zai_org_glm_5_3_flash.tt import dsa as D
from models.autoports.zai_org_glm_5_3_flash.tt import dsa_buckets as B


def test_bucket_contexts_rules(monkeypatch):
    monkeypatch.delenv("GLM53_DSA_BUCKETS", raising=False)
    monkeypatch.delenv("GLM53_DSA_BUCKET_MIN_T", raising=False)
    assert B.bucket_contexts(16384) is None  # below the 65,536 default threshold: single trace
    ctx = B.bucket_contexts(280576)
    assert ctx[0] == 4096 and ctx[-1] == 280576 and ctx == sorted(ctx)
    assert B.bucket_contexts(280576, sharded=True) is None  # block-sharded DSA: no buckets
    monkeypatch.setenv("GLM53_DSA_BUCKETS", "0")
    assert B.bucket_contexts(280576) is None


def test_install_gates_sharded_layers(monkeypatch):
    """install(): a sharded DSA model takes no DSA decode buckets (MoE / views decide the rest)."""
    monkeypatch.delenv("GLM53_DSA_BUCKETS", raising=False)
    layer = types.SimpleNamespace(T=1 << 20, dsa_shard=True)
    dec = types.SimpleNamespace(attn_kind="dsa", attn=layer, mlp_kind="dense", mlp=None)
    gen = types.SimpleNamespace(model=types.SimpleNamespace(layers=[dec]), max_batch_size=1, mesh=None)
    layer.__dict__  # SimpleNamespace: _inner() walks __dict__
    assert B.install(gen) is None
    assert gen._dsa_buckets is None


def test_moe_row_buckets_and_paths(monkeypatch):
    monkeypatch.delenv("GLM53_MOE_BUCKETS", raising=False)
    monkeypatch.delenv("GLM53_MOE_BUCKET_PATH", raising=False)
    assert B.moe_row_buckets(1) is None
    assert B.moe_row_buckets(16) == [1, 2, 4, 8, 16]
    assert B.moe_row_buckets(32) == [1, 2, 4, 8, 16, 32]
    assert B.moe_bucket_paths([1, 2, 4, 8]) == {1: "pairs", 2: "pairs", 4: "pairs", 8: "pairs"}
    monkeypatch.setenv("GLM53_MOE_BUCKET_PATH", "8:sorted")
    assert B.moe_bucket_paths([1, 2, 4, 8])[8] == "sorted"
    monkeypatch.setenv("GLM53_MOE_BUCKETS", "0")
    assert B.moe_row_buckets(16) is None


def _ctl(S=16, T=1000):
    c = B.DsaBuckets.__new__(B.DsaBuckets)
    c.tbs, c.S, c.max_context, c.widths = B.moe_row_buckets(S), S, T, None
    c.moe_layers, c._active, c.mesh = [object()], None, None
    c.gidx = c.sidx = c.cpad = "buf"
    return c


def test_moe_bucket_and_compaction_index(monkeypatch):
    c = _ctl()
    assert c.moe_bucket([-1] * 16) == 0
    assert c.moe_bucket([5] * 3 + [-1] * 13) == 2  # 3 active -> 4 rows
    assert c.moe_bucket([5] * 9 + [-1] * 7) == 4  # 9 active -> the S bucket
    assert c.moe_bucket([1000] + [-1] * 15) == 0  # a slot at max_seq_len counts as inactive
    writes = []
    monkeypatch.setattr(ttnn, "from_torch", lambda t, **k: t)
    monkeypatch.setattr(ttnn, "ReplicateTensorToMesh", lambda m: None)
    monkeypatch.setattr(ttnn, "copy_host_to_device_tensor", lambda h, d: writes.append(h.flatten().tolist()))
    pos = [-1] * 16
    pos[7], pos[3] = 10, 20
    c._set_active(pos)
    g, sidx, pad = writes
    assert g[:2] == [3, 7] and all(x == 3 for x in g[2:])  # active slots first, padding repeats the first
    assert sidx[3] == 0 and sidx[7] == 1 and all(sidx[s] == 15 for s in range(16) if s not in (3, 7))
    assert pad[:2] == [0.0, 0.0] and all(x == 1.0 for x in pad[2:])
    c._set_active(pos)  # unchanged active set: no copy
    assert len(writes) == 3


@pytest.mark.parametrize("T,NB", [(280576, 2193), (243712, 1933), (325632, 2546), (151552, 1229), (16384, 129)])
def test_view_geometry(T, NB):
    MB = T // 128 + 1
    PM = D._smooth64(max(NB, MB))
    m = PM // 64
    for p in (2, 3, 5):
        while m % p == 0:
            m //= p
    assert PM % 64 == 0 and m == 1 and PM >= max(NB, MB)
    ds = D._view_blocks(PM, MB, T)
    assert ds == sorted(ds) and all(PM % d == 0 for d in ds)
    assert 32 * (ds[-1] - 1) >= 32 * (MB - 1)  # the largest view covers every pool of the context
    assert 32 * (ds[0] - 1) >= 4096 // 4 or T <= 4096


def test_release_drops_owned_tensors(monkeypatch):
    freed = []
    monkeypatch.setattr(ttnn, "deallocate", lambda t, *a, **k: freed.append(t))
    monkeypatch.setattr(ttnn, "release_trace", lambda *a, **k: None)
    layer = types.SimpleNamespace(_sel_buf="sel", score_widths=[1056], P_alloc=70176, _const_cache={("iota", 1056): 1})
    moe = types.SimpleNamespace(_bucket_bufs={"out": "moeout"})
    c = B.DsaBuckets.__new__(B.DsaBuckets)
    c.mesh, c.main, c.points, c.widths, c.view_ds = object(), [1], [("dsa", [2])], [1056], []
    c.layers, c.moe_layers = [layer], [moe]
    c.gidx, c.sidx, c.cpad = "g", "s", "p"
    D._SHARED_IOTA[(id(c.mesh), 1056)] = (c.mesh, "iota")
    names = set(c.device_tensors())
    assert {"dsa0.sel_buf", "iota1056", "moe.gidx", "moe0.out"} <= names
    c.release()
    assert {"sel", "moeout", "g", "s", "p"} <= set(freed)
    assert not c.layers and not c.moe_layers and ("iota", 1056) not in layer._const_cache
    assert (id(c.mesh), 1056) not in D._SHARED_IOTA


def test_fit_row_buckets_to_trace_region():
    """Each row bucket below S costs one point trace per MoE layer: 64 MiB holds the 1-row bucket, 96 MiB all of them."""
    assert B.fit_row_buckets([1, 2, 4, 8, 16], 42, 64 << 20) == [1, 16]
    assert B.fit_row_buckets([1, 2, 4, 8, 16], 42, 96 << 20) == [1, 2, 4, 8, 16]
    assert B.fit_row_buckets([1, 2, 4, 8, 16], 42, 40 << 20) is None
