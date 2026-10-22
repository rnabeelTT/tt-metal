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


@pytest.mark.parametrize("S,T", [(2, 65536), (16, 239616), (16, 243712), (32, 137216), (32, 151552), (8, 131072)])
def test_multi_slot_takes_no_auto_dsa_buckets(monkeypatch, S, T):
    """S >= 2: no auto DSA decode buckets, so a request's selection does not depend on the other slots' positions
    (review #1 P2-2: t20 mixed modes, t23 served). 1 slot keeps the 4,096 start; an explicit list still applies."""
    monkeypatch.delenv("GLM53_DSA_BUCKETS", raising=False)
    monkeypatch.delenv("GLM53_DSA_BUCKET_MIN_T", raising=False)
    assert B.bucket_contexts(T, slots=S) is None
    assert B.bucket_contexts(T, slots=1)[0] == 4096
    monkeypatch.setenv("GLM53_DSA_BUCKETS", "65536")
    assert B.bucket_contexts(T, slots=S) == ([65536, T] if T > 65536 else [T])


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
    assert B.moe_bucket_paths([1, 2, 4, 8, 16]) == {t: "pairs" for t in (1, 2, 4, 8, 16)}
    monkeypatch.setenv("GLM53_MOE_BUCKET_PATH", "8:sorted")
    assert B.moe_bucket_paths([1, 2, 4, 8, 16])[8] == "sorted"


class _Moe:
    def _decode_sorted(self, T):  # Qwen35MoEOptimized default: expert-sorted from 32 decode rows
        return T >= 32


@pytest.mark.parametrize("S", [2, 4, 8, 16, 32])
def test_all_buckets_share_the_s_bucket_family(monkeypatch, S):
    """Batch invariance rule: every row bucket, the S bucket included, runs the routed-path family of the S bucket
    (stage-10 wide regression: pairs below S and sorted at S = 32 gave 4/32 burst identity)."""
    monkeypatch.delenv("GLM53_MOE_BUCKET_PATH", raising=False)
    tbs = B.moe_row_buckets(S)
    fam = B.moe_family(_Moe(), S)
    paths = B.moe_bucket_paths(tbs, fam)
    assert set(paths) == set(tbs) and set(paths.values()) == {fam}
    assert fam == ("sorted" if S >= 32 else "pairs")
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


@pytest.mark.parametrize(
    "S,region_mib,with_dsa,n_layers,want",
    [
        # measured configurations: the same lists as the 0a2663d0b6b rule
        (16, 96, True, 42, [1, 2, 4, 8, 16]),  # p300x2 / p150x4 (243,712: DSA buckets on), stage-9 G9 16-slot capture
        (16, 96, False, 42, [1, 2, 4, 8, 16]),  # t15b pairs96
        (16, 64, False, 42, [1, 16]),  # t15b pairs (64 MiB)
        (16, 64, True, 42, [1, 16]),
        (8, 64, False, 42, [1, 8]),  # standalone 8 slots
        (8, 64, True, 42, [1, 8]),
        (4, 64, False, 2, [1, 2, 4]),  # reduced adapter validator (t16)
        (16, 64, False, 2, [1, 2, 4, 8, 16]),  # reduced 16-slot runs (t13)
        # 32 slots: fit by the estimate (0a2663d0b6b picked 1/2/4/8 at 96 MiB: G9 OOM; 1 at 64 MiB: estimated 66.1 MiB)
        (32, 96, True, 42, [1, 2, 4, 32]),
        (32, 96, False, 42, [1, 2, 4, 32]),
        (32, 64, False, 42, None),
        (32, 64, True, 42, None),
    ],
)
def test_fit_row_buckets_to_trace_region(S, region_mib, with_dsa, n_layers, want):
    tbs = B.moe_row_buckets(S)
    got = B.fit_row_buckets(tbs, n_layers, region_mib << 20, with_dsa=with_dsa)
    assert got == want
    k = len(got) - 1 if got else 0
    assert B.decode_trace_estimate(S, n_layers, k, with_dsa) <= region_mib << 20 or k == 0


def test_trace_estimate_matches_measured_captures():
    MiB = 1 << 20
    # 16 slots, k = 4 + DSA points: measured 93.2 MiB with the sampler (stage-9 G9) -> the estimate covers it
    assert B.decode_trace_estimate(16, 42, 4, True) >= 93.2 * MiB
    # 32 slots, DSA, k = 4: the G9 capture ran out of a 96 MiB region -> the estimate must exceed 96 MiB
    assert B.decode_trace_estimate(32, 42, 4, True) > 96 * MiB
