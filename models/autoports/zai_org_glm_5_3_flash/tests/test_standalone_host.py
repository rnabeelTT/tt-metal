# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only tests of the standalone generator's sizing rule (tt/generator.plan_context) and the weight cache's
manifest handling (tt/weight_cache.weight_cache_from_env). No device.

pytest models/autoports/zai_org_glm_5_3_flash/tests/test_standalone_host.py
"""

import json
import os
import time
import types

import pytest

from models.autoports.zai_org_glm_5_3_flash.tt import generator as G
from models.autoports.zai_org_glm_5_3_flash.tt import serving_profiles as sp
from models.autoports.zai_org_glm_5_3_flash.tt import weight_cache as W


@pytest.fixture
def default_env(monkeypatch):
    for k in ("GLM53_PRECISION", "GLM53_DRAM_RESERVE_MIB", "GLM53_DSA_SHARD", "GLM53_BUDGET_FREE_BYTES"):
        monkeypatch.delenv(k, raising=False)
    G.budget()
    if G.BUDGET["source"].startswith("built-in"):
        pytest.skip("serving_profiles.json does not match the default precision")
    yield
    G.budget()


def _replicated_default(data):
    """The documented rule: replicated, reserve max(serving workspace, 1,200 MiB), 1 slot owns T / 128 blocks + 1."""
    b = sp.layout_budget(data, "replicated")
    reserve = max(int(b["reserve_bytes"]), 1200 << 20)
    blocks = (int(b["free_after_load_bytes"]) - reserve) // G.BUDGET["block_bytes"]["replicated"]
    t = (blocks - 1) * 128
    t = min(t, sp.max_model_len_for(data, 1, layout="replicated"))
    return t // 2048 * 2048


def test_default_plan_is_replicated(default_env):
    p = G.plan_context(1)
    assert p["layout"] == "replicated"
    assert p["max_seq_len"] == _replicated_default(sp.load())


def test_advertised_context_plans_sharded(default_env):
    p = G.plan_context(1, 1048576)
    assert p["layout"] == "sharded" and p["dsa_shard"]
    # the sharded layout uses its own measured budget (600 MiB workspace), not the 1,200 MiB replicated floor
    assert G.BUDGET["layouts"]["sharded"]["reserve"] == int(sp.layout_budget(sp.load(), "sharded")["reserve_bytes"])


def test_low_reserve_is_clamped_to_measured_replicated_max(default_env, monkeypatch, expect_error):
    monkeypatch.setenv("GLM53_DRAM_RESERVE_MIB", "600")
    G.budget()
    cap = sp.max_model_len_for(sp.load(), 1, layout="replicated")
    assert G.max_context(1, "replicated") <= cap
    assert G.plan_context(1, 395264)["layout"] == "sharded"  # above the measured replicated maximum
    with expect_error(ValueError, "Recipe"):
        G.plan_context(1, 395264, layout="replicated")


# ------------------------------------------------------------------------------------------------- weight cache
@pytest.fixture
def ckpt(tmp_path, monkeypatch):
    monkeypatch.setattr(W, "WeightCache", lambda root, mesh: root)
    monkeypatch.setattr(W, "policy_weight_dtypes", lambda opt: {"x": 1})
    monkeypatch.setattr(W, "checkpoint_digest", lambda rd: "ck")
    d = tmp_path / "ck"
    d.mkdir()
    (d / "a.safetensors").write_text("x")
    (d / "config.json").write_text("{}")
    monkeypatch.setenv("GLM53_CACHE_DIR", str(tmp_path / "cache"))
    rd = types.SimpleNamespace(path=d, _files={"t": "a.safetensors"})
    return rd, types.SimpleNamespace(name="p"), types.SimpleNamespace(shape=(1, 4))


def _legacy(root, created):
    m = json.loads((root / "manifest.json").read_text())
    m.pop("checkpoint_files", None)
    m["created"] = created
    (root / "manifest.json").write_text(json.dumps(m))


def _ts(dt):
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() + dt))


def test_legacy_manifest_adopted_when_shards_older(ckpt):
    rd, opt, mesh = ckpt
    root = W.weight_cache_from_env(mesh, rd, opt)
    _legacy(root, _ts(60))
    assert W.weight_cache_from_env(mesh, rd, opt) == root
    assert "checkpoint_files" in json.loads((root / "manifest.json").read_text())


def test_legacy_manifest_refused_when_a_shard_is_newer(ckpt):
    rd, opt, mesh = ckpt
    root = W.weight_cache_from_env(mesh, rd, opt)
    _legacy(root, "2000-01-01T00:00:00Z")
    fresh = W.weight_cache_from_env(mesh, rd, opt)
    assert fresh != root and fresh.name.startswith(root.name + "-f")


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores directory permissions")
def test_read_only_root(ckpt, tmp_path):
    rd, opt, mesh = ckpt
    root = W.weight_cache_from_env(mesh, rd, opt)
    base = tmp_path / "cache"
    try:
        _legacy(root, _ts(60))
        os.chmod(root, 0o555)
        os.chmod(base, 0o555)
        assert W.weight_cache_from_env(mesh, rd, opt) == root  # verified legacy cache: reused read-only
        os.chmod(root, 0o755)
        _legacy(root, "2000-01-01T00:00:00Z")
        os.chmod(root, 0o555)
        assert W.weight_cache_from_env(mesh, rd, opt) is None  # stale and no fresh directory possible: uncached
    finally:
        os.chmod(base, 0o755)
        os.chmod(root, 0o755)


def test_device_oom_is_not_an_unreadable_cache(tmp_path, monkeypatch, expect_error):
    c = W.WeightCache.__new__(W.WeightCache)
    monkeypatch.setattr(c, "has", lambda name: True, raising=False)
    monkeypatch.setattr(c, "file", lambda name: tmp_path / name, raising=False)

    def oom(name):
        raise RuntimeError("TT_FATAL: Out of Memory: Not enough space to allocate 226492416 B DRAM buffer")

    monkeypatch.setattr(c, "load", oom, raising=False)
    with expect_error(RuntimeError, "Out of Memory"):
        c.get("x", lambda: pytest.fail("must not rebuild on a device OOM"))

    def corrupt(name):
        raise RuntimeError("short read at 0 of 8 bytes")

    monkeypatch.setattr(c, "load", corrupt, raising=False)
    monkeypatch.setattr(c, "store", lambda name, t: t, raising=False)
    assert c.get("x", lambda: "rebuilt") == "rebuilt"
