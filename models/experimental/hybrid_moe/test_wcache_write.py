# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Host test of hybrid/wcache.py's image write (no device): a write() that returns a short count (a filling disk does
this rather than raise) is continued until every byte is written, and one that makes no progress raises instead of
leaving a short file to be renamed into the cache.
usage: python -m pytest test_wcache_write.py"""

import importlib.util
import os
import sys
import types

import pytest


def _wcache():
    if "ttnn" not in sys.modules:
        try:
            import ttnn  # noqa: F401
        except ImportError:
            sys.modules["ttnn"] = types.SimpleNamespace(bfloat4_b=object(), bfloat8_b=object())
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hybrid", "wcache.py")
    spec = importlib.util.spec_from_file_location("wcache_under_test", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class ShortFile:
    """A raw file whose write() takes at most `step` bytes, and none once `cap` bytes are in."""

    def __init__(self, path, step, cap):
        self.f, self.step, self.cap, self.n = open(path, "wb", buffering=0), step, cap, 0

    def write(self, b):
        k = min(len(b), self.step, self.cap - self.n)
        if k <= 0:
            return 0
        self.n += self.f.write(bytes(b[:k]))
        return k

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.f.close()


def test_short_writes_are_continued(tmp_path, monkeypatch):
    W = _wcache()
    data = bytes(range(256)) * 41
    monkeypatch.setattr(W, "open", lambda p, *a, **k: ShortFile(p, 1000, 1 << 30), raising=False)
    W._write_all(str(tmp_path / "x"), memoryview(data))
    assert (tmp_path / "x").read_bytes() == data


def test_stalled_write_raises(tmp_path, monkeypatch):
    W = _wcache()
    monkeypatch.setattr(W, "open", lambda p, *a, **k: ShortFile(p, 1000, 4096), raising=False)
    with pytest.raises(OSError):
        W._write_all(str(tmp_path / "x"), memoryview(bytes(10000)))


def test_hybrid_image_cut_from_full_image(tmp_path):
    """A demoted layer's missing hybrid image is cut from the full image on the host: per device shard, slot s holds
    expert ids[s]'s bytes (zero bytes for an empty slot), as wcache.gather builds it on the device."""
    W = _wcache()
    tp, E_full, c = 4, 8, 96  # 4 shards of 8 experts, 96 bytes per expert per shard
    ids = [5, 0, -1, 7, -1]
    plan = {3: list(range(E_full))}
    wc = W.ExpertImageCache(str(tmp_path), resident_fn=lambda L: plan[L], tag="t")
    wc.px = types.SimpleNamespace(tp=tp)
    full = dict(
        E=E_full,
        H=64,
        Nlp=32,
        tp=tp,
        gate_up_dtype="DataType.BFLOAT4_B",
        down_dtype="DataType.BFLOAT4_B",
        sfx=".qx",
        gate_up_tile="None",
    )
    imgs = {}
    for path, k in zip(wc._paths(3, wc._key(3, full)), (1, 2)):
        imgs[path] = bytes((k * 100 + d * 10 + e + i) % 251 for d in range(tp) for e in range(E_full) for i in range(c))
        (tmp_path / os.path.basename(path)).write_bytes(imgs[path])
    plan[3] = ids
    hyb = dict(full, E=len(ids))
    assert not wc._slice_full_image(None, 3, hyb, dict(full, sfx=".other"))  # another rounding: not a source
    assert wc._slice_full_image(None, 3, hyb, full)
    for src, dst in zip(wc._paths(3, wc._key(3, full, ids=list(range(E_full)))), wc._paths(3, wc._key(3, hyb))):
        data, got = imgs[src], open(dst, "rb").read()
        n = E_full * c
        want = b"".join(
            bytes(c) if e < 0 else data[d * n + e * c : d * n + (e + 1) * c] for d in range(tp) for e in ids
        )
        assert got == want
