# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Fast reads of large files on a host whose page cache is pathological.

On the QB2 host a buffered (page-cache) read of the NVMe runs at ~210 MB/s (kswapd reclaims file pages as fast as they
are added; Cached stays < 1 GB with ~100 GB free) while an O_DIRECT read of the same file runs at 5 GB/s. Every
tensor-cache file ttnn loads is mmap'ed (page cache), so the ~99 GB device-model build took 430-507 s at ~11 s per layer.

- `read_direct(path, buf, off0, n)`: O_DIRECT reads of [off0, off0 + n) of `path` into the page-aligned buffer `buf`,
  by `threads` parallel readers in `chunk` pieces (no page cache, one copy).
- `install()` (HY4_FAST_LOAD=1, default): ttnn's tensor-cache load (`load_tensor_flatbuffer`, used by ttnn.as_tensor)
  stages each file >= HY4_FAST_LOAD_MIN_MB (16) into a /dev/shm file with O_DIRECT reads, loads it from there (the same
  bytes: identical tensors) and removes it. Peak extra host memory: one file (<= 1.4 GB). Any failure falls back to the
  plain load of the original path.
"""

from __future__ import annotations

import mmap
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

ALIGN = 4096
_pool = None
_lock = threading.Lock()
STATS = {"files": 0, "bytes": 0, "read_s": 0.0, "load_s": 0.0, "fallbacks": 0}


def _threads():
    return int(os.environ.get("HY4_FAST_LOAD_THREADS", "8"))


def _shared():
    global _pool
    with _lock:
        if _pool is None:
            _pool = ThreadPoolExecutor(max_workers=_threads(), thread_name_prefix="hy4-direct")
    return _pool


def read_direct(path, buf, off0=0, n=None, chunk=64 << 20, wait=None, pool=None):
    """O_DIRECT read of bytes [off0, off0 + n) of `path` into memoryview/buffer `buf` (page-aligned address, len >= n
    rounded up to 4096; off0 a multiple of 4096). wait: optional callable run before each chunk (throttle); pool:
    the executor of the readers (default the shared one, HY4_FAST_LOAD_THREADS).
    Returns the bytes read (n, or up to the file end)."""
    size = os.path.getsize(path)
    n = size - off0 if n is None else min(n, size - off0)
    assert off0 % ALIGN == 0
    mv = memoryview(buf).cast("B")
    fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
    try:
        def one(a):
            if wait is not None:
                wait()
            want = min(chunk, -(-(n - a) // ALIGN) * ALIGN)  # the last piece rounded up to the block size
            got = 0
            while got < want:
                k = os.preadv(fd, [mv[a + got : a + want]], off0 + a + got)
                if k <= 0:
                    break  # end of file (the rounded-up tail)
                got += k
            if a + got < min(a + chunk, n):
                raise IOError(f"short O_DIRECT read of {path} at {off0 + a + got}")

        list((pool or _shared()).map(one, range(0, n, chunk)))
    finally:
        os.close(fd)
    return n


def stage(path, shm_dir="/dev/shm"):
    """Copy `path` into a fresh tmpfs file with O_DIRECT reads (no page cache of the NVMe); returns its path."""
    size = os.path.getsize(path)
    rsize = -(-size // ALIGN) * ALIGN
    dst = os.path.join(shm_dir, f"hy4fl-{os.getpid()}-{threading.get_ident()}-{os.path.basename(path)}")
    fd = os.open(dst, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.ftruncate(fd, rsize)
        mm = mmap.mmap(fd, rsize, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
        try:
            read_direct(path, mm, 0, size)
        finally:
            mm.close()
        os.ftruncate(fd, size)
    except BaseException:
        os.close(fd)
        os.unlink(dst)
        raise
    os.close(fd)
    return dst


def install():
    """Route ttnn's tensor-cache loads through `stage` (idempotent; HY4_FAST_LOAD=0 disables)."""
    if os.environ.get("HY4_FAST_LOAD", "1") != "1":
        return False
    import ttnn

    mod = ttnn._ttnn.tensor
    orig = mod.load_tensor_flatbuffer
    if getattr(orig, "_hy4_fast", False):
        return True
    min_bytes = int(float(os.environ.get("HY4_FAST_LOAD_MIN_MB", "16")) * 2**20)
    shm = os.environ.get("HY4_FAST_LOAD_SHM", "/dev/shm")
    if not os.path.isdir(shm) or not os.access(shm, os.W_OK):
        return False

    def load_tensor_flatbuffer(file_name, device=None, *args, **kw):
        path = str(file_name)
        try:
            big = device is not None and os.path.getsize(path) >= min_bytes
        except OSError:
            big = False
        if not big:
            return orig(file_name, device, *args, **kw)
        t0 = time.perf_counter()
        try:
            tmp = stage(path, shm)
        except Exception:  # noqa: BLE001
            STATS["fallbacks"] += 1
            return orig(file_name, device, *args, **kw)
        t1 = time.perf_counter()
        try:
            return orig(tmp, device, *args, **kw)
        finally:
            os.unlink(tmp)
            STATS["files"] += 1
            STATS["bytes"] += os.path.getsize(path)
            STATS["read_s"] += t1 - t0
            STATS["load_s"] += time.perf_counter() - t1

    load_tensor_flatbuffer._hy4_fast = True
    mod.load_tensor_flatbuffer = load_tensor_flatbuffer
    return True


def summary():
    s = STATS
    return (f"fast load: {s['files']} files {s['bytes'] / 2**30:.1f} GiB, O_DIRECT staging {s['read_s']:.0f} s, "
            f"ttnn load {s['load_s']:.0f} s, fallbacks {s['fallbacks']}")
