# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Startup host-RAM check for the served model (tt/generator_vllm.py, before anything loads).

Host memory the server uses:
- BASE: the vLLM API server + the engine process besides the Engram tables (tensor-cache loading, sampling buffers)
  - HY4_RAM_BASE_GIB (8). Measured on the QB2 (tt-model serve, default profile, 2026-11-23): MemAvailable 118 GB
  before the start, 55 GB once ready = ~63 GB with the 60 GiB Engram copy, i.e. ~3-5 GB besides it.
- ENGRAM: the four Engram n-gram tables (FP8 rows + scales, ~61 GiB, read from the weight cache). With
  MemAvailable >= tables + HY4_ENGRAM_WARM_MARGIN_GB (48) when the store opens (right after this check, before the
  model loads), they are copied into process memory in the background (engram_store._load_anon; ~13 s) and mlocked
  when RLIMIT_MEMLOCK allows; otherwise every lookup reads the memory map (the NVMe through the page cache): same
  output, slower decode when the rows are not cached.

HY4_RAM_CHECK: error (default: refuse to start below BASE + HY4_RAM_FLOOR_GIB, warn below the fast path), warn (never
refuse), off. Compared against MemAvailable, capped by the container's cgroup v2 limit: RAM that is free right now
(other jobs count), not swap.
"""

import os
import sys

GIB = 1 << 30
ENGRAM_GIB_EST = 61.0  # the FP8 tables + scales of layers 1/11/21/31 (before the first boot has written them)


def available():
    """Bytes of RAM this process can still get: MemAvailable, capped by a cgroup v2 memory limit."""
    avail = 0
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                avail = int(line.split()[1]) * 1024
    try:
        lim = open("/sys/fs/cgroup/memory.max").read().strip()
        if lim != "max":
            avail = min(avail, int(lim) - int(open("/sys/fs/cgroup/memory.current").read()))
    except (OSError, ValueError):
        pass
    return avail


def engram_bytes(store_dir):
    """Bytes of the converted Engram tables (the estimate when the first boot has not written them yet)."""
    try:
        n = sum(e.stat().st_size for e in os.scandir(store_dir) if e.name.endswith(".npy"))
    except OSError:
        n = 0
    return n if n > 30 * GIB else int(ENGRAM_GIB_EST * GIB)


def check_startup(store_dir):
    """Called once per server, before the model loads. Raises (HY4_RAM_CHECK=error) when even the memory-mapped
    Engram path does not fit; logs which Engram path the free RAM allows."""
    mode = os.environ.get("HY4_RAM_CHECK", "error")
    if mode == "off" or os.environ.get("_HY4_RAM_CHECKED") == "1":
        return
    os.environ["_HY4_RAM_CHECKED"] = "1"
    base = float(os.environ.get("HY4_RAM_BASE_GIB", "8")) * GIB
    floor = float(os.environ.get("HY4_RAM_FLOOR_GIB", "8")) * GIB  # page cache for the mapped Engram rows + headroom
    margin = float(os.environ.get("HY4_ENGRAM_WARM_MARGIN_GB", "48")) * GIB
    tables = engram_bytes(store_dir)
    avail = available()
    need_min, need_fast = base + floor, tables + margin  # the Engram loader's own rule
    gb = lambda n: f"{n / 1e9:.0f} GB"  # noqa: E731
    sys.stderr.write(
        f"[hy4] host RAM: {gb(avail)} free; needs {gb(need_min)} to start (engine {gb(base)} + {gb(floor)} headroom), "
        f"{gb(need_fast)} for the Engram tables in RAM ({gb(tables)} copied into RAM + {gb(margin)} margin)\n"
    )
    try:
        import resource

        lim = resource.getrlimit(resource.RLIMIT_MEMLOCK)[0]
    except (ImportError, OSError, ValueError):
        lim = -1
    if avail < need_min:
        msg = (
            f"host RAM check failed: {gb(avail)} free, {gb(need_min)} needed to start this server. Free host RAM (stop "
            f"other jobs on this machine), or set HY4_RAM_CHECK=warn to start anyway (the kernel may swap or kill the "
            f"server)."
        )
        if mode != "warn":
            raise RuntimeError(msg)
        sys.stderr.write(f"[hy4] WARNING: {msg}\n")
    elif avail < need_fast:
        sys.stderr.write(
            f"[hy4] WARNING: {gb(need_fast - avail)} short of the in-RAM Engram path: the Engram tables are read from "
            f"their memory map (the weight cache on disk through the page cache). Same output; decode is slower "
            f"when the rows are not cached. Free host RAM (stop other jobs) to get the fast path.\n"
        )
    elif lim != -1 and 0 <= lim < tables:
        sys.stderr.write(
            f"[hy4] note: the Engram copy cannot be locked (memlock limit {lim} bytes; `--ulimit memlock=-1` lifts it): "
            f"under memory pressure its rows can be swapped out (slower decode)\n"
        )
