# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""First-serve build of the hybrid's expert copies from the model's own checkpoint (no separate weights repo).

A package ships only its routing plan (HYB_BUILD_PLAN: the compact sim_*.py output, per layer the routed mass of every
expert, ~0.5 MB). On the first start the server builds into the weight cache (HYB_LIB / HYB_DEV_EXPERTS, plain paths
under the tt-model weight-cache mount):
  - the host int4 library: every expert outside each layer's top HYB_BUILD_S (build_int4_library.py, CPU);
  - with the fetch on (HYB_LRU > 0): the device-format (bfp4) copies of the experts the fetch may copy, plan ranks
    [C - 1 - HYB_LRU, + HYB_PIN_TOP) (build_dev_experts.py, host-side packing; the device must be attached, no mesh
    is opened).
Both builders skip finished files, so an interrupted first start resumes; a marker file records a finished build.
Later starts find the markers and go on at once. Runs in the first process that installs the overlay (the vLLM API
server, before the engine opens the device), under a file lock.

env: HYB_AUTOBUILD (1 = on when HYB_BUILD_PLAN is set; 0 = off), HYB_BUILD_PLAN (path or hf://), HYB_BUILD_S (192),
HYB_LIB_KEYS (checkpoint key style), HYB_BUILD_THREADS (default: all cores), HF_MODEL (the checkpoint, pinned by
tt-model's TT_MODEL_WEIGHTS_REVISION).
"""

import fcntl
import os
import subprocess
import sys
import time

from . import hfpath

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # models/experimental/hybrid_moe


def _log(msg):
    print(f"[hybrid build] {time.strftime('%H:%M:%S')} {msg}", flush=True)


def _done(path):
    return os.path.exists(os.path.join(path, ".complete"))


def _run(script, args, env, what, out):
    t0 = time.time()
    _log(f"{what}: building into {out} (first start only; resumes if interrupted)")
    r = subprocess.run([sys.executable, os.path.join(_HERE, script)] + args, env={**os.environ, **env})
    if r.returncode != 0:
        raise RuntimeError(f"[hybrid build] {what} failed (exit {r.returncode}); restart to resume")
    with open(os.path.join(out, ".complete"), "w") as f:
        f.write(f"{script} {' '.join(args)}\n")
    _log(f"{what}: done in {(time.time() - t0) / 60:.1f} min")


def ensure():
    plan = hfpath.resolve(os.environ.get("HYB_BUILD_PLAN", ""))
    if not plan or os.environ.get("HYB_AUTOBUILD", "1") != "1":
        return
    lib = hfpath.resolve(os.environ.get("HYB_LIB", ""))
    dev = hfpath.resolve(os.environ.get("HYB_DEV_EXPERTS", ""))
    lru = int(os.environ.get("HYB_LRU", "0") or 0)
    need_dev = bool(dev) and lru > 0
    if (not lib or _done(lib)) and (not need_dev or _done(dev)):
        return
    if not os.path.exists(plan):
        raise RuntimeError(f"[hybrid build] HYB_BUILD_PLAN {plan} not found")
    snap = hfpath.snapshot(os.environ["HF_MODEL"])
    if not snap or not os.path.exists(os.path.join(snap, "model.safetensors.index.json")):
        raise RuntimeError(f"[hybrid build] no checkpoint snapshot for {os.environ.get('HF_MODEL')} (got {snap})")
    S = os.environ.get("HYB_BUILD_S", "192")
    threads = os.environ.get("HYB_BUILD_THREADS", str(os.cpu_count() or 8))
    keys = os.environ.get("HYB_LIB_KEYS", "qwen35")
    os.makedirs(os.path.dirname(lib or dev), exist_ok=True)
    lock = open(os.path.join(os.path.dirname(lib or dev), ".hybrid-build.lock"), "w")
    fcntl.flock(lock, fcntl.LOCK_EX)  # one builder at a time (a second process waits, then finds the markers)
    try:
        if lib and not _done(lib):
            os.makedirs(lib, exist_ok=True)
            _run(
                "build_int4_library.py",
                [snap, lib],
                {"HYB_INT4_GS": "32", "HYB_LIB_KEYS": keys, "HYB_LIB_COLD": f"{plan}:{S}", "THREADS": threads},
                "host int4 expert library (CPU, ~1 hour)",
                lib,
            )
        if need_dev and not _done(dev):
            os.makedirs(dev, exist_ok=True)
            C = int(os.environ.get("HYB_SLOTS", "256"))
            lo = C - 1 - lru
            pin = int(os.environ.get("HYB_PIN_TOP", "0") or 0)
            hi = lo + pin if pin else 1 << 30
            _run(
                "build_dev_experts.py",
                [snap, dev, f"{plan}:{S}"],
                {"HYB_LIB_KEYS": keys, "HYB_DEV_RANKS": f"{lo}:{hi}", "THREADS": threads},
                f"device-format expert copies, ranks {lo}-{hi if pin else 'end'} per layer",
                dev,
            )
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()
