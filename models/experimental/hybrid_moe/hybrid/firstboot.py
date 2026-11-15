# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""First boot: build the host copies of the experts that can miss from the served checkpoint, so a package downloads
only the official weights. Two directories, each built once and kept (the package's weight cache):

    HYB_LIB          int4 host library, one layer_<L>.pt per hybrid layer (build_int4_library.py, CPU)
    HYB_DEV_EXPERTS  bfp4 device-format copies, <L>_<e>_gu / _dn.tensorbin (build_dev_experts.py; ttnn's bfp4 packing
                     needs the cluster, so it runs in a subprocess that exits before the server opens the device)

for every expert outside each layer's top HYB_BUILD_STATIC of the plan's build `order` (HYB_PLAN, a compact plan with
`order` and `freq`), in the layers HYB_BUILD_LAYERS. Both builders resume per layer / expert and write atomically, so
an interrupted first boot continues where it stopped. A `.complete` marker records what was built (checkpoint
revision, plan hash, static count, layers); a marker that does not match the current settings is an error: delete
the directory to rebuild. Runs in the first process only (the vLLM API server, before the EngineCore starts), like
the RAM check. Not configured (HYB_BUILD_LAYERS unset) = the directories are expected to exist already (dev serving).
A local build is byte-identical to the published library (layer 0: every tensor equal, 2026-11-14).
"""

import glob
import hashlib
import json
import os
import subprocess
import sys
import time

from . import hfpath

MARKER = ".complete"


def _log(msg):
    sys.stderr.write(f"[hybrid] {msg}\n")
    sys.stderr.flush()


def _spec(plan):
    snap = hfpath.snapshot(os.environ["HF_MODEL"])
    rev = os.path.basename(os.path.normpath(snap))
    with open(plan, "rb") as f:
        plan_sha = hashlib.sha256(f.read()).hexdigest()[:16]
    return snap, {
        "checkpoint": f"{os.environ['HF_MODEL']}@{rev}",
        "plan": plan_sha,
        "static": int(os.environ["HYB_BUILD_STATIC"]),
        "layers": sorted(int(v) for v in os.environ["HYB_BUILD_LAYERS"].split(",")),
        "keys": os.environ.get("HYB_LIB_KEYS", "minimax_m2"),
    }


def _done(out, spec, what):
    path = os.path.join(out, MARKER)
    if not os.path.exists(path):
        return False
    with open(path) as f:
        have = json.load(f)
    if have != spec:
        raise RuntimeError(
            f"{what} in {out} was built for {have}, the server now asks for {spec}: delete {out} to rebuild it"
        )
    return True


def _run(what, cmd, env):
    t0 = time.time()
    _log(f"first boot: building the {what} (one time; resumes if interrupted): {' '.join(cmd)}")
    r = subprocess.run(cmd, env=env)
    if r.returncode != 0:
        raise RuntimeError(f"building the {what} failed (exit {r.returncode}): {' '.join(cmd)}")
    _log(f"first boot: {what} built in {(time.time() - t0) / 60:.1f} min")


def ensure():
    if not os.environ.get("HYB_BUILD_LAYERS") or os.environ.get("_HYB_EXPERTS_READY") == "1":
        return
    os.environ["_HYB_EXPERTS_READY"] = "1"  # the EngineCore child inherits it
    lib, dev, plan = (hfpath.resolve(os.environ.get(k, "")) for k in ("HYB_LIB", "HYB_DEV_EXPERTS", "HYB_PLAN"))
    if not (lib and dev and plan):
        raise ValueError("HYB_BUILD_LAYERS needs HYB_LIB, HYB_DEV_EXPERTS and HYB_PLAN (a compact plan with `order`)")
    snap, spec = _spec(plan)
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    layers = ",".join(str(v) for v in spec["layers"])
    cold = f"{plan}:{spec['static']}"
    env = dict(os.environ, HYB_LIB_KEYS=spec["keys"], THREADS=str(os.cpu_count() or 8))
    for what, out, cmd, extra in (
        (
            "int4 host expert library",
            lib,
            [os.path.join(here, "build_int4_library.py"), snap, lib, layers],
            {"HYB_LIB_COLD": cold},
        ),
        ("device-format expert copies", dev, [os.path.join(here, "build_dev_experts.py"), snap, dev, cold, layers], {}),
    ):
        if _done(out, spec, what):
            continue
        os.makedirs(out, exist_ok=True)
        for tmp in glob.glob(os.path.join(out, "*.tmp*")):  # a write cut off by an earlier interrupted boot
            os.remove(tmp)
        _run(what, [sys.executable, *cmd], dict(env, **extra))
        with open(os.path.join(out, MARKER), "w") as f:
            json.dump(spec, f)
