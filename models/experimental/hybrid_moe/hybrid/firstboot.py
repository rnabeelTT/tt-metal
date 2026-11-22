# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""First boot: build the host copies of the experts that can miss from the served checkpoint, so a package downloads
only the official weights. Two directories, each built once and kept (the package's weight cache):

    HYB_LIB          int4 host library, one layer_<L>.pt per hybrid layer (build_int4_library.py, CPU)
    HYB_DEV_EXPERTS  bfp4 device-format copies, <L>_<e>_gu / _dn.tensorbin (build_dev_experts.py; ttnn's bfp4 packing
                     needs the cluster, so it runs in a subprocess that exits before the server opens the device)

for every expert outside each layer's top HYB_BUILD_STATIC of the plan's build `order` (HYB_PLAN, a compact plan with
`order` and `freq`), in the layers HYB_BUILD_LAYERS. With HYB_BFP4_ROUND's gptq methods, a calibration capture
(bfp4_calib.py, HYB_BFP4_CAL) is built first: the device copies and the port's expert conversion round with it. Both builders resume per layer / expert and write atomically, so
an interrupted first boot continues where it stopped. A `.complete` marker records what was built (checkpoint
revision, plan hash, static count, layers); a marker that does not match the current settings is an error: delete
the directory to rebuild. Runs in the first process only (the vLLM API server, before the EngineCore starts), like
the RAM check. Not configured (HYB_BUILD_LAYERS unset) = the directories are expected to exist already (dev serving).
A first boot's build is byte-identical to the copies v1.4 published (all 42 + 8736 files, 2026-11-15).
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


def _calibrate(snap, here, spec):
    """HYB_BFP4_ROUND's gptq methods round with a calibration capture (bfp4_calib.py, CPU): build it in HYB_BFP4_CAL
    first, for the device copies here and the port's conversion in the EngineCore."""
    if not os.environ.get("HYB_BFP4_ROUND"):
        return
    import bfp4_round

    if not bfp4_round.uses_cal():
        return
    out = hfpath.resolve(os.environ.get("HYB_BFP4_CAL", ""))
    if not out:
        raise ValueError(f"HYB_BFP4_ROUND={os.environ['HYB_BFP4_ROUND']} uses gptq methods: set HYB_BFP4_CAL")
    with open(os.path.join(here, "plans", "bfp4_cal_tokens.npy"), "rb") as f:
        want = {"checkpoint": spec["checkpoint"], "tokens": hashlib.sha256(f.read()).hexdigest()[:16]}
    if _done(out, want, "bfp4 calibration capture"):
        return
    os.makedirs(out, exist_ok=True)
    for tmp in glob.glob(os.path.join(out, "*.tmp")):
        os.remove(tmp)
    threads = str(os.cpu_count() or 8)
    _run(
        "bfp4 calibration capture",
        [sys.executable, os.path.join(here, "bfp4_calib.py"), snap, out],
        dict(os.environ, REF_THREADS=threads),
    )
    with open(os.path.join(out, MARKER), "w") as f:
        json.dump(want, f)


def ensure():
    if not os.environ.get("HYB_BUILD_LAYERS") or os.environ.get("_HYB_EXPERTS_READY") == "1":
        return
    os.environ["_HYB_EXPERTS_READY"] = "1"  # the EngineCore child inherits it
    lib, dev, plan = (hfpath.resolve(os.environ.get(k, "")) for k in ("HYB_LIB", "HYB_DEV_EXPERTS", "HYB_PLAN"))
    if not (lib and plan):
        raise ValueError(
            "HYB_BUILD_LAYERS needs HYB_LIB and HYB_PLAN (a compact plan with `order`); HYB_DEV_EXPERTS "
            "too when the profile fetches / streams experts over PCIe"
        )
    snap, spec = _spec(plan)
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    layers = ",".join(str(v) for v in spec["layers"])
    cold = f"{plan}:{spec['static']}"
    env = dict(os.environ, HYB_LIB_KEYS=spec["keys"], THREADS=str(os.cpu_count() or 8))
    _calibrate(snap, here, spec)
    builds = [
        (
            "int4 host expert library",
            lib,
            [os.path.join(here, "build_int4_library.py"), snap, lib, layers],
            {"HYB_LIB_COLD": cold},
            spec,
        ),
        (
            "device-format expert copies",
            dev,
            [os.path.join(here, "build_dev_experts.py"), snap, dev, cold, layers],
            {},
            spec,
        ),
    ]
    # HYB_FETCH_DN_EXPERTS / HYB_FETCH_DN_LAYERS: the downs of the layers whose device format differs from the copies
    # above, in the served precision policy's format (hybrid/fetch.py, decode fetch of a mixed-precision profile)
    dn_dir = hfpath.resolve(os.environ.get("HYB_FETCH_DN_EXPERTS", ""))
    dn_layers = os.environ.get("HYB_FETCH_DN_LAYERS", "")
    if dn_dir and dn_layers:
        from .fetch import _parse_layers

        dl = ",".join(str(v) for v in sorted(_parse_layers(dn_layers)))
        pol = os.environ.get("QWEN35_PRECISION_CONFIG", "")
        builds.append(
            (
                "own-format down copies",
                dn_dir,
                [os.path.join(here, "build_dev_experts.py"), snap, dn_dir, cold, dl],
                {"HYB_DEV_PARTS": "dn", "HYB_DEV_POLICY": pol},
                dict(spec, layers=[int(v) for v in dl.split(",")], parts="dn", policy=os.path.basename(pol)),
            )
        )
    for what, out, cmd, extra, spec in builds:
        if not out:  # no HYB_DEV_EXPERTS: a profile without the PCIe fetch / prefill streaming
            continue
        want = spec
        if out == dev and os.environ.get("HYB_BFP4_ROUND"):  # the device copies also depend on the bfp4 rounding
            import bfp4_round

            want = dict(spec, bfp4_round={str(L): bfp4_round.tag(L) for L in spec["layers"]})
        if _done(out, want, what):
            continue
        os.makedirs(out, exist_ok=True)
        for tmp in glob.glob(os.path.join(out, "*.tmp*")):  # a write cut off by an earlier interrupted boot
            os.remove(tmp)
        _run(what, [sys.executable, *cmd], dict(env, **extra))
        with open(os.path.join(out, MARKER), "w") as f:
            json.dump(want, f)
