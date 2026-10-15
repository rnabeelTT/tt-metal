# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Compute HF layer references recorded by a dry run (GLM53_REF_DRYRUN=<dir> pytest ...) in parallel on the CPU, into
the normal reference cache (tests/hf_ref/client.py), so the device tests later find every reference cached.

usage: python tools/compute_refs.py <dir> [--workers N]"""

import argparse
import concurrent.futures as cf
import os
import time
from pathlib import Path

import torch

from models.autoports.zai_org_glm_5_3_flash.tests.hf_ref.client import cache_path, hf_layer_reference


def _one(path: str) -> str:
    req = torch.load(path, weights_only=False)
    t0 = time.time()
    if not cache_path(req).exists():
        hf_layer_reference(req, timeout=8 * 3600)
    Path(path).unlink(missing_ok=True)
    return f"{Path(path).name}: {time.time() - t0:.0f} s"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    reqs = sorted(str(p) for p in Path(a.dir).glob("*.req.pt"))
    # long-context requests first (the longest jobs)
    reqs.sort(key=lambda p: 0 if "long" in torch.load(p, weights_only=False) else 1)
    print(f"{len(reqs)} requests, {a.workers} workers", flush=True)
    os.environ.pop("GLM53_REF_DRYRUN", None)
    with cf.ProcessPoolExecutor(a.workers) as ex:
        for msg in ex.map(_one, reqs):
            print(msg, flush=True)


if __name__ == "__main__":
    main()
