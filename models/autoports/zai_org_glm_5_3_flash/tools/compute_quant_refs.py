# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Compute recorded quantization-aware HF references (tools/quant_ref_requests.py) in batches on the CPU: requests of
the same (layer, weights, quant) share one quantized layer build (tests/hf_ref/hf_quant_batch.py); each batch is split
over `--split` processes. Results go to the normal reference cache (the client cache path of each request).

--check: also recompute one request with the single-request script (hf_quant_ref.py) and compare it with the batch
result (must be bit-identical).

usage: python -m models.autoports.zai_org_glm_5_3_flash.tools.compute_quant_refs <dir> [--split 2] [--check]"""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import torch

from models.autoports.zai_org_glm_5_3_flash.tests.hf_ref.client import QUANT_SCRIPT, _hf_python, cache_path

BATCH = Path(QUANT_SCRIPT).with_name("hf_quant_batch.py")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--split", type=int, default=2)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--stagger", type=int, default=150, help="seconds between process starts")
    a = ap.parse_args()
    d = Path(a.dir)
    groups = {}
    for p in sorted(d.glob("*.req.pt")):
        req = torch.load(p, weights_only=False)
        assert "quant" in req, p
        out = cache_path(req)
        if out.exists():
            p.unlink()
            continue
        key = (int(req["layer"]), json.dumps(req["weights"], sort_keys=True))
        groups.setdefault(key, []).append((p, out, sum(req["segments"])))
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    procs = []
    for i, (key, items) in enumerate(sorted(groups.items())):
        items.sort(key=lambda t: -t[2])  # longest first, dealt round-robin over the processes
        n = min(a.split if len(items) > 6 else 1, len(items))
        for j in range(n):
            part = items[j::n]
            lst = d / f"batch_{i}_{j}.txt"
            lst.write_text("".join(f"{p} {o}\n" for p, o, _ in part))
            log = open(d / f"batch_{i}_{j}.log", "w")
            cmd = [_hf_python(), "-I", str(BATCH), "--batch", str(lst), "--threads", str(a.threads)]
            procs.append((subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT), lst, part))
            print(f"batch {key} part {j}: {len(part)} requests", flush=True)
            time.sleep(a.stagger)  # the layer builds peak at ~28 GB each: do not overlap them
    check = None
    if a.check and procs:  # the smallest request of the first batch, single-request script into a scratch file
        p, o, _ = min(procs[0][2], key=lambda t: t[2])
        check = (o, d / "check_single.pt")
        r = subprocess.run(
            [_hf_python(), "-I", str(QUANT_SCRIPT), "--request", str(p), "--out", str(check[1])],
            env=env,
            capture_output=True,
            text=True,
        )
        print(r.stdout[-500:], r.stderr[-2000:], flush=True)
    rc = 0
    for proc, lst, part in procs:
        rc |= proc.wait()
        for p, o, _ in part:
            if o.exists():
                Path(p).unlink(missing_ok=True)
    left = list(d.glob("*.req.pt"))
    print(f"batches done (rc {rc}); {len(left)} requests left", flush=True)
    if check is not None:
        b, s = torch.load(check[0], weights_only=False), torch.load(check[1], weights_only=False)
        for k in ("y", "y_swap", "route_margin"):
            if k in b or k in s:
                same = torch.equal(b[k], s[k])
                print(f"check {k}: batch == single {same}", flush=True)
                rc |= int(not same)
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
