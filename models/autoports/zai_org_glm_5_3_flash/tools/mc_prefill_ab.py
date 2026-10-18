# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage-5 A/B of the multichip CCL options (tt/multichip_decoder.MCOpts topology / num_links) on warmed prefill of
one real layer (one build; options are read at run time). Per variant: one compile pass, then `--reps` timed passes
(host wall with a device sync, as tests/test_perf_decoder.py::test_perf_prefill) and the output PCC against the
first variant.

  python -m models.autoports.zai_org_glm_5_3_flash.tools.mc_prefill_ab --kind kda_moe --rows 2048 \
      --variants "base:;ring:topology=ring" --out ab.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import time

os.environ.setdefault("GLM53_TP", "4")

import torch  # noqa: E402

import ttnn  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tests.harness import (  # noqa: E402
    H4,
    build_layer,
    close_mesh,
    open_mesh,
    stream_in,
    stream_out,
)
from models.autoports.zai_org_glm_5_3_flash.tools.mc_decode_ab import KINDS, Stage4Base, pcc, set_opts  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tt.ckpt import CheckpointReader  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", required=True)
    ap.add_argument("--rows", type=int, default=2048)
    ap.add_argument("--variants", required=True)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", required=True)
    ap.add_argument("--save", help="torch.save the first variant's output here")
    ap.add_argument("--ref", help="compare every variant with this saved output (e.g. another build's)")
    args = ap.parse_args()
    ext = torch.load(args.ref) if args.ref and os.path.exists(args.ref) else None
    out = open(args.out, "a")
    rd = CheckpointReader()
    mesh = open_mesh()
    try:
        n = args.rows
        dec = build_layer(
            mesh, rd, KINDS[args.kind], batch=8, prefill_lengths=(n,), max_context=4096, decoder_cls=Stage4Base
        )
        g = torch.Generator().manual_seed(1)
        x = (torch.randn(1, 1, n, H4, generator=g) * 0.02).float()
        X = stream_in(mesh, x)
        inputs = dec.prefill_inputs(n, start=0, length=n, slot=0)
        ref = None
        for item in [v for v in args.variants.split(";") if v.strip()]:
            name, spec = item.split(":", 1) if ":" in item else (item, "")  # bash read strips a trailing ":"
            set_opts(dec, spec)
            rec = {"kind": args.kind, "rows": n, "variant": name, "spec": spec}
            try:
                Y, _ = dec.prefill_forward(X, inputs=inputs)  # compile
                y = stream_out(Y)[0, 0]
                ttnn.deallocate(Y)
                ttnn.synchronize_device(mesh)
                ts = []
                for _ in range(args.reps):
                    t = time.time()
                    Y, _ = dec.prefill_forward(X, inputs=inputs)
                    ttnn.synchronize_device(mesh)
                    ts.append((time.time() - t) * 1e3)
                    ttnn.deallocate(Y)
                rec["ms"] = min(ts)
                rec["ms_all"] = ts
                if ref is None:
                    ref = y
                    if args.save:
                        torch.save(y, args.save)
                rec["pcc_vs_first"] = pcc(ref, y)
                d, d0 = (y - x[0, 0]), (ref - x[0, 0])  # layer updates (output - input stream)
                rec["update_pcc_vs_first"] = pcc(d0, d)
                rec["update_rms_vs_first"] = float(d.norm() / d0.norm())
                if ext is not None:
                    rec["pcc_vs_ref"] = pcc(ext, y)
                    rec["maxabs_vs_ref"] = float((ext - y).abs().max())
            except Exception as e:  # noqa: BLE001
                rec["error"] = f"{type(e).__name__}: {str(e)[:400]}"
            print("PF", json.dumps(rec))
            out.write(json.dumps(rec) + "\n")
            out.flush()
    finally:
        close_mesh(mesh)


if __name__ == "__main__":
    main()
