# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Teacher-forced quality of the Inkling-2-Mini TT model against the CPU reference (reference/ref_texts.py), on the
held-out texts (models/experimental/hybrid_moe/quality_texts.py): the quality gate, same metrics as the other ports.

  top1       device argmax == reference argmax
  top1_conf  the same where the reference is confident (p > 0.5)
  kl32       KL(ref || device) over the reference's top 32 (both log-probs over the full vocab)
  nll        device next-token NLL (ref's alongside)

usage (in the image): tools/run_dev.sh python models/autoports/thinkingmachines_inkling_2_mini/tools/teacher_force.py
env: EXPERT_DT bf16|bfp8|bfp4 (or a per-layer spec, see tt/precision.py), MESH (devices: 1, 2, 4), CHUNK (256),
     REF (/cache/fixtures/ref_bf16.pt), ONLY (comma list of text names)
"""

import json
import os
import time

import numpy as np
import torch

import ttnn
from models.autoports.thinkingmachines_inkling_2_mini.tt.model import Inkling2Model

DT = {"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b, "bfp4": ttnn.bfloat4_b}
REF = os.environ.get("REF", "/cache/fixtures/ref_bf16.pt")
CHUNK = int(os.environ.get("CHUNK", "256"))


def main():
    ref = torch.load(REF)
    only = os.environ.get("ONLY")
    names = [n for n in ref if not only or n in only.split(",")]
    n_dev = int(os.environ.get("MESH", "1"))
    if n_dev > 1:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, n_dev))
    dt = os.environ.get("EXPERT_DT", "bfp4")
    t0 = time.time()
    kv = os.environ.get("KV_DT", "bfp8")
    m = Inkling2Model(mesh, max_seq_len=1024 + CHUNK, expert_dtype=DT.get(dt, dt), prefill_chunk=CHUNK,
                      kv_dtype=DT[kv])
    print(f"model up in {time.time() - t0:.0f}s (experts {dt}, kv {kv}, {n_dev} devices)", flush=True)
    results = {}
    for name in names:
        r = ref[name]
        ids = r["ids"]
        T = len(ids)
        t0 = time.time()
        m.reset()
        lps = []
        for p0 in range(0, T, CHUNK):
            chunk = ids[p0 : p0 + CHUNK]
            n = len(chunk)
            if n % 64:
                chunk = torch.cat([chunk, torch.zeros((-n) % 64, dtype=chunk.dtype)])
            lg = m.forward(chunk, p0, valid=n)
            lg = ttnn.to_torch(lg if n_dev == 1 else ttnn.get_device_tensors(lg)[0]).float()
            lps.append(torch.log_softmax(lg.reshape(len(chunk), -1)[:n, : m.V], -1))
        lp = torch.cat(lps)
        top_ids, top_lp = r["top_ids"], r["top_lp"]
        top1 = (lp[:-1].argmax(-1) == top_ids[:-1, 0]).float()
        conf = top_lp[:-1, 0].exp() > 0.5
        dev_lp32 = lp.gather(-1, top_ids)
        kl = (top_lp.exp() * (top_lp - dev_lp32)).sum(-1)[:-1]
        nll = -lp[:-1].gather(-1, ids[1:, None])[:, 0]
        res = {"top1": top1.mean().item(), "top1_conf": top1[conf].mean().item(), "kl32": kl.mean().item(),
               "nll": nll.mean().item(), "nll_ref": r["nll"].mean().item()}
        results[name] = res
        print(f"{name}: top1 {res['top1']:.4f}  top1_conf {res['top1_conf']:.4f}  kl32 {res['kl32']:.4f}  "
              f"nll {res['nll']:.3f} (ref {res['nll_ref']:.3f})  [{time.time() - t0:.0f}s]", flush=True)
    agg = {k: float(np.mean([v[k] for v in results.values()])) for k in ("top1", "top1_conf", "kl32", "nll", "nll_ref")}
    print("MEAN " + json.dumps({k: round(v, 4) for k, v in agg.items()}), flush=True)
    ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
