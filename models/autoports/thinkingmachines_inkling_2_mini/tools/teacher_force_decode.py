# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Teacher-forced quality of the DECODE path (what a served user gets after the prompt): each held-out text's first P
tokens are prefilled into a slot, then the next D tokens are fed one at a time through the batched decoder (traced,
B slots: B texts decode together), and every step's logits are scored against the CPU reference like
tools/teacher_force.py (top1, top1_conf, kl32, nll).

usage: tools/run_dev.sh python .../tools/teacher_force_decode.py      env: B (4), P (256), D (256), EXPERT_DT (bfp8),
       KV_DT (bf16), MESH (1), REF (/cache/fixtures/ref_bf16.pt), TRACE (1), VERIFY (0; R: one text at a time through
       the speculative verify pass, R teacher-forced rows per pass, all accepted)
"""

import json
import os
import time

import numpy as np
import torch

import ttnn
from models.autoports.thinkingmachines_inkling_2_mini.tt.batched import BatchedDecoder, TracedDecoder
from models.autoports.thinkingmachines_inkling_2_mini.tt.model import Inkling2Model

DT = {"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b, "bfp4": ttnn.bfloat4_b}


def main():
    ref = torch.load(os.environ.get("REF", "/cache/fixtures/ref_bf16.pt"))
    names = list(ref)
    B, P, D = int(os.environ.get("B", "4")), int(os.environ.get("P", "256")), int(os.environ.get("D", "256"))
    N = int(os.environ.get("MESH", "1"))
    V = int(os.environ.get("VERIFY", "0"))
    if V:
        B = V
    if N > 1:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, N), trace_region_size=400 << 20)
    m = Inkling2Model(mesh, max_seq_len=P + D + 64, expert_dtype=DT[os.environ.get("EXPERT_DT", "bfp8")],
                      kv_dtype=DT[os.environ.get("KV_DT", "bf16")], prefill_chunk=256, slots=B)
    dec = BatchedDecoder(m, B)
    if V:
        dec = BatchedDecoder(m, V, shared=dec, verify=V)
    G = 1 if V else B  # (texts per group)
    d0 = (lambda t: ttnn.get_device_tensors(t)[0]) if N > 1 else (lambda t: t)
    tr = None
    results = {}
    for g0 in range(0, len(names), G):
        group = names[g0 : g0 + G]
        for b, name in enumerate(group):
            ids = ref[name]["ids"]
            m.reset()
            for p0 in range(0, P, 256):
                c = ids[p0 : min(P, p0 + 256)]
                n = len(c)
                if n % 64:
                    c = torch.cat([c, torch.zeros((-n) % 64, dtype=c.dtype)])
                m.forward(c, p0, valid=n, slot=b, last_only=True, last_row=n - 1)
            dec.load_slot(b)
        if tr is None and os.environ.get("TRACE", "1") == "1":
            tr = TracedDecoder(dec)
            tr.capture()
            for b, name in enumerate(group):  # the capture advanced the conv states: prefill again
                ids = ref[name]["ids"]
                m.reset()
                for p0 in range(0, P, 256):
                    c = ids[p0 : min(P, p0 + 256)]
                    n = len(c)
                    if n % 64:
                        c = torch.cat([c, torch.zeros((-n) % 64, dtype=c.dtype)])
                    m.forward(c, p0, valid=n, slot=b, last_only=True, last_row=n - 1)
                dec.load_slot(b)
        d = tr or dec
        lps = [[] for _ in group]
        t0 = time.time()
        if V:
            dec.set_accept([0])
        for t in range(0, D, V) if V else []:
            ids = ref[group[0]]["ids"]
            d.set_inputs(ids[P + t : P + t + V], torch.arange(P + t, P + t + V))
            lg = m.logits_host(d.step(), V)
            dec.set_accept([V - 1])
            lps[0] += [torch.log_softmax(lg[r], -1) for r in range(V)]
        for t in range(0 if V else D):
            tok = torch.tensor([ref[n_]["ids"][P + t] for n_ in group] + [0] * (B - len(group)))
            pos = torch.tensor([P + t] * len(group) + [-1] * (B - len(group)))
            d.set_inputs(tok, pos)
            lg = m.logits_host(d.step(), B)
            for b in range(len(group)):
                lps[b].append(torch.log_softmax(lg[b], -1))
        dt = time.time() - t0
        for b, name in enumerate(group):
            r = ref[name]
            lp = torch.stack(lps[b])  # logits for positions P .. P+D-1 (predicting P+1 .. P+D)
            top_ids, top_lp = r["top_ids"][P : P + D], r["top_lp"][P : P + D]
            nxt = r["ids"][P + 1 : P + D + 1]
            k = min(len(nxt), D)
            top1 = (lp[:k].argmax(-1) == top_ids[:k, 0]).float()
            conf = top_lp[:k, 0].exp() > 0.5
            kl = (top_lp[:k].exp() * (top_lp[:k] - lp[:k].gather(-1, top_ids[:k]))).sum(-1)
            nll = -lp[:k].gather(-1, nxt[:k, None])[:, 0]
            results[name] = {"top1": top1.mean().item(), "top1_conf": top1[conf].mean().item(), "kl32": kl.mean().item(),
                             "nll": nll.mean().item(), "nll_ref": r["nll"][P : P + k].mean().item()}
            x = results[name]
            print(f"{name}: top1 {x['top1']:.4f}  top1_conf {x['top1_conf']:.4f}  kl32 {x['kl32']:.4f}  "
                  f"nll {x['nll']:.3f} (ref {x['nll_ref']:.3f})", flush=True)
        print(f"  group of {len(group)}: {D} steps in {dt:.1f}s ({dt / D * 1000:.1f} ms/step incl. readback)", flush=True)
    agg = {k: float(np.mean([v[k] for v in results.values()])) for k in ("top1", "top1_conf", "kl32", "nll", "nll_ref")}
    print("MEAN " + json.dumps({k: round(v, 4) for k, v in agg.items()}), flush=True)
    if tr is not None:
        tr.release()
    ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
