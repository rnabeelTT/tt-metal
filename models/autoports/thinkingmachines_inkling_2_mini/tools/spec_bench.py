# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Greedy speculative decode (tt/spec.py) vs plain traced decode on one die: ms per token, tokens per pass, and how
long the two token streams agree (they run at different decode widths, so a near-tie can split them).
env: D (3: drafts per pass), NEW (128: tokens per prompt), PROMPT (256: prompt tokens), TEXTS (fixture texts)"""

import os
import time

import torch

import ttnn
from models.autoports.thinkingmachines_inkling_2_mini.tt.batched import BatchedDecoder, TracedDecoder
from models.autoports.thinkingmachines_inkling_2_mini.tt.drafter import DSparkDrafter
from models.autoports.thinkingmachines_inkling_2_mini.tt.model import Inkling2Model
from models.autoports.thinkingmachines_inkling_2_mini.tt.spec import SpecGreedy


def main():
    d, NEW, PL = int(os.environ.get("D", "3")), int(os.environ.get("NEW", "128")), int(os.environ.get("PROMPT", "256"))
    R = d + 1
    texts = torch.load("/cache/fixtures/dspark_taps.pt")["texts"]
    names = os.environ.get("TEXTS", ",".join(texts)).split(",")
    N = int(os.environ.get("MESH", "1"))
    if N > 1:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, N), trace_region_size=400 << 20)
    m = Inkling2Model(mesh, max_seq_len=4096, expert_dtype=ttnn.bfloat8_b, kv_dtype=ttnn.bfloat16, prefill_chunk=512,
                      slots=R)
    full = BatchedDecoder(m, R)
    one = BatchedDecoder(m, 1, shared=full)
    dr = DSparkDrafter(mesh, model=m, max_len=4096)
    spec = SpecGreedy(m, full, dr, d=d)
    plain = TracedDecoder(one)

    def am(lg):
        return spec._argmax(lg, 1)[0]

    pad = lambda ids: torch.cat([ids, torch.zeros((-int(ids.numel())) % 64, dtype=ids.dtype)])  # noqa: E731
    # compile everything eagerly, then capture both traces (a program compiled after a capture can be overwritten)
    w = texts[names[0]]["ids"][:PL].long()
    one.set_inputs(torch.zeros(1, dtype=torch.long), torch.full((1,), -1))
    am(one.step())
    spec.warm(w)
    plain.set_inputs(torch.zeros(1, dtype=torch.long), torch.full((1,), -1))
    plain.capture(compile=False)

    tot = {"plain": 0.0, "spec": 0.0, "n": 0, "passes": 0, "acc": 0}
    for name in names:
        ids = texts[name]["ids"][:PL].long()
        # spec (captures its trace on the first prompt)
        t0 = spec.prefill(ids)
        sp = [t0]
        p0, a0 = spec.passes, spec.accepted
        t = time.time()
        while len(sp) < NEW:
            sp += spec.step()
        ts = time.time() - t
        sp = sp[:NEW]
        np_, na = spec.passes - p0, spec.accepted - a0
        # plain decode of the same prompt
        m.reset()
        P = int(ids.numel())
        lg = m.forward(pad(ids), 0, valid=P, slot=0, last_only=True, last_row=P - 1)
        full.load_slot(0)
        pt = [am(lg)]
        t = time.time()
        while len(pt) < NEW:
            plain.set_inputs(torch.tensor([pt[-1]]), torch.tensor([P + len(pt) - 1]))
            pt.append(am(plain.step()))
        tp = time.time() - t
        agree = next((i for i in range(NEW) if sp[i] != pt[i]), NEW)
        print(f"[{name}] plain {1000 * tp / (NEW - 1):.2f} ms/tok | spec d={d} {1000 * ts / (NEW - 1):.2f} ms/tok, "
              f"{(NEW - 1) / np_:.2f} tok/pass ({na / np_:.2f} drafts accepted / pass, {1000 * ts / np_:.1f} ms/pass) | "
              f"streams agree for {agree} / {NEW} tokens", flush=True)
        tot["plain"] += tp
        tot["spec"] += ts
        tot["n"] += NEW - 1
        tot["passes"] += np_
        tot["acc"] += na
    print(f"MEAN plain {1000 * tot['plain'] / tot['n']:.2f} ms/tok, spec {1000 * tot['spec'] / tot['n']:.2f} ms/tok "
          f"({tot['plain'] / tot['spec']:.2f}x), {tot['n'] / tot['passes']:.2f} tok/pass", flush=True)
    spec.dec.release()
    plain.release()
    ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
