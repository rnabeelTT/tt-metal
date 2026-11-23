# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Speculative verify rows: R consecutive tokens of one sequence in one pass (BatchedDecoder verify=R) must give the
logits of R sequential decode steps. Pass 1 from the prefill (accepted row 0), pass 2 after accepting A + 1 rows of
pass 1 (its state from row A); eager and traced.
env: MESH (1), R (4), A (1)"""

import os

import torch

import ttnn
from models.autoports.thinkingmachines_inkling_2_mini.tt.batched import BatchedDecoder, TracedDecoder
from models.autoports.thinkingmachines_inkling_2_mini.tt.model import Inkling2Model


def pcc(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def main():
    g = torch.load("/cache/fixtures/golden_t256.pt")
    ids = g["ids"]
    P, R, A = 100, int(os.environ.get("R", "4")), int(os.environ.get("A", "1"))
    N = int(os.environ.get("MESH", "1"))
    if N > 1:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, N), trace_region_size=400 << 20)
    m = Inkling2Model(mesh, max_seq_len=512, expert_dtype=ttnn.bfloat8_b, kv_dtype=ttnn.bfloat16, prefill_chunk=256,
                      slots=R)
    full = BatchedDecoder(m, R)
    one = BatchedDecoder(m, 1, shared=full)
    ver = BatchedDecoder(m, R, shared=full, verify=R)

    def prefill():
        m.reset()
        m.forward(torch.cat([ids[:P], torch.zeros((-P) % 64, dtype=ids.dtype)]), 0, valid=P, slot=0)
        full.load_slot(0)

    # sequential reference: A + 1 + R steps from position P
    prefill()
    seq = []
    for t in range(A + 1 + R):
        one.set_inputs(ids[P + t : P + t + 1], torch.tensor([P + t]))
        seq.append(m.logits_host(one.step(), 1)[0])
    seq = torch.stack(seq)
    # the same steps on the full-width decoder (slot 0 live, the others idle): the verify pass's kernels, one row each
    W = full.B
    prefill()
    seqw = []
    for t in range(A + 1 + R):
        full.set_inputs(torch.tensor([int(ids[P + t])] + [0] * (W - 1)), torch.tensor([P + t] + [-1] * (W - 1)))
        seqw.append(m.logits_host(full.step(), W)[0])
    seqw = torch.stack(seqw)
    print("1-wide vs full-width sequential, per row pcc:",
          " ".join(f"{pcc(seq[r], seqw[r]):.5f}" for r in range(A + 1 + R)), flush=True)

    def two_passes(d):
        prefill()
        ver.set_accept([0])
        d.set_inputs(ids[P : P + R], torch.arange(P, P + R))
        a = m.logits_host(d.step(), R)
        ver.set_accept([A])
        q0 = P + A + 1
        d.set_inputs(ids[q0 : q0 + R], torch.arange(q0, q0 + R))
        b = m.logits_host(d.step(), R)
        return a, b

    worst, ok = 1.0, True
    for name, d in (("eager", ver), ("traced", None)):
        if d is None:
            prefill()
            d = TracedDecoder(ver)
            d.set_inputs(ids[P : P + R], torch.arange(P, P + R))
            d.capture()
        a, b = two_passes(d)
        for tag, got, ref in (("pass 1", a, seqw[:R]), (f"pass 2 (accept {A + 1})", b, seqw[A + 1 :])):
            print(f"  {name} {tag} per row pcc: " + " ".join(f"{pcc(got[r], ref[r]):.6f}" for r in range(R)))
            p = min(pcc(got[r], ref[r]) for r in range(R))
            top = (got.argmax(-1) == ref.argmax(-1)).float().mean().item()
            md = (got - ref).abs().max().item()
            worst, ok = min(worst, p), ok and top == 1.0
            print(f"{name} {tag}: min row pcc {p:.6f}, top-1 agree {top:.3f}, max |diff| {md:.4f}", flush=True)
        if isinstance(d, TracedDecoder):
            d.release()
    # the drafter's taps vs the HF target's (tools/dspark_taps.py fixture), one verify pass on a fixture text
    tf = "/cache/fixtures/dspark_taps.pt"
    if os.path.exists(tf):
        rec = next(iter(torch.load(tf)["texts"].values()))
        tid, ref_taps = rec["ids"], rec["taps"].float()
        Q = 128
        m.reset()
        m.forward(tid[:Q], 0, valid=Q, slot=0)
        full.load_slot(0)
        ver.set_accept([0])
        ver.set_inputs(tid[Q : Q + R], torch.arange(Q, Q + R))
        ver.step()
        tp = ttnn.to_torch(ttnn.get_device_tensors(ver.taps)[0] if N > 1 else ver.taps).float()[0, :, :R]  # [4, R, H]
        for j in range(4):
            p = min(pcc(tp[j, r], ref_taps[Q + r, j]) for r in range(R))
            print(f"tap {j}: min row pcc vs HF {p:.5f}", flush=True)
            ok = ok and p > 0.97  # (the device model vs HF: ~0.98 on the final state)
    ttnn.close_mesh_device(mesh)
    print(f"{'PASS' if worst > 0.9999 and ok else 'FAIL'}: worst pcc {worst:.6f}")


if __name__ == "__main__":
    main()
