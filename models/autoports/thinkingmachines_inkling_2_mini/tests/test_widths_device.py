# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Decode width buckets: a narrower BatchedDecoder sharing a wider one's state must give the same logits for the slots
it covers. Slots 0 / 1 prefilled; D steps of {slot 0} on the 4-wide decoder vs the 1-wide one, {0, 1} on 4-wide vs
2-wide (re-prefilled in between); eager and traced; plus the CPU reference top-1 at those positions.
env: MESH (1), D (6)"""

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
    ids, ref = g["ids"], g["logits"]
    prefix, D, B = [100, 160], int(os.environ.get("D", "6")), 4
    N = int(os.environ.get("MESH", "1"))
    if N > 1:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, N), trace_region_size=400 << 20)
    m = Inkling2Model(mesh, max_seq_len=512, expert_dtype=ttnn.bfloat8_b, kv_dtype=ttnn.bfloat16, prefill_chunk=256,
                      slots=B)
    full = BatchedDecoder(m, B)
    narrow = {w: BatchedDecoder(m, w, shared=full) for w in (1, 2)}
    d0 = (lambda t: ttnn.get_device_tensors(t)[0]) if N > 1 else (lambda t: t)

    def prefill():
        for b, P in enumerate(prefix):
            m.reset()
            m.forward(torch.cat([ids[:P], torch.zeros((-P) % 64, dtype=ids.dtype)]), 0, valid=P, slot=b)
            full.load_slot(b)

    def run(d, W, active):
        outs = []
        for t in range(D):
            tok = torch.tensor([ids[prefix[b] + t] if b in active else 0 for b in range(W)])
            pos = torch.tensor([prefix[b] + t if b in active else -1 for b in range(W)])
            d.set_inputs(tok, pos)
            outs.append(ttnn.to_torch(d0(d.step())).float().reshape(W, -1)[:, : m.V])
        return torch.stack(outs)  # [D, W, V]

    worst = 1.0
    for W, active in ((1, [0]), (2, [0, 1])):
        prefill()
        a = run(full, B, active)
        prefill()
        b = run(narrow[W], W, active)
        for s in active:
            p = pcc(a[:, s], b[:, s])
            top = (a[:, s].argmax(-1) == b[:, s].argmax(-1)).float().mean().item()
            refp = ref[prefix[s] : prefix[s] + D].float()
            tr = (b[:, s].argmax(-1) == refp.argmax(-1)).float().mean().item()
            worst = min(worst, p)
            print(f"width {W} slot {s}: vs 4-wide pcc {p:.6f} top-1 agree {top:.3f}; vs ref top-1 {tr:.3f}", flush=True)
    # traced narrow decoders (both widths captured, replayed alternately)
    prefill()
    trs = {w: TracedDecoder(narrow[w]) for w in (1, 2)}
    for w, t in trs.items():
        t.set_inputs(torch.zeros(w, dtype=torch.long), torch.full((w,), -1))
        t.capture()
    prefill()
    a = run(trs[2], 2, [0, 1])
    prefill()
    b = run(narrow[2], 2, [0, 1])
    p = pcc(a, b)
    worst = min(worst, p)
    print(f"traced width 2 vs eager: pcc {p:.6f}", flush=True)
    for t in trs.values():
        t.release()
    ttnn.close_mesh_device(mesh)
    print(f"{'PASS' if worst > 0.9999 else 'FAIL'}: worst pcc {worst:.6f}")


if __name__ == "__main__":
    main()
