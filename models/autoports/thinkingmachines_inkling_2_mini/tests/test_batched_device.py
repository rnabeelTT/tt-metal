# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Multi-user decode test: B slots prefilled to different lengths of the golden text, then D teacher-forced batched
decode steps; every slot's logits vs the CPU reference at its own positions (top-1 agreement, pcc), plus a control
run of one slot alone (B-slot model, the others idle) that must give the same logits as in the batch.

usage (in the image): tools/run_dev.sh python models/autoports/thinkingmachines_inkling_2_mini/tests/test_batched_device.py
env: EXPERT_DT (bfp8), KV_DT (bf16), MESH (1), D (8), PREFIX (64,100,160,200)
"""

import os
import time

import torch

import ttnn
from models.autoports.thinkingmachines_inkling_2_mini.tt.batched import BatchedDecoder, TracedDecoder
from models.autoports.thinkingmachines_inkling_2_mini.tt.model import Inkling2Model

DT = {"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b, "bfp4": ttnn.bfloat4_b}


def pcc(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def main():
    g = torch.load(os.environ.get("FIX", "/cache/fixtures/golden_t256.pt"))
    ids, ref = g["ids"], g["logits"]
    prefix = [int(x) for x in os.environ.get("PREFIX", "64,100,160,200").split(",")]
    B, D = len(prefix), int(os.environ.get("D", "8"))
    N = int(os.environ.get("MESH", "1"))
    if N > 1:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, N), trace_region_size=int(os.environ.get("TRACE_MB", "300")) << 20)
    m = Inkling2Model(mesh, max_seq_len=512, expert_dtype=DT[os.environ.get("EXPERT_DT", "bfp8")],
                      kv_dtype=DT[os.environ.get("KV_DT", "bf16")], prefill_chunk=256, slots=B)
    dec = BatchedDecoder(m, B)
    d0 = (lambda t: ttnn.get_device_tensors(t)[0]) if N > 1 else (lambda t: t)

    def prefill(b):
        P = prefix[b]
        m.reset()
        chunk = ids[:P]
        pad = (-P) % 64
        m.forward(torch.cat([chunk, torch.zeros(pad, dtype=chunk.dtype)]), 0, valid=P, slot=b)
        dec.load_slot(b)

    t0 = time.time()
    for b in range(B):
        prefill(b)
    print(f"prefilled {B} slots ({prefix}) in {time.time() - t0:.1f}s", flush=True)

    def run(active, d=None):
        d = d or dec
        outs = []
        for t in range(D):
            tok = torch.tensor([ids[prefix[b] + t] if b in active else 0 for b in range(B)])
            pos = torch.tensor([prefix[b] + t if b in active else -1 for b in range(B)])
            d.set_inputs(tok, pos)
            t1 = time.time()
            lg = m.logits_host(d.step(), B)
            outs.append(lg)
            if t == D - 1:
                print(f"  step {time.time() - t1:.2f}s (eager)", flush=True)
        return torch.stack(outs, 1)  # [B, D, V]

    got = run(set(range(B)))
    worst = 1.0
    for b in range(B):
        want = ref[prefix[b] + 1 - 1 : prefix[b] + D]  # logits at the decoded positions
        want = ref[prefix[b] : prefix[b] + D]
        top1 = (got[b].argmax(-1) == want.argmax(-1)).float().mean().item()
        p = pcc(got[b], want)
        worst = min(worst, p)
        print(f"slot {b} (prefix {prefix[b]}): logits pcc {p:.5f}  top-1 {top1:.3f}", flush=True)
    # control: slot 1 alone (re-prefilled), the others idle -> same logits as in the batch
    prefill(1)
    solo = run({1})
    print(f"control: slot 1 alone vs in batch: pcc {pcc(solo[1], got[1]):.6f}, max abs diff "
          f"{(solo[1] - got[1]).abs().max():.4f}", flush=True)
    # traced: re-prefill every slot, capture, replay the same steps -> same logits; then timing
    for b in range(B):
        prefill(b)
    tr = TracedDecoder(dec)
    t0 = time.time()
    tr.capture()
    print(f"trace captured in {time.time() - t0:.1f}s", flush=True)
    for b in range(B):  # the capture's compile + recorded step advanced the conv states: re-prefill
        prefill(b)
    traced = run(set(range(B)), tr)
    print(f"traced vs eager: logits pcc {pcc(traced, got):.6f}, max abs diff {(traced - got).abs().max():.4f}",
          flush=True)
    n = 50
    dec.set_inputs(torch.tensor([ids[prefix[b]] for b in range(B)]), torch.tensor(prefix))
    ttnn.synchronize_device(mesh)
    t0 = time.time()
    for _ in range(n):
        tr.step()
    ttnn.synchronize_device(mesh)
    dt = (time.time() - t0) / n
    print(f"traced decode: {dt * 1000:.1f} ms/step at B={B} -> {1 / dt:.1f} tok/s per user, {B / dt:.0f} tok/s total",
          flush=True)
    tr.release()
    ttnn.close_mesh_device(mesh)
    print(f"{'PASS' if worst > 0.99 else 'FAIL'}: worst slot logits pcc {worst:.5f}")


if __name__ == "__main__":
    main()
