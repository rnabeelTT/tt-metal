# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Debug: the MoE of a 32-wide decode step, sorted vs masked on the SAME inputs at every application."""

import os

import torch

import ttnn
import models.autoports.thinkingmachines_inkling_2_mini.tt.moe as mm
from models.autoports.thinkingmachines_inkling_2_mini.tt.batched import BatchedDecoder
from models.autoports.thinkingmachines_inkling_2_mini.tt.model import Inkling2Model


def pcc(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def main():
    N = int(os.environ.get("MESH", "4"))
    B = 32
    if N > 1:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, N))
    d0 = (lambda t: ttnn.get_device_tensors(t)[0]) if N > 1 else (lambda t: t)
    m = Inkling2Model(mesh, max_seq_len=512, expert_dtype=ttnn.bfloat8_b, kv_dtype=ttnn.bfloat16, prefill_chunk=256,
                      slots=B)
    g = torch.load("/cache/fixtures/golden_t256.pt")
    ids = g["ids"]
    dec = BatchedDecoder(m, B)
    for b in range(B):
        P = 40 + 6 * b
        m.reset()
        m.forward(torch.cat([ids[:P], torch.zeros((-P) % 64, dtype=ids.dtype)]), 0, valid=P, slot=b)
        dec.load_slot(b)
    orig = mm.MoE.forward
    stats = []

    def both(self, x):
        mm.SORTED = False
        a = ttnn.to_torch(d0(orig(self, x))).float()
        mm.SORTED = True
        out = orig(self, x)
        b = ttnn.to_torch(d0(out)).float()
        xin = ttnn.to_torch(d0(x)).float()
        stats.append((pcc(a, b), (a - b).abs().max().item(), list(x.shape), str(x.memory_config().buffer_type),
                      xin.abs().max().item()))
        return out

    mm.MoE.forward = both
    dec.set_inputs(torch.tensor([int(ids[40 + 6 * b]) for b in range(B)]), torch.tensor([40 + 6 * b for b in range(B)]))
    dec.step()
    for i, s in enumerate(stats):
        print(f"app {i:2d}: sorted vs masked pcc {s[0]:.6f} max {s[1]:.3e}  x {s[2]} {s[3]} |x| {s[4]:.2f}", flush=True)
    ttnn.close_mesh_device(mesh)
    print("== dbg done")


if __name__ == "__main__":
    main()
