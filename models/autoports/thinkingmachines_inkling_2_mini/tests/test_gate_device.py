# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""tt/gate.py (the message gate in one program) vs transformers' Inkling2LoopMessageGate math in torch (fp32 norms,
bf16 down projection, fp32 forget / gain projections of bf16 weights): T in (1, 4, 32), H 2304, rank 64; several calls
in a row (semaphore reuse) and two weight sets. These random gates are ill-conditioned (|g| ~ 8: one bf16 ulp of g
moves the tanh inputs by ~1e-2), so the bar is the ttnn-op path's own error on the same inputs (tests/dbg_gate_device.py:
op path 2.2e-1 max abs vs torch, this kernel 1.2e-1): pass below 3e-2 max relative."""

import torch

import ttnn
from models.autoports.thinkingmachines_inkling_2_mini.tt.gate import GateKernel


def ref_gate(h, m, wd, wf, wg, eps):
    n = lambda x: x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)  # noqa: E731
    g = (torch.cat([n(h), n(m)], -1).bfloat16().float() @ wd.float().T).bfloat16().float()
    f, a = torch.tanh(g @ wf.float().T), torch.tanh(g @ wg.float().T)
    return (1 - f) * h + a * m


def main():
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    H, Rk, eps = 2304, 64, 1e-6
    k = GateKernel(mesh, H, Rk, eps)
    up = lambda t, dt: ttnn.from_torch(t, dtype=dt, layout=ttnn.TILE_LAYOUT, device=mesh,  # noqa: E731
                                       memory_config=ttnn.L1_MEMORY_CONFIG)
    worst = 0.0
    torch.manual_seed(0)
    for wi in range(2):
        wd = (torch.randn(Rk, 2 * H) * 0.05).bfloat16()
        wf, wg = (torch.randn(H, Rk) * 0.3).bfloat16(), (torch.randn(H, Rk) * 0.3).bfloat16()
        twd = up(wd.float().T.reshape(1, 1, 2 * H, Rk), ttnn.bfloat16)
        twfg = up(torch.cat([wf.float().T, wg.float().T], 1).reshape(1, 1, Rk, 2 * H), ttnn.bfloat16)
        for T in (1, 4, 32):
            for rep in range(3):
                h, m = torch.randn(1, 1, T, H) * 3, torch.randn(1, 1, T, H) * 2
                ref = ref_gate(h, m, wd, wf, wg, eps)
                got = ttnn.to_torch(k(up(h, ttnn.float32), up(m, ttnn.float32), twd, twfg)).float()[:, :, :T]
                d = ((got - ref).abs().max() / ref.abs().max()).item()
                worst = max(worst, d)
                if rep == 0:
                    print(f"weights {wi} T={T}: max rel err {d:.2e}", flush=True)
    ttnn.close_mesh_device(mesh)
    print(f"{'PASS' if worst < 3e-2 else 'FAIL'}: worst rel err {worst:.2e}")


if __name__ == "__main__":
    main()
