# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""tt/gatet.py (fused message-gate tail) vs fp32 torch and the ttnn-op tail: T in (1, 4, 32), H 2304."""

import torch

import ttnn
from models.autoports.thinkingmachines_inkling_2_mini.tt.gatet import GateTail


def main():
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    H = 2304
    k = GateTail(mesh, H)
    up = lambda t: ttnn.from_torch(t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh)  # noqa: E731
    worst_k = worst_o = 0.0
    torch.manual_seed(0)
    for T in (1, 4, 32):
        fg, h, m = torch.randn(1, 1, T, 2 * H) * 2, torch.randn(1, 1, T, H), torch.randn(1, 1, T, H)
        ref = h - torch.tanh(fg[..., :H]) * h + torch.tanh(fg[..., H:]) * m
        out = ttnn.to_torch(k(up(fg), up(h), up(m))).float()[:, :, :T]
        tf = ttnn.tanh(up(fg))
        f = ttnn.slice(tf, [0, 0, 0, 0], [1, 1, T, H])
        a = ttnn.slice(tf, [0, 0, 0, H], [1, 1, T, 2 * H])
        th, tm = up(h), up(m)
        oo = ttnn.to_torch(ttnn.add(ttnn.subtract(th, ttnn.multiply(f, th)), ttnn.multiply(a, tm))).float()[:, :, :T]
        dk, do = (out - ref).abs().max().item(), (oo - ref).abs().max().item()
        worst_k, worst_o = max(worst_k, dk), max(worst_o, do)
        print(f"T={T}: kernel {dk:.2e}  ops {do:.2e}", flush=True)
    ttnn.close_mesh_device(mesh)
    print(f"{'PASS' if worst_k <= max(2 * worst_o, 1e-3) else 'FAIL'}: worst kernel {worst_k:.2e}, ops {worst_o:.2e}")


if __name__ == "__main__":
    main()
