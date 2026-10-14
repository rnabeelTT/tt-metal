# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Probe: exactness of the DSA pool-id -> token-id expansion (tok = 4 p + j) at long-context pool ids, for the matmul
expansion and the elementwise alternatives. Prints the max |error| in tokens per method."""

import torch

import ttnn

dev = ttnn.open_device(device_id=0)
try:
    rows, K, KP = 32, 512, 4
    p = torch.randint(0, 262144, (1, 1, rows, K)).float()
    ref = (p.repeat_interleave(KP, dim=-1) * KP + torch.arange(KP).float().repeat(K)).double()
    up = lambda t, dt=ttnn.float32: ttnn.from_torch(t, dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev)
    pd = up(p)
    expand = up(torch.kron(torch.eye(K), torch.full((1, KP), float(KP))).reshape(1, 1, K, K * KP))
    offs = up(torch.arange(KP).float().repeat(K).reshape(1, 1, 1, K * KP))
    for fid in (ttnn.MathFidelity.HiFi2, ttnn.MathFidelity.HiFi4):
        ckc = ttnn.init_device_compute_kernel_config(
            dev.arch(), math_fidelity=fid, fp32_dest_acc_en=True, packer_l1_acc=False, math_approx_mode=False
        )
        tok = ttnn.add(ttnn.matmul(pd, expand, compute_kernel_config=ckc, dtype=ttnn.float32), offs)
        print(f"matmul {fid}: max err {(ttnn.to_torch(tok).double() - ref).abs().max().item()}")
    ri = ttnn.repeat_interleave(pd, KP, dim=3)
    tok = ttnn.add(ttnn.multiply(ri, float(KP)), offs)
    print(f"repeat_interleave: shape {list(ri.shape)} max err {(ttnn.to_torch(tok).double() - ref).abs().max().item()}")
finally:
    ttnn.close_device(dev)
