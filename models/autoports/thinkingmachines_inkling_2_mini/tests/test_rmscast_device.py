# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Ops.rms with the dtype casts folded into its reshards (I2M_RMS_CAST) vs the explicit typecasts: bf16 / fp32 inputs,
bf16 / fp32 outputs, T in (1, 4, 32): must be bit-identical.
usage: tools/run_dev.sh python .../tests/test_rmscast_device.py"""

import torch

import ttnn
from models.autoports.thinkingmachines_inkling_2_mini.tt.model import Ops


def main():
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    ops = Ops(mesh, 1e-6)
    H = 2304
    torch.manual_seed(0)
    w = ops.up(torch.randn(1, 1, 1, H) * 0.2 + 1)
    ok = True
    for T in (1, 4, 32):
        for din in (ttnn.bfloat16, ttnn.float32):
            x = ttnn.from_torch(torch.randn(1, 1, T, H) * 3, dtype=din, layout=ttnn.TILE_LAYOUT, device=mesh,
                                memory_config=ttnn.L1_MEMORY_CONFIG)
            for dout in (ttnn.bfloat16, ttnn.float32):
                for ww in (None, w):
                    ref = ops._rms(x, ww)
                    ref = ttnn.typecast(ref, dout) if ref.dtype != dout else ref
                    got = ops.rms(x, ww, dout)
                    a, b = ttnn.to_torch(ref).float(), ttnn.to_torch(got).float()
                    d = (a - b).abs().max().item()
                    ok = ok and d == 0 and got.dtype == dout
                    print(f"T={T} in {din} out {dout} weight {ww is not None}: max|d| {d:.3g}", flush=True)
    print("PASS" if ok else "FAIL", flush=True)
    ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
