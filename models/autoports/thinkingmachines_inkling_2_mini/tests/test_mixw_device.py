# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""tt/mixw.py (fused block mix, one tile row) vs the stacked ttnn-op Ops.mix and an fp32 torch reference:
n sources in 1..6, nq in (1, 3, 4), T in (1, 4, 32), H 2304."""

import os

os.environ["I2M_MIX_KERNEL"] = "1"

import torch  # noqa: E402

import ttnn  # noqa: E402
import models.autoports.thinkingmachines_inkling_2_mini.tt.model as mdl  # noqa: E402
from models.autoports.thinkingmachines_inkling_2_mini.tt.model import Ops  # noqa: E402


def main():
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    ops = Ops(mesh, 1e-6)
    H = 2304
    worst_k, worst_o = 0.0, 0.0
    torch.manual_seed(0)
    for nq in (1, 3, 4):
        Qc = torch.randn(H, nq) * H**-0.5 * 3
        Qk = ops.query(Qc)
        for n in range(1, 7):
            for T in ((1, 4, 32) if os.environ.get("BIG") != "1" else (64, 256, 1024)):
                src = [torch.randn(1, 1, T, H) for _ in range(n)]
                tt = [ops.up(s) for s in src]
                S = torch.cat(src, 1)  # [1, n, T, H]
                w = torch.softmax(torch.einsum("bntH,Hq->bntq", S, Qc), dim=1)
                ref = [(S * w[..., q : q + 1]).sum(1, keepdim=True) for q in range(nq)]
                mdl.MIX_KERNEL = True
                ok = [ttnn.to_torch(o).float()[:, :, :T] for o in ops.mix(tt, Qk, nq)] if T <= 32 else ref
                mdl.MIX_KERNEL = False  # the stacked op path (prefill)
                oo = [ttnn.to_torch(o).float()[:, :, :T] for o in ops.mix(tt, Qk, nq)]
                dk = max((a - r).abs().max().item() for a, r in zip(ok, ref))
                do = max((a - r).abs().max().item() for a, r in zip(oo, ref))
                worst_k, worst_o = max(worst_k, dk), max(worst_o, do)
                print(f"nq={nq} n={n} T={T}: kernel {dk:.2e}  ops {do:.2e}", flush=True)
    ttnn.close_mesh_device(mesh)
    print(f"{'PASS' if worst_k < 0.03 and worst_o < 0.03 else 'FAIL'}: worst kernel {worst_k:.2e}, ops {worst_o:.2e}")


if __name__ == "__main__":
    main()
