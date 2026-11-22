# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""tt/gate.py (the message gate in one program) vs transformers' Inkling2LoopMessageGate math in torch (fp32 norms,
bf16 down projection, fp32 forget / gain projections of bf16 weights): T in (1, 4, 32), H 2304, rank 64; several calls
in a row (semaphore reuse) and two weight sets. These random gates are ill-conditioned (|g| ~ 8: one bf16 ulp of g
moves the tanh inputs by ~1e-2), so the bar is the ttnn-op path's own error on the same inputs (tests/dbg_gate_device.py:
op path 2.2e-1 max abs vs torch, this kernel 1.2e-1): pass below 3e-2 max relative."""

import os

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
    G2 = os.environ.get("GATE2", "0") == "1"  # tt/gate.py GateKernel2 (the post-attention rectangle; T <= 16)
    if G2:
        from models.autoports.thinkingmachines_inkling_2_mini.tt.gate import GateKernel2
        from models.autoports.thinkingmachines_inkling_2_mini.tt.post import PostKernel

        k = GateKernel2(PostKernel(mesh, H, eps), Rk)
    else:
        k = GateKernel(mesh, H, Rk, eps)
    up = lambda t, dt: ttnn.from_torch(t, dtype=dt, layout=ttnn.TILE_LAYOUT, device=mesh,  # noqa: E731
                                       memory_config=ttnn.L1_MEMORY_CONFIG)
    worst, worst_n = 0.0, 0.0
    torch.manual_seed(0)
    for wi in range(2):
        wd = (torch.randn(Rk, 2 * H) * 0.05).bfloat16()
        wf, wg = (torch.randn(H, Rk) * 0.3).bfloat16(), (torch.randn(H, Rk) * 0.3).bfloat16()
        wup = (lambda t, dt: ttnn.from_torch(t, dtype=dt, layout=ttnn.TILE_LAYOUT, device=mesh)) if G2 else up  # noqa: E731
        twd = wup(wd.float().T.reshape(1, 1, 2 * H, Rk), ttnn.bfloat16)
        twfg = wup(torch.cat([wf.float().T, wg.float().T], 1).reshape(1, 1, Rk, 2 * H), ttnn.bfloat16)
        for T in (1, 4, 16) if G2 else (1, 4, 16, 32):
            for rep in range(3):
                h, m = torch.randn(1, 1, T, H) * 3, torch.randn(1, 1, T, H) * 2
                ref = ref_gate(h, m, wd, wf, wg, eps)
                got = ttnn.to_torch(k(up(h, ttnn.float32), up(m, ttnn.float32), twd, twfg)).float()[:, :, :T]
                d = ((got - ref).abs().max() / ref.abs().max()).item()
                worst = max(worst, d)
                if rep == 0:
                    print(f"weights {wi} T={T}: max rel err {d:.2e}", flush=True)
                if G2:  # + the RMS norm (and at <= 8 rows the router logits) vs this gate's out
                    nw = 1 + 0.1 * torch.randn(H)
                    W = torch.randn(H, 128) * 0.02
                    hi = W.bfloat16()
                    rw = (wup(hi.float().reshape(1, 1, H, 128), ttnn.bfloat16),
                          wup((W - hi.float()).bfloat16().float().reshape(1, 1, H, 128), ttnn.bfloat16))
                    tnw = wup(nw.reshape(1, 1, 1, H).expand(1, 1, 32, H).contiguous(), ttnn.float32)
                    rt = T <= GateKernel2.ROUTER_ROWS
                    xo, lg = k(up(h, ttnn.float32), up(m, ttnn.float32), twd, twfg, nw=tnw, router=rw if rt else None,
                               dup=True)
                    xo = ttnn.to_torch(xo).float()[:, :, :T]
                    lg = ttnn.to_torch(lg).float() if rt else None
                    o = got.double()
                    xr = (o * torch.rsqrt(o.pow(2).mean(-1, keepdim=True) + eps) * nw.double()).bfloat16().float()
                    dx = ((xo - xr).abs().max() / xr.abs().max()).item()
                    lr = xo.double() @ W.double()
                    dl = ((lg[:, :, :T].double() - lr).abs().max() / lr.abs().max()).item() if rt else 0.0
                    dd = (lg[:, :, 1] - lg[:, :, 0]).abs().max().item() if T == 1 else 0.0
                    if rep == 0 and rt:  # the op path's router (tt/moe_fused.py _logits) on the same bf16 input
                        ckc = ttnn.init_device_compute_kernel_config(mesh.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
                                                                     math_approx_mode=False, fp32_dest_acc_en=True,
                                                                     packer_l1_acc=False)
                        xb = ttnn.from_torch(xo.reshape(1, 1, T, H), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                             device=mesh)
                        wf32 = ttnn.from_torch(W.reshape(1, 1, H, 128), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
                                               device=mesh)
                        lo_ = ttnn.to_torch(ttnn.matmul(ttnn.typecast(xb, ttnn.float32), wf32, compute_kernel_config=ckc,
                                                        dtype=ttnn.float32)).double()
                        print(f"  op-path router: {((lo_ - lr).abs().max() / lr.abs().max()).item():.2e}", flush=True)
                    worst_n = max(worst_n, dx, dl, dd * 1e9)
                    if rep == 0:
                        print(f"  norm: max rel err {dx:.2e}  router: {dl:.2e}  dup row diff {dd:.1e}", flush=True)
    ttnn.close_mesh_device(mesh)
    ok = worst < 3e-2 and worst_n < 1e-2
    print(f"{'PASS' if ok else 'FAIL'}: worst rel err {worst:.2e}" + (f", norm / router {worst_n:.2e}" if G2 else ""))


if __name__ == "__main__":
    main()
