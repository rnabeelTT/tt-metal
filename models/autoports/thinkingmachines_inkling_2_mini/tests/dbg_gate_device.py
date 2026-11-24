# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Debug tt/gate.py: (a) zero forget / gain weights -> out == h; (b) error per hidden tile (core = tile // 3);
(c) the implied f / a error from a with h = 0 (out = a m) and m = 1."""

import torch

import ttnn
from models.autoports.thinkingmachines_inkling_2_mini.tt.gate import GateKernel
from models.autoports.thinkingmachines_inkling_2_mini.tests.test_gate_device import ref_gate


def main():
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    H, Rk, eps = 2304, 64, 1e-6
    k = GateKernel(mesh, H, Rk, eps)
    up = lambda t, dt: ttnn.from_torch(t, dtype=dt, layout=ttnn.TILE_LAYOUT, device=mesh,  # noqa: E731
                                       memory_config=ttnn.L1_MEMORY_CONFIG)
    torch.manual_seed(0)
    T = 1
    h, m = torch.randn(1, 1, T, H) * 3, torch.randn(1, 1, T, H) * 2
    wd = (torch.randn(Rk, 2 * H) * 0.05).bfloat16()
    wf, wg = (torch.randn(H, Rk) * 0.3).bfloat16(), (torch.randn(H, Rk) * 0.3).bfloat16()
    twd = up(wd.float().T.reshape(1, 1, 2 * H, Rk), ttnn.bfloat16)
    run = lambda h_, m_, wf_, wg_: ttnn.to_torch(k(up(h_, ttnn.float32), up(m_, ttnn.float32), twd, up(  # noqa: E731
        torch.cat([wf_.float().T, wg_.float().T], 1).reshape(1, 1, Rk, 2 * H), ttnn.bfloat16))).float()[:, :, :T]
    z = torch.zeros_like(wf)
    o = run(h, m, z, z)
    print(f"(a) zero W: max |out - h| {(o - h).abs().max().item():.2e}", flush=True)
    ref = ref_gate(h, m, wd, wf, wg, eps)
    o = run(h, m, wf, wg)
    e = (o - ref).abs().reshape(-1, 32).amax(-1)  # per hidden tile (T = 1: row 0)
    print("(b) per-tile max err: " + " ".join(f"{v:.1e}" for v in e.tolist()), flush=True)
    # (c) h = 0, m = 1 -> out = a (gain gate) ; the norms of h = 0 make g depend on m only
    hz, m1 = torch.zeros_like(h), torch.ones_like(m)
    ref_a = ref_gate(hz + 1e-30, m1, wd, wf, wg, eps)
    o = run(hz, m1, wf, wg)
    print(f"(c) a error max {(o - ref_a).abs().max().item():.2e}, |a| max {ref_a.abs().max().item():.2e}", flush=True)
    # (d) g sensitivity: perturb g by one bf16 ulp in the reference
    n = lambda x: x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)  # noqa: E731
    g = (torch.cat([n(h), n(m)], -1).bfloat16().float() @ wd.float().T)
    print(f"(d) g fp32 vs bf16 rel {((g - g.bfloat16().float()).abs().max() / g.abs().max()).item():.1e}; "
          f"|g| max {g.abs().max().item():.2f}", flush=True)
    def trunc(x):  # fp32 -> bf16 toward zero
        return (x.view(torch.int32) & ~0xFFFF).view(torch.float32)

    o = run(h, m, wf, wg)
    for nr, gr in (("rne", "rne"), ("trunc", "rne"), ("rne", "trunc"), ("trunc", "trunc"), ("rne", "none")):
        R = {"rne": lambda x: x.bfloat16().float(), "trunc": trunc, "none": lambda x: x}
        hn = R[nr](torch.cat([n(h), n(m)], -1))
        gg = R[gr](hn @ wd.float().T)
        f, a = torch.tanh(gg @ wf.float().T), torch.tanh(gg @ wg.float().T)
        rr = (1 - f) * h + a * m
        print(f"(e) norm {nr:5s} g {gr:5s}: max err {(o - rr).abs().max().item():.2e}", flush=True)
    # the op path (model.MessageGate's ttnn ops) on the same inputs
    from models.autoports.thinkingmachines_inkling_2_mini.tt.model import Ops, _bf, _f32
    from models.autoports.thinkingmachines_inkling_2_mini.tt.dmm import dmm
    ops = Ops(mesh, eps)
    th, tm = up(h, ttnn.float32), up(m, ttnn.float32)
    hm = ttnn.concat([_bf(ops.rms(th)), _bf(ops.rms(tm))], dim=-1)
    g = dmm(hm, twd, ops.ckc, dtype=ttnn.bfloat16)
    wfg32 = up(torch.cat([wf.float().T, wg.float().T], 1).reshape(1, 1, Rk, 2 * H), ttnn.float32)
    fg = torch.tanh(ttnn.to_torch(dmm(_f32(g), wfg32, ops.ckc, dtype=ttnn.float32)).float()[:, :, :T])
    oo = (1 - fg[..., :H]) * h + fg[..., H:] * m
    print(f"(f) op path vs torch ref: max err {(oo - ref).abs().max().item():.2e}; fused vs op path "
          f"{(o - oo).abs().max().item():.2e}", flush=True)
    gt = ttnn.to_torch(g).float()[0, 0, 0]
    gr = (torch.cat([n(h), n(m)], -1).bfloat16().float() @ wd.float().T).bfloat16().float()[0, 0, 0]
    print(f"(g) op-path g vs ref g: max {(gt - gr).abs().max().item():.2e}, ulps differing "
          f"{(gt != gr).float().mean().item():.2f}", flush=True)
    ttnn.close_mesh_device(mesh)
    print("== dbg done")


if __name__ == "__main__":
    main()
