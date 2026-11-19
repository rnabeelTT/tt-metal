# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Microbenchmark: one hyper-connection pre + post on decode rows (X [1, 4, 32, 3072]), the day-0 form vs split-K."""

from __future__ import annotations

import os

import torch
import ttnn

from models.autoports.tencent_hy4_flash.tools.bench_dsh import timed


def main():
    from models.autoports.tencent_hy4_flash.tt.hc import HyperConnection

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=100_000_000)
    try:
        rep = ttnn.ReplicateTensorToMesh(mesh)
        torch.manual_seed(0)
        hc, H = 4, 3072
        fn = torch.randn(8, hc * H) * 0.02
        base, scale = torch.randn(8) * 0.5, torch.tensor([0.7, 1.3])
        X = (torch.randn(1, hc, 32, H) * 2).to(torch.bfloat16)
        out = torch.randn(1, 1, 32, H).to(torch.bfloat16)
        L1 = ttnn.L1_MEMORY_CONFIG
        Xt = ttnn.from_torch(X, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep, memory_config=L1)
        ot = ttnn.from_torch(out, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep, memory_config=L1)
        res = {}
        for mode in ("0", "1"):
            os.environ["HY4_HC_SPLITK"] = mode
            h = HyperConnection(mesh, fn, base, scale, hc, H, 1e-5, 1e-6, 2.0)

            def f():
                a, post = h.pre(Xt, L1)
                y = h.post(Xt, ot, post, L1)
                ttnn.deallocate(a)
                ttnn.deallocate(post)
                return y

            def g():
                a, post = h.pre(Xt, L1)
                ttnn.deallocate(post)
                return a

            t, y = timed(mesh, f)
            tp, a = timed(mesh, g)
            res[mode] = (ttnn.to_torch(ttnn.get_device_tensors(a)[0]).float(), ttnn.to_torch(ttnn.get_device_tensors(y)[0]).float())
            print(f"HC splitk={mode}: pre+post {t:6.1f} us, pre {tp:6.1f} us", flush=True)
        # reference (fp32)
        xf = X.float().reshape(hc, 32, H)
        flat = xf.permute(1, 0, 2).reshape(32, hc * H)
        mix = (flat @ fn.T) * torch.rsqrt(flat.pow(2).mean(-1, keepdim=True) + 1e-5)
        pre = torch.sigmoid(mix[:, :hc] * scale[0] + base[:hc]) + 1e-6
        inp = (pre.T[:, :, None] * xf).sum(0).to(torch.bfloat16).float()
        for mode, (a, y) in res.items():
            print(f"splitk={mode}: sublayer input max abs err vs fp32 ref {(a.reshape(32, H) - inp).abs().max():.3e}; "
                  f"post vs splitk=0 max diff {(y - res['0'][1]).abs().max():.3e}", flush=True)
        # fused (tt/hc_fused): POST (previous HC's post weights) + PRE of a second HC vs the unfused post + pre
        os.environ["HY4_HC_SPLITK"] = "1"
        from models.autoports.tencent_hy4_flash.tt.hc_fused import HCFusedOp

        h1 = HyperConnection(mesh, fn, base, scale, hc, H, 1e-5, 1e-6, 2.0)
        fn2 = torch.randn(8, hc * H) * 0.02
        h2 = HyperConnection(mesh, fn2, base.flip(0), scale.flip(0), hc, H, 1e-5, 1e-6, 2.0)
        a1, post1 = h1.pre(Xt, L1)
        w1 = h1._weights_splitk(ttnn.typecast(Xt, ttnn.float32, memory_config=L1), L1)  # [1, 1, 32, 32]
        X2 = h1.post(Xt, ot, post1, L1)
        a2, post2 = h2.pre(X2, L1)
        op = HCFusedOp.get(mesh, H)
        Xf, inf_, wf = op(Xt, ot, w1, h2)
        t2 = lambda t: ttnn.to_torch(ttnn.get_device_tensors(t)[0]).float()  # noqa: E731
        print(f"fused X' vs unfused: max diff {(t2(Xf) - t2(X2)).abs().max():.3e} "
              f"(frac differing {(t2(Xf) != t2(X2)).float().mean():.4f})", flush=True)
        print(f"fused in vs unfused: max diff {(t2(inf_) - t2(a2)).abs().max():.3e} "
              f"(frac differing {(t2(inf_) != t2(a2)).float().mean():.4f})", flush=True)
        wu = h2._weights_splitk(ttnn.typecast(X2, ttnn.float32, memory_config=L1), L1)
        print(f"fused w vs unfused: max diff {(t2(wf) - t2(wu)).abs().max():.3e}", flush=True)
        # fp32 torch reference of post (h1's weights from X) then pre (h2) -> X', in; both device paths vs it
        def wref(Xb, fn_, base_, scale_):
            flat = Xb.float().permute(2, 0, 1, 3).reshape(32, hc * H)
            mix = (flat @ fn_.T) * torch.rsqrt(flat.pow(2).mean(-1, keepdim=True) + 1e-5)
            sc_ = torch.tensor([float(scale_[0])] * hc + [float(scale_[1])] * hc)
            a_ = torch.tensor([1.0] * hc + [2.0] * hc)
            return a_ * torch.sigmoid(mix * sc_ + base_) + 1e-6  # [32, 8]

        w1r = wref(X, fn, base, scale)
        X2r = (w1r[:, hc:].T[None, :, :, None] * out.float() + X.float()).to(torch.bfloat16)  # [1, 4, 32, H]
        w2r = wref(X2r, fn2, base.flip(0), scale.flip(0))
        inr = (w2r[:, :hc].T[:, :, None] * X2r[0].float()).sum(0).to(torch.bfloat16).float()  # [32, H]
        for name, xp, ip in (("fused", t2(Xf), t2(inf_)), ("unfused", t2(X2), t2(a2))):
            print(f"{name}: X' frac != ref {(xp != X2r.float()).float().mean():.4f}, in frac != ref "
                  f"{(ip.reshape(32, H) != inr).float().mean():.4f} (max {(ip.reshape(32, H) - inr).abs().max():.3e})",
                  flush=True)
        t, _ = timed(mesh, lambda: op(Xt, ot, w1, h2)[1])
        tu, _ = timed(mesh, lambda: h2.pre(h1.post(Xt, ot, post1, L1), L1)[0])
        print(f"fused post+pre {t:6.1f} us vs unfused (split-K) {tu:6.1f} us", flush=True)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
