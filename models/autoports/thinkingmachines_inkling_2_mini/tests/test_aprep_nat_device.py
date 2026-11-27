# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""tt/aprep.py prefill prep on whole tiles (I2M_APREP_NAT) vs the per-row prefill prep: q / r / sink rows < NH and the
K / V fill tensors bitwise, sliding and global (tau) layers, per-die head counts of TP 1 / 2 / 4, DR 16. Also vs fp32
torch. env: ROWS (default 256)."""

import os
import types

import torch

import ttnn
import models.autoports.thinkingmachines_inkling_2_mini.tt.aprep as ap


def l2clip(x, thr):
    return x * torch.clamp(thr / x.norm(dim=-1, keepdim=True), max=1.0)


def main():
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    T = int(os.environ.get("ROWS", "256"))
    HD, DR = 128, 16
    ok = True
    torch.manual_seed(0)
    cap = []
    real_fill = ttnn.experimental.paged_fill_cache

    def fake_fill(cache, t, tab, batch_idx=0):  # (the fill tensors, kept)
        cap.append(ttnn.to_torch(t).float())

    ttnn.experimental.paged_fill_cache = fake_fill
    for tp in [int(t) for t in os.environ.get("TPS", "4,2,1").split(",")]:
        NH, NKV = 24 // tp, 8 // tp
        qd, rd, sd, gd, kd = NH * HD, NH * DR, 32, NH * HD, NKV * HD
        fq = qd + rd + sd + gd
        for sliding in (True, False):
            w = types.SimpleNamespace(NH=NH, NKV=NKV, HD=HD, DR=DR, qd=qd, rd=rd, sd=sd, gd=gd, mesh=mesh,
                                      sliding=sliding, qk_clip=22.6, rel_clip=8.0, eps=1e-6, scale=HD**-0.5)
            pq = torch.randn(T, fq) * torch.rand(T, fq).mul(4.0)
            k = torch.randn(T, kd) * 2.5
            v = torch.randn(T, kd) * 3.0
            tau = torch.rand(T) + 1.0
            up = lambda t, dt, sh: ttnn.from_torch(t.reshape(sh), dtype=dt, layout=ttnn.TILE_LAYOUT,  # noqa: E731
                                                   device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            args = (up(pq, ttnn.float32, (1, 1, T, fq)), up(k, ttnn.bfloat16, (1, 1, T, kd)),
                    up(v, ttnn.bfloat16, (1, 1, T, kd)), None if sliding else up(tau, ttnn.float32, (1, T, 1, 1)))
            tab = ttnn.from_torch(torch.arange(T // 64, dtype=torch.int32).reshape(1, -1), dtype=ttnn.int32,
                                  layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh)
            res = {}
            for nat in (False, True):
                ap.NAT = nat
                cap.clear()
                q, r, _, _, s = ap.AttnPrep(w, T, None)(*args, fill=(None, None, tab))
                res[nat] = [ttnn.to_torch(t).float()[:, :, :NH] for t in (q, r, s)] + list(cap)
            names = ["q", "r", "sink", "k fill", "v fill"]
            for i, n in enumerate(names):
                a, b = res[False][i], res[True][i]
                same = torch.equal(a, b)
                ok &= same
                d = (a - b).abs().max().item()
                print(f"TP{tp} {'sliding' if sliding else 'global '} {n}: {'bitwise' if same else 'DIFF'} max|d| {d:.3g}"
                      f" shape {tuple(a.shape)}", flush=True)
                if not same:
                    bad = (a != b).nonzero()
                    print(f"   first diffs at {bad[:6].tolist()}", flush=True)
            tf = tau[:, None, None] if not sliding else 1.0
            ref_q = l2clip(pq[:, :qd].reshape(T, NH, HD), 22.6) * tf
            for nat in (False, True):
                g = res[nat][0].reshape(T, NH, HD)
                e = (g - ref_q).abs().amax(-1)
                print(f"   NAT={int(nat)} q vs fp32 torch max rel {(e.max() / ref_q.abs().max()).item():.2e} "
                      f"bad (token, head) {(e > 1).nonzero()[:4].tolist()}", flush=True)
    ttnn.experimental.paged_fill_cache = real_fill
    print("PASS" if ok else "FAIL", flush=True)
    ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
