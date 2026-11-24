# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""tt/aprep.py (decode attention prep in one program) vs an fp32 torch reference: B in (1, 4, 32), sliding and global
(tau) layers, the real dims (NH 24, NKV 8, HD 128, DR 32) and clips; inputs scaled so some heads clip and some do not."""

import types

import torch

import ttnn
from models.autoports.thinkingmachines_inkling_2_mini.tt.aprep import AttnPrep


def l2clip(x, thr):
    n = x.norm(dim=-1, keepdim=True)
    return x * torch.clamp(thr / n, max=1.0)


def main():
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    NH, NKV, HD, DR = 24, 8, 128, 16
    qd, rd, sd, gd, kd = NH * HD, NH * DR, 32, NH * HD, NKV * HD
    fq = qd + rd + sd + gd
    worst = 0.0
    torch.manual_seed(0)
    grid = mesh.compute_with_storage_grid_size()
    for B in (1, 4, 32):
        kv_mc = ttnn.create_sharded_memory_config(
            shape=(32, HD), core_grid=ttnn.num_cores_to_corerangeset(B, grid, row_wise=True),
            strategy=ttnn.ShardStrategy.HEIGHT, orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True)
        for sliding in (True, False):
            w = types.SimpleNamespace(NH=NH, NKV=NKV, HD=HD, DR=DR, qd=qd, rd=rd, sd=sd, gd=gd, mesh=mesh,
                                      sliding=sliding, qk_clip=22.6, rel_clip=8.0, eps=1e-6, scale=HD**-0.5)
            # per-head magnitudes spread around the clips
            pq = torch.randn(B, fq) * torch.rand(B, fq).mul(4.0)
            k = torch.randn(B, kd) * 2.5
            v = torch.randn(B, kd) * 3.0
            tau = torch.rand(B) + 1.0
            up = lambda t, dt, sh: ttnn.from_torch(t.reshape(sh), dtype=dt, layout=ttnn.TILE_LAYOUT,  # noqa: E731
                                                   device=mesh, memory_config=ttnn.L1_MEMORY_CONFIG)
            kb, vb = k.bfloat16().float(), v.bfloat16().float()
            outs = AttnPrep(w, B, kv_mc)(up(pq, ttnn.float32, (1, 1, B, fq)), up(k, ttnn.bfloat16, (1, 1, B, kd)),
                                         up(v, ttnn.bfloat16, (1, 1, B, kd)), up(tau, ttnn.float32, (1, B, 1, 1)))
            tf = tau[:, None, None] if not sliding else 1.0
            ref_q = l2clip(pq[:, :qd].reshape(B, NH, HD), 22.6) * tf
            ref_r = l2clip(pq[:, qd : qd + rd].reshape(B, NH, DR), 8.0) * tf
            ref_k = l2clip(kb.reshape(B, NKV, HD), 22.6)
            ref_v = vb.reshape(B, NKV, HD) * torch.rsqrt(vb.reshape(B, NKV, HD).pow(2).mean(-1, keepdim=True) + 1e-6)
            ref_s = (pq[:, qd + rd : qd + rd + NH] / w.scale)[:, :, None].expand(B, NH, 32)
            got = [ttnn.to_torch(t).float() for t in outs]
            got[2], got[3] = got[2][:, :, :NKV], got[3][:, :, :NKV]
            rpad = got[1][..., DR:].abs().max().item()
            got[1] = got[1][..., :DR]
            print(f"  r padding cols max |x| {rpad:.1e}", flush=True)
            worst = max(worst, rpad)
            for name, g, r in zip("q r k v s".split(), got, (ref_q, ref_r, ref_k, ref_v, ref_s)):
                g = g.reshape(r.shape)
                rel = ((g - r).abs().max() / r.abs().max()).item()
                worst = max(worst, rel)
                print(f"B={B} {'sliding' if sliding else 'global '} {name}: max rel err {rel:.2e}", flush=True)
            pad = ttnn.to_torch(outs[2]).float()[:, :, NKV:].abs().max().item()
            print(f"  k padding rows max |x| {pad:.1e}", flush=True)
            worst = max(worst, pad)
    ttnn.close_mesh_device(mesh)
    print(f"{'PASS' if worst < 1e-2 else 'FAIL'}: worst rel err {worst:.2e}")


if __name__ == "__main__":
    main()
