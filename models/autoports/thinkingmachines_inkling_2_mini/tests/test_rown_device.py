# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""tt/rown.py RowNorm against the op path ([typecast ->] ttnn.rms_norm [-> typecast]): bitwise, and timed.
env: ROWS (comma list, default 64,512,2048)"""

import os
import time

import torch

import ttnn
from models.autoports.thinkingmachines_inkling_2_mini.tests._devtime import devtime
from models.autoports.thinkingmachines_inkling_2_mini.tt.rown import RowNorm

H, EPS = 2304, 1e-6


def main():
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    ck = ttnn.init_device_compute_kernel_config(mesh.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
                                                math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False)
    up = lambda t, dt: ttnn.from_torch(t, dtype=dt, layout=ttnn.TILE_LAYOUT, device=mesh,  # noqa: E731
                                       memory_config=ttnn.DRAM_MEMORY_CONFIG)
    rn = RowNorm(mesh, H, EPS)
    g = torch.Generator().manual_seed(0)
    w_h = (torch.randn(1, 1, 1, H, generator=g) * 0.3 + 1.0).to(torch.bfloat16).float()
    w = up(w_h, ttnn.float32)
    ok = True
    for T in [int(v) for v in os.environ.get("ROWS", "64,512,2048").split(",")]:
        xh = torch.randn(1, 1, T, H, generator=g) * torch.rand(1, 1, T, 1, generator=g) * 5
        for in_dt in (ttnn.float32, ttnn.bfloat16):
            x = up(xh, in_dt)
            for gam in (None, w):
                for out_dt in (ttnn.float32, ttnn.bfloat16):
                    xf = ttnn.typecast(x, ttnn.float32) if in_dt != ttnn.float32 else x
                    ref = ttnn.rms_norm(xf, epsilon=EPS, weight=gam, compute_kernel_config=ck)
                    if out_dt != ttnn.float32:
                        ref = ttnn.typecast(ref, out_dt)
                    y = rn(x, gam, out_dtype=out_dt)
                    a, b = ttnn.to_torch(ref).float(), ttnn.to_torch(y).float()
                    same = torch.equal(a, b)
                    md = (a - b).abs().max().item()
                    ok &= same
                    print(f"T={T} in={in_dt} gamma={gam is not None} out={out_dt}: bitwise {same} maxdiff {md:.3g}",
                          flush=True)
        # the mix logits with the norm (qrep: each query in all 32 columns of its tile)
        for nq in (1, 3):
            qh = (torch.randn(H, nq, generator=g) * H**-0.5).repeat_interleave(32, dim=1).reshape(1, 1, H, nq * 32)
            qrep = up(qh, ttnn.float32)
            for in_dt in (ttnn.float32, ttnn.bfloat16):
                x = up(xh, in_dt)
                xf = ttnn.typecast(x, ttnn.float32) if in_dt != ttnn.float32 else x
                ref = ttnn.rms_norm(xf, epsilon=EPS, compute_kernel_config=ck)
                rlg = ttnn.matmul(ref, qrep, compute_kernel_config=ck, dtype=ttnn.float32,
                                  memory_config=ttnn.DRAM_MEMORY_CONFIG)
                y, lg = rn(x, None, qrep=qrep)
                s1 = torch.equal(ttnn.to_torch(ref).float(), ttnn.to_torch(y).float())
                a, b = ttnn.to_torch(rlg).float(), ttnn.to_torch(lg).float()
                s2 = torch.equal(a, b)
                ok &= s1 and s2
                ex = (ttnn.to_torch(y).double().reshape(T, H) @ qh.double().reshape(H, -1)).reshape(a.shape)
                print(f"T={T} nq={nq} in={in_dt} logits: out bitwise {s1}, logits bitwise {s2} maxdiff "
                      f"{(a - b).abs().max().item():.3g}; max err vs fp64: op path {(a - ex).abs().max().item():.3g}, "
                      f"RowNorm {(b - ex).abs().max().item():.3g}", flush=True)
                if in_dt == ttnn.bfloat16 and nq == 1:
                    d_new = devtime(mesh, lambda: rn(x, None, qrep=qrep))
                    d_old = devtime(mesh, lambda: ttnn.matmul(ttnn.rms_norm(ttnn.typecast(x, ttnn.float32), epsilon=EPS,
                                                                            compute_kernel_config=ck), qrep,
                                                              compute_kernel_config=ck, dtype=ttnn.float32,
                                                              memory_config=ttnn.DRAM_MEMORY_CONFIG))
                    print(f"T={T} bf16->f32+lg: RowNorm {d_new} us, op path {d_old} us (device)", flush=True)
        # two destinations: the halves of a [T, 2H] bf16 buffer
        cat = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, T, 2 * H]), ttnn.bfloat16, ttnn.TILE_LAYOUT, mesh,
                                             ttnn.DRAM_MEMORY_CONFIG)
        x = up(xh, ttnn.float32)
        x2 = up(xh * 0.5 + 1, ttnn.float32)
        rn(x, None, out_dtype=ttnn.bfloat16, dests=[(cat, 0)])
        rn(x2, None, out_dtype=ttnn.bfloat16, dests=[(cat, H // 32)])
        ref = ttnn.concat([ttnn.typecast(ttnn.rms_norm(t, epsilon=EPS, compute_kernel_config=ck), ttnn.bfloat16)
                           for t in (x, x2)], dim=-1)
        same = torch.equal(ttnn.to_torch(ref).float(), ttnn.to_torch(cat).float())
        ok &= same
        print(f"T={T} concat dests: bitwise {same}", flush=True)
        # timing: fp32 in + gamma + bf16 out vs rms_norm + typecast; bf16 in, no gamma, fp32 out vs typecast + rms
        for name, f_new, f_old in (
            ("f32+w->bf16", lambda: rn(x, w, out_dtype=ttnn.bfloat16),
             lambda: ttnn.typecast(ttnn.rms_norm(x, epsilon=EPS, weight=w, compute_kernel_config=ck), ttnn.bfloat16)),
            ("bf16->f32", lambda: rn(xb, None), lambda: ttnn.rms_norm(ttnn.typecast(xb, ttnn.float32), epsilon=EPS,
                                                                       compute_kernel_config=ck)),
            ("f32->f32", lambda: rn(x, None), lambda: ttnn.rms_norm(x, epsilon=EPS, compute_kernel_config=ck))):
            xb = up(xh, ttnn.bfloat16)
            dts = [devtime(mesh, f) for f in (f_new, f_old)]
            if dts[0] is not None:
                print(f"T={T} {name}: RowNorm {dts[0]:.1f} us, op path {dts[1]:.1f} us (device)", flush=True)
            res = []
            for f in (f_new, f_old):
                for _ in range(3):
                    f()
                ttnn.synchronize_device(mesh)
                t0 = time.perf_counter()
                for _ in range(50):
                    f()
                ttnn.synchronize_device(mesh)
                res.append((time.perf_counter() - t0) / 50 * 1e6)
            print(f"T={T} {name}: RowNorm {res[0]:.1f} us, op path {res[1]:.1f} us (wall per call)", flush=True)
    print("PASS" if ok else "FAIL", flush=True)
    print("== rown done", flush=True)
    from models.autoports.thinkingmachines_inkling_2_mini.tt.clean_exit import close
    close(mesh)


if __name__ == "__main__":
    main()
