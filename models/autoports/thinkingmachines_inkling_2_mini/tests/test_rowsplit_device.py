# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""I2M_PF_ROWSPLIT pieces on one die: (1) the packed conv with a row offset (sconv_kernels/pfseg_reader.cpp PF_ROWOFF)
for each of N emulated dies vs the full-rows kernel: output rows and every lane's written states bitwise; (2) the
prefill glue ops on T rows vs on each T / N row slice: bitwise. env: ROWS (default 2048), NSPLIT (default 4)."""

import os

import torch

import ttnn
from models.autoports.thinkingmachines_inkling_2_mini.tt.model import MessageGate, Ops, _lane_bufs, _lane_copy
from models.autoports.thinkingmachines_inkling_2_mini.tt.sconv import ShortConv

H, EPS, RANK = 2304, 1e-6, 64


def bits(t):
    a = ttnn.to_torch(t)
    return a.view(torch.int16) if a.dtype == torch.bfloat16 else a.float().view(torch.int32)


def main():
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    T = int(os.environ.get("ROWS", "2048"))
    NS = int(os.environ.get("NSPLIT", "4"))
    Tl = T // NS
    g = torch.Generator().manual_seed(0)
    DR = ttnn.DRAM_MEMORY_CONFIG
    up = lambda t, dt: ttnn.from_torch(t.contiguous(), dtype=dt, layout=ttnn.TILE_LAYOUT, device=mesh,  # noqa: E731
                                       memory_config=DR)
    ok = True

    def check(name, full, parts):
        nonlocal ok
        a = bits(full)
        same = all(torch.equal(a[..., d * Tl:(d + 1) * Tl, :], bits(p)) for d, p in enumerate(parts))
        ok &= same
        print(f"{name}: {'bitwise' if same else 'DIFF'}", flush=True)

    rows = lambda t, d: ttnn.slice(t, [0, 0, d * Tl, 0], [1, 1, (d + 1) * Tl, int(t.shape[-1])])  # noqa: E731

    # (1) the packed conv with a row offset
    w = (torch.randn(H, 4, generator=g) * 0.5).to(torch.bfloat16).float()
    cv = _lane_bufs(ShortConv(mesh, w, None, H))
    lanes = [cv, _lane_copy(cv), _lane_copy(cv)]
    st0 = [torch.randn(1, 1, 3, H, generator=g).to(torch.bfloat16).float() for _ in lanes]
    up_rm = lambda t: ttnn.from_torch(t, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh)  # noqa: E731
    st0_tt = [up_rm(s) for s in st0]
    # segments crossing the slice boundaries; commit rows inside; a snapshot row on lane 1 (64-row aligned)
    a, b = (5 * T) // 16 // 64 * 64, (3 * T) // 4 // 64 * 64
    segs_h = [(0, a, a - 40, 0), (a, b - a, b - a, 1), (b, T - b, T - b - 1, 2)]
    x = up(torch.randn(1, 1, T, H, generator=g), ttnn.bfloat16)
    variants = [("resid fp32 -> bf16", ttnn.float32, ttnn.bfloat16), ("resid bf16 -> fp32", ttnn.bfloat16, ttnn.float32),
                ("no resid", None, ttnn.bfloat16)]
    for snap in (None, 37):
        for vname, rdt, odt in variants:
            resid = up(torch.randn(1, 1, T, H, generator=g) * 4, rdt) if rdt is not None else None

            def run(rs, rr):
                for ln, s in zip(lanes, st0_tt):
                    ttnn.copy(s, ln.st)
                    ln.snap_row = None
                lanes[1].snap_row = snap
                segs = [(r0, R, n, lanes[li]) for r0, R, n, li in segs_h]
                y = cv.prefill_packed(x, segs, resid=rr, out_dtype=odt, rs=rs)
                return y, [bits(ln.st) for ln in lanes] + ([bits(lanes[1].snap_buf)] if snap is not None else [])

            yf, sf = run(None, resid)
            yp = []
            for d in range(NS):
                k0 = torch.zeros(1, 1, 1, 16, dtype=torch.int32)
                k0[0, 0, 0, 0] = d * Tl // 32
                k0t = ttnn.from_torch(k0, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, memory_config=DR)
                y, s = run((k0t, Tl), rows(resid, d) if resid is not None else None)
                same_st = all(torch.equal(u, v) for u, v in zip(s, sf))
                ok &= same_st
                if not same_st:
                    print(f"conv {vname} snap {snap}: die {d} states DIFF", flush=True)
                yp.append(y)
            check(f"conv {vname} snap {snap}", yf, yp)
            # PF_RSX: x only this die's rows + the side tiles every die contributes (RowSplit.slots / side, emulated)
            from models.autoports.thinkingmachines_inkling_2_mini.tt.model import RowSplit

            rsh = RowSplit.__new__(RowSplit)
            rsh.N, rsh.T, rsh.Tl = NS, T, Tl
            slots = rsh.slots([(r0, n, snap if li == 1 else None) for r0, R, n, li in segs_h])
            xh = ttnn.to_torch(x)
            Rt = Tl // 32
            A = torch.cat([xh[..., (o * Rt + l) * 32:(o * Rt + l + 1) * 32, :] for o in range(NS) for l in slots], 2)
            A_tt = up(A.float(), ttnn.bfloat16)
            yp = []
            for d in range(NS):
                k0 = torch.zeros(1, 1, 1, 16, dtype=torch.int32)
                k0[0, 0, 0, 0] = d * Tl // 32
                k0t = ttnn.from_torch(k0, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, memory_config=DR)
                xl = rows(x, d)
                for ln, s0 in zip(lanes, st0_tt):
                    ttnn.copy(s0, ln.st)
                    ln.snap_row = None
                lanes[1].snap_row = snap
                segs = [(r0, R, n, lanes[li]) for r0, R, n, li in segs_h]
                y = cv.prefill_packed(xl, segs, resid=rows(resid, d) if resid is not None else None, out_dtype=odt,
                                      rs=(k0t, Tl, A_tt, slots))
                s = [bits(ln.st) for ln in lanes] + ([bits(lanes[1].snap_buf)] if snap is not None else [])
                same_st = all(torch.equal(u, v) for u, v in zip(s, sf))
                ok &= same_st
                if not same_st:
                    print(f"rsx conv {vname} snap {snap}: die {d} states DIFF", flush=True)
                yp.append(y)
            check(f"rsx conv {vname} snap {snap} slots {slots}", yf, yp)

    # (2) the glue ops
    ops = Ops(mesh, EPS)
    from models.autoports.thinkingmachines_inkling_2_mini.tt.mixw import MixKernel

    ops.mixk = MixKernel(mesh, H)
    assert ops.rown_ok(ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, T, H]), ttnn.float32, ttnn.TILE_LAYOUT, mesh, DR))
    xf = up(torch.randn(1, 1, T, H, generator=g) * torch.rand(1, 1, T, 1, generator=g) * 5, ttnn.float32)
    xb = up(torch.randn(1, 1, T, H, generator=g) * 3, ttnn.bfloat16)
    wn = up((torch.randn(1, 1, 1, H, generator=g) * 0.3 + 1.0).to(torch.bfloat16).float(), ttnn.float32)
    Q = ops.query(torch.randn(H, 3, generator=g) * H**-0.5)
    qrep = ops.qrep[id(Q)]
    sl = lambda t: [rows(t, d) for d in range(NS)]  # noqa: E731
    for nm, src in (("fp32", xf), ("bf16", xb)):
        for out in (ttnn.float32, ttnn.bfloat16):
            check(f"rown {nm} w -> {out}", ops.rown(src, wn, out_dtype=out), [ops.rown(s, wn, out_dtype=out) for s in sl(src)])
    o, lg = ops.rown(xf, None, qrep=qrep)
    parts = [ops.rown(s, None, qrep=qrep) for s in sl(xf)]
    check("rown + logits: norm", o, [p[0] for p in parts])
    check("rown + logits: logits", lg, [p[1] for p in parts])
    three = ops.rown.multi([dict(x=t, w=wn, out_dtype=ttnn.bfloat16) for t in (xf, xb, xf)])
    tp = [ops.rown.multi([dict(x=t, w=wn, out_dtype=ttnn.bfloat16) for t in (s0, s1, s0)]) for s0, s1 in zip(sl(xf), sl(xb))]
    for i in range(3):
        check(f"rown multi {i}", three[i], [p[i] for p in tp])
    for width in (32, 96, 480, 1536):
        qc = up(torch.randn(1, 1, H, width, generator=g) * H**-0.5, ttnn.float32)
        mm = lambda t: ttnn.matmul(t, qc, compute_kernel_config=ops.ckc, dtype=ttnn.float32, memory_config=DR)  # noqa: E731
        check(f"logits matmul x {width}", mm(xf), [mm(s) for s in sl(xf)])
    # mix: 4 blocks (logits as (tensor, offset, stride) against 5 queries) + the current source
    srcs = [up(torch.randn(1, 1, T, H, generator=g), ttnn.float32) for _ in range(4)] + [xf]
    lgs = [(up(torch.randn(1, 1, T, 5 * 32, generator=g), ttnn.float32), 2, 5) for _ in range(4)] \
        + [up(torch.randn(1, 1, T, 3 * 32, generator=g), ttnn.float32)]
    full = ops.mixk(srcs, lgs, 5, 3, out_mc=DR)
    sp = [ops.mixk([rows(s, d) for s in srcs], [(rows(t[0], d), t[1], t[2]) if isinstance(t, tuple) else rows(t, d)
                                                for t in lgs], 5, 3, out_mc=DR) for d in range(NS)]
    for i in range(3):
        check(f"mix query {i}", full[i], [p[i] for p in sp])

    class _Ck:  # a message gate's weights
        def __init__(self):
            self.w = {"down_weight": torch.randn(RANK, 2 * H, generator=g) * 0.02,
                      "forget_weight": torch.randn(H, RANK, generator=g) * 0.1,
                      "gain_weight": torch.randn(H, RANK, generator=g) * 0.1}

        def L(self, li, n):
            return self.w[n.split(".")[-1]].to(torch.bfloat16)

    gate = MessageGate(ops, _Ck(), 0, "attn")
    m = up(torch.randn(1, 1, T, H, generator=g), ttnn.float32)
    check("message gate", gate(xf, m), [gate(s, t) for s, t in zip(sl(xf), sl(m))])

    print("PASS" if ok else "FAIL", flush=True)
    print("== rowsplit done", flush=True)
    from models.autoports.thinkingmachines_inkling_2_mini.tt.clean_exit import close

    close(mesh)


if __name__ == "__main__":
    main()
