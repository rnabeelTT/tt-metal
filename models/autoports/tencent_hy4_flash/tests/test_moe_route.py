# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Router top-k in tt/router_post (RP_ROUTE), the one-row one-bank gate|up (qwen36 moe_decode ExpertGateUp1B) and the
routed-scale fold (route weights x 2^wexp) vs the HEAD path, one chip, Hy4's per-die shapes (latent 1536, expert width
384, 512 experts, top-16, bfp4, routed only, "keys" router), random weights:
  1 row:  router_post -> [keys | s] -> ExpertGateUp (router tail) -> ExpertDown   (HEAD)
      vs  router_post(route=) -> ExpertGateUp(ROUTE_IN) -> ExpertDown           (route record + act + out bitwise)
      vs  router_post(route=) -> ExpertGateUp1B -> ExpertDown                   (act + out bitwise)
      vs  wexp=2 -> out == ttnn.multiply(out_HEAD, 3.998)                       (bitwise)
  M rows: RouteRows(wexp=2) -> ExpertGateUpRows -> ExpertDownRows == ttnn.multiply(HEAD out, 3.998), and router_post's
          route table == RouteRows' (bitwise, used fields).
Prints per-call trace-replay us. usage: python -m models.autoports.tencent_hy4_flash.tests.test_moe_route"""

import os
import time

import torch
import ttnn

from models.autoports.tencent_hy4_flash.tt.router_post import RouterPostOp
from models.demos.blackhole.qwen36.tt.moe_decode import experts as X
from models.demos.blackhole.qwen36.tt.moe_decode import union as U

H, I, E, K = 1536, 384, 512, 16
DT = ttnn.bfloat4_b
SCALE = 3.998


def timed(dev, fn, n=20):
    fn()
    ttnn.synchronize_device(dev)
    tid = ttnn.begin_trace_capture(dev, cq_id=0)
    for _ in range(n):
        fn()
    ttnn.end_trace_capture(dev, tid, cq_id=0)
    ttnn.execute_trace(dev, tid, cq_id=0, blocking=True)
    t0 = time.time()
    ttnn.execute_trace(dev, tid, cq_id=0, blocking=True)
    dt = (time.time() - t0) / n * 1e6
    ttnn.release_trace(dev, tid)
    return dt


def main():
    dev = ttnn.open_device(device_id=int(os.environ.get("TDEV", "0")), l1_small_size=24576, trace_region_size=64 << 20)
    os.environ["MOE_SWIGLU_LIMIT"] = "10.0"
    os.environ.setdefault("QWEN35_MOE_GU1B", "2-16")
    os.environ.setdefault("QWEN35_MOE_GU1B_READERS", "2:2,5:3,9:4")
    os.environ.setdefault("QWEN35_MOE_GU1B_ONROW", "2:2,5:3,9:4")
    ok = True
    try:
        g = torch.Generator().manual_seed(0)
        dram, l1 = ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG
        up = lambda t: ttnn.from_torch(t, dtype=DT, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=dram)  # noqa
        ug = up(torch.randn(1, E, H, 2 * I, generator=g) * 0.03)
        dn = up(torch.randn(1, E, I, H, generator=g) * 0.03)
        bias_h = torch.randn(E, generator=g) * 0.05
        bias = U.upload_selection_bias(dev, bias_h)
        f32 = lambda t, mem=l1: ttnn.from_torch(t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev,  # noqa
                                                 memory_config=mem)
        bias_rows = f32(bias_h.reshape(1, 1, 1, E).expand(1, 1, 32, E).contiguous(), dram)
        a = f32(torch.randn(1, 1, 32, E, generator=g) * 2.0)
        b = f32(torch.randn(1, 1, 32, E, generator=g) * 0.01)
        rp = RouterPostOp.get(dev)

        # ---- scalar multiply: BF16 x 3.998 == x 4 bit for bit? ----
        xr = torch.randn(1, 1, 32, H, generator=g)
        xt = ttnn.from_torch(xr, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=dram)
        m1 = ttnn.to_torch(ttnn.multiply(xt, SCALE, memory_config=dram))
        same4 = bool(torch.equal(m1, xr.to(torch.bfloat16) * 4))
        print(f"SCALAR: ttnn.multiply(bf16, {SCALE}) == x * 4 bitwise: {same4}", flush=True)
        ok &= same4

        # ---- one row ----
        x = ttnn.from_torch(torch.randn(1, 1, 1, H, generator=g), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                            device=dev, memory_config=dram)
        bufs = X.MoEDecodeBuffers(dev, H, I, K)
        gu = X.ExpertGateUp(dev, H, I, E, K, DT, readers=4, router="keys", shared=False)
        dop = X.ExpertDown(dev, H, I, E, K, DT, shared=False)
        gu1b = X.ExpertGateUp1B(dev, H, I, E, K, DT, readers=int(os.environ.get("GU1B1_READERS", "2")),
                                on_row=int(os.environ.get("GU1B1_ONROW", "2")))
        res, times = {}, {}

        def head():
            row = rp(a, b, bias_rows, 1, l1)
            gu(x, row, ug, ug, bufs.act, bufs.route, bias=bias)
            ttnn.deallocate(row)
            dop(bufs.act, bufs.route, dn, dn, bufs.out)

        def rin(op, wexp=0):
            def f():
                row = rp(a, b, bias_rows, 1, l1, route=bufs.route, wexp=wexp)
                ttnn.deallocate(row)
                op(x, bufs.route, ug, ug, bufs.act, bufs.route, bias=bias, route_in=True)
                dop(bufs.act, bufs.route, dn, dn, bufs.out)
            return f

        for name, fn in (("head", head), ("route_in", rin(gu)), ("gu1b", rin(gu1b)), ("gu1b_w4", rin(gu1b, 2))):
            fn()
            ttnn.synchronize_device(dev)
            res[name] = (ttnn.to_torch(bufs.route).reshape(-1)[:64].clone(),
                         ttnn.to_torch(bufs.act).reshape(K + 1, I)[1:].clone(),
                         ttnn.to_torch(bufs.out).reshape(-1, H)[:1].clone())
            times[name] = timed(dev, fn)
        r0, a0, o0 = res["head"]
        used = list(range(0, 33))
        for name in ("route_in", "gu1b"):
            r1, a1, o1 = res[name]
            sr = bool(torch.equal(r0[used], r1[used]))
            sa, so = bool(torch.equal(a0, a1)), bool(torch.equal(o0, o1))
            ok &= sr and sa and so
            print(f"R=1 {name}: route record {sr}, act {sa}, out {so} bitwise vs head", flush=True)
        o4 = res["gu1b_w4"][2]
        ref4 = ttnn.to_torch(ttnn.multiply(ttnn.from_torch(o0.reshape(1, 1, 1, H), dtype=ttnn.bfloat16,
                                                           layout=ttnn.TILE_LAYOUT, device=dev), SCALE))
        s4 = bool(torch.equal(o4, ref4.reshape(1, H)))
        ok &= s4
        fin = bool(torch.isfinite(o0).all()) and float(o0.abs().max()) > 0
        print(f"R=1 wexp=2: out == multiply(out_head, {SCALE}) bitwise {s4}; head finite/nonzero {fin}", flush=True)
        print("R=1 us per call (router_post + gate|up + down): " + ", ".join(f"{k} {v:.1f}" for k, v in times.items()),
              flush=True)

        # ---- one-row components (trace replay us per call) ----
        row1 = rp(a, b, bias_rows, 1, l1)
        comp = {
            "rp": lambda: ttnn.deallocate(rp(a, b, bias_rows, 1, l1)),
            "rp_route": lambda: ttnn.deallocate(rp(a, b, bias_rows, 1, l1, route=bufs.route)),
            "gu_tail": lambda: gu(x, row1, ug, ug, bufs.act, bufs.route, bias=bias),
            "gu_route_in": lambda: gu(x, bufs.route, ug, ug, bufs.act, bufs.route, bias=bias, route_in=True),
            "dn": lambda: dop(bufs.act, bufs.route, dn, dn, bufs.out),
        }
        for rd, onr in ((2, 2), (3, 2), (3, 3), (4, 2), (4, 4)):
            op = X.ExpertGateUp1B(dev, H, I, E, K, DT, readers=rd, on_row=onr)
            comp[f"gu1b_r{rd}o{onr}"] = (lambda op=op: op(x, bufs.route, ug, ug, bufs.act, bufs.route, route_in=True))
        ok_c = True
        for name, fn in comp.items():
            t = timed(dev, fn)
            if name.startswith("gu1b"):
                same = bool(torch.equal(ttnn.to_torch(bufs.act).reshape(K + 1, I)[1:], a0))
                ok_c &= same
                name += f"({'exact' if same else 'DIFF'})"
            print(f"R=1 component {name}: {t:.1f} us", flush=True)
        ok &= ok_c
        ttnn.deallocate(row1)
        for R in (2, 4, 8, 16):
            for G in (1, 2, 4, 6):
                if G <= R:
                    rt_ = U.RowsBuffers(dev, H, I, E, K, R, DT, act=None).table
                    t = timed(dev, lambda: ttnn.deallocate(rp(a, b, bias_rows, R, l1, route=rt_, groups=G)))
                    print(f"R={R} component rp_route G={G}: {t:.1f} us", flush=True)
            t = timed(dev, lambda: ttnn.deallocate(rp(a, b, bias_rows, R, l1)))
            rr_ = U.RouteRows(dev, E, K, R, router="keys")
            rowR = rp(a, b, bias_rows, R, l1)
            t2 = timed(dev, lambda: rr_(rowR, rt_, bias=bias))
            print(f"R={R} component rp {t:.1f} us, RouteRows {t2:.1f} us", flush=True)
            ttnn.deallocate(rowR)

        # ---- M rows: RouteRows wexp, router_post route table ----
        for R in (2, 4, 8, 16):
            xR = ttnn.from_torch(torch.randn(1, 1, R, H, generator=g), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                 device=dev, memory_config=dram)
            act = U.act_buffer(dev, I, E, K, R)
            rb = U.RowsBuffers(dev, H, I, E, K, R, DT, act=act)
            route = U.RouteRows(dev, E, K, R, router="keys")
            gur = U.ExpertGateUpRows(dev, H, I, E, K, DT, R, shared=False)
            dnr = U.ExpertDownRows(dev, H, I, E, K, DT, R, shared=False)
            outs, tabs, tm = {}, {}, {}
            for name, wexp, via_rp in (("head", 0, False), ("w4", 2, False), ("rp", 0, True)):
                def f(wexp=wexp, via_rp=via_rp):
                    if via_rp:
                        row = rp(a, b, bias_rows, R, l1, route=rb.table, wexp=wexp)
                    else:
                        row = rp(a, b, bias_rows, R, l1)
                        route(row, rb.table, bias=bias, wexp=wexp)
                    ttnn.deallocate(row)
                    gur(xR, rb.table, rb.identity, ug, ug, act)
                    dnr(act, rb.table, dn, dn, rb.out)
                f()
                ttnn.synchronize_device(dev)
                outs[name] = ttnn.to_torch(rb.out).reshape(-1, H)[:R].clone()
                tabs[name] = ttnn.to_torch(rb.table).reshape(R, -1)[:, :33].clone()
                tm[name] = timed(dev, f)
            ref = ttnn.to_torch(ttnn.multiply(ttnn.from_torch(outs["head"].reshape(1, 1, R, H), dtype=ttnn.bfloat16,
                                                              layout=ttnn.TILE_LAYOUT, device=dev), SCALE)).reshape(R, H)
            s4 = bool(torch.equal(outs["w4"], ref))
            st = bool(torch.equal(tabs["rp"], tabs["head"]))
            so = bool(torch.equal(outs["rp"], outs["head"]))
            ok &= s4 and st and so
            print(f"R={R}: wexp=2 out == multiply(head, {SCALE}) {s4}; router_post table == RouteRows {st}, out {so}; "
                  f"us head {tm['head']:.1f} w4 {tm['w4']:.1f} rp {tm['rp']:.1f}", flush=True)
            ttnn.deallocate(act)
    finally:
        ttnn.close_device(dev)
    print("TEST_MOE_ROUTE", "PASS" if ok else "FAIL", flush=True)


if __name__ == "__main__":
    main()
