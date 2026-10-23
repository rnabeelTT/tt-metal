# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""On-device checks of the fused MoE decode ops (models/demos/blackhole/qwen36/tt/moe_decode) against torch.

    python test_moe_decode_ops.py            # all
    python test_moe_decode_ops.py topk
"""

import os
import sys

import torch

import ttnn
from models.demos.blackhole.qwen36.tt.moe_decode.experts import ExpertDown, ExpertGateUp, ExpertMoE, MoEDecodeBuffers
from models.demos.blackhole.qwen36.tt.moe_decode.topk import RouterTopK
from models.demos.blackhole.qwen36.tt.moe_decode.union import ExpertDownRows, ExpertGateUpRows, RouteRows, RowsBuffers, act_buffer

torch.manual_seed(0)
# experts per test tensor: 64 keeps every kernel path (E % 64 == 0) and the per-expert shapes, at a quarter of the
# host conversion time of the model's 256; TEST_E=256 for the full size
TEST_E = int(os.environ.get("TEST_E", "64"))
_WEIGHTS = {}


def expert_weights(dev, E, H, I, dtype, rounded):
    """(up_gate, shared_up_gate, down, shared_down) device tensors + their device-rounded torch copies, cached."""
    key = (E, H, I, dtype, rounded)
    if key not in _WEIGHTS:
        from models.demos.blackhole.qwen36.tt.moe_decode.fused import round_sig

        g = torch.Generator().manual_seed(1)
        rnd = round_sig if rounded else (lambda t: t)
        dev_t = [
            ttnn.from_torch(rnd(torch.randn(*shape, generator=g) * 0.03), dtype=dtype, layout=ttnn.TILE_LAYOUT,
                            device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            for shape in ((1, E, H, 2 * I), (1, 1, H, 2 * I), (1, E, I, H), (1, 1, I, H))
        ]
        _WEIGHTS[key] = (dev_t, [ttnn.to_torch(t).float() for t in dev_t])
    return _WEIGHTS[key]


def check_topk(dev, E=256, k=8, trials=20):
    op = RouterTopK(dev, E, k)
    ids, scores = op.alloc_outputs()
    worst = 0.0
    for t in range(trials):
        logits = torch.randn(1, 1, 32, E) * (0.5 + 3 * torch.rand(1))
        if t % 5 == 4:  # exact bf16 ties inside and at the edge of the top-k
            logits[0, 0, 0, 7] = logits[0, 0, 0, 200] = logits[0, 0, 0, :].max()
        lg = logits.bfloat16()
        x = ttnn.from_torch(lg, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.L1_MEMORY_CONFIG)
        op(x, ids, scores)
        got_ids = ttnn.to_torch(ids).reshape(-1)[:k].long()
        got_w = ttnn.to_torch(scores).float().reshape(-1)[:k]
        row = lg[0, 0, 0].float()
        # reference: descending, ties to the lower id
        order = sorted(range(E), key=lambda i: (-row[i].item(), i))[:k]
        ref_ids = torch.tensor(order)
        ref_w = torch.softmax(row[ref_ids], dim=-1)
        assert torch.equal(got_ids, ref_ids), (t, got_ids.tolist(), ref_ids.tolist())
        err = ((got_w - ref_w).abs() / ref_w.clamp_min(1e-3)).max().item()
        worst = max(worst, err)
        assert err < 1e-2, (t, got_w.tolist(), ref_w.tolist())
        ttnn.deallocate(x)
    print(f"topk E={E} k={k}: {trials} trials, ids exact, max rel weight err {worst:.2e}")


def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def round_sig(t, bits):
    """Round to nearest with `bits` significant bits (None: unchanged)."""
    if bits is None:
        return t
    m, e = torch.frexp(t.float())
    return torch.ldexp(torch.round(m * 2**bits) / 2**bits, e)


def check_experts(
    dev, H=2048, I=128, E=256, k=8, dtype=ttnn.bfloat8_b, gu_readers=4, dn_readers=2, trials=3, w_bits=None, x_bits=None,
    one_program=False,
):
    """Fused gate|up + down vs torch on the device-rounded weights (Ornith TP4 per-device shapes by default).
    w_bits / x_bits: pre-round the weights / the activation to that many significant bits (LoFi window probes)."""
    dram = ttnn.DRAM_MEMORY_CONFIG

    def up(t, dt=dtype):
        return ttnn.from_torch(round_sig(t, w_bits), dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=dram)

    def back(t):
        return ttnn.to_torch(t).float()

    s = 0.03
    ug = up(torch.randn(1, E, H, 2 * I) * s)
    ug_s = up(torch.randn(1, 1, H, 2 * I) * s)
    dn = up(torch.randn(1, E, I, H) * s)
    dn_s = up(torch.randn(1, 1, I, H) * s)
    W_ug, W_ugs, W_dn, W_dns = back(ug)[0], back(ug_s)[0, 0], back(dn)[0], back(dn_s)[0, 0]
    buf = MoEDecodeBuffers(dev, H, I, k)
    gu = ExpertGateUp(dev, H, I, E, k, dtype, readers=gu_readers)
    dn_op = ExpertDown(dev, H, I, E, k, dtype, readers=dn_readers)
    one = ExpertMoE(dev, H, I, E, k, dtype, gu_readers=gu_readers, dn_readers=dn_readers) if one_program else None
    for t in range(trials):
        xr = torch.zeros(1, 1, 32, H)
        xr[0, 0, 0] = torch.randn(H)
        xb = round_sig(xr, x_bits).bfloat16()
        lg = torch.zeros(1, 1, 32, E + 32)
        lg[0, 0, 0, : E + 1] = torch.randn(E + 1) * 2
        lgb = lg.bfloat16()
        x_tt = ttnn.from_torch(xb, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.L1_MEMORY_CONFIG)
        lg_tt = ttnn.from_torch(lgb, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.L1_MEMORY_CONFIG)
        if one is not None:
            one(x_tt, lg_tt, ug, ug_s, dn, dn_s, buf.act, buf.out)
            gu(x_tt, lg_tt, ug, ug_s, buf.act, buf.route)  # the route record for the id check
        else:
            gu(x_tt, lg_tt, ug, ug_s, buf.act, buf.route)
            dn_op(buf.act, buf.route, dn, dn_s, buf.out)
        route = ttnn.to_torch(buf.route).reshape(-1).long()
        act = back(buf.act)[0, 0]
        out = back(buf.out)[0, 0, 0]
        # reference
        row = lgb[0, 0, 0, :E].float()
        ids = torch.tensor(sorted(range(E), key=lambda i: (-row[i].item(), i))[:k])
        w = torch.cat([torch.sigmoid(lgb[0, 0, 0, E].float()).reshape(1), torch.softmax(row[ids], -1)])
        x = xb[0, 0, 0].float()
        Wg = [W_ugs] + [W_ug[e] for e in ids]
        Wd = [W_dns] + [W_dn[e] for e in ids]
        ref_act = torch.stack([w[i] * torch.nn.functional.silu(x @ Wg[i][:, I:]) * (x @ Wg[i][:, :I]) for i in range(k + 1)])
        ref_out = sum(ref_act[i] @ Wd[i] for i in range(k + 1))
        assert torch.equal(route[:k], ids), (route[:k].tolist(), ids.tolist())
        print(
            f"{'one-program ' if one_program else ''}w_bits {w_bits} x_bits {x_bits} experts trial {t}: ids exact, act pcc {pcc(act, ref_act):.5f}, out pcc {pcc(out, ref_out):.5f}, "
            f"out rel err {((out - ref_out).norm() / ref_out.norm()).item():.4f}, "
            f"scale act {(act * ref_act).sum().item() / (ref_act * ref_act).sum().item():.4f} "
            f"out {(out * ref_out).sum().item() / (ref_out * ref_out).sum().item():.4f}"
        )
        for tt_ in (x_tt, lg_tt):
            ttnn.deallocate(tt_)


def check_rows(dev, rows, H=2048, I=128, E=None, k=8, dtype=ttnn.bfloat8_b, trials=2):
    """Multi-row fused MoE (RouteRows -> ExpertGateUpRows -> ExpertDownRows) vs torch per row, weights pre-rounded to
    5 significant bits (as the model stores them)."""
    E = E or TEST_E
    back = lambda t: ttnn.to_torch(t).float()
    (ug, ug_s, dn, dn_s), (W_ug, W_ugs, W_dn, W_dns) = expert_weights(dev, E, H, I, dtype, True)
    W_ug, W_ugs, W_dn, W_dns = W_ug[0], W_ugs[0, 0], W_dn[0], W_dns[0, 0]
    buf = RowsBuffers(dev, H, I, E, k, rows, dtype, act=act_buffer(dev, I, E, k, rows))
    route = RouteRows(dev, E, k, rows)
    gu = ExpertGateUpRows(dev, H, I, E, k, dtype, rows)
    dn_op = ExpertDownRows(dev, H, I, E, k, dtype, rows)
    for t in range(trials):
        xr = torch.zeros(1, 1, 32, H)
        xr[0, 0, :rows] = torch.randn(rows, H)
        xb = xr.bfloat16()
        lg = torch.zeros(1, 1, 32, E + 32)
        lg[0, 0, :rows, : E + 1] = torch.randn(rows, E + 1) * 2
        lgb = lg.bfloat16()
        x_tt = ttnn.from_torch(xb, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.L1_MEMORY_CONFIG)
        lg_tt = ttnn.from_torch(lgb, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.L1_MEMORY_CONFIG)
        route(lg_tt, buf.table)
        ttnn.synchronize_device(dev)
        print(f"rows {rows}: route done", flush=True)
        gu(x_tt, buf.table, buf.identity, ug, ug_s, buf.act)
        ttnn.synchronize_device(dev)
        print(f"rows {rows}: gate|up done", flush=True)
        dn_op(buf.act, buf.table, dn, dn_s, buf.out)
        ttnn.synchronize_device(dev)
        print(f"rows {rows}: down done", flush=True)
        out = back(buf.out)[0, 0, :rows]
        table = ttnn.to_torch(buf.table).reshape(rows, 32).long()
        worst = 1.0
        for r in range(rows):
            row = lgb[0, 0, r, :E].float()
            ids = torch.tensor(sorted(range(E), key=lambda i: (-row[i].item(), i))[:k])
            assert torch.equal(table[r, :k], ids), (r, table[r, :k].tolist(), ids.tolist())
            w = torch.cat([torch.sigmoid(lgb[0, 0, r, E].float()).reshape(1), torch.softmax(row[ids], -1)])
            x = xb[0, 0, r].float()
            Wg = [W_ugs] + [W_ug[e] for e in ids]
            Wd = [W_dns] + [W_dn[e] for e in ids]
            ref = sum(
                (w[i] * torch.nn.functional.silu(x @ Wg[i][:, I:]) * (x @ Wg[i][:, :I])) @ Wd[i] for i in range(k + 1)
            )
            worst = min(worst, pcc(out[r], ref))
        print(f"rows {rows} trial {t}: ids exact, worst row out pcc {worst:.5f}")
        if worst < 0.99:  # debug: the activation buffer per union slot vs torch
            mp = gu.trows
            raw = ttnn.to_torch(buf.act).float().reshape(-1, mp * 32)  # page = slot * pairs + j: [face0 mp x 16 | face1]
            pairs = I // 32
            sel = [sorted(range(E), key=lambda i: (-lgb[0, 0, r, i].item(), i))[:k] for r in range(rows)]
            uni = sorted(set(e for r in range(rows) for e in sel[r]))
            for sl in range(min(4, len(uni) + 1)):
                e = None if sl == 0 else uni[sl - 1]
                got = torch.zeros(rows, I)
                for j in range(pairs):
                    pg = raw[sl * pairs + j]
                    for r in range(rows):
                        got[r, 32 * j : 32 * j + 16] = pg[16 * r : 16 * r + 16]
                        got[r, 32 * j + 16 : 32 * j + 32] = pg[16 * mp + 16 * r : 16 * mp + 16 * r + 16]
                W = W_ugs if e is None else W_ug[e]
                ref = torch.zeros(rows, I)
                for r in range(rows):
                    row = lgb[0, 0, r, :E].float()
                    ids = sel[r]
                    sw = torch.softmax(row[torch.tensor(ids)], -1)
                    w = torch.sigmoid(lgb[0, 0, r, E].float()) if e is None else (sw[ids.index(e)] if e in ids else 0.0)
                    x = xb[0, 0, r].float()
                    ref[r] = w * torch.nn.functional.silu(x @ W[:, I:]) * (x @ W[:, :I])
                print(f"   slot {sl} expert {e}: pcc {pcc(got, ref):.4f} scale {(got * ref).sum().item() / max((ref * ref).sum().item(), 1e-12):.3f} "
                      f"row-norms got {[round(v, 3) for v in got.norm(dim=1).tolist()]} ref {[round(v, 3) for v in ref.norm(dim=1).tolist()]}")
        for tt_ in (x_tt, lg_tt):
            ttnn.deallocate(tt_)


def time_experts(
    dev, H=2048, I=128, E=256, k=8, dtype=ttnn.bfloat8_b, variants=((4, 2), (3, 2), (4, 4), (8, 2)), prof=False
):
    """Trace-replay time per call of the gate|up and down ops (and both), per (gu_readers, dn_readers)."""
    import time

    dram = ttnn.DRAM_MEMORY_CONFIG
    up = lambda t: ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=dram)
    ug, ug_s = up(torch.randn(1, E, H, 2 * I) * 0.03), up(torch.randn(1, 1, H, 2 * I) * 0.03)
    dn, dn_s = up(torch.randn(1, E, I, H) * 0.03), up(torch.randn(1, 1, I, H) * 0.03)
    buf = MoEDecodeBuffers(dev, H, I, k)
    xr = torch.zeros(1, 1, 32, H)
    xr[0, 0, 0] = torch.randn(H)
    x = ttnn.from_torch(xr, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.L1_MEMORY_CONFIG)
    lg = ttnn.from_torch(torch.randn(1, 1, 32, E + 32), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                         memory_config=ttnn.L1_MEMORY_CONFIG)

    def bench(fn, reps=20, iters=20):
        fn()
        tid = ttnn.begin_trace_capture(dev, cq_id=0)
        for _ in range(reps):
            fn()
        ttnn.end_trace_capture(dev, tid, cq_id=0)
        ttnn.execute_trace(dev, tid, cq_id=0, blocking=True)
        t0 = time.time()
        for _ in range(iters):
            ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(dev)
        ttnn.release_trace(dev, tid)
        return (time.time() - t0) / (reps * iters) * 1e6

    for gr, dr in variants:
        gu = ExpertGateUp(dev, H, I, E, k, dtype, readers=gr)
        dn_op = ExpertDown(dev, H, I, E, k, dtype, readers=dr)
        one = ExpertMoE(dev, H, I, E, k, dtype, gu_readers=gr, dn_readers=dr)
        if prof:
            for _ in range(3):
                gu(x, lg, ug, ug_s, buf.act, buf.route)
                dn_op(buf.act, buf.route, dn, dn_s, buf.out)
            for _ in range(3):
                one(x, lg, ug, ug_s, dn, dn_s, buf.act, buf.out)
            ttnn.synchronize_device(dev)
            ttnn.ReadDeviceProfiler(dev)
            continue
        t_gu = bench(lambda: gu(x, lg, ug, ug_s, buf.act, buf.route))
        t_dn = bench(lambda: dn_op(buf.act, buf.route, dn, dn_s, buf.out))
        t_both = bench(lambda: (gu(x, lg, ug, ug_s, buf.act, buf.route), dn_op(buf.act, buf.route, dn, dn_s, buf.out)))
        t_one = bench(lambda: one(x, lg, ug, ug_s, dn, dn_s, buf.act, buf.out))
        print(f"time gu_readers {gr} ({len(gu.assign)} cores) dn_readers {dr} ({len(dn_op.assign)} cores): "
              f"gate|up {t_gu:5.1f} us, down {t_dn:5.1f} us, both {t_both:5.1f} us, one program {t_one:5.1f} us")


def main():
    what = sys.argv[1:] or ["topk", "experts"]
    dev = ttnn.open_device(device_id=0, trace_region_size=32 << 20)
    try:
        if "topk" in what:
            check_topk(dev)
            check_topk(dev, E=512, k=10)
        if "profrows" in what:  # eager multi-row ops for the device profiler: route / gate|up / down per row count
            I = int(os.environ.get("TEST_I", "128"))
            for m in [int(v) for v in os.environ.get("ROWS_LIST", "2,4,8").split(",")]:
                (ug, ug_s, dn, dn_s), _ = expert_weights(dev, TEST_E, 2048, I, ttnn.bfloat8_b, True)
                buf = RowsBuffers(dev, 2048, I, TEST_E, 8, m, ttnn.bfloat8_b, act=act_buffer(dev, I, TEST_E, 8, m))
                ops = (RouteRows(dev, TEST_E, 8, m), ExpertGateUpRows(dev, 2048, I, TEST_E, 8, ttnn.bfloat8_b, m),
                       ExpertDownRows(dev, 2048, I, TEST_E, 8, ttnn.bfloat8_b, m))
                xr = torch.zeros(1, 1, 32, 2048)
                xr[0, 0, :m] = torch.randn(m, 2048)
                x = ttnn.from_torch(xr, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.L1_MEMORY_CONFIG)
                lg = ttnn.from_torch(torch.randn(1, 1, 32, TEST_E + 32), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                     device=dev, memory_config=ttnn.L1_MEMORY_CONFIG)
                for _ in range(2):
                    ops[0](lg, buf.table)
                    ops[1](x, buf.table, buf.identity, ug, ug_s, buf.act)
                    ops[2](buf.act, buf.table, dn, dn_s, buf.out)
                ttnn.synchronize_device(dev)
            ttnn.ReadDeviceProfiler(dev)
        if "prof" in what:  # eager calls for the device profiler (TT_METAL_DEVICE_PROFILER=1)
            time_experts(dev, variants=((4, 2),), prof=True)
        if "rows" in what:
            for m in [int(v) for v in os.environ.get("ROWS_LIST", "2,3,4,8").split(",")]:
                check_rows(dev, m)
        if "tp2" in what:  # p300 (2 chips) / Qwen3.5-122B on 4 chips: 256 expert columns per device, strided up|gate reads
            check_experts(dev, I=256, trials=1)
            check_rows(dev, 4, I=256, trials=1)
            check_rows(dev, 16, I=256, trials=1)
        if "cn" in what:  # Qwen3-Coder-Next shapes: 512 experts, top-10, 256 expert columns per device (TP2)
            for m in [int(v) for v in os.environ.get("ROWS_LIST", "1,2,8,16,32").split(",")]:
                check_rows(dev, m, I=256, E=512, k=10, trials=1)
        if "time" in what:
            time_experts(dev)
        if "experts" in what:
            check_experts(dev)
            check_experts(dev, one_program=True)
            if "probe" in what:
                for wb, xb in ((5, None), (6, None), (5, 7), (6, 7)):
                    check_experts(dev, trials=1, w_bits=wb, x_bits=xb)
            if "bfp4" in what:
                check_experts(dev, dtype=ttnn.bfloat4_b)
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
