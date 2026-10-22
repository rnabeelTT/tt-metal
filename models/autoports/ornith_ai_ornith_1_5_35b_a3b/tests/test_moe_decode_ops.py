# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""On-device checks of the fused MoE decode ops (models/demos/blackhole/qwen36/tt/moe_decode) against torch.

    python test_moe_decode_ops.py            # all
    python test_moe_decode_ops.py topk
"""

import sys

import torch

import ttnn
from models.demos.blackhole.qwen36.tt.moe_decode.experts import ExpertDown, ExpertGateUp, ExpertMoE, MoEDecodeBuffers
from models.demos.blackhole.qwen36.tt.moe_decode.topk import RouterTopK

torch.manual_seed(0)


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
        if "prof" in what:  # eager calls for the device profiler (TT_METAL_DEVICE_PROFILER=1)
            time_experts(dev, variants=((4, 2),), prof=True)
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
