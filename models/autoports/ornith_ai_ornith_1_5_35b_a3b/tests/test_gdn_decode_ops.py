"""On-device checks of the fused GDN decode mixer ops (models/demos/blackhole/qwen36/tt/gdn_decode) against torch FP32.

    python test_gdn_decode_ops.py              # Ornith TP4 shapes, B = 1 / 2 / 8 / 32
    python test_gdn_decode_ops.py tp2 q122     # + Ornith TP2 (p300) and Qwen3.5-122B TP4 shapes
    python test_gdn_decode_ops.py prof         # eager calls for the device profiler
"""

import os
import sys

import torch

import ttnn
from models.demos.blackhole.qwen36.tt.gdn_decode.ops import TILE, GdnDecode

torch.manual_seed(0)

SHAPES = {  # (Nk, Nv) per die, Dk = Dv = 128
    "tp4": (4, 8),   # Ornith-1.5-35B-A3B on 4 dies
    "tp2": (8, 16),  # Ornith on 2 dies (p300)
    "q122": (4, 16),  # Qwen3.5-122B-A10B on 4 dies
}


def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def bf(t):
    return t.to(torch.bfloat16).float()


def reference(proj, st, taps, neg_a, dt, norm_w, S, B, Nk, Nv, D, scale, eps=1e-6):
    """-> (y [B, Nv D], S_new [B, Nv, D, D], conv [32, qkv]) in FP32 from the bf16 inputs (the unfused path's math)."""
    qkv = 2 * Nk * D + Nv * D
    x = proj[:, :qkv]
    conv = torch.nn.functional.silu(taps[0] * st[0] + taps[1] * st[1] + taps[2] * st[2] + taps[3] * x)
    q = conv[:B, : Nk * D].reshape(B, Nk, D)
    k = conv[:B, Nk * D : 2 * Nk * D].reshape(B, Nk, D)
    v = conv[:B, 2 * Nk * D :].reshape(B, Nv, D)
    z = proj[:B, qkv : qkv + Nv * D].reshape(B, Nv, D)
    a_col = (qkv + Nv * D)
    a = proj[:B, a_col : a_col + Nv]
    b = proj[:B, a_col + TILE : a_col + TILE + Nv]
    rf = Nv // Nk
    q = q.repeat_interleave(rf, dim=1)
    k = k.repeat_interleave(rf, dim=1)
    qn = q * scale / torch.sqrt((q * q).sum(-1, keepdim=True) + eps)
    kn = k / torch.sqrt((k * k).sum(-1, keepdim=True) + eps)
    g = neg_a * torch.nn.functional.softplus(a + dt)
    decay = torch.exp(g)[..., None, None]
    beta = torch.sigmoid(b)[..., None]
    if os.environ.get("GDN_DEBUG"):
        print(f"GDNREF d {torch.exp(g)[0, 0].item():.6g} beta {torch.sigmoid(b)[0, 0].item():.6g} "
              f"rq {(scale / torch.sqrt((q * q).sum(-1) + eps))[0, 0].item():.6g} "
              f"rk {(1 / torch.sqrt((k * k).sum(-1) + eps))[0, 0].item():.6g} qk {(qn * kn).sum(-1)[0, 0].item():.6g}")
    if os.environ.get("GDN_UNITS"):
        for n in map(int, os.environ["GDN_UNITS"].split(",")):
            bb, hh = divmod(n, Nv)
            if bb < B:
                print(f"GDNREF n{n} d {torch.exp(g)[bb, hh].item():.6g} beta {torch.sigmoid(b)[bb, hh].item():.6g} "
                      f"rq {(scale / torch.sqrt((q * q).sum(-1) + eps))[bb, hh].item():.6g} "
                      f"rk {(1 / torch.sqrt((k * k).sum(-1) + eps))[bb, hh].item():.6g} "
                      f"qk {(qn * kn).sum(-1)[bb, hh].item():.6g}")
    sd = decay * S[:B]
    vread = torch.einsum("bhk,bhkv->bhv", kn, sd)
    u = beta * (v - vread)
    S_new = sd + kn[..., :, None] * u[..., None, :]
    o = torch.einsum("bhk,bhkv->bhv", qn, S_new)
    y = o / torch.sqrt((o * o).mean(-1, keepdim=True) + eps) * norm_w * torch.nn.functional.silu(z)
    return y.reshape(B, Nv * D), S_new, conv


def unit_state(proj, st, taps, neg_a, dt, S, n, Nk, Nv, D, scale, over=None, eps=1e-6):
    """S_new of unit n = (b, h) with optional scalar overrides {d, beta, rk, rq} (debug)."""
    over = over or {}
    b_, h = divmod(n, Nv)
    qkv = 2 * Nk * D + Nv * D
    x = proj[:, :qkv]
    conv = torch.nn.functional.silu(taps[0] * st[0] + taps[1] * st[1] + taps[2] * st[2] + taps[3] * x)[b_]
    hk = h // (Nv // Nk)
    k = conv[Nk * D + hk * D : Nk * D + (hk + 1) * D]
    v = conv[2 * Nk * D + h * D : 2 * Nk * D + (h + 1) * D]
    a_col = qkv + Nv * D
    a, bb = proj[b_, a_col + h], proj[b_, a_col + TILE + h]
    d = over.get("d", torch.exp(neg_a[h] * torch.nn.functional.softplus(a + dt[h])))
    beta = over.get("beta", torch.sigmoid(bb))
    rk = over.get("rk", 1 / torch.sqrt((k * k).sum() + eps))
    kn = k * rk
    sd = d * S[b_, h]
    u = beta * (v - kn @ sd)
    return sd + kn[:, None] * u[None, :], dict(d=d, beta=beta, rk=rk)


def check(dev, B, Nk, Nv, label, D=128, trials=2):
    qkv = 2 * Nk * D + Nv * D
    P = qkv + Nv * D + 2 * TILE
    scale = D**-0.5
    rep = {}
    ops = GdnDecode(dev, Nk, Nv, D, D, P, scale)
    ops.keep_conv = True
    for t in range(trials):
        proj = bf(torch.randn(TILE, P))
        st = [bf(torch.randn(TILE, qkv)) for _ in range(3)]
        taps = [bf(torch.randn(qkv) * 0.5) for _ in range(4)]
        neg_a = bf(-torch.exp(torch.rand(Nv) * 2.7))
        dt = bf(torch.randn(Nv) * 0.5)
        norm_w = bf(1 + 0.1 * torch.randn(D))
        S = bf(torch.randn(TILE, Nv, D, D) * 0.05)
        to = lambda x, dtype=ttnn.bfloat16, mc=ttnn.DRAM_MEMORY_CONFIG: ttnn.from_torch(
            x, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=mc)
        proj_t = to(proj[:B].reshape(1, B, P), mc=ttnn.L1_MEMORY_CONFIG)
        cs = [to(torch.zeros(1, TILE, qkv))] + [to(s.reshape(1, TILE, qkv)) for s in st]
        state = to(S)
        tap_t = [to(x.reshape(1, 1, qkv)) for x in taps]
        taps_t, params_t = ops.prepare(tap_t, to(neg_a.reshape(1, 1, Nv)), to(dt.reshape(1, 1, Nv)),
                                       to(norm_w.reshape(1, 1, D)))
        x_rows = torch.zeros(TILE, P)
        x_rows[:B] = proj[:B]
        y_ref, S_ref, conv_ref = reference(x_rows, st, taps, neg_a, dt, norm_w, S, B, Nk, Nv, D, scale)
        y = ops(proj_t, cs, taps_t, params_t, state, B)
        y_got = ttnn.to_torch(y).reshape(B, Nv * D).float()
        S_got = ttnn.to_torch(state).float()
        conv_got = ttnn.to_torch(ops.conv_out).reshape(TILE, qkv)
        st_got = [ttnn.to_torch(cs[j]).reshape(TILE, qkv).float() for j in (1, 2, 3)]
        # shift register: rows 1..3 <- old 2, old 3, x (the whole tile, padded rows included)
        assert torch.equal(st_got[0], st[1]) and torch.equal(st_got[1], st[2]), f"{label} B={B}: conv shift"
        assert torch.equal(st_got[2][:B], proj[:B, :qkv]), f"{label} B={B}: conv input row"
        assert torch.equal(S_got[B:], S[B:]), f"{label} B={B}: users >= B touched"
        p_conv = pcc(conv_got[:B], conv_ref[:B])
        p_y = pcc(y_got, y_ref)
        p_s = pcc(S_got[:B], S_ref)
        err_y = ((y_got - y_ref).abs().max() / y_ref.abs().max()).item()
        err_s = ((S_got[:B] - S_ref).abs().max() / S_ref.abs().max()).item()
        print(f"{label} B={B} trial {t}: conv pcc {p_conv:.6f}, y pcc {p_y:.6f} (max err {err_y:.2e} of max), "
              f"state pcc {p_s:.6f} (max err {err_s:.2e} of max)")
        if os.environ.get("GDN_DEBUG") and p_s <= 0.9999:
            e = ((S_got[:B] - S_ref).abs().amax(dim=(-1, -2)) / S_ref.abs().amax()).reshape(-1)  # per unit n = b Nv + h
            bad = (e > 0.02).nonzero().reshape(-1).tolist()
            from models.demos.blackhole.qwen36.tt.gdn_decode.ops import _split
            grid = dev.compute_with_storage_grid_size()
            work = _split(B * Nv, min(B * Nv, grid.x * grid.y))
            pos = {}
            for c, (s0, cnt) in enumerate(work):
                for i in range(cnt):
                    pos[s0 + i] = (c, i, cnt)
            for n in bad[:2]:
                bb, hh = divmod(n, Nv)
                R = (S_got[bb, hh] - S_ref[bb, hh]).double()
                U, sv, Vh = torch.linalg.svd(R)
                rows = R.abs().amax(dim=1)
                cols = R.abs().amax(dim=0)
                print(f"  n{n}: |R| max {R.abs().max().item():.3g}, singular values {[round(x, 4) for x in sv[:4].tolist()]}, "
                      f"nonzero rows {(rows > 1e-2).sum().item()}, nonzero cols {(cols > 1e-2).sum().item()}, "
                      f"rows idx {(rows > 1e-2).nonzero().reshape(-1)[:12].tolist()}, cols idx {(cols > 1e-2).nonzero().reshape(-1)[:12].tolist()}")
            for n in bad[:4]:
                c, i, cnt = pos[n]
                bb, hh = divmod(n, Nv)
                base, sc = unit_state(x_rows, st, taps, neg_a, dt, S, n, Nk, Nv, D, scale)
                e0 = (S_got[bb, hh] - base).abs().max().item()
                tries = []
                for m in [n - j for j in range(1, i + 1)]:
                    _, scm = unit_state(x_rows, st, taps, neg_a, dt, S, m, Nk, Nv, D, scale)
                    for key in ("d", "beta", "rk"):
                        alt, _ = unit_state(x_rows, st, taps, neg_a, dt, S, n, Nk, Nv, D, scale, {key: scm[key]})
                        tries.append((f"{key}<-n{m}", (S_got[bb, hh] - alt).abs().max().item()))
                    alt, _ = unit_state(x_rows, st, taps, neg_a, dt, S, n, Nk, Nv, D, scale, {k_: scm[k_] for k_ in ("d", "beta")})
                    tries.append((f"d+beta<-n{m}", (S_got[bb, hh] - alt).abs().max().item()))
                tries.sort(key=lambda t: t[1])
                print(f"  n{n} (core {c} #{i}): err vs own scalars {e0:.3g}; best swaps {[(k_, round(v_, 4)) for k_, v_ in tries[:3]]}")
            print(f"{label} B={B}: {len(bad)} bad units of {B * Nv}: " + ", ".join(f"n{n}(core {pos[n][0]} #{pos[n][1]}/{pos[n][2]})" for n in bad[:40]))
        assert p_conv > 0.9999 and p_y > 0.999 and p_s > 0.9999, label
        for t_ in [proj_t, state, taps_t, params_t, y, ops.conv_out] + cs + tap_t:
            ttnn.deallocate(t_)


def main():
    what = sys.argv[1:] or ["tp4"]
    dev = ttnn.open_device(device_id=0, l1_small_size=0, trace_region_size=0)
    try:
        bs = [int(v) for v in os.environ.get("BS", "1,2,8,32").split(",")]
        for s in [w for w in what if w in SHAPES]:
            Nk, Nv = SHAPES[s]
            for B in bs:
                check(dev, B, Nk, Nv, s)
        if "prof" in what:
            Nk, Nv = SHAPES[os.environ.get("PROF_SHAPE", "tp4")]
            for B in bs:
                check(dev, B, Nk, Nv, "prof", trials=2)
            ttnn.ReadDeviceProfiler(dev)
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
