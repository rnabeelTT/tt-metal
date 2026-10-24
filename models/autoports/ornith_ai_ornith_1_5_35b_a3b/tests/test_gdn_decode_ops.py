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
    sd = decay * S[:B]
    vread = torch.einsum("bhk,bhkv->bhv", kn, sd)
    u = beta * (v - vread)
    S_new = sd + kn[..., :, None] * u[..., None, :]
    o = torch.einsum("bhk,bhkv->bhv", qn, S_new)
    y = o / torch.sqrt((o * o).mean(-1, keepdim=True) + eps) * norm_w * torch.nn.functional.silu(z)
    return y.reshape(B, Nv * D), S_new, conv


def check(dev, B, Nk, Nv, label, D=128, trials=2):
    qkv = 2 * Nk * D + Nv * D
    P = qkv + Nv * D + 2 * TILE
    scale = D**-0.5
    rep = {}
    ops = GdnDecode(dev, Nk, Nv, D, D, P, scale)
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
        assert p_conv > 0.9999 and p_y > 0.999 and p_s > 0.9999, label
        for t_ in [proj_t, state, taps_t, params_t] + cs + tap_t:
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
