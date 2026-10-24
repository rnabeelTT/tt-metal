"""On-device check of the attention decode prep op (models/demos/blackhole/qwen36/tt/attn_decode) against torch.

    python test_attn_decode_ops.py              # Ornith TP4 heads (4 q / 1 kv), B = 1 / 2 / 8 / 32
    python test_attn_decode_ops.py tp2 q122     # + Ornith TP2 (8 / 1) and Qwen3.5-122B TP4 (8 / 1)
    python test_attn_decode_ops.py prof         # eager calls for the device profiler
"""

import os
import sys

import torch

import ttnn
from models.demos.blackhole.qwen36.tt.attn_decode.ops import AttnDecodePost, AttnDecodePrep

torch.manual_seed(0)
SHAPES = {"tp4": (4, 1), "tp2": (8, 1), "q122": (8, 1)}
HD, RD = 256, 64


def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def bf(t):
    return t.to(torch.bfloat16).float()


def reference(qkv, cos, sin, wq, wk, B, NH, NKV, eps=1e-6):
    x = qkv[0, 0, :B]
    q = x[:, : NH * HD].reshape(B, NH, HD)
    k = x[:, NH * HD : (NH + NKV) * HD].reshape(B, NKV, HD)
    v = x[:, (NH + NKV) * HD : (NH + 2 * NKV) * HD].reshape(B, NKV, HD)
    g = x[:, (NH + 2 * NKV) * HD : (2 * NH + 2 * NKV) * HD].reshape(B, NH, HD)

    def norm_rope(t, w):
        t = t / torch.sqrt((t * t).mean(-1, keepdim=True) + eps) * w
        c, s = cos[0, :B, 0][:, None, :], sin[0, :B, 0][:, None, :]
        r = t[..., :RD]
        rot = torch.cat([-r[..., RD // 2 :], r[..., : RD // 2]], -1)
        return torch.cat([r * c + rot * s, t[..., RD:]], -1)

    return norm_rope(q, wq), norm_rope(k, wk), v, torch.sigmoid(g)


def check(dev, B, NH, NKV, label, trials=2):
    W = (2 * NH + 2 * NKV) * HD
    op = AttnDecodePrep(dev, NH, NKV, HD, RD)
    post = AttnDecodePost(dev, NH, HD)
    for t in range(trials):
        qkv = bf(torch.randn(1, 1, 32, W) * 2)
        pos = torch.randint(0, 100000, (B,)).float()
        inv = 1.0 / (10000000 ** (torch.arange(0, RD, 2).float() / RD))
        f = pos[:, None] * inv[None, :]
        emb = torch.cat([f, f], -1)
        cos, sin = bf(emb.cos()).reshape(1, B, 1, RD), bf(emb.sin()).reshape(1, B, 1, RD)
        wq, wk = 1 + 0.2 * torch.randn(HD), 1 + 0.2 * torch.randn(HD)
        to = lambda x, mc=ttnn.DRAM_MEMORY_CONFIG: ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                                                 device=dev, memory_config=mc)
        qkv_t = to(qkv[:, :, :B].contiguous(), ttnn.L1_MEMORY_CONFIG if t % 2 else ttnn.DRAM_MEMORY_CONFIG)  # model: L1
        cos_t, sin_t = to(cos), to(sin)
        const = op.prepare(wq, wk)
        outs = op(qkv_t, cos_t, sin_t, const, B)
        refs = reference(qkv, cos, sin, wq.float(), wk.float(), B, NH, NKV)
        msg = []
        for name, o, r in zip(("q", "k", "v", "gate"), outs, refs):
            got = ttnn.to_torch(o).float().reshape(B, -1, HD)
            p = pcc(got, r)
            e = ((got - r).abs().max() / r.abs().max()).item()
            msg.append(f"{name} pcc {p:.6f} err {e:.1e}")
            if os.environ.get("ATTN_DEBUG") and p <= 0.9999:
                d = (got - r).abs()
                print(f"{label} B={B} {name}: per-user max err", d.amax((1, 2)).tolist())
                print("  per-head", d.amax((0, 2)).tolist())
                print("  per-tile", d.reshape(B, -1, HD // 32, 32).amax((0, 1, 3)).tolist())
                print("  ratio got/ref per head (tile 3)", (got[..., 96:128].norm(dim=-1) / r[..., 96:128].norm(dim=-1)).tolist())
                print("  got[0,0,:8]", got[0, 0, :8].tolist(), "ref", r[0, 0, :8].tolist())
                print("  got[0,0,96:104]", got[0, 0, 96:104].tolist(), "ref", r[0, 0, 96:104].tolist())
            assert p > 0.9999 or (name == "v" and torch.equal(got, r)), f"{label} B={B} {name}"
        att = bf(torch.randn(1, B, NH, HD))
        att_t = ttnn.from_torch(att, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                                memory_config=ttnn.L1_MEMORY_CONFIG)
        y = ttnn.to_torch(post(att_t, outs[3], B)).float().reshape(B, NH * HD)
        g_got = ttnn.to_torch(outs[3]).float().reshape(B, NH, HD)
        y_ref = (att[0] * g_got).reshape(B, NH * HD)
        p = pcc(y, y_ref)
        msg.append(f"post pcc {p:.6f}")
        assert p > 0.9999, f"{label} B={B} post"
        ttnn.deallocate(att_t)
        print(f"{label} B={B} trial {t}: " + ", ".join(msg))
        for x in (qkv_t, cos_t, sin_t, const):
            ttnn.deallocate(x)


def main():
    what = sys.argv[1:] or ["tp4"]
    dev = ttnn.open_device(device_id=0, l1_small_size=0, trace_region_size=0)
    try:
        bs = [int(v) for v in os.environ.get("BS", "1,2,8,32").split(",")]
        for s in [w for w in what if w in SHAPES]:
            for B in bs:
                check(dev, B, *SHAPES[s], s)
        if "prof" in what:
            for B in bs:
                check(dev, B, *SHAPES[os.environ.get("PROF_SHAPE", "tp4")], "prof")
            ttnn.ReadDeviceProfiler(dev)
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
