"""On-device check of the decode add + RMSNorm op (models/demos/blackhole/qwen36/tt/addnorm) against torch.

    python test_addnorm_ops.py          # H = 2048 (Ornith / Coder-Next) and 3072 (122B), B = 1 / 8 / 32, add / norm only
    python test_addnorm_ops.py prof     # + eager calls for the device profiler
"""

import os
import sys

import torch

import ttnn
from models.demos.blackhole.qwen36.tt.addnorm.ops import AddNorm
from models.demos.blackhole.qwen36.tt.decode_boundary import residual_memory_config

torch.manual_seed(0)


def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def bf(t):
    return t.to(torch.bfloat16).float()


def check(dev, H, B, add, op, trials=2):
    mc = residual_memory_config(H)
    for t in range(trials):
        res = bf(torch.randn(1, 1, B, H) * 3)
        red = bf(torch.randn(1, 1, B, H))
        gamma = 1 + 0.3 * torch.randn(H)
        to = lambda x: ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=mc)
        res_t, red_t = to(res), (to(red) if add else None)
        g = op.prepare(gamma)
        h, out = op(res_t, red_t, g)
        hr = bf(res + red) if add else res
        ref = hr / torch.sqrt((hr * hr).mean(-1, keepdim=True) + 1e-6) * bf(gamma)
        got = ttnn.to_torch(out).float()[..., :B, :]
        p = pcc(got, ref)
        e = ((got - ref).abs().max() / ref.abs().max()).item()
        msg = f"H={H} B={B} add={int(add)} trial {t}: out pcc {p:.6f} err {e:.1e}"
        if add:
            hg = ttnn.to_torch(h).float()[..., :B, :]
            dh = (hg - hr).abs().max().item()
            msg += f", h max diff {dh:.2e}"
            assert dh <= hr.abs().max().item() / 128, msg  # one bf16 ulp at the largest magnitude (packer rounding)
            ttnn.deallocate(h)
        print(msg)
        assert p > 0.99999, msg
        for x in (res_t, red_t, g, out):
            if x is not None:
                ttnn.deallocate(x)


def check_live(dev, H, B, op, n=3):
    """n residual / reduced pairs alive at once (distinct addresses) through the same cached programs."""
    mc = residual_memory_config(H)
    to = lambda x: ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=mc)
    gamma = 1 + 0.3 * torch.randn(H)
    g = op.prepare(gamma)
    xs = [(bf(torch.randn(1, 1, B, H) * 3), bf(torch.randn(1, 1, B, H))) for _ in range(n)]
    ts = [(to(r), to(d)) for r, d in xs]
    for add in (True, False):
        for i, ((r, d), (rt, dt)) in enumerate(zip(xs, ts)):
            h, out = op(rt, dt if add else None, g)
            hr = bf(r + d) if add else r
            ref = hr / torch.sqrt((hr * hr).mean(-1, keepdim=True) + 1e-6) * bf(gamma)
            p = pcc(ttnn.to_torch(out).float()[..., :B, :], ref)
            msg = f"live H={H} B={B} add={int(add)} tensor {i} (addr {rt.buffer_address()}): out pcc {p:.6f}"
            print(msg)
            assert p > 0.99999, msg
            for x in (h, out):
                if x is not None:
                    ttnn.deallocate(x)


def main():
    dev = ttnn.open_device(device_id=0, l1_small_size=0, trace_region_size=0)
    try:
        bs = [int(v) for v in os.environ.get("BS", "1,8,32").split(",")]
        for H in (2048, 3072):
            op = AddNorm(dev, H)
            for B in bs:
                for add in (True, False):
                    check(dev, H, B, add, op)
                check_live(dev, H, B, op)
        if "prof" in sys.argv[1:]:
            ttnn.ReadDeviceProfiler(dev)
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
