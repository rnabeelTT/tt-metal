"""On-device check of the two fused GDN decode ops against a torch reference of the Qwen3.5 GatedDeltaNet decode step.

Shapes are Ornith's per-device GDN at TP2 (8 q/k heads, 16 value heads, K = V = 128, conv width 4096) and TP4 (4 / 8).
For each (B active rows of a Bmax-row state) it steps T tokens through
    kda_decode_conv (4-tap causal conv + SiLU, history shifted in place)
    kda.recurrent_decode (grouped q/k heads, per-head decay, fp32 state in place)
and compares outputs and states with torch (fp32), and checks that idle rows [B, Bmax) are untouched.

    python test_gdn_fused_ops.py
"""

import sys

import torch

import ttnn

torch.manual_seed(0)


def l2n(x, eps=1e-6):
    return x / torch.sqrt((x * x).sum(-1, keepdim=True) + eps)


def ref_step(S, q, k, v, g, beta, scale):
    """HF Qwen3.5 torch_recurrent_gated_delta_rule, one token. S [B,H,K,V]; q,k [B,H,K]; v [B,H,V]; g,beta [B,H]."""
    q = l2n(q) * scale
    k = l2n(k)
    S = S * g.exp()[..., None, None]
    kv_mem = (S * k[..., None]).sum(-2)
    delta = (v - kv_mem) * beta[..., None]
    S = S + k[..., None] * delta[..., None, :]
    o = (S * q[..., None]).sum(-2)
    return S, o


def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def tt(x, dev, dtype=ttnn.float32):
    return ttnn.from_torch(x, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def back(x):
    return ttnn.to_torch(x).float()


def run_case(dev, Hk, H, B, Bmax, T=6, K=128, V=128):
    W = 2 * Hk * K + H * V
    P = W + 512  # proj carries more columns after qkv ([qkv|z|a|b]); the conv reads the first W
    scale = K**-0.5
    taps = torch.randn(4, W) * 0.5
    hist = [torch.randn(1, 1, Bmax, W) for _ in range(3)]
    S0 = torch.randn(Bmax, H, K, V) * 0.1

    # device state
    h_tt = [tt(h.bfloat16(), dev, ttnn.bfloat16) for h in hist]
    taps_tt = [tt(taps[j].reshape(1, 1, 1, W), dev) for j in range(4)]
    m = torch.zeros(1, 1, Bmax, 1)
    m[:, :, :B] = 1
    act_tt = tt(m, dev)
    S_tt = tt(S0.clone(), dev)

    # reference state (history kept in bf16 like the device buffers)
    h_ref = [h.bfloat16().float() for h in hist]
    S_ref = S0.clone()
    worst = {"qkv": 1.0, "o": 1.0, "S": 1.0}
    for _ in range(T):
        proj = torch.randn(1, 1, B, P)
        a = torch.randn(B, H)
        b = torch.randn(B, H)
        g = -torch.exp(torch.randn(H) * 0.3) * torch.nn.functional.softplus(a + 0.5)  # neg_exp_A * softplus(a + dt)
        beta = torch.sigmoid(b)

        cur = proj.bfloat16().float()[..., :W]
        acc = cur * taps[3] + h_ref[0][:, :, :B] * taps[0] + h_ref[1][:, :, :B] * taps[1] + h_ref[2][:, :, :B] * taps[2]
        qkv_ref = torch.nn.functional.silu(acc)
        new_h = [h_ref[1].clone(), h_ref[2].clone(), h_ref[2].clone()]
        new_h[2][:, :, :B] = cur.bfloat16().float()
        for j in range(2):
            new_h[j][:, :, B:] = h_ref[j][:, :, B:]
        new_h[2][:, :, B:] = h_ref[2][:, :, B:]
        h_ref = new_h

        qkv_dev = ttnn.experimental.deepseek_prefill.kda_decode_conv(
            tt(proj.bfloat16(), dev, ttnn.bfloat16), h_tt[0], h_tt[1], h_tt[2], *taps_tt, act_tt, W
        )
        qkv_d = back(qkv_dev)[..., :B, :]
        worst["qkv"] = min(worst["qkv"], pcc(qkv_d, qkv_ref))

        q = qkv_ref[0, 0, :, : Hk * K].reshape(B, Hk, K).repeat_interleave(H // Hk, dim=1)
        k = qkv_ref[0, 0, :, Hk * K : 2 * Hk * K].reshape(B, Hk, K).repeat_interleave(H // Hk, dim=1)
        v = qkv_ref[0, 0, :, 2 * Hk * K :].reshape(B, H, V)
        S_new, o_ref = ref_step(S_ref[:B], q, k, v, g, beta, scale)
        S_ref[:B] = S_new

        o_dev = ttnn.experimental.kda.recurrent_decode(
            S_tt,
            qkv_dev,
            tt(g.exp().reshape(1, 1, B, H), dev),
            tt(beta.reshape(1, 1, B, H), dev),
            scale,
            eps=1e-6,
            num_qk_heads=Hk,
            per_head_decay=True,
            batch=B,
        )
        o_d = back(o_dev)[0, 0, : B * H].reshape(B, H, V)
        worst["o"] = min(worst["o"], pcc(o_d, o_ref))
        worst["S"] = min(worst["S"], pcc(back(S_tt)[:B], S_ref[:B]))
    S_d = back(S_tt)
    idle_state_ok = torch.equal(S_d[B:], S0[B:]) if B < Bmax else True
    h_d = [back(h) for h in h_tt]
    idle_hist_ok = (
        all(torch.equal(h_d[j][:, :, B:], hist[j].bfloat16().float()[:, :, B:]) for j in range(3)) if B < Bmax else True
    )
    hist_pcc = min(pcc(h_d[j][:, :, :B], h_ref[j][:, :, :B]) for j in range(3))
    return worst, hist_pcc, idle_state_ok, idle_hist_ok


def main():
    dev = ttnn.open_device(device_id=0)
    ok = True
    try:
        for Hk, H in ((8, 16), (4, 8)):
            for B, Bmax in ((1, 32), (2, 32), (4, 32), (32, 32)):
                worst, hist_pcc, idle_s, idle_h = run_case(dev, Hk, H, B, Bmax)
                good = min(worst.values()) > 0.999 and hist_pcc > 0.9999 and idle_s and idle_h
                ok &= good
                print(
                    f"Hk={Hk} H={H} B={B}/{Bmax}: pcc qkv {worst['qkv']:.6f} o {worst['o']:.6f} S {worst['S']:.6f} "
                    f"hist {hist_pcc:.6f} idle state untouched {idle_s} idle hist untouched {idle_h} -> "
                    f"{'PASS' if good else 'FAIL'}",
                    flush=True,
                )
    finally:
        ttnn.close_device(dev)
    print("ALL PASS" if ok else "SOME FAILED")
    sys.exit(0 if ok else 1)


def check_reshape_is_view():
    """The fused decode shifts conv_states[1..3] ([1, Bmax, W] bf16) in place through 4-D reshapes of them."""
    dev = ttnn.open_device(device_id=0)
    try:
        a = ttnn.from_torch(
            torch.zeros(1, 32, 4096),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=dev,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        v = ttnn.reshape(a, (1, 1, 32, 4096))
        same = a.buffer_address() == v.buffer_address()
        print(f"reshape [1,32,4096] -> [1,1,32,4096] is a view: {same}", flush=True)
        return same
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    if not check_reshape_is_view():
        sys.exit(2)
    main()
