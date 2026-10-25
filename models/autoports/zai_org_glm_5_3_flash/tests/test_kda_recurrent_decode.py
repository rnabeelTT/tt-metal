# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""ttnn.experimental.kda.recurrent_decode: one KDA decode step per (user, head), state updated in place, vs a torch
fp32 reference; chained steps (the state feeds the next step), rows R > B, and a traced replay."""

import pytest
import torch

import ttnn

H, K, V = 64, 128, 128
EPS = 1e-6


def _pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm()))


def _ref_step(S, qkv, decay, beta, B, q_scale):
    """S [B,H,K,V]; qkv [R, 2HK+HV]; decay [R, HK]; beta [R, H] -> (o [B*H, V], S')."""
    q = qkv[:B, : H * K].reshape(B, H, K)
    k = qkv[:B, H * K : 2 * H * K].reshape(B, H, K)
    v = qkv[:B, 2 * H * K :].reshape(B, H, V)
    qn = q / torch.sqrt((q * q).sum(-1, keepdim=True) + EPS) * q_scale
    kn = k / torch.sqrt((k * k).sum(-1, keepdim=True) + EPS)
    Sd = decay[:B].reshape(B, H, K, 1) * S
    vread = torch.einsum("bhk,bhkv->bhv", kn, Sd)
    u = beta[:B].reshape(B, H, 1) * (v - vread)
    S2 = Sd + kn.unsqueeze(-1) * u.unsqueeze(-2)
    o = torch.einsum("bhk,bhkv->bhv", qn, S2)
    return o.reshape(B * H, V), S2


def _inputs(B, R, gen):
    qkv = torch.randn(R, 2 * H * K + H * V, generator=gen)
    decay = torch.exp(-5 * torch.sigmoid(torch.randn(R, H * K, generator=gen)))  # GLM bounded gate range
    beta = torch.sigmoid(torch.randn(R, H, generator=gen))
    return qkv, decay, beta


def _dev(t, device):
    return ttnn.from_torch(t.reshape(1, 1, *t.shape[-2:]), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)


@pytest.mark.parametrize("B,R", [(1, 1), (8, 8), (8, 32), (32, 32)])
def test_kda_recurrent_decode(device, B, R):
    gen = torch.Generator().manual_seed(B * 100 + R)
    q_scale = K**-0.5
    S = torch.randn(B, H, K, V, generator=gen) * 0.05
    S_dev = ttnn.from_torch(S, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    addr = S_dev.buffer_address()
    for step in range(3):
        qkv, decay, beta = _inputs(B, R, gen)
        o_ref, S = _ref_step(S, qkv, decay, beta, B, q_scale)
        o = ttnn.experimental.kda.recurrent_decode(
            S_dev, _dev(qkv, device), _dev(decay, device), _dev(beta, device), q_scale
        )
        assert list(o.shape) == [1, 1, B * H, V]
        o_tt = ttnn.to_torch(o)[0, 0]
        S_tt = ttnn.to_torch(S_dev)
        assert S_dev.buffer_address() == addr
        po, ps = _pcc(o_tt, o_ref), _pcc(S_tt, S)
        eo, es = (o_tt - o_ref).abs().max().item(), (S_tt - S).abs().max().item()
        print(f"B={B} R={R} step {step}: o PCC {po:.7f} max|err| {eo:.2e}; state PCC {ps:.7f} max|err| {es:.2e}")
        assert po > 0.9999 and ps > 0.99999


def test_kda_recurrent_decode_traced(device_params_trace):
    """Capture once, replay with new inputs copied into the captured input buffers; the state keeps its address."""
    device = device_params_trace
    B, R = 8, 32
    gen = torch.Generator().manual_seed(7)
    q_scale = K**-0.5
    S = torch.randn(B, H, K, V, generator=gen) * 0.05
    S_dev = ttnn.from_torch(S, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    qkv, decay, beta = _inputs(B, R, gen)
    bufs = [_dev(qkv, device), _dev(decay, device), _dev(beta, device)]
    # warm (compile), restore the state, then capture
    ttnn.deallocate(ttnn.experimental.kda.recurrent_decode(S_dev, *bufs, q_scale))
    ttnn.copy_host_to_device_tensor(ttnn.from_torch(S, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT), S_dev)
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    o = ttnn.experimental.kda.recurrent_decode(S_dev, *bufs, q_scale)
    ttnn.end_trace_capture(device, tid, cq_id=0)
    try:
        for step in range(3):
            if step:
                qkv, decay, beta = _inputs(B, R, gen)
            for t, buf in zip((qkv, decay, beta), bufs):
                host = ttnn.from_torch(t.reshape(1, 1, *t.shape), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)
                ttnn.copy_host_to_device_tensor(host, buf)
            ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
            o_ref, S = _ref_step(S, qkv, decay, beta, B, q_scale)
            po, ps = _pcc(ttnn.to_torch(o)[0, 0], o_ref), _pcc(ttnn.to_torch(S_dev), S)
            print(f"traced step {step}: o PCC {po:.7f} state PCC {ps:.7f}")
            assert po > 0.9999 and ps > 0.99999
    finally:
        ttnn.release_trace(device, tid)


@pytest.fixture
def device_params_trace():
    dev = ttnn.open_device(device_id=0, trace_region_size=8 << 20)
    yield dev
    ttnn.close_device(dev)


@pytest.mark.parametrize("decay_lo", [0.99, 0.999])
def test_kda_recurrent_decode_long_chain(device, decay_lo):
    """1000 chained steps (state fed back in place) with decays in [decay_lo, 1): the state must track the fp32 torch
    recurrence without drift (a per-step bias in the decay or the state reload would compound)."""
    B, R, steps = 1, 1, 1000
    gen = torch.Generator().manual_seed(11)
    q_scale = K**-0.5
    S = torch.zeros(B, H, K, V)
    S_dev = ttnn.from_torch(S, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    worst, worst_r = 1.0, 0.0
    for step in range(steps):
        qkv = torch.randn(R, 2 * H * K + H * V, generator=gen)
        decay = decay_lo + (1 - decay_lo) * torch.rand(R, H * K, generator=gen)
        beta = torch.sigmoid(torch.randn(R, H, generator=gen))
        o_ref, S = _ref_step(S, qkv, decay, beta, B, q_scale)
        o = ttnn.experimental.kda.recurrent_decode(
            S_dev, _dev(qkv, device), _dev(decay, device), _dev(beta, device), q_scale
        )
        if step % 100 == 99 or step == steps - 1:
            S_tt = ttnn.to_torch(S_dev)
            ps = _pcc(S_tt, S)
            r = float(S_tt.norm() / S.norm())
            po = _pcc(ttnn.to_torch(o)[0, 0], o_ref)
            worst, worst_r = min(worst, ps), max(worst_r, abs(r - 1))
            rel = float((S_tt - S).abs().max() / S.abs().max())
            print(
                f"decay>={decay_lo} step {step + 1}: state PCC {ps:.9f} rms ratio {r:.7f} max rel err {rel:.2e}; "
                f"o PCC {po:.9f}"
            )
        ttnn.deallocate(o)
    # fp32 state / decay must not be read at TF32 (a per-step bias compounds): no drift in scale
    assert worst > 0.99999 and worst_r < 1e-4, (worst, worst_r)


def _flags(device, B, live):
    n = -(-B // 16) * 16
    f = torch.zeros(1, n, dtype=torch.int32)
    f[0, list(live)] = 1
    return ttnn.from_torch(f, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)


def _flags_host(B, live):
    n = -(-B // 16) * 16
    f = torch.zeros(1, n, dtype=torch.int32)
    f[0, list(live)] = 1
    return ttnn.from_torch(f, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)


@pytest.mark.parametrize("live", [(5,), (0, 15), (3, 4, 9), tuple(range(16)), ()])
def test_kda_recurrent_decode_active(device, live):
    """Stage 12 `active` slot flags: active rows bit-identical to the op without flags (same per-head math on another
    core), inactive slots' state untouched (bit-identical to before the step) and their output rows zero."""
    B, R = 16, 32
    gen = torch.Generator().manual_seed(11 + len(live))
    q_scale = K**-0.5
    S0 = torch.randn(B, H, K, V, generator=gen) * 0.05
    qkv, decay, beta = _inputs(B, R, gen)
    ins = [_dev(qkv, device), _dev(decay, device), _dev(beta, device)]
    S_a = ttnn.from_torch(S0, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    o_a = ttnn.to_torch(ttnn.experimental.kda.recurrent_decode(S_a, *ins, q_scale))[0, 0]
    S_b = ttnn.from_torch(S0, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    o_b = ttnn.to_torch(ttnn.experimental.kda.recurrent_decode(S_b, *ins, q_scale, active=_flags(device, B, live)))[
        0, 0
    ]
    Sa, Sb = ttnn.to_torch(S_a), ttnn.to_torch(S_b)
    for b in range(B):
        rows = slice(b * H, (b + 1) * H)
        if b in live:
            assert torch.equal(o_b[rows], o_a[rows]), f"slot {b}: output differs"
            assert torch.equal(Sb[b], Sa[b]), f"slot {b}: state differs"
        else:
            assert torch.equal(Sb[b], S0[b]), f"slot {b}: idle state changed"
            assert not o_b[rows].any(), f"slot {b}: idle output not zero"


def test_kda_recurrent_decode_active_traced(device_params_trace):
    """One capture, then replays with different flags written into the captured flags buffer (the active set changes
    without a recapture): each replay matches an eager step with the same flags."""
    device = device_params_trace
    B, R = 16, 32
    gen = torch.Generator().manual_seed(23)
    q_scale = K**-0.5
    S0 = torch.randn(B, H, K, V, generator=gen) * 0.05
    qkv, decay, beta = _inputs(B, R, gen)
    ins = [_dev(qkv, device), _dev(decay, device), _dev(beta, device)]
    flags = _flags(device, B, range(B))
    S_t = ttnn.from_torch(S0, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn.deallocate(ttnn.experimental.kda.recurrent_decode(S_t, *ins, q_scale, active=flags))  # compile
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    o = ttnn.experimental.kda.recurrent_decode(S_t, *ins, q_scale, active=flags)
    ttnn.end_trace_capture(device, tid, cq_id=0)
    try:
        for live in [(7,), (1, 2, 14), tuple(range(16)), (0,)]:
            ttnn.copy_host_to_device_tensor(ttnn.from_torch(S0, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT), S_t)
            ttnn.copy_host_to_device_tensor(_flags_host(B, live), flags)
            ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
            S_e = ttnn.from_torch(S0, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
            o_e = ttnn.experimental.kda.recurrent_decode(S_e, *ins, q_scale, active=_flags(device, B, live))
            assert torch.equal(ttnn.to_torch(o), ttnn.to_torch(o_e)), f"live {live}: traced output differs"
            assert torch.equal(ttnn.to_torch(S_t), ttnn.to_torch(S_e)), f"live {live}: traced state differs"
    finally:
        ttnn.release_trace(device, tid)
