# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Fused-decoder tests (stage 2): the fused graph must match HF at the stage-1 bar and match the functional
graph, for both layer kinds, non-aligned lengths, batch, traced decode, determinism; perf signposts before/after.

    pytest models/autoports/qwen_qwen3_coder_next/tests/test_fused_decoder.py -v -s
"""
from __future__ import annotations

import os
import time

import pytest
import torch

import ttnn
from models.autoports.qwen_qwen3_coder_next.tests import reference as ref
from models.autoports.qwen_qwen3_coder_next.tests.test_functional_decoder import (
    DEVICE_PARAMS,
    L0,
    L3,
    _Harness,
    _hidden,
    _pcc,
    _prefill_users,
    _state,
    signpost,
)
from models.autoports.qwen_qwen3_coder_next.tt.fused_decoder import FusedDecoder, FusionFlags

os.environ.setdefault("HF_MODEL", "Qwen/Qwen3-Coder-Next")
PCC_BAR = 0.995
EQUIV_BAR = 0.999  # fused vs functional graph on identical weights/inputs (both bf16 device paths)


def _fused(device, layer_idx, *, batch, max_seq_len, real=False, flags=None, state=None, hf=True):
    return _Harness(
        device,
        layer_idx,
        batch=batch,
        max_seq_len=max_seq_len,
        real=real,
        state=state,
        hf=hf,
        decoder_cls=FusedDecoder,
        decoder_kwargs={"fusion_flags": flags or FusionFlags()},
    )


@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize(
    ("layer_idx", "seq_len"),
    [(L0, 5), (L3, 5), (L0, 129), (L3, 129), (L0, 333), (L3, 333)],
    ids=["gdn_seq5", "attn_seq5", "gdn_seq129", "attn_seq129", "gdn_seq333", "attn_seq333"],
)
def test_fused_prefill_decode_against_hf(device, layer_idx, seq_len):
    h = _fused(device, layer_idx, batch=1, max_seq_len=4096)
    hidden = _hidden(1, seq_len, h.H, seed=100 + seq_len)
    ref_pre, tt_pre = _prefill_users(h, hidden)
    _pcc(f"FUSED {h.kind} prefill T={seq_len}", ref_pre, tt_pre)
    dec = _hidden(1, 1, h.H, seed=200 + seq_len)
    pos = torch.tensor([seq_len])
    _pcc(f"FUSED {h.kind} decode pos={seq_len}", h.hf_decode(dec, pos), h.tt_decode(dec, pos))


@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("layer_idx", [L0, L3], ids=["gdn", "attn"])
def test_fused_matches_functional(device, layer_idx):
    """Same weights, same inputs: fused graph vs functional graph (device to device), prefill T=333 and decode."""
    cfg, text = ref.text_config()
    state = _state(text, layer_idx, real=False)
    seq_len = 333
    hidden = _hidden(1, seq_len, text.hidden_size, seed=900)
    dec = _hidden(1, 1, text.hidden_size, seed=901)
    pos = torch.tensor([seq_len])
    a = _Harness(device, layer_idx, batch=1, max_seq_len=4096, state=dict(state), hf=False)
    pre_a = a.tt_prefill(hidden)
    dec_a = a.tt_decode(dec, pos)
    b = _fused(device, layer_idx, batch=1, max_seq_len=4096, state=state, hf=False)
    pre_b = b.tt_prefill(hidden)
    dec_b = b.tt_decode(dec, pos)
    _pcc(f"{a.kind} fused-vs-functional prefill", pre_a, pre_b, EQUIV_BAR)
    _pcc(f"{a.kind} fused-vs-functional decode", dec_a, dec_b, EQUIV_BAR)


@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize(("layer_idx", "batch"), [(L0, 32), (L3, 32)], ids=["gdn_b32", "attn_b32"])
def test_fused_batched_decode_against_hf(device, layer_idx, batch):
    seq_len = 37
    h = _fused(device, layer_idx, batch=batch, max_seq_len=1024)
    hidden = _hidden(batch, seq_len, h.H, seed=300 + batch)
    ref_pre, tt_pre = _prefill_users(h, hidden)
    _pcc(f"FUSED {h.kind} prefill B={batch}", ref_pre, tt_pre)
    dec = _hidden(batch, 1, h.H, seed=400 + batch)
    pos = torch.full((batch,), seq_len)
    _pcc(f"FUSED {h.kind} batched decode B={batch}", h.hf_decode(dec, pos), h.tt_decode(dec, pos))


@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("layer_idx", [L0, L3], ids=["gdn", "attn"])
def test_fused_traced_decode_and_determinism(device, layer_idx):
    seq_len = 40
    h = _fused(device, layer_idx, batch=1, max_seq_len=1024)
    hidden = _hidden(1, seq_len, h.H, seed=500)
    ref_pre, tt_pre = _prefill_users(h, hidden)
    _pcc(f"FUSED {h.kind} prefill", ref_pre, tt_pre)
    steps = [_hidden(1, 1, h.H, seed=600 + i) for i in range(3)]
    x, kw = h.decode_inputs(steps[0], torch.tensor([seq_len]))
    out = h.tt.decode_forward(x, **kw)
    ttnn.deallocate(out)
    if h.kind == "linear_attention":
        h.tt.reset_gdn_state()
        h.tt_prefill(hidden)
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    out = h.tt.decode_forward(x, **kw)
    ttnn.end_trace_capture(device, tid, cq_id=0)
    if h.kind == "linear_attention":
        h.tt.reset_gdn_state()
        h.tt_prefill(hidden)
    for i, step in enumerate(steps):
        pos = torch.tensor([seq_len + i])
        ttnn.copy_host_to_device_tensor(ttnn.from_torch(step, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT), x)
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(pos.to(torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT), kw["current_pos"]
        )
        if h.kind == "full_attention":
            cos_h, sin_h = h.tt.rope_decode_host(pos)
            ttnn.copy_host_to_device_tensor(
                ttnn.from_torch(cos_h, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT), kw["cos"]
            )
            ttnn.copy_host_to_device_tensor(
                ttnn.from_torch(sin_h, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT), kw["sin"]
            )
        ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
        got = ttnn.to_torch(out)
        _pcc(f"FUSED {h.kind} traced decode step {i}", h.hf_decode(step, pos), got)
    # determinism: replay the same step twice at the same position (attention) / from the same state (gdn)
    if h.kind == "full_attention":
        ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
        g1 = ttnn.to_torch(out)
        ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
        g2 = ttnn.to_torch(out)
        assert torch.equal(g1, g2), "traced decode not deterministic"
    ttnn.release_trace(device, tid)
    if h.kind == "linear_attention":
        h.tt.reset_gdn_state()
        p1 = h.tt_prefill(hidden)
        h.tt.reset_gdn_state()
        p2 = h.tt_prefill(hidden)
        assert torch.equal(p1, p2), "fused prefill not deterministic"


@pytest.mark.real_weights
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize(
    ("layer_idx", "seq_len"),
    [(L0, 37), (L3, 37), (L0, 300), (L3, 300)],
    ids=["gdn_37", "attn_37", "gdn_300", "attn_300"],
)
def test_fused_real_weight_against_hf(device, layer_idx, seq_len):
    h = _fused(device, layer_idx, batch=1, max_seq_len=4096, real=True)
    hidden = _hidden(1, seq_len, h.H, seed=900 + seq_len)
    ref_pre, tt_pre = _prefill_users(h, hidden)
    _pcc(f"FUSED REAL {h.kind} prefill T={seq_len}", ref_pre, tt_pre)
    dec = _hidden(1, 1, h.H, seed=901 + seq_len)
    pos = torch.tensor([seq_len])
    _pcc(f"FUSED REAL {h.kind} decode pos={seq_len}", h.hf_decode(dec, pos), h.tt_decode(dec, pos))


@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("layer_idx", [L0, L3], ids=["gdn", "attn"])
def test_fused_runtime_fallback_audit(device, layer_idx):
    """No host transfers in the fused aligned prefill, chunk continuation and decode (same boundary rule as stage 1)."""
    from models.autoports.qwen_qwen3_coder_next.tests.test_functional_decoder import _HostTransferCounter, _tt

    h = _fused(device, layer_idx, batch=1, max_seq_len=4096, hf=False)
    for label, x, start in (
        ("aligned prefill", _tt(_hidden(1, 256, h.H, seed=1700), h.device), 0),
        ("continuation prefill", _tt(_hidden(1, 128, h.H, seed=1701), h.device), 256),
    ):
        with _HostTransferCounter() as c:
            h.tt.prefill_forward(x, start_pos=start, page_table=h.page_table_dev, user_id=0)
        assert not c.calls, f"{label}: {[(n, f.filename.split('/')[-1], f.lineno) for n, f in c.calls]}"
    xd, kw = h.decode_inputs(_hidden(1, 1, h.H, seed=1702), torch.tensor([384]))
    with _HostTransferCounter() as c:
        h.tt.decode_forward(xd, **kw)
    assert not c.calls, f"decode: {[(n, f.filename.split('/')[-1], f.lineno) for n, f in c.calls]}"


@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("layer_idx", [L0, L3], ids=["gdn", "attn"])
def test_fused_stress_repeated_runs(device, layer_idx):
    """Stress: 20 prefill+decode cycles with varying non-aligned lengths on one fused layer; outputs finite and the
    repeated (identical) cycle bit-identical."""
    h = _fused(device, layer_idx, batch=1, max_seq_len=2048, hf=False)
    lens = [5, 37, 128, 129, 333, 600, 1023, 1024, 1500] * 2 + [37, 37]
    ref = None
    for i, n in enumerate(lens):
        if h.kind == "linear_attention":
            h.tt.reset_gdn_state()
        out = h.tt_prefill(_hidden(1, n, h.H, seed=1800 + n))
        assert torch.isfinite(out.float()).all(), f"cycle {i} T={n}: non-finite prefill"
        dec = h.tt_decode(_hidden(1, 1, h.H, seed=1900 + n), torch.tensor([n]))
        assert torch.isfinite(dec.float()).all(), f"cycle {i} T={n}: non-finite decode"
        if n == 37 and i >= len(lens) - 2:
            if ref is None:
                ref = (out, dec)
            else:
                assert torch.equal(ref[0], out) and torch.equal(ref[1], dec), "repeated identical cycle differs"


@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("layer_idx", [L0, L3], ids=["gdn", "attn"])
def test_fused_chunked_prefill_matches_hf(device, layer_idx):
    """Fused graph at the production prefill chunk: 2048 + 2048 + 333 start_pos chunks vs one HF pass; for the attention
    layer also the K/V cache content per 512-position window; decode at 4429."""
    from models.autoports.qwen_qwen3_coder_next.tests.test_functional_decoder import CHUNK, _check_kv_cache

    lens = [CHUNK, CHUNK, 333]
    total = sum(lens)
    h = _fused(device, layer_idx, batch=1, max_seq_len=8192)
    hidden = _hidden(1, total, h.H, seed=800)
    ref_out = h.hf_prefill(hidden)
    outs, start = [], 0
    for n in lens:
        outs.append(h.tt_prefill(hidden[:, start : start + n], start=start))
        start += n
    _pcc(f"FUSED {h.kind} chunked prefill T={total}", ref_out, torch.cat(outs, dim=1))
    if h.kind == "full_attention":
        _check_kv_cache(h, layer_idx, total, "FUSED chunked")
    dec = _hidden(1, 1, h.H, seed=801)
    pos = torch.tensor([total])
    _pcc(f"FUSED {h.kind} decode after chunked prefill pos={total}", h.hf_decode(dec, pos), h.tt_decode(dec, pos))


# ---- perf: functional (before) vs fused (after), same harness ----------------------------------------------
def _perf_prefill(device, layer_idx, seq_len, fused, flags=None):
    mk = (
        (lambda: _fused(device, layer_idx, batch=1, max_seq_len=4096, real=True, flags=flags, hf=False))
        if fused
        else (lambda: _Harness(device, layer_idx, batch=1, max_seq_len=4096, real=True, hf=False))
    )
    h = mk()
    hidden = _hidden(1, seq_len, h.H, seed=2000)
    h.tt_prefill(hidden)
    if h.kind == "linear_attention":
        h.tt.reset_gdn_state()
    ttnn.synchronize_device(device)
    t0 = time.time()
    signpost("PERF_PREFILL")
    h.tt_prefill(hidden)
    ttnn.synchronize_device(device)
    signpost("PERF_PREFILL_END")
    print(f"{'fused' if fused else 'functional'} {h.kind}: prefill T={seq_len} wall {1e3 * (time.time() - t0):.1f} ms")


def _perf_traced_decode(device, layer_idx, fused, iters=3, flags=None):
    seq_len = 128
    h = (
        _fused(device, layer_idx, batch=1, max_seq_len=4096, real=True, flags=flags, hf=False)
        if fused
        else _Harness(device, layer_idx, batch=1, max_seq_len=4096, real=True, hf=False)
    )
    h.tt_prefill(_hidden(1, seq_len, h.H, seed=2100))
    x, kw = h.decode_inputs(_hidden(1, 1, h.H, seed=2101), torch.tensor([seq_len]))
    out = h.tt.decode_forward(x, **kw)
    ttnn.deallocate(out)
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    out = h.tt.decode_forward(x, **kw)
    ttnn.end_trace_capture(device, tid, cq_id=0)
    ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
    ttnn.synchronize_device(device)
    signpost("PERF_DECODE")
    t0 = time.time()
    for _ in range(iters):
        ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(device)
    dt = (time.time() - t0) / iters
    signpost("PERF_DECODE_END")
    print(f"{'fused' if fused else 'functional'} {h.kind}: traced decode {dt * 1e3:.3f} ms/step ({iters} replays)")
    ttnn.release_trace(device, tid)


@pytest.mark.perf
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576, "trace_region_size": 0}], indirect=True)
@pytest.mark.parametrize("fused", [False, True], ids=["functional", "fused"])
@pytest.mark.parametrize("layer_idx", [L0, L3], ids=["gdn", "attn"])
def test_perf_prefill_128(device, layer_idx, fused):
    _perf_prefill(device, layer_idx, 128, fused)


@pytest.mark.perf
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576, "trace_region_size": 64_000_000}], indirect=True)
@pytest.mark.parametrize("fused", [False, True], ids=["functional", "fused"])
@pytest.mark.parametrize("layer_idx", [L0, L3], ids=["gdn", "attn"])
def test_perf_traced_decode(device, layer_idx, fused):
    _perf_traced_decode(device, layer_idx, fused)
