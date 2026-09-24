# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Optimized-decoder tests (stage 3): the SELECTED policy vs HF (real + synthetic weights), batch, traced decode,
determinism, chunked prefill + K/V content, long HF parity with the selected KV dtype, runtime audit, stress; perf signposts.

    pytest models/autoports/qwen_qwen3_5_122b_a10b/tests/test_optimized_decoder.py -v -s
"""
from __future__ import annotations

import os
import time

import pytest
import torch

import ttnn
from models.autoports.qwen_qwen3_5_122b_a10b.tests.test_functional_decoder import (
    CHUNK,
    DEVICE_PARAMS,
    L0,
    L3,
    _check_kv_cache,
    _flush_profiler,
    _Harness,
    _hidden,
    _HostTransferCounter,
    _pcc,
    _prefill_users,
    _tt,
    signpost,
)
from models.autoports.qwen_qwen3_5_122b_a10b.tt.fused_decoder import FusedDecoder
from models.autoports.qwen_qwen3_5_122b_a10b.tt.optimized_decoder import SELECTED, OptimizedDecoder

os.environ.setdefault("HF_MODEL", "Qwen/Qwen3.5-122B-A10B")


def _opt(device, layer_idx, *, batch, max_seq_len, real=False, opt=None, hf=True, state=None):
    return _Harness(
        device,
        layer_idx,
        batch=batch,
        max_seq_len=max_seq_len,
        real=real,
        hf=hf,
        state=state,
        decoder_cls=OptimizedDecoder,
        decoder_kwargs={"opt": opt or SELECTED},
    )


@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize(
    ("layer_idx", "seq_len"),
    [(L0, 5), (L3, 5), (L0, 129), (L3, 129), (L0, 333), (L3, 333)],
    ids=["gdn_seq5", "attn_seq5", "gdn_seq129", "attn_seq129", "gdn_seq333", "attn_seq333"],
)
def test_opt_prefill_decode_against_hf(device, layer_idx, seq_len):
    h = _opt(device, layer_idx, batch=1, max_seq_len=4096)
    hidden = _hidden(1, seq_len, h.H, seed=100 + seq_len)
    ref_pre, tt_pre = _prefill_users(h, hidden)
    _pcc(f"OPT {h.kind} prefill T={seq_len}", ref_pre, tt_pre)
    dec = _hidden(1, 1, h.H, seed=200 + seq_len)
    pos = torch.tensor([seq_len])
    _pcc(f"OPT {h.kind} decode pos={seq_len}", h.hf_decode(dec, pos), h.tt_decode(dec, pos))


@pytest.mark.real_weights
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize(
    ("layer_idx", "seq_len"),
    [(L0, 37), (L3, 37), (L0, 300), (L3, 300)],
    ids=["gdn_37", "attn_37", "gdn_300", "attn_300"],
)
def test_opt_real_weight_against_hf(device, layer_idx, seq_len):
    h = _opt(device, layer_idx, batch=1, max_seq_len=4096, real=True)
    hidden = _hidden(1, seq_len, h.H, seed=900 + seq_len)
    ref_pre, tt_pre = _prefill_users(h, hidden)
    _pcc(f"OPT REAL {h.kind} prefill T={seq_len}", ref_pre, tt_pre)
    dec = _hidden(1, 1, h.H, seed=901 + seq_len)
    pos = torch.tensor([seq_len])
    _pcc(f"OPT REAL {h.kind} decode pos={seq_len}", h.hf_decode(dec, pos), h.tt_decode(dec, pos))


@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize(("layer_idx", "batch"), [(L0, 32), (L3, 32)], ids=["gdn_b32", "attn_b32"])
def test_opt_batched_decode_against_hf(device, layer_idx, batch):
    seq_len = 37
    h = _opt(device, layer_idx, batch=batch, max_seq_len=1024)
    hidden = _hidden(batch, seq_len, h.H, seed=300 + batch)
    ref_pre, tt_pre = _prefill_users(h, hidden)
    _pcc(f"OPT {h.kind} prefill B={batch}", ref_pre, tt_pre)
    dec = _hidden(batch, 1, h.H, seed=400 + batch)
    pos = torch.full((batch,), seq_len)
    _pcc(f"OPT {h.kind} batched decode B={batch}", h.hf_decode(dec, pos), h.tt_decode(dec, pos))


@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("layer_idx", [L0, L3], ids=["gdn", "attn"])
def test_opt_traced_decode_and_determinism(device, layer_idx):
    seq_len = 40
    h = _opt(device, layer_idx, batch=1, max_seq_len=1024)
    hidden = _hidden(1, seq_len, h.H, seed=500)
    ref_pre, tt_pre = _prefill_users(h, hidden)
    _pcc(f"OPT {h.kind} prefill", ref_pre, tt_pre)
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
        _pcc(f"OPT {h.kind} traced decode step {i}", h.hf_decode(step, pos), ttnn.to_torch(out))
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
        if not torch.equal(p1, p2):
            d = (p1.float() - p2.float()).abs()
            print(
                f"prefill determinism: {int((d > 0).sum())} of {d.numel()} elements differ, max abs {d.max().item():.3e}"
            )
        assert torch.equal(p1, p2), "prefill not deterministic"


@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("layer_idx", [L0, L3], ids=["gdn", "attn"])
def test_opt_chunked_prefill_matches_hf(device, layer_idx):
    """2048 + 2048 + 333 start_pos chunks vs HF; attention also checks the (selected-dtype) K/V cache content."""
    lens = [CHUNK, CHUNK, 333]
    total = sum(lens)
    h = _opt(device, layer_idx, batch=1, max_seq_len=8192)
    hidden = _hidden(1, total, h.H, seed=800)
    ref_out = h.hf_prefill(hidden)
    outs, start = [], 0
    for n in lens:
        outs.append(h.tt_prefill(hidden[:, start : start + n], start=start))
        start += n
    _pcc(f"OPT {h.kind} chunked prefill T={total}", ref_out, torch.cat(outs, dim=1))
    if h.kind == "full_attention":
        _check_kv_cache(h, layer_idx, total, "OPT chunked")
    dec = _hidden(1, 1, h.H, seed=801)
    pos = torch.tensor([total])
    _pcc(f"OPT {h.kind} decode after chunked prefill pos={total}", h.hf_decode(dec, pos), h.tt_decode(dec, pos))


@pytest.mark.long
@pytest.mark.timeout(7200)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize(
    ("layer_idx", "total", "variant"),
    [
        (L3, 16384 + 37, None),
        (L0, 65536 + 37, None),
        pytest.param(
            L0,
            65536 + 37,
            "gdn_bfp4",
            marks=pytest.mark.xfail(
                strict=True,
                reason="rejected candidate kept as evidence (synthetic weights): bfp4 GDN in-projection weights fall below "
                "the 0.995 bar at 65K tokens (last 2048: 0.99387; bfp8 0.99732), 2026-09-24",
            ),
        ),
        (L3, 16384 + 37, "real"),
        (L0, 65536 + 37, "real"),
        pytest.param(
            L0,
            65536 + 37,
            "real_bfp4",
            marks=pytest.mark.xfail(
                strict=True,
                reason="rejected candidate kept as evidence: bfp4 GDN in-projection weights on REAL weights fall below the "
                "0.995 bar at 65K tokens (last 2048: 0.99304; bfp8 0.99552), 2026-09-24",
            ),
        ),
    ],
    ids=["attn_16421", "gdn_65573", "gdn_65573_bfp4_mega", "attn_16421_real", "gdn_65573_real", "gdn_65573_real_bfp4"],
)
def test_opt_long_hf_parity(device, layer_idx, total, variant):
    """Long HF parity with the SELECTED policy (incl. the selected KV cache dtype); decode after the long prefill."""
    variant = variant or ""
    opt = SELECTED.with_(gdn_weight_dtype=ttnn.bfloat4_b) if variant.endswith("bfp4") else SELECTED
    h = _opt(device, layer_idx, batch=1, max_seq_len=total + 128, opt=opt, real=variant.startswith("real"))
    hidden = _hidden(1, total, h.H, seed=1400 + layer_idx)
    ref_out = h.hf_prefill(hidden)
    outs, start = [], 0
    while start < total:
        n = min(CHUNK, total - start)
        outs.append(h.tt_prefill(hidden[:, start : start + n], start=start))
        start += n
    tt_out = torch.cat(outs, dim=1)
    _pcc(f"OPT {h.kind} long prefill T={total} (last 2048)", ref_out[:, -2048:], tt_out[:, -2048:])
    _pcc(f"OPT {h.kind} long prefill T={total} (all)", ref_out, tt_out)
    if h.kind == "full_attention":
        _check_kv_cache(h, layer_idx, total, "OPT long")
    dec = _hidden(1, 1, h.H, seed=1401)
    pos = torch.tensor([total])
    _pcc(f"OPT {h.kind} decode after long prefill pos={total}", h.hf_decode(dec, pos), h.tt_decode(dec, pos))


@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("layer_idx", [L0, L3], ids=["gdn", "attn"])
def test_opt_decode_then_chunk_continuation(device, layer_idx):
    """Prefill 128, decode 128 single tokens (positions 128..255), then a 100-token chunk continuation at start_pos 256:
    the GDN conv window / recurrent state and the paged KV cache carry across decode into the continuation."""
    T1, ND, T2 = 128, 128, 100
    total = T1 + ND + T2
    h = _opt(device, layer_idx, batch=1, max_seq_len=2048)
    hidden = _hidden(1, total, h.H, seed=2600)
    ref_out = h.hf_prefill(hidden)
    out1 = h.tt_prefill(hidden[:, :T1])
    _pcc(f"OPT {h.kind} prefill T={T1}", ref_out[:, :T1], out1)
    dec_outs = []
    for i in range(ND):
        pos = T1 + i
        dec_outs.append(h.tt_decode(hidden[:, pos : pos + 1], torch.tensor([pos])))
    _pcc(f"OPT {h.kind} {ND} decode steps", ref_out[:, T1 : T1 + ND], torch.cat(dec_outs, dim=1))
    out3 = h.tt_prefill(hidden[:, T1 + ND :], start=T1 + ND)
    _pcc(f"OPT {h.kind} continuation T={T2} at start_pos {T1 + ND} after decode", ref_out[:, T1 + ND :], out3)


@pytest.mark.real_weights
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize(
    ("layer_idx", "seq_len"), [(L0, 37), (L3, 333), (L0, 1023)], ids=["gdn_37", "attn_333", "gdn_1023"]
)
def test_opt_sorted_prefill_matches_masked(device, layer_idx, seq_len):
    """Expert-sorted dispatch prefill == masked-union prefill on real weights (same experts, same routing weights)."""
    base = SELECTED.with_(prefill_mode="masked")
    h_m = _opt(device, layer_idx, batch=1, max_seq_len=4096, real=True, hf=False, opt=base)
    hidden = _hidden(1, seq_len, h_m.H, seed=2400 + seq_len)
    out_m = h_m.tt_prefill(hidden)
    del h_m
    h_s = _opt(
        device,
        layer_idx,
        batch=1,
        max_seq_len=4096,
        real=True,
        hf=True,
        opt=SELECTED.with_(prefill_mode="sorted", sorted_min_tokens=1),
    )
    out_s = h_s.tt_prefill(hidden)
    _pcc(f"OPT sorted vs masked {h_s.kind} T={seq_len}", out_m, out_s, bar=0.999)
    _pcc(f"OPT sorted vs HF {h_s.kind} T={seq_len}", h_s.hf_prefill(hidden), out_s)


@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("layer_idx", [L0, L3], ids=["gdn", "attn"])
def test_opt_runtime_fallback_audit(device, layer_idx):
    h = _opt(device, layer_idx, batch=1, max_seq_len=4096, hf=False)
    for label, x, start in (
        ("aligned prefill", _tt(_hidden(1, 256, h.H, seed=1700), h.device), 0),
        ("continuation prefill", _tt(_hidden(1, 128, h.H, seed=1701), h.device), 256),
        ("sorted-dispatch prefill", _tt(_hidden(1, 640, h.H, seed=1703), h.device), 384),
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
def test_opt_stress_repeated_runs(device, layer_idx):
    h = _opt(device, layer_idx, batch=1, max_seq_len=2048, hf=False)
    lens = [5, 37, 128, 129, 333, 600, 1023, 1024, 1500] * 2 + [37, 37]
    ref_pair = None
    for i, n in enumerate(lens):
        if h.kind == "linear_attention":
            h.tt.reset_gdn_state()
        out = h.tt_prefill(_hidden(1, n, h.H, seed=1800 + n))
        assert torch.isfinite(out.float()).all()
        dec = h.tt_decode(_hidden(1, 1, h.H, seed=1900 + n), torch.tensor([n]))
        assert torch.isfinite(dec.float()).all()
        if n == 37 and i >= len(lens) - 2:
            if ref_pair is None:
                ref_pair = (out, dec)
            else:
                for label, a, b in (("prefill", ref_pair[0], out), ("decode", ref_pair[1], dec)):
                    if not torch.equal(a, b):
                        d = (a.float() - b.float()).abs()
                        print(
                            f"stress determinism {label}: {int((d > 0).sum())} of {d.numel()} elements differ, max abs {d.max().item():.3e}"
                        )
                assert torch.equal(ref_pair[0], out) and torch.equal(ref_pair[1], dec)


# ---- perf: fused (before) vs optimized (after) --------------------------------------------------------------
def _mk(device, layer_idx, optimized, ctx_len=4096):
    if optimized:
        return _opt(device, layer_idx, batch=1, max_seq_len=ctx_len, real=True, hf=False)
    return _Harness(device, layer_idx, batch=1, max_seq_len=ctx_len, real=True, hf=False, decoder_cls=FusedDecoder)


def _perf_prefill(device, layer_idx, seq_len, optimized, chunk=None):
    """Warmed prefill; chunk = start_pos chunks with a profiler flush between them (device profiler buffers hold ~500 ops)."""
    h = _mk(device, layer_idx, optimized)
    hidden = _hidden(1, seq_len, h.H, seed=2000)
    chunk = chunk or seq_len

    def run():
        for s0 in range(0, seq_len, chunk):
            h.tt_prefill(hidden[:, s0 : s0 + chunk], start=s0)
            if chunk != seq_len:
                _flush_profiler(device)

    run()
    if h.kind == "linear_attention":
        h.tt.reset_gdn_state()
    ttnn.synchronize_device(device)
    _flush_profiler(device)
    t0 = time.time()
    signpost("PERF_PREFILL")
    run()
    ttnn.synchronize_device(device)
    signpost("PERF_PREFILL_END")
    print(
        f"{'optimized' if optimized else 'fused'} {h.kind}: prefill T={seq_len} wall {1e3 * (time.time() - t0):.1f} ms"
    )


def _perf_traced_decode(device, layer_idx, optimized, iters=3):
    seq_len = 128
    h = _mk(device, layer_idx, optimized)
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
    print(f"{'optimized' if optimized else 'fused'} {h.kind}: traced decode {dt * 1e3:.3f} ms/step ({iters} replays)")
    ttnn.release_trace(device, tid)


@pytest.mark.perf
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576, "trace_region_size": 0}], indirect=True)
@pytest.mark.parametrize("optimized", [False, True], ids=["fused", "optimized"])
@pytest.mark.parametrize("layer_idx", [L0, L3], ids=["gdn", "attn"])
def test_perf_prefill_128(device, layer_idx, optimized):
    _perf_prefill(device, layer_idx, 128, optimized)


@pytest.mark.perf
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576, "trace_region_size": 0}], indirect=True)
@pytest.mark.parametrize("optimized", [False, True], ids=["fused", "optimized"])
@pytest.mark.parametrize("layer_idx", [L0, L3], ids=["gdn", "attn"])
def test_perf_prefill_2048(device, layer_idx, optimized):
    _perf_prefill(device, layer_idx, 2048, optimized, chunk=512)


@pytest.mark.perf
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576, "trace_region_size": 64_000_000}], indirect=True)
@pytest.mark.parametrize("optimized", [False, True], ids=["fused", "optimized"])
@pytest.mark.parametrize("layer_idx", [L0, L3], ids=["gdn", "attn"])
def test_perf_traced_decode(device, layer_idx, optimized):
    _perf_traced_decode(device, layer_idx, optimized)
