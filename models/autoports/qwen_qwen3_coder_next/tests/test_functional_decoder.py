# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Functional-decoder tests for Qwen/Qwen3-Coder-Next (stage 1).

Run on one Blackhole device (1x1 mesh)::

    export HF_MODEL=Qwen/Qwen3-Coder-Next
    pytest models/autoports/qwen_qwen3_coder_next/tests/test_functional_decoder.py -v -s
    # real weights (needs the local HF snapshot), long context, perf signposts:
    pytest ... -m real_weights ; pytest ... -m long ; python -m tracy -r -p -v -m pytest ... -m perf

Every test compares against the HF ``Qwen3_5MoeDecoderLayer`` (bf16, eager, CPU) with the real
config shapes. Synthetic weights are drawn from recorded real-weight statistics
(``weight_stats.json``); real-weight tests read the checkpoint layer directly.
"""
from __future__ import annotations

import ast
import os
import time
from pathlib import Path

import pytest
import torch

import ttnn
from models.autoports.qwen_qwen3_coder_next.tests import reference as ref
from models.autoports.qwen_qwen3_coder_next.tt.functional_decoder import FunctionalDecoder
from models.common.utility_functions import comp_pcc

os.environ.setdefault("HF_MODEL", "Qwen/Qwen3-Coder-Next")

MODEL_ID = "Qwen/Qwen3-Coder-Next"
PCC_BAR = 0.995
BLOCK_SIZE = 64
CHUNK = 2048  # prefill chunk for long prompts (start_pos increments)
DEVICE_PARAMS = {"l1_small_size": 24576, "trace_region_size": 64_000_000}
L0, L3 = 0, 3  # one layer per kind: linear_attention, full_attention

try:  # perf signposts are optional (Tracy)
    from tracy import signpost
except Exception:  # pragma: no cover

    def signpost(name):
        return None


# ---------------------------------------------------------------------------------------------------
def _pcc(label, expected, actual, bar=PCC_BAR):
    ok, pcc = comp_pcc(expected.float(), actual.float(), bar)
    print(f"{label}: PCC {pcc}")
    assert ok, f"{label}: PCC {pcc} < {bar}"
    return float(str(pcc).split()[-1]) if isinstance(pcc, str) else float(pcc)


def _tt(t, device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(t, dtype=dtype, layout=layout, device=device)


def _int32(t, device):
    return ttnn.from_torch(t.to(torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)


def _hidden(batch, seq, H, seed, scale=0.5):
    gen = torch.Generator().manual_seed(seed)
    return (torch.randn(batch, seq, H, generator=gen) * scale).to(torch.bfloat16)


def _state(text, layer_idx, real):
    return ref.real_layer_state(layer_idx) if real else ref.synthetic_layer_state(text, layer_idx, seed=layer_idx)


def _page_table(batch, blocks_per_seq, seed=0):
    gen = torch.Generator().manual_seed(seed)
    return torch.randperm(batch * blocks_per_seq, generator=gen).to(torch.int32).reshape(batch, blocks_per_seq)


class _Harness:
    """One HF layer + one TT layer with the paged/GDN state wired for `batch` users."""

    def __init__(
        self,
        device,
        layer_idx,
        *,
        batch,
        max_seq_len,
        real=False,
        max_batch_size=None,
        decoder_cls=FunctionalDecoder,
        decoder_kwargs=None,
        state=None,
        hf=True,
    ):
        self.cfg, self.text = ref.text_config()
        self.device = device
        self.layer_idx = layer_idx
        self.kind = self.text.layer_types[layer_idx]
        self.batch = batch
        if state is None:
            state = _state(self.text, layer_idx, real)
        self.hf = ref.torch_layer(self.text, layer_idx, state) if hf else None
        self.tt = decoder_cls.from_state_dict(
            state,
            hf_config=self.cfg,
            layer_idx=layer_idx,
            mesh_device=device,
            max_batch_size=max_batch_size or batch,
            max_seq_len=max_seq_len,
            **(decoder_kwargs or {}),
        )
        del state
        self.H = self.text.hidden_size
        if self.kind == "full_attention":
            self.hf_cache = ref.DynamicCache()
            self.tt.allocate_kv_cache(max_batch_size=batch, max_seq_len=max_seq_len, block_size=BLOCK_SIZE)
            self.page_table = _page_table(batch, self.tt.kv_cache["blocks_per_seq"], seed=layer_idx)
            self.page_table_dev = _int32(self.page_table, device)
        else:
            self.hf_cache = ref.LinearCache(layer_idx)
            self.tt.allocate_gdn_state(batch_size=batch)
            self.page_table = self.page_table_dev = None

    # -- HF ----------------------------------------------------------------------------------------
    def hf_prefill(self, hidden, start=0):
        B, T, _ = hidden.shape
        pos = torch.arange(start, start + T).reshape(1, T).expand(B, T)
        pe = ref.rotary(self.text, hidden, pos)
        mask = None
        if self.kind == "full_attention":
            mask = ref.causal_mask(B, T) if start == 0 else None
            if start > 0:  # chunk continuation: mask over the full [start+T] key range
                m = torch.full((B, 1, T, start + T), torch.finfo(torch.float32).min)
                m = torch.triu(m, diagonal=start + 1).to(torch.bfloat16)
                mask = m
        with torch.no_grad():
            return self.hf(hidden, position_embeddings=pe, attention_mask=mask, past_key_values=self.hf_cache)

    def hf_decode(self, hidden, pos_per_user):
        pe = ref.rotary(self.text, hidden, pos_per_user.reshape(-1, 1))
        with torch.no_grad():
            return self.hf(hidden, position_embeddings=pe, attention_mask=None, past_key_values=self.hf_cache)

    # -- TT ----------------------------------------------------------------------------------------
    def tt_prefill(self, hidden, start=0, user_id=0):
        out = self.tt.prefill_forward(
            _tt(hidden, self.device), start_pos=start, page_table=self.page_table_dev, user_id=user_id
        )
        return ttnn.to_torch(out)

    def decode_inputs(self, hidden, pos_per_user):
        x = _tt(hidden, self.device)  # [B, 1, H]
        cur = _int32(pos_per_user.reshape(-1), self.device)
        if self.kind == "full_attention":
            cos_h, sin_h = self.tt.rope_decode_host(pos_per_user)
            return x, dict(
                current_pos=cur,
                page_table=self.page_table_dev,
                cos=_tt(cos_h, self.device),
                sin=_tt(sin_h, self.device),
            )
        return x, dict(current_pos=cur)

    def tt_decode(self, hidden, pos_per_user):
        x, kw = self.decode_inputs(hidden, pos_per_user)
        return ttnn.to_torch(self.tt.decode_forward(x, **kw))


def _prefill_users(h: _Harness, hidden):
    """Prefill each user separately on TT (vLLM semantics) and all users at once on HF."""
    ref_out = h.hf_prefill(hidden)
    outs = [h.tt_prefill(hidden[u : u + 1], user_id=u) for u in range(hidden.shape[0])]
    return ref_out, torch.cat(outs, dim=0)


# ---------------------------------------------------------------------------------------------------
def test_target_decoder_contract():
    cfg, text = ref.text_config()
    assert cfg.architectures == ["Qwen3_5MoeForConditionalGeneration"]
    assert text.hidden_size == 3072 and text.num_hidden_layers == 48
    assert text.layer_types[L0] == "linear_attention" and text.layer_types[L3] == "full_attention"
    assert text.layer_types.count("full_attention") == 12
    assert text.num_experts == 256 and text.num_experts_per_tok == 8 and text.moe_intermediate_size == 1024
    assert text.num_attention_heads == 32 and text.num_key_value_heads == 2 and text.head_dim == 256
    assert text.max_position_embeddings == 262144
    assert text.rope_parameters["partial_rotary_factor"] == 0.25


@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize(
    ("layer_idx", "seq_len"),
    [(L0, 5), (L3, 5), (L0, 128), (L3, 128), (L0, 129), (L3, 129), (L0, 333), (L3, 333)],
    ids=[
        "gdn_seq5",
        "attn_seq5",
        "gdn_seq128_boundary",
        "attn_seq128_boundary",
        "gdn_seq129_across",
        "attn_seq129_across",
        "gdn_seq333",
        "attn_seq333",
    ],
)
def test_synthetic_prefill_decode_against_hf(device, layer_idx, seq_len):
    h = _Harness(device, layer_idx, batch=1, max_seq_len=4096)
    hidden = _hidden(1, seq_len, h.H, seed=100 + seq_len)
    ref_pre, tt_pre = _prefill_users(h, hidden)
    _pcc(f"{h.kind} prefill T={seq_len}", ref_pre, tt_pre)
    dec = _hidden(1, 1, h.H, seed=200 + seq_len)
    pos = torch.tensor([seq_len])
    _pcc(f"{h.kind} decode pos={seq_len}", h.hf_decode(dec, pos), h.tt_decode(dec, pos))


@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize(
    ("layer_idx", "batch"), [(L0, 2), (L3, 2), (L0, 32), (L3, 32)], ids=["gdn_b2", "attn_b2", "gdn_b32", "attn_b32"]
)
def test_synthetic_batched_decode_against_hf(device, layer_idx, batch):
    """All users prefilled one at a time with the same length, then one batched decode step (distinct-position case below)."""
    seq_len = 37
    h = _Harness(device, layer_idx, batch=batch, max_seq_len=1024)
    hidden = _hidden(batch, seq_len, h.H, seed=300 + batch)
    ref_pre, tt_pre = _prefill_users(h, hidden)
    _pcc(f"{h.kind} prefill B={batch}", ref_pre, tt_pre)
    dec = _hidden(batch, 1, h.H, seed=400 + batch)
    pos = torch.full((batch,), seq_len)
    _pcc(f"{h.kind} batched decode B={batch}", h.hf_decode(dec, pos), h.tt_decode(dec, pos))


@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("layer_idx", [L0, L3], ids=["gdn", "attn"])
def test_synthetic_traced_decode(device, layer_idx):
    """Capture one decode step as a trace; replay it for 3 further steps with refreshed inputs; compare to HF."""
    seq_len = 40
    h = _Harness(device, layer_idx, batch=1, max_seq_len=1024)
    hidden = _hidden(1, seq_len, h.H, seed=500)
    ref_pre, tt_pre = _prefill_users(h, hidden)
    _pcc(f"{h.kind} prefill", ref_pre, tt_pre)
    steps = [_hidden(1, 1, h.H, seed=600 + i) for i in range(4)]
    x, kw = h.decode_inputs(steps[0], torch.tensor([seq_len]))
    # compile
    out = h.tt.decode_forward(x, **kw)
    ttnn.deallocate(out)
    if h.kind == "linear_attention":  # compile pass advanced the GDN state: rebuild it from the prefill
        h.tt.reset_gdn_state()
        h.tt_prefill(hidden)
    # capture
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
        _pcc(f"{h.kind} traced decode step {i} pos={seq_len + i}", h.hf_decode(step, pos), got)
    ttnn.release_trace(device, tid)


@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("layer_idx", [L0, L3], ids=["gdn", "attn"])
def test_synthetic_repeated_input_determinism(device, layer_idx):
    seq_len = 65
    h = _Harness(device, layer_idx, batch=1, max_seq_len=1024)
    hidden = _hidden(1, seq_len, h.H, seed=700)
    a = h.tt_prefill(hidden)
    if h.kind == "linear_attention":
        h.tt.reset_gdn_state()
    b = h.tt_prefill(hidden)
    assert torch.equal(a, b), "prefill is not deterministic for identical inputs"
    dec = _hidden(1, 1, h.H, seed=701)
    pos = torch.tensor([seq_len])
    x, kw = h.decode_inputs(dec, pos)
    if h.kind == "linear_attention":
        d1 = ttnn.to_torch(h.tt.decode_forward(x, **kw))
        h.tt.reset_gdn_state()
        h.tt_prefill(hidden)
        d2 = ttnn.to_torch(h.tt.decode_forward(x, **kw))
    else:
        d1 = ttnn.to_torch(h.tt.decode_forward(x, **kw))
        d2 = ttnn.to_torch(h.tt.decode_forward(x, **kw))  # same position: cache slot rewritten identically
    assert torch.equal(d1, d2), "decode is not deterministic for identical inputs"


@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("layer_idx", [L0, L3], ids=["gdn", "attn"])
def test_synthetic_chunked_prefill_matches_hf(device, layer_idx):
    """Prefill 2048 + 2048 + 333 tokens as three start_pos chunks (the long-prompt path) vs one HF pass."""
    lens = [CHUNK, CHUNK, 333]
    total = sum(lens)
    h = _Harness(device, layer_idx, batch=1, max_seq_len=8192)
    hidden = _hidden(1, total, h.H, seed=800)
    ref_out = h.hf_prefill(hidden)
    outs, start = [], 0
    for n in lens:
        outs.append(h.tt_prefill(hidden[:, start : start + n], start=start))
        start += n
    _pcc(f"{h.kind} chunked prefill T={total}", ref_out, torch.cat(outs, dim=1))
    dec = _hidden(1, 1, h.H, seed=801)
    pos = torch.tensor([total])
    _pcc(f"{h.kind} decode after chunked prefill pos={total}", h.hf_decode(dec, pos), h.tt_decode(dec, pos))


@pytest.mark.real_weights
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize(
    ("layer_idx", "seq_len"),
    [(L0, 37), (L3, 37), (L0, 300), (L3, 300)],
    ids=["gdn_37", "attn_37", "gdn_300", "attn_300"],
)
def test_real_weight_prefill_decode_against_hf(device, layer_idx, seq_len):
    h = _Harness(device, layer_idx, batch=1, max_seq_len=4096, real=True)
    hidden = _hidden(1, seq_len, h.H, seed=900 + seq_len)
    ref_pre, tt_pre = _prefill_users(h, hidden)
    _pcc(f"REAL {h.kind} prefill T={seq_len}", ref_pre, tt_pre)
    dec = _hidden(1, 1, h.H, seed=901 + seq_len)
    pos = torch.tensor([seq_len])
    _pcc(f"REAL {h.kind} decode pos={seq_len}", h.hf_decode(dec, pos), h.tt_decode(dec, pos))


@pytest.mark.long
@pytest.mark.timeout(5400)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("layer_idx", [L0, L3], ids=["gdn", "attn"])
def test_full_context_capacity(device, layer_idx):
    """Advertised context: prefill 262144-37 tokens in 2048 chunks, decode at that position.

    HF parity at this length is not computable on this host (eager attention over 262K keys); the
    kernels are the same ones validated against HF up to 4429 tokens above. This test proves that the
    full context fits and runs (paged KV for 262144 positions, GDN state continuation), that outputs are
    finite and non-constant, and records the wall time.
    """
    cfg, text = ref.text_config()
    ctx = text.max_position_embeddings
    total = ctx - 37
    h = _Harness(device, layer_idx, batch=1, max_seq_len=ctx)
    t0 = time.time()
    start = 0
    last = None
    gen = torch.Generator().manual_seed(1000)
    while start < total:
        n = min(CHUNK, total - start)
        chunk = (torch.randn(1, n, h.H, generator=gen) * 0.5).to(torch.bfloat16)
        last = h.tt_prefill(chunk, start=start)
        start += n
    prefill_s = time.time() - t0
    assert torch.isfinite(last.float()).all() and last.float().std() > 0
    dec = _hidden(1, 1, h.H, seed=1001)
    pos = torch.tensor([total])
    t0 = time.time()
    out = h.tt_decode(dec, pos)
    decode_s = time.time() - t0
    assert torch.isfinite(out.float()).all() and out.float().std() > 0
    print(
        f"{h.kind}: prefill {total} tokens in {prefill_s:.1f}s ({total / prefill_s:.0f} tok/s), decode at {total}: {decode_s * 1e3:.1f} ms (eager)"
    )


@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("layer_idx", [L0, L3], ids=["gdn", "attn"])
def test_synthetic_batched_decode_distinct_positions(device, layer_idx):
    """4 users with different prompt lengths (5, 37, 129, 300); one batched decode step at their own positions.
    Reference: an independent HF layer run per user."""
    lens = [5, 37, 129, 300]
    B = len(lens)
    h = _Harness(device, layer_idx, batch=B, max_seq_len=1024)
    dec = _hidden(B, 1, h.H, seed=450)
    refs = []
    for u, n in enumerate(lens):
        hidden = _hidden(1, n, h.H, seed=460 + u)
        # per-user HF reference with its own cache
        hu = _Harness.__new__(_Harness)
        hu.__dict__.update(h.__dict__)
        hu.hf_cache = ref.DynamicCache() if h.kind == "full_attention" else ref.LinearCache(layer_idx)
        hu.hf_prefill(hidden)
        refs.append(hu.hf_decode(dec[u : u + 1], torch.tensor([n])))
        h.tt_prefill(hidden, user_id=u)
    pos = torch.tensor(lens)
    got = h.tt_decode(dec, pos)
    for u in range(B):
        _pcc(f"{h.kind} distinct-position decode user {u} pos={lens[u]}", refs[u], got[u : u + 1])


class _HostTransferCounter:
    """Counts ttnn.from_torch / ttnn.to_torch calls (the host-fallback boundaries) during a measured pass."""

    def __enter__(self):
        self.calls = []
        self._orig = (ttnn.from_torch, ttnn.to_torch)

        def wrap(name, fn):
            def inner(*a, **k):
                import traceback

                self.calls.append((name, traceback.extract_stack(limit=6)[-2]))
                return fn(*a, **k)

            return inner

        ttnn.from_torch = wrap("from_torch", self._orig[0])
        ttnn.to_torch = wrap("to_torch", self._orig[1])
        return self

    def __exit__(self, *exc):
        ttnn.from_torch, ttnn.to_torch = self._orig


@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("layer_idx", [L0, L3], ids=["gdn", "attn"])
def test_runtime_fallback_audit_device_passes(device, layer_idx):
    """No host transfers inside an aligned prefill pass, a chunk-continuation pass, or a decode pass.

    The only accepted boundary is the fixed-bucket GDN prefill mask for a NON-aligned length (valid_len):
    the qwen36 kernel builds a small one-hot selector on the host (ttnn_gated_deltanet.py). It is counted
    and reported here so the exception is explicit."""
    h = _Harness(device, layer_idx, batch=1, max_seq_len=4096)
    aligned = _hidden(1, 256, h.H, seed=1300)
    x = _tt(aligned, h.device)
    with _HostTransferCounter() as c:
        h.tt.prefill_forward(x, start_pos=0, page_table=h.page_table_dev, user_id=0)
    assert not c.calls, f"host transfers in aligned prefill: {[(n, f.filename, f.lineno) for n, f in c.calls]}"
    x2 = _tt(_hidden(1, 128, h.H, seed=1301), h.device)
    with _HostTransferCounter() as c:
        h.tt.prefill_forward(x2, start_pos=256, page_table=h.page_table_dev, user_id=0)
    assert not c.calls, f"host transfers in continuation prefill: {[(n, f.filename, f.lineno) for n, f in c.calls]}"
    dec = _hidden(1, 1, h.H, seed=1302)
    xd, kw = h.decode_inputs(dec, torch.tensor([384]))
    with _HostTransferCounter() as c:
        h.tt.decode_forward(xd, **kw)
    assert not c.calls, f"host transfers in decode: {[(n, f.filename, f.lineno) for n, f in c.calls]}"
    # non-aligned prefill: report the accepted valid_len mask boundary
    if h.kind == "linear_attention":
        h.tt.reset_gdn_state()
    x3 = _tt(_hidden(1, 37, h.H, seed=1303), h.device)
    with _HostTransferCounter() as c:
        h.tt.prefill_forward(x3, start_pos=0, page_table=h.page_table_dev, user_id=0)
    sites = sorted({(n, f.filename.split("/")[-1], f.lineno) for n, f in c.calls})
    print(f"{h.kind}: non-aligned prefill host transfers = {len(c.calls)} at {sites}")
    if h.kind == "full_attention":
        assert not c.calls, f"host transfers in non-aligned attention prefill: {sites}"
    else:
        assert all(n == "from_torch" and ("gated_deltanet" in f or "delta_rule" in f) for n, f, _ in sites), sites
        assert len(c.calls) <= 4, sites


@pytest.mark.long
@pytest.mark.timeout(7200)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize(
    ("layer_idx", "total"),
    [(L0, 65536 + 37), (L3, 16384 + 37), (L0, 262144 - 37)],
    ids=["gdn_65573", "attn_16421", "gdn_262107"],
)
def test_long_hf_parity(device, layer_idx, total):
    """HF parity at long lengths: GDN 65573 and 262107 tokens (chunked 2048, state carried), attention 16421 tokens.
    Limits: HF eager attention scores are B*Hq*T^2 fp32 (16421 -> 34 GB; 32K -> 137 GB on a 249 GB host with the
    layer weights resident), GDN is linear in T. Decode after the long prefill is compared too."""
    h = _Harness(device, layer_idx, batch=1, max_seq_len=total + 128)
    hidden = _hidden(1, total, h.H, seed=1400 + layer_idx)
    t0 = time.time()
    ref_out = h.hf_prefill(hidden)
    print(f"{h.kind}: HF prefill {total} tokens {time.time() - t0:.0f}s")
    outs, start = [], 0
    t0 = time.time()
    while start < total:
        n = min(CHUNK, total - start)
        outs.append(h.tt_prefill(hidden[:, start : start + n], start=start))
        start += n
    print(f"{h.kind}: TT chunked prefill {total} tokens {time.time() - t0:.1f}s")
    tt_out = torch.cat(outs, dim=1)
    _pcc(f"{h.kind} long prefill T={total} (last 2048 positions)", ref_out[:, -2048:], tt_out[:, -2048:])
    _pcc(f"{h.kind} long prefill T={total} (all positions)", ref_out, tt_out)
    dec = _hidden(1, 1, h.H, seed=1401)
    pos = torch.tensor([total])
    _pcc(f"{h.kind} decode after long prefill pos={total}", h.hf_decode(dec, pos), h.tt_decode(dec, pos))


def _tt_kv_by_position(h: _Harness, num_positions: int):
    """Read the paged K/V cache back through user 0's page table -> [n_kv, num_positions, head_dim] each."""
    bs = h.tt.kv_cache["block_size"]
    k = ttnn.to_torch(h.tt.kv_cache["k"])  # [num_blocks, n_kv, bs, D]
    v = ttnn.to_torch(h.tt.kv_cache["v"])
    blocks = h.page_table[0, : -(-num_positions // bs)].to(torch.long)
    k = k[blocks].permute(1, 0, 2, 3).reshape(k.shape[1], -1, k.shape[3])[:, :num_positions]
    v = v[blocks].permute(1, 0, 2, 3).reshape(v.shape[1], -1, v.shape[3])[:, :num_positions]
    return k, v


def _hf_kv(h: _Harness, layer_idx: int):
    layer = h.hf_cache.layers[layer_idx]
    return layer.keys[0], layer.values[0]  # [n_kv, T, D] (post k_norm / RoPE, as stored by HF)


def _check_kv_cache(h: _Harness, layer_idx: int, total: int, label: str):
    hk, hv = _hf_kv(h, layer_idx)
    tk, tv = _tt_kv_by_position(h, total)
    assert hk.shape[1] == total, (hk.shape, total)
    # every 512-position window separately: a chunk written to the wrong blocks fails its own window
    for s0 in range(0, total, 512):
        e = min(total, s0 + 512)
        _pcc(f"{label} K cache positions [{s0},{e})", hk[:, s0:e], tk[:, s0:e], 0.99)
        _pcc(f"{label} V cache positions [{s0},{e})", hv[:, s0:e], tv[:, s0:e], 0.99)


@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
def test_chunked_prefill_kv_cache_matches_hf(device):
    """Discriminating check for chunked paged prefill: the K/V written through a random-permutation page table for
    2048 + 2048 + 333 tokens must match HF's cache position by position (whole-layer PCC is residual-dominated
    for attention at long context, so it cannot see a chunk written to the wrong blocks)."""
    lens = [CHUNK, CHUNK, 333]
    total = sum(lens)
    h = _Harness(device, L3, batch=1, max_seq_len=8192)
    hidden = _hidden(1, total, h.H, seed=1500)
    h.hf_prefill(hidden)
    start = 0
    for n in lens:
        h.tt_prefill(hidden[:, start : start + n], start=start)
        start += n
    _check_kv_cache(h, L3, total, "chunked")
    dec = _hidden(1, 1, h.H, seed=1501)
    pos = torch.tensor([total])
    _pcc("decode after chunked prefill (layer)", h.hf_decode(dec, pos), h.tt_decode(dec, pos))


@pytest.mark.real_weights
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
def test_real_weight_chunked_attention_prefill(device):
    """Real weights: 2048 + 333 tokens as two start_pos chunks; layer PCC, K/V cache content, decode at 2381."""
    lens = [CHUNK, 333]
    total = sum(lens)
    h = _Harness(device, L3, batch=1, max_seq_len=4096, real=True)
    hidden = _hidden(1, total, h.H, seed=1600)
    ref_out = h.hf_prefill(hidden)
    outs, start = [], 0
    for n in lens:
        outs.append(h.tt_prefill(hidden[:, start : start + n], start=start))
        start += n
    _pcc(f"REAL full_attention chunked prefill T={total}", ref_out, torch.cat(outs, dim=1))
    _check_kv_cache(h, L3, total, "REAL chunked")
    dec = _hidden(1, 1, h.H, seed=1601)
    pos = torch.tensor([total])
    _pcc(f"REAL full_attention decode after chunked prefill pos={total}", h.hf_decode(dec, pos), h.tt_decode(dec, pos))


def test_runtime_fallback_audit_source():
    """No torch / from_torch / to_torch in the runtime forward paths of the TT modules."""
    root = Path(__file__).resolve().parents[1] / "tt"
    banned = {"from_torch", "to_torch", "torch"}
    findings = []
    for path in ["functional_decoder.py", "moe.py", "layer.py"]:
        tree = ast.parse((root / path).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name in {
                "forward",
                "prefill_forward",
                "decode_forward",
                "_route",
                "_routed_gather",
                "_routed_masked",
                "_shared",
                "_routing_matrix",
                "_swiglu_rows",
            }:
                for sub in ast.walk(node):
                    if isinstance(sub, ast.Attribute) and sub.attr in banned:
                        findings.append(f"{path}:{sub.lineno} {node.name} uses .{sub.attr}")
                    if isinstance(sub, ast.Name) and sub.id == "torch":
                        findings.append(f"{path}:{sub.lineno} {node.name} references torch")
    assert not findings, findings


# ---------------------------------------------------------------------------------------------------
# Perf (run under Tracy: python -m tracy -r -p -v -m pytest ... -m perf -k <case>)
def _flush_profiler(device):
    """Dump the device profiler buffers mid-run (only when profiling) so long windows keep every op."""
    if os.environ.get("TT_METAL_DEVICE_PROFILER") == "1" and hasattr(ttnn, "ReadDeviceProfiler"):
        ttnn.synchronize_device(device)
        ttnn.ReadDeviceProfiler(device)


def _perf_prefill(device, layer_idx, seq_len, chunk=None):
    """Warmed prefill of seq_len tokens. chunk: prefill in start_pos chunks with a profiler flush between them
    (the device profiler buffers hold ~500 ops; a 2048-token pass is ~750)."""
    h = _Harness(device, layer_idx, batch=1, max_seq_len=4096, real=True)
    hidden = _hidden(1, seq_len, h.H, seed=2000)
    chunk = chunk or seq_len

    def run():
        for s0 in range(0, seq_len, chunk):
            h.tt_prefill(hidden[:, s0 : s0 + chunk], start=s0)
            if chunk != seq_len:
                _flush_profiler(device)

    run()  # compile
    if h.kind == "linear_attention":
        h.tt.reset_gdn_state()
    ttnn.synchronize_device(device)
    _flush_profiler(device)
    signpost("PERF_PREFILL")
    run()
    ttnn.synchronize_device(device)
    signpost("PERF_PREFILL_END")


def _perf_traced_decode(device, layer_idx, iters=3):
    seq_len = 128
    h = _Harness(device, layer_idx, batch=1, max_seq_len=4096, real=True)
    h.tt_prefill(_hidden(1, seq_len, h.H, seed=2100))
    x, kw = h.decode_inputs(_hidden(1, 1, h.H, seed=2101), torch.tensor([seq_len]))
    out = h.tt.decode_forward(x, **kw)
    ttnn.deallocate(out)
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    out = h.tt.decode_forward(x, **kw)
    ttnn.end_trace_capture(device, tid, cq_id=0)
    ttnn.execute_trace(device, tid, cq_id=0, blocking=True)  # warm
    ttnn.synchronize_device(device)
    signpost("PERF_DECODE")
    t0 = time.time()
    for _ in range(iters):
        ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(device)
    dt = (time.time() - t0) / iters
    signpost("PERF_DECODE_END")
    print(f"{h.kind}: traced decode {dt * 1e3:.3f} ms/step (host wall, {iters} replays)")
    ttnn.release_trace(device, tid)


@pytest.mark.perf
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576, "trace_region_size": 0}], indirect=True)
@pytest.mark.parametrize("layer_idx", [L0, L3], ids=["gdn", "attn"])
def test_perf_prefill_128(device, layer_idx):
    _perf_prefill(device, layer_idx, 128)


@pytest.mark.perf
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576, "trace_region_size": 0}], indirect=True)
@pytest.mark.parametrize("layer_idx", [L0, L3], ids=["gdn", "attn"])
def test_perf_prefill_2048(device, layer_idx):
    _perf_prefill(device, layer_idx, 2048, chunk=512)


@pytest.mark.perf
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576, "trace_region_size": 64_000_000}], indirect=True)
@pytest.mark.parametrize("layer_idx", [L0, L3], ids=["gdn", "attn"])
def test_perf_traced_decode(device, layer_idx):
    _perf_traced_decode(device, layer_idx)
