# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage 1 (functional decoder) tests of the GLM-5.3-Flash autoport on one die, against the HF decoder layer.

Layer kinds: kda_dense (layer 0), dsa_moe (layer 3), kda_moe (layer 4). Every case runs chunked prefill for one or
more users (each in its own decode slot of an 8-slot layer), then decode steps: the first eager (compiles the
programs), the rest replayed from a captured trace on fixed-address inputs.

Acceptance (doc/functional_decoder/README.md): PCC of the layer update X' - X >= 0.995 for prefill rows and decode
steps; PCC of the full stream X' >= 0.99 (the HF reference rounds the 4-stream residual to bf16 at every mix, the
port keeps it fp32; doc measures that floor).
"""

from __future__ import annotations

import os

import pytest
import torch

import ttnn
from models.autoports.zai_org_glm_5_3_flash.tests.harness import (
    H4,
    DecodeRunner,
    build_layer,
    close_mesh,
    hidden_out,
    open_mesh,
    pcc,
    prefill_user,
    replicated_in,
    stream_in,
    stream_out,
)
from models.autoports.zai_org_glm_5_3_flash.tests.hf_ref.client import DryRunResult, flip_reference, hf_layer_reference
from models.autoports.zai_org_glm_5_3_flash.tt.ckpt import CheckpointReader

KINDS = {"kda_dense": 0, "dsa_moe": 3, "kda_moe": 4}
BATCH = 8
DELTA_BAR = 0.995
FULL_BAR = 0.99
# Router near-ties: a decoded token whose HF top-8 vs 9th expert selection margin is below NEAR_TIE can take a
# different 8th expert. The port's FFN input differs from HF's by ~1 % relative (PCC ~0.99994: HF rounds its 4-stream
# residual to bf16 at every mix, the port keeps fp32), which moves the router's selection scores sigmoid(l) + bias by up
# to ~2.5e-3 (sigmoid slope <= 0.25) -> margins of a few 1e-3 can flip. A flipped expert changes the token's function,
# not only its rounding: over 4606 L3 / 2812 L4 HF tokens, the 8th -> 9th swap moves the update scale by up to 17.7 %
# (L3; 99th percentile 7.9 %) and the delta PCC down to 0.870. So the HF reference also gives, for every decode row
# with margin < NEAR_TIE, y_swap: the HF layer output with the 8th expert replaced by the 9th, weights renormalized
# (tests/hf_ref/hf_flip_ref.py). Such a token passes when it meets the normal bars (DELTA_BAR, FULL_BAR, TOKEN_RMS_TOL)
# against y or against y_swap; a y_swap match is logged in SWAPS. A test may match y_swap on at most
# max(1, SWAP_MAX_FRAC x its MoE decode tokens) tokens, the module (suite) on at most
# max(SWAP_SUITE_MIN, SWAP_SUITE_FRAC x its MoE decode tokens): a router defect that flips near-tie tokens
# systematically fails. A token that matches neither fails.
NEAR_TIE = 5e-3
SWAP_MAX_FRAC = 0.05
SWAP_SUITE_FRAC = 0.03
SWAP_SUITE_MIN = 2
# Update scale |X' - X| TT / HF (PCC is scale-blind; review #1: the routed experts were 0.976 before the sparse_matmul
# fp32-partials fix). A prefill sequence is held to 1 +- SEQ_RMS_TOL, which that bug fails: injected again
# (GLM53_FAULT_ROUTED_SCALE=0.9722, the routed output x 0.9722; doc/functional_decoder/s1_fault_routed2.log), it fails
# the multi-row real-weight prefills of L3 and most of L4 and the L3 decode means (the per-test list is in the
# README). The bug's layer-level factor (faulted / clean update scale) is 0.977 .. 0.979 on L3 and 0.984 .. 0.991
# on L4 (real weights). One decode token is held to 1 +- TOKEN_RMS_TOL only: HF runs the layer in bf16 and its mHC
# post / comb mixing rounds, so a single token's update scale scatters around the fp32 result
# (tools/decode_stage_probe.py, layer 4: an fp32 host recomputation of HF's own mHC on HF's inputs misses HF's update
# by 0.978 .. 1.019). The mean over a test's decode tokens (at least MEAN_MIN_TOKENS; a y_swap match enters with its
# ratio against y_swap) is held to 1 +- SEQ_RMS_TOL.
SEQ_RMS_TOL = 0.01
TOKEN_RMS_TOL = 0.05
MEAN_MIN_TOKENS = 8
DECODE_RATIOS = []  # update rms ratio of every decode token of the running test
# Per-layer bar overrides {layer: {"delta": bar, "seq": tol, "token": tol}}: a precision policy whose weights are
# quantized below the checkpoint (bfp4 routed experts) is held to its measured quantization floor on the layers it
# changes (tests/test_optimized_decoder.py): every PCC bar of the layer (update, synthetic-weights) moves down
# by DELTA_BAR - delta, the scale bounds become seq / token. Empty here: the functional bars apply.
BAR_OVERRIDES = {}
SWAPS = []  # (tag, delta PCC vs y_swap, update rms ratio vs y_swap, HF margin) of every token that matched y_swap
MOE_TOKENS = []  # one entry per MoE decode token (HF margin known) of the running test
SUITE = {"swaps": 0, "moe_tokens": 0}

pytestmark = pytest.mark.skipif(not os.environ.get("GLM53_CKPT"), reason="needs the E192 checkpoint (GLM53_CKPT)")


@pytest.fixture(scope="module", autouse=True)
def suite_swap_cap():
    """Module-level y_swap cap, checked once at teardown (a running check fails early on a chance cluster)."""
    yield
    if SUITE["moe_tokens"]:
        suite_cap = max(SWAP_SUITE_MIN, int(SWAP_SUITE_FRAC * SUITE["moe_tokens"]))
        print(f"module y_swap matches {SUITE['swaps']} of {SUITE['moe_tokens']} MoE decode tokens (cap {suite_cap})")
        assert SUITE["swaps"] <= suite_cap, f"module y_swap matches {SUITE['swaps']} > {suite_cap}"


@pytest.fixture(autouse=True)
def decode_scale_mean():
    """Mean update scale over the test's decode tokens (see SEQ_RMS_TOL)."""
    DECODE_RATIOS.clear()
    MOE_TOKENS.clear()
    n_swaps = len(SWAPS)
    yield
    swaps = len(SWAPS) - n_swaps
    SUITE["swaps"] += swaps
    SUITE["moe_tokens"] += len(MOE_TOKENS)
    if MOE_TOKENS:
        cap = max(1, int(SWAP_MAX_FRAC * len(MOE_TOKENS)))
        suite_cap = max(SWAP_SUITE_MIN, int(SWAP_SUITE_FRAC * SUITE["moe_tokens"]))
        print(
            f"router near-tie y_swap matches: {swaps} of {len(MOE_TOKENS)} MoE decode tokens (cap {cap}); "
            f"module total {SUITE['swaps']} of {SUITE['moe_tokens']} (cap {suite_cap})"
        )
        assert swaps <= cap, f"{swaps} y_swap matches in {len(MOE_TOKENS)} MoE decode tokens > {cap}"
    if len(DECODE_RATIOS) >= MEAN_MIN_TOKENS:
        m = sum(r for _, r in DECODE_RATIOS) / len(DECODE_RATIOS)
        print(f"decode update rms ratio mean {m:.4f} over {len(DECODE_RATIOS)} tokens")
        tol = max([SEQ_RMS_TOL] + [BAR_OVERRIDES[ly]["seq"] for ly, _ in DECODE_RATIOS if ly in BAR_OVERRIDES])
        assert abs(m - 1) <= tol, f"decode update rms ratio mean {m:.4f} outside 1 +- {tol}"


@pytest.fixture(scope="module")
def mesh():
    m = open_mesh()
    yield m
    close_mesh(m)


@pytest.fixture(scope="module")
def rd():
    return CheckpointReader()


_LAYERS = {}
# decoder under test; tests/test_fused_decoder.py loads its own instance of this module with the fused decoder
DECODER_CLS = None


def get_layer(mesh, rd, layer, weights="real"):
    key = (layer, str(weights))
    if key not in _LAYERS:
        for k in list(_LAYERS):  # keep one layer resident (DRAM)
            del _LAYERS[k]
        _LAYERS[key] = build_layer(mesh, rd, layer, weights, batch=BATCH, decoder_cls=DECODER_CLS)
    return _LAYERS[key]


def tokens(seed: int, n: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 150000, (1, n), generator=g)


def synthetic_stream(layer: int, seed: int, n: int) -> torch.Tensor:
    """[1, n, 4, 4096] bf16 drawn per stream-channel from the recorded statistics of the real input stream of this
    layer kind (weight_stats.json "stream": per-channel mean / std over 512 real tokens; the 4 streams are identical
    at layer 0 and nearly uncorrelated from layer 3 on)."""
    import base64
    import json

    from models.autoports.zai_org_glm_5_3_flash.tests.harness import STATS
    from models.autoports.zai_org_glm_5_3_flash.tt.synthetic import stats_layer_for

    rec = json.loads(STATS.read_text())["stream"][str(stats_layer_for(layer))]
    dec = lambda k: torch.frombuffer(bytearray(base64.b64decode(rec[k])), dtype=torch.float16).float()
    mean, std = dec("mean_b64").reshape(4, 4096), dec("std_b64").reshape(4, 4096)
    g = torch.Generator().manual_seed(seed)
    if layer == 0:  # embedding copied into the four streams
        z = torch.randn(1, n, 1, 4096, generator=g).expand(1, n, 4, 4096)
    else:
        z = torch.randn(1, n, 4, 4096, generator=g)
    return (mean + std * z).to(torch.bfloat16)


def reference(layer, weights, seed, prefill_len, decode_steps, decode_rows=None):
    """HF reference of one sequence: prefill_len rows, then decode_steps single-row segments. decode_rows: the rows the
    test decodes on the port (default: the decode segments), whose near-tie y_swap is computed (make_ref)."""
    segs = [prefill_len] + [1] * decode_steps
    n = prefill_len + decode_steps
    if weights == "real":
        req = {"layer": layer, "weights": "real", "tokens": tokens(seed, n), "segments": segs}
    else:
        from models.autoports.zai_org_glm_5_3_flash.tests.harness import STATS

        req = {
            "layer": layer,
            "weights": {"synthetic": 0, "stats": str(STATS)},
            "x": synthetic_stream(layer, seed, n),
            "segments": segs,
        }
    r = hf_layer_reference(req)
    if isinstance(r, DryRunResult):
        return _DryRef()
    return make_ref(req, r, range(prefill_len, n) if decode_rows is None else decode_rows)


def make_ref(req, r, decode_rows):
    """Ref of a computed HF reference; for real-weight MoE layers, also y_swap of the decode_rows whose HF router
    margin is below NEAR_TIE (tests/hf_ref/hf_flip_ref.py)."""
    if isinstance(r, DryRunResult):
        return _DryRef()
    n = sum(req["segments"])
    x = (r["x"] if "x" in r else req["x"]).float().reshape(n, H4)
    margin = r.get("route_margin")
    swap = None
    if margin is not None and req["weights"] == "real":
        m = margin.reshape(-1).float()
        rows = sorted({int(i) for i in decode_rows if float(m[i]) < NEAR_TIE})
        swap = flip_reference(req, rows)
        if isinstance(swap, DryRunResult):
            return _DryRef()
    return Ref((x, r["y"].float().reshape(n, H4)), margin, swap)


class _DryRef(tuple):
    """Dry run (GLM53_REF_DRYRUN): the reference was only recorded; using it skips the test."""

    def __iter__(self):
        pytest.skip("reference recorded (GLM53_REF_DRYRUN)")

    __getitem__ = __iter__


class Ref(tuple):
    """(x, y) of a reference sequence; .margin = HF router margin per row (None for dense-MLP layers); .swap = {row:
    y_swap row} for the near-tie decode rows (None if there are none)."""

    def __new__(cls, xy, margin, swap=None):
        obj = super().__new__(cls, xy)
        obj.margin = None if margin is None else margin.reshape(-1).float()
        obj.swap = swap or {}
        return obj

    def row_margin(self, i):
        return None if self.margin is None else float(self.margin[i])

    def alt(self, i):
        """y_swap of row i (router 8th expert -> 9th), or None."""
        a = self.swap.get(i)
        return None if a is None else a.float().reshape(-1)


def check(tag, got, x, y, delta_bar=DELTA_BAR, full_bar=FULL_BAR, margin=None, y_alt=None):
    """margin: the HF router margin of this (single) token, if any. y_alt: HF's output with the router's 8th expert
    replaced by its 9th (given for margin < NEAR_TIE): the token passes the normal bars against y or against y_alt;
    a y_alt match is recorded in SWAPS."""
    layer = int(tag[1:].split(" ")[0]) if tag.startswith("L") else None
    ov = BAR_OVERRIDES.get(layer)
    if ov:  # every PCC bar of the layer moves down by the same quantization allowance
        delta_bar -= DELTA_BAR - ov["delta"]
    token = got.numel() == x.shape[-1]  # one decode token (else a prefill sequence)
    tol = TOKEN_RMS_TOL if token else SEQ_RMS_TOL
    if ov:
        tol = ov["token"] if token else ov["seq"]

    def stats(ref):
        d = pcc(got - x, ref - x)
        f = pcc(got, ref)
        r = float((got - x).norm() / (ref - x).norm())  # update scale vs HF (PCC is scale-blind)
        return d, f, r, d >= delta_bar and f >= full_bar and abs(r - 1) <= tol

    note = f" (HF router margin {margin:.1e})" if margin is not None else ""
    d, f, r, ok = stats(y)
    print(f"{tag}: delta PCC {d:.6f} full PCC {f:.6f} update rms ratio {r:.4f}{note}")
    if token and margin is not None:
        MOE_TOKENS.append(tag)
    if not ok and token and y_alt is not None and margin is not None and margin < NEAR_TIE:
        d2, f2, r2, ok2 = stats(y_alt)
        print(f"{tag}: vs y_swap (8th expert -> 9th): delta PCC {d2:.6f} full PCC {f2:.6f} update rms ratio {r2:.4f}")
        assert ok2, f"{tag}: matches neither y nor y_swap{note}"
        SWAPS.append((tag, d2, r2, margin))
        DECODE_RATIOS.append((layer, r2))
        print(f"{tag}: matched y_swap (router near-tie)")
        return d2, f2
    assert abs(r - 1) <= tol, f"{tag}: update rms ratio {r:.4f} outside 1 +- {tol}{note}"
    assert d >= delta_bar, f"{tag}: delta PCC {d:.6f} < {delta_bar}{note}"
    assert f >= full_bar, f"{tag}: full PCC {f:.6f} < {full_bar}"
    if token:
        DECODE_RATIOS.append((layer, r))
    return d, f


def run_user(mesh, dec, layer, weights, seed, prefill_len, decode_steps, slot, traced=True):
    ref = reference(layer, weights, seed, prefill_len, decode_steps)
    x, y = ref
    out = prefill_user(dec, mesh, x, prefill_len, slot)
    check(f"L{layer} slot {slot} prefill {prefill_len}", out, x[:prefill_len], y[:prefill_len])
    runner = DecodeRunner(dec, mesh, BATCH)
    outs = []
    for i in range(decode_steps):
        pos = [-1] * BATCH
        pos[slot] = prefill_len + i
        rows = {slot: x[prefill_len + i]}
        if i == 0 or not traced:
            o = runner.eager_step(rows, pos)[slot]
            if traced and decode_steps > 1:
                pos2 = list(pos)
                pos2[slot] += 1
                runner.capture({slot: x[prefill_len + 1]}, pos2)
        else:
            o = runner.traced_step(rows, pos)[slot]
        outs.append(o)
        check(
            f"L{layer} slot {slot} decode pos {prefill_len + i}",
            o,
            x[prefill_len + i],
            y[prefill_len + i],
            margin=ref.row_margin(prefill_len + i),
            y_alt=ref.alt(prefill_len + i),
        )
    runner.release()
    return out, outs


# --------------------------------------------------------------------------------------------- real weights
@pytest.mark.parametrize("kind", list(KINDS))
def test_real_weights(mesh, rd, kind):
    layer = KINDS[kind]
    dec = get_layer(mesh, rd, layer)
    run_user(mesh, dec, layer, "real", seed=1, prefill_len=100, decode_steps=3, slot=5)


# ---------------------------------------------------------------------------------------- synthetic weights
# Synthetic MoE layers route in a near-tie regime that real activations do not reach: on the synthetic stream 12-14 % of
# tokens have a top-8 vs 9th selection margin < 1e-3 (real stream: 8.7 %), so the ~6e-5 relative difference between
# the TT and HF FFN inputs flips experts of a few tokens (doc/functional_decoder/README.md, "Synthetic MoE"). For those
# layers the MoE block is checked strictly on the HF FFN input, and the layer update with a routing-sensitive bar.
SYNTH_MOE_LAYER_BAR = 0.98


@pytest.mark.parametrize("kind", list(KINDS))
def test_synthetic_weights(mesh, rd, kind):
    layer = KINDS[kind]
    weights = {"synthetic": 0}
    dec = get_layer(mesh, rd, layer, weights)
    moe = kind.endswith("moe")
    if moe:
        from models.autoports.zai_org_glm_5_3_flash.tests.harness import STATS

        n = 200
        r = hf_layer_reference(
            {
                "layer": layer,
                "weights": {"synthetic": 0, "stats": str(STATS)},
                "x": synthetic_stream(layer, 2, n + 3),
                "segments": [n, 1, 1, 1],
                "taps": True,
            }
        )
        fin = r["taps"]["ffn_in"].reshape(n, -1)
        fout = r["taps"]["ffn_out"].float().reshape(n, -1)
        rows = 256
        t_in = replicated_in(
            mesh, torch.nn.functional.pad(fin.float(), (0, 0, 0, rows - n)).reshape(1, 1, rows, -1).bfloat16()
        )
        got = hidden_out(dec.mlp.forward(t_in, mode="prefill"))[0, 0, :n].float()
        p = pcc(got, fout)
        print(f"L{layer} synthetic MoE on the HF FFN input: PCC {p:.6f}")
        assert p >= BAR_OVERRIDES.get(layer, {}).get("delta", DELTA_BAR), p  # one MoE: the update bar itself
    bar = SYNTH_MOE_LAYER_BAR if moe else DELTA_BAR
    x, y = reference(layer, weights, 2, 200, 3)
    out = prefill_user(dec, mesh, x, 200, 3)
    check(f"L{layer} slot 3 prefill 200", out, x[:200], y[:200], delta_bar=bar)
    runner = DecodeRunner(dec, mesh, BATCH)
    for i in range(3):
        pos = [-1] * BATCH
        pos[3] = 200 + i
        o = runner.eager_step({3: x[200 + i]}, pos)[3]
        check(f"L{layer} slot 3 decode pos {200 + i}", o, x[200 + i], y[200 + i], delta_bar=bar)


# ------------------------------------------------------------------------------ non-aligned logical lengths
LENGTHS = [
    ("kda_dense", 1),
    ("kda_dense", 31),
    ("kda_dense", 33),
    ("kda_dense", 128),
    ("kda_dense", 129),
    ("kda_dense", 2047),
    ("kda_dense", 2049),
    ("kda_dense", 4000),
    ("dsa_moe", 1),
    ("dsa_moe", 33),
    ("dsa_moe", 129),
    ("dsa_moe", 2049),  # decode crosses the dense -> sparse boundary at position 2051
    ("dsa_moe", 2051),
    ("dsa_moe", 2177),
    ("dsa_moe", 4000),
    ("kda_moe", 33),
    ("kda_moe", 2049),
]


@pytest.mark.parametrize("kind,length", LENGTHS, ids=[f"{k}-{n}" for k, n in LENGTHS])
def test_lengths(mesh, rd, kind, length):
    layer = KINDS[kind]
    dec = get_layer(mesh, rd, layer)
    run_user(mesh, dec, layer, "real", seed=1000 + length, prefill_len=length, decode_steps=3, slot=length % BATCH)


# ------------------------------------------------------------------------- several users decoding together
@pytest.mark.parametrize("kind", list(KINDS))
def test_batched_users(mesh, rd, kind):
    """Four users with different prompt lengths in slots 1, 3, 4, 6 (other slots inactive) prefill one after another,
    then decode three steps together (eager, then traced): every user must match its own HF sequence."""
    layer = KINDS[kind]
    dec = get_layer(mesh, rd, layer)
    users = {1: 37, 3: 100, 4: 2100 if kind == "dsa_moe" else 300, 6: 64}
    steps = 3
    refs = {s: reference(layer, "real", 500 + s, n, steps) for s, n in users.items()}
    for s, n in users.items():
        x, y = refs[s]
        out = prefill_user(dec, mesh, x, n, s)
        check(f"L{layer} user slot {s} prefill {n}", out, x[:n], y[:n])
    runner = DecodeRunner(dec, mesh, BATCH)
    for i in range(steps):
        pos = [-1] * BATCH
        rows = {}
        for s, n in users.items():
            pos[s] = n + i
            rows[s] = refs[s][0][n + i]
        if i == 0:
            got = runner.eager_step(rows, pos)
            nxt = {s: refs[s][0][n + 1] for s, n in users.items()}
            runner.capture(nxt, [p + 1 if p >= 0 else -1 for p in pos])
        else:
            got = runner.traced_step(rows, pos)
        for s, n in users.items():
            x, y = refs[s]
            check(
                f"L{layer} user slot {s} decode pos {n + i}",
                got[s],
                x[n + i],
                y[n + i],
                margin=refs[s].row_margin(n + i),
                y_alt=refs[s].alt(n + i),
            )
    runner.release()


# ------------------------------------------------------------------------------------------- determinism
@pytest.mark.parametrize("kind", list(KINDS))
def test_determinism(mesh, rd, kind):
    """Identical inputs twice (same slot, fresh sequence each time): prefill and traced decode outputs bit-identical."""
    layer = KINDS[kind]
    dec = get_layer(mesh, rd, layer)
    a_pre, a_dec = run_user(mesh, dec, layer, "real", seed=7, prefill_len=150, decode_steps=4, slot=2)
    b_pre, b_dec = run_user(mesh, dec, layer, "real", seed=7, prefill_len=150, decode_steps=4, slot=2)
    assert torch.equal(a_pre, b_pre), "prefill output differs between identical runs"
    for i, (a, b) in enumerate(zip(a_dec, b_dec)):
        assert torch.equal(a, b), f"decode step {i} differs between identical runs"


# ------------------------------------------------------------------------------------- host fallback audit
class _HostCall(RuntimeError):
    pass


@pytest.mark.parametrize("kind", list(KINDS))
def test_no_host_fallback(mesh, rd, kind, monkeypatch):
    """One prefill pass (prefill_forward) and one decode pass (decode_forward) with every host <-> device transfer entry
    point patched to raise: inputs are prepared before the pass, outputs read after it."""
    layer = KINDS[kind]
    dec = get_layer(mesh, rd, layer)
    x, _ = reference(layer, "real", 1, 100, 1)
    n = 128
    xp = torch.zeros(1, 1, n, H4)
    xp[0, 0, :100] = x[:100]
    X = stream_in(mesh, xp)
    inputs = dec.prefill_inputs(n, start=0, length=100, slot=4)
    runner = DecodeRunner(dec, mesh, BATCH)
    pos = [-1] * BATCH
    pos[4] = 100
    runner._set_inputs({4: x[100]}, pos)

    def boom(*a, **k):
        raise _HostCall("host transfer inside a pass")

    real = {name: getattr(ttnn, name) for name in ("from_torch", "to_torch", "copy_host_to_device_tensor")}
    for name in real:
        monkeypatch.setattr(ttnn, name, boom)
    try:
        Y, st = dec.prefill_forward(X, inputs=inputs)
        dec.commit_prefill(4, st)
        Yd = dec.decode_forward(runner.x_buf, inputs=runner.dsa_bufs)
    finally:
        for name, fn in real.items():
            monkeypatch.setattr(ttnn, name, fn)
    assert torch.isfinite(stream_out(Y)[0, 0, :100]).all()
    assert torch.isfinite(stream_out(Yd)[0, 0, 4]).all()


# ------------------------------------------------------------------------------- decode from an empty state
@pytest.mark.parametrize("kind", list(KINDS))
def test_decode_only(mesh, rd, kind):
    """Decode positions 0..3 of a fresh sequence (no prefill: zero KDA state and conv history, empty DSA cache), eager
    first step then traced. Also the watcher case of the KDA layers (whose prefill kernel does not fit the kernel
    config buffer once watcher-instrumented; doc/functional_decoder/README.md)."""
    layer = KINDS[kind]
    dec = get_layer(mesh, rd, layer)
    steps, slot = 4, 6
    req = {"layer": layer, "weights": "real", "tokens": tokens(77, steps), "segments": [1] * steps}
    ref = make_ref(req, hf_layer_reference(req), range(steps))
    x, y = ref
    if dec.attn_kind == "kda":  # fresh sequence: zero carry in the slot
        st = dec.attn.new_prefill_state()
        dec.commit_prefill(slot, st)
    runner = DecodeRunner(dec, mesh, BATCH)
    for i in range(steps):
        pos = [-1] * BATCH
        pos[slot] = i
        if i == 0:
            o = runner.eager_step({slot: x[0]}, pos)[slot]
            runner.capture({slot: x[1]}, [p + 1 if p >= 0 else -1 for p in pos])
        else:
            o = runner.traced_step({slot: x[i]}, pos)[slot]
        check(f"L{layer} slot {slot} decode-only pos {i}", o, x[i], y[i], margin=ref.row_margin(i), y_alt=ref.alt(i))
    runner.release()


# ------------------------------------------------------------------- paged DSA caches: permuted block table
def test_paged_permuted_blocks(mesh, rd):
    """DSA latent / pool-key caches through a random, interleaved, non-contiguous block table (every block of the pool
    in use, slots' blocks shuffled together): users in the sparse regime (4000), across the 2048 chunk (2100) and short
    (129) prefill and decode (eager, then traced) against HF."""
    layer = KINDS["dsa_moe"]
    dec = get_layer(mesh, rd, layer)
    attn = dec.attn
    bps = attn.T // 128
    perm = torch.randperm(attn.NB - 1, generator=torch.Generator().manual_seed(11))[: BATCH * bps].reshape(BATCH, bps)
    assert not torch.equal(perm, attn.default_page_table()[:, :bps])
    dec.set_page_table(perm)
    try:
        users = {0: 4000, 5: 2100, 7: 129}
        steps = 3
        refs = {s: reference(layer, "real", 900 + s, n, steps) for s, n in users.items()}
        for s, n in users.items():
            x, y = refs[s]
            check(f"L{layer} paged slot {s} prefill {n}", prefill_user(dec, mesh, x, n, s), x[:n], y[:n])
        runner = DecodeRunner(dec, mesh, BATCH)
        for i in range(steps):
            pos = [-1] * BATCH
            rows = {}
            for s, n in users.items():
                pos[s] = n + i
                rows[s] = refs[s][0][n + i]
            if i == 0:
                got = runner.eager_step(rows, pos)
                runner.capture({s: refs[s][0][n + 1] for s, n in users.items()}, [p + 1 if p >= 0 else -1 for p in pos])
            else:
                got = runner.traced_step(rows, pos)
            for s, n in users.items():
                x, y = refs[s]
                check(
                    f"L{layer} paged slot {s} decode pos {n + i}",
                    got[s],
                    x[n + i],
                    y[n + i],
                    margin=refs[s].row_margin(n + i),
                    y_alt=refs[s].alt(n + i),
                )
        runner.release()
    finally:
        dec.set_page_table(attn.default_page_table()[:, :bps])


# ------------------------------------------------------------------------- 32 active slots, mixed positions
@pytest.mark.parametrize("kind", list(KINDS))
def test_batch32_all_active(mesh, rd, kind):
    """All 32 decode slots active at mixed positions (8 sequences x 4 prompt lengths; DSA: two lengths past the
    dense / sparse switch at 2051), real weights: 32-row decode (the MoE takes its 32-row path), eager then traced."""
    layer = KINDS[kind]
    for k in list(_LAYERS):  # one layer resident (DRAM)
        del _LAYERS[k]
    dec = build_layer(mesh, rd, layer, "real", batch=32, decoder_cls=DECODER_CLS)
    lengths = [37, 300, 2100, 2300] if kind == "dsa_moe" else [37, 129, 300, 700]
    steps = 3
    rows = [n + i for n in lengths for i in range(steps)]  # the rows decoded (HF: rows of its prefill)
    refs = {seed: reference(layer, "real", 1300 + seed, max(lengths), steps, decode_rows=rows) for seed in range(8)}
    users = {}  # slot -> (seed, prompt length)
    for slot in range(32):
        users[slot] = (slot % 8, lengths[slot // 8])
    for slot, (seed, n) in users.items():
        x, y = refs[seed]
        prefill_user(dec, mesh, x, n, slot)
    runner = DecodeRunner(dec, mesh, 32)
    worst = 1.0
    for i in range(steps):
        pos = [n + i for _, (_, n) in sorted(users.items())]
        rows = {slot: refs[seed][0][n + i] for slot, (seed, n) in users.items()}
        if i == 0:
            got = runner.eager_step(rows, pos)
            runner.capture({slot: refs[seed][0][n + 1] for slot, (seed, n) in users.items()}, [p + 1 for p in pos])
        else:
            got = runner.traced_step(rows, pos)
        for slot, (seed, n) in users.items():
            x, y = refs[seed]
            d, _ = check(
                f"L{layer} 32-slot decode slot {slot} pos {n + i}",
                got[slot],
                x[n + i],
                y[n + i],
                margin=refs[seed].row_margin(n + i),
                y_alt=refs[seed].alt(n + i),
            )
            worst = min(worst, d)
    runner.release()
    print(f"L{layer} 32 active slots: min decode delta PCC {worst:.6f}")
    del dec


# ------------------------------------------------------------------------- inactive slots keep their state
@pytest.mark.parametrize("kind", list(KINDS))
def test_inactive_slot_state(mesh, rd, kind):
    """User B (slot 2, prompt 102: its last pool is open) is prefilled; user A (slot 5) then decodes 4 steps while B is
    inactive (position -1) with a NaN input row. B then decodes 2 steps together with A and must match HF: decode leaves an inactive slot's
    KDA state / conv history and DSA open-pool state and caches untouched."""
    layer = KINDS[kind]
    dec = get_layer(mesh, rd, layer)
    nb, na = 102, 64
    rb = reference(layer, "real", 41, nb, 2)
    ra = reference(layer, "real", 42, na, 6)
    (xb, yb), (xa, ya) = rb, ra
    prefill_user(dec, mesh, xb, nb, 2)
    prefill_user(dec, mesh, xa, na, 5)
    runner = DecodeRunner(dec, mesh, BATCH)
    nan_row = torch.full_like(xa[0], float("nan"))
    for i in range(4):
        pos = [-1] * BATCH
        pos[5] = na + i
        # the idle slot B gets a non-finite row (as a stacked model's idle rows may carry): its state must not change
        o = runner.eager_step({5: xa[na + i], 2: nan_row}, pos)
        check(
            f"L{layer} A alone pos {na + i}",
            o[5],
            xa[na + i],
            ya[na + i],
            margin=ra.row_margin(na + i),
            y_alt=ra.alt(na + i),
        )
    for i in range(2):
        pos = [-1] * BATCH
        pos[5], pos[2] = na + 4 + i, nb + i
        o = runner.eager_step({5: xa[na + 4 + i], 2: xb[nb + i]}, pos)
        check(
            f"L{layer} A pos {na + 4 + i}",
            o[5],
            xa[na + 4 + i],
            ya[na + 4 + i],
            margin=ra.row_margin(na + 4 + i),
            y_alt=ra.alt(na + 4 + i),
        )
        check(
            f"L{layer} B after idling, pos {nb + i}",
            o[2],
            xb[nb + i],
            yb[nb + i],
            margin=rb.row_margin(nb + i),
            y_alt=rb.alt(nb + i),
        )
