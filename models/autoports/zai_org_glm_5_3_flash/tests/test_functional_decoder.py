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
    open_mesh,
    pcc,
    prefill_user,
)
from models.autoports.zai_org_glm_5_3_flash.tests.hf_ref.client import hf_layer_reference
from models.autoports.zai_org_glm_5_3_flash.tt.ckpt import CheckpointReader

KINDS = {"kda_dense": 0, "dsa_moe": 3, "kda_moe": 4}
BATCH = 8
DELTA_BAR = 0.995
FULL_BAR = 0.99

pytestmark = pytest.mark.skipif(not os.environ.get("GLM53_CKPT"), reason="needs the E192 checkpoint (GLM53_CKPT)")


@pytest.fixture(scope="module")
def mesh():
    m = open_mesh()
    yield m
    ttnn.close_mesh_device(m)


@pytest.fixture(scope="module")
def rd():
    return CheckpointReader()


_LAYERS = {}


def get_layer(mesh, rd, layer, weights="real"):
    key = (layer, str(weights))
    if key not in _LAYERS:
        for k in list(_LAYERS):  # keep one layer resident (DRAM)
            del _LAYERS[k]
        _LAYERS[key] = build_layer(mesh, rd, layer, weights, batch=BATCH)
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


def reference(layer, weights, seed, prefill_len, decode_steps):
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
    x = (r["x"] if "x" in r else req["x"]).float().reshape(n, H4)
    return x, r["y"].float().reshape(n, H4)


def check(tag, got, x, y, delta_bar=DELTA_BAR, full_bar=FULL_BAR):
    d = pcc(got - x, y - x)
    f = pcc(got, y)
    print(f"{tag}: delta PCC {d:.6f} full PCC {f:.6f}")
    assert d >= delta_bar, f"{tag}: delta PCC {d:.6f} < {delta_bar}"
    assert f >= full_bar, f"{tag}: full PCC {f:.6f} < {full_bar}"
    return d, f


def run_user(mesh, dec, layer, weights, seed, prefill_len, decode_steps, slot, traced=True):
    x, y = reference(layer, weights, seed, prefill_len, decode_steps)
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
        check(f"L{layer} slot {slot} decode pos {prefill_len + i}", o, x[prefill_len + i], y[prefill_len + i])
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
        t_in = ttnn.from_torch(
            torch.nn.functional.pad(fin.float(), (0, 0, 0, rows - n)).reshape(1, 1, rows, -1).bfloat16(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
        )
        got = ttnn.to_torch(dec.mlp.forward(t_in, mode="prefill"))[0, 0, :n].float()
        p = pcc(got, fout)
        print(f"L{layer} synthetic MoE on the HF FFN input: PCC {p:.6f}")
        assert p >= DELTA_BAR, p
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
            check(f"L{layer} user slot {s} decode pos {n + i}", got[s], x[n + i], y[n + i])
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
