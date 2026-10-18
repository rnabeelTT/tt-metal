# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage-4 review #1 P1-1: the block-sharded DSA latent cache (tt/dsa_shard.py, MCOpts.dsa_shard) against the
replicated layout on the same (1, 4) mesh, same weights (layer 3, DSA + MoE), same inputs.

Each case builds the layer twice (replicated, then sharded) with the MCOpts of the run (DEFAULT_MC, or GLM53_MC on top),
installs the same randomly permuted GLOBAL block table (every die owns an irregular block subset), prefills several
users (flash first chunk, sparse chunks past position 2050, non-aligned lengths), then decodes all of them together
(one eager step, then traced replays). Compared on the layer updates X' - X: PCC >= 0.999 and |rms ratio - 1| <= 0.005
per prefill sequence and per decode token, on the ATTENTION sub-block output (the DSA module's reduce-scattered y,
captured by a tap; the layout change lives there). The layer update is printed and held to the same bars, except that a
token / sequence whose attention output passes and whose layer update does not is counted as a MoE router near-tie
(a ~1e-4 relative change of the FFN input flips the 8th expert of a near-tie token: tests/test_functional_decoder.py
NEAR_TIE; doc/multichip_decoder/README.md, diagnosis of the first A/B run), at most max(2, 10 %) of the compared keys.
Also printed: wall time of the prefill per user and of a traced decode replay (one layer; host sync per step).
"""

from __future__ import annotations

import dataclasses
import os
import time

os.environ.setdefault("GLM53_TP", "4")
os.environ.pop("GLM53_DSA_SHARD", None)  # the cases choose the layout themselves

import json  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

import ttnn  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tests.harness import (  # noqa: E402
    H4,
    STATS,
    DecodeRunner,
    build_layer,
    close_mesh,
    hidden_out,
    open_mesh,
    pcc,
    prefill_user,
)
from models.autoports.zai_org_glm_5_3_flash.tt.ckpt import CheckpointReader  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tt.multichip_decoder import (  # noqa: E402
    DEFAULT_MC,
    MCOpts,
    MultichipDecoder,
)
from models.autoports.zai_org_glm_5_3_flash.tt.synthetic import synthetic_stream_segment  # noqa: E402

pytestmark = pytest.mark.skipif(not os.environ.get("GLM53_CKPT"), reason="needs the E192 checkpoint (GLM53_CKPT)")

LAYER = 3
PCC_BAR = 0.999
RMS_TOL = 0.005
STEPS = 3  # one eager step, two traced replays (the cached real-token reference has 3 decode rows)
TIMED_REPLAYS = 20


@pytest.fixture(scope="module")
def mesh():
    m = open_mesh()
    yield m
    close_mesh(m)


@pytest.fixture(scope="module")
def rd():
    return CheckpointReader()


def _cls(shard: bool):
    mc = dataclasses.replace(MCOpts.from_env(DEFAULT_MC), dsa_shard=shard)

    class _C(MultichipDecoder):
        @classmethod
        def from_state_dict(cls, *a, **kw):
            kw["mc"] = mc
            return super().from_state_dict(*a, **kw)

    return _C


def _synthetic(seed: int, n: int) -> torch.Tensor:
    stats = json.loads(STATS.read_text())
    return synthetic_stream_segment(stats, LAYER, seed, 0, n).float().reshape(n, H4)


def _real(n: int, seed: int) -> torch.Tensor | None:
    """The real-token input stream of a cached HF reference (test_lengths dsa_moe-4000: seed 5000), or None if it is not
    cached (never computed here: the device lock is held)."""
    from models.autoports.zai_org_glm_5_3_flash.tests.hf_ref.client import cache_path

    g = torch.Generator().manual_seed(seed)
    req = {
        "layer": LAYER,
        "weights": "real",
        "tokens": torch.randint(0, 150000, (1, n + 3), generator=g),
        "segments": [n, 1, 1, 1],
    }
    p = cache_path(req)
    if not p.exists():
        return None
    r = torch.load(p)
    return r["x"].float().reshape(n + 3, H4)


class _Tap:
    """Wraps the decoder's attention module: keeps the attention sub-block output (prefill: host rows per chunk; decode:
    a persistent device copy, written inside the captured trace too)."""

    def __init__(self, inner):
        self.__dict__["inner"] = inner
        self.__dict__["chunks"] = []
        self.__dict__["buf"] = None

    def __getattr__(self, k):
        return getattr(self.inner, k)

    def __setattr__(self, k, v):
        setattr(self.inner, k, v)

    def prefill_forward(self, *a, **kw):
        y, st = self.inner.prefill_forward(*a, **kw)
        self.chunks.append(hidden_out(y)[0, 0].float())
        return y, st

    def decode_forward(self, *a, **kw):
        y = self.inner.decode_forward(*a, **kw)
        if self.buf is None:  # eager step 0: allocated outside any trace
            self.__dict__["buf"] = ttnn.clone(y, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.copy(y, self.buf)  # also eagerly once: the copy program compiles before the trace capture
        return y

    def decode_rows(self):
        return hidden_out(self.buf)[0, 0].float()


def _run(mesh, rd, shard, batch, max_context, users, perm_seed):
    dec = build_layer(
        mesh,
        rd,
        LAYER,
        "real",
        batch=batch,
        prefill_lengths=(128, 2048),
        max_context=max_context,
        decoder_cls=_cls(shard),
    )
    tap = _Tap(dec.attn)
    dec.attn = tap
    assert getattr(dec.attn, "dsa_shard", False) == shard
    bps = max_context // 128
    g = torch.Generator().manual_seed(perm_seed)
    perm = torch.randperm(dec.attn.NB - 1, generator=g)[: batch * bps].reshape(batch, bps).to(torch.int32)
    dec.set_page_table(perm)
    out, times = {}, {}
    for slot, x, L in users:
        ttnn.synchronize_device(mesh)
        t0 = time.time()
        tap.chunks.clear()
        out[("prefill", slot)] = prefill_user(dec, mesh, x, L, slot) - x[:L]
        out[("attn prefill", slot)] = torch.cat(tap.chunks, dim=0)[:L]
        times[("prefill", slot, L)] = time.time() - t0
    runner = DecodeRunner(dec, mesh, batch)

    def step_inputs(i):
        pos = [-1] * batch
        rows = {}
        for slot, x, L in users:
            pos[slot] = L + i
            rows[slot] = x[L + i]
        return rows, pos

    for i in range(STEPS):
        rows, pos = step_inputs(i)
        if i == 0:
            o = runner.eager_step(rows, pos)
            runner.capture(*step_inputs(1))
        else:
            o = runner.traced_step(rows, pos)
        att = tap.decode_rows()
        for slot, x, L in users:
            out[("decode", slot, i)] = o[slot] - x[L + i]
            out[("attn decode", slot, i)] = att[slot]
    # timing: replays of the captured step on fixed inputs (positions of the last step: state advances are irrelevant
    # for timing; the outputs are not compared)
    rows, pos = step_inputs(STEPS - 1)
    runner.traced_step(rows, pos)
    ttnn.synchronize_device(mesh)
    t0 = time.time()
    for _ in range(TIMED_REPLAYS):
        ttnn.execute_trace(mesh, runner.trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh)
    times["decode_replay_us"] = (time.time() - t0) / TIMED_REPLAYS * 1e6
    runner.release()
    a = dec.attn
    times["latent_bytes_per_die"] = int(a.latent.shape[-2]) * int(a.latent.shape[-1]) * 2
    del runner, dec, tap
    return out, times


CASES = {
    # 8 slots: a 9001-token user (4 sparse chunks), a 4000-token real-token user (flash + sparse), a 2049-token user
    # whose decode crosses the dense -> sparse boundary at 2051, a 100-token user (flash + dense decode)
    "b8": dict(
        batch=8,
        max_context=16384,
        users=[(5, "syn:77", 9001), (2, "real:4000", 4000), (0, "syn:78", 2049), (7, "syn:79", 100)],
    ),
    # the 1-slot serving profile (16 query rows per die in the all-gather / reduce-scatter)
    "b1": dict(batch=1, max_context=8192, users=[(0, "syn:80", 6001)]),
}


@pytest.mark.parametrize("case", list(CASES))
def test_shard_matches_replicated(mesh, rd, case):
    cfg = CASES[case]
    users = []
    for slot, src, L in cfg["users"]:
        kind, arg = src.split(":")
        x = _real(L, 1000 + L) if kind == "real" else None
        if x is None:
            x = _synthetic(int(arg) if kind == "syn" else 81, L + STEPS)
            src = f"syn (real stream not cached)" if kind == "real" else src
        users.append((slot, x, L))
        print(f"{case}: slot {slot} {src} prefill {L}")
    ref, t_ref = _run(mesh, rd, False, cfg["batch"], cfg["max_context"], users, perm_seed=11)
    got, t_got = _run(mesh, rd, True, cfg["batch"], cfg["max_context"], users, perm_seed=11)
    worst_p, worst_r = 1.0, 0.0
    fails, near_tie, n_layer = [], [], 0
    for k in sorted(ref, key=lambda k: not k[0].startswith("attn")):  # attention keys first
        a, b = ref[k], got[k]
        assert torch.isfinite(b).all(), k
        p = pcc(b, a)
        r = float(b.norm() / a.norm())
        ok = p >= PCC_BAR and abs(r - 1) <= RMS_TOL
        print(f"{case} {k}: PCC sharded vs replicated {p:.6f} rms ratio {r:.5f}")
        if k[0].startswith("attn"):
            worst_p, worst_r = min(worst_p, p), max(worst_r, abs(r - 1))
            if not ok:
                fails.append((k, p, r))
        else:
            n_layer += 1
            if not ok:
                ka = ("attn " + k[0],) + k[1:]
                (near_tie if ka not in [f[0] for f in fails] else fails).append((k, p, r))
    print(f"{case}: attention output worst PCC {worst_p:.6f}, worst |rms ratio - 1| {worst_r:.5f}")
    print(f"{case}: layer updates outside the bars with a matching attention output (MoE near-tie): {near_tie}")
    for k in t_ref:
        if k in t_got:
            print(f"{case} {k}: replicated {t_ref[k]:.4g} sharded {t_got[k]:.4g}")
    assert not fails, fails
    assert len(near_tie) <= max(2, n_layer // 10), near_tie
