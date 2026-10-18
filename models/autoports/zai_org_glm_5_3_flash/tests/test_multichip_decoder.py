# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage 4: the functional-decoder test suite (tests/test_functional_decoder.py: real / synthetic weights, lengths
around every boundary, batched users, determinism, host-fallback audit, decode-only, paging, 32 active users) run on
tt/multichip_decoder.py over a (1, 4) mesh (GLM53_TP, default 4). The harness shards the residual stream in the
decoder's stream-interleaved layout and gathers it back, so the same HF references and bars apply.

GLM53_OPT=<candidate> selects the precision policy (tt/optimized_decoder.CANDIDATES; default SELECTED)."""

import importlib.util
import os
from pathlib import Path

os.environ.setdefault("GLM53_TP", "4")

import ttnn  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tt.multichip_decoder import MultichipDecoder  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tt.optimized_decoder import CANDIDATES, SELECTED  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "glm53_multichip_decoder_suite", Path(__file__).with_name("test_functional_decoder.py")
)
_suite = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_suite)


def _decoder_cls():
    name = os.environ.get("GLM53_OPT")
    if not name:
        return MultichipDecoder
    opt = CANDIDATES[name]

    class CandidateDecoder(MultichipDecoder):
        @classmethod
        def from_state_dict(cls, *args, **kw):
            kw.setdefault("opt", opt)
            return super().from_state_dict(*args, **kw)

    return CandidateDecoder


_suite.DECODER_CLS = _decoder_cls()
# Reference: the quantization-aware HF layer, as tests/test_optimized_decoder.py (stage-3 review #1 P1-2, stage-4 review
# #1 P2-5): the MoE layers' reference rounds the weights the policy stores below bf16 (OptConfig.quant_spec: routed
# experts bfp4, shared expert bfp8) exactly as the port's host packing does (tests/hf_ref/hf_quant_ref.py, bfp.py), and
# the functional bars apply unchanged (no BAR_OVERRIDES). GLM53_REF_QUANT=0: the checkpoint reference with the earlier
# bfp4 bar shift (comparison only).
_opt = CANDIDATES[os.environ["GLM53_OPT"]] if os.environ.get("GLM53_OPT") else SELECTED
if os.environ.get("GLM53_REF_QUANT", "1") != "0":
    _suite.QUANT_REF.update(_opt.quant_spec())
    print(f"multichip suite: quantization-aware reference {_suite.QUANT_REF}")
elif _opt.moe.routed_gate_up_dtype == ttnn.bfloat4_b:
    _suite.BAR_OVERRIDES.update(
        {layer: {"delta": 0.98, "seq": 0.03, "token": 0.06} for k, layer in _suite.KINDS.items() if k.endswith("moe")}
    )

pytestmark = _suite.pytestmark
mesh = _suite.mesh
rd = _suite.rd
decode_scale_mean = _suite.decode_scale_mean
suite_swap_cap = _suite.suite_swap_cap
globals().update({k: v for k, v in vars(_suite).items() if k.startswith("test_")})


def test_suite_runs_multichip_decoder(mesh):
    assert issubclass(_suite.DECODER_CLS, MultichipDecoder)
    assert tuple(mesh.shape) == (1, int(os.environ["GLM53_TP"]))


BASELINE = os.environ.get("GLM53_BASELINE", os.path.expanduser("~/glm53-flash-bringup/refs/baseline_opt.pt"))
BASELINE_BAR = 0.999  # multichip vs the single-die TTNN decoder, same weights / inputs / policy, on the layer updates
# |update rms ratio - 1| vs the single die (PCC is scale-blind: a partial added twice or a low-precision reduce-scatter
# payload passes a PCC bar). Matched-baseline spread on the final kernels: 0.00007 .. 0.0013 (stage-4 review #1 P2-5).
BASELINE_RMS_TOL = 0.005


def test_matches_single_die_baseline(mesh, rd):
    """Same inputs and policy as tools/baseline_dump.py on one die: prefill rows and decode steps of every kind match
    the single-die TTNN decoder (isolates sharding / collective errors from HF-vs-TT numerics)."""
    import pytest
    import torch

    from models.autoports.zai_org_glm_5_3_flash.tools.baseline_dump import decoder_cls_from_env, run_case

    if not os.path.exists(BASELINE):  # a missing baseline is a failed gate, not a skip (review #1 hard-check gap)
        if os.environ.get("GLM53_BASELINE_OPTIONAL") == "1":
            pytest.skip(f"no single-die baseline at {BASELINE} (tools/baseline_dump.py)")
        pytest.fail(f"no single-die baseline at {BASELINE}: run tools/baseline_dump.py at GLM53_TP=1 first")
    base = torch.load(BASELINE)
    cls, name = decoder_cls_from_env(multichip=True)
    assert base["policy"] == name, (base["policy"], name)
    from models.autoports.zai_org_glm_5_3_flash.tools.baseline_dump import N_A, N_B, STEPS

    near_ties = []
    for kind, layer in _suite.KINDS.items():
        got = run_case(mesh, rd, layer, cls)
        margin = {"a": _hf_margins(layer, 503, N_A), "b": _hf_margins(layer, 506, N_B)}
        for key in ("prefill_a", "prefill_b", "decode"):
            ref = base["cases"][kind][key]
            g = got[key]
            p = _suite.pcc(g, ref)
            r = float(g.norm() / ref.norm())
            print(f"{kind} {key}: PCC vs single die {p:.6f} rms ratio {r:.5f}")
            if p >= BASELINE_BAR and abs(r - 1) <= BASELINE_RMS_TOL:
                continue
            # router near-tie allowance (per token): a row whose HF router margin is below NEAR_TIE may take a different
            # 8th expert on one side (a ~1e-4 input difference flips it; stage-5 rework: one kda_moe decode token at
            # rel L2 0.085, every other row <= 0.0025). Such rows are set aside, at most max(1, 5 %) of the rows; the
            # remaining rows must meet both bars.
            if key == "decode":  # [STEPS, 2 users, H4]: user A rows N_A + i, user B rows N_B + i
                rows_g, rows_r = g.reshape(-1, g.shape[-1]), ref.reshape(-1, ref.shape[-1])
                marg = [m for i in range(STEPS) for m in (_row(margin["a"], N_A + i), _row(margin["b"], N_B + i))]
            else:
                rows_g, rows_r = g, ref
                mg = margin["a" if key == "prefill_a" else "b"]
                marg = [_row(mg, i) for i in range(g.shape[0])]
            rp = [_suite.pcc(rows_g[i], rows_r[i]) for i in range(rows_g.shape[0])]
            out = [i for i, v in enumerate(rp) if v < BASELINE_BAR]
            cap = max(1, int(0.05 * len(rp)))
            ok_rows = [i for i in range(len(rp)) if i not in out]
            allowed = (
                len(out) <= cap
                and all(marg[i] is not None and marg[i] < _suite.NEAR_TIE for i in out)
                and len(ok_rows) > 0
            )
            p2 = _suite.pcc(rows_g[ok_rows], rows_r[ok_rows]) if ok_rows else 0.0
            r2 = float(rows_g[ok_rows].norm() / rows_r[ok_rows].norm()) if ok_rows else 0.0
            print(
                f"{kind} {key}: rows below {BASELINE_BAR}: {[(i, round(rp[i], 5), marg[i]) for i in out]} (cap {cap}); "
                f"other rows PCC {p2:.6f} rms ratio {r2:.5f}"
            )
            assert allowed, (kind, key, p, r, "rows", [(i, rp[i], marg[i]) for i in out])
            assert p2 >= BASELINE_BAR, (kind, key, "other rows", p2)
            assert abs(r2 - 1) <= BASELINE_RMS_TOL, (kind, key, "other rows rms ratio", r2)
            near_ties += [(kind, key, i, rp[i], marg[i]) for i in out]
    print(f"single-die compare: router near-tie rows set aside: {near_ties}")


def _hf_margins(layer, seed, n):
    """HF router margin per row of the cached real-token reference behind tools/baseline_dump.py's inputs (None: dense
    layer, or not cached; never computed here)."""
    import torch

    from models.autoports.zai_org_glm_5_3_flash.tests.hf_ref.client import cache_path
    from models.autoports.zai_org_glm_5_3_flash.tools.baseline_dump import STEPS

    req = {"layer": layer, "weights": "real", "tokens": _suite.tokens(seed, n + STEPS), "segments": [n] + [1] * STEPS}
    p = cache_path(req)
    if not p.exists():
        return None
    m = torch.load(p).get("route_margin")
    return None if m is None else m.reshape(-1).float()


def _row(m, i):
    return None if m is None else float(m[i])


def test_single_slot_kda_decode(mesh, rd):
    """max_batch = 1 (single-user serving): TP4 leaves 16 KDA heads per die and kda.recurrent_decode needs batch x
    heads % 32 == 0, so tt/multichip_decoder.MultichipKDA pads the decode batch with one inactive slot (stage 5; the
    stage-4 decoder failed here). Slot 0 of a 1-slot decoder must match slot 0 of a 2-slot decoder (slot 1 inactive)
    on the same rows: 3 eager steps and one traced replay."""
    import torch

    from models.autoports.zai_org_glm_5_3_flash.tests.harness import H4, DecodeRunner, build_layer

    outs = {}
    for B in (1, 2):
        dec = build_layer(
            mesh, rd, 0, batch=B, prefill_lengths=(128,), max_context=4096, decoder_cls=_suite.DECODER_CLS
        )
        run = DecodeRunner(dec, mesh, B)
        rows = {0: torch.randn(H4, generator=torch.Generator().manual_seed(3)) * 0.02}
        pos = lambda p: [p] + [-1] * (B - 1)
        ys = [run.eager_step(rows, pos(128 + s))[0] for s in range(3)]
        run.capture(rows, pos(131))
        ys.append(run.traced_step(rows, pos(131))[0])
        run.release()
        outs[B] = torch.stack(ys)
        del run, dec
    assert torch.isfinite(outs[1]).all()
    p = _suite.pcc(outs[1], outs[2])
    print(f"1-slot vs 2-slot KDA decode: PCC {p:.7f}, max abs {float((outs[1] - outs[2]).abs().max()):.3e}")
    assert p >= 0.99999, p
