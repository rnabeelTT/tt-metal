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
# bfp4 routed experts: the MoE layers are held to the bfp4 quantization floor (as tests/test_optimized_decoder.py)
_opt = CANDIDATES[os.environ["GLM53_OPT"]] if os.environ.get("GLM53_OPT") else SELECTED
if _opt.moe.routed_gate_up_dtype == ttnn.bfloat4_b:
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


def test_matches_single_die_baseline(mesh, rd):
    """Same inputs and policy as tools/baseline_dump.py on one die: prefill rows and decode steps of every kind match
    the single-die TTNN decoder (isolates sharding / collective errors from HF-vs-TT numerics)."""
    import pytest
    import torch

    from models.autoports.zai_org_glm_5_3_flash.tools.baseline_dump import decoder_cls_from_env, run_case

    if not os.path.exists(BASELINE):
        pytest.skip(f"no single-die baseline at {BASELINE} (tools/baseline_dump.py)")
    base = torch.load(BASELINE)
    cls, name = decoder_cls_from_env(multichip=True)
    assert base["policy"] == name, (base["policy"], name)
    for kind, layer in _suite.KINDS.items():
        got = run_case(mesh, rd, layer, cls)
        for key in ("prefill_a", "prefill_b", "decode"):
            ref = base["cases"][kind][key]
            p = _suite.pcc(got[key], ref)
            r = float(got[key].norm() / ref.norm())
            print(f"{kind} {key}: PCC vs single die {p:.6f} rms ratio {r:.5f}")
            assert p >= BASELINE_BAR, (kind, key, p)


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
