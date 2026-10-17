# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Single-die TTNN baseline for the multichip comparison (stage 4): run the optimized decoder (one die, policy
GLM53_OPT, default SELECTED) on fixed inputs per layer kind and save the outputs. tests/test_multichip_decoder.py
::test_matches_single_die_baseline runs the same inputs through MultichipDecoder on the (1, 4) mesh and compares.

Inputs per kind: real-token streams of the HF reference (the cached test_batched_users requests): user A 100 prefill
rows into slot 5, user B 64 rows into slot 2, then 3 decode steps with both slots active. Saved: the layer updates
X' - X (the residual itself would dominate a PCC).

usage: python -m models.autoports.zai_org_glm_5_3_flash.tools.baseline_dump --out ~/glm53-flash-bringup/refs/baseline_opt.pt
"""

import argparse
import os
from pathlib import Path

import torch

from models.autoports.zai_org_glm_5_3_flash.tests.harness import (
    DecodeRunner,
    build_layer,
    close_mesh,
    open_mesh,
    prefill_user,
)
from models.autoports.zai_org_glm_5_3_flash.tests.test_functional_decoder import KINDS, reference
from models.autoports.zai_org_glm_5_3_flash.tt.ckpt import CheckpointReader

BATCH = 8
N_A, N_B, STEPS = 100, 64, 3  # = reference(layer, 'real', 503 / 506, n, 3) of test_batched_users


def run_case(mesh, rd, layer, decoder_cls):
    xa, _ = reference(layer, "real", 503, N_A, STEPS)
    xb, _ = reference(layer, "real", 506, N_B, STEPS)
    dec = build_layer(mesh, rd, layer, "real", batch=BATCH, decoder_cls=decoder_cls)
    out = {
        "prefill_a": prefill_user(dec, mesh, xa, N_A, 5) - xa[:N_A],
        "prefill_b": prefill_user(dec, mesh, xb, N_B, 2) - xb[:N_B],
    }
    runner = DecodeRunner(dec, mesh, BATCH)
    steps = []
    for i in range(STEPS):
        pos = [-1] * BATCH
        pos[5], pos[2] = N_A + i, N_B + i
        o = runner.eager_step({5: xa[N_A + i], 2: xb[N_B + i]}, pos)
        steps.append(torch.stack([o[5] - xa[N_A + i], o[2] - xb[N_B + i]]))
    out["decode"] = torch.stack(steps)
    return out


def decoder_cls_from_env(multichip: bool):
    from models.autoports.zai_org_glm_5_3_flash.tt.optimized_decoder import CANDIDATES, SELECTED, OptimizedDecoder

    base = OptimizedDecoder
    if multichip:
        from models.autoports.zai_org_glm_5_3_flash.tt.multichip_decoder import MultichipDecoder as base
    opt = CANDIDATES[os.environ["GLM53_OPT"]] if os.environ.get("GLM53_OPT") else SELECTED

    class _Cls(base):
        @classmethod
        def from_state_dict(cls, *a, **kw):
            kw.setdefault("opt", opt)
            return super().from_state_dict(*a, **kw)

    return _Cls, opt.name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    assert os.environ.get("GLM53_TP", "1") == "1", "the baseline runs on one die"
    rd = CheckpointReader()
    mesh = open_mesh()
    cls, name = decoder_cls_from_env(multichip=False)
    try:
        res = {"policy": name, "cases": {k: run_case(mesh, rd, layer, cls) for k, layer in KINDS.items()}}
    finally:
        close_mesh(mesh)
    Path(a.out).expanduser().parent.mkdir(parents=True, exist_ok=True)
    torch.save(res, Path(a.out).expanduser())
    print(f"saved {a.out} (policy {name})")


if __name__ == "__main__":
    main()
