# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Record (CPU, no device) the quantization-aware HF references (tests/hf_ref/hf_quant_ref.py) that the optimized suite
(tests/test_optimized_decoder.py) requests for the MoE layers, for tools/compute_refs.py to compute in parallel. The
calls mirror the reference() / make_ref() calls of every test of tests/test_functional_decoder.py; the base
(checkpoint) references must already be cached.

usage: GLM53_REF_DRYRUN=<dir> python -m models.autoports.zai_org_glm_5_3_flash.tools.quant_ref_requests [--policy opt]
       python -m models.autoports.zai_org_glm_5_3_flash.tools.compute_refs <dir> --workers 3"""

import argparse
import os

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="opt")
    a = ap.parse_args()
    assert os.environ.get("GLM53_REF_DRYRUN"), "set GLM53_REF_DRYRUN=<dir>"
    import models.autoports.zai_org_glm_5_3_flash.tests.test_functional_decoder as S
    from models.autoports.zai_org_glm_5_3_flash.tests.harness import STATS
    from models.autoports.zai_org_glm_5_3_flash.tests.hf_ref.client import DryRunResult, quant_reference
    from models.autoports.zai_org_glm_5_3_flash.tt.optimized_decoder import CANDIDATES

    S.QUANT_REF.update(CANDIDATES[a.policy].quant_spec())
    print("quant spec", S.QUANT_REF)
    calls = []
    for kind in ("dsa_moe", "kda_moe"):
        L = S.KINDS[kind]
        calls.append((L, "real", 1, 100, 3, None))  # test_real_weights
        calls.append((L, {"synthetic": 0}, 2, 200, 3, None))  # test_synthetic_weights
        calls += [(L, "real", 1000 + n, n, 3, None) for k, n in S.LENGTHS if k == kind]  # test_lengths
        users = {1: 37, 3: 100, 4: 2100 if kind == "dsa_moe" else 300, 6: 64}
        calls += [(L, "real", 500 + s, n, 3, None) for s, n in users.items()]  # test_batched_users
        calls.append((L, "real", 7, 150, 4, None))  # test_determinism
        calls.append((L, "real", 1, 100, 1, None))  # test_no_host_fallback
        lengths = [37, 300, 2100, 2300] if kind == "dsa_moe" else [37, 129, 300, 700]
        rows = [n + i for n in lengths for i in range(3)]
        calls += [(L, "real", 1300 + seed, max(lengths), 3, rows) for seed in range(8)]  # test_batch32_all_active
        calls += [(L, "real", 41, 102, 2, None), (L, "real", 42, 64, 6, None)]  # test_inactive_slot_state
        if kind == "dsa_moe":
            calls += [(L, "real", 900 + s, n, 3, None) for s, n in {0: 4000, 5: 2100, 7: 129}.items()]  # paged
    n_ok = n_dry = 0
    for L, w, seed, n, steps, rows in calls:
        r = S.reference(L, w, seed, n, steps, decode_rows=rows)
        if isinstance(r, S._DryRef):
            n_dry += 1
        else:
            n_ok += 1
    for kind in ("dsa_moe", "kda_moe"):  # test_decode_only and the synthetic FFN taps
        L = S.KINDS[kind]
        req = {"layer": L, "weights": "real", "tokens": S.tokens(77, 4), "segments": [1] * 4}
        r = S.make_ref(req, S.hf_layer_reference(req), range(4))
        n_dry += isinstance(r, S._DryRef)
        req = {
            "layer": L,
            "weights": {"synthetic": 0, "stats": str(STATS)},
            "x": S.synthetic_stream(L, 2, 203),
            "segments": [200, 1, 1, 1],
            "taps": True,
        }
        base = S.hf_layer_reference(req)
        if not isinstance(base, DryRunResult):
            n_dry += isinstance(quant_reference(req, base, S.QUANT_REF, taps=True), DryRunResult)
    print(f"references: {n_ok} complete, {n_dry} with recorded requests in {os.environ['GLM53_REF_DRYRUN']}")


if __name__ == "__main__":
    torch.set_num_threads(4)
    main()
