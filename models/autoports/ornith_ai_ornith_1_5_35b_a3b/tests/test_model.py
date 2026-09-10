# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Full 40-layer model against a cached HF reference.

The stage-05 gate: the whole stack -- embeddings, 30 Gated DeltaNet layers, 10 gated
full-attention layers, 40 MoE blocks, final norm and LM head -- producing next-token
logits for real prompts.

Graded on TOP-1 AGREEMENT first and PCC second, deliberately. A model with a subtly
broken router or a mis-split expert still yields smooth, well-correlated logits; what it
does not do is keep picking the same token as the reference. PCC alone has already let
one real defect through in this bring-up.

The reference is generated once by scripts/lib/gen_reference.py and cached, because
running a 35B reference costs ~70 GB of host RAM and minutes per prompt.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch

import ttnn
from models.autoports.ornith_ai_ornith_1_5_35b_a3b.tt.model import OrnithModel
from models.autoports.ornith_ai_ornith_1_5_35b_a3b.tt.model_config import OrnithModelArgs
from models.autoports.ornith_ai_ornith_1_5_35b_a3b.tt.weight_mapping import load_ornith_state_dict
from models.common.utility_functions import comp_pcc

# Reference logits are ~1 MB each and are NOT committed: regenerate them with
#   python tools/gen_reference.py --out <model_dir>/reference
# The default lives beside the model so a checkout is self-contained once generated;
# ORNITH_REFERENCE points somewhere else (e.g. a shared cache) when you have one.
REFERENCE = Path(os.environ.get("ORNITH_REFERENCE", Path(__file__).resolve().parent.parent / "reference"))

# Logit PCC over a 248320-wide vocabulary, and how many of the reference's top-8 the
# device must also rank in its top-8. Top-1 must always match.
PCC_BY_DTYPE = {"bfloat16": 0.99, "bfloat8_b": 0.98, "bfloat4_bfloat8": 0.97, "bfloat4_b": 0.95}
MIN_TOP8_OVERLAP = 6


def _weights_dir() -> Path:
    d = os.environ.get("ORNITH_WEIGHTS") or os.environ.get("HF_MODEL")
    if not d or not Path(d, "config.json").is_file():
        pytest.skip("ORNITH_WEIGHTS is not pointing at a local checkpoint")
    return Path(d)


@pytest.fixture(scope="module")
def reference():
    f = REFERENCE / "reference.json"
    if not f.is_file():
        pytest.skip(f"no cached reference at {f}; run tools/gen_reference.py --out {REFERENCE}")
    return json.loads(f.read_text())


def test_model_next_token(mesh_device, reference):
    """One build, every prompt.

    Deliberately not parametrized per prompt: `mesh_device` is function-scoped, so a
    parametrized test reopens the device and re-uploads all 40 layers of experts for
    each prompt -- minutes of weight upload to check three next tokens.
    """
    weights = _weights_dir()
    # Prompts here are tens of tokens; a large max_seq_len only inflates the KV cache
    # allocation, which is one of the L1 residents the GDN op has to fit around.
    args = OrnithModelArgs(mesh_device, max_batch_size=1, max_seq_len=256)
    cfg = json.loads((weights / "config.json").read_text())["text_config"]
    assert reference["layers"] == cfg["num_hidden_layers"], (
        f"reference was generated for {reference['layers']} layers but the checkpoint has "
        f"{cfg['num_hidden_layers']}; regenerate it"
    )
    state = load_ornith_state_dict(weights, cfg)
    # A real cache path is required, not optional: the embedding builds its cache name as
    # `weight_cache_path / base_name` unconditionally, so None fails with a bare TypeError
    # about dividing NoneType by str. It also makes the second run of this test much
    # faster, since 40 layers of expert weights are then read back from cache.
    cache = Path(args.weight_cache_path(args.routed_expert_dtype))
    cache.mkdir(parents=True, exist_ok=True)
    model = OrnithModel(mesh_device, args, state, tensor_cache_path=cache)

    failures = []
    for prompt_idx in range(len(reference["prompts"])):
        failures.extend(_check_prompt(mesh_device, args, model, reference, prompt_idx))
    assert not failures, "\n".join(failures)


def _check_prompt(mesh_device, args, model, reference, prompt_idx):
    rec = reference["prompts"][prompt_idx]
    ids = torch.tensor(rec["input_ids"]).unsqueeze(0)
    seq = ids.shape[-1]

    # prefill() returns the LAST position's logits, which is exactly what a next-token
    # check needs. The TP variant is a separate entry point on multi-device meshes.
    out = model.prefill_tp(ids) if args.num_devices > 1 else model.prefill(ids)
    logits = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
    got = logits.reshape(-1, logits.shape[-1])[0].float()[: len(torch.load(REFERENCE / f"logits_{prompt_idx}.pt"))]
    exp = torch.load(REFERENCE / f"logits_{prompt_idx}.pt").float()

    problems = []
    if not torch.isfinite(got).all():
        return [f"[{prompt_idx}] device logits contain inf/nan"]

    got_top = torch.topk(got, 8).indices.tolist()
    exp_top = rec["top_tokens"]
    overlap = len(set(got_top) & set(exp_top))
    threshold = PCC_BY_DTYPE[args.routed_expert_dtype]
    _, pcc = comp_pcc(exp, got, threshold)
    print(
        f"  [{prompt_idx}] top1 {'OK ' if got_top[0] == exp_top[0] else 'MISS'} "
        f"top8 {overlap}/8  pcc {pcc:.5f}  {rec['prompt'][:40]!r}"
    )

    if got_top[0] != exp_top[0]:
        problems.append(
            f"[{prompt_idx}] top-1 differs: device {got_top[0]} vs reference "
            f"{exp_top[0]} (seq={seq}, pcc={pcc:.5f})"
        )
    if overlap < MIN_TOP8_OVERLAP:
        problems.append(f"[{prompt_idx}] only {overlap}/8 of the reference top-8 survived " f"(pcc={pcc:.5f})")
    if pcc < threshold:
        problems.append(f"[{prompt_idx}] logit PCC {pcc} < {threshold}")
    return problems
