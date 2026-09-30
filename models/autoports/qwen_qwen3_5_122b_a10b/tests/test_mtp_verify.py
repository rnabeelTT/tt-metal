# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""MTP verify pass vs sequential decode on the reduced 2-layer generator (layers 0 = GDN, 3 = attention): after a prompt
prefill, the logits of R = 5 tokens fed one per decode step must equal (PCC / argmax) the logits of the same 5 tokens run
as ONE verify pass from the same state."""
from __future__ import annotations

import os

import pytest
import torch

import ttnn
from models.autoports.qwen_qwen3_5_122b_a10b.tt.generator import build_generator
from models.autoports.qwen_qwen3_5_122b_a10b.tt.mtp_verify import copy_slot_into_scratch, verify_pass
from models.common.utility_functions import comp_pcc

MODEL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "l1_small_size": 24576, "trace_region_size": 128 * 1024 * 1024}],
    indirect=True,
)
def test_verify_pass_matches_sequential_decode(mesh_device):
    os.environ.setdefault("MESH_DEVICE", "P300x2")
    gen = build_generator(MODEL_DIR, mesh_device, max_seq_len=4096, layer_indices=[0, 3], warm_decode_trace=False)
    model = gen.model
    ids = [1000 + 7 * i for i in range(24)]
    P = len(ids)
    page_table = gen.page_table[:1]
    R = 5
    toks = [2000, 2011, 2022, 2033, 2044]
    # (a) sequential eager decode, host logits
    gen.reset()
    gen.prefill_forward(
        torch.tensor(ids).reshape(1, -1), page_table=page_table, prompt_lens=[P], return_all_logits=True
    )
    seq_logits = []
    for i, t in enumerate(toks):
        out = gen.decode_forward(
            torch.tensor([[t]]), torch.tensor([P + i]), enable_trace=False, on_device_sampling=False, reload_inputs=True
        )
        seq_logits.append(out[0].float().reshape(-1)[: gen.args.vocab_size].clone())
    # (b) fresh state, the same 5 tokens as one verify pass
    gen.reset()
    gen.prefill_forward(
        torch.tensor(ids).reshape(1, -1), page_table=page_table, prompt_lens=[P], return_all_logits=True
    )
    prev = model._bind_gdn_prefill_scratch()
    try:
        copy_slot_into_scratch(prev, 0)
        h, logits, _ = verify_pass(model, toks, [P + i for i in range(R)], page_table)
        lg = ttnn.to_torch(logits, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=-1)).float()
        lg = lg.reshape(-1, lg.shape[-1])[:R, : gen.args.vocab_size]
    finally:
        model._unbind_gdn_prefill_scratch(prev)
    pccs, agree = [], 0
    for i in range(R):
        _, pcc = comp_pcc(seq_logits[i], lg[i], 0.99)
        pccs.append(pcc)
        agree += int(seq_logits[i].argmax().item() == lg[i].argmax().item())
        print(f"VERIFY row {i}: PCC {pcc} argmax seq {seq_logits[i].argmax().item()} verify {lg[i].argmax().item()}")
    print(f"VERIFY_PASS R={R}: min PCC {min(pccs):.5f} argmax agreement {agree}/{R}")
    assert min(pccs) >= 0.98, pccs
    assert agree == R, agree
