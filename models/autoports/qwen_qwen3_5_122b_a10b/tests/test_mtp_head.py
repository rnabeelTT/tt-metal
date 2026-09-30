# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""MTP head on the mesh vs the torch reference (tests/reference_mtp.py): a reduced model (layers 0 and 3, for the shared
embedding / LM head / mesh plumbing) plus the MTP head, fed T single-row decode steps at positions 0..T-1 with random
token ids and random 'final-normed hidden' rows; the reference runs the same rows as one causal prefill. Checks the
per-position logits (PCC) and the argmax agreement (the draft token)."""
from __future__ import annotations

import os

import pytest
import torch

import ttnn
from models.autoports.qwen_qwen3_5_122b_a10b.tests import reference as ref
from models.autoports.qwen_qwen3_5_122b_a10b.tests.reference_mtp import MtpReference
from models.autoports.qwen_qwen3_5_122b_a10b.tt.model import create_model
from models.autoports.qwen_qwen3_5_122b_a10b.tt.mtp import Qwen35MtpHead
from models.common.utility_functions import comp_pcc

BLOCK = 64


@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
@pytest.mark.parametrize(
    "device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "l1_small_size": 24576}], indirect=True
)
@pytest.mark.parametrize("T", [8], ids=["t8"])
def test_mtp_head_matches_reference(mesh_device, T):
    os.environ.setdefault("MESH_DEVICE", "P300x2")
    args, model = create_model(mesh_device, max_batch_size=1, max_seq_len=4096, layer_indices=[0, 3])
    sd = args.load_state_dict()
    head = Qwen35MtpHead(model, sd)
    text = ref.text_config()[1]
    reference = MtpReference(text, sd, embed_weight=sd["tok_embeddings.weight"], lm_head_weight=sd["output.weight"])
    # paged KV for the main attention layer(s) and the head: 1 user, 4096 tokens of 64-token blocks
    n_blocks = 4096 // BLOCK
    kv_shape = (
        model.layers[1].attention.paged_k.shape
        if getattr(model.layers[1].attention, "paged_k", None) is not None
        else None
    )
    if kv_shape is None:
        kv_heads = text.num_key_value_heads
        kv_shape = (n_blocks, kv_heads if args.num_devices == 1 else 1, BLOCK, text.head_dim)
        model.allocate_kv_caches(kv_shape, ttnn.bfloat16, batch_size=1)
    head.allocate_kv_cache(kv_shape)
    page_table = torch.arange(n_blocks, dtype=torch.int32).reshape(1, n_blocks)
    H = text.hidden_size
    torch.manual_seed(7)
    tok = torch.randint(1000, 30000, (1, T))
    hidden = (torch.randn(1, T, H) * 0.7).to(torch.bfloat16)
    pos = torch.arange(T).reshape(1, T)
    ref_logits, _ = reference(tok, hidden, pos, ref.DynamicCache(), attention_mask=ref.causal_mask(1, T))
    pccs, agree = [], 0
    rep = ttnn.ReplicateTensorToMesh(mesh_device)
    for t in range(T):
        tokens_tt, cur_pos_tt, rot_idxs, pt_tt = model.prepare_decode_inputs_host(
            tok[:, t], pos[0, t].item(), page_table
        )
        tokens_tt, cur_pos_tt, rot_idxs, pt_tt = (
            ttnn.to_device(x, mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            for x in (tokens_tt, cur_pos_tt, rot_idxs, pt_tt)
        )
        hid = ttnn.from_torch(
            hidden[:, t].reshape(1, 1, 1, H),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=rep,
        )
        logits, h = head.decode_step(tokens_tt, hid, cur_pos_tt, rot_idxs, pt_tt)
        out = ttnn.to_torch(logits, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=-1)).float()
        out = out.reshape(-1, out.shape[-1])[:1, : text.vocab_size]
        r = ref_logits[0, t : t + 1]
        ok, pcc = comp_pcc(r, out, 0.99)
        pccs.append(pcc)
        agree += int(out.argmax(-1).item() == r.argmax(-1).item())
        print(f"MTP step {t}: PCC {pcc} argmax tt {out.argmax(-1).item()} ref {r.argmax(-1).item()}")
    print(f"MTP_HEAD T={T}: min PCC {min(pccs):.5f} argmax agreement {agree}/{T}")
    assert min(pccs) >= 0.99, pccs
    assert agree >= T - 1, agree
