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
    ref_logits, ref_h = reference(tok, hidden, pos, ref.DynamicCache(), attention_mask=ref.causal_mask(1, T))
    lm_head_bf16 = reference.lm_head_weight.float()
    pccs, pccs_h, pccs_l16, agree, top5 = [], [], [], 0, 0
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
        # isolation 1: the mix (norms + fc) alone vs the reference mix
        x_mix = head.mix(tokens_tt, hid)
        mix_dev = (
            ttnn.to_torch(x_mix, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=-1)).float().reshape(-1, H)[:1]
        )
        mix_ref = reference.mix(tok[:, t : t + 1], hidden[:, t : t + 1]).float().reshape(-1, H)
        _, pcc_mix = comp_pcc(mix_ref, mix_dev, 0.99)
        ttnn.deallocate(x_mix)
        print(f"MTP step {t}: mix PCC {pcc_mix}")
        logits, h = head.decode_step(tokens_tt, hid, cur_pos_tt, rot_idxs, pt_tt)
        out = ttnn.to_torch(logits, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=-1)).float()
        out = out.reshape(-1, out.shape[-1])[:1, : text.vocab_size]
        h_dev = ttnn.to_torch(ttnn.get_device_tensors(h)[0]).float().reshape(-1, H)[:1]
        r = ref_logits[0, t : t + 1]
        r_h = ref_h[0, t : t + 1].float()
        ok, pcc = comp_pcc(r, out, 0.99)
        _, pcc_h = comp_pcc(r_h, h_dev, 0.99)
        logits_from_dev_h = h_dev @ lm_head_bf16.T  # the device hidden through a bf16 LM head: isolates the bfp4 head
        _, pcc_l16 = comp_pcc(r, logits_from_dev_h, 0.99)
        pccs.append(pcc)
        pccs_h.append(pcc_h)
        pccs_l16.append(pcc_l16)
        agree += int(out.argmax(-1).item() == r.argmax(-1).item())
        top5 += int(r.argmax(-1).item() in out.topk(5, dim=-1).indices[0].tolist())
        print(
            f"MTP step {t}: hidden PCC {pcc_h} | logits PCC {pcc} (device h x bf16 head: {pcc_l16}) | "
            f"argmax tt {out.argmax(-1).item()} ref {r.argmax(-1).item()}"
        )
    print(
        f"MTP_HEAD T={T}: min hidden PCC {min(pccs_h):.5f} min logits PCC {min(pccs):.5f} "
        f"(bf16 head on device hidden {min(pccs_l16):.5f}) argmax agreement {agree}/{T} ref-top1-in-device-top5 {top5}/{T}"
    )
    # isolation 2: the layer + final norm alone, fed the REFERENCE mix output (fresh KV cache), vs the reference hidden
    head.allocate_kv_cache(kv_shape)
    from models.tt_transformers.tt.common import Mode

    pccs_layer = []
    for t in range(T):
        _, cur_pos_tt, rot_idxs, pt_tt = model.prepare_decode_inputs_host(tok[:, t], pos[0, t].item(), page_table)
        cur_pos_tt, rot_idxs, pt_tt = (
            ttnn.to_device(x, mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG) for x in (cur_pos_tt, rot_idxs, pt_tt)
        )
        mix_ref = reference.mix(tok[:, t : t + 1], hidden[:, t : t + 1]).reshape(1, 1, 1, H)
        x_ref = ttnn.from_torch(
            mix_ref,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=-1),
        )
        cos, sin = model._rope_from_idxs(rot_idxs)
        y = head.layer.forward(x_ref, cos, sin, position_tensor=cur_pos_tt, page_table=pt_tt, mode="decode")
        nc = dict(args.get_norm_config("lm_head", Mode.DECODE))
        nc["output_mem_config"] = ttnn.DRAM_MEMORY_CONFIG
        h2 = head.norm(y, mode=Mode.DECODE, norm_config=nc)
        h2_dev = ttnn.to_torch(ttnn.get_device_tensors(h2)[0]).float().reshape(-1, H)[:1]
        _, pcc_l = comp_pcc(ref_h[0, t : t + 1].float(), h2_dev, 0.99)
        pccs_layer.append(pcc_l)
        print(f"MTP step {t}: layer-only (reference mix in) hidden PCC {pcc_l}")
    print(f"MTP_HEAD_LAYER_ONLY T={T}: min hidden PCC {min(pccs_layer):.5f}")
    assert min(pccs_h) >= 0.99, pccs_h
    assert top5 >= T - 1, top5


@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
@pytest.mark.parametrize(
    "device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "l1_small_size": 24576}], indirect=True
)
@pytest.mark.parametrize("which", ["main3", "mtp"], ids=["main3", "mtp"])
def test_layer_decode_rows_harness(mesh_device, which):
    """Harness control: the SAME row-by-row decode harness (positions 0..7, fresh KV, random fractured residual in) on the
    main full-attention layer 3 (reference: the HF layer with the real weights) and on the MTP layer; if the main layer
    also lands at ~0.95 the harness / reference is off, if it lands at ~0.995 the MTP layer is the odd one."""
    os.environ.setdefault("MESH_DEVICE", "P300x2")
    T = 8
    args, model = create_model(mesh_device, max_batch_size=1, max_seq_len=4096, layer_indices=[0, 3])
    sd = args.load_state_dict()
    text = ref.text_config()[1]
    n_blocks = 4096 // BLOCK
    kv_heads = text.num_key_value_heads
    kv_shape = (n_blocks, kv_heads if args.num_devices == 1 else 1, BLOCK, text.head_dim)
    model.allocate_kv_caches(kv_shape, ttnn.bfloat16, batch_size=1)
    if which == "mtp":
        head = Qwen35MtpHead(model, sd)
        head.allocate_kv_cache(kv_shape)
        layer = head.layer
        st = sd.mtp_state(fuse_experts=True)
        prefix = "mtp.layers.0."
        hf = ref.torch_layer(
            head.args
            and __import__(
                "models.autoports.qwen_qwen3_5_122b_a10b.tests.reference_mtp", fromlist=["mtp_text_config"]
            ).mtp_text_config(text),
            len(text.layer_types),
            {k[len(prefix) :]: v for k, v in st.items() if k.startswith(prefix)},
        )
    else:
        layer = model.layers[1]  # checkpoint layer 3 (full attention)
        hf = ref.torch_layer(text, 3, ref.real_layer_state(3))
    page_table = torch.arange(n_blocks, dtype=torch.int32).reshape(1, n_blocks)
    H = text.hidden_size
    torch.manual_seed(11)
    x_all = (torch.randn(1, T, H) * 0.7).to(torch.bfloat16)
    pos = torch.arange(T).reshape(1, T)
    pe = ref.rotary(text, x_all, pos)
    with torch.no_grad():
        y_ref = hf(
            x_all, position_embeddings=pe, attention_mask=ref.causal_mask(1, T), past_key_values=ref.DynamicCache()
        )
    y_ref = (y_ref[0] if isinstance(y_ref, tuple) else y_ref).float()
    pccs = []
    for t in range(T):
        _, cur_pos_tt, rot_idxs, pt_tt = model.prepare_decode_inputs_host(
            torch.zeros(1, dtype=torch.int32), t, page_table
        )
        cur_pos_tt, rot_idxs, pt_tt = (
            ttnn.to_device(x, mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG) for x in (cur_pos_tt, rot_idxs, pt_tt)
        )
        x_t = ttnn.from_torch(
            x_all[:, t].reshape(1, 1, 1, H),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=-1),
        )
        cos, sin = model._rope_from_idxs(rot_idxs)
        y = layer.forward(x_t, cos, sin, position_tensor=cur_pos_tt, page_table=pt_tt, mode="decode")
        y_dev = ttnn.to_torch(y, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=-1)).float().reshape(-1, H)[:1]
        _, pcc = comp_pcc(y_ref[0, t : t + 1], y_dev, 0.99)
        pccs.append(pcc)
        print(f"HARNESS {which} step {t}: layer output PCC {pcc}")
    print(f"HARNESS_{which.upper()} T={T}: min PCC {min(pccs):.5f}")
