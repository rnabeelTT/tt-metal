# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""MTP verify-pass building blocks: the "rows" mode of one layer = R consecutive positions of ONE user in a single
batched decode call (rows in the batch dimension, the same page-table row for every row, positions P..P+R-1). For the
attention layer the KV of all R rows is written before the SDPA reads, so row j must see rows < j (causal within the
step); the reference is the HF layer's causal prefill over the same R rows."""
from __future__ import annotations

import os

import pytest
import torch

import ttnn
from models.autoports.qwen_qwen3_5_122b_a10b.tests import reference as ref
from models.autoports.qwen_qwen3_5_122b_a10b.tt.model import create_model
from models.common.utility_functions import comp_pcc

BLOCK = 64
BW = 8  # decode batch width (allowed widths are powers of two); rows beyond R are dummies at unattended positions


@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
@pytest.mark.parametrize(
    "device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "l1_small_size": 24576}], indirect=True
)
@pytest.mark.parametrize("R,prefix", [(5, 0), (5, 40), (8, 40)], ids=["r5", "r5_p40", "r8_p40"])
def test_attention_rows_mode(mesh_device, R, prefix):
    """Layer 3 (full attention): `prefix` positions filled row by row, then R rows at positions prefix..prefix+R-1 in ONE
    batched decode call vs the HF layer's causal prefill over prefix+R rows."""
    os.environ.setdefault("MESH_DEVICE", "P300x2")
    args, model = create_model(mesh_device, max_batch_size=BW, max_seq_len=4096, layer_indices=[0, 3])
    text = ref.text_config()[1]
    n_blocks = 4096 // BLOCK
    kv_shape = (n_blocks, 1 if args.num_devices > 1 else text.num_key_value_heads, BLOCK, text.head_dim)
    model.allocate_kv_caches(kv_shape, ttnn.bfloat16, batch_size=BW)
    layer = model.layers[1]
    hf = ref.torch_layer(text, 3, ref.real_layer_state(3))
    H = text.hidden_size
    T = prefix + R
    torch.manual_seed(23)
    x_all = (torch.randn(1, T, H) * 0.7).to(torch.bfloat16)
    pos_all = torch.arange(T).reshape(1, T)
    with torch.no_grad():
        y_ref = hf(
            x_all,
            position_embeddings=ref.rotary(text, x_all, pos_all),
            attention_mask=ref.causal_mask(1, T),
            past_key_values=ref.DynamicCache(),
        )
    y_ref = (y_ref[0] if isinstance(y_ref, tuple) else y_ref).float()
    page_row = torch.arange(n_blocks, dtype=torch.int32)

    def step(rows_x, positions):
        """rows_x [n, H] torch, positions list -> layer output [n, H] torch (one batched decode call over BW rows: the
        n real rows first, then dummy rows at positions above every real one, which no real row attends)."""
        n = len(positions)
        B = BW
        if n < B:
            pad_pos = [max(positions) + 1 + j for j in range(B - n)]
            positions = list(positions) + pad_pos
            rows_x = torch.cat([rows_x, torch.zeros(B - n, rows_x.shape[-1], dtype=rows_x.dtype)], dim=0)
        tokens_tt, cur_pos_tt, rot_idxs, pt_tt = model.prepare_decode_inputs_host(
            torch.zeros(B, dtype=torch.int32),
            torch.tensor(positions, dtype=torch.int32),
            page_row.reshape(1, -1).repeat(B, 1),
        )
        cur_pos_tt, rot_idxs, pt_tt = (
            ttnn.to_device(t, mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG) for t in (cur_pos_tt, rot_idxs, pt_tt)
        )
        x_t = ttnn.from_torch(
            rows_x.reshape(1, 1, B, H),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=-1),
        )
        cos, sin = model._rope_from_idxs(rot_idxs)
        y = layer.forward(x_t, cos, sin, position_tensor=cur_pos_tt, page_table=pt_tt, mode="decode")
        return ttnn.to_torch(y, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=-1)).float().reshape(-1, H)[:n]

    # prefix: one row per step (plain decode)
    for t in range(prefix):
        step(x_all[0, t : t + 1], [t])
    # the verify rows: R positions in ONE call
    y_rows = step(x_all[0, prefix : prefix + R], list(range(prefix, prefix + R)))
    pccs = []
    for j in range(R):
        _, pcc = comp_pcc(y_ref[0, prefix + j : prefix + j + 1], y_rows[j : j + 1], 0.99)
        pccs.append(pcc)
        print(f"ROWS attn R={R} prefix={prefix} row {j} (pos {prefix + j}): PCC {pcc}")
    print(f"ROWS_ATTN R={R} prefix={prefix}: min PCC {min(pccs):.5f}")
    assert min(pccs) >= 0.99, pccs
