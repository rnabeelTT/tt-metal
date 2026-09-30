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

    # every call here carries BW rows of one user (dummies included), so every call needs the per-row K/V write: with
    # the batched per-user update the dummy rows of the prefix steps race the real row inside its KV tile
    layer.attention.rows_kv_write = True
    # prefix: one real row per step
    for t in range(prefix):
        step(x_all[0, t : t + 1], [t])
    # the verify rows: R positions in ONE call
    y_rows = step(x_all[0, prefix : prefix + R], list(range(prefix, prefix + R)))
    layer.attention.rows_kv_write = False
    pccs = []
    for j in range(R):
        _, pcc = comp_pcc(y_ref[0, prefix + j : prefix + j + 1], y_rows[j : j + 1], 0.99)
        pccs.append(pcc)
        print(f"ROWS attn R={R} prefix={prefix} row {j} (pos {prefix + j}): PCC {pcc}")
    print(f"ROWS_ATTN R={R} prefix={prefix}: min PCC {min(pccs):.5f}")
    assert min(pccs) >= 0.99, pccs


@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
@pytest.mark.parametrize(
    "device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "l1_small_size": 24576}], indirect=True
)
def test_gdn_rows_cost(mesh_device):
    """Cost of the GDN verify-rows candidates on one GDN layer (real weights): (a) one decode step (B = 1), (b) a prefill
    chunk continuation of T = 32 rows (padded internally to the 128-row chunk), (c) T = 128; host wall with sync, 10 iters.
    Decides whether the verify pass can afford the GDN prefill form per layer (36 GDN layers per pass)."""
    import time as _time

    from models.autoports.qwen_qwen3_5_122b_a10b.tests.test_multichip_decoder import L0, _hidden, _mc

    os.environ.setdefault("MESH_DEVICE", "P300x2")
    hm = _mc(mesh_device, L0, batch=1, max_seq_len=8192, real=True, hf=False)
    H = hm.H
    # prefix so the recurrent state is non-trivial
    hm.tt_prefill(_hidden(1, 256, H, seed=1))
    results = {}
    for name, T in (("prefill_T32", 32), ("prefill_T128", 128)):
        start = 256
        hidden = _hidden(1, T, H, seed=2)
        hm.tt_prefill(hidden, start=start)  # compile
        ttnn.synchronize_device(mesh_device)
        t0 = _time.perf_counter()
        for it in range(10):
            hm.tt_prefill(hidden, start=start)  # chunk starts must be 128-aligned: re-run the same continuation
        ttnn.synchronize_device(mesh_device)
        results[name] = 1e3 * (_time.perf_counter() - t0) / 10
    x, kw = hm.decode_inputs(_hidden(1, 1, H, seed=3), torch.tensor([2000]))
    hm.tt.decode_forward(x, **kw)
    ttnn.synchronize_device(mesh_device)
    t0 = _time.perf_counter()
    for it in range(10):
        out = hm.tt.decode_forward(x, **kw)
        ttnn.deallocate(out)
    ttnn.synchronize_device(mesh_device)
    results["decode_B1"] = 1e3 * (_time.perf_counter() - t0) / 10
    print("GDN_ROWS_COST " + " | ".join(f"{k} {v:.3f} ms" for k, v in results.items()))


@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
@pytest.mark.parametrize(
    "device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "l1_small_size": 24576}], indirect=True
)
@pytest.mark.parametrize("R", [5, 8], ids=["r5", "r8"])
def test_gdn_rows_mode(mesh_device, R):
    """GDN layer 0 (real weights, B = 1 state): a 256-token prefill, 3 plain decode steps (the taps advance), then R rows
    in ONE rows-mode call (`layer_forward_rows`) vs the HF layer's causal prefill over the same 256 + 3 + R rows."""
    import time as _time

    from models.autoports.qwen_qwen3_5_122b_a10b.tests.test_multichip_decoder import L0, _hidden, _mc
    from models.autoports.qwen_qwen3_5_122b_a10b.tt.mtp_rows import layer_forward_rows

    os.environ.setdefault("MESH_DEVICE", "P300x2")
    BW = 8
    hm = _mc(mesh_device, L0, batch=1, max_seq_len=8192, real=True, hf=True, max_batch_size=1)
    H = hm.H
    P0, D = 256, 3
    T = P0 + D + R
    hidden = _hidden(1, T, H, seed=5)
    y_ref = hm.hf_prefill(hidden).float()
    hm.tt_prefill(hidden[:, :P0])
    for t in range(D):
        x, kw = hm.decode_inputs(hidden[:, P0 + t].reshape(1, 1, H), torch.tensor([P0 + t]))
        out = hm.tt.decode_forward(x, **kw)
        ttnn.deallocate(out)
    layer = hm.tt.layer
    rows = torch.cat([hidden[0, P0 + D : P0 + D + R], torch.zeros(BW - R, H, dtype=hidden.dtype)], dim=0)
    x_rows = ttnn.from_torch(
        rows.reshape(1, 1, BW, H),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=-1),
    )
    t0 = _time.perf_counter()
    y = layer_forward_rows(layer, x_rows, R)
    ttnn.synchronize_device(mesh_device)
    dt = 1e3 * (_time.perf_counter() - t0)
    y_dev = ttnn.to_torch(y, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=-1)).float().reshape(-1, H)[:R]
    pccs = []
    for j in range(R):
        _, pcc = comp_pcc(y_ref[0, P0 + D + j : P0 + D + j + 1], y_dev[j : j + 1], 0.99)
        pccs.append(pcc)
        print(f"ROWS gdn R={R} row {j} (pos {P0 + D + j}): PCC {pcc}")
    print(f"ROWS_GDN R={R}: min PCC {min(pccs):.5f} (rows call {dt:.2f} ms host wall incl. compile)")
    assert min(pccs) >= 0.99, pccs
