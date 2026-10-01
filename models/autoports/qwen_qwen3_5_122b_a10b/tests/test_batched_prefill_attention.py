# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Batched short-prompt prefill, step 1: ``TPAttention.forward_prefill_paged_batched`` (one QKV / rope / output
projection over B users' T-row blocks, per-user K/V fill + causal SDPA) must reproduce B single-user
``forward_prefill_paged`` calls: identical attention outputs and identical paged K/V contents."""
import pytest
import torch

import ttnn
from models.autoports.qwen_qwen3_5_122b_a10b.tests.test_multichip_decoder import (
    BLOCK_SIZE,
    L3,
    _compose,
    _hidden,
    _MeshHarness,
    _replicate,
    _shard,
)
from models.tt_transformers.tt.common import Mode


@pytest.mark.parametrize("T", [128, 512], ids=["t128", "t512"])
def test_attention_prefill_batched_matches_single(mesh_device, T):
    B = 4
    h = _MeshHarness(mesh_device, L3, batch=B, max_seq_len=4096, hf=False)
    layer, attn = h.tt.layer, h.tt.layer.attention
    H, rd = h.H, h.tt.args.rope_head_dim
    bs = BLOCK_SIZE
    n_blocks = -(-T // bs)
    pt = h.page_table  # [B, blocks_per_seq] host
    hidden = _hidden(B, T, H, seed=7).reshape(B, T, H)
    norm_cfg = {
        **h.tt.args.get_norm_config("attn", Mode.PREFILL),
        "distributed_output_mem_config": ttnn.L1_MEMORY_CONFIG,
    }

    def normed(x_host_rows):  # [1, rows, H] -> the layer's prefill attention input (fractured, as the layer builds it)
        x = _shard(mesh_device, x_host_rows.reshape(1, 1, -1, H))
        n = layer._decode_norm(layer.attention_norm, x, Mode.PREFILL, norm_cfg, keep_sharded=False)
        return n

    cos3 = ttnn.slice(h.tt.rope.cos_device, [0, 0, 0], [1, T, rd])
    sin3 = ttnn.slice(h.tt.rope.sin_device, [0, 0, 0], [1, T, rd])
    cos1 = ttnn.reshape(cos3, (1, 1, T, rd))
    sin1 = ttnn.reshape(sin3, (1, 1, T, rd))
    # per-user path (the production single-user prefill)
    pt_rows = [_replicate(mesh_device, pt[u : u + 1], dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT) for u in range(B)]
    chunk_rows = [ttnn.slice(pt_rows[u], [0, 0], [1, n_blocks]) for u in range(B)]
    single = []
    for u in range(B):
        n_u = normed(hidden[u : u + 1])
        out_u = attn.forward_prefill_paged(
            n_u, cos1, sin1, pt_rows[u], chunk_page_table=chunk_rows[u], chunk_start_idx=0, user_id=0
        )
        single.append(_compose(mesh_device, out_u).float().reshape(T, -1))
        ttnn.deallocate(out_u)
    k_single = ttnn.to_torch(ttnn.get_device_tensors(attn.paged_k)[0]).float().clone()
    v_single = ttnn.to_torch(ttnn.get_device_tensors(attn.paged_v)[0]).float().clone()
    # batched path: all users in one pass; rope tables repeat 0..T-1 per user
    cos_b = ttnn.reshape(ttnn.concat([cos3] * B, dim=1), (1, 1, B * T, rd))
    sin_b = ttnn.reshape(ttnn.concat([sin3] * B, dim=1), (1, 1, B * T, rd))
    n_all = normed(hidden.reshape(1, B * T, H))
    out_b = attn.forward_prefill_paged_batched(
        n_all, cos_b, sin_b, pt_rows, chunk_page_tables=chunk_rows, user_ids=[0] * B
    )
    got = _compose(mesh_device, out_b).float().reshape(B, T, -1)
    k_batched = ttnn.to_torch(ttnn.get_device_tensors(attn.paged_k)[0]).float()
    v_batched = ttnn.to_torch(ttnn.get_device_tensors(attn.paged_v)[0]).float()
    worst = 1.0
    for u in range(B):
        ref = single[u]
        c = torch.corrcoef(torch.stack([ref.flatten(), got[u].flatten()]))[0, 1].item()
        worst = min(worst, c)
        assert torch.allclose(ref, got[u], atol=2e-2, rtol=2e-2), f"user {u}: max diff {(ref - got[u]).abs().max()}"
    assert torch.equal(k_single, k_batched) and torch.equal(v_single, v_batched), "paged K/V differ between the paths"
    print(f"BATCHED_ATTN_PREFILL B={B} T={T}: per-user PCC min {worst:.6f}, K/V cache identical")
