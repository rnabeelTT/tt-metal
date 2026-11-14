# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Self-test for ref_inkling.run_layer: a tiny random Inkling text model, full HF forward vs ref_inkling's
layer-at-a-time path, whole-sequence and chunked (chunks smaller and larger than the sliding window and rel_extent).
No checkpoint needed. usage: python ref_inkling_selftest.py"""

import importlib.util
import os

import torch
from transformers.models.inkling import modeling_inkling as M
from transformers.models.inkling.configuration_inkling import InklingTextConfig

spec = importlib.util.spec_from_file_location("ref_inkling", os.path.join(os.path.dirname(__file__), "ref_inkling.py"))
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)

torch.manual_seed(0)
cfg = InklingTextConfig(
    hidden_size=128,
    num_hidden_layers=6,
    vocab_size=512,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=32,
    swa_head_dim=32,
    swa_num_attention_heads=4,
    swa_num_key_value_heads=2,
    d_rel=8,
    rel_extent=96,
    sliding_window_size=48,
    local_layer_ids=[0, 1, 3, 4],
    dense_mlp_idx=1,
    n_routed_experts=16,
    num_experts_per_tok=3,
    n_shared_experts=2,
    moe_intermediate_size=64,
    dense_intermediate_size=256,
    intermediate_size=64,
    log_scaling_n_floor=64,  # exercise log scaling inside the test length
)
cfg._attn_implementation = "sdpa"
model = M.InklingTextModel(cfg).eval().float()
with torch.no_grad():
    for n, p in model.named_parameters():  # non-trivial values for every param (norms, scales, conv, rel bias)
        p.add_(torch.randn_like(p) * 0.1)
    for lyr in model.layers:
        if hasattr(lyr.mlp, "gate"):
            lyr.mlp.gate.e_score_correction_bias.normal_(0, 0.1)

B, T = 2, 200
ids = torch.randint(0, cfg.vocab_size, (B, T))
with torch.no_grad():
    want = model(input_ids=ids).last_hidden_state
    x0 = model.embed_norm(model.embed_tokens(ids))
    for chunk in (0, 7, 40, 64, 200):
        x = x0
        for lyr in model.layers:
            x = R.run_layer(cfg, lyr, x, chunk)
        got = model.norm(x)
        err = (got - want).abs().max().item()
        print(f"chunk {chunk:3d}: max abs err {err:.2e}")
        assert err < 1e-3, err
print("OK")
