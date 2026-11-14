# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only: the port's MoE formulation equals transformers' InklingMoE.

Port (tt/optimized_decoder.py + tt/weights.py): the 2 shared experts are experts 256, 257; every token takes the 6
sigmoid+bias picks plus the 2 shared, weighted by sigmoid / (sum of the 8 sigmoids), and every expert's down projection
carries route_scale * global_scale. HF: routed experts weighted by the 8-way logsigmoid softmax x route_scale x
global_scale, plus the shared experts with their gammas (from the same normalisation) inside the SwiGLU."""

import pytest
import torch


def test_moe_formulation_matches_hf():
    M = pytest.importorskip("transformers.models.inkling.modeling_inkling")
    from transformers.models.inkling.configuration_inkling import InklingTextConfig

    torch.manual_seed(0)
    cfg = InklingTextConfig(hidden_size=64, n_routed_experts=16, num_experts_per_tok=3, n_shared_experts=2,
                            moe_intermediate_size=32, route_scale=8.0)
    moe = M.InklingMoE(cfg).float().eval()
    with torch.no_grad():
        for p in moe.parameters():
            p.normal_(0, 0.2)
        moe.gate.global_scale.fill_(0.37)
        moe.gate.e_score_correction_bias.normal_(0, 0.3)
    x = torch.randn(5, 64)
    with torch.no_grad():
        want = moe(x.unsqueeze(0))[0]

        E, S, K = 16, 2, 3
        scale = 8.0 * 0.37
        logits = x @ moe.gate.weight.T  # [T, 18]
        sig = logits.sigmoid()
        picks = torch.topk(sig[:, :E] + moe.gate.e_score_correction_bias, K, dim=-1).indices
        ids = torch.cat([picks, torch.arange(E, E + S).expand(x.shape[0], -1)], -1)  # [T, 5]
        w = sig.gather(1, ids)
        w = w / w.sum(-1, keepdim=True)  # plain normalisation (sum 1)

        gu_r = moe.experts.gate_up_proj  # [E, 2I, H] gate rows first
        dn_r = moe.experts.down_proj  # [E, H, I]
        gu_s = torch.cat([moe.shared_experts.gate_proj, moe.shared_experts.up_proj], 1)  # [S, 2I, H]
        dn_s = moe.shared_experts.down_proj
        gu = torch.cat([gu_r, gu_s], 0)
        dn = torch.cat([dn_r, dn_s], 0) * scale  # the folded constant
        out = torch.zeros_like(x)
        for t in range(x.shape[0]):
            for j in range(ids.shape[1]):
                e = ids[t, j]
                g, u = (gu[e] @ x[t]).chunk(2)
                out[t] += w[t, j] * (dn[e] @ (torch.nn.functional.silu(g) * u))
    torch.testing.assert_close(out, want, rtol=1e-4, atol=1e-4)
