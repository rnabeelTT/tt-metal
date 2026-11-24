# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device modules vs the HF reference modules (transformers PR 50903) with the checkpoint's weights, one layer.

    HY4_LAYER=1 python -m pytest -q -s models/autoports/tencent_hy4_flash/tests/test_layer_device.py

Inputs are realistic: the embedding of a short text expanded to the 4 streams, so norms see real scales. Reports PCC
and max abs error per module; the thresholds are bring-up gates (bfp4 experts / bfp8 attention weights move outputs
at the 1e-2 level), the real quality gate is the teacher-forced comparison against ref_hy4.
"""

from __future__ import annotations

import os

import pytest
import torch

import ttnn

LAYER = int(os.environ.get("HY4_LAYER", "1"))
T = int(os.environ.get("HY4_T", "40"))


def pcc(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm() + 1e-30))


@pytest.fixture(scope="module")
def env():
    from models.autoports.tencent_hy4_flash.tt.config import Hy4Config
    from models.autoports.tencent_hy4_flash.tt.weights import Hy4Weights

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4))
    cfg = Hy4Config.load()
    w = Hy4Weights()
    yield mesh, cfg, w
    ttnn.close_mesh_device(mesh)


def hf_layer(cfg, w, li):
    """The HF decoder layer with this layer's checkpoint weights (bf16), engram excluded."""
    from transformers import AutoConfig
    from transformers.models.hy_v4_flash import modeling_hy_v4_flash as M

    full = AutoConfig.from_pretrained(w.dir)
    tc = full.text_config
    tc._attn_implementation = "eager"
    layer = M.HYV4FlashDecoderLayer(tc, li).to(torch.bfloat16)
    sd = {}
    for k in w.layer(li):
        if "engram_layer" in k:
            continue
        sd[k] = w.layer(li)[k]
    # checkpoint -> module names (the PR's conversion mapping): hc_*_layer.hc_pre.hc_{fn,base,scale} -> *_hc.{fn,base,scale}
    ren = {}
    for k, v in sd.items():
        k2 = k.replace("hc_attn_layer.hc_pre.hc_", "attn_hc.").replace("hc_mlp_layer.hc_pre.hc_", "ffn_hc.")
        k2 = k2.replace("self_attn.linear_gate.", "self_attn.gate_proj.").replace("self_attn.learnable_sink_param", "self_attn.sinks")
        k2 = k2.replace("self_attn.compressed_indexer.", "self_attn.compressed_indexer.")
        ren[k2] = v
    missing, unexpected = layer.load_state_dict(ren, strict=False)
    missing = [m for m in missing if "engram" not in m]
    assert not missing, f"missing {missing[:8]}"
    assert not unexpected, f"unexpected {unexpected[:8]}"
    return layer, tc, M


def streams(cfg, w, T, seed=0):
    g = torch.Generator().manual_seed(seed)
    emb = w.get("model.embed_tokens.weight")
    ids = torch.randint(0, 100000, (T,), generator=g)
    x = emb[ids].to(torch.bfloat16)
    # perturb the streams a little so they differ (as after a few layers)
    xs = [x + 0.02 * torch.randn(x.shape, generator=g).to(torch.bfloat16) * x.abs().mean() for _ in range(cfg.hc_mult)]
    return torch.stack(xs, dim=1)  # [T, 4, H]


def test_layer(env):
    mesh, cfg, w = env
    from models.autoports.tencent_hy4_flash.tt.common import to_torch0, upload
    from models.autoports.tencent_hy4_flash.tt.model import Hy4Layer
    from models.autoports.tencent_hy4_flash.tt.rope import DeviceRope

    li = LAYER
    hl, tc, M = hf_layer(cfg, w, li)
    X = streams(cfg, w, T)  # [T, 4, H]
    # ---------------- HF reference (dense regime: T < 2048, no cache)
    pos = torch.arange(T).reshape(1, -1)
    rot = M.HYV4FlashRotaryEmbedding(tc)
    pe = {lt: rot(X, position_ids=pos[None].expand(3, 1, -1), layer_type=lt) for lt in rot.layer_types}
    lt = hl.attention_type
    from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask

    inp = X.unsqueeze(0)
    mask_fn = create_sliding_window_causal_mask if lt == "sliding_attention" else create_causal_mask
    mask = mask_fn(config=tc, inputs_embeds=X.unsqueeze(0)[:, :, 0], attention_mask=None, cache=None, position_ids=pos)
    with torch.no_grad():
        # piecewise, to localize errors
        post_a, a_in = hl.attn_hc(inp)
        a_norm = hl.input_layernorm(a_in)
        a_out, _ = hl.self_attn(hidden_states=a_norm, attention_mask=mask, position_ids=pos[None].expand(3, 1, -1),
                                position_embeddings=pe[lt], cache=None)
        ref_full = hl(inp, attention_mask=mask, position_ids=pos[None].expand(3, 1, -1), position_embeddings=pe[lt])
        x2 = (post_a.float().unsqueeze(-1) * a_out.float().unsqueeze(-2) + inp.float()).to(torch.bfloat16)
        post_f, f_in = hl.ffn_hc(x2)
        f_norm = hl.post_attention_layernorm(f_in)
        f_out = hl.mlp(f_norm)
    # ---------------- device
    layer = Hy4Layer(mesh, cfg, w, li, max_ctx=4096, chunk=256)
    Tp = -(-T // 32) * 32
    t = torch.zeros(1, cfg.hc_mult, Tp, cfg.hidden, dtype=torch.bfloat16)
    t[0, :, :T] = X.transpose(0, 1)
    Xd = upload(mesh, t, ttnn.bfloat16)
    a_in_d, post_d = layer.attn_hc.pre(Xd)
    got = to_torch0(a_in_d)[0, 0, :T]
    print(f"attn HC pre: pcc {pcc(got, a_in[0]):.6f} max {float((got.float() - a_in[0].float()).abs().max()):.4g}")
    rope = DeviceRope(mesh, cfg.qk_rope)
    p = torch.arange(Tp)
    cos_sin = {k: rope.tables(p, cfg.rope_theta[k]) for k in ("sparse", "swa")}
    a_d = layer.attn.forward(layer.in_norm(a_in_d), 0, T, cos_sin)
    got = to_torch0(a_d)[0, 0, :T]
    pa = pcc(got, a_out[0])
    print(f"attention ({layer.attn.kind}): pcc {pa:.6f} max {float((got.float() - a_out[0].float()).abs().max()):.4g}")
    # FFN half from the reference's own x2 (isolates it from attention error)
    t2 = torch.zeros(1, cfg.hc_mult, Tp, cfg.hidden, dtype=torch.bfloat16)
    t2[0, :, :T] = x2[0].transpose(0, 1)
    X2d = upload(mesh, t2, ttnn.bfloat16)
    f_in_d, post_fd = layer.ffn_hc.pre(X2d)
    got = to_torch0(f_in_d)[0, 0, :T]
    print(f"ffn HC pre: pcc {pcc(got, f_in[0]):.6f} max {float((got.float() - f_in[0].float()).abs().max()):.4g}")
    fn_d = layer.post_norm(f_in_d)
    got = to_torch0(fn_d)[0, 0, :T]
    print(f"post_attention_layernorm: pcc {pcc(got, f_norm[0]):.6f}")
    # the MLP on the reference's normed input
    tn = torch.zeros(1, 1, Tp, cfg.hidden, dtype=torch.bfloat16)
    tn[0, 0, :T] = f_norm[0]
    fnd = upload(mesh, tn, ttnn.bfloat16)
    from models.autoports.tencent_hy4_flash.tt.moe import Hy4MoE

    mo = layer.mlp.forward(fnd, T) if isinstance(layer.mlp, Hy4MoE) else layer.mlp.forward(fnd)
    got = to_torch0(mo)[0, 0, :T]
    print(f"mlp ({type(layer.mlp).__name__}): pcc {pcc(got, f_out[0]):.6f} max {float((got.float() - f_out[0].float()).abs().max()):.4g} "
          f"(ref max {float(f_out[0].float().abs().max()):.3g})")
    layer.reset()
    layer2 = layer
    layer2.attn.indexer and layer2.attn.indexer.reset()
    out = layer2.forward(Xd, 0, T, cos_sin)
    got = to_torch0(out)[0, :, :T].transpose(0, 1)
    pl = pcc(got, ref_full[0])
    print(f"layer {li}: pcc {pl:.6f} max {float((got.float() - ref_full[0].float()).abs().max()):.4g}")
    assert pa > 0.98 and pl > 0.98
