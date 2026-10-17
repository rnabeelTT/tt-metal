# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Diagnostic (stage-1 review #5 P2-B): the DSA attention of one layer, stage by stage, on HF's own attention input,
against an fp32 torch computation of the same absorbed-MLA math with the port's own weights (read back from the
device: bfp8 q_a / q_b / kv_a / o, bf16 W_UK / W_UV), so every ratio is device compute only (no weight rounding).

Stages: qr = RMSNorm(x Wqa), q = qr Wqb, q_abs = q W_UK (per head), c = RMSNorm(x Wkva) (latent), o_lat = softmax(
q_abs c^T / 16, causal) c (flash_mla_prefill; and sparse_sdpa with the dense index list), o = o_lat W_UV^T, y = o Wo.
Each stage is also fed with the torch value of its input where possible (isolates that op).

usage: python -m models.autoports.zai_org_glm_5_3_flash.tools.dsa_stage_probe --layer 3 --n 100 --ref-n 108"""

import argparse

import torch

import ttnn
from models.autoports.zai_org_glm_5_3_flash.tests.harness import build_layer, open_mesh, pcc
from models.autoports.zai_org_glm_5_3_flash.tests.hf_ref.client import hf_layer_reference
from models.autoports.zai_org_glm_5_3_flash.tests.test_functional_decoder import tokens
from models.autoports.zai_org_glm_5_3_flash.tt.ckpt import CheckpointReader


def rms(x, w, eps):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=3)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--ref-n", type=int, default=None)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    N = a.ref_n or a.n
    r = hf_layer_reference(
        {"layer": a.layer, "weights": "real", "tokens": tokens(a.seed, N), "segments": [N], "taps": True}
    )
    xin = r["taps"]["attn_in"].float().reshape(N, -1)[: a.n]
    hf_out = r["taps"]["attn_out"].float().reshape(N, -1)[: a.n]
    rd = CheckpointReader()
    mesh = open_mesh()
    try:
        dec = build_layer(mesh, rd, a.layer, "real", batch=8, prefill_lengths=(128, 2048))
        at = dec.attn
        n = 128
        get = lambda t: ttnn.to_torch(t).float()
        Wqa, Wqb, Wkva, Wo = (get(t).reshape(t.shape[-2], t.shape[-1]) for t in (at.w_q_a, at.w_q_b, at.w_kv_a, at.w_o))
        Wuk = get(at.w_uk).reshape(at.H, at.Dn, at.R)  # q_abs = q @ W_UK
        Wuv = get(at.w_uv_t).reshape(at.H, at.R, at.Dv)  # o = o_lat @ W_UV^T
        qan = get(at.q_a_norm).reshape(-1)
        kvn = get(at.kv_a_norm).reshape(-1)
        eps = at.eps
        # torch fp32 reference with the device weights
        x = xin
        t_qa = x @ Wqa
        t_qr = rms(t_qa, qan, eps)
        t_q = (t_qr @ Wqb).reshape(a.n, at.H, at.Dn).transpose(0, 1)  # [H, n, 256]
        t_qabs = t_q @ Wuk  # [H, n, 512]
        t_c = rms(x @ Wkva, kvn, eps)  # [n, 512]
        s = (t_qabs @ t_c.T) * at.scale
        s = s.masked_fill(torch.triu(torch.ones(a.n, a.n, dtype=torch.bool), 1), float("-inf"))
        t_olat = torch.softmax(s, -1) @ t_c  # [H, n, 512]
        t_o = t_olat @ Wuv  # [H, n, 256]
        t_y = t_o.transpose(0, 1).reshape(a.n, -1) @ Wo

        def dev(t, dtype=ttnn.bfloat16):
            """[rows, D] or [H, rows, D] torch -> [1, 1|H, n, D] device tensor (rows zero-padded to n)."""
            tp = torch.nn.functional.pad(t, (0, 0, 0, n - t.shape[-2]))
            while tp.dim() < 4:
                tp = tp.unsqueeze(0)
            return ttnn.from_torch(tp, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=mesh)

        def rep(name, tt, ref):
            print(f"{name:26s} PCC {pcc(tt, ref):.6f} rms ratio {(tt.norm() / ref.norm()).item():.5f}", flush=True)

        rows = lambda t: get(t)[..., : a.n, :]
        X = dev(xin)
        qa = at._mm(X, at.w_q_a)
        rep("q_a (x Wqa)", rows(qa)[0, 0], t_qa)
        qr = at._rms(qa, at.q_a_norm)
        rep("qr", rows(qr)[0, 0], t_qr)
        rep("qr on torch q_a", rows(at._rms(dev(t_qa, dtype=qa.dtype), at.q_a_norm))[0, 0], t_qr)
        q_abs, _ = at._q_abs(X, n)
        rep("q_abs", rows(q_abs)[0], t_qabs)
        q_t = dev(t_q)  # [1, H, n, 256]
        rep("q_abs on torch q", rows(at._bmm(q_t, at.w_uk, n))[0], t_qabs)
        c = at._latent(X)
        rep("latent c", rows(c)[0, 0], t_c)
        o_lat = ttnn.transformer.flash_mla_prefill(
            q_abs,
            c,
            at.R,
            is_causal=True,
            scale=at.scale,
            compute_kernel_config=at.sdpa_ckc,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        rep("o_lat (flash)", rows(o_lat)[0], t_olat)
        qa_t = dev(t_qabs)
        c_t = dev(t_c)
        o_lat2 = ttnn.transformer.flash_mla_prefill(
            qa_t,
            c_t,
            at.R,
            is_causal=True,
            scale=at.scale,
            compute_kernel_config=at.sdpa_ckc,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        rep("o_lat (flash) torch in", rows(o_lat2)[0], t_olat)
        o = at._bmm(o_lat, at.w_uv_t, n)
        rep("o (o_lat W_UV)", rows(o)[0], t_o)
        rep("o on torch o_lat", rows(at._bmm(dev(t_olat), at.w_uv_t, n))[0], t_o)
        y = at._out(o_lat, n)
        rep("attn out", rows(y)[0, 0], t_y)
        o_cat = dev(t_o.transpose(0, 1).reshape(a.n, -1))
        rep("attn out on torch o", rows(at._mm(o_cat, at.w_o))[0, 0], t_y)
        rep("torch(dev weights) vs HF bf16", t_y, hf_out)
        rep("attn out vs HF bf16", rows(y)[0, 0], hf_out)
        # the full prefill path (flash for this length) for comparison
        inputs = dec.prefill_inputs(n, start=0, length=a.n, slot=0)
        yp, _ = at.prefill_forward(X, inputs=inputs["attn"])
        rep("prefill_forward", rows(yp)[0, 0], t_y)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
