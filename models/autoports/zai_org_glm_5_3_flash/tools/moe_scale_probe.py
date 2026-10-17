# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Diagnostic (stage-1 review #1, P2 "output-scale"): where does the routed-MoE output scale deficit come from?

On one layer's real weights and MoE-input rows drawn from the real stream statistics (as tools/moe_route_probe.py
--stream), compare
  ref      torch MoE with the checkpoint weights (fp32 math)
  ref_q8   the same with every expert / shared weight passed through the device block-float conversion
           (ttnn.from_torch(W^T, bfloat8_b, TILE) -> to_torch: the exact host packing the port uses)
  ref_q4   ditto with bfloat4_b
  tt       the port's MoE (Qwen35MoEOptimized, functional-decoder precision)
and report RMS ratios and PCC: tt/ref vs ref_q8/ref separates conversion loss from device compute.

usage: python tools/moe_scale_probe.py --layer 4 --rows 128 [--mode decode]"""

import argparse

import torch

import ttnn
from models.autoports.zai_org_glm_5_3_flash.tests.harness import open_mesh, pcc
from models.autoports.zai_org_glm_5_3_flash.tools.moe_route_probe import hf_moe
from models.autoports.zai_org_glm_5_3_flash.tt.ckpt import CheckpointReader
from models.autoports.zai_org_glm_5_3_flash.tt.functional_decoder import FUNCTIONAL_MOE, moe_args, moe_state_dict
from models.autoports.zai_org_glm_5_3_flash.tt.moe_opt import Qwen35MoEOptimized


def quantized(sd: dict, dtype) -> dict:
    """Expert / shared projection weights through the host block-float packing ([K, N] = W^T orientation, TILE)."""
    out = dict(sd)
    for k, w in sd.items():
        if k.startswith(("mlp.experts.", "mlp.shared_experts.")) and k.endswith(".weight") and w.dim() == 2:
            t = ttnn.from_torch(w.float().T.contiguous(), dtype=dtype, layout=ttnn.TILE_LAYOUT)
            out[k] = ttnn.to_torch(t).float()[: w.shape[1], : w.shape[0]].T.contiguous()
    return out


def stream_rows(sd, layer, rows):
    from models.autoports.zai_org_glm_5_3_flash.tests.test_functional_decoder import synthetic_stream

    X = synthetic_stream(layer, 2, rows).float().reshape(rows, 4 * 4096)
    fn, base, sc = sd["hc_ffn_fn"].float(), sd["hc_ffn_base"].bfloat16().float(), sd["hc_ffn_scale"].bfloat16().float()
    f = X * torch.rsqrt(X.pow(2).mean(-1, keepdim=True) + 1e-5)
    pre = torch.sigmoid((f @ fn.T)[:, :4] * sc[0] + base[:4]) + 1e-6
    h = (pre.unsqueeze(-1) * X.reshape(rows, 4, 4096)).sum(1).bfloat16().float()
    hn = (h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-5)).bfloat16()
    return (hn * sd["post_attention_layernorm.weight"].bfloat16()).bfloat16()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=4)
    ap.add_argument("--rows", type=int, default=128)
    ap.add_argument("--grouped-in0", type=int, default=None, help="K block (tiles) of the grouped sparse matmuls")
    ap.add_argument("--no-grouped", action="store_true", help="non-grouped sorted sparse matmuls")
    ap.add_argument("--skip-ref", action="store_true", help="TT only (reuse the printed reference numbers)")
    ap.add_argument("--policy", default=None, help="MoE policy of a tt/optimized_decoder.CANDIDATES entry")
    ap.add_argument("--mode", default="prefill", choices=("prefill", "decode"), help="MoE path (decode: 32-row tiles)")
    a = ap.parse_args()
    rd = CheckpointReader()
    sd = rd.layer_state(a.layer)
    x = stream_rows(sd, a.layer, a.rows)
    ref_r, ref_s, ids = hf_moe(x, sd, parts=True)
    q8 = quantized(sd, ttnn.bfloat8_b)
    q8_r, q8_s, ids8 = hf_moe(x, q8, parts=True)
    q4 = quantized(sd, ttnn.bfloat4_b)
    q4_r, q4_s, _ = hf_moe(x, q4, parts=True)
    same = torch.tensor([set(ids[t].tolist()) == set(ids8[t].tolist()) for t in range(a.rows)])
    rat = lambda u, v: (u.norm() / v.norm()).item()
    print(f"layer {a.layer}, {a.rows} rows (routing identical in ref / ref_q8: {int(same.sum())}/{a.rows})")
    print(
        f"ref_q8 vs ref: routed rms {rat(q8_r, ref_r):.4f} PCC {pcc(q8_r, ref_r):.6f}; shared rms {rat(q8_s, ref_s):.4f}"
    )
    print(
        f"ref_q4 vs ref: routed rms {rat(q4_r, ref_r):.4f} PCC {pcc(q4_r, ref_r):.6f}; shared rms {rat(q4_s, ref_s):.4f}"
    )
    mesh = open_mesh()
    try:
        import dataclasses

        opt = FUNCTIONAL_MOE
        if a.policy:
            from models.autoports.zai_org_glm_5_3_flash.tt.optimized_decoder import CANDIDATES

            opt = CANDIDATES[a.policy].moe
        if a.grouped_in0:
            opt = dataclasses.replace(opt, grouped_gate_up_in0=a.grouped_in0, grouped_down_in0=a.grouped_in0)
        if a.no_grouped:
            opt = dataclasses.replace(opt, sorted_grouped=False)
        print(
            f"config: sorted_grouped={opt.sorted_grouped} grouped_in0={opt.grouped_gate_up_in0}/{opt.grouped_down_in0}"
        )
        margs = moe_args(rd.text_config, 1, opt=opt)
        moe = Qwen35MoEOptimized(
            mesh, margs, moe_state_dict(sd, 192), tensor_cache_path=None, tt_ccl=None, layer_num=a.layer
        )
        tx = ttnn.from_torch(x.reshape(1, 1, a.rows, 4096), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)
        sh = ttnn.to_torch(moe._shared(tx)).reshape(a.rows, 4096).float()
        if a.mode == "prefill":
            y = ttnn.to_torch(moe.forward(tx, mode="prefill")).reshape(a.rows, 4096).float()
        else:  # decode path: one 32-row tile of users per call (rows % 32 == 0)
            ys = []
            for r0 in range(0, a.rows, 32):
                xt = ttnn.from_torch(
                    x[r0 : r0 + 32].reshape(1, 1, 32, 4096), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh
                )
                ys.append(ttnn.to_torch(moe.forward(xt, mode="decode")).reshape(32, 4096).float())
            y = torch.cat(ys)
        rt = y - sh
        # per-row routed scale (decode checks are single rows)
        rr = rt.norm(dim=-1) / ref_r.norm(dim=-1)
        print(f"per-row routed rms ratio: mean {rr.mean():.4f} min {rr.min():.4f} max {rr.max():.4f}")
        print(
            f"tt vs ref:    routed rms {rat(rt, ref_r):.4f} PCC {pcc(rt, ref_r):.6f}; shared rms {rat(sh, ref_s):.4f} PCC {pcc(sh, ref_s):.6f}"
        )
        print(
            f"tt vs ref_q8: routed rms {rat(rt, q8_r):.4f} PCC {pcc(rt, q8_r):.6f}; shared rms {rat(sh, q8_s):.4f} PCC {pcc(sh, q8_s):.6f}"
        )
        print(
            f"tt vs ref_q4: routed rms {rat(rt, q4_r):.4f} PCC {pcc(rt, q4_r):.6f}; shared rms {rat(sh, q4_s):.4f} PCC {pcc(sh, q4_s):.6f}"
        )
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
