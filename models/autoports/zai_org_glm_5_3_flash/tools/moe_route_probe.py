# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Diagnostic: MoE routing agreement TT vs torch (HF semantics) for a layer's router on random unit-RMS rows."""

import argparse
import json

import torch

import ttnn
from models.autoports.zai_org_glm_5_3_flash.tests.harness import STATS, open_mesh, pcc
from models.autoports.zai_org_glm_5_3_flash.tt.ckpt import CheckpointReader
from models.autoports.zai_org_glm_5_3_flash.tt.functional_decoder import moe_args, moe_state_dict
from models.autoports.zai_org_glm_5_3_flash.tt.moe_opt import Qwen35MoEOptimized as Qwen35MoE
from models.autoports.zai_org_glm_5_3_flash.tt.synthetic import synthetic_layer_state


def hf_route(x, W, b, k=8, scale=2.5):
    s = torch.sigmoid(x.float() @ W.float().T)
    ids = (s + b.float()).topk(k, dim=-1).indices
    w = s.gather(1, ids)
    return ids, w / (w.sum(-1, keepdim=True) + 1e-20) * scale


def hf_moe(x, sd, L=10.0, parts=False):
    ids, w = hf_route(x, sd["mlp.gate.weight"], sd["mlp.gate.e_score_correction_bias"])
    out = torch.zeros(x.shape[0], x.shape[1])
    xf = x.float()

    def mlp(xx, g, u, d):
        gg = (xx @ g.float().T).clamp(max=L)
        uu = (xx @ u.float().T).clamp(-L, L)
        return (torch.nn.functional.silu(gg) * uu) @ d.float().T

    for t in range(x.shape[0]):
        for j in range(ids.shape[1]):
            e = int(ids[t, j])
            out[t] += (
                w[t, j]
                * mlp(
                    xf[t : t + 1],
                    sd[f"mlp.experts.{e}.gate_proj.weight"],
                    sd[f"mlp.experts.{e}.up_proj.weight"],
                    sd[f"mlp.experts.{e}.down_proj.weight"],
                )[0]
            )
    sh = mlp(
        xf,
        sd["mlp.shared_experts.gate_proj.weight"],
        sd["mlp.shared_experts.up_proj.weight"],
        sd["mlp.shared_experts.down_proj.weight"],
    )
    if parts:
        return out, sh, ids
    return out + sh, ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=3)
    ap.add_argument("--weights", default="synthetic")
    ap.add_argument("--rows", type=int, default=64)
    ap.add_argument("--stream", action="store_true", help="MoE input = rmsnorm(ffn_hc.pre(synthetic stream))")
    args = ap.parse_args()
    rd = CheckpointReader()
    sd = (
        rd.layer_state(args.layer)
        if args.weights == "real"
        else synthetic_layer_state(json.loads(STATS.read_text()), args.layer, 0)
    )
    torch.manual_seed(0)
    if args.stream:
        from models.autoports.zai_org_glm_5_3_flash.tests.test_functional_decoder import synthetic_stream

        X = synthetic_stream(args.layer, 2, args.rows).float().reshape(args.rows, 4 * 4096)
        fn, base, sc = (
            sd["hc_ffn_fn"].float(),
            sd["hc_ffn_base"].bfloat16().float(),
            sd["hc_ffn_scale"].bfloat16().float(),
        )
        f = X * torch.rsqrt(X.pow(2).mean(-1, keepdim=True) + 1e-5)
        mix = f @ fn.T
        pre = torch.sigmoid(mix[:, :4] * sc[0] + base[:4]) + 1e-6
        h = (pre.unsqueeze(-1) * X.reshape(args.rows, 4, 4096)).sum(1).bfloat16().float()
        hn = (h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-5)).bfloat16()
        x = (hn * sd["post_attention_layernorm.weight"].bfloat16()).bfloat16()
        print("MoE input rms", x.float().pow(2).mean().sqrt().item())
    else:
        x = torch.randn(args.rows, 4096).bfloat16()
    ref, ref_ids = hf_moe(x, sd)
    mesh = open_mesh()
    try:
        import os

        from models.autoports.zai_org_glm_5_3_flash.tt.moe import MoEPrecision

        margs = moe_args(rd.text_config, 1)
        if os.environ.get("PROBE_PREC") == "bf16":
            margs.moe_precision = MoEPrecision(
                routed_gate_up=ttnn.bfloat16,
                routed_down=ttnn.bfloat16,
                shared=ttnn.bfloat16,
                routed_fidelity=ttnn.MathFidelity.HiFi4,
                shared_fidelity=ttnn.MathFidelity.HiFi4,
            )
        if os.environ.get("PROBE_NOCLAMP") == "1":
            margs.swiglu_limit = None
        moe = Qwen35MoE(mesh, margs, moe_state_dict(sd, 192), layer_num=args.layer)
        tx = ttnn.from_torch(
            x.reshape(1, 1, args.rows, 4096), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh
        )
        scores, ids = moe._route(tx)
        tt_ids = ttnn.to_torch(ids).reshape(args.rows, -1)[:, :8].long()
        agree = sum(set(tt_ids[t].tolist()) == set(ref_ids[t].tolist()) for t in range(args.rows))
        tt_w = ttnn.to_torch(scores).reshape(args.rows, -1)[:, :8].float()
        _, ref_w = hf_route(x, sd["mlp.gate.weight"], sd["mlp.gate.e_score_correction_bias"])
        # align by expert id
        ratios = []
        for t in range(min(args.rows, 16)):
            m = {int(e): float(w) for e, w in zip(ref_ids[t], ref_w[t])}
            for e, w in zip(tt_ids[t].tolist(), tt_w[t].tolist()):
                if e in m:
                    ratios.append(w / m[e])
        rt = torch.tensor(ratios)
        print(
            f"routing weight ratio tt/hf: mean {rt.mean():.4f} min {rt.min():.4f} max {rt.max():.4f}; row sums tt {tt_w.sum(-1)[:4].tolist()} hf {ref_w.sum(-1)[:4].tolist()}"
        )
        print(f"routing sets equal on {agree}/{args.rows} rows")
        y = ttnn.to_torch(moe.forward(tx, mode="prefill")).reshape(args.rows, 4096).float()
        print(f"MoE out PCC {pcc(y, ref):.6f} rms ratio tt/ref {(y.norm() / ref.norm()).item():.4f}")
        r_routed, r_shared, _ = hf_moe(x, sd, parts=True)
        sh_tt = ttnn.to_torch(moe._shared(tx)).reshape(args.rows, 4096).float()
        print(f"shared PCC {pcc(sh_tt, r_shared):.6f} rms ratio {(sh_tt.norm() / r_shared.norm()).item():.4f}")
        rt = y - sh_tt
        print(f"routed PCC {pcc(rt, r_routed):.6f} rms ratio {(rt.norm() / r_routed.norm()).item():.4f}")
        if os.environ.get("PROBE_DENSE") == "1":  # one expert densely from the MoE's own packed weights
            e = int(ref_ids[0, 0])
            N = moe.inter_local
            wgu = ttnn.slice(moe.up_gate, (0, e, 0, 0), (1, e + 1, 4096, 2 * N))
            wd = ttnn.slice(moe.down, (0, e, 0, 0), (1, e + 1, N, 4096))
            x0 = ttnn.from_torch(
                x[:1].reshape(1, 1, 1, 4096), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh
            )
            fu = ttnn.to_torch(ttnn.matmul(x0, wgu)).reshape(-1)[: 2 * N].float()
            up_ref = x[0].float() @ sd[f"mlp.experts.{e}.up_proj.weight"].float().T
            g_ref = x[0].float() @ sd[f"mlp.experts.{e}.gate_proj.weight"].float().T
            print(
                f"dense expert {e}: up PCC {pcc(fu[:N], up_ref):.5f} ratio {(fu[:N].norm()/up_ref.norm()).item():.4f}; "
                f"gate PCC {pcc(fu[N:], g_ref):.5f} ratio {(fu[N:].norm()/g_ref.norm()).item():.4f}"
            )
            h = (torch.nn.functional.silu(g_ref) * up_ref).bfloat16()
            hd = ttnn.from_torch(h.reshape(1, 1, 1, N), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)
            dn = ttnn.to_torch(ttnn.matmul(hd, wd)).reshape(-1)[:4096].float()
            dn_ref = h.float() @ sd[f"mlp.experts.{e}.down_proj.weight"].float().T
            print(f"dense expert {e}: down PCC {pcc(dn, dn_ref):.5f} ratio {(dn.norm()/dn_ref.norm()).item():.4f}")
            # the sparse gather path for this single expert: silu on device
            tt_h = ttnn.multiply(
                ttnn.from_torch(
                    up_ref.bfloat16().reshape(1, 1, 1, N), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh
                ),
                ttnn.from_torch(
                    g_ref.bfloat16().reshape(1, 1, 1, N), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh
                ),
                input_tensor_b_activations=[ttnn.UnaryOpType.SILU],
            )
            th = ttnn.to_torch(tt_h).reshape(-1)[:N].float()
            print(f"fused silu-mul PCC {pcc(th, h.float()):.5f} ratio {(th.norm()/h.float().norm()).item():.4f}")
            sw = (
                ttnn.to_torch(
                    ttnn.swiglu(
                        ttnn.from_torch(
                            torch.cat([up_ref, g_ref]).bfloat16().reshape(1, 1, 1, 2 * N),
                            dtype=ttnn.bfloat16,
                            layout=ttnn.TILE_LAYOUT,
                            device=mesh,
                        )
                    )
                )
                .reshape(-1)[:N]
                .float()
            )
            print(f"ttnn.swiglu PCC {pcc(sw, h.float()):.5f} ratio {(sw.norm()/h.float().norm()).item():.4f}")
        save = os.environ.get("PROBE_SAVE")
        if save:
            torch.save({"x": x, "tt_routed": rt, "tt_shared": sh_tt, "ids": tt_ids, "w": tt_w}, save)
        good = [t for t in range(args.rows) if set(tt_ids[t].tolist()) == set(ref_ids[t].tolist())]
        print(f"MoE out PCC on agreeing rows {pcc(y[good], ref[good]):.6f}")
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
