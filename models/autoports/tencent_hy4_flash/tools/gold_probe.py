# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Layer-by-layer check of the eager device forward against the CPU BF16 reference's streams (ref_hy4 gold_<name>.pt:
x_in [42, T, 4, 3072] = the 4 streams entering every layer, hidden [T, 3072] = the final normed hidden).

    python -m models.autoports.tencent_hy4_flash.tools.gold_probe GOLD.pt [--T 64] [--layers 0,1,...] [--teacher]

Default: one forward over the first T tokens, the device streams after each layer vs gold x_in[li + 1] (PCC, max abs).
--teacher: each layer starts from the gold streams entering it (isolates per-layer error from accumulation).
"""

from __future__ import annotations

import argparse

import torch
import ttnn


def pcc(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm() + 1e-30))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gold")
    ap.add_argument("--T", type=int, default=64)
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--layers", default="")
    ap.add_argument("--no-engram", action="store_true")
    ap.add_argument("--teacher", action="store_true")
    args = ap.parse_args()
    from models.autoports.tencent_hy4_flash.tools.generate import build, open_mesh
    from models.autoports.tencent_hy4_flash.tt.common import to_torch0, upload

    g = torch.load(args.gold, weights_only=False)
    ids, x_in = g["ids"][: args.T], g["x_in"][:, : args.T]
    mesh = open_mesh()
    try:
        model = build(mesh, args)
        cfg, T = model.cfg, args.T
        Tp = -(-T // 32) * 32
        model.reset()
        orig = {}
        for layer in model.layers:
            orig[layer.li] = layer.forward

            def hook(X, p0, T_, cos_sin, engram_emb=None, mem=None, engram_keep=None, _l=layer):
                if args.teacher:
                    xg = torch.zeros(1, cfg.hc_mult, Tp, cfg.hidden, dtype=torch.bfloat16)
                    xg[0, :, :T] = x_in[_l.li].permute(1, 0, 2)
                    ttnn.deallocate(X)
                    X = upload(mesh, xg, ttnn.bfloat16)
                if _l.li == 0 or args.teacher:
                    di = to_torch0(X).float()[0, :, :T].permute(1, 0, 2)
                    print(f"layer {_l.li:2d} input: pcc {pcc(di, x_in[_l.li].float()):.5f} "
                          f"max abs {float((di - x_in[_l.li].float()).abs().max()):.3e}", flush=True)
                out = orig[_l.li](X, p0, T_, cos_sin, engram_emb, mem, engram_keep)
                dev = to_torch0(out).float()[0, :, :T].permute(1, 0, 2)  # [T, 4, H]
                ref = x_in[_l.li + 1].float() if _l.li + 1 < x_in.shape[0] else None
                if ref is not None:
                    print(f"layer {_l.li:2d} ({_l.attn.kind:6s}{' engram' if _l.engram is not None else ''}): "
                          f"pcc {pcc(dev, ref):.5f} max abs {float((dev - ref).abs().max()):.3e} "
                          f"(ref max {float(ref.abs().max()):.2e})", flush=True)
                return out

            layer.forward = hook
        lg = model.forward(ids)
        print("next-token argmax", int(lg.argmax()), "ref next id", int(g["ids"][T]) if T < len(g["ids"]) else None)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
