# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Rows-mode attention of MiniMax-M2.7 layer 0 with the stock full-width QK norm (_full_qk_norm + RoPE-only prep)
against the fused one (QWEN35_QKNORM_FUSED: QKStats + all_gather + prep norm="full"), same inputs: the q / k the prep
writes (captured by wrapping both prep calls), the layer outputs, and q / k of both against an FP32 host reference
built from the stock path's raw qkv (the projection before any norm).
usage (in the image, QWEN35_QKNORM_FUSED=1): python dbg_qkfused.py"""

import os

import torch

import ttnn
from models.autoports.minimaxai_minimax_m2_7.tt import mtp_rows as rows
from models.autoports.minimaxai_minimax_m2_7.tt.generator import build_generator

DRAM = ttnn.DRAM_MEMORY_CONFIG


def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def main():
    if os.environ.get("QWEN35_QKNORM_FUSED") != "1":
        raise RuntimeError("run with QWEN35_QKNORM_FUSED=1 (the fused path is built at layer setup)")
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), l1_small_size=24576, trace_region_size=20_000_000)
    try:
        gen = build_generator(
            "models/autoports/minimaxai_minimax_m2_7",
            mesh,
            max_seq_len=4096,
            warm_decode_trace=False,
            warm_prefill_trace=False,
            layer_indices=[0],
        )
        m = gen.model
        layer = m.layers[0]
        att = layer.attention
        if att._rows_ops is None or att._qk_fused is None:
            raise RuntimeError(f"rows ops {att._rows_ops is not None}, fused {att._qk_fused is not None}")
        R = 4
        BW = rows.rows_width(R)
        pt_row = gen.page_table[0]
        cat = lambda t: ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)).float()  # noqa: E731
        torch.manual_seed(0)
        toks = torch.randint(0, 1000, (BW,), dtype=torch.int32)
        cap = {}

        def wrap(fn, tag):
            def w(*a, **k):
                out = fn(*a, **k)
                cap[tag + "_qkv"] = cat(a[0])
                cap[tag + "_q"], cap[tag + "_k"] = cat(out[0]), cat(out[1])
                return out

            return w

        stock_ops, fused = att._rows_ops, att._qk_fused
        orig_qkv = att._qkv

        def qkv_spy(x, qk_norm=True):
            raw = orig_qkv(x, qk_norm=False)
            cap["raw_qkv"] = cat(raw[0])
            print(f"QKF raw qkv dtype {raw[0].dtype} layout {raw[0].layout} mem {raw[0].memory_config()}", flush=True)
            if not qk_norm:
                return raw
            return (att._full_qk_norm(raw[0]),) + tuple(raw[1:])

        att._qkv = qkv_spy

        def run(p0, use_fused):
            pos = list(range(p0, p0 + R))
            tok, pos_t, rot, pt = m.prepare_decode_inputs_host(
                toks, torch.tensor(pos, dtype=torch.int32), pt_row.repeat(BW, 1)
            )
            dev = lambda t: ttnn.to_device(t, mesh, memory_config=DRAM)  # noqa: E731
            tok, pos_t, rot, pt = dev(tok), dev(pos_t), dev(rot), dev(pt)
            masks = dev(rows.rows_pos_masks_host(pos, BW))
            cos, sin = m._rope_from_idxs(rot)
            x = m.embd(m._decode_tokens_view(tok, BW))
            x = ttnn.to_memory_config(ttnn.reshape(x, (1, 1, BW, x.shape[-1])), DRAM)
            att._rows_ops = (wrap(stock_ops[0], "stock"),) + tuple(stock_ops[1:])
            att._qk_fused = (wrap(fused[0], "fused"),) + tuple(fused[1:]) if use_fused else None
            try:
                y = rows.layer_forward_rows(
                    layer,
                    x,
                    R,
                    cos=cos,
                    sin=sin,
                    position_tensor=pos_t,
                    page_table=pt,
                    rows_pos_masks=masks,
                    gdn_index=0,
                )
            finally:
                att._rows_ops, att._qk_fused = stock_ops, fused
            return ttnn.to_torch(y, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=-1)).float()

        ys = run(64, False)
        s_q, s_k, raw_s = cap["stock_q"], cap["stock_k"], cap["raw_qkv"]
        yf = run(64, True)
        f_q, f_k, raw_f = cap["fused_q"], cap["fused_k"], cap["raw_qkv"]
        print(f"QKF raw qkv identical: {torch.equal(raw_s, raw_f)} shape {tuple(raw_s.shape)}", flush=True)
        print(
            f"QKF stock prep input = normed qkv {tuple(cap['stock_qkv'].shape)}; fused prep input = raw: "
            f"{torch.equal(cap['fused_qkv'], raw_f)}",
            flush=True,
        )
        rr = slice(0, R)
        print(f"QKF q shape {tuple(s_q.shape)} k shape {tuple(s_k.shape)}", flush=True)
        for name, a, b in (("q", s_q, f_q), ("k", s_k, f_k)):
            a1, b1 = a.reshape(4, -1, *a.shape[2:]), b.reshape(4, -1, *b.shape[2:])
            print(f"QKF {name} all rows: pcc {pcc(a, b):.6f} max {float((a - b).abs().max()):.4f}", flush=True)
            for d in range(4):
                print(f"QKF {name} die {d}: pcc {pcc(a1[d], b1[d]):.6f} max {float((a1[d] - b1[d]).abs().max()):.4f}")
        print(f"QKF layer out pcc {pcc(ys, yf):.6f} max {float((ys - yf).abs().max()):.4f}", flush=True)
        torch.save(dict(cap=cap, ys=ys, yf=yf), "/poc/runs/m27/dbg_qkfused.pt")
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()
