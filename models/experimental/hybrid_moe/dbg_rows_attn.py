# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Rows-mode attention (4 verify rows) of MiniMax-M2.7 layer 0: the prep / post ops path (QWEN35_ROWS_ATTN_OPS) against
the op chain, same inputs, two consecutive passes (the second reads the first's K / V rows). Prints the PCC of the
layer outputs and the device time per call. usage (in the image): python dbg_rows_attn.py <out dir>"""

import time

import torch

import ttnn
from models.autoports.minimaxai_minimax_m2_7.tt import mtp_rows as rows
from models.autoports.minimaxai_minimax_m2_7.tt.generator import build_generator

DRAM = ttnn.DRAM_MEMORY_CONFIG


def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def main():
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
        if att._rows_ops is None:
            raise RuntimeError("rows ops not built for this layer")
        R = 4
        BW = rows.rows_width(R)
        W = int(gen.page_table.shape[-1])
        pt_row = gen.page_table[0]
        comp = ttnn.ConcatMeshToTensor(mesh, dim=-1)
        torch.manual_seed(0)
        toks = torch.randint(0, 1000, (BW,), dtype=torch.int32)

        def run(p0, use_ops):
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
            saved = att._rows_ops
            att._rows_ops = saved if use_ops else None
            try:
                ttnn.synchronize_device(mesh)
                t0 = time.perf_counter()
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
                ttnn.synchronize_device(mesh)
                dt = time.perf_counter() - t0
            finally:
                att._rows_ops = saved
            return ttnn.to_torch(y, mesh_composer=comp).float(), dt

        res = {}
        for use_ops in (False, True):
            outs = [run(64, use_ops), run(68, use_ops)]  # the second pass attends to the first's rows
            res[use_ops] = outs
        for i in range(2):
            a, b = res[False][i][0], res[True][i][0]
            print(
                f"ROWS_ATTN pass {i}: pcc {pcc(a, b):.6f} max diff {float((a - b).abs().max()):.4f} "
                f"(chain {res[False][i][1] * 1e3:.2f} ms, ops {res[True][i][1] * 1e3:.2f} ms eager)",
                flush=True,
            )
        # timing: eager repeats
        for use_ops in (False, True):
            ts = [run(72, use_ops)[1] for _ in range(5)]
            print(
                f"ROWS_ATTN eager layer step {'ops' if use_ops else 'chain'}: {sorted(ts)[2] * 1e3:.2f} ms", flush=True
            )
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()
