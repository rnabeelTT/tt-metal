# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Bring-up smoke: one real-weight decoder layer on a (1,1) mesh vs the HF layer (prefill + decode steps).

python -m models.autoports.zai_org_glm_5_3_flash.tools.smoke_layer --layer 0 --prefill 60 --decode 4
"""

import argparse
import time

import torch

import ttnn
from models.autoports.zai_org_glm_5_3_flash.tests.hf_ref.client import hf_layer_reference
from models.autoports.zai_org_glm_5_3_flash.tt.ckpt import CheckpointReader
from models.autoports.zai_org_glm_5_3_flash.tt.functional_decoder import FunctionalDecoder


def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm() + 1e-30))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--prefill", type=int, default=60)
    ap.add_argument("--decode", type=int, default=4)
    ap.add_argument("--bucket", type=int, default=128)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--slot", type=int, default=0)
    args = ap.parse_args()
    rd = CheckpointReader()
    tc = rd.text_config
    torch.manual_seed(0)
    S = args.prefill + args.decode
    ids = torch.randint(0, 150000, (1, S))
    ref = hf_layer_reference(
        {
            "layer": args.layer,
            "weights": "real",
            "tokens": ids,
            "segments": [args.prefill] + [1] * args.decode,
            "capture": True,
        }
    )
    x, y = ref["x"].float(), ref["y"].float()  # [1,S,4,H]
    H4 = x.shape[-1] * x.shape[-2]

    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=24576, trace_region_size=0)
    try:
        t0 = time.time()
        bs, mbu = 64, 8  # block size, blocks per user (512 tokens per slot in this smoke)
        dec = FunctionalDecoder.from_state_dict(
            rd.layer_state(args.layer),
            hf_config=tc,
            layer_idx=args.layer,
            mesh_device=mesh,
            prefill_lengths=(args.bucket,),
            max_batch=args.batch,
            dsa_kwargs={"num_blocks": args.batch * mbu, "block_size": bs},
        )
        perm = torch.randperm(args.batch * mbu, generator=torch.Generator().manual_seed(1))
        pt_host = perm.reshape(args.batch, mbu).to(torch.int32)
        page_table = ttnn.from_torch(pt_host, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh)
        user_pt = ttnn.from_torch(
            pt_host[args.slot : args.slot + 1], dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh
        )
        dsa = dec.attn_kind == "dsa"
        print(f"build {time.time() - t0:.1f} s", flush=True)
        n = args.bucket
        xp = torch.zeros(1, 1, n, H4)
        xp[0, 0, : args.prefill] = x[0, : args.prefill].reshape(args.prefill, H4)
        X = ttnn.from_torch(xp, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh)
        if dsa:
            dec.allocate_decode_state(args.batch)
            Y, st = dec.prefill_forward(X, length=args.prefill, page_table=user_pt)
        else:
            Y, st = dec.prefill_forward(X, length=args.prefill)
        yt = ttnn.to_torch(Y)[0, 0, : args.prefill]
        xr = x[0, : args.prefill].reshape(args.prefill, H4)
        yr = y[0, : args.prefill].reshape(args.prefill, H4)
        print(f"prefill PCC {pcc(yt, yr):.6f} delta PCC {pcc(yt - xr, yr - xr):.6f}", flush=True)
        if not dsa:
            dec.allocate_decode_state(args.batch)
        dec.commit_prefill(args.slot, st)
        for i in range(args.decode):
            xd = torch.zeros(1, 1, args.batch, H4)
            xd[0, 0, args.slot] = x[0, args.prefill + i].reshape(H4)
            Xd = ttnn.from_torch(xd, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh)
            if dsa:
                pos = torch.full((args.batch,), -1, dtype=torch.int32)
                pos[args.slot] = args.prefill + i
                cur = ttnn.from_torch(pos, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh)
                Yd = dec.decode_forward(Xd, cur_pos=cur, page_table=page_table)
            else:
                Yd = dec.decode_forward(Xd)
            yd = ttnn.to_torch(Yd)[0, 0, args.slot]
            xr = x[0, args.prefill + i].reshape(H4)
            yr = y[0, args.prefill + i].reshape(H4)
            print(f"decode {i} PCC {pcc(yd, yr):.6f} delta PCC {pcc(yd - xr, yr - xr):.6f}", flush=True)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
