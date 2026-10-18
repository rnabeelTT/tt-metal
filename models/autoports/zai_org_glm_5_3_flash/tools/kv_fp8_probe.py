# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage 8 (datatype sweep) KV-cache smoke probe: can the DSA latent pool be stored in 1 byte per element?

The DSA latent pool (tt/dsa.py GlmDSA.latent) is a ROW_MAJOR bf16 [1, 1, NB*128, 512] tensor written by
ttnn.scatter_rows and read by ttnn.transformer.sparse_sdpa (kv_format BF16). BFP8 is a TILE-only format, so a bfp8 pool
does not fit this row-paged contract; sparse_sdpa accepts FP8_E4M3 rows (kv_format FP8_E4M3, tilized in-op to bfp8_b).
The probe checks, on one die with the GLM geometry (16 heads per die padded to 32, K = V = 512, top-k 2048 keys):
  1. the device bf16 -> fp8_e4m3 cast of a latent chunk (TILE and ROW_MAJOR inputs),
  2. scatter_rows of fp8 rows into an fp8 pool,
  3. sparse_sdpa over the fp8 pool vs the bf16 pool (PCC of the attention output, decode rows and a 128-row chunk).

usage: python -m models.autoports.zai_org_glm_5_3_flash.tools.kv_fp8_probe
"""

from __future__ import annotations

import json
import traceback

import torch

import ttnn


def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm() + 1e-30))


def main():
    out = {}
    dev = ttnn.open_device(device_id=0, l1_small_size=1 << 15)
    try:
        torch.manual_seed(0)
        T, R, H, TOPK = 8192, 512, 32, 2048
        # latent rows of the real model are kv_a_layernorm outputs: O(1) values; use a matching scale
        lat = torch.randn(T, R) * 0.8
        rep = lambda t, dt, lay: ttnn.from_torch(t, dtype=dt, layout=lay, device=dev)
        # ---- 1. device casts
        for name, lay in (("tile", ttnn.TILE_LAYOUT), ("row_major", ttnn.ROW_MAJOR_LAYOUT)):
            try:
                c = rep(lat[:128].reshape(1, 1, 128, R), ttnn.bfloat16, lay)
                c8 = ttnn.typecast(c, ttnn.fp8_e4m3)
                c8 = c8 if c8.layout == ttnn.ROW_MAJOR_LAYOUT else ttnn.to_layout(c8, ttnn.ROW_MAJOR_LAYOUT)
                back = ttnn.to_torch(ttnn.typecast(ttnn.to_layout(c8, ttnn.TILE_LAYOUT), ttnn.bfloat16)).float()
                ref = lat[:128].to(torch.float8_e4m3fn).float()
                out[f"cast_{name}"] = {
                    "ok": True,
                    "out_layout": str(c8.layout),
                    "max_abs_vs_torch_fp8": float((back.reshape(128, R) - ref).abs().max()),
                    "pcc_vs_bf16": pcc(back, lat[:128]),
                }
            except Exception as e:  # noqa: BLE001
                out[f"cast_{name}"] = {"ok": False, "error": str(e).splitlines()[0][:400]}
        print("KVFP8 casts " + json.dumps(out), flush=True)

        # ---- 2. scatter_rows into an fp8 pool
        pool16 = rep(lat.reshape(1, 1, T, R), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
        try:
            pool8 = rep(torch.zeros(1, 1, T, R), ttnn.fp8_e4m3, ttnn.ROW_MAJOR_LAYOUT)
            rows = torch.arange(T, dtype=torch.int32)
            for r0 in range(0, T, 2048):
                src = rep(lat[r0 : r0 + 2048], ttnn.fp8_e4m3, ttnn.ROW_MAJOR_LAYOUT)
                idx = ttnn.from_torch(
                    rows[r0 : r0 + 2048].reshape(1, -1), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev
                )
                ttnn.scatter_rows(pool8, src, idx)
            out["scatter_rows_fp8"] = {"ok": True}
        except Exception as e:  # noqa: BLE001
            out["scatter_rows_fp8"] = {"ok": False, "error": str(e).splitlines()[0][:400]}
            pool8 = rep(lat.reshape(1, 1, T, R), ttnn.fp8_e4m3, ttnn.ROW_MAJOR_LAYOUT)
        print("KVFP8 scatter " + json.dumps(out["scatter_rows_fp8"]), flush=True)

        # ---- 3. sparse_sdpa: bf16 pool vs fp8 pool
        for S in (32, 128):
            q = torch.randn(1, H, S, R) * 0.5
            q[:, 16:] = 0.0  # 16 real heads per die, zero-padded to 32
            ids = torch.stack([torch.randperm(T)[:TOPK].sort().values for _ in range(S)]).to(torch.int32)
            idx = ttnn.from_torch(
                ids.reshape(1, 1, S, TOPK), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev
            )
            qd = rep(q, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
            res = {}
            for name, kv, fmt in (
                ("bf16", pool16, ttnn.transformer.SparseKVFormat.BF16),
                ("fp8", pool8, ttnn.transformer.SparseKVFormat.FP8_E4M3),
            ):
                try:
                    o = ttnn.transformer.sparse_sdpa(qd, kv, idx, R, kv_format=fmt, scale=R**-0.5, k_chunk_size=32)
                    res[name] = ttnn.to_torch(o).float()[:, :16]
                except Exception as e:  # noqa: BLE001
                    res[name] = None
                    out[f"sdpa_{name}_S{S}"] = {"ok": False, "error": str(e).splitlines()[0][:400]}
            # torch reference on bf16 latent
            kv_sel = lat[ids.long()]  # [S, TOPK, R]
            sc = torch.einsum("hsr,skr->hsk", q[0, :16], kv_sel) * R**-0.5
            ref = torch.einsum("hsk,skr->hsr", sc.softmax(-1), kv_sel)
            rec = {}
            for name in ("bf16", "fp8"):
                if res[name] is not None:
                    rec[f"pcc_{name}_vs_torch"] = pcc(res[name][0], ref)
                    rec[f"rms_{name}_over_ref"] = float(res[name][0].norm() / ref.norm())
            if res["bf16"] is not None and res["fp8"] is not None:
                rec["pcc_fp8_vs_bf16"] = pcc(res["fp8"], res["bf16"])
            out[f"sdpa_S{S}"] = rec
            print(f"KVFP8 sdpa S={S} " + json.dumps(rec), flush=True)
    except Exception:  # noqa: BLE001
        out["fatal"] = traceback.format_exc()[-2000:]
    finally:
        print("KVFP8_RESULT " + json.dumps(out), flush=True)
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
