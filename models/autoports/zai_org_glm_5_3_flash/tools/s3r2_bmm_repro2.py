# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage-3 re-review #2 P2-2a, step 2: isolate the wrong output of the DSA per-head reuse matmul config at 2048 prefill
rows (s3r2_bmm_repro.py: per_core_M 8 < Mt 64 -> W_UK non-finite, W_UV PCC 0.14 inside the decoder; per_core_M = Mt = 4
at 128 rows correct; tools/s3r_probe_dsa.py: the same config on fresh random tensors PCC 0.999999).

One real layer-3 decoder (opt_s5_pf); one 2048-row prefill chunk with the default matmuls, the W_UK / W_UV inputs cloned.
Then for each weight, the reuse config (per_core_M 8 / 4 / 2 / 1) on:
  A the decoder's device tensors (x clone, layer weight)
  B a fresh device copy of x (to_torch -> from_torch), layer weight
  C x clone, a fresh device copy of the weight (from_torch bfp8 of its values)
  D fresh random x of the same shape, layer weight
  E fresh x and fresh weight (the probe setting)
  F A again after clearing the program cache
and the default config on A, each against a torch fp32 product of the device operands.

usage: python -m models.autoports.zai_org_glm_5_3_flash.tools.s3r2_bmm_repro2"""

import types

import torch

import ttnn
from models.autoports.zai_org_glm_5_3_flash.tests.harness import H4, build_layer, open_mesh, pcc, prefill_user
from models.autoports.zai_org_glm_5_3_flash.tests.test_functional_decoder import synthetic_stream
from models.autoports.zai_org_glm_5_3_flash.tt.ckpt import CheckpointReader
from models.autoports.zai_org_glm_5_3_flash.tt.optimized_decoder import CANDIDATES, OptimizedDecoder

DRAM = ttnn.DRAM_MEMORY_CONFIG


def cfg(mesh, w, Mt, pm):
    grid = mesh.compute_with_storage_grid_size()
    Kt, Nt = int(w.shape[-2]) // 32, int(w.shape[-1]) // 32
    return ttnn.MatmulMultiCoreReuseProgramConfig(
        compute_with_storage_grid_size=(grid.x, grid.y),
        in0_block_w=min(Kt, 8),
        out_subblock_h=1,
        out_subblock_w=max(d for d in (4, 2, 1) if Nt % d == 0),
        per_core_M=pm,
        per_core_N=Nt,
    )


def main():
    rd = CheckpointReader()
    mesh = open_mesh()
    opt = CANDIDATES["opt_s5_pf"]

    class Dec(OptimizedDecoder):
        @classmethod
        def from_state_dict(cls, *a, **kw):
            kw["opt"] = opt
            return super().from_state_dict(*a, **kw)

    try:
        dec = build_layer(mesh, rd, 3, "real", batch=8, decoder_cls=Dec)
        attn = dec.attn
        attn.bmm_prefill_pm = 0
        orig = attn._bmm
        keep = {}

        def _bmm(self, x, w, rows):
            if rows > 32 and len(keep) < 2:
                keep["W_UK" if w is self.w_uk else "W_UV"] = (ttnn.clone(x, memory_config=DRAM), w)
            return orig(x, w, rows)

        attn._bmm = types.MethodType(_bmm, attn)
        x = synthetic_stream(3, 21, 2048).float().reshape(2048, H4)
        prefill_user(dec, mesh, x, 2048, 0)
        ckc = attn.ckc
        for name, (xd, wd) in keep.items():
            tx, tw = ttnn.to_torch(xd).float(), ttnn.to_torch(wd).float()
            ref = tx @ tw
            Mt = int(xd.padded_shape[-2]) // 32
            fresh_x = ttnn.from_torch(
                tx.bfloat16(), dtype=xd.dtype, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=DRAM
            )
            fresh_w = ttnn.from_torch(
                tw.bfloat16(), dtype=wd.dtype, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=DRAM
            )
            g = torch.Generator().manual_seed(3)
            rnd = torch.randn(tuple(tx.shape), generator=g).bfloat16()
            rnd_x = ttnn.from_torch(rnd, dtype=xd.dtype, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=DRAM)
            ref_rnd = rnd.float() @ tw
            print(f"{name}: x {tuple(xd.shape)} {xd.dtype} w {tuple(wd.shape)} {wd.dtype} Mt {Mt}", flush=True)
            o = ttnn.to_torch(ttnn.matmul(xd, wd, compute_kernel_config=ckc, memory_config=DRAM)).float()
            print(f"  default config, A: PCC {pcc(o, ref):.6f}", flush=True)
            for pm in (8, 4, 2, 1):  # (per_core_M = Mt = 64 hung the device in stage 3: not run)
                pc = cfg(mesh, wd, Mt, pm)
                for tag, a, b, r in (
                    ("A device tensors", xd, wd, ref),
                    ("B fresh x", fresh_x, wd, ref),
                    ("C fresh w", xd, fresh_w, ref),
                    ("D random x", rnd_x, wd, ref_rnd),
                    ("E fresh x + w", fresh_x, fresh_w, ref),
                ):
                    try:
                        o = ttnn.to_torch(
                            ttnn.matmul(a, b, compute_kernel_config=ckc, memory_config=DRAM, program_config=pc)
                        ).float()
                        bad = (o - r).abs().amax(dim=(-1, -2))[0]  # per head
                        nb = int((bad > 0.05 * r.abs().amax()).sum())
                        print(
                            f"  per_core_M {pm:2d} {tag:18}: PCC {pcc(o, r):.6f} finite {bool(torch.isfinite(o).all())} "
                            f"heads wrong {nb}/{o.shape[1]}",
                            flush=True,
                        )
                    except Exception as e:  # noqa: BLE001
                        print(f"  per_core_M {pm:2d} {tag:18}: ERROR {str(e).splitlines()[0][:200]}", flush=True)
            mesh.disable_and_clear_program_cache()
            mesh.enable_program_cache()
            for pm in (8, 1):
                o = ttnn.to_torch(
                    ttnn.matmul(
                        xd, wd, compute_kernel_config=ckc, memory_config=DRAM, program_config=cfg(mesh, wd, Mt, pm)
                    )
                ).float()
                print(f"  per_core_M {pm:2d} F cleared cache   : PCC {pcc(o, ref):.6f}", flush=True)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
