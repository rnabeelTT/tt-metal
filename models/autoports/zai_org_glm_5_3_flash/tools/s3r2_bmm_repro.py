# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage-3 re-review #2 P2-2a: why the DSA per-head reuse matmul config (GlmDSA._bmm, MatmulMultiCoreReuseProgramConfig,
full grid, per_core_M <= 8) gives wrong DSA output on prefill continuation chunks.

Real layer-3 decoder (policy opt_s5_pf), one die. GlmDSA._bmm is wrapped: every prefill call (rows > 32) records the
input / weight tensor specs and computes, on the SAME inputs, (a) the default ttnn.matmul and (b) the reuse config
(per_core_M = largest divisor of the padded M tiles <= 8, as tt/dsa.py), and a torch fp32 product of the device
operands, then returns (a) or (b) per the run. Two prefills of the same prompt (lengths 4000 and 2049: a 2048-row chunk
then a continuation chunk) run with the wrapper returning (a), then (b); the layer outputs are compared per chunk.
After each run the reuse output is recomputed once more after clearing the program cache (a program-cache collision
check).

usage: python -m models.autoports.zai_org_glm_5_3_flash.tools.s3r2_bmm_repro"""

import types

import torch

import ttnn
from models.autoports.zai_org_glm_5_3_flash.tests.harness import H4, build_layer, open_mesh, pcc, prefill_user
from models.autoports.zai_org_glm_5_3_flash.tests.test_functional_decoder import synthetic_stream
from models.autoports.zai_org_glm_5_3_flash.tt.ckpt import CheckpointReader
from models.autoports.zai_org_glm_5_3_flash.tt.optimized_decoder import CANDIDATES, OptimizedDecoder

DRAM = ttnn.DRAM_MEMORY_CONFIG


def spec(t):
    return (
        f"shape {tuple(t.shape)} padded {tuple(t.padded_shape)} {t.layout} {t.dtype} "
        f"{'sharded' if t.is_sharded() else 'interleaved'} {t.memory_config().buffer_type}"
    )


def reuse_cfg(attn, x, w):
    grid = attn.mesh.compute_with_storage_grid_size()
    Kt, Nt, Mt = int(w.shape[-2]) // 32, int(w.shape[-1]) // 32, int(x.padded_shape[-2]) // 32
    pm = max(d for d in range(1, 9) if Mt % d == 0)
    return (
        ttnn.MatmulMultiCoreReuseProgramConfig(
            compute_with_storage_grid_size=(grid.x, grid.y),
            in0_block_w=min(Kt, 8),
            out_subblock_h=1,
            out_subblock_w=max(d for d in (4, 2, 1) if Nt % d == 0),
            per_core_M=pm,
            per_core_N=Nt,
        ),
        Mt,
        pm,
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
        attn.bmm_prefill_pm = 0  # the wrapper below decides
        orig = attn._bmm
        state = {"mode": "default", "log": []}

        def _bmm(self, x, w, rows):
            if rows <= 32:
                return orig(x, w, rows)
            name = "W_UK" if w is self.w_uk else "W_UV"
            pc, Mt, pm = reuse_cfg(self, x, w)
            a = ttnn.matmul(x, w, compute_kernel_config=self.ckc, memory_config=DRAM)
            b = ttnn.matmul(x, w, compute_kernel_config=self.ckc, memory_config=DRAM, program_config=pc)
            ta, tb = ttnn.to_torch(a).float(), ttnn.to_torch(b).float()
            tx, tw = ttnn.to_torch(x).float(), ttnn.to_torch(w).float()
            ref = tx @ tw
            r = int(x.shape[-2])
            fin = bool(torch.isfinite(tb).all())
            # per-head agreement: which heads / rows differ
            d = (ta - tb).abs().amax(dim=-1)  # [1, H, rows]
            bad_heads = (d.amax(dim=-1) > 1e-2).nonzero()[:, 1].tolist() if d.numel() else []
            bad_rows = (d.amax(dim=1) > 1e-2).nonzero()[:, 1].tolist() if d.numel() else []
            line = (
                f"[{state['mode']}] {name} rows {rows} | x {spec(x)} | w {spec(w)} | Mt {Mt} per_core_M {pm} | "
                f"out default {spec(a)} | out reuse {spec(b)} | default vs torch PCC {pcc(ta[..., :r, :], ref[..., :r, :]):.6f} "
                f"| reuse vs torch PCC {pcc(tb[..., :r, :], ref[..., :r, :]):.6f} | reuse finite {fin} | max|a-b| "
                f"{float((ta - tb).abs().max()):.3e} | heads differing {len(bad_heads)} {bad_heads[:8]} | rows differing "
                f"{len(bad_rows)} {bad_rows[:4]}..{bad_rows[-4:] if bad_rows else ''}"
            )
            print(line, flush=True)
            state["log"] = (state["log"] + [(name, rows, tx, w, pc, x.dtype)])[-2:]
            if state["mode"] == "default":
                ttnn.deallocate(b)
                return a
            ttnn.deallocate(a)
            return b

        attn._bmm = types.MethodType(_bmm, attn)
        for length, slot, seed in ((4000, 0, 11), (2049, 1, 12)):
            x = synthetic_stream(3, seed, length).float().reshape(length, H4)
            outs = {}
            for mode in ("default", "reuse"):
                state["mode"] = mode
                print(f"=== prompt {length} slot {slot} mode {mode}", flush=True)
                outs[mode] = prefill_user(dec, mesh, x, length, slot)
            a, b = outs["default"], outs["reuse"]
            for c0, c1 in ((0, 2048), (2048, length)):
                pa, pb = a[c0:c1], b[c0:c1]
                xs = x[c0:c1]
                print(
                    f"prompt {length} rows {c0}..{c1}: layer output PCC reuse vs default {pcc(pb, pa):.6f}, update rms "
                    f"ratio {float((pb - xs).norm() / (pa - xs).norm()):.4f}, finite {bool(torch.isfinite(pb).all())}",
                    flush=True,
                )
        # program-cache collision check: recompute the last reuse products on a cleared cache
        mesh.disable_and_clear_program_cache()
        mesh.enable_program_cache()
        for name, rows, txx, ww, pc, xdt in state["log"]:
            try:
                xx = ttnn.from_torch(txx.bfloat16(), dtype=xdt, layout=ttnn.TILE_LAYOUT, device=mesh)
                b1 = ttnn.to_torch(
                    ttnn.matmul(xx, ww, compute_kernel_config=attn.ckc, memory_config=DRAM, program_config=pc)
                ).float()
                a1 = ttnn.to_torch(ttnn.matmul(xx, ww, compute_kernel_config=attn.ckc, memory_config=DRAM)).float()
                print(f"cleared cache: {name} rows {rows}: max|default-reuse| {float((a1 - b1).abs().max()):.3e}")
            except Exception as e:  # noqa: BLE001 - tensors may have been freed
                print(f"cleared cache: {name}: {str(e).splitlines()[0][:200]}")
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
