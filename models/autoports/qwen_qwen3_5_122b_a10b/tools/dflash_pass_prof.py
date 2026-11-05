# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Where the verify pass's time goes: the rows-mode layer forward (tt/mtp_rows.py) timed per layer kind and sub-block
(mixer / MoE) with a device sync around each piece, at several row counts, after a warm-up of each program. Host walls
with syncs (dispatch included): compare widths / pieces, not absolute device time.

    python -m models.autoports.qwen_qwen3_5_122b_a10b.tools.dflash_pass_prof --widths 1,4,8,16 [--reps 3]
"""
from __future__ import annotations

import argparse
import time
from collections import defaultdict

import torch

import ttnn
from models.autoports.qwen_qwen3_5_122b_a10b.tt.generator import build_generator
from models.autoports.qwen_qwen3_5_122b_a10b.tt.mtp_rows import layer_forward_rows, rows_pos_masks_host, rows_width

MODEL_DIR = "models/autoports/qwen_qwen3_5_122b_a10b"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--widths", default="1,4,8,16")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--traced", type=int, default=1)
    args = ap.parse_args()
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 4), l1_small_size=24576, trace_region_size=256 << 20)
    gen = build_generator(MODEL_DIR, mesh, max_seq_len=4096, warm_decode_trace=False, warm_prefill_trace=False)
    m = gen.model
    ids = [1000 + (i % 977) for i in range(300)]
    gen.prefill_forward(torch.tensor(ids).reshape(1, -1), page_table=gen.page_table[:1], prompt_lens=[len(ids)])
    sync = lambda: ttnn.synchronize_device(mesh)
    acc = defaultdict(float)

    # sub-block timers: wrap the MoE forward and the mixers
    restore = []

    def timed(obj, name, key):
        orig = getattr(obj, name)
        restore.append((obj, name, orig))

        def f(*a, **k):
            sync()
            t0 = time.perf_counter()
            out = orig(*a, **k)
            sync()
            acc[key()] += time.perf_counter() - t0
            return out

        setattr(obj, name, f)

    cur = {"kind": None}
    for layer in m.layers:
        kind = "attn" if layer.is_full_attention else "gdn"
        timed(layer.feed_forward, "forward", lambda kind=kind: f"{kind}.moe")
        if not layer.is_full_attention:
            timed(layer.attention, "forward_prefill", lambda: "gdn.mixer(prefill form)")
    for w in [int(x) for x in args.widths.split(",")]:
        BW = rows_width(w) if w > 1 else 1
        P = len(ids)
        pos = [P + i for i in range(BW)]
        tokens_tt, cur_pos_tt, rot_idxs, pt_tt = m.prepare_decode_inputs_host(
            torch.tensor([5] * BW, dtype=torch.int32), torch.tensor(pos, dtype=torch.int32),
            gen.page_table[:1].repeat(BW, 1),
        )
        dev = lambda t: ttnn.to_device(t, mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        tokens_tt, cur_pos_tt, rot_idxs, pt_tt = (dev(t) for t in (tokens_tt, cur_pos_tt, rot_idxs, pt_tt))
        cos, sin = m._rope_from_idxs(rot_idxs)
        masks = dev(rows_pos_masks_host(pos, BW))
        for rep in range(args.reps + 1):
            acc.clear()
            per_kind = defaultdict(float)
            x = m.embd(m._decode_tokens_view(tokens_tt, BW))
            x = ttnn.to_memory_config(ttnn.reshape(x, (1, 1, BW, x.shape[-1])), ttnn.DRAM_MEMORY_CONFIG)
            saved = []
            gi = 0
            sync()
            t_all = time.perf_counter()
            for layer in m.layers:
                t0 = time.perf_counter()
                y = layer_forward_rows(layer, x, w, cos=cos, sin=sin, position_tensor=cur_pos_tt, page_table=pt_tt,
                                       saved=saved, rows_pos_masks=masks, gdn_index=gi)
                if not layer.is_full_attention:
                    gi += 1
                sync()
                per_kind["attn" if layer.is_full_attention else "gdn"] += time.perf_counter() - t0
                ttnn.deallocate(x)
                x = y
            total = time.perf_counter() - t_all
            ttnn.deallocate(x)
            for sink in saved:
                for k, t in sink.items():
                    if k != "meta":
                        ttnn.deallocate(t)
            if rep == 0:
                continue  # warm-up (compiles)
            print(
                f"PASS_PROF rows {w} (BW {BW}) rep {rep}: layers total {total * 1e3:.1f} ms | "
                + " ".join(f"{k} {v * 1e3:.1f}" for k, v in sorted(per_kind.items()))
                + " | " + " ".join(f"{k} {v * 1e3:.1f}" for k, v in sorted(acc.items())),
                flush=True,
            )
        # ---- traced: one GDN layer / one attention layer (whole rows-mode layer), the GDN mixer alone, the MoE alone
        if args.traced:
            for obj, name, orig in restore:
                setattr(obj, name, orig)
            restore.clear()
            from models.autoports.qwen_qwen3_5_122b_a10b.tt.mtp_rows import GDN_CHUNK
            from models.autoports.qwen_qwen3_5_122b_a10b.tt.prefill_len import PrefillLen
            from models.tt_transformers.tt.common import Mode

            plen = PrefillLen(mesh, GDN_CHUNK, next(l.attention.K for l in m.layers if not l.is_full_attention))
            plen.stage(w)

            gl = next(l for l in m.layers if not l.is_full_attention)
            al = next(l for l in m.layers if l.is_full_attention)
            x0 = m.embd(m._decode_tokens_view(tokens_tt, BW))
            x0 = ttnn.to_memory_config(ttnn.reshape(x0, (1, 1, BW, x0.shape[-1])), ttnn.DRAM_MEMORY_CONFIG)

            def gdn_layer():
                y = layer_forward_rows(gl, ttnn.clone(x0), plen, cos=cos, sin=sin, position_tensor=cur_pos_tt,
                                       page_table=pt_tt, saved=None, rows_pos_masks=masks, gdn_index=0)
                ttnn.deallocate(y)

            def attn_layer():
                y = layer_forward_rows(al, ttnn.clone(x0), plen, cos=cos, sin=sin, position_tensor=cur_pos_tt,
                                       page_table=pt_tt, saved=None, rows_pos_masks=masks, gdn_index=0)
                ttnn.deallocate(y)

            def moe_only():
                ff_in = gl._decode_norm(gl.ffn_norm, ttnn.clone(x0), Mode.DECODE, gl.args.get_norm_config("attn", Mode.DECODE))
                gl.feed_forward.mtp_rows_mode = True
                try:
                    y = gl.feed_forward.forward(ff_in, mode="decode")
                finally:
                    gl.feed_forward.mtp_rows_mode = False
                ttnn.deallocate(y)

            for name, fn in (("gdn_layer", gdn_layer), ("attn_layer", attn_layer), ("moe", moe_only)):
                fn()
                fn()
                sync()
                tid = ttnn.begin_trace_capture(mesh, cq_id=0)
                fn()
                ttnn.end_trace_capture(mesh, tid, cq_id=0)
                sync()
                t0 = time.perf_counter()
                for _ in range(10):
                    ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
                sync()
                print(f"TRACED rows {w}: {name} {(time.perf_counter() - t0) / 10 * 1e6:.0f} us", flush=True)
                ttnn.release_trace(mesh, tid)
            ttnn.deallocate(x0)
    gen.teardown()
    ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
