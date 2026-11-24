# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Day-0 device driver: chunked prefill + greedy decode of one chat prompt (eager), printing the text and timings.

    python -m models.autoports.tencent_hy4_flash.tools.generate --prompt "..." --max-new 64 [--layers 0,1,2] [--ctx 32768]

--teacher FILE.pt: teacher-forced mode instead; FILE holds {"ids": [T] tensor} (e.g. from ref_hy4) and the logits of
every position are saved to --out for compare_hy4.
"""

from __future__ import annotations

import argparse
import os
import time

import torch
import ttnn
from loguru import logger


def open_mesh():
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    return ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=int(os.environ.get("HY4_TRACE_REGION", "0")))


def build(mesh, args):
    from models.autoports.tencent_hy4_flash.tt.config import Hy4Config, checkpoint_dir
    from models.autoports.tencent_hy4_flash.tt.model import Hy4Model

    cfg = Hy4Config.load()
    layers = [int(x) for x in args.layers.split(",")] if args.layers else None
    engram = None
    if not args.no_engram:
        from models.experimental.hybrid_moe.engram_store import EngramRows

        engram = EngramRows(os.environ.get("HY4_ENGRAM_DIR", os.path.expanduser("~/.cache/hybrid-moe/hy4_engram")),
                             checkpoint_dir())
    cache_dir = os.environ.get("HY4_TENSOR_CACHE", os.path.expanduser("~/.cache/hybrid-moe/hy4_tt"))
    os.makedirs(cache_dir, exist_ok=True)
    return Hy4Model(mesh, cfg, max_ctx=args.ctx, chunk=args.chunk, layers=layers, engram=engram, cache_dir=cache_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="Write a haiku about the ocean.")
    ap.add_argument("--raw", action="store_true", help="no chat template")
    ap.add_argument("--think", action="store_true", help="reasoning_effort high (default no_think)")
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--layers", default="")
    ap.add_argument("--no-engram", action="store_true")
    ap.add_argument("--teacher", default="")
    ap.add_argument("--eager", action="store_true")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    from transformers import AutoTokenizer

    from models.autoports.tencent_hy4_flash.tt.config import checkpoint_dir

    tok_ = AutoTokenizer.from_pretrained(checkpoint_dir())
    mesh = open_mesh()
    try:
        t0 = time.time()
        model = build(mesh, args)
        logger.info(f"build {time.time() - t0:.0f} s")
        if args.teacher:
            ids = torch.load(args.teacher)["ids"].reshape(-1)
            outs = []
            t0 = time.time()
            for c0 in range(0, ids.numel(), args.chunk):
                outs.append(model.forward(ids[c0 : c0 + args.chunk], logits_rows="all"))
            lg = torch.cat(outs)
            logger.info(f"teacher forced {ids.numel()} tokens in {time.time() - t0:.1f} s")
            torch.save({"ids": ids, "logits": lg.to(torch.float32)}, args.out or "tt_logits.pt")
            return
        if args.raw:
            ids = torch.tensor(tok_(args.prompt)["input_ids"])
        else:
            text = tok_.apply_chat_template([{"role": "user", "content": args.prompt}], tokenize=False,
                                           add_generation_prompt=True,
                                           reasoning_effort="high" if args.think else "no_think")
            ids = torch.tensor(tok_(text, add_special_tokens=False)["input_ids"])
        logger.info(f"prompt {ids.numel()} tokens")
        from models.autoports.tencent_hy4_flash.tt.engine import Hy4Engine

        eng = Hy4Engine(model, 0, trace=not args.eager)
        t0 = time.time()
        tok = eng.prefill(ids.tolist())
        t_pf = time.time() - t0
        out = [tok]
        ts = []
        while len(out) < args.max_new and out[-1] != model.cfg.eos:
            t1 = time.time()
            out += eng.step()
            ts.append(time.time() - t1)
        print("=" * 80)
        print(tok_.decode(out))
        print("=" * 80)
        warm = ts[2:] or ts
        print(f"prefill {ids.numel()} tok in {t_pf:.2f} s; decode {len(out)} tok, warm step "
              f"{1e3 * sum(warm) / max(1, len(warm)):.1f} ms ({len(warm) / max(sum(warm), 1e-9):.2f} tok/s)")
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
