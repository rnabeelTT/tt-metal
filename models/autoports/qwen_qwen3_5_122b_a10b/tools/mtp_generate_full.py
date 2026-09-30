# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Full-model check of the v1 MTP loop: plain greedy vs MTP (draft k, verify pass, exact commit) on one prompt; the
tokens must agree (bf16 near-ties aside) and the passes-per-token ratio is reported.

    python -m models.autoports.qwen_qwen3_5_122b_a10b.tools.mtp_generate_full --prompt prose --k 4 --new 48
"""
from __future__ import annotations

import argparse
import os
import time

from loguru import logger

import ttnn
from models.autoports.qwen_qwen3_5_122b_a10b.tools.mtp_acceptance import PROMPTS
from models.autoports.qwen_qwen3_5_122b_a10b.tt.generator import build_generator
from models.autoports.qwen_qwen3_5_122b_a10b.tt.mtp import Qwen35MtpHead
from models.autoports.qwen_qwen3_5_122b_a10b.tt.mtp_verify import mtp_generate_eager
from models.autoports.qwen_qwen3_5_122b_a10b.tt.weights import resolve_checkpoint_dir

MODEL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="prose")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--new", type=int, default=48)
    ap.add_argument("--layers", default=None, help="comma-separated checkpoint layer indices for a reduced model")
    ap.add_argument("--gdn-commit", action="store_true", help="GDN-only exact commit instead of re-running the pass")
    args = ap.parse_args()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(resolve_checkpoint_dir()))
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(1, 4), l1_small_size=24576, trace_region_size=256 * 1024 * 1024
    )
    layer_indices = [int(s) for s in args.layers.split(",")] if args.layers else None
    gen = build_generator(MODEL_DIR, mesh, max_seq_len=4096, warm_decode_trace=False, layer_indices=layer_indices)
    model = gen.model
    head = Qwen35MtpHead(model, gen.args.load_state_dict())
    kv_shape = next(tuple(l.attention.paged_k.shape) for l in model.layers if l.is_full_attention)
    ids = tok(PROMPTS[args.prompt], add_special_tokens=False)["input_ids"]
    t0 = time.time()
    greedy = gen.generate(ids, args.new, enable_trace=False, on_device_sampling=False)
    t_greedy = time.time() - t0
    logger.info(f"greedy {args.new} tokens in {t_greedy:.1f}s (eager): {tok.decode(greedy)!r}")
    head.allocate_kv_cache(kv_shape)
    t0 = time.time()
    mtp, passes = mtp_generate_eager(gen, head, ids, args.new, args.k, log=logger.info, gdn_commit=args.gdn_commit)
    t_mtp = time.time() - t0
    agree = sum(int(a == b) for a, b in zip(greedy, mtp))
    first_diff = next((i for i, (a, b) in enumerate(zip(greedy, mtp)) if a != b), None)
    print(
        f"MTP_GENERATE_FULL {args.prompt}: k={args.k} N={args.new} gdn_commit={args.gdn_commit} passes {passes} -> {args.new / passes:.2f} tokens per pass; "
        f"tokens equal to greedy {agree}/{args.new} (first difference at {first_diff}); eager walls greedy {t_greedy:.1f}s mtp {t_mtp:.1f}s",
        flush=True,
    )
    print(f"  greedy: {tok.decode(greedy)!r}\n  mtp   : {tok.decode(mtp)!r}", flush=True)
    gen.teardown()
    ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
