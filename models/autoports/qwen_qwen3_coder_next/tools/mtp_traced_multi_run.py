# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Traced multi-user MTP step (tt/mtp_traced_multi.py) vs the eager multi-user loop on B users: token equality, passes,
per-program walls. Reduced model (--layers 0,3) for the plumbing, full model for the numbers."""
import argparse
import time

from loguru import logger

import ttnn
from models.autoports.qwen_qwen3_coder_next.tools.mtp_acceptance import PROMPTS
from models.autoports.qwen_qwen3_coder_next.tt.generator import build_generator
from models.autoports.qwen_qwen3_coder_next.tt.mtp import Qwen35MtpHead
from models.autoports.qwen_qwen3_coder_next.tt.mtp_multi import mtp_generate_multi, prefill_multi_with_head
from models.autoports.qwen_qwen3_coder_next.tt.mtp_traced_multi import MtpTracedMulti
from models.autoports.qwen_qwen3_coder_next.tt.mtp_verify import (
    free_gdn_snapshot,
    restore_gdn_state,
    snapshot_gdn_state,
)
from models.autoports.qwen_qwen3_coder_next.tt.weights import resolve_checkpoint_dir

MODEL_DIR = "models/autoports/qwen_qwen3_coder_next"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", type=int, default=2)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--new", type=int, default=32)
    ap.add_argument("--layers", default=None)
    ap.add_argument("--prompts", default="prose,code,json,chat")
    args = ap.parse_args()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(resolve_checkpoint_dir()))
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(1, 4), l1_small_size=24576, trace_region_size=512 * 1024 * 1024
    )
    layer_indices = [int(s) for s in args.layers.split(",")] if args.layers else None
    B = args.users
    gen = build_generator(
        MODEL_DIR,
        mesh,
        max_batch_size=B,
        max_seq_len=4096,
        warm_decode_trace=False,
        warm_prefill_trace=False,
        layer_indices=layer_indices,
    )
    model = gen.model
    head = Qwen35MtpHead(model, gen.args.load_state_dict())
    kv_shape = next(tuple(l.attention.paged_k.shape) for l in model.layers if l.is_full_attention)
    head.allocate_kv_cache(kv_shape)
    names = args.prompts.split(",")[:B]
    prompts = [tok(PROMPTS[n], add_special_tokens=False)["input_ids"] for n in names]

    # eager reference loop (real head drafts) from a fresh prefill
    tok0, hids, P = prefill_multi_with_head(gen, head, prompts)
    snap = snapshot_gdn_state(model)
    t0 = time.time()
    eager, passes_e = mtp_generate_multi(gen, head, prompts, args.new, args.k, prefilled=(tok0, hids, P))
    t_eager = time.time() - t0
    logger.info(f"eager multi loop: {passes_e} passes for {B} x {args.new} tokens in {t_eager:.1f}s")

    # traced step: slots from the same prefill, warm (compiles, scrambles the state) -> restore -> capture -> restore
    runner = MtpTracedMulti(gen, head, args.k, B)
    for u in range(B):
        runner.set_slot(u, tok0[u], P[u], gen.page_table[u], hids[u])
    restore_gdn_state(model, snap)
    runner.warm()
    restore_gdn_state(model, snap)
    runner.capture()
    restore_gdn_state(model, snap)
    for u in range(B):
        runner.set_slot(u, tok0[u], P[u], gen.page_table[u], hids[u])
    for v in runner.timing.values():
        v.clear()
    out = [[tok0[u]] for u in range(B)]
    passes = 0
    t0 = time.time()
    while min(len(o) for o in out) < args.new:
        emitted = runner.step()
        for u in range(B):
            out[u].extend(emitted[u])
        passes += 1
    t_traced = time.time() - t0
    tm = runner.timing_summary()
    per_pass = sum(n * ms for n, ms in tm.values()) / passes
    tot = B * args.new
    eq = sum(sum(int(a == b) for a, b in zip(eager[u], out[u][: args.new])) for u in range(B))
    print(
        f"MTP_TRACED_MULTI B={B} k={args.k} N={args.new}: passes {passes} (eager {passes_e}) -> {tot / passes:.2f} tokens per pass "
        f"over {B} users; traced tokens equal to eager {eq}/{tot}; loop wall {t_traced:.2f}s = {t_traced / passes * 1e3:.1f} ms per pass "
        f"-> {tot / t_traced:.1f} tok/s aggregate; program walls (count, ms): {tm}; summed per pass {per_pass:.1f} ms",
        flush=True,
    )
    for u in range(B):
        print(f"  user {u} ({names[u]}): {tok.decode(out[u][: args.new])!r}", flush=True)
    free_gdn_snapshot(snap)
    runner.release()
    ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
