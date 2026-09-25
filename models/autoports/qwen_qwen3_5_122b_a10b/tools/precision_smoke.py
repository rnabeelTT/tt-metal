# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""One-candidate device smoke of a precision policy on the reduced 2-layer model (layers 0 and 3, one of each kind):
build through the normal path (create_model applies the policy and asserts the built tensors match it), warm the traces,
prefill a 64-token prompt and run 8 greedy traced decode steps, then print the tokens, the memory report and the wall
times. A candidate that cannot be constructed or run fails here before it costs a 48-layer sweep run.

    QWEN35_PRECISION_CONFIG=doc/datatype_sweep/configs/<id>.json python -m ...tools.precision_smoke [--layers 0,3] [--steps 8]
"""
import argparse
import json
import os
import time
from pathlib import Path

import torch

MODEL_DIR = Path(__file__).resolve().parents[1]
L1_SMALL_SIZE = 24576


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="0,3")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--prompt-len", type=int, default=64)
    ap.add_argument("--trace-region", type=int, default=int(os.environ.get("QWEN35_TRACE_REGION", 512 * 1024 * 1024)))
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    os.environ.setdefault("HF_MODEL", "Qwen/Qwen3.5-122B-A10B")
    import ttnn
    from models.autoports.qwen_qwen3_5_122b_a10b.tt.generator import build_generator

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(1, 4),
        l1_small_size=L1_SMALL_SIZE,
        trace_region_size=a.trace_region,
        num_command_queues=1,
    )
    rec = {"config": os.environ.get("QWEN35_PRECISION_CONFIG", "none"), "layers": a.layers}
    try:
        t0 = time.perf_counter()
        gen = build_generator(
            str(MODEL_DIR), mesh, layer_indices=[int(x) for x in a.layers.split(",")], max_seq_len=4096
        )
        rec["build_s"] = round(time.perf_counter() - t0, 1)
        rec["config_id"] = gen.model.precision_policy.get("config_id")
        text = "The Tenstorrent Blackhole mesh runs a mixture-of-experts language model. " * 8
        ids = gen.tokenizer.encode(text)[: a.prompt_len]
        gen.reset()
        t1 = time.perf_counter()
        lg = gen.prefill_forward(
            torch.tensor(ids).reshape(1, -1), page_table=gen.page_table[:1], prompt_lens=[len(ids)]
        )
        rec["prefill_ms"] = round((time.perf_counter() - t1) * 1e3, 1)
        tok = int(torch.argmax(lg[0, -1]))
        toks = [tok]
        t2 = time.perf_counter()
        for i in range(a.steps):
            tok = int(gen.decode_forward(torch.tensor([[tok]]), torch.tensor([len(ids) + i]), enable_trace=True)[0])
            toks.append(tok)
        rec["decode_ms_per_step"] = round((time.perf_counter() - t2) * 1e3 / a.steps, 2)
        rec["tokens"] = toks
        rec["text"] = gen.tokenizer.decode(toks)
        rec["program_cache"] = gen.program_cache_entries()
        rec["memory"] = gen._memory_report()
        print("PRECISION_SMOKE " + json.dumps(rec, default=str))
        if a.out:
            a.out.parent.mkdir(parents=True, exist_ok=True)
            a.out.write_text(json.dumps(rec, indent=2, default=str) + "\n")
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()
