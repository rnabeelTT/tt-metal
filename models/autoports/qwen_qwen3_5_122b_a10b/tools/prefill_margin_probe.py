# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Classify a step-0 greedy fork: run the token-out bench prompt's prefill N times (fresh reset each) on the 48-layer model
under the active precision policy and print, per run, the top-2 next tokens and their logit margin. A margin of a few
hundredths of a logit with the argmax flipping between runs = a near tie under the model's run-to-run LSB non-determinism;
a large margin with a flip = a real path difference.

    QWEN35_PRECISION_CONFIG=... python -m ...tools.prefill_margin_probe [--repeats 3] [--out file.json]
"""
import argparse
import json
import os
from pathlib import Path

import torch

from models.autoports.qwen_qwen3_5_122b_a10b.tools.readiness import _open_mesh
from models.autoports.qwen_qwen3_5_122b_a10b.tools.tokenout_bench import PROMPT

MODEL_DIR = Path(__file__).resolve().parents[1]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument(
        "--decode-steps", type=int, default=2, help="also record the first decode steps' top-2 margins (blocking)"
    )
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    os.environ.setdefault("HF_MODEL", "Qwen/Qwen3.5-122B-A10B")
    import ttnn
    from models.autoports.qwen_qwen3_5_122b_a10b.tt.generator import build_generator

    mesh = _open_mesh(int(os.environ.get("QWEN35_TRACE_REGION", 512 * 1024 * 1024)))
    rec = {"policy": os.environ.get("QWEN35_PRECISION_CONFIG", "shipped default"), "prompt": PROMPT, "runs": []}
    try:
        gen = build_generator(str(MODEL_DIR), mesh, max_seq_len=4096, max_batch_size=1, warm_host_trace=True)
        rec["config_id"] = gen.model.precision_policy.get("config_id")
        ids = gen.tokenizer.encode(PROMPT)
        toks = torch.tensor(ids).reshape(1, -1)
        for r in range(a.repeats):
            gen.reset()
            lg = gen.prefill_forward(toks, page_table=gen.page_table[:1], prompt_lens=[len(ids)])[0, -1].float()
            top = torch.topk(lg, 2)
            run = {
                "prefill_top2": [int(top.indices[0]), int(top.indices[1])],
                "prefill_top2_text": [
                    gen.tokenizer.decode([int(top.indices[0])]),
                    gen.tokenizer.decode([int(top.indices[1])]),
                ],
                "prefill_margin": float(top.values[0] - top.values[1]),
                "decode": [],
            }
            tok = int(top.indices[0])
            for i in range(a.decode_steps):  # host-logits compatibility mode (untraced) for the decode margins
                lgd = gen.decode_forward(
                    torch.tensor([[tok]]), torch.tensor([len(ids) + i]), enable_trace=True, on_device_sampling=False
                )
                lgd = torch.as_tensor(lgd).float().reshape(-1)
                t2 = torch.topk(lgd, 2)
                run["decode"].append(
                    {"top2": [int(t2.indices[0]), int(t2.indices[1])], "margin": float(t2.values[0] - t2.values[1])}
                )
                tok = int(t2.indices[0])
            rec["runs"].append(run)
            print("MARGIN_PROBE " + json.dumps(run))
        print(
            "MARGIN_PROBE_DONE "
            + json.dumps(
                {
                    "config_id": rec["config_id"],
                    "prefill_argmax": [x["prefill_top2"][0] for x in rec["runs"]],
                    "prefill_margins": [round(x["prefill_margin"], 4) for x in rec["runs"]],
                }
            )
        )
        if a.out:
            a.out.parent.mkdir(parents=True, exist_ok=True)
            a.out.write_text(json.dumps(rec, indent=2) + "\n")
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()
