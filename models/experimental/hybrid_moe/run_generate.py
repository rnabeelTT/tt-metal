# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Hybrid-MoE prototype driver: build the Qwen3.5 autoport with the hybrid overlay and generate greedily (eager).

Env: HF_MODEL (122B or 397B snapshot id), HYB_SLOTS, HYB_PLAN, POC_PROMPTS (count), POC_NEW_TOKENS, POC_LAYERS
(comma list -> reduced model), POC_MAX_SEQ (default 2048), HYB_RECORD=1 (save decode routing as routing.npz).
Writes <out_dir>/result.json.
"""

import traceback
import json
import os
import sys
import time

sys.path.insert(0, "/poc")
import hybrid  # noqa: E402

HYB = hybrid.install()

import numpy as np  # noqa: E402
import torch  # noqa: E402

import ttnn  # noqa: E402
from capture_routing import PROMPTS  # noqa: E402
from models.autoports.qwen_qwen3_5_122b_a10b.tt.generator import build_generator  # noqa: E402

N_PROMPTS = int(os.environ.get("POC_PROMPTS", "2"))
NEW = int(os.environ.get("POC_NEW_TOKENS", "64"))


def main():
    out_dir = sys.argv[1]
    os.makedirs(out_dir, exist_ok=True)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), l1_small_size=24576, trace_region_size=64_000_000)
    res = {"env": {k: v for k, v in os.environ.items() if k.startswith(("HYB_", "POC_", "HF_MODEL", "QWEN35_"))}}
    try:
        t0 = time.time()
        kw = {}
        if os.environ.get("POC_LAYERS"):
            kw["layer_indices"] = [int(i) for i in os.environ["POC_LAYERS"].split(",")]
        gen = build_generator(
            "models/autoports/qwen_qwen3_5_122b_a10b",
            mesh,
            max_seq_len=int(os.environ.get("POC_MAX_SEQ", "2048")),
            warm_decode_trace=os.environ.get("POC_TRACE") == "1",
            warm_prefill_trace=False,
            **kw,
        )
        res["build_s"] = round(time.time() - t0, 1)
        res["E"], res["C"] = HYB.E, HYB.C
        print(f"BUILD {res['build_s']}s E={HYB.E} C={HYB.C}", flush=True)
        res["memory"] = gen._memory_report()
        print("MEMORY " + json.dumps(res["memory"]), flush=True)
        tok = gen.tokenizer
        runs = []
        rec_arrays = {}
        for i, p in enumerate(PROMPTS[:N_PROMPTS]):
            chat = tok.apply_chat_template(
                [{"role": "user", "content": p}], add_generation_prompt=True, enable_thinking=False, tokenize=False
            )
            ids = tok.encode(chat, add_special_tokens=False)
            HYB.reset_stats()
            HYB.record = [] if os.environ.get("HYB_RECORD") == "1" else None
            t1 = time.time()
            out = gen.generate(list(ids), NEW, enable_trace=os.environ.get("POC_TRACE") == "1")
            dt = time.time() - t1
            st = dict(HYB.st)
            from hybrid.seg import SEG

            st["seg"] = dict(SEG.stats)
            text = tok.decode(out)
            r = {
                "prompt": p[:80],
                "prompt_tokens": len(ids),
                "wall_s": round(dt, 1),
                "ms_per_token_incl_prefill": round(1e3 * dt / NEW, 1),
                "hit_rate": round(st["hits"] / max(1, st["picks"]), 4),
                "decode_steps": st["steps"],
                "trace_replays": st["seg"]["replays"],
                "dyn_hits": st["dyn_hits"],
                "promoted": st["promoted"],
                "promote_s": round(st["t_promote"], 2),
                "host_s": round(st["t_host"], 2),
                "readback_s": round(st["t_read"], 2),
                "upload_s": round(st["t_up"], 2),
                "prefill_rows": st["prefill_rows"],
                "prefill_miss_pairs": st["prefill_miss_pairs"],
                "text": text,
                "tokens": [int(t) for t in out],
            }
            runs.append(r)
            print(f"P{i} {json.dumps({k: v for k, v in r.items() if k not in ('text', 'tokens')})}", flush=True)
            print(f"P{i} TEXT {text[:300]!r}", flush=True)
            if HYB.record:
                layers = sorted({l for l, _ in HYB.record})
                nl = len(layers)
                steps = len(HYB.record) // nl
                arr = np.stack([v.numpy() for _, v in HYB.record[: steps * nl]]).reshape(steps, nl, -1)
                rec_arrays[f"p{i}"] = arr
                np.savez_compressed(os.path.join(out_dir, "routing.npz"), layers=np.array(layers), **rec_arrays)
            res["runs"] = runs
            json.dump(res, open(os.path.join(out_dir, "result.json"), "w"), indent=1)
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush()
        os._exit(1)  # the hybrid prep thread pool and device threads otherwise keep a failed run alive
    sys.stdout.flush()
    os._exit(0)
