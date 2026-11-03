# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Hybrid-MoE POC, step 2: record the real decode routing of Qwen3.5-122B-A10B on the QB2 (mesh 1x4).

Runs inside the 122B tt-model image (code at /opt/tt-metal). Greedy single-user generation with the decode trace OFF so
the router runs through Python every step; a wrapper on Qwen35MoE._route reads the top-k expert ids of the decode row
(row 0) for every MoE layer. Output: routing.npz with ids[step, layer, k] per prompt, plus the generated texts.
"""

import json
import os
import sys
import time

import numpy as np
import torch

import ttnn

os.environ.setdefault("HF_MODEL", "Qwen/Qwen3.5-122B-A10B")
from models.autoports.qwen_qwen3_5_122b_a10b.tt import fused_decoder as fd  # noqa: E402
from models.autoports.qwen_qwen3_5_122b_a10b.tt import moe as moe_mod  # noqa: E402
from models.autoports.qwen_qwen3_5_122b_a10b.tt.generator import build_generator  # noqa: E402

PROMPTS = [
    "Write a detailed, 400-word explanation of how a refrigerator works, for a curious teenager.",
    "Tell a 400-word short story about a lighthouse keeper who finds a message in a bottle.",
    "Explain in about 400 words why the sky is blue and why sunsets are red, with the physics.",
    "Write a Python function that parses an Apache access log and returns the top 10 IPs by request count, "
    "with type hints, docstrings, and unit tests using pytest.",
    "I'm getting 'TypeError: Cannot read properties of undefined (reading map)' in my React component that "
    "fetches users from an API in useEffect. Explain the likely causes and show a fixed version.",
    "Solve step by step: a train leaves at 3pm at 60 km/h, another leaves the same station at 4pm at 90 km/h on "
    "the same track. When and where does the second catch the first? Then generalize the formula.",
    "Compare PostgreSQL and MongoDB for an e-commerce order system. Cover transactions, schema evolution, "
    "indexing and operational cost, and end with a recommendation.",
    "Write a SQL query schema and queries for a library system: books, members, loans, with overdue fines. "
    "Include the CREATE TABLE statements.",
    "Draft a polite but firm email to a contractor who is three weeks late on a kitchen renovation, asking for a "
    "revised schedule and mentioning the penalty clause.",
    "Explain how transformers in machine learning work: attention, positional encoding, and why they replaced RNNs.",
    "Summarize the causes and consequences of the 2008 financial crisis in about 400 words.",
    "Write a bash script that backs up a directory to a timestamped tar.gz, keeps the last 7 backups, and logs "
    "to syslog. Explain each part.",
    "Translate the following into French and Spanish, then explain any idioms: 'It's raining cats and dogs, so "
    "let's call it a day and hit the hay early.'",
    "Give me a 5-day meal plan for a vegetarian athlete with roughly 140 g protein per day, with a shopping list.",
    "Implement a thread-safe LRU cache in Rust with get and put, explain the ownership choices.",
    "What are the main differences between TCP and QUIC? Explain head-of-line blocking and connection migration.",
]
# the image ships no tests package: the (1,4) mesh parameters of tests/test_multichip_decoder.py
MESH_PARAMS = {
    "fabric_config": ttnn.FabricConfig.FABRIC_1D,
    "l1_small_size": 24576,  # generator.L1_SMALL_SIZE (GDN prefill conv1d)
    "trace_region_size": 64_000_000,
}
NEW_TOKENS = int(os.environ.get("POC_NEW_TOKENS", "320"))

LOG = []  # (layer, ids[k]) for the current prompt's decode steps
_depth = [0]


def wrap(cls):
    orig = cls._route

    def _route(self, x4):
        _depth[0] += 1
        try:
            scores, ids = orig(self, x4)
        finally:
            _depth[0] -= 1
        if _depth[0] == 0 and x4.shape[-2] <= 32:  # outermost call, decode row(s)
            t = ttnn.to_torch(ttnn.get_device_tensors(ids)[0]).reshape(-1, self.top_k)
            LOG.append((self.layer_num, t[0].to(torch.int32).numpy().copy()))
        return scores, ids

    cls._route = _route


def main():
    wrap(moe_mod.Qwen35MoE)
    wrap(fd.Qwen35MoEFused)
    out_dir = sys.argv[1]
    os.makedirs(out_dir, exist_ok=True)
    ttnn.set_fabric_config(MESH_PARAMS["fabric_config"])
    mesh = ttnn.open_mesh_device(
        ttnn.MeshShape(1, 4),
        l1_small_size=MESH_PARAMS["l1_small_size"],
        trace_region_size=MESH_PARAMS["trace_region_size"],
    )
    try:
        t0 = time.time()
        gen = build_generator(
            "models/autoports/qwen_qwen3_5_122b_a10b",
            mesh,
            max_seq_len=4096,
            warm_decode_trace=False,
            warm_prefill_trace=False,
        )
        print(f"BUILD {time.time() - t0:.0f}s", flush=True)
        tok = gen.tokenizer
        arrays, texts = {}, []
        for i, p in enumerate(PROMPTS):
            chat = tok.apply_chat_template(
                [{"role": "user", "content": p}], add_generation_prompt=True, enable_thinking=False, tokenize=False
            )
            ids = tok.encode(chat, add_special_tokens=False)
            LOG.clear()
            t1 = time.time()
            out = gen.generate(list(ids), NEW_TOKENS, enable_trace=False)
            dt = time.time() - t1
            layers = sorted({l for l, _ in LOG})
            nl = len(layers)
            steps = len(LOG) // nl
            arr = np.stack([v for _, v in LOG[: steps * nl]]).reshape(steps, nl, -1)
            # every step must visit the layers in the same order
            order = [l for l, _ in LOG[:nl]]
            if order != layers:
                raise RuntimeError(f"unexpected layer order {order[:5]}... vs {layers[:5]}")
            arrays[f"p{i}"] = arr
            text = tok.decode(out)
            texts.append({"prompt": p, "prompt_tokens": len(ids), "steps": steps, "text": text})
            print(f"P{i} steps={steps} layers={nl} {dt:.0f}s {text[:90]!r}", flush=True)
            np.savez_compressed(os.path.join(out_dir, "routing.npz"), layers=np.array(layers), **arrays)
            json.dump(texts, open(os.path.join(out_dir, "texts.json"), "w"), indent=1)
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()
