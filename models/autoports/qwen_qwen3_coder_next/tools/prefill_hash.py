# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Bit-exactness fingerprint of the served prefill: sha256 of the next-token logits (and the argmax) for fixed prompts
of several lengths, through ``prefill_serving`` (traced buckets, device slot writes). Run it on two builds / env
settings and compare the PREFILL_HASH lines.

    python -m models.autoports.qwen_qwen3_coder_next.tools.prefill_hash [T ...]

env: BATCHED=n also hashes n same-length prompts admitted in one call (the batched pass when it is enabled).
"""
import hashlib
import os
import sys

import torch

import ttnn
from models.autoports.qwen_qwen3_coder_next.tt.generator import build_generator


def main():
    lens = [int(v) for v in sys.argv[1:]] or [20, 128, 500, 2048, 3000]
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), l1_small_size=24576, trace_region_size=768 << 20)
    try:
        gen = build_generator("models/autoports/qwen_qwen3_coder_next", mesh, max_seq_len=16384,
                              max_batch_size=int(os.environ.get("QWEN35_MAX_BATCH", 32)), warm_decode_trace=True)
        n = int(os.environ.get("BATCHED", 1))
        for T in lens:
            g = torch.Generator().manual_seed(T)
            toks = torch.randint(1000, 100000, (n, T), dtype=torch.int64, generator=g)
            lg = gen.prefill_serving(toks, gen.page_table[:n], [T] * n, list(range(n)))
            lg = torch.as_tensor(lg).float().reshape(n, -1)
            for u in range(n):
                h = hashlib.sha256(lg[u].numpy().tobytes()).hexdigest()[:16]
                print(f"PREFILL_HASH T={T} n={n} u={u}: {h} argmax {int(lg[u].argmax())} max {float(lg[u].max()):.4f}",
                      flush=True)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
