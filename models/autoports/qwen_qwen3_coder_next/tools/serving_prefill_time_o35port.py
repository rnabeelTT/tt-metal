# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Wall time of the SERVED prefill entry (``prefill_serving``: traced bucket, device GDN snapshot + slot write, host
logits) per prompt length, split into its stages (host-synchronized, so the stage sums equal the total).

    python -m models.autoports.qwen_qwen3_coder_next.tools.serving_prefill_time [T ...]

env: REPS (8), QWEN35_MAX_BATCH (32), BATCHED=n (time n same-bucket prompts in one call; batched prefill if enabled).
"""
import collections
import os
import sys
import time

import torch

import ttnn
from models.autoports.qwen_qwen3_coder_next.tt.generator import build_generator


def main():
    lens = [int(v) for v in sys.argv[1:]] or [32, 128, 512, 2048]
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), l1_small_size=24576, trace_region_size=768 << 20)
    try:
        gen = build_generator("models/autoports/qwen_qwen3_coder_next", mesh, max_seq_len=16384,
                              max_batch_size=int(os.environ.get("QWEN35_MAX_BATCH", 32)), warm_decode_trace=True)
        model = gen.model
        stage = collections.defaultdict(float)

        def timed(obj, name, label):
            f = getattr(obj, name)

            def w(*a, **k):
                ttnn.synchronize_device(mesh)
                t0 = time.perf_counter()
                r = f(*a, **k)
                ttnn.synchronize_device(mesh)
                stage[label] += time.perf_counter() - t0
                return r

            setattr(obj, name, w)

        timed(model, "prefill_masked_bucket", "bucket")
        timed(gen, "_host_logits", "host_logits")
        timed(model, "_snapshot_gdn_scratch_device", "snapshot")
        timed(model, "_write_gdn_slot_device", "slot_write")
        timed(model, "_reset_gdn_state_for_new_sequence", "gdn_reset")
        timed(model, "_build_request_rope", "rope")
        n = int(os.environ.get("BATCHED", 1))
        reps = int(os.environ.get("REPS", 8))
        for T in lens:
            toks = torch.randint(1000, 100000, (n, T), dtype=torch.int64)
            tot = []
            per = collections.defaultdict(list)
            for i in range(reps):
                stage.clear()
                ttnn.synchronize_device(mesh)
                t0 = time.perf_counter()
                gen.prefill_serving(toks, gen.page_table[:n], [T] * n, list(range(n)))
                ttnn.synchronize_device(mesh)
                tot.append(1e3 * (time.perf_counter() - t0))
                for k, v in stage.items():
                    per[k].append(1e3 * v)
            med = lambda xs: sorted(xs[1:])[len(xs[1:]) // 2]
            parts = " ".join(f"{k} {med(v):.1f}" for k, v in sorted(per.items()))
            print(f"SERVE_PREFILL T={T} n={n}: total {med(tot):.1f} ms | {parts}", flush=True)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
