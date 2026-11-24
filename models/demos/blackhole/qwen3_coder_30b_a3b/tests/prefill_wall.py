# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Wall time of one user's prefill (device sampling, bucketed, warm programs) per prompt length, and the host-side
dispatch share: the same call with the device synchronized only at the end vs the device time from decode_prof.

    python -m models.demos.blackhole.qwen3_coder_30b_a3b.tests.prefill_wall [T ...]

env: REPS (6). Also prints PREFILL_HASH (sha256 of host logits) so a change to the prefill path can be checked
bit-exact against this run.
"""
import hashlib
import os
import sys
import time

import torch

import ttnn
from models.demos.blackhole.qwen3_coder_30b_a3b.tt import generator as gen_mod


def main():
    lens = [int(v) for v in sys.argv[1:]] or [32, 128, 512, 2048]
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 4), trace_region_size=48 << 20)
    try:
        gen = gen_mod.build_generator("models/demos/blackhole/qwen3_coder_30b_a3b", mesh, max_context_len=8192,
                                      max_batch_size=1)
        kv = gen._ensure_kv_cache()
        reps = int(os.environ.get("REPS", 6))
        for T in lens:
            g = torch.Generator().manual_seed(T)
            ids = torch.randint(1000, 100000, (1, T), dtype=torch.int64, generator=g)
            pt = gen.make_page_table([T + 64])
            lg = gen.prefill_forward(ids, page_table=pt, kv_cache=kv, prompt_lens=[T], sampling_mode="host")
            lg = torch.as_tensor(lg).float().reshape(-1)
            print(f"PREFILL_HASH T={T}: {hashlib.sha256(lg.numpy().tobytes()).hexdigest()[:16]} argmax {int(lg.argmax())}",
                  flush=True)
            for mode in ("host", "device"):
                ws = []
                for _ in range(reps):
                    ttnn.synchronize_device(mesh)
                    t0 = time.perf_counter()
                    gen.prefill_forward(ids, page_table=pt, kv_cache=kv, prompt_lens=[T], sampling_mode=mode)
                    ttnn.synchronize_device(mesh)
                    ws.append(1e3 * (time.perf_counter() - t0))
                ws = sorted(ws[1:])
                print(f"PREFILL_WALL T={T} sampling={mode}: median {ws[len(ws) // 2]:.1f} ms min {ws[0]:.1f}", flush=True)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
