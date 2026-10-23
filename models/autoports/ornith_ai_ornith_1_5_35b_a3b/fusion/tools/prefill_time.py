"""Full-model prefill wall time (the served traced bucket path) on MESH_DEVICE, for A/B runs of env settings.

    python prefill_time.py [T ...]      (default 128 2048 8192; REPS - 1 timed calls each after one warm call)

env: REPS (6), WARM_DECODE_TRACE (1), SIGNPOST=1 (tracy signposts around the last call), QWEN35_MAX_BATCH (32).
"""

import os
import sys
import time

import torch

import ttnn
from models.autoports.ornith_ai_ornith_1_5_35b_a3b.tt.generator import build_generator


def main():
    lens = [int(v) for v in sys.argv[1:]] or [128, 2048, 8192]
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    shape = {"P300": (1, 2), "P300x2": (1, 4)}[os.environ.get("MESH_DEVICE", "P300")]
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(*shape), l1_small_size=24576, trace_region_size=768 << 20)
    try:
        gen = build_generator("models/autoports/ornith_ai_ornith_1_5_35b_a3b", mesh, max_seq_len=16384,
                              max_batch_size=int(os.environ.get("QWEN35_MAX_BATCH", 32)),
                              warm_decode_trace=os.environ.get("WARM_DECODE_TRACE", "1") == "1")
        for T in lens:
            toks = torch.randint(1000, 100000, (1, T), dtype=torch.int64)
            ts = []
            reps = int(os.environ.get("REPS", 6))
            for i in range(reps):
                gen.reset()
                ttnn.synchronize_device(mesh)
                if i == reps - 1 and os.environ.get("SIGNPOST") == "1":
                    from tracy import signpost

                    signpost(f"PREFILL_{T}")
                t0 = time.perf_counter()
                gen.prefill_forward(toks, page_table=gen.page_table[:1], prompt_lens=[T])
                ttnn.synchronize_device(mesh)
                ts.append(1e3 * (time.perf_counter() - t0))
                if i == reps - 1 and os.environ.get("SIGNPOST") == "1":
                    signpost(f"PREFILL_{T}_END")
                    ttnn.ReadDeviceProfiler(mesh)
            ts = sorted(ts[1:])
            print(f"PREFILL_TIME T={T}: median {ts[len(ts) // 2]:.1f} ms min {ts[0]:.1f} ({len(ts)} calls)", flush=True)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
