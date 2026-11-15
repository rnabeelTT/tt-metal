# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Where prefill time goes: the engine's prefill of an N-token prompt (after one untimed warm prefill), wall time, and
the hybrid overlay's host counters per chunk (t_* keys of HYB.st). Dev signal.

usage (hybrid env as tools/eval_hybrid.sh, TOOL=prefill_bench.py): prefill_bench.py [N=8192]
env: PF_MTP_PROMPT (1: with the drafter's prompt pass, as served), ENGINE_CTX
"""

import os
import sys
import time

if os.environ.get("HYB_ENABLE") == "1":
    sys.path.insert(0, os.environ.get("HYB_POC_DIR", "/opt/tt-metal/models/experimental/hybrid_moe"))
    os.environ.setdefault("HYB_PORT", "thinkingmachines_inkling_small")
    import hybrid

    hybrid.install()
    from hybrid.patch import HYB
else:
    HYB = None

import torch  # noqa: E402

import ttnn  # noqa: E402
from models.autoports.thinkingmachines_inkling_small.tt.inkling_serve import InklingSpecServing  # noqa: E402
from models.autoports.thinkingmachines_inkling_small.tt.model import create_model  # noqa: E402


def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 8192
    os.environ.setdefault("INKLING_MTP_PROMPT", os.environ.get("PF_MTP_PROMPT", "1"))
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), l1_small_size=24576, trace_region_size=64_000_000)
    try:
        args, model = create_model(mesh, max_seq_len=int(os.environ.get("ENGINE_CTX", "32768")))
        eng = InklingSpecServing(model, args, mesh, int(os.environ.get("INKLING_SPEC_K", "2")), trace=False)
        ids = (torch.arange(N + 1) * 7919 % 150000 + 1000).tolist()
        eng.warm()  # compiles every prefill bucket; pins the device-format copies when streaming (HYB_DEV_EXPERTS)
        ttnn.synchronize_device(mesh)
        st0 = dict(HYB.st) if HYB is not None else {}
        eng._reset()
        t0 = time.time()
        eng._prefill(ids)
        ttnn.synchronize_device(mesh)
        dt = time.time() - t0
        nch = -(-N // eng.chunk)
        print(f"prefill {N} tokens: {dt:.2f} s ({N / dt:.0f} tok/s), {nch} chunks of {eng.chunk}: "
              f"{dt / nch * 1e3:.0f} ms per chunk", flush=True)
        if HYB is not None:
            d = {k: HYB.st[k] - st0.get(k, 0) for k in HYB.st if isinstance(HYB.st[k], (int, float))}
            print("hybrid host timers (ms per chunk): " + " ".join(
                f"{k} {1e3 * v / nch:.0f}" for k, v in sorted(d.items()) if k.startswith("t_") and v > 0), flush=True)
            print("counters per chunk: " + " ".join(
                f"{k} {v / nch:.0f}" for k, v in sorted(d.items()) if not k.startswith("t_") and v), flush=True)
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()
