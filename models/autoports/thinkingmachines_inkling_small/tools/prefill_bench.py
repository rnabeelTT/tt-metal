# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Where prefill time goes: the engine's prefill of an N-token prompt (after one untimed warm prefill), wall time, and
the hybrid overlay's host counters per chunk (t_* keys of HYB.st). Dev signal.

usage (hybrid env as tools/eval_hybrid.sh, TOOL=prefill_bench.py): prefill_bench.py [N=8192]
env: PF_MTP_PROMPT (1: with the drafter's prompt pass, as served), ENGINE_CTX, PF_CAPTURE (1: traces captured and a\n     generation before the timed prefill, as served), PF_PRE (a long prefill of this many tokens before the captures)
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
    Ns = [int(v) for v in (sys.argv[1] if len(sys.argv) > 1 else "8192").split(",")]
    N = Ns[-1]
    os.environ.setdefault("INKLING_MTP_PROMPT", os.environ.get("PF_MTP_PROMPT", "1"))
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(
        ttnn.MeshShape(1, 4),
        l1_small_size=24576,
        trace_region_size=int(os.environ.get("SPEC_TRACE_REGION", "335544320")),
    )
    try:
        args, model = create_model(mesh, max_seq_len=int(os.environ.get("ENGINE_CTX", "32768")))
        cap = os.environ.get("PF_CAPTURE") == "1"  # as served: traces captured and replayed before the timed prefill
        eng = InklingSpecServing(model, args, mesh, int(os.environ.get("INKLING_SPEC_K", "2")), trace=cap)
        ids = (torch.arange(N + 1) * 7919 % 150000 + 1000).tolist()
        eng.warm()  # compiles every prefill bucket; pins the device-format copies when streaming (HYB_DEV_EXPERTS)
        pre = int(os.environ.get("PF_PRE", "0"))  # a long prefill before the captures (lazy device caches then exist)
        if pre:
            eng._reset()
            eng._prefill((torch.arange(pre + 1) * 31337 % 150000 + 1000).tolist())
            ttnn.synchronize_device(mesh)
            eng._reset()
        if cap:
            eng.capture()
            print("generated before the prefill:", len(eng.generate(ids[:300], 64)), "tokens", flush=True)
        ttnn.synchronize_device(mesh)
        for n_prev in Ns[:-1]:  # earlier requests: a prefill of n_prev tokens, then 32 generated (PF_CAPTURE: traced)
            t0 = time.time()
            out = eng.generate((torch.arange(n_prev + 1) * 104729 % 150000 + 1000).tolist(), 32) if cap else None
            if not cap:
                eng._reset()
                eng._prefill((torch.arange(n_prev + 1) * 104729 % 150000 + 1000).tolist())
            ttnn.synchronize_device(mesh)
            print(f"earlier request: {n_prev} prompt tokens in {time.time() - t0:.1f} s", flush=True)
        st0 = dict(HYB.st) if HYB is not None else {}
        eng._reset()
        t0 = time.time()
        eng._prefill(ids)
        ttnn.synchronize_device(mesh)
        dt = time.time() - t0
        nch = -(-N // eng.chunk)
        print(
            f"prefill {N} tokens: {dt:.2f} s ({N / dt:.0f} tok/s), {nch} chunks of {eng.chunk}: "
            f"{dt / nch * 1e3:.0f} ms per chunk",
            flush=True,
        )
        if HYB is not None:
            d = {k: HYB.st[k] - st0.get(k, 0) for k in HYB.st if isinstance(HYB.st[k], (int, float))}
            print(
                "hybrid host timers (ms per chunk): "
                + " ".join(f"{k} {1e3 * v / nch:.0f}" for k, v in sorted(d.items()) if k.startswith("t_") and v > 0),
                flush=True,
            )
            print(
                "counters per chunk: "
                + " ".join(f"{k} {v / nch:.0f}" for k, v in sorted(d.items()) if not k.startswith("t_") and v),
                flush=True,
            )
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()
