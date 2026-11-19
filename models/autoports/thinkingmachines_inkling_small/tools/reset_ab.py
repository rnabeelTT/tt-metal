# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Speed of HYB_RESET_PER_REQUEST in one process (the engine as served, traces captured): rounds alternating the reset
on and off, each round a long prompt (AB_LONG tokens of held-out text) and AB_SHORT short prompts, AB_GEN tokens
generated greedily per request; prefill tok/s of the long prompt and decode tok/s per mode.

usage (in the image, hybrid env as tools/eval_hybrid.sh TOOL=reset_ab.py): AB_LONG (16384), AB_SHORT (3), AB_GEN (128),
AB_ROUNDS (4: on, off, on, off), PF_TEXT (held-out texts json)
"""

import json
import os
import sys
import time

import ttnn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prefill_bench import HYB, create_model  # noqa: E402

from models.autoports.thinkingmachines_inkling_small.tt.inkling_serve import InklingSpecServing  # noqa: E402


def main():
    n_long = int(os.environ.get("AB_LONG", "16384"))
    n_short = int(os.environ.get("AB_SHORT", "3"))
    gen = int(os.environ.get("AB_GEN", "128"))
    rounds = int(os.environ.get("AB_ROUNDS", "4"))
    os.environ.setdefault("INKLING_MTP_PROMPT", "1")
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(
        ttnn.MeshShape(1, 4), l1_small_size=24576, trace_region_size=int(os.environ.get("SPEC_TRACE_REGION", "335544320"))
    )
    try:
        args, model = create_model(mesh, max_seq_len=int(os.environ.get("ENGINE_CTX", "32768")))
        eng = InklingSpecServing(model, args, mesh, int(os.environ.get("INKLING_SPEC_K", "2")), trace=True)
        docs = json.load(open(os.environ["PF_TEXT"]))
        text = [t for d in (docs.values() if isinstance(docs, dict) else docs) for t in d]
        eng.warm()
        eng.capture()

        def req(off, n):
            ids = [text[(off + j) % len(text)] for j in range(n)]
            t0 = time.time()
            out = [eng.prefill(ids)]
            t1 = time.time()
            while len(out) < gen and out[-1] not in eng.eos:
                out += eng.step()
            t2 = time.time()
            return n / (t1 - t0), (len(out) - 1) / max(t2 - t1, 1e-9), len(out)

        res = {0: [], 1: []}
        for r in range(rounds):
            mode = 1 - r % 2  # on first
            HYB.reset_per_request = bool(mode)
            pf, dec, n = req(r * 104729 % len(text), n_long)
            decs = [dec]
            for s in range(n_short):
                _, d, _ = req((r * 31 + s) * 7919 % len(text), 30)
                decs.append(d)
            res[mode].append((pf, sum(decs) / len(decs)))
            print(f"round {r + 1} reset {'on ' if mode else 'off'}: long prefill {pf:.0f} tok/s, decode mean "
                  f"{sum(decs) / len(decs):.1f} tok/s ({', '.join(f'{d:.1f}' for d in decs)})", flush=True)
        for mode in (1, 0):
            v = res[mode]
            if v:
                print(f"AB reset {'on ' if mode else 'off'}: prefill {sum(x[0] for x in v) / len(v):.0f} tok/s, "
                      f"decode {sum(x[1] for x in v) / len(v):.1f} tok/s over {len(v)} rounds", flush=True)
    finally:
        os._exit(0)


if __name__ == "__main__":
    main()
