# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Does greedy generation repeat? The engine as served (traces captured), three prompts of different lengths from the
held-out texts generated greedily in an interleaved order (no prompt directly after itself, so the prefix cache never
resumes one), each prompt's outputs compared with its first. Prints per prompt the runs identical to the first and,
for a run that differs, the first differing position. Every run's tokens go to DET_OUT (json) for cross-process
comparison (DET_CMP=<earlier json>: compares with it too).

usage (in the image, hybrid env as tools/eval_hybrid.sh TOOL=determinism_check.py): DET_LENS (default 300,1500,4500),
DET_GEN (generated tokens, default 96), DET_ORDER (default ABCACBA), PF_TEXT (the held-out texts json)
"""

import json
import os
import sys

import torch

import ttnn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prefill_bench import create_model  # noqa: E402

from models.autoports.thinkingmachines_inkling_small.tt.inkling_serve import InklingSpecServing  # noqa: E402


def main():
    lens = [int(v) for v in os.environ.get("DET_LENS", "300,1500,4500").split(",")]
    gen = int(os.environ.get("DET_GEN", "96"))
    order = os.environ.get("DET_ORDER", "ABCACBA")
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
        prompts = {}
        for i, n in enumerate(lens):
            off = (i * 7919 * 37) % len(text)
            prompts["ABCDEFGH"[i]] = [text[(off + j) % len(text)] for j in range(n)]
        eng.warm()
        eng.capture()
        runs = {k: [] for k in prompts}
        for k in order:
            out = eng.generate(prompts[k], gen)
            runs[k].append(out)
            same = out == runs[k][0]
            print(f"run {k} (#{len(runs[k])}, {len(prompts[k])} prompt tokens): {len(out)} tokens, "
                  f"{'same as its first' if same else 'DIFFERS from its first'}", flush=True)
        ok = True
        for k, outs in runs.items():
            diffs = []
            for o in outs[1:]:
                if o != outs[0]:
                    pos = next((i for i, (a, b) in enumerate(zip(o, outs[0])) if a != b), min(len(o), len(outs[0])))
                    diffs.append(pos)
            ok = ok and not diffs
            print(f"DET {k}: {len(outs)} runs, {len(outs) - 1 - len(diffs)} of {len(outs) - 1} repeats identical"
                  + (f"; first differing positions {diffs}" if diffs else ""), flush=True)
        cmp = os.environ.get("DET_CMP")
        if cmp and os.path.exists(cmp):
            prev = json.load(open(cmp))
            for k, outs in runs.items():
                if k in prev:
                    same = prev[k][0] == outs[0]
                    ok = ok and same
                    print(f"DET {k} vs {os.path.basename(cmp)}: {'identical' if same else 'DIFFERS'}", flush=True)
        if os.environ.get("DET_OUT"):
            json.dump(runs, open(os.environ["DET_OUT"], "w"))
        print("DET ALL IDENTICAL" if ok else "DET NOT DETERMINISTIC", flush=True)
    finally:
        os._exit(0)  # (close_mesh_device can hang after fetch runs; the container is removed by the caller)


if __name__ == "__main__":
    main()
