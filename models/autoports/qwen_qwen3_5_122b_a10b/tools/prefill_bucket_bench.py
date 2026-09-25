# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage-7 prefill evidence: the persistent-input masked-bucket prefill (eager and TRACED) against the stock
host-upload path, on real weights and the (1,4) mesh.

For each prompt length: logits of the stock path (``Qwen36Model.prefill_masked_bucket`` through the generator's
segmenting) vs the persistent-input path (eager) vs the traced replays: max |diff|, argmax equality, PCC; GDN state
(recurrent + conv) equality after the prefill; the first decode token after each. Then TTFT (prefill wall) per path.

    python -m models.autoports.qwen_qwen3_5_122b_a10b.tools.prefill_bucket_bench --layers 0,3 --lens 1,17,161,500,2048,2049,4097
"""
import argparse
import json
import os
import time
from pathlib import Path

import torch

import ttnn
from models.autoports.qwen_qwen3_5_122b_a10b.tools.readiness import _open_mesh
from models.autoports.qwen_qwen3_5_122b_a10b.tt.generator import build_generator
from models.demos.blackhole.qwen36.tt.model import Qwen36Model

MODEL_DIR = "models/autoports/qwen_qwen3_5_122b_a10b"


def _pcc(a, b):
    a = a.flatten().double()
    b = b.flatten().double()
    if torch.allclose(a, b):
        return 1.0
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


class _Paths:
    """Switch the model between the stock path, the persistent-input eager path and the traced path."""

    def __init__(self, gen):
        self.gen = gen
        self.model = gen.model
        self.scratch = gen.model._kv_scratch_block
        self.traces = dict(getattr(gen.model, "_bucket_traces", {}))

    def use(self, mode):
        m = self.model
        if mode == "stock":
            m._kv_scratch_block = None  # prefill_masked_bucket falls back to Qwen36Model's path
            m._bucket_traces = {}
        elif mode == "dev":
            m._kv_scratch_block = self.scratch
            m._bucket_traces = {}
        elif mode == "traced":
            m._kv_scratch_block = self.scratch
            m._bucket_traces = dict(self.traces)
        else:
            raise ValueError(mode)


def _gdn_state(gen):
    comp = ttnn.ConcatMeshToTensor(gen.mesh_device, dim=0)
    out = []
    for layer in gen.model.layers:
        if layer.is_full_attention:
            continue
        dn = layer.attention
        out.append(ttnn.to_torch(dn.rec_state, mesh_composer=comp).float())
        out.extend(ttnn.to_torch(c, mesh_composer=comp).float() for c in dn.conv_states)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="0,3")
    ap.add_argument("--lens", default="1,17,128,129,161,500,2048,2049,4097")
    ap.add_argument("--max-seq-len", type=int, default=8192)
    ap.add_argument("--timing-lens", default="128,161,2048")
    ap.add_argument("--trace-region", type=int, default=int(os.environ.get("QWEN35_TRACE_REGION", 512 * 1024 * 1024)))
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--no-equiv", action="store_true", help="timing only (48 layers)")
    ap.add_argument(
        "--modes",
        default="stock,dev,traced",
        help="paths to run; the STOCK path compiles programs after the captures unless --warm-stock, and a replay in "
        "between then clobbers them (README section 4) -> use dev,traced for 48-layer timing runs",
    )
    ap.add_argument(
        "--warm-stock",
        action="store_true",
        help="compile the STOCK host-upload path's programs before the bucket traces are captured (a program compiled "
        "after a capture gets its binaries clobbered by the next replay -> mesh-wide NoC hang; the stock path is bench-only)",
    )
    a = ap.parse_args(argv)
    layers = None if a.layers == "all" else [int(s) for s in a.layers.split(",")]
    mesh = _open_mesh(a.trace_region)
    rep = {"layers": a.layers, "equiv": [], "timing": []}
    try:
        # the stock path is bench-only: compile ALL of its programs before ANY trace (decode or bucket) is captured —
        # a program compiled after a capture can have its binaries overwritten by a later replay (README section 4)
        gen = build_generator(
            MODEL_DIR,
            mesh,
            layer_indices=layers,
            max_seq_len=a.max_seq_len,
            warm_decode_trace=not a.warm_stock,
            warm_prefill_trace=not a.warm_stock,
        )
        if a.warm_stock:
            m = gen.model
            for b in m._PREFILL_MASK_BUCKETS:
                for n in (1, max(1, b // 2), b):
                    for cs in (0, 2048):
                        if cs == 0:
                            m._reset_gdn_state_for_new_sequence()
                        Qwen36Model.prefill_masked_bucket(
                            m,
                            torch.zeros(1, n, dtype=torch.int32),
                            gen.page_table[:1],
                            actual_len=n,
                            chunk_start=cs,
                            bucket=b,
                        )
            ttnn.synchronize_device(mesh)
            gen._zero_gdn_state()
            gen._warm_decode_trace(False)
            gen._warm_prefill_traces()
        paths = _Paths(gen)
        modes = [m for m in a.modes.split(",") if m]
        torch.manual_seed(0)
        V = gen.args.vocab_size
        if not a.no_equiv:
            for L in [int(s) for s in a.lens.split(",")]:
                toks = torch.randint(1000, 30000, (1, L))
                res = {"len": L}
                ref = None
                for mode in modes:
                    paths.use(mode)
                    gen.reset()
                    print(f"PREFILL_STEP len={L} mode={mode} prefill", flush=True)
                    lg = gen.prefill_forward(toks, page_table=gen.page_table[:1], prompt_lens=[L]).float().reshape(-1)
                    print(f"PREFILL_STEP len={L} mode={mode} state", flush=True)
                    st = _gdn_state(gen)
                    print(f"PREFILL_STEP len={L} mode={mode} decode", flush=True)
                    t1 = int(
                        gen.decode_forward(
                            torch.tensor([[int(torch.argmax(lg))]]),
                            torch.tensor([L]),
                            enable_trace=True,
                            reload_inputs=True,
                        )[0]
                    )
                    print(f"PREFILL_STEP len={L} mode={mode} done", flush=True)
                    if ref is None:
                        ref = (lg, st, t1)
                        res["reference_mode"] = mode
                        res["argmax"] = int(torch.argmax(lg))
                        res["next_token_stock"] = t1
                    else:
                        res[f"{mode}_max_abs_diff"] = float((lg - ref[0]).abs().max())
                        res[f"{mode}_pcc"] = _pcc(lg, ref[0])
                        res[f"{mode}_argmax_equal"] = bool(torch.argmax(lg) == torch.argmax(ref[0]))
                        res[f"{mode}_state_equal"] = all(torch.equal(x, y) for x, y in zip(st, ref[1]))
                        res[f"{mode}_state_max_diff"] = max(float((x - y).abs().max()) for x, y in zip(st, ref[1]))
                        res[f"{mode}_next_token_equal"] = t1 == ref[2]
                rep["equiv"].append(res)
                print("PREFILL_EQUIV " + json.dumps(res), flush=True)
        for L in [int(s) for s in a.timing_lens.split(",") if s and s != "none"]:
            toks = torch.randint(1000, 30000, (1, L))
            row = {"len": L}
            for mode in modes:
                paths.use(mode)
                gen.reset()
                gen.prefill_forward(toks, page_table=gen.page_table[:1], prompt_lens=[L])  # warm this length's programs
                times = []
                for _ in range(3):
                    gen.reset()
                    ttnn.synchronize_device(mesh)
                    t0 = time.perf_counter()
                    lg = gen.prefill_forward(toks, page_table=gen.page_table[:1], prompt_lens=[L])
                    _ = int(torch.argmax(lg.float()))  # host logits consumed = TTFT boundary
                    times.append(1e3 * (time.perf_counter() - t0))
                row[f"{mode}_ms"] = round(min(times), 1)
                row[f"{mode}_ms_all"] = [round(t, 1) for t in times]
            rep["timing"].append(row)
            print("PREFILL_TIMING " + json.dumps(row), flush=True)
        paths.use("traced")
        print("PREFILL_BENCH_DONE", flush=True)
        if a.out:
            a.out.parent.mkdir(parents=True, exist_ok=True)
            a.out.write_text(json.dumps(rep, indent=1) + "\n")
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()
