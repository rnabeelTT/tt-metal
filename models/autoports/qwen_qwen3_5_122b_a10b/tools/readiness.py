# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Run the readiness_check runners for this model on a mesh opened with the device parameters the model needs.

The stock CLIs open the mesh with the bare ``ttnn.open_mesh_device(mesh_shape=...)`` (trace region 0, small-L1 0); this
model's decode trace and GDN prefill conv1d need both, and a FABRIC_1D mesh cannot be re-opened inside one process on
p300x2. This wrapper opens the (1,4) mesh with ``l1_small_size=24576`` and a trace region, then calls the runners'
programmatic entry points unchanged (same references, same reports).

    python -m models.autoports.qwen_qwen3_5_122b_a10b.tools.readiness prefill-check --reference bringup/references/X.refpt
    python -m models.autoports.qwen_qwen3_5_122b_a10b.tools.readiness teacher-forcing --reference ...
    python -m models.autoports.qwen_qwen3_5_122b_a10b.tools.readiness autoregressive --prompt-file P --output-dir D [--max-new-tokens N]
"""

import argparse
import json
import os
import sys
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parents[1]
HF_MODEL = "Qwen/Qwen3.5-122B-A10B"
L1_SMALL_SIZE = 24576


def _open_mesh(trace_region):
    import ttnn

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    return ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(1, 4),
        l1_small_size=L1_SMALL_SIZE,
        trace_region_size=trace_region,
        num_command_queues=1,
    )


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["prefill-check", "teacher-forcing", "autoregressive"])
    ap.add_argument("--reference", type=Path)
    ap.add_argument("--prompt-file", type=Path)
    ap.add_argument("--output-dir", type=Path)
    ap.add_argument("--max-new-tokens", type=int, default=None)
    ap.add_argument("--hf-model", default=HF_MODEL)
    ap.add_argument("--trace-region", type=int, default=int(os.environ.get("QWEN35_TRACE_REGION", 512 * 1024 * 1024)))
    ap.add_argument(
        "--build-kwargs", default="{}", help="JSON kwargs for build_generator (max_seq_len, layer_indices, ...)"
    )
    ap.add_argument("--report", type=Path, help="write the runner's result list as JSON")
    a = ap.parse_args(argv)
    os.environ.setdefault("HF_MODEL", a.hf_model)
    import ttnn

    mesh = _open_mesh(a.trace_region)
    try:
        kw = json.loads(a.build_kwargs)
        if a.mode == "prefill-check":
            from readiness_check.run_prefill_check import run_prefill_check

            res = run_prefill_check(
                model_dir=MODEL_DIR, reference_path=a.reference.resolve(), mesh_device=mesh, build_kwargs=kw
            )
        elif a.mode == "teacher-forcing":
            from readiness_check.run_teacher_forcing import run_teacher_forcing

            res = run_teacher_forcing(
                model_dir=MODEL_DIR, reference_path=a.reference.resolve(), mesh_device=mesh, build_kwargs=kw
            )
        else:
            from readiness_check.run_autoregressive import run_autoregressive

            extra = {"max_new_tokens": a.max_new_tokens} if a.max_new_tokens else {}
            res = run_autoregressive(
                model_dir=MODEL_DIR,
                hf_model_id=a.hf_model,
                prompt_file=a.prompt_file.resolve(),
                mesh_device=mesh,
                output_dir=a.output_dir.resolve(),
                build_kwargs=kw,
                **extra,
            )
            res = {k: str(v) for k, v in res.items()}
        print("READINESS_RESULT " + json.dumps(res, default=str))
        if a.report:
            a.report.parent.mkdir(parents=True, exist_ok=True)
            a.report.write_text(json.dumps(res, indent=2, default=str) + "\n")
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    sys.exit(main())
