# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Run the readiness_check runners for this model on a mesh opened with the device parameters the model needs.

The stock CLIs open the mesh with the bare ``ttnn.open_mesh_device(mesh_shape=...)`` (trace region 0, small-L1 0); this
model's decode trace and KDA prefill conv need both, and a FABRIC_1D mesh cannot be re-opened inside one process on
p300x2. This wrapper opens the (1,4) mesh with ``l1_small_size=24576`` and a trace region, then calls the runners'
programmatic entry points unchanged (same references, same reports).

    python -m models.autoports.zai_org_glm_5_3_flash.tools.readiness prefill-check --reference bringup/references/X.refpt
    python -m models.autoports.zai_org_glm_5_3_flash.tools.readiness teacher-forcing --reference ...
    python -m models.autoports.zai_org_glm_5_3_flash.tools.readiness autoregressive --prompt-file P --output-dir D \
        [--max-new-tokens N] [--hf-tokens doc/full_model/autoregressive/hf_tokens.json]

--hf-tokens: run_autoregressive's HF side (`_hf_generate_greedy`: AutoModelForCausalLM.from_pretrained of the 328 GB
parent in this process; python_env's transformers has no glm5_next) cannot run here. The HF greedy completion is
precomputed on CPU by `tools/hf_full_ref.py --mode autoregressive` (same E192 checkpoint, same tokenization, greedy,
generation_config eos ids) and `_hf_generate_greedy` is monkeypatched to return those ids after checking the prompt ids
and token budget match. autoregressive_meta.json then records the substitution under "hf_reference".
"""

import argparse
import json
import os
import sys
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parents[1]
HF_MODEL = "zai-org/GLM-5.3-Flash"
L1_SMALL_SIZE = 24576


def _open_mesh(trace_region):
    import ttnn

    shape = (1, int(os.environ.get("GLM53_TP", "4")))
    if shape != (
        1,
        1,
    ):  # one die runs no CCLs; a fabric on a single die of a multi-chip host never finishes its handshake
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    kw = {}
    if os.environ.get("GLM53_WORKER_L1_SIZE"):  # watcher runs: a smaller worker L1 grows the kernel-config ring
        kw["worker_l1_size"] = int(os.environ["GLM53_WORKER_L1_SIZE"])
    return ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(*shape),
        l1_small_size=L1_SMALL_SIZE,
        trace_region_size=trace_region,
        num_command_queues=1,
        **kw,
    )


def patch_hf_reference(hf_tokens: Path) -> dict:
    """Replace readiness_check.run_autoregressive._hf_generate_greedy with the precomputed hf_full_ref.py tokens."""
    import readiness_check.run_autoregressive as ra

    ref = json.loads(Path(hf_tokens).read_text())

    def _precomputed_hf_generate_greedy(*, hf_model_id, prompt_token_ids, max_new_tokens, device):
        if [int(t) for t in prompt_token_ids] != ref["prompt_token_ids"]:
            raise RuntimeError(f"{hf_tokens}: prompt token ids differ from the runner's tokenization")
        if int(max_new_tokens) != int(ref["max_new_tokens"]):
            raise RuntimeError(f"{hf_tokens}: max_new_tokens {ref['max_new_tokens']} != runner's {max_new_tokens}")
        print(
            f"HF side PRECOMPUTED by tools/hf_full_ref.py --mode autoregressive ({hf_tokens}): {ref['num_tokens']} tokens"
        )
        return list(ref["token_ids"])

    ra._hf_generate_greedy = _precomputed_hf_generate_greedy
    return ref


def annotate_hf_reference(output_dir: Path, hf_tokens: Path, ref: dict) -> None:
    meta_path = Path(output_dir) / "autoregressive_meta.json"
    meta = json.loads(meta_path.read_text())
    meta["hf_reference"] = {
        "precomputed": True,
        "tokens_file": str(Path(hf_tokens).resolve()),
        "how": "readiness_check.run_autoregressive._hf_generate_greedy monkeypatched (tools/readiness.py --hf-tokens) "
        "to return the ids generated on CPU by tools/hf_full_ref.py --mode autoregressive: unmodified HF "
        "Glm5NextTextModel on the same E192 FP8 checkpoint the TT model loads, prompt tokenized identically "
        "(checked), greedy, stops on generation_config eos ids like model.generate",
        "checkpoint": ref.get("checkpoint"),
        "hf_revision": ref.get("hf_revision"),
        "stop_ids": ref.get("stop_ids"),
        "stopped_on_eos": ref.get("stopped_on_eos"),
        "command": ref.get("command"),
        "created": ref.get("created"),
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["prefill-check", "teacher-forcing", "autoregressive"])
    ap.add_argument("--reference", type=Path)
    ap.add_argument("--prompt-file", type=Path)
    ap.add_argument("--output-dir", type=Path)
    ap.add_argument("--max-new-tokens", type=int, default=None)
    ap.add_argument("--hf-model", default=HF_MODEL)
    ap.add_argument("--trace-region", type=int, default=int(os.environ.get("GLM53_TRACE_REGION", 64 * 1024 * 1024)))
    ap.add_argument(
        "--build-kwargs", default="{}", help="JSON kwargs for build_generator (max_seq_len, layer_indices, ...)"
    )
    ap.add_argument("--report", type=Path, help="write the runner's result list as JSON")
    ap.add_argument("--hf-tokens", type=Path, help="autoregressive: precomputed HF ids (tools/hf_full_ref.py)")
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
            hf_ref = patch_hf_reference(a.hf_tokens) if a.hf_tokens else None
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
            if hf_ref is not None:
                annotate_hf_reference(a.output_dir, a.hf_tokens, hf_ref)
                res["hf_reference"] = f"precomputed by tools/hf_full_ref.py ({a.hf_tokens})"
        print("READINESS_RESULT " + json.dumps(res, default=str))
        if a.report:
            a.report.parent.mkdir(parents=True, exist_ok=True)
            a.report.write_text(json.dumps(res, indent=2, default=str) + "\n")
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    sys.exit(main())
