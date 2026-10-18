# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage 8 (datatype sweep) per-candidate driver: ONE full-model build per precision candidate serves

  policy     the precision policy read back from the live model objects (tt/precision.runtime_policy) and the
             candidate's applied overrides: the propagation check
  memory     DRAM allocated / free per bank after build + warm-up + trace capture
  ready      readiness teacher forcing + prefill check (AIME24 chat reference, 100 positions) on the same generator;
             the number of decode-trace replays during teacher forcing is counted (trace verification)
  ttft       warm prefill to device logits and the generator's host-logits path (tools/s7_probe.run_ttft)
  decode     model-trace, sampler-trace, token-out (model + sampler replays, device token feedback, no readback) and
             generate (one token readback per step) ms per step (tools/s7_probe.run_decode)
  ptcheck    non-aligned prompts: traced vs eager prefill logits and greedy continuations (tools/s7_probe.run_ptcheck)
  checks     qualitative suite + autoregressive check (tools/full_model_checks) into --checks-dir
  fidcount   LAST: one eager (untraced) decode step with the precision wrappers' fidelity counters reset
             (tt/precision.FIDELITY_SEEN): which compute fidelity each wrapped decode matmul group used in one step

The policy is GLM53_PRECISION (tt/precision.py): unset = the selected default (doc/datatype_sweep/
selected_precision_config.json), "none" = the stage-7 policy, <json> = a candidate. The record holds the argv, the
GLM53_* / MESH environment, the calling script (S8_SCRIPT), the base commit (S8_BASE_COMMIT or <cwd>/BASE_COMMIT)
and the md5 of every tt/*.py with a digest of that map.

usage:
  GLM53_TP=4 GLM53_CACHE_DIR=~/.cache/glm53_flash GLM53_PRECISION=doc/datatype_sweep/configs/<id>.json \\
  python -m models.autoports.zai_org_glm_5_3_flash.tools.dtype_sweep --id <id> --do policy,ready,ttft,decode \\
      --out doc/datatype_sweep/runs/<id>/record.json --report-dir doc/datatype_sweep/runs/<id>
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import ttnn
from models.autoports.zai_org_glm_5_3_flash.tools import s7_probe as s7

MODEL_DIR = Path(__file__).resolve().parents[1]


def _count_traces():
    """Wrap ttnn.execute_trace with a call counter (the readiness decode loop replays the model trace)."""
    orig = ttnn.execute_trace
    n = {"calls": 0}

    def counted(*a, **kw):
        n["calls"] += 1
        return orig(*a, **kw)

    ttnn.execute_trace = counted
    return n, lambda: setattr(ttnn, "execute_trace", orig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", required=True)
    ap.add_argument("--slots", type=int, default=1)
    ap.add_argument("--seq-len", type=int, default=16384)
    ap.add_argument("--do", default="policy,ready,ttft,decode")
    ap.add_argument("--prompts", default="128,161,2048")
    ap.add_argument("--ptcheck", default="100,161,300")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--reference", default=str(MODEL_DIR / "readiness_aime24_chat.refpt"))
    ap.add_argument("--report-dir", default=None)
    ap.add_argument("--extra-refs", default="", help="comma list of further .refpt references (do: extra)")
    ap.add_argument("--checks-dir", default=None)
    ap.add_argument("--trace-region", type=int, default=int(os.environ.get("GLM53_TRACE_REGION", 64 << 20)))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.environ.setdefault("HF_MODEL", "zai-org/GLM-5.3-Flash")
    from models.autoports.zai_org_glm_5_3_flash.tt.generator import Glm53Generator
    from models.autoports.zai_org_glm_5_3_flash.tt.precision import runtime_policy

    todo = a.do.split(",")
    rec = {
        "id": a.id,
        "args": vars(a),
        "precision_env": os.environ.get("GLM53_PRECISION", ""),
        "env_mc": os.environ.get("GLM53_MC", ""),
        "cache_dir": os.environ.get("GLM53_CACHE_DIR", ""),
        "commit": subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True
        ).stdout.strip(),
        "mesh": os.environ.get("MESH_DEVICE", ""),
        "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "argv": sys.argv,
        "env": {
            k: v for k, v in sorted(os.environ.items()) if k.startswith(("GLM53_", "MESH_", "S8_", "TT_METAL_TRACE"))
        },
        "script": os.environ.get("S8_SCRIPT", ""),
        "cwd": os.getcwd(),
        "base_commit": os.environ.get("S8_BASE_COMMIT")
        or (Path("BASE_COMMIT").read_text().strip() if Path("BASE_COMMIT").exists() else None),
    }
    import hashlib

    import models.autoports.zai_org_glm_5_3_flash.tt as _tt

    tt_dir = Path(_tt.__file__).parent
    rec["code"] = {
        "package": str(tt_dir),
        "md5": {f.name: hashlib.md5(f.read_bytes()).hexdigest()[:12] for f in sorted(tt_dir.glob("*.py"))},
    }
    rec["code"]["md5_map_digest"] = hashlib.sha256(json.dumps(rec["code"]["md5"], sort_keys=True).encode()).hexdigest()[
        :12
    ]
    print(f"S8 CODE {tt_dir} " + json.dumps(rec["code"]), flush=True)
    if os.environ.get("S8_REQUIRE_PKG") and not str(tt_dir).startswith(os.environ["S8_REQUIRE_PKG"]):
        raise SystemExit(f"S8: package {tt_dir} is not under S8_REQUIRE_PKG={os.environ['S8_REQUIRE_PKG']}")
    mesh = s7._open_mesh(a.trace_region)
    try:
        t0 = time.time()
        g = Glm53Generator(mesh, max_batch_size=a.slots, max_seq_len=a.seq_len)
        rec["build_s"] = time.time() - t0
        m = g.model
        from models.autoports.zai_org_glm_5_3_flash.tt.precision import PrecisionOverrides

        rec["precision"] = PrecisionOverrides.from_env().record()
        rec["weight_cache"] = (
            {"root": str(m.weight_cache.root), "hits": m.weight_cache.hits, "stored": m.weight_cache.misses}
            if m.weight_cache is not None
            else None
        )
        v = ttnn.get_memory_view(mesh, ttnn.BufferType.DRAM)
        rec["dram_alloc_per_bank"] = int(v.total_bytes_allocated_per_bank)
        rec["dram_free_per_bank"] = int(v.total_bytes_free_per_bank)
        rec["dram_largest_free_block_per_bank"] = int(getattr(v, "largest_contiguous_bytes_free_per_bank", 0))
        print(f"S8 built {a.id} in {rec['build_s']:.0f}s; DRAM alloc/bank {rec['dram_alloc_per_bank']}", flush=True)
        if "policy" in todo:
            rec["runtime_policy"] = runtime_policy(m)
            print("S8 POLICY " + json.dumps(rec["runtime_policy"]), flush=True)
            print("S8 APPLIED " + json.dumps(rec["precision"]["applied"][:12]), flush=True)
        if "ready" in todo:
            n, undo = _count_traces()
            try:
                s7.run_ready(g, a.reference, a.report_dir, rec)
            finally:
                undo()
            rec["readiness_trace_replays"] = n["calls"]
            # teacher forcing: prefill logits give position 0, then 99 decode steps, each = model + sampler trace
            steps = int(rec["teacher_forcing"][0].get("decode_tokens", 99))
            rec["readiness_trace_replays_expected"] = 2 * steps
            rec["trace_verified"] = n["calls"] == 2 * steps
            print(f"S8 readiness trace replays: {n['calls']}", flush=True)
        if "extra" in todo:
            rec["extra_refs"] = {}
            for ref in [x for x in a.extra_refs.split(",") if x]:
                sub = {}
                s7.run_ready(g, ref, str(Path(a.report_dir) / Path(ref).stem) if a.report_dir else None, sub)
                rec["extra_refs"][Path(ref).stem] = {k: sub[k] for k in ("teacher_forcing", "prefill_check")}
        if "ptcheck" in todo:
            s7.run_ptcheck(g, [int(x) for x in a.ptcheck.split(",")], rec)
        if "checks" in todo:
            from types import SimpleNamespace

            from models.autoports.zai_org_glm_5_3_flash.tools import full_model_checks as fmc

            rc = Path(os.environ["TT_MODEL_BRINGUP_ROOT"]) / "runtime" / "readiness_check"
            ca = SimpleNamespace(
                prompt_file=rc / "autoregressive_prompt.txt",
                hf_tokens=fmc.DOC / "autoregressive" / "hf_tokens.json",
                qual_max_new_tokens=int(os.environ.get("S7_QUAL_MAX_NEW", "2048")),
                skip_autoregressive=False,
                wait_hf_min=0.0,
                skip_qualitative=False,
            )
            rec["checks"] = fmc.run_with(g, mesh, ca, Path(a.checks_dir))
            g.reset()
        if "ttft" in todo:
            s7.run_ttft(g, [int(x) for x in a.prompts.split(",")], rec)
        if "decode" in todo:
            s7.run_decode(g, [1], a.steps, rec)
        if "fidcount" in todo:  # last: an eager step after capture allocates outside the trace's view
            import torch

            from models.autoports.zai_org_glm_5_3_flash.tt import precision as prec_mod

            rec["fidelity_seen_build"] = dict(prec_mod.FIDELITY_SEEN)  # warm-up eager step + trace capture
            prec_mod.FIDELITY_SEEN.clear()
            g.reset()
            g.decode_forward(torch.tensor([[1000]]), torch.tensor([300]), enable_trace=False, on_device_sampling=False)
            rec["fidelity_seen_one_eager_step"] = dict(prec_mod.FIDELITY_SEEN)
            print("S8 FIDELITY one step " + json.dumps(rec["fidelity_seen_one_eager_step"]), flush=True)
            g.reset()
        g.teardown()
    finally:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(rec, indent=2, default=str) + "\n")
        print("S8_RESULT " + json.dumps(rec, default=str)[:4000], flush=True)
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()
