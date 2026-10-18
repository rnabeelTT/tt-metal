# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage 7 (optimized full model) measurement driver: one generator build serves every measurement of a job.

  ttft      warmed prefill of synthetic prompts (--prompts 128,2048,8192): host dispatch time (prefill returns with the
            last-row logits on device) and total time (after a device sync), cold (first) and warm (second) run, plus
            the generator's host-logits path (what readiness times as TTFT).
  decode    per active-slot count (--active 1,4,8): 'model' = back-to-back model trace replays (non-blocking, one sync),
            'sampler' = sampler trace replays, 'tokenout' = model + sampler replays with device token feedback (no host
            work per step), 'generate' = Glm53Generator.decode_forward(reload_inputs=False) per step (token readback each
            step, the readiness / generate loop).
  tracy     signposted windows for tt-perf-report: PERF_DECODE = one model replay + one sampler replay;
            PERF_PREFILL = one warmed prefill of --tracy-prefill rows.
  ready     teacher forcing + prefill check (readiness_check runners) on the SAME generator (no second build).

usage (GLM53_LAYERS=0,3,4 for the reduced model):
  python -m models.autoports.zai_org_glm_5_3_flash.tools.s7_probe --slots 1 --seq-len 16384 --do ttft,decode \
      --prompts 128,2048,8192 --active 1 --out probe.json
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch

import ttnn

try:
    from tracy import signpost
except ImportError:  # noqa: D103

    def signpost(name):
        pass


MODEL_DIR = Path(__file__).resolve().parents[1]


def _open_mesh(trace_region):
    from models.autoports.zai_org_glm_5_3_flash.tools.readiness import _open_mesh as om

    return om(trace_region)


def _sync(mesh):
    ttnn.synchronize_device(mesh)


def _ids(L, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(1000, 100000, (L,), generator=g, dtype=torch.int64)


class _NoPrefillTrace:
    """Context: the eager prefill path of a model that has prefill traces (same build, A/B)."""

    def __init__(self, m, on=True):
        self.m, self.on = m, on

    def __enter__(self):
        self.saved = self.m._ptraces
        if self.on:
            self.m._ptraces = {}

    def __exit__(self, *a):
        self.m._ptraces = self.saved


def run_ttft(g, prompts, rec):
    out = {}
    variants = ["traced", "eager"] if g.model._ptraces else ["eager"]
    for v in variants:
        with _NoPrefillTrace(g.model, v == "eager"):
            out[v] = _ttft(g, prompts)
    rec["moe_split"] = getattr(g.model, "moe_split", None)
    if os.environ.get("S7_MOE_SPLIT_AB"):  # the other MoE row-split setting on the same build (cold + warm runs)
        cur = os.environ.get("GLM53_MOE_SORTED_ROWS", "1024")
        os.environ["GLM53_MOE_SORTED_ROWS"] = "1024" if cur == "0" else "0"
        out["moe_rows_" + os.environ["GLM53_MOE_SORTED_ROWS"]] = _ttft(g, [L for L in prompts if L >= 1024])
        os.environ["GLM53_MOE_SORTED_ROWS"] = cur
    rec["ttft"] = out
    g.reset()


def run_ptcheck(g, prompts, rec, gen=8):
    """Traced vs eager prefill on the same build: last-row logits and greedy continuations (decode replays in between,
    so a prefill replay after decode replays is covered)."""
    m = g.model
    out = {}
    for L in prompts:
        ids = _ids(L, seed=L + 7)
        res = {}
        for v in ("traced", "eager", "eager2", "traced2"):
            with _NoPrefillTrace(m, v.startswith("eager")):
                g.reset()
                lg = g.prefill_forward(ids.reshape(1, -1), prompt_lens=[L])[0, -1, : m.vocab]
                toks = g.generate(ids.tolist(), gen)
                res[v] = (lg, toks)
        (lt, tt), (le, te) = res["traced"], res["eager"]
        out[L] = {
            "max_abs_diff": float((lt - le).abs().max()),
            "argmax": [int(lt.argmax()), int(le.argmax())],
            "tokens_traced": tt,
            "tokens_eager": te,
            "tokens_equal": tt == te,
            "tokens_eager2": res["eager2"][1],
            "tokens_traced2": res["traced2"][1],
            "eager_repeat_equal": res["eager2"][1] == te,
            "traced_repeat_equal": res["traced2"][1] == tt,
        }
        print(f"S7 PTCHECK L={L}: {json.dumps(out[L])}", flush=True)
    rec["ptcheck"] = out
    g.reset()


def _ttft(g, prompts):
    m = g.model
    out = {}
    for L in prompts:
        if L > g.max_seq_len:
            continue
        r = {"plan": m.chunk_plan(L)}
        for run in ("cold", "warm"):
            g.reset()
            ids = _ids(L, seed=L)
            _sync(g.mesh)
            t0 = time.perf_counter()
            lg = m.prefill(ids, 0, device_logits=True)
            t1 = time.perf_counter()
            _sync(g.mesh)
            t2 = time.perf_counter()
            ttnn.deallocate(lg)
            r[run] = {"dispatch_ms": (t1 - t0) * 1e3, "total_ms": (t2 - t0) * 1e3}
        g.reset()
        ids = _ids(L, seed=L)
        t0 = time.perf_counter()
        logits = g.prefill_forward(ids.reshape(1, -1), prompt_lens=[L])
        r["host_logits_ms"] = (time.perf_counter() - t0) * 1e3
        r["first_token"] = int(torch.argmax(logits[0, -1, : m.vocab]))
        out[L] = r
        print(f"S7 TTFT L={L} traced={sorted(m._ptraces)}: {json.dumps(r)}", flush=True)
    return out


def _set_active(g, A, p0, tok=1000):
    S = g.max_batch_size
    pos = [p0 if s < A else -1 for s in range(S)]
    g.model.inputs.set_positions(pos)
    g._write_tokens([tok] * S)
    _sync(g.mesh)


def run_decode(g, actives, steps, rec, p0=300):
    m, mesh = g.model, g.mesh
    sm = m.sampling
    out = {}
    for A in actives:
        if A > g.max_batch_size:
            continue
        r = {}
        # warm one of each
        _set_active(g, A, p0)
        ttnn.execute_trace(mesh, g.trace_id, cq_id=0, blocking=False)
        sm.sample(g.logits_out, tt_out_tok=g.tok_buf, enable_trace=True)
        _sync(mesh)

        _set_active(g, A, p0)
        t0 = time.perf_counter()
        for _ in range(steps):
            ttnn.execute_trace(mesh, g.trace_id, cq_id=0, blocking=False)
        t1 = time.perf_counter()
        _sync(mesh)
        r["model_ms"] = (time.perf_counter() - t0) * 1e3 / steps
        r["model_enqueue_ms"] = (t1 - t0) * 1e3 / steps

        t0 = time.perf_counter()
        for _ in range(steps):
            sm.sample(g.logits_out, tt_out_tok=g.tok_buf, enable_trace=True)
        _sync(mesh)
        r["sampler_ms"] = (time.perf_counter() - t0) * 1e3 / steps

        _set_active(g, A, p0)
        t0 = time.perf_counter()
        for _ in range(steps):
            ttnn.execute_trace(mesh, g.trace_id, cq_id=0, blocking=False)
            sm.sample(g.logits_out, tt_out_tok=g.tok_buf, enable_trace=True)
        _sync(mesh)
        r["tokenout_ms"] = (time.perf_counter() - t0) * 1e3 / steps

        _set_active(g, A, p0)
        g.positions = list(m.inputs.positions)
        tok = torch.full((A, 1), 1000)
        pos = torch.full((A,), p0)
        g.decode_forward(tok, pos, reload_inputs=True, slots=list(range(A)))
        t0 = time.perf_counter()
        for i in range(steps):
            g.decode_forward(tok, pos + 1 + i, reload_inputs=False, slots=list(range(A)))
        r["generate_ms"] = (time.perf_counter() - t0) * 1e3 / steps
        r["generate_tsu"] = 1e3 / r["generate_ms"]
        out[A] = r
        print(f"S7 DECODE S={g.max_batch_size} active={A}: {json.dumps(r)}", flush=True)
    rec["decode"] = out
    g.reset()


def _flush_profiler(mesh):
    """Drain the device profiler buffers (they hold a few hundred ops: the build and warm-up overflow them)."""
    if os.environ.get("TT_METAL_DEVICE_PROFILER") == "1" and hasattr(ttnn, "ReadDeviceProfiler"):
        ttnn.synchronize_device(mesh)
        ttnn.ReadDeviceProfiler(mesh)


def run_tracy(g, prefill_rows, rec):
    m, mesh = g.model, g.mesh
    _set_active(g, g.max_batch_size, 300)
    for _ in range(2):  # warm
        ttnn.execute_trace(mesh, g.trace_id, cq_id=0, blocking=False)
        m.sampling.sample(g.logits_out, tt_out_tok=g.tok_buf, enable_trace=True)
    _flush_profiler(mesh)
    signpost("PERF_DECODE")
    t0 = time.perf_counter()
    ttnn.execute_trace(mesh, g.trace_id, cq_id=0, blocking=False)
    m.sampling.sample(g.logits_out, tt_out_tok=g.tok_buf, enable_trace=True)
    _sync(mesh)
    dt = time.perf_counter() - t0
    signpost("PERF_DECODE_END")
    _flush_profiler(mesh)
    print(f"PERF decode step (model + sampler replay): {dt * 1e3:.3f} ms host wall", flush=True)
    rec["tracy_decode_ms"] = dt * 1e3
    g.reset()
    if prefill_rows:
        ids = _ids(prefill_rows, seed=1)
        ttnn.deallocate(m.prefill(ids, 0, device_logits=True))
        g.reset()
        _flush_profiler(mesh)
        signpost("PERF_PREFILL")
        t0 = time.perf_counter()
        lg = m.prefill(ids, 0, device_logits=True)
        t1 = time.perf_counter()
        _sync(mesh)
        dt = time.perf_counter() - t0
        signpost("PERF_PREFILL_END")
        _flush_profiler(mesh)
        ttnn.deallocate(lg)
        print(f"PERF prefill {prefill_rows} rows: {dt * 1e3:.2f} ms host wall (dispatch {(t1 - t0) * 1e3:.2f} ms)")
        rec["tracy_prefill_ms"] = dt * 1e3
        g.reset()


class _Shared:
    """The generator handed to the readiness runners: teardown is deferred to this driver."""

    def __init__(self, g):
        self._g = g

    def __getattr__(self, k):
        return getattr(self._g, k)

    def teardown(self):
        pass


def run_ready(g, reference, report_dir, rec):
    import readiness_check.run_prefill_check as rpc
    import readiness_check.run_teacher_forcing as rtf

    shared = _Shared(g)
    for mod in (rtf, rpc):
        mod._import_build_generator = lambda model_dir: (lambda **kw: shared)
    res = rtf.run_teacher_forcing(model_dir=MODEL_DIR, reference_path=Path(reference).resolve(), mesh_device=g.mesh)
    print("READINESS_RESULT teacher-forcing " + json.dumps(res, default=str), flush=True)
    rec["teacher_forcing"] = res
    g.reset()
    res2 = rpc.run_prefill_check(model_dir=MODEL_DIR, reference_path=Path(reference).resolve(), mesh_device=g.mesh)
    print("READINESS_RESULT prefill-check " + json.dumps(res2, default=str), flush=True)
    rec["prefill_check"] = res2
    if report_dir:
        d = Path(report_dir)
        d.mkdir(parents=True, exist_ok=True)
        (d / "teacher_forcing.json").write_text(json.dumps(res, indent=2, default=str) + "\n")
        (d / "prefill_check.json").write_text(json.dumps(res2, indent=2, default=str) + "\n")
    g.reset()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slots", type=int, default=1)
    ap.add_argument("--seq-len", type=int, default=4096)
    ap.add_argument("--do", default="ttft,decode")
    ap.add_argument("--prompts", default="128,2048")
    ap.add_argument("--active", default="1")
    ap.add_argument("--ptcheck", default="100,128,161")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--tracy-prefill", type=int, default=128)
    ap.add_argument("--reference", default=str(MODEL_DIR / "readiness_aime24_chat.refpt"))
    ap.add_argument("--report-dir", default=None)
    ap.add_argument("--trace-region", type=int, default=int(os.environ.get("GLM53_TRACE_REGION", 64 << 20)))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    os.environ.setdefault("HF_MODEL", "zai-org/GLM-5.3-Flash")
    from models.autoports.zai_org_glm_5_3_flash.tt.generator import Glm53Generator

    todo = a.do.split(",")
    rec = {"args": vars(a), "layers": os.environ.get("GLM53_LAYERS", "all"), "env_mc": os.environ.get("GLM53_MC", "")}
    mesh = _open_mesh(a.trace_region)
    try:
        t0 = time.time()
        g = Glm53Generator(mesh, max_batch_size=a.slots, max_seq_len=a.seq_len)
        rec["build_s"] = time.time() - t0
        try:
            tv = ttnn.get_memory_view(mesh, ttnn.BufferType.TRACE)
            rec["trace_alloc_per_bank"] = int(tv.total_bytes_allocated_per_bank)
            rec["trace_total_per_bank"] = int(tv.total_bytes_per_bank)
        except Exception as e:  # noqa: BLE001
            rec["trace_view_error"] = str(e)
        print(f"S7 built S={a.slots} seq={a.seq_len} in {rec['build_s']:.0f}s", flush=True)
        try:
            v = ttnn.get_memory_view(mesh, ttnn.BufferType.DRAM)
            rec["dram_free_per_bank"] = int(v.total_bytes_free_per_bank)
            rec["dram_alloc_per_bank"] = int(v.total_bytes_allocated_per_bank)
        except Exception as e:  # noqa: BLE001
            rec["dram_view_error"] = str(e)
        if "ready" in todo:
            run_ready(g, a.reference, a.report_dir, rec)
        if "ptcheck" in todo:
            run_ptcheck(g, [int(x) for x in a.ptcheck.split(",")], rec)
        if "checks" in todo:  # qualitative suite + autoregressive (tools/full_model_checks.run_with) on this build
            from types import SimpleNamespace

            from models.autoports.zai_org_glm_5_3_flash.tools import full_model_checks as fmc

            rc = Path(os.environ["TT_MODEL_BRINGUP_ROOT"]) / "runtime" / "readiness_check"
            ca = SimpleNamespace(
                prompt_file=rc / "autoregressive_prompt.txt",
                hf_tokens=fmc.DOC / "autoregressive" / "hf_tokens.json",
                qual_max_new_tokens=int(os.environ.get("S7_QUAL_MAX_NEW", "2048")),
                skip_autoregressive=False,
                wait_hf_min=0.0,
                skip_qualitative=os.environ.get("S7_SKIP_QUAL") == "1",
            )
            rec["checks"] = fmc.run_with(g, mesh, ca, MODEL_DIR / "doc" / "optimized_full_model")
            g.reset()
        if "ttft" in todo:
            run_ttft(g, [int(x) for x in a.prompts.split(",")], rec)
        if "decode" in todo:
            run_decode(g, [int(x) for x in a.active.split(",")], a.steps, rec)
        if "tracy" in todo:
            run_tracy(g, a.tracy_prefill, rec)
        g.teardown()
    finally:
        if a.out:
            Path(a.out).parent.mkdir(parents=True, exist_ok=True)
            Path(a.out).write_text(json.dumps(rec, indent=2, default=str) + "\n")
        print("S7_RESULT " + json.dumps(rec, default=str), flush=True)
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()
