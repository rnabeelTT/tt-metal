# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage-6 device DRAM / trace-region probe of the full-model wrapper (tt/model.py, tt/generator.py).

Every number is read from the mesh allocator (ttnn.get_memory_view: DRAM / TRACE bytes per bank x banks = per die; the
mesh allocates in lockstep, so one view is every die). Peaks are sampled after every top-level ttnn op through a
post-operation hook (transients inside one op's C++ body are not seen; op outputs are).

  reduced   GLM53_LAYERS=0,3,4 (one layer of each kind: KDA+dense, DSA+MoE, KDA+MoE). Construction bytes per layer
            (weights / decode state / constants), prefill peak per layer kind and chunk path (flash / dense / sparse)
            at 2048- and 1024-row chunks with the MoE row-block cap GLM53_MOE_SORTED_ROWS in {0, 1024, 512} (time +
            bit-identity of the last logits and of one MoE layer's output), decode-step workspace, decode trace bytes per
            layer kind, model trace and sampler trace bytes.
  capacity  the full model on the vLLM serving path (Glm53Generator(defer_caches=True), 1 slot by default): weights
            loaded -> free DRAM measured -> the DSA block pool sized from it (minus --workspace-mib) -> bind_kv_cache ->
            capture_traces (model + sampler variants) -> one prompt of length near the pool capacity (non-aligned) ->
            traced greedy decode at the top positions. The prompt is a natural-text passage repeated: the greedy
            continuation must copy the passage (induction over the whole context).

usage:
  GLM53_TP=4 GLM53_LAYERS=0,3,4 python -m models.autoports.zai_org_glm_5_3_flash.tools.memory_probe reduced --out DIR
  GLM53_TP=4 python -m models.autoports.zai_org_glm_5_3_flash.tools.memory_probe capacity --out DIR --trace-region N
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch

import ttnn

MiB = 1 << 20
L1_SMALL_SIZE = 24576


class Dram:
    """Per-die DRAM / trace accounting and the per-op peak sampler."""

    def __init__(self, mesh):
        self.mesh = mesh
        v = ttnn.get_memory_view(mesh, ttnn.BufferType.DRAM)
        self.banks = int(v.num_banks)
        self.bank_size = int(v.total_bytes_per_bank)
        self.on = False
        self.peak = 0
        self.peak_op = None
        self.min_largest_free = None
        self.samples = 0
        self.sample_seconds = 0.0

    def view(self, bt=ttnn.BufferType.DRAM):
        return ttnn.get_memory_view(self.mesh, bt)

    def alloc(self) -> int:
        return int(self.view().total_bytes_allocated_per_bank) * self.banks

    def largest_free(self) -> int:
        return int(self.view().largest_contiguous_bytes_free_per_bank) * self.banks

    def total(self) -> int:
        return self.bank_size * self.banks

    def trace(self) -> int:
        v = self.view(ttnn.BufferType.TRACE)
        return int(v.total_bytes_allocated_per_bank) * int(v.num_banks)

    def trace_total(self) -> int:
        v = self.view(ttnn.BufferType.TRACE)
        return int(v.total_bytes_per_bank) * int(v.num_banks)

    def snapshot(self) -> dict:
        v = self.view()
        return {
            "allocated": int(v.total_bytes_allocated_per_bank) * self.banks,
            "free": int(v.total_bytes_free_per_bank) * self.banks,
            "largest_free_block_per_bank": int(v.largest_contiguous_bytes_free_per_bank),
            "trace_allocated": self.trace(),
        }

    def hook(self, op, args, kwargs, out):
        if not self.on:
            return None
        t0 = time.perf_counter()
        v = self.view()
        a = int(v.total_bytes_allocated_per_bank) * self.banks
        lf = int(v.largest_contiguous_bytes_free_per_bank)
        self.samples += 1
        if a > self.peak:
            self.peak = a
            self.peak_op = getattr(op, "python_fully_qualified_name", None) or str(op)
        if self.min_largest_free is None or lf < self.min_largest_free:
            self.min_largest_free = lf
        self.sample_seconds += time.perf_counter() - t0
        return None

    def start(self):
        # the operation hooks run only on ttnn's slow (validating) dispatch path: fast runtime mode off while sampling
        self._fast = ttnn.CONFIG.enable_fast_runtime_mode
        ttnn.CONFIG.enable_fast_runtime_mode = False
        self.peak = self.alloc()
        self.peak_op = None
        self.min_largest_free = None
        self.on = True
        return self.peak

    def stop(self) -> int:
        self.on = False
        ttnn.CONFIG.enable_fast_runtime_mode = self._fast
        return self.peak


def open_mesh(trace_region: int):
    tp = int(os.environ.get("GLM53_TP", "4"))
    if tp > 1:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    return ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(1, tp),
        l1_small_size=L1_SMALL_SIZE,
        trace_region_size=trace_region,
        num_command_queues=1,
    )


def instrument_construction(dram: Dram, rec: dict):
    """Allocation deltas of every layer's construction and decode-state allocation (attention state separately)."""
    from models.autoports.zai_org_glm_5_3_flash.tt import dsa as dsa_mod
    from models.autoports.zai_org_glm_5_3_flash.tt import kda as kda_mod
    from models.autoports.zai_org_glm_5_3_flash.tt.multichip_decoder import MultichipDecoder

    orig_fsd = MultichipDecoder.from_state_dict.__func__
    orig_ads = MultichipDecoder.allocate_decode_state

    def fsd(cls, *a, **kw):
        a0 = dram.alloc()
        dec = orig_fsd(cls, *a, **kw)
        r = rec.setdefault(kw["layer_idx"], {"kind": f"{dec.attn_kind}+{dec.mlp_kind}"})
        r["construct_bytes"] = dram.alloc() - a0
        dec._probe_idx = kw["layer_idx"]
        return dec

    def ads(self, batch=None):
        a0 = dram.alloc()
        rec[self._probe_idx]["_attn_state"] = 0
        rec["_current"] = self._probe_idx
        out = orig_ads(self, batch)
        r = rec[self._probe_idx]
        r["decode_state_total_bytes"] = r.get("decode_state_total_bytes", 0) + dram.alloc() - a0
        r["attn_state_bytes"] = r.pop("_attn_state")
        return out

    MultichipDecoder.from_state_dict = classmethod(fsd)
    MultichipDecoder.allocate_decode_state = ads

    for mod, cls_name in ((kda_mod, "GlmKDA"), (dsa_mod, "GlmDSA")):
        cls = getattr(mod, cls_name)
        orig = cls.allocate_decode_state

        def wrap(self, *a, _orig=orig, **kw):
            a0 = dram.alloc()
            out = _orig(self, *a, **kw)
            cur = rec.get("_current")
            if cur is not None and "_attn_state" in rec[cur]:
                rec[cur]["_attn_state"] += dram.alloc() - a0
            return out

        cls.allocate_decode_state = wrap


def track_layers(model, dram: Dram, rows: list, enabled=lambda chunk: True):
    """Wrap every layer's prefill_forward: entry bytes, peak bytes, wall time (synchronized) per (layer, chunk)."""
    mesh = model.mesh
    state = {"chunk": -1}
    for li, d in enumerate(model.layers):
        orig = d.prefill_forward

        def pf(X, *, inputs, attn_state=None, _orig=orig, _li=li, _d=d):
            if _li == 0:
                state["chunk"] += 1
            on = enabled(state["chunk"])
            if not on:
                return _orig(X, inputs=inputs, attn_state=attn_state)
            ttnn.synchronize_device(mesh)
            entry = dram.start()
            t0 = time.perf_counter()
            out = _orig(X, inputs=inputs, attn_state=attn_state)
            ttnn.synchronize_device(mesh)
            dt = time.perf_counter() - t0
            peak = dram.stop()
            a = inputs["attn"]
            path = a.get("path", "kda") if isinstance(a, dict) else "kda"
            rows.append(
                {
                    "layer": int(getattr(_d, "_probe_idx", _li)),
                    "kind": f"{_d.attn_kind}+{_d.mlp_kind}",
                    "chunk": state["chunk"],
                    "rows": int(X.shape[-2]),
                    "dsa_path": path,
                    "entry_bytes": entry,
                    "peak_bytes": peak,
                    "peak_over_entry_bytes": peak - entry,
                    "peak_op": dram.peak_op,
                    "min_largest_free_block_per_bank": dram.min_largest_free,
                    "seconds": round(dt, 4),
                }
            )
            return out

        d.prefill_forward = pf
    return state


def summarize_peaks(rows):
    out = {}
    for r in rows:
        key = f"{r['kind']}/{r['rows']}/{r['dsa_path']}"
        o = out.setdefault(key, {"peak_over_entry_bytes": 0, "seconds": []})
        if r["peak_over_entry_bytes"] >= o["peak_over_entry_bytes"]:
            o["peak_over_entry_bytes"] = r["peak_over_entry_bytes"]
            o["peak_op"] = r["peak_op"]
        o["seconds"].append(r["seconds"])
    for o in out.values():
        o["peak_over_entry_MiB"] = round(o["peak_over_entry_bytes"] / MiB, 1)
        o["seconds_min"] = min(o["seconds"])
    return out


def _tok_buf(mesh, S):
    return ttnn.from_torch(
        torch.zeros(1, 1, 1, S, dtype=torch.int32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


# ------------------------------------------------------------------------------------------------------------ reduced
def run_reduced(a):
    from models.autoports.zai_org_glm_5_3_flash.tt.model import model_from_env

    report = {"mode": "reduced", "layers": os.environ.get("GLM53_LAYERS"), "S": a.slots, "max_context": a.max_context}
    mesh = open_mesh(a.trace_region)
    dram = Dram(mesh)
    report["bank_size"] = dram.bank_size
    report["banks"] = dram.banks
    report["die_allocatable_bytes"] = dram.total()
    report["trace_region_bytes"] = dram.trace_total()
    rec = {}
    instrument_construction(dram, rec)
    a_open = dram.alloc()
    report["after_open"] = dram.snapshot()
    hook_cm = ttnn.register_post_operation_hook(dram.hook)
    hook_cm.__enter__()
    try:
        S, C = a.slots, a.max_context
        model = model_from_env(mesh, max_batch=S, max_context=C, prefill_lengths=(128, 1024, 2048), weight_cache=None)
        report["after_build"] = dram.snapshot()
        rec.pop("_current", None)
        report["layers_construct"] = {str(k): v for k, v in rec.items()}
        report["model_level_bytes"] = (
            report["after_build"]["allocated"]
            - a_open
            - sum(v.get("construct_bytes", 0) + v.get("decode_state_total_bytes", 0) for v in rec.values())
        )
        dsa = next((d.attn for d in model.layers if d.attn_kind == "dsa"), None)
        pt = dsa.default_page_table()[:, : C // 128] if dsa is not None else None
        if pt is not None:
            model.set_page_table(pt)
        report["dsa_NB"] = dsa.NB if dsa is not None else None
        report["dsa_P_alloc"] = dsa.P_alloc if dsa is not None else None
        print(json.dumps({k: report[k] for k in ("after_build", "model_level_bytes")}), flush=True)

        # ---- prefill peaks / time / bit identity
        g = torch.Generator().manual_seed(7)
        L = a.prompt_len
        ids = torch.randint(100, 150000, (L,), generator=g)
        rows = []
        state = track_layers(model, dram, rows)
        runs = {}
        configs = [  # (tag, bucket, GLM53_MOE_SORTED_ROWS, run name, GLM53_DSA_INDEX_BLOCK_MB)
            ("2048", 2048, "1024", "warm", "32"),
            ("2048", 2048, "1024", "s1024_b128", "128"),
            ("2048", 2048, "1024", "s1024_b32", "32"),
            ("2048", 2048, "1024", "s1024_b16", "16"),
            ("2048", 2048, "0", "s0_b128", "128"),
            ("2048", 2048, "512", "s512_b32", "32"),
            ("1024", 1024, "1024", "c1024_b32", "32"),
            ("1024", 1024, "0", "c1024_s0_b128", "128"),
        ]
        logits = {}
        for tag, bucket, sub, name, blk in configs:
            os.environ["GLM53_MOE_SORTED_ROWS"] = sub
            os.environ["GLM53_DSA_INDEX_BLOCK_MB"] = blk
            model.prefill_lengths = (128, 1024, 2048) if bucket == 2048 else (128, 1024)
            model.reset_state()
            if pt is not None:
                model.set_page_table(pt)
            n0 = len(rows)
            state["chunk"] = -1
            ttnn.synchronize_device(mesh)
            t0 = time.perf_counter()
            lg = model.prefill(ids, 0)
            ttnn.synchronize_device(mesh)
            wall = time.perf_counter() - t0
            logits[name] = lg.clone()
            mine = rows[n0:]
            for r in mine:
                r["run"] = name
            runs[name] = {
                "bucket": bucket,
                "moe_sorted_rows": int(sub),
                "dsa_index_block_mb": int(blk),
                "wall_seconds": round(wall, 3),
                "peak_over_entry_bytes_max": max(r["peak_over_entry_bytes"] for r in mine),
                "peaks": summarize_peaks(mine),
            }
            print(name, json.dumps(runs[name]["peaks"]), flush=True)
        model.prefill_lengths = (128, 1024, 2048)
        ref = logits["s0_b128"]  # the pre-change configuration
        report["bit_identity_last_logits_vs_s0_b128"] = {
            n: bool(torch.equal(logits[n], ref)) for n in logits if n.startswith("s") or n == "warm"
        }
        report["bit_identity_last_logits_c1024_vs_c1024_s0_b128"] = bool(
            torch.equal(logits["c1024_b32"], logits["c1024_s0_b128"])
        )
        print(
            json.dumps(report["bit_identity_last_logits_vs_s0_b128"]),
            report["bit_identity_last_logits_c1024_vs_c1024_s0_b128"],
            flush=True,
        )
        report["prefill_runs"] = runs
        report["prefill_rows"] = rows
        for d in model.layers:  # unwrap
            d.__dict__.pop("prefill_forward", None)

        # ---- one MoE layer's output, direct (same input, three row-block caps)
        moe_d = next((d for d in model.layers if d.mlp_kind == "moe"), None)
        if moe_d is not None:
            x = ttnn.from_torch(
                torch.randn(1, 1, 2048, model.hidden, generator=g).to(torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            )
            outs, moe_peak = {}, {}
            for sub in ("0", "1024", "512", "0"):
                os.environ["GLM53_MOE_SORTED_ROWS"] = sub
                ttnn.synchronize_device(mesh)
                e = dram.start()
                t0 = time.perf_counter()
                y = moe_d.mlp.forward(x, mode="prefill")
                ttnn.synchronize_device(mesh)
                dt = time.perf_counter() - t0
                moe_peak[sub] = {
                    "peak_over_entry_bytes": dram.stop() - e,
                    "seconds": round(dt, 4),
                    "peak_op": dram.peak_op,
                }
                outs[sub] = ttnn.to_torch(y, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=-1))
                ttnn.deallocate(y)
            ttnn.deallocate(x)
            report["moe_layer_direct"] = {
                "layer": int(moe_d._probe_idx),
                "peaks": moe_peak,
                "bit_identical_1024": bool(torch.equal(outs["1024"], outs["0"])),
                "bit_identical_512": bool(torch.equal(outs["512"], outs["0"])),
            }
            print("moe direct", json.dumps(report["moe_layer_direct"]), flush=True)
        os.environ["GLM53_MOE_SORTED_ROWS"] = str(a.moe_rows)

        # ---- decode: eager step workspace, per-layer trace bytes, model + sampler trace bytes
        model.reset_state()
        if pt is not None:
            model.set_page_table(pt)
        model.inputs.set_positions([-1] * S)
        tok_buf = _tok_buf(mesh, S)
        view = ttnn.reshape(tok_buf, (1, S))
        ttnn.synchronize_device(mesh)
        e = dram.start()
        lg = model.decode_step(view)
        ttnn.synchronize_device(mesh)
        report["decode_step_eager"] = {"peak_over_entry_bytes": dram.stop() - e, "peak_op": dram.peak_op}
        sm = model.sampling
        if sm is not None:
            from models.common.sampling.generator import format_sampling_params
            from models.common.sampling.sampling_params import SamplingParams

            sm.reset_sampling_params(format_sampling_params(SamplingParams(temperature=0.0, top_k=1, top_p=1.0), S))
            sm.precompile(lg, tt_out_tok=tok_buf, all_configs=False)
        ttnn.deallocate(lg)
        ttnn.synchronize_device(mesh)
        report["after_decode_warm"] = dram.snapshot()

        per_layer = {}
        model.inputs.refresh()
        X = model.embed(view)
        ttnn.synchronize_device(mesh)
        for d in model.layers:
            bufs = model.inputs.kda_bufs if d.attn_kind == "kda" else model.inputs.dsa_bufs
            Y = d.decode_forward(X, inputs=bufs)  # eager warm of this exact call
            ttnn.deallocate(Y)
            ttnn.synchronize_device(mesh)
            t0 = dram.trace()
            tid = ttnn.begin_trace_capture(mesh, cq_id=0)
            try:
                Y = d.decode_forward(X, inputs=bufs)
            finally:
                ttnn.end_trace_capture(mesh, tid, cq_id=0)
            tb = dram.trace() - t0
            ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
            ttnn.release_trace(mesh, tid)
            ttnn.deallocate(Y)
            per_layer[str(d._probe_idx)] = {"kind": f"{d.attn_kind}+{d.mlp_kind}", "trace_bytes": tb}
        ttnn.deallocate(X)
        report["decode_trace_per_layer"] = per_layer
        ttnn.synchronize_device(mesh)
        t0 = dram.trace()
        a0 = dram.alloc()
        tid = ttnn.begin_trace_capture(mesh, cq_id=0)
        try:
            logits_out = model.decode_step(view)
        finally:
            ttnn.end_trace_capture(mesh, tid, cq_id=0)
        report["model_trace_bytes"] = dram.trace() - t0
        report["model_trace_persistent_dram_bytes"] = dram.alloc() - a0  # the trace's output buffers
        if sm is not None:  # the greedy sampler trace (Glm53Generator._warm's capture)
            t1 = dram.trace()
            sm.sample(logits_out, tt_out_tok=tok_buf, enable_trace=True, skip_precompile=True)
            report["sampler_trace_bytes_total"] = dram.trace() - t1
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
        ttnn.synchronize_device(mesh)
        report["after_traces"] = dram.snapshot()
        nl = len(model.layers)
        report["model_trace_minus_layers_bytes"] = report["model_trace_bytes"] - sum(
            v["trace_bytes"] for v in per_layer.values()
        )
        report["hook_samples"] = dram.samples
        report["hook_seconds"] = round(dram.sample_seconds, 2)
        print(
            json.dumps({k: report[k] for k in report if "trace" in k or k.startswith("decode")}, indent=1), flush=True
        )
    finally:
        hook_cm.__exit__(None, None, None)
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "reduced.json").write_text(json.dumps(report, indent=1, default=str))
        from models.autoports.zai_org_glm_5_3_flash.tests.harness import close_mesh

        close_mesh(mesh)


# ----------------------------------------------------------------------------------------------------------- capacity
PASSAGE_FILES = ("LICENSE", "README.md")


def copy_prompt(tok, L: int):
    """A natural-text passage (tokenized) repeated to L tokens; returns (ids [L], passage ids, period)."""
    root = Path(os.environ.get("TT_METAL_HOME", "."))
    text = ""
    for f in PASSAGE_FILES:
        p = root / f
        if p.exists():
            text += p.read_text(errors="ignore")
        if len(text) > 12000:
            break
    pas = tok.encode(text[:12000], add_special_tokens=False)[:1500]
    reps = -(-L // len(pas))
    ids = torch.tensor((pas * reps)[:L], dtype=torch.int64)
    return ids, pas


def run_capacity(a):
    from models.autoports.zai_org_glm_5_3_flash.tt.generator import Glm53Generator

    report = {"mode": "capacity", "S": a.slots, "max_seq_len": a.max_context, "workspace_mib": a.workspace_mib}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    def save():
        (out / "capacity.json").write_text(json.dumps(report, indent=1, default=str))

    mesh = open_mesh(a.trace_region)
    dram = Dram(mesh)
    report["bank_size"] = dram.bank_size
    report["die_allocatable_bytes"] = dram.total()
    report["trace_region_bytes"] = dram.trace_total()
    report["after_open"] = dram.snapshot()
    hook_cm = ttnn.register_post_operation_hook(dram.hook)
    hook_cm.__enter__()
    try:
        t0 = time.time()
        g = Glm53Generator(
            mesh, max_batch_size=a.slots, max_seq_len=a.max_context, defer_caches=True, weight_cache=None
        )
        m = g.model
        report["load_seconds"] = round(time.time() - t0, 1)
        report["after_load"] = dram.snapshot()
        save()
        dsa_layers = [d for d in m.layers if d.attn_kind == "dsa"]
        d0 = dsa_layers[0].attn
        # replicated: latent 128 x 512 bf16 + 32 pool keys x 128 bf16 per layer; block-sharded (GLM53_DSA_SHARD=1 /
        # MCOpts.dsa_shard, tt/dsa_shard.py): the 544-wide latent row over the tp dies + the replicated pool keys
        from models.autoports.zai_org_glm_5_3_flash.tt.dsa_shard import dsa_bytes_per_token

        per_block = int(round(128 * dsa_bytes_per_token([d.attn for d in dsa_layers])))
        report["dsa_shard"] = bool(getattr(d0, "dsa_shard", False))
        free = dram.total() - dram.alloc()
        ws = int(a.workspace_mib * MiB)
        nb_fit = int((free - ws) // per_block)
        nb_cap = a.slots * (a.max_context // 128) + 1  # vLLM's null block 0 + every slot at max_seq_len
        nb = max(2, min(nb_fit, nb_cap))
        report.update(
            {
                "free_after_load_bytes": free,
                "dsa_bytes_per_block_all_layers": per_block,
                "num_blocks_fit": nb_fit,
                "num_blocks": nb,
                "capped_by_max_seq_len": nb_fit > nb_cap,
            }
        )
        print(json.dumps(report, default=str), flush=True)
        save()
        a0 = dram.alloc()
        g.bind_kv_cache(nb)
        report["dsa_pool_bytes_measured"] = dram.alloc() - a0
        report["after_bind"] = dram.snapshot()
        save()
        ttnn.synchronize_device(mesh)
        dram.start()
        t1 = time.time()
        # warm_programs + the model trace + the GREEDY sampler trace only (Glm53Generator._warm's capture). The serving
        # capture_traces also records the penalties / force-argmax sampler variants; capturing the penalties variant
        # after precompile hit "Cannot load new binaries during trace capture" in the reduced probe (2026-10-17), so
        # this probe does not risk the full-model run on it (variant bytes: reduced.json sampler trace x 4).
        g.warm_programs()
        ttnn.synchronize_device(mesh)
        g.trace_id = ttnn.begin_trace_capture(mesh, cq_id=0)
        try:
            g.logits_out = m.decode_step(g._tok_view())
        finally:
            ttnn.end_trace_capture(mesh, g.trace_id, cq_id=0)
        report["model_trace_bytes"] = dram.trace()
        if m.sampling is not None:
            m.sampling.sample(g.logits_out, tt_out_tok=g.tok_buf, enable_trace=True, skip_precompile=True)
            report["sampler_trace_bytes"] = dram.trace() - report["model_trace_bytes"]
        ttnn.synchronize_device(mesh)
        report["warm_and_capture_peak_bytes"] = dram.stop()
        report["warm_and_capture_peak_op"] = dram.peak_op
        report["warm_and_capture_seconds"] = round(time.time() - t1, 1)
        report["after_capture"] = dram.snapshot()
        report["trace_bytes_model_plus_greedy_sampler"] = dram.trace()
        print(
            json.dumps(
                {k: report[k] for k in ("after_bind", "after_capture", "trace_bytes_model_plus_greedy_sampler")}
            ),
            flush=True,
        )
        save()

        # ---- long prompt near the pool capacity
        n_dec = a.decode_steps
        tokens_cap = min(a.max_context, (nb - 1) * 128)  # block 0 is vLLM's null block
        L = tokens_cap - n_dec - a.tail_gap
        ids, pas = copy_prompt(g.tokenizer, L)
        nblk = -(-(L + n_dec + 1) // 128)
        assert nblk <= nb - 1, (nblk, nb)
        pt = torch.arange(1, nblk + 1, dtype=torch.int32).reshape(1, -1)
        if a.slots > 1:
            pt = torch.cat([pt, torch.zeros(a.slots - 1, nblk, dtype=torch.int32)], dim=0)
        nchunks = -(-L // 2048)
        rows = []
        track_layers(m, dram, rows, enabled=lambda c: c in (0, 1, nchunks - 2, nchunks - 1))
        report["prompt_len"] = L
        report["passage_tokens"] = len(pas)
        save()
        ttnn.synchronize_device(mesh)
        t2 = time.time()
        logits = g.prefill_forward(ids.reshape(1, -1), page_table=pt, prompt_lens=[L])
        report["prefill_seconds"] = round(time.time() - t2, 1)
        report["prefill_tracked_rows"] = rows
        report["prefill_peaks"] = summarize_peaks(rows)
        report["after_prefill"] = dram.snapshot()
        save()
        first = int(torch.argmax(logits[0, -1, : m.vocab]))
        preds = [first]
        pos = L
        tok = first
        t3 = time.time()
        for i in range(n_dec - 1):
            # host argmax of the traced model step's logits (the sampler traces are captured and measured above;
            # this check needs only the model): tokens / positions written by the host every step
            out_t = g.decode_forward(
                torch.tensor([[tok]] + [[0]] * (a.slots - 1)),
                torch.tensor([pos] + [-1] * (a.slots - 1)),
                page_table=pt,
                enable_trace=True,
                on_device_sampling=False,
                reload_inputs=True,
            )
            tok = int(torch.argmax(out_t[0, : m.vocab]))
            preds.append(tok)
            pos += 1
        report["decode_seconds"] = round(time.time() - t3, 2)
        expect = [pas[(L + j) % len(pas)] for j in range(n_dec)]
        match = sum(int(p == e) for p, e in zip(preds, expect))
        report.update(
            {
                "decode_positions": [L, L + n_dec - 1],
                "pred_tokens": preds,
                "expected_copy_tokens": expect,
                "copy_matches": match,
                "copy_accuracy": round(match / n_dec, 4),
                "pred_text": g.tokenizer.decode(preds) if g.tokenizer is not None else None,
                "expected_text": g.tokenizer.decode(expect) if g.tokenizer is not None else None,
                "supported_tokens_one_slot": tokens_cap,
            }
        )
        report["after_decode"] = dram.snapshot()
        report["hook_samples"] = dram.samples
        print(
            json.dumps(
                {k: report[k] for k in ("prompt_len", "prefill_seconds", "copy_accuracy", "pred_text", "expected_text")}
            ),
            flush=True,
        )
        g.teardown()
    finally:
        hook_cm.__exit__(None, None, None)
        save()
        from models.autoports.zai_org_glm_5_3_flash.tests.harness import close_mesh

        close_mesh(mesh)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["reduced", "capacity"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--trace-region", type=int, default=512 * MiB)
    ap.add_argument("--slots", type=int, default=None)
    ap.add_argument("--max-context", type=int, default=None)
    ap.add_argument(
        "--prompt-len", type=int, default=6221, help="reduced: prompt tokens (3 x 2048 + 77: flash, sparse)"
    )
    ap.add_argument("--moe-rows", type=int, default=1024, help="reduced: GLM53_MOE_SORTED_ROWS after the A/B")
    ap.add_argument("--workspace-mib", type=float, default=1024.0, help="capacity: DRAM kept free besides the pool")
    ap.add_argument("--decode-steps", type=int, default=24)
    ap.add_argument("--tail-gap", type=int, default=37, help="capacity: tokens left unused below the pool top")
    a = ap.parse_args()
    if a.mode == "reduced":
        a.slots = a.slots or 32
        a.max_context = a.max_context or 131072
        run_reduced(a)
    else:
        a.slots = a.slots or 1
        a.max_context = a.max_context or 131072
        run_capacity(a)


if __name__ == "__main__":
    main()
