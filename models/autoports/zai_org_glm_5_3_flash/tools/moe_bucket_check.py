# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage 10 (MoE row buckets) record of one multi-slot build: run once with GLM53_MOE_BUCKETS=0 (every slot row
through the routed experts) and once bucketed, then `compare` (active rows: bit-identical tokens and logits).

record: Glm53Generator(max_batch_size S, max_seq_len T). For each active count A in --actives: A prompts (own seed per
(A, slot): the same prompts in every run) prefilled into slots spread over the S slots, then --steps TRACED decode
steps of those A slots together (host logits; own argmax fed back). Then the decode time per step at each active count
(model + sampler trace replays, as tools/s7_probe run_decode), the bucket hits and the program-cache entries.

Usage:
  python -m ...tools.moe_bucket_check record --slots 16 --seq-len 16384 --actives 1,2,3,5,8,16 --out a.pt
  python -m ...tools.moe_bucket_check compare a.pt b.pt [--json out.json]
"""

import argparse
import json
import os
import sys
import time

import torch

import ttnn


def _slots(A, S):
    """A active slots spread over S (not only the first rows: the compaction must move rows)."""
    if A >= S:
        return list(range(S))
    step = S / A
    return sorted({min(S - 1, int(i * step + step / 2)) for i in range(A)})


def record(a):
    from models.autoports.zai_org_glm_5_3_flash.tests.harness import close_mesh
    from models.autoports.zai_org_glm_5_3_flash.tt import dsa_buckets
    from models.autoports.zai_org_glm_5_3_flash.tt.generator import L1_SMALL_SIZE, Glm53Generator

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(
        ttnn.MeshShape(1, int(os.environ.get("GLM53_TP", "4"))),
        l1_small_size=L1_SMALL_SIZE,
        trace_region_size=a.trace_region,
    )
    rec = {"env": {k: v for k, v in os.environ.items() if k.startswith(("GLM53_", "QWEN35_", "TT_METAL_TRACE"))}}
    ok = False
    try:
        t0 = time.perf_counter()
        g = Glm53Generator(mesh, max_batch_size=a.slots, max_seq_len=a.seq_len)
        rec["build_s"] = time.perf_counter() - t0
        ctl = g.__dict__.get("_dsa_buckets")
        rec["buckets"] = None if ctl is None else {"tbs": ctl.tbs, "paths": ctl.paths, "widths": ctl.widths}
        rec["trace_bytes"] = getattr(ctl, "trace_bytes", None)
        rec["programs_after_capture"] = g.program_cache_entries()
        print(
            f"BUILD {rec['build_s']:.1f} s buckets {rec['buckets']} programs {rec['programs_after_capture']}",
            flush=True,
        )
        V, S = g.model.vocab, a.slots
        runs = []
        for A in [int(x) for x in a.actives.split(",")]:
            slots = _slots(A, S)
            g.reset()
            pre = []
            for s in slots:
                ids = torch.randint(
                    100, V - 100, (a.prompt_len + 7 * s,), generator=torch.Generator().manual_seed(1000 * A + s)
                )
                lg = g.prefill_forward(ids.reshape(1, -1), prompt_lens=[len(ids)], slots=[s])
                pre.append(lg.reshape(-1)[:V].to(torch.bfloat16))
            tok = torch.tensor([int(torch.argmax(p.float())) for p in pre])
            pos = torch.tensor([a.prompt_len + 7 * s for s in slots])
            steps = []
            for i in range(a.steps):
                out = g.decode_forward(
                    tok.reshape(-1, 1), pos, slots=slots, on_device_sampling=False, enable_trace=True
                )[:, :V]
                b = ctl.moe_bucket(list(g.model.inputs.positions)) if ctl is not None and ctl.tbs else None
                steps.append({"pos": pos.tolist(), "bucket": b, "logits": out.to(torch.bfloat16)})
                tok = torch.argmax(out.float(), dim=-1)
                pos = pos + 1
            print(f"ACTIVE {A} slots {slots} tokens {tok.tolist()}", flush=True)
            runs.append({"A": A, "slots": slots, "prefill": pre, "steps": steps})
        rec["runs"] = runs
        if a.invariance:
            rec["invariance"] = _invariance(g, a, ctl)
        rec["programs_after_checks"] = g.program_cache_entries()
        # ---- decode time per step by active count (position 300; tokens fed back on the device)
        times = {}
        sm = g.model.sampling
        for A in [int(x) for x in a.time_actives.split(",") if x]:
            slots = _slots(A, S)
            pos_l = [300 if s in slots else -1 for s in range(S)]
            g.model.inputs.set_positions(pos_l)
            g._write_tokens([1000] * S)
            dsa_buckets.replay_decode(g)
            sm.sample(g.logits_out, tt_out_tok=g.tok_buf, enable_trace=True)
            ttnn.synchronize_device(mesh)
            g.model.inputs.set_positions(pos_l)
            ttnn.synchronize_device(mesh)
            t0 = time.perf_counter()
            for _ in range(a.time_steps):
                dsa_buckets.replay_decode(g)
                sm.sample(g.logits_out, tt_out_tok=g.tok_buf, enable_trace=True)
            t1 = time.perf_counter()
            ttnn.synchronize_device(mesh)
            times[A] = {
                "tokenout_ms": (time.perf_counter() - t0) * 1e3 / a.time_steps,
                "enqueue_ms": (t1 - t0) * 1e3 / a.time_steps,
                "bucket": ctl.moe_bucket(pos_l) if ctl is not None and ctl.tbs else None,
            }
            print(f"TIME active {A}: {json.dumps(times[A])}", flush=True)
        rec["decode_times"] = times
        rec["programs_end"] = g.program_cache_entries()
        rec["hits"] = None if ctl is None else ctl.hits
        g.reset()
        g.teardown()
        ok = rec["programs_after_capture"] == rec["programs_end"]
    finally:
        torch.save(rec, a.out)
        close_mesh(mesh)
    print(f"RECORD DONE program cache stable: {ok}", flush=True)
    return 0 if ok else 3


def _invariance(g, a, ctl):
    """Batch invariance: one fixed request in slot 0, decoded with A active slots (fillers in spread slots, filler
    prompts by seed). Its prefill + traced decode logits must not depend on A or on the fillers' content."""
    V, S = g.model.vocab, a.slots
    target = torch.randint(100, V - 100, (a.prompt_len + 13,), generator=torch.Generator().manual_seed(777))
    out, ref = [], None
    for spec in a.invariance.split(","):
        A, fs = (int(x) for x in spec.split(":")) if ":" in spec else (int(spec), 0)
        others = [s for s in _slots(A - 1, S - 1)] if A > 1 else []
        slots = [0] + [s + 1 for s in others]
        g.reset()
        pre = None
        for s in slots:
            ids = (
                target
                if s == 0
                else torch.randint(
                    100,
                    V - 100,
                    (a.prompt_len + 5 * s,),
                    generator=torch.Generator().manual_seed(fs * 100000 + 100 * A + s),
                )
            )
            lg = g.prefill_forward(ids.reshape(1, -1), prompt_lens=[len(ids)], slots=[s])
            if s == 0:
                pre = lg.reshape(-1)[:V].to(torch.bfloat16)
        pos = torch.tensor([len(target) if s == 0 else a.prompt_len + 5 * s for s in slots])
        tok = None
        steps = []
        for i in range(a.steps):
            if tok is None:
                tok = torch.tensor([int(torch.argmax(pre.float()))] + [1000] * (len(slots) - 1))
            o = g.decode_forward(tok.reshape(-1, 1), pos, slots=slots, on_device_sampling=False, enable_trace=True)[
                :, :V
            ]
            steps.append(o[0].to(torch.bfloat16))
            tok = torch.argmax(o.float(), dim=-1)
            pos = pos + 1
        bucket = ctl.moe_bucket([0] * len(slots) + [-1] * (S - len(slots))) if ctl is not None and ctl.tbs else None
        cur = {"A": A, "filler_seed": fs, "bucket": bucket, "prefill": pre, "steps": steps}
        if ref is None:
            ref = cur
        same = torch.equal(pre, ref["prefill"]) and all(torch.equal(x, y) for x, y in zip(steps, ref["steps"]))
        dmax = max(float((x.float() - y.float()).abs().max()) for x, y in zip(steps, ref["steps"]))
        print(
            f"INVARIANCE A {A} fillers {fs} bucket {bucket}: identical to A={ref['A']}: {same} max|d| {dmax}",
            flush=True,
        )
        out.append({"A": A, "filler_seed": fs, "bucket": bucket, "identical": bool(same), "max_abs_d": dmax})
    return out


def compare(a):
    x, y = torch.load(a.a), torch.load(a.b)
    bad, top1 = [], []
    for rx, ry in zip(x["runs"], y["runs"]):
        assert rx["A"] == ry["A"] and rx["slots"] == ry["slots"]
        for i, (p, q) in enumerate(zip(rx["prefill"], ry["prefill"])):
            top1.append(int(p.float().argmax()) == int(q.float().argmax()))
            if not torch.equal(p, q):
                bad.append(
                    f"A {rx['A']} prefill slot {rx['slots'][i]}: max |d| {float((p.float() - q.float()).abs().max())}"
                )
        for i, (s, t) in enumerate(zip(rx["steps"], ry["steps"])):
            top1.append(bool(torch.equal(s["logits"].float().argmax(-1), t["logits"].float().argmax(-1))))
            if not torch.equal(s["logits"], t["logits"]):
                d = (s["logits"].float() - t["logits"].float()).abs().amax(-1)
                bad.append(f"A {rx['A']} step {i} buckets {s['bucket']}/{t['bucket']}: max |d| per slot {d.tolist()}")
    res = {
        "a": a.a,
        "b": a.b,
        "bit_identical": not bad,
        "top1_agree": f"{sum(top1)}/{len(top1)}",
        "mismatches": bad[:40],
        "decode_times_a": x.get("decode_times"),
        "decode_times_b": y.get("decode_times"),
        "buckets_b": y.get("buckets"),
        "programs": [
            [x.get("programs_after_capture"), x.get("programs_end")],
            [y.get("programs_after_capture"), y.get("programs_end")],
        ],
        "trace_bytes": [x.get("trace_bytes"), y.get("trace_bytes")],
    }
    print(json.dumps(res, indent=1))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, indent=1)
    return 0 if not bad else 1


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record")
    r.add_argument("--slots", type=int, default=16)
    r.add_argument("--seq-len", type=int, default=16384)
    r.add_argument("--actives", default="1,2,3,5,8,16")
    r.add_argument("--invariance", default="", help="A[:filler_seed],...: target-row batch invariance (slot 0)")
    r.add_argument("--prompt-len", type=int, default=300)
    r.add_argument("--steps", type=int, default=6)
    r.add_argument("--time-actives", default="1,2,4,8,16")
    r.add_argument("--time-steps", type=int, default=20)
    r.add_argument("--trace-region", type=int, default=64 << 20)
    r.add_argument("--out", required=True)
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    c.add_argument("--json", default=None)
    a = ap.parse_args()
    sys.exit(record(a) if a.cmd == "record" else compare(a))


if __name__ == "__main__":
    main()
