# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage 10 (DSA buckets) correctness and cost record of one build; run once with GLM53_DSA_BUCKETS=0 (full-width
code) and once bucketed, then `compare` the two records (bit-identical tokens and logits required).

record: Glm53Generator(max_batch_size S, max_seq_len T) (GLM53_LAYERS selects a reduced stack), one random-token
prompt per case, prefilled into its slot (last-row host logits), then `--steps` TRACED decode steps of all case slots
together (host logits; own argmax fed back). Prompt lengths sit just below a bucket switch, so the decode crosses it
(bucket context T_b switches at position T_b + 131). Also: program-cache entries after capture and at the end, decode
time per step at a set of positions (--time-positions; model trace + sampler, as tools/s7_probe), the bucket hits.

Usage:
  python -m ...tools.dsa_bucket_check record --slots 1 --seq-len 262144 --cases 2100,4220,... --out a.pt
  python -m ...tools.dsa_bucket_check compare a.pt b.pt
"""

import argparse
import json
import os
import sys
import time

import torch

import ttnn


def _sync(mesh):
    ttnn.synchronize_device(mesh)


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
    rec = {"env": {k: v for k, v in os.environ.items() if k.startswith("GLM53_") or k.startswith("TT_METAL_TRACE")}}
    try:
        t0 = time.perf_counter()
        pt = None
        if a.serving:  # vLLM-style pool: vLLM's null block 0 + every slot's blocks (+ the model's dummy block)
            g = Glm53Generator(
                mesh, max_batch_size=a.slots, max_seq_len=a.seq_len, defer_caches=True, budget_check=False
            )
            nbs = a.seq_len // 128
            g.bind_kv_cache(a.slots * nbs + 1)
            g.capture_traces()
            pt = torch.arange(1, a.slots * nbs + 1, dtype=torch.int32).reshape(a.slots, nbs)
            d0 = next(d.attn for d in g.model.layers if d.attn_kind == "dsa")
            rec["pool"] = {
                "NB": int(d0.NB),
                "KB": int(d0.KB),
                "pool_alias": bool(getattr(d0, "pool_alias", False)),
                "pooled_bytes_per_layer": int(d0.KB * d0.P_alloc * d0.ID * 2),
            }
            print(f"POOL {rec['pool']}", flush=True)
        else:
            g = Glm53Generator(mesh, max_batch_size=a.slots, max_seq_len=a.seq_len)
        rec["build_s"] = time.perf_counter() - t0
        ctl = g.__dict__.get("_dsa_buckets")
        rec["buckets"] = (
            None if ctl is None else {"widths": ctl.widths, "trace_bytes": getattr(ctl, "trace_bytes", None)}
        )
        rec["programs_after_capture"] = g.program_cache_entries()
        print(
            f"BUILD {rec['build_s']:.1f} s buckets {rec['buckets']} programs {rec['programs_after_capture']}",
            flush=True,
        )
        V = g.model.vocab
        cases = [int(x) for x in a.cases.split(",")]
        gen = torch.Generator().manual_seed(1234)
        if (
            a.group_seeds
        ):  # one prompt stream per case group: the seed-1234 group's prompts equal a single-group record's
            seeds = [int(x) for x in a.group_seeds.split(",")]
            prompts = []
            for gi, g0 in enumerate(range(0, len(cases), a.slots)):
                gen = torch.Generator().manual_seed(seeds[gi])
                prompts += [torch.randint(100, V - 100, (L,), generator=gen) for L in cases[g0 : g0 + a.slots]]
        else:  # one stream: a later group's prompts differ from a single-group record of the same lengths
            prompts = [torch.randint(100, V - 100, (L,), generator=gen) for L in cases]
        pre, steps, groups = [], [], []
        for g0 in range(0, len(cases), a.slots):  # groups of `slots` cases decoded together
            group = list(range(g0, min(g0 + a.slots, len(cases))))
            if g0 == 0 or not a.no_reset:  # --no-reset: serving reuses a slot without reset_state (prefill overwrites)
                g.reset()
            gi = g0 // a.slots
            if a.force_ctx:  # exactness check: every prefill chunk of this group scores the T_b bucket width
                from models.autoports.zai_org_glm_5_3_flash.tt.dsa_buckets import width_of

                assert ctl is not None, "--force-ctx needs buckets on"
                Wf = width_of(int(a.force_ctx.split(",")[gi]))
                for d in ctl.layers:
                    d._force_score_width = Wf
            n_pre, n_steps = len(pre), len(steps)
            for s, c in enumerate(group):
                ids = prompts[c]
                lg = g.prefill_forward(
                    ids.reshape(1, -1),
                    prompt_lens=[len(ids)],
                    slots=[s],
                    page_table=None if pt is None else pt[s : s + 1],
                )
                pre.append(lg.reshape(-1)[:V].clone())
                print(f"PREFILL slot {s} L {len(ids)} argmax {int(torch.argmax(pre[-1]))}", flush=True)
            slots = list(range(len(group)))
            tok = torch.tensor([int(torch.argmax(pre[c])) for c in group])
            pos = torch.tensor([cases[c] for c in group])
            for i in range(a.steps):
                out = g.decode_forward(
                    tok.reshape(-1, 1),
                    pos,
                    slots=slots,
                    on_device_sampling=False,
                    enable_trace=True,
                    page_table=None if pt is None else pt[: len(slots)],
                )
                out = out[:, :V].clone()
                b = ctl.bucket([int(p) for p in pos]) if ctl is not None else None
                steps.append({"pos": pos.tolist(), "bucket": b, "logits": out.to(torch.bfloat16)})
                tok = torch.argmax(out.float(), dim=-1)
                pos = pos + 1
                print(f"DECODE step {i} pos {steps[-1]['pos']} bucket {b} tokens {tok.tolist()}", flush=True)
            groups.append({"pre": (n_pre, len(pre)), "steps": (n_steps, len(steps))})
            if ctl is not None:
                for d in ctl.layers:
                    d.__dict__.pop("_force_score_width", None)
        rec["prefill_logits"] = [p.to(torch.bfloat16) for p in pre]
        rec["steps"] = steps
        rec["groups"] = groups
        rec["programs_end_check"] = g.program_cache_entries()
        # ---- decode cost by position: 1 active slot (slot 0), model + sampler replays, as tools/s7_probe run_decode
        times = {}
        sm = g.model.sampling
        for p0 in [int(x) for x in a.time_positions.split(",") if x] if a.time_positions else []:
            if p0 >= a.seq_len - a.time_steps - 1:
                continue
            pos_l = [p0] + [-1] * (a.slots - 1)
            g.model.inputs.set_positions(pos_l)
            g._write_tokens([1000] * a.slots)
            _sync(mesh)
            dsa_buckets.replay_decode(g)
            sm.sample(g.logits_out, tt_out_tok=g.tok_buf, enable_trace=True)
            _sync(mesh)
            g.model.inputs.set_positions(pos_l)
            _sync(mesh)
            t0 = time.perf_counter()
            for _ in range(a.time_steps):
                dsa_buckets.replay_decode(g)
                sm.sample(g.logits_out, tt_out_tok=g.tok_buf, enable_trace=True)
            t1 = time.perf_counter()
            _sync(mesh)
            times[p0] = {
                "tokenout_ms": (time.perf_counter() - t0) * 1e3 / a.time_steps,
                "enqueue_ms": (t1 - t0) * 1e3 / a.time_steps,
                "bucket": ctl.bucket(pos_l) if ctl is not None else None,
            }
            print(f"TIME pos {p0}: {json.dumps(times[p0])}", flush=True)
        rec["decode_times"] = times
        # ---- TTFT: warm prefill to last-row device logits + sync (tools/s7_probe._ttft), random tokens, slot 0
        ttft = {}
        for L in [int(x) for x in a.ttft.split(",") if x]:
            g.reset()
            ids = torch.randint(100, V - 100, (L,), generator=torch.Generator().manual_seed(L))
            runs = []
            for _ in range(2):
                g.reset()
                _sync(mesh)
                t0 = time.perf_counter()
                lg = g.model.prefill(ids, 0, device_logits=True)
                _sync(mesh)
                runs.append((time.perf_counter() - t0) * 1e3)
                ttnn.deallocate(lg)
            ttft[L] = {"cold_ms": runs[0], "warm_ms": runs[1]}
            print(f"TTFT L {L}: {json.dumps(ttft[L])}", flush=True)
        rec["ttft"] = ttft
        rec["programs_end"] = g.program_cache_entries()
        rec["bucket_hits"] = None if ctl is None else list(ctl.hits)
        print(
            f"PROGRAMS after capture {rec['programs_after_capture']} after checks {rec['programs_end_check']} "
            f"end {rec['programs_end']}",
            flush=True,
        )
        g.reset()
        g.teardown()
    finally:
        torch.save(rec, a.out)
        close_mesh(mesh)
    ok = rec.get("programs_after_capture") == rec.get("programs_end")
    print(f"RECORD DONE program cache stable: {ok}", flush=True)
    return 0 if ok else 3


def _group(rec, i):
    """Case group i of a record as a record of its own."""
    g = rec["groups"][i]
    return {
        **rec,
        "prefill_logits": rec["prefill_logits"][g["pre"][0] : g["pre"][1]],
        "steps": rec["steps"][g["steps"][0] : g["steps"][1]],
    }


def compare(a):
    x, y = torch.load(a.a), torch.load(a.b)
    if a.group_a is not None:  # one case group of `a` vs the whole of `b` (exactness check vs a T_b build)
        x = _group(x, a.group_a)
    bad = []
    for i, (p, q) in enumerate(zip(x["prefill_logits"], y["prefill_logits"])):
        if not torch.equal(p, q):
            bad.append(f"prefill slot {i}: max |d| {float((p.float() - q.float()).abs().max())}")
    assert len(x["steps"]) == len(y["steps"])
    for i, (s, t) in enumerate(zip(x["steps"], y["steps"])):
        assert s["pos"] == t["pos"]
        if not torch.equal(s["logits"], t["logits"]):
            d = (s["logits"].float() - t["logits"].float()).abs().amax(dim=-1)
            bad.append(f"step {i} pos {s['pos']} buckets {s['bucket']}/{t['bucket']}: max |d| per slot {d.tolist()}")
    top1 = [
        bool(torch.equal(s["logits"].float().argmax(-1), t["logits"].float().argmax(-1)))
        for s, t in zip(x["steps"], y["steps"])
    ]
    top1 += [
        int(p.float().argmax()) == int(q.float().argmax()) for p, q in zip(x["prefill_logits"], y["prefill_logits"])
    ]
    res = {
        "top1_agree": f"{sum(top1)}/{len(top1)}",
        "a": a.a,
        "b": a.b,
        "steps": len(x["steps"]),
        "buckets_a": [s["bucket"] for s in x["steps"]],
        "buckets_b": [t["bucket"] for t in y["steps"]],
        "bit_identical": not bad,
        "mismatches": bad,
        "decode_times_a": x.get("decode_times"),
        "decode_times_b": y.get("decode_times"),
        "ttft_a": x.get("ttft"),
        "ttft_b": y.get("ttft"),
        "programs": [
            [x.get("programs_after_capture"), x.get("programs_end")],
            [y.get("programs_after_capture"), y.get("programs_end")],
        ],
        "buckets": y.get("buckets"),
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
    r.add_argument("--slots", type=int, default=1)
    r.add_argument("--seq-len", type=int, required=True)
    r.add_argument("--cases", required=True)
    r.add_argument("--steps", type=int, default=12)
    r.add_argument("--time-positions", default="")
    r.add_argument("--time-steps", type=int, default=20)
    r.add_argument("--ttft", default="")
    r.add_argument("--serving", action="store_true", help="defer_caches build on a vLLM-style pool (null block 0)")
    r.add_argument("--force-ctx", default="", help="per case group: bucket context whose width every prefill scores")
    r.add_argument("--trace-region", type=int, default=64 << 20)
    r.add_argument("--group-seeds", default="", help="per case group: prompt seed (default: one seed-1234 stream)")
    r.add_argument("--no-reset", action="store_true", help="no generator reset between case groups (serving)")
    r.add_argument("--out", required=True)
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    c.add_argument("--json", default=None)
    c.add_argument("--group-a", type=int, default=None)
    a = ap.parse_args()
    sys.exit(record(a) if a.cmd == "record" else compare(a))


if __name__ == "__main__":
    main()
