# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Prefill benchmark on the hybrid MiniMax-M2.7 (the eager served prefill, no drafter, no decode trace): wall time of
prompts of several lengths, the hybrid counters of each prefill, and the next-token logits (saved, for comparing two
prefill implementations on the same build).

env: BENCH_LENS (prompt token counts, default "512,2048,8192"), BENCH_REPEAT (runs per length, default 2),
BENCH_LAYER_TIMES=1 (synchronize around every decoder layer and report full vs hybrid layer time; slows the run),
BENCH_REF (a previous run's out dir, mounted under /poc: compare the logits), POC_MAX_SEQ (default 131072). usage (in the image): python bench_prefill.py <out dir>
"""

import json
import os
import sys
import time

sys.path.insert(0, "/poc")
import hybrid  # noqa: E402

HYB = hybrid.install()

import torch  # noqa: E402

import ttnn  # noqa: E402
from hybrid.patch import PORT, _port  # noqa: E402

PREFILL_KEYS = (
    "prefill_rows",
    "prefill_miss_pairs",
    "host_experts",
    "pf_cpu_experts",
    "prefetched",
    "streamed",
    "stream_rounds",
    "picks",
    "hits",
)
TIME_KEYS = (
    "t_read",
    "t_sync",
    "t_host",
    "t_host_wait",
    "t_up",
    "t_stream",
    "t_pf_routed",
    "t_pf_upload",
    "t_pf_sroute",
    "t_pf_hostadd",
    "t_pf_reduce",
)
SOURCES = ("/poc/hybrid/patch.py", "/poc/dflash_serve.py", "/poc/dflash_tt.py", "/poc/hybrid/fetch.py")


def prompt_ids(tok, n):
    body = "\n\n".join(f"# file: {os.path.basename(f)}\n" + open(f).read() for f in SOURCES)
    ids = tok.encode(body, add_special_tokens=False)
    while len(ids) < n:
        ids = ids + ids
    return ids[:n]


VARIANT_KEYS = {
    "seg": lambda gen, v: setattr(gen.model, "PREFILL_SEGMENT", int(v)),
    "sent": lambda gen, v: setattr(HYB, "prefill_sentinel", v == "1"),
    "rounds": lambda gen, v: setattr(HYB.stream, "rounds", int(v)),
    "min": lambda gen, v: (setattr(HYB.stream, "min_short", int(v)), setattr(HYB.stream, "min_long", int(v))),
    "cq1": lambda gen, v: setattr(HYB.stream, "cq1", v == "1"),  # needs HYB_STREAM_CQ1=1 (the mesh's second queue)
    "group": lambda gen, v: setattr(gen, "prefill_group_size", int(v)),
    "merge": lambda gen, v: setattr(HYB.stream, "merge", v == "1"),
    "split": lambda gen, v: setattr(HYB, "pf_split", v == "1"),
    "pfmin": lambda gen, v: setattr(HYB.stream, "prefetch_min", int(v)),
    "cpufix": lambda gen, v: setattr(HYB, "pf_cpu_fixed_ms", [float(x) for x in v.split("/")]),
    "maskpad": lambda gen, v: setattr(HYB, "pf_mask_pad", v == "1"),
    "hkrows": lambda gen, v: setattr(HYB, "prefill_hostk_rows", int(v)),
    "minlong": lambda gen, v: (setattr(HYB.stream, "min_long", int(v)), setattr(HYB.stream, "long_from", 1024)),
}


def parse_variants(spec):
    """BENCH_VARIANTS "name:key=v,key=v;name:..." (keys: VARIANT_KEYS) -> [(name, {key: v})]; empty = one
    variant with the environment's settings."""
    out = []
    for part in filter(None, (p.strip() for p in spec.split(";"))):
        name, _, kv = part.partition(":")
        cfg = dict(item.split("=", 1) for item in filter(None, kv.split(",")))
        bad = set(cfg) - set(VARIANT_KEYS)
        if bad:
            raise ValueError(f"BENCH_VARIANTS {name!r}: unknown keys {sorted(bad)} (known: {sorted(VARIANT_KEYS)})")
        out.append((name, cfg))
    return out or [("env", {})]


def apply_variant(gen, cfg):
    for k, v in cfg.items():
        VARIANT_KEYS[k](gen, v)


def compare(lg, ref, tag):
    return {
        f"{tag}_top1": int(ref.argmax()),
        f"{tag}_pcc": round(float(torch.corrcoef(torch.stack([lg, ref]))[0, 1]), 6),
        f"{tag}_max_abs": round(float((lg - ref).abs().max()), 4),
        f"{tag}_top5_overlap": len(set(lg.topk(5).indices.tolist()) & set(ref.topk(5).indices.tolist())),
    }


def wrap_layer_times(gen, times):
    for li, layer in enumerate(gen.model.layers):
        orig = layer.forward

        def fwd(*a, _f=orig, _li=li, **kw):
            ttnn.synchronize_device(gen.mesh_device)
            t0 = time.perf_counter()
            y = _f(*a, **kw)
            ttnn.synchronize_device(gen.mesh_device)
            times[_li] = times.get(_li, 0.0) + time.perf_counter() - t0
            return y

        layer.forward = fwd


def main():
    out_dir = sys.argv[1]
    os.makedirs(out_dir, exist_ok=True)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), l1_small_size=24576, trace_region_size=96_000_000)
    res = {"runs": [], "env": {k: v for k, v in os.environ.items() if k.startswith(("HYB_", "BENCH_", "QWEN35_"))}}
    try:
        gen = _port("generator").build_generator(
            f"models/autoports/{PORT}",
            mesh,
            max_seq_len=int(os.environ.get("POC_MAX_SEQ", "131072")),
            warm_decode_trace=False,
            warm_prefill_trace=False,
        )
        nums = [int(getattr(layer, "layer_num", li)) for li, layer in enumerate(gen.model.layers)]
        for n in nums:
            HYB.lib_layer(n)
        HYB.fetch.finalize(HYB.loader, HYB.stream)
        full = [li for li, n in enumerate(nums) if n in HYB.full_layers]
        hyb = [li for li, n in enumerate(nums) if n not in HYB.full_layers]
        print(f"BENCH layers: {len(full)} full {full}, {len(hyb)} hybrid", flush=True)
        times = {}
        layer_times = os.environ.get("BENCH_LAYER_TIMES") == "1"
        if layer_times:
            wrap_layer_times(gen, times)
        lens = [int(v) for v in os.environ.get("BENCH_LENS", "512,2048,8192").split(",")]
        repeat = int(os.environ.get("BENCH_REPEAT", "2"))
        variants = parse_variants(os.environ.get("BENCH_VARIANTS", ""))
        first = {}  # tokens -> logits of the first variant (the in-run reference)
        for (vname, cfg), n in ((v, n) for v in variants for n in lens):
            apply_variant(gen, cfg)
            ids = prompt_ids(gen.tokenizer, n)
            for rep in range(repeat):
                gen.reset()
                HYB.reset_stats()
                times.clear()
                ttnn.synchronize_device(mesh)
                reused0 = HYB.stream.reused
                t0 = time.perf_counter()
                lg = gen.prefill_forward(
                    torch.tensor(ids).reshape(1, -1), page_table=gen.page_table[:1], prompt_lens=[n]
                )
                ttnn.synchronize_device(mesh)
                dt = time.perf_counter() - t0
                lg = torch.as_tensor(lg).float().reshape(-1)[: int(gen.args.vocab_size)]
                st = HYB.st
                r = {
                    "variant": vname,
                    "tokens": n,
                    "rep": rep,
                    "prefill_s": round(dt, 3),
                    "tok_s": round(n / dt, 1),
                    "top1": int(lg.argmax()),
                    **{k: int(st.get(k, 0)) for k in PREFILL_KEYS},
                    "stream_reused": HYB.stream.reused - reused0,
                    **{k: round(float(st.get(k, 0.0)), 3) for k in TIME_KEYS},
                }
                if layer_times:
                    r["full_layers_s"] = round(sum(times.get(li, 0.0) for li in full), 3)
                    r["hybrid_layers_s"] = round(sum(times.get(li, 0.0) for li in hyb), 3)
                    r["per_full_layer_ms"] = round(1e3 * r["full_layers_s"] / max(1, len(full)), 1)
                    r["per_hybrid_layer_ms"] = round(1e3 * r["hybrid_layers_s"] / max(1, len(hyb)), 1)
                ref_path = os.path.join(os.environ.get("BENCH_REF", ""), f"logits_{n}_0.pt")
                if os.environ.get("BENCH_REF") and os.path.exists(ref_path):
                    r.update(compare(lg, torch.load(ref_path), "ref"))
                first.setdefault(n, lg)
                if vname != variants[0][0]:
                    r.update(compare(lg, first[n], "v0"))
                print("PREFILL " + json.dumps(r), flush=True)
                res["runs"].append(r)
                tag = "" if vname == variants[0][0] else f"{vname}_"
                torch.save(lg, os.path.join(out_dir, f"logits_{tag}{n}_{rep}.pt"))
                json.dump(res, open(os.path.join(out_dir, "result.json"), "w"), indent=1)
    finally:
        HYB.fetch.release()
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()
