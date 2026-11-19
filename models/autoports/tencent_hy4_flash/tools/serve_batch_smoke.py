# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Multi-user serving smoke test through the vLLM adapter (tt/generator_vllm.py) without vLLM.

A: each prompt alone (greedy, NEW tokens) -> reference streams.
B: staggered arrivals (user u prefills into row u at step STAGGER * u), one row remap (rows 0 <-> 1) midway, a host-
   sampled step, rows reused by second-wave requests -> per-user agreement with A (prefix match / all tokens).
C: decode tok/s at 1, 2, 4, ... users (all rows live, median step).

    python -m models.autoports.tencent_hy4_flash.tools.serve_batch_smoke TEXTS --users 8 --ctx 16384
"""

from __future__ import annotations

import argparse
import json
import os
import time
from types import SimpleNamespace

import torch
import ttnn
from loguru import logger


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("texts")
    ap.add_argument("--users", type=int, default=8)
    ap.add_argument("--ctx", type=int, default=16384)
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--layers", default="")
    ap.add_argument("--no-engram", action="store_true")
    ap.add_argument("--new", type=int, default=40)
    ap.add_argument("--stagger", type=int, default=3)
    ap.add_argument("--bench-steps", type=int, default=48)
    ap.add_argument("--variants", default="base")
    ap.add_argument("--no-bench", action="store_true")
    ap.add_argument("--stale-check", action="store_true")
    ap.add_argument("--long-check", action="store_true")
    args = ap.parse_args()
    B = args.users
    os.environ["HY4_SLOTS"] = str(B)
    from models.autoports.tencent_hy4_flash.tools.generate import build, open_mesh
    from models.autoports.tencent_hy4_flash.tt.engine_batch import Hy4BatchEngine
    from models.autoports.tencent_hy4_flash.tt.generator_vllm import HYV4FlashForConditionalGeneration as G

    seqs = json.load(open(args.texts))
    names = sorted(seqs)
    prompts = [seqs[names[u % len(names)]][: 40 + 37 * u] for u in range(B)]  # varied lengths / 32-row alignment
    second = [seqs[names[(u + 3) % len(names)]][: 50 + 29 * u] for u in range(B)]
    mesh = open_mesh()
    try:
        model = build(mesh, args)
        if model.engram is not None and hasattr(model.engram.store, "wait_ram"):
            model.engram.store.wait_ram()
        eng = Hy4BatchEngine(model)
        ad = G(model, eng, mesh, args.ctx - 64)
        t0 = time.time()
        ad.warmup_model_prefill(None, True)
        logger.info(f"warm-up {time.time() - t0:.0f} s")
        greedy = SimpleNamespace(temperature=[0.0] * B, top_k=[0] * B, top_p=[1.0] * B, seed=[None] * B)

        def prefill(rows, ps):
            n = max(len(p) for p in ps)
            tok = torch.zeros(len(ps), n, dtype=torch.long)
            for i, p in enumerate(ps):
                tok[i, : len(p)] = torch.tensor(p)
            out = ad.prefill_forward(tok, None, None, torch.tensor([len(p) for p in ps]), empty_slots=rows,
                                     sampling_params=greedy)
            return [int(x) for x in out.reshape(-1)]

        def decode(st, sp=greedy, logits=False, **kw):
            """st: {row: [stream tokens, pos]} -> {row: next}."""
            tok = torch.zeros(B, dtype=torch.int32)
            pos = torch.full((B,), -1, dtype=torch.int32)
            for r, (s, p) in st.items():
                tok[r], pos[r] = s[-1], p
            out = ad.decode_forward(tokens=tok.reshape(B, 1), start_pos=pos, sampling_params=sp, **kw)
            if sp is None:
                return {r: (out[r, 0] if logits else int(out[r, 0].argmax())) for r in st}
            return {r: int(out[r, 0]) for r in st}

        # A: alone
        ref, ref2, margins = [], [], []
        for plist, dst in ((prompts, ref), (second, ref2)):
            for p in plist:
                first = prefill([0], [p])[0]
                st = {0: [[first], len(p)]}
                mg = [float("inf")]
                while len(st[0][0]) < args.new:
                    lg = decode(st, sp=None, logits=True)[0]
                    top = torch.topk(lg.float(), 2)
                    mg.append(float(top.values[0] - top.values[1]))
                    st[0][0].append(int(top.indices[0]))
                    st[0][1] += 1
                dst.append(st[0][0])
                margins.append(mg)
        if args.stale_check:
            # the same prompt into the same slot after a long / a short previous occupant: streams must match
            def alone(p, junk):
                prefill([0], [junk])
                first = prefill([0], [p])[0]
                st = {0: [[first], len(p)]}
                mg = [float("inf")]
                while len(st[0][0]) < args.new:
                    lg = decode(st, sp=None, logits=True)[0]
                    top = torch.topk(lg.float(), 2)
                    mg.append(float(top.values[0] - top.values[1]))
                    st[0][0].append(int(top.indices[0]))
                    st[0][1] += 1
                return st[0][0], mg

            gj = torch.Generator().manual_seed(1)
            long_junk = torch.randint(1000, 100000, (3000,), generator=gj).tolist()
            short_junk = torch.randint(1000, 100000, (20,), generator=gj).tolist()
            for u in range(B):
                a1, m1 = alone(prompts[u], long_junk)
                a2, _ = alone(prompts[u], short_junk)
                k = next((i for i in range(len(a1)) if a1[i] != a2[i]), len(a1))
                print(f"STALE user {u}: after-long vs after-short prefix {k}/{len(a1)}"
                      + (f" (margin {m1[k]:.3f}); vs phase A {sum(x == y for x, y in zip(a1, ref[u]))}"
                         f"/{sum(x == y for x, y in zip(a2, ref[u]))}" if k < len(a1) else ""), flush=True)
        # B: staggered arrivals, a remap, a host-sampled step, second wave into reused rows (variants: --variants)
        def phase_b(tag, stagger, do_remap, do_host, order):
            st, owner, done = {}, {}, {}
            step, remapped, hosted, wave2 = 0, not do_remap, not do_host, set()
            arrive = {u: stagger * i for i, u in enumerate(order)}
            while len(done) < 2 * B:
                for u in range(B):
                    if step == arrive[u] and u not in owner.values():
                        st[u] = [[prefill([u], [prompts[u]])[0]], len(prompts[u])]
                        owner[u] = u
                if not remapped and step == stagger * B // 2 and 0 in st and 1 in st:
                    st[0], st[1] = st[1], st[0]
                    owner[0], owner[1] = owner[1], owner[0]
                    remap = torch.arange(B, dtype=torch.int32)
                    remap[0], remap[1] = 1, 0
                    nx = decode(dict(st), slot_remap=remap)
                    remapped = True
                elif not hosted and len(st) >= 3:
                    nx = decode(st, sp=None)  # host sampling (greedy argmax of the returned logits)
                    hosted = True
                else:
                    nx = decode(st)
                for r, v in nx.items():
                    st[r][0].append(v)
                    st[r][1] += 1
                for r in list(st):
                    u = owner[r]
                    if len(st[r][0]) >= args.new:
                        done[u] = st.pop(r)[0]
                        if u < B and u + B not in done and u + B not in wave2:  # second wave reuses the row
                            wave2.add(u + B)
                            st[r] = [[prefill([r], [second[u]])[0]], len(second[u])]
                            owner[r] = u + B
                step += 1
            agree, bad = [], 0
            for u in range(2 * B):
                a, b = (ref + ref2)[u], done[u]
                k = next((i for i in range(min(len(a), len(b))) if a[i] != b[i]), min(len(a), len(b)))
                agree.append((k, sum(x == y for x, y in zip(a, b)) / len(a)))
                if k < min(len(a), len(b)):
                    bad += margins[u][k] > 1.0
                    print(f"DIVERGE {tag} user {u} at {k} (arrival step {arrive.get(u % B)}): alone {a[k]} batch {b[k]}, "
                          f"alone margin {margins[u][k]:.3f}", flush=True)
            print(f"AGREE {tag}: " + " ".join(f"{k}/{f:.2f}" for k, f in agree), flush=True)
            print(f"AGREE {tag} mean all-token {sum(f for _, f in agree) / len(agree):.3f}; full-prefix users "
                  f"{sum(k >= args.new for k, _ in agree)}/{len(agree)}; confident divergences (margin > 1) {bad}",
                  flush=True)

        rev = list(range(B))[::-1]
        variants = {"base": (args.stagger, True, True, list(range(B))), "noremap": (args.stagger, False, True, list(range(B))),
                    "nohost": (args.stagger, True, False, list(range(B))), "plain": (args.stagger, False, False, list(range(B))),
                    "s5": (5, False, False, list(range(B))), "rev": (args.stagger, False, False, rev),
                    "once": (0, False, False, list(range(B)))}
        for v in filter(None, args.variants.split(",")):
            phase_b(v, *variants[v])
        if args.long_check:
            # worst case: every row live, row 0 re-prefilled with a near-max prompt (its TTFT), then batched steps
            gj = torch.Generator().manual_seed(2)
            firsts = prefill(list(range(1, B)), [prompts[u] for u in range(1, B)])
            st = {u: [[firsts[u - 1]], len(prompts[u])] for u in range(1, B)}
            n_long = args.ctx - 64 - 2 * args.new
            lp = torch.randint(1000, 100000, (n_long,), generator=gj).tolist()
            t1 = time.perf_counter()
            st[0] = [[prefill([0], [lp])[0]], n_long]
            ttft = time.perf_counter() - t1
            ts = []
            for _ in range(args.new):
                t1 = time.perf_counter()
                nx = decode(st)
                ts.append(time.perf_counter() - t1)
                for r, v in nx.items():
                    st[r][0].append(v)
                    st[r][1] += 1
            print(f"LONG ok: {n_long}-token prompt with {B - 1} live users: TTFT {ttft:.2f} s; then {B} users "
                  f"median step {sorted(ts)[len(ts) // 2] * 1e3:.2f} ms; {model.dram_report()}", flush=True)
        # C: throughput
        for n in [b for b in (1, 2, 4, 8, 16) if b <= B and not args.no_bench]:
            firsts = prefill(list(range(n)), [prompts[u] for u in range(n)])
            st = {u: [[firsts[u]], len(prompts[u])] for u in range(n)}
            ts = []
            for _ in range(args.bench_steps):
                t1 = time.perf_counter()
                nx = decode(st)
                ts.append(time.perf_counter() - t1)
                for r, v in nx.items():
                    st[r][0].append(v)
                    st[r][1] += 1
            med = sorted(ts[4:])[len(ts[4:]) // 2]
            print(f"SERVE {n} users: step {med * 1e3:.2f} ms -> {n / med:.1f} tok/s total, {1 / med:.1f} per user",
                  flush=True)
        logger.info(f"engine stats {eng.stats}; traces {len(eng.dec.btraces)}")
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
