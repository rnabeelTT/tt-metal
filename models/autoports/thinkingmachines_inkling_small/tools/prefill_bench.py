# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Where prefill time goes: the engine's prefill of an N-token prompt (after one untimed warm prefill), wall time, and
the hybrid overlay's host counters per chunk (t_* keys of HYB.st). Dev signal.

usage (hybrid env as tools/eval_hybrid.sh, TOOL=prefill_bench.py): prefill_bench.py [N=8192]
env: PF_PROF (1: cProfile of the timed prefill's main thread), PF_OPS (1: ttnn op calls by op / call site),
     PF_DEVPROF (1: device time per op by call site; see _dev_tally), PF_MTP_PROMPT (1: with the drafter's prompt pass, as served), ENGINE_CTX, PF_CAPTURE (1: traces captured and a\n     generation before the timed prefill, as served), PF_PRE (a long prefill of this many tokens before the captures),
     PF_TEXT (a json of token-id documents: real text instead of spread synthetic ids), PF_DUMP (a path: the last
     prompt row's logits saved, for bit-identity checks: tools/pf_cmp.py)
"""

import os
import sys
import time

if os.environ.get("HYB_ENABLE") == "1":
    sys.path.insert(0, os.environ.get("HYB_POC_DIR", "/opt/tt-metal/models/experimental/hybrid_moe"))
    os.environ.setdefault("HYB_PORT", "thinkingmachines_inkling_small")
    import hybrid

    hybrid.install()
    from hybrid.patch import HYB
else:
    HYB = None

import torch  # noqa: E402
import ttnn  # noqa: E402

from models.autoports.thinkingmachines_inkling_small.tt.inkling_serve import InklingSpecServing  # noqa: E402
from models.autoports.thinkingmachines_inkling_small.tt.model import create_model  # noqa: E402


class _op_tally:
    """PF_OPS=1: every ttnn op call of the timed prefill by (op, call site), with its host time (where the dispatch
    cost per layer-chunk goes)."""

    def __init__(self):
        import collections

        from ttnn import decorators as dec

        self.n, self.t = collections.Counter(), collections.Counter()
        self.ops, self.sites = collections.Counter(), collections.Counter()
        tally = self
        orig = dec.FastOperation.__call__

        def call(op, *a, **k):
            f = sys._getframe(1)
            while f is not None and ("/ttnn/ttnn/" in f.f_code.co_filename):
                f = f.f_back
            site = f"{f.f_code.co_filename.split('/tt-metal/')[-1]}:{f.f_lineno}({f.f_code.co_name})" if f else "?"
            t0 = time.perf_counter()
            try:
                return orig(op, *a, **k)
            finally:
                dt = time.perf_counter() - t0
                key = (op.python_fully_qualified_name, site)
                tally.n[key] += 1
                tally.t[key] += dt
                tally.ops[op.python_fully_qualified_name] += 1
                tally.sites[site.split(":")[0] + ":" + site.split("(")[-1]] += 1

        dec.FastOperation.__call__ = call
        self._undo = lambda: setattr(dec.FastOperation, "__call__", orig)

    def report(self, units):
        self._undo()
        tot_n, tot_t = sum(self.n.values()), sum(self.t.values())
        print(f"OPS: {tot_n} calls ({tot_n / units:.1f} per layer-chunk over {units}), {tot_t:.2f} s host", flush=True)
        print("OPS by op:", flush=True)
        for op, c in self.ops.most_common(40):
            print(f"  {c / units:7.2f}/lc  {op}", flush=True)
        print("OPS by function:", flush=True)
        for s, c in self.sites.most_common(40):
            print(f"  {c / units:7.2f}/lc  {s}", flush=True)
        print("OPS by site (top 80 by host time):", flush=True)
        for key in sorted(self.t, key=self.t.get, reverse=True)[:80]:
            print(f"  {self.n[key] / units:6.2f}/lc {1e3 * self.t[key]:8.1f} ms  {key[0]}  @ {key[1]}", flush=True)


class _dev_tally:
    """PF_DEVPROF=1: exact device time (DEVICE FW DURATION, die 0) of every top-level ttnn op of the timed prefill,
    by call site: each op is followed by a device sync and a profiler read (tools/op_prof.py's method), so the wall time
    of that run is not the served one; the device total vs the unprofiled wall time is the host-bound share. Needs
    TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_MID_RUN_DUMP=1 TT_METAL_PROFILER_CPP_POST_PROCESS=1."""

    def __init__(self, mesh):
        import collections

        import ttnn._ttnn.profiler as P
        from ttnn import decorators as dec

        self.us, self.n, self.progs = collections.Counter(), collections.Counter(), collections.Counter()
        self.done = 0
        tally = self

        def read():
            ttnn.synchronize_device(mesh)
            ttnn.ReadDeviceProfiler(mesh)
            data = P.get_latest_programs_perf_data()
            if not data:
                return 0, 0.0
            n, us = 0, 0.0
            for prog in data[sorted(data)[0]]:
                res = prog.program_analyses_results
                k = "DEVICE FW DURATION [ns]" if "DEVICE FW DURATION [ns]" in res else next(iter(res))
                us += res[k].duration / 1e3
                n += 1
            return n, us

        read()  # drop what ran before
        orig = dec.FastOperation.__call__

        def call(op, *a, **k):
            if dec.OPERATION_CALL_STACK:
                return orig(op, *a, **k)
            f = sys._getframe(1)
            while f is not None and ("/ttnn/ttnn/" in f.f_code.co_filename):
                f = f.f_back
            site = f"{f.f_code.co_filename.split('/tt-metal/')[-1]}:{f.f_lineno}({f.f_code.co_name})" if f else "?"
            out = orig(op, *a, **k)
            n, us = read()
            key = (op.python_fully_qualified_name.replace("ttnn.", ""), site)
            tally.us[key] += us
            tally.n[key] += 1
            tally.progs[key] += n
            tally.done += 1
            if tally.done % 500 == 0:  # progress (a profiled prefill is slow; a silent run looks hung)
                print(f"DEV progress: {tally.done} ops, {sum(tally.us.values()) / 1e3:.0f} ms device so far", flush=True)
            return out

        dec.FastOperation.__call__ = call
        self._undo = lambda: setattr(dec.FastOperation, "__call__", orig)

    def report(self, nch):
        self._undo()
        tot = sum(self.us.values())
        print(f"DEV: device total {tot / 1e3:.1f} ms ({tot / 1e3 / nch:.0f} ms per chunk, die 0), "
              f"{sum(self.n.values())} top-level ops, {sum(self.progs.values())} programs", flush=True)
        fam = {}
        for (op, site), us in self.us.items():
            fn = site.split("(")[-1].rstrip(")")
            fam[fn] = fam.get(fn, 0.0) + us
        print("DEV by function (ms per chunk):", "  ".join(f"{k} {v / 1e3 / nch:.1f}" for k, v in
              sorted(fam.items(), key=lambda kv: -kv[1])[:25]), flush=True)
        print("DEV by site (top 50, ms per chunk | calls per chunk):", flush=True)
        for key in sorted(self.us, key=self.us.get, reverse=True)[:50]:
            print(f"  {self.us[key] / 1e3 / nch:8.2f} ms  {self.n[key] / nch:7.1f}  {key[0]}  @ {key[1]}", flush=True)


def main():
    Ns = [int(v) for v in (sys.argv[1] if len(sys.argv) > 1 else "8192").split(",")]
    N = Ns[-1]
    os.environ.setdefault("INKLING_MTP_PROMPT", os.environ.get("PF_MTP_PROMPT", "1"))
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(
        ttnn.MeshShape(1, 4),
        l1_small_size=24576,
        trace_region_size=int(os.environ.get("SPEC_TRACE_REGION", "335544320")),
    )
    try:
        args, model = create_model(mesh, max_seq_len=int(os.environ.get("ENGINE_CTX", "32768")))
        cap = os.environ.get("PF_CAPTURE") == "1"  # as served: traces captured and replayed before the timed prefill
        eng = InklingSpecServing(model, args, mesh, int(os.environ.get("INKLING_SPEC_K", "2")), trace=cap)
        text = None
        if os.environ.get("PF_TEXT"):  # real text: the held-out documents (token ids) concatenated, repeated to length
            import json

            docs = json.load(open(os.environ["PF_TEXT"]))
            text = [t for d in (docs.values() if isinstance(docs, dict) else docs) for t in d]

        def prompt(n, salt):
            if text is None:
                return (torch.arange(n + 1) * salt % 150000 + 1000).tolist()
            off = (salt * 37) % len(text)
            return [text[(off + i) % len(text)] for i in range(n + 1)]

        ids = prompt(N, 7919)
        if os.environ.get("PF_WATCHDOG"):  # a hang: every thread's stack after this many seconds, then exit
            import faulthandler

            faulthandler.dump_traceback_later(int(os.environ["PF_WATCHDOG"]), exit=True)
        eng.warm()  # compiles every prefill bucket; pins the device-format copies when streaming (HYB_DEV_EXPERTS)
        pre = int(os.environ.get("PF_PRE", "0"))  # a long prefill before the captures (lazy device caches then exist)
        if pre:
            eng._reset()
            eng._prefill((torch.arange(pre + 1) * 31337 % 150000 + 1000).tolist())
            ttnn.synchronize_device(mesh)
            eng._reset()
        if cap:
            eng.capture()
            print("generated before the prefill:", len(eng.generate(ids[:300], 64)), "tokens", flush=True)
        ttnn.synchronize_device(mesh)
        for n_prev in Ns[:-1]:  # earlier requests: a prefill of n_prev tokens, then 32 generated (PF_CAPTURE: traced)
            t0 = time.time()
            out = eng.generate(prompt(n_prev, 104729), 32) if cap else None
            if not cap:
                eng._reset()
                eng._prefill(prompt(n_prev, 104729))
            ttnn.synchronize_device(mesh)
            print(f"earlier request: {n_prev} prompt tokens in {time.time() - t0:.1f} s", flush=True)
        st0 = dict(HYB.st) if HYB is not None else {}
        eng._reset()
        if HYB is not None:
            HYB.reset_placement()
        prof = None
        if os.environ.get("PF_PROF") == "1":  # the main thread's time by function (where the host blocks or works)
            import cProfile

            prof = cProfile.Profile()
            prof.enable()
        ops = _op_tally() if os.environ.get("PF_OPS") == "1" else None
        dev = _dev_tally(mesh) if os.environ.get("PF_DEVPROF") == "1" else None
        if os.environ.get("PF_WATCHDOG"):  # (re-armed: the engine's own watchdog cancels it after a request)
            import faulthandler

            faulthandler.dump_traceback_later(int(os.environ["PF_WATCHDOG"]), exit=True)
        if hasattr(eng.model, "layer_hashes"):
            eng.model.layer_hashes = []
        t0 = time.time()
        last = eng._prefill(ids)
        ttnn.synchronize_device(mesh)
        dt = time.time() - t0
        # PF_REPEAT=n: n more timed prefills of the same prompt (run-to-run noise); PF_SWEEP="ENV=v1,v2,..": one
        # repeat per value, the env var set before each (e.g. INKLING_PREFILL_HOSTK_THREADS=8,7,6)
        reps = int(os.environ.get("PF_REPEAT", "0"))
        sweep = os.environ.get("PF_SWEEP")
        vals = [None] * reps
        if sweep:
            key, vs = sweep.split("=", 1)
            vals = [(key, v) for v in vs.split(",")] * max(1, reps)
        for kv in vals:
            if kv is not None and kv[0].startswith("HYB."):  # an attribute of the hybrid state, e.g. HYB.pf_copy_ms
                obj, *path = kv[0].split(".")[1:]
                tgt = HYB
                for part in [obj] + path[:-1]:
                    tgt = getattr(tgt, part)
                name = (path or [obj])[-1] if path else obj
                tgt = HYB if not path else tgt
                cur = getattr(tgt, name)
                setattr(tgt, name, type(cur)(kv[1]))
            elif kv is not None:
                os.environ[kv[0]] = kv[1]
            eng._reset()
            sr = dict(HYB.st) if HYB is not None else {}
            t1 = time.time()
            eng._prefill(ids)
            ttnn.synchronize_device(mesh)
            d1 = time.time() - t1
            print(f"repeat{'' if kv is None else ' ' + kv[0] + '=' + kv[1]}: {N} tokens {d1:.2f} s ({N / d1:.0f} tok/s)",
                  flush=True)
            if HYB is not None:  # this repeat's host timers (ms per chunk): what grows in a slow repeat
                nc = -(-N // eng.chunk)
                print(
                    "  repeat timers: "
                    + " ".join(
                        f"{k} {1e3 * (v - sr.get(k, 0)) / nc:.0f}"
                        for k, v in sorted(HYB.st.items())
                        if k.startswith("t_") and isinstance(v, (int, float)) and v - sr.get(k, 0) > 0.0005 * nc
                    ),
                    flush=True,
                )
        det = int(os.environ.get("PF_DET", "0"))  # PF_DET=n: n more prefills of the prompt, last-row logits vs the first
        verify_ref = None
        if det and last is not None and os.environ.get("PF_DET_VERIFY") == "1":  # + the first verify pass (row P - 1)
            verify_ref = eng._verify([int(ids[-1])], len(ids) - 1)[0].float().cpu().clone()
        if det and last is not None:
            ref, same = last.float().cpu().clone(), 0
            ref_h = list(getattr(eng.model, "layer_hashes", []))
            mode = os.environ.get("PF_DET_PT")  # the repeats through a vLLM-style page table (ids 1.., shifted / permuted)
            if mode:
                n = eng.pt_host.numel()
                row = torch.arange(n) + 1 if mode == "shift" else torch.randperm(n, generator=torch.Generator().manual_seed(0)) + 1
                eng.enable_vllm_paging(n + 1, 8, eng.max_seq_len)
                eng.set_page_row(row)
                print(f"det repeats through page table {mode}: {row[:6].tolist()} ...", flush=True)
            for r in range(det):
                eng._reset()
                if HYB is not None:
                    HYB.reset_placement()
                eng.model.layer_hashes = []
                lr = eng._prefill(ids).float().cpu()
                if verify_ref is not None:
                    vr = eng._verify([int(ids[-1])], len(ids) - 1)[0].float().cpu()
                    dv = (vr - verify_ref).abs().max().item()
                    print(f"det repeat {r + 1}: first verify max |d| {dv:.4g}, argmax {int(vr[0].argmax())} vs "
                          f"{int(verify_ref[0].argmax())}", flush=True)
                if ref_h:  # INKLING_PF_LAYER_HASH=1: the first layer whose residual differs from the first prefill's
                    first = next(((a, b) for a, b in zip(ref_h, eng.model.layer_hashes) if a != b), None)
                    if first:
                        print(f"det repeat {r + 1}: first difference after layer {first[0][0]} chunk {first[0][1]}: "
                              f"valid rows {'differ' if first[0][2] != first[1][2] else 'same'}, padding rows "
                              f"{'differ' if first[0][3] != first[1][3] else 'same'}", flush=True)
                    else:
                        print(f"det repeat {r + 1}: every layer's residual identical", flush=True)
                d = (lr - ref).abs().max().item()
                same += d == 0
                print(f"det repeat {r + 1}: max |d| {d:.4g}, argmax {int(lr.argmax())} vs {int(ref.argmax())}", flush=True)
            print(f"DET {N} tokens: {same} of {det} repeats bit-identical to the first", flush=True)
        if os.environ.get("PF_DUMP") and last is not None:  # the last row's logits, for bit-identity checks
            torch.save(last.cpu(), os.environ["PF_DUMP"])
            print(f"saved {os.environ['PF_DUMP']}: logits {tuple(last.shape)}, argmax {int(last.argmax())}", flush=True)
        if dev is not None:
            dev.report(-(-N // eng.chunk))
        if ops is not None:
            ops.report(-(-N // eng.chunk) * len(eng.model.layers))
        if prof is not None:
            import pstats

            prof.disable()
            for key in ("tottime", "cumulative"):
                pstats.Stats(prof, stream=sys.stdout).sort_stats(key).print_stats(45)
            # who calls the host-side conversions (uploads / readbacks)
            pstats.Stats(prof, stream=sys.stdout).sort_stats("tottime").print_callers(
                r"core.py.*\((from_torch|to_torch)\)"
            )
            sys.stdout.flush()
        nch = -(-N // eng.chunk)
        print(
            f"prefill {N} tokens: {dt:.2f} s ({N / dt:.0f} tok/s), {nch} chunks of {eng.chunk}: "
            f"{dt / nch * 1e3:.0f} ms per chunk",
            flush=True,
        )
        if HYB is not None:
            d = {k: HYB.st[k] - st0.get(k, 0) for k in HYB.st if isinstance(HYB.st[k], (int, float))}
            print(
                f"hybrid host timers (ms per chunk, summed over the {1 + len(vals)} prefills): "
                + " ".join(f"{k} {1e3 * v / nch:.0f}" for k, v in sorted(d.items()) if k.startswith("t_") and v > 0),
                flush=True,
            )
            print(
                "counters per chunk: "
                + " ".join(f"{k} {v / nch:.0f}" for k, v in sorted(d.items()) if not k.startswith("t_") and v),
                flush=True,
            )
    finally:
        from models.autoports.thinkingmachines_inkling_small.tt import clean_exit

        clean_exit.close(mesh)


if __name__ == "__main__":
    main()
