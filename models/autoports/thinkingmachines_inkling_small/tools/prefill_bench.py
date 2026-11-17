# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Where prefill time goes: the engine's prefill of an N-token prompt (after one untimed warm prefill), wall time, and
the hybrid overlay's host counters per chunk (t_* keys of HYB.st). Dev signal.

usage (hybrid env as tools/eval_hybrid.sh, TOOL=prefill_bench.py): prefill_bench.py [N=8192]
env: PF_PROF (1: cProfile of the timed prefill's main thread), PF_OPS (1: ttnn op calls by op / call site), PF_MTP_PROMPT (1: with the drafter's prompt pass, as served), ENGINE_CTX, PF_CAPTURE (1: traces captured and a\n     generation before the timed prefill, as served), PF_PRE (a long prefill of this many tokens before the captures),
     PF_TEXT (a json of token-id documents: real text instead of spread synthetic ids)
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
        prof = None
        if os.environ.get("PF_PROF") == "1":  # the main thread's time by function (where the host blocks or works)
            import cProfile

            prof = cProfile.Profile()
            prof.enable()
        ops = _op_tally() if os.environ.get("PF_OPS") == "1" else None
        t0 = time.time()
        eng._prefill(ids)
        ttnn.synchronize_device(mesh)
        dt = time.time() - t0
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
                "hybrid host timers (ms per chunk): "
                + " ".join(f"{k} {1e3 * v / nch:.0f}" for k, v in sorted(d.items()) if k.startswith("t_") and v > 0),
                flush=True,
            )
            print(
                "counters per chunk: "
                + " ".join(f"{k} {v / nch:.0f}" for k, v in sorted(d.items()) if not k.startswith("t_") and v),
                flush=True,
            )
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()
