# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""DFlash speculative decoding on the hybrid MiniMax-M2.7, traced (the step itself: dflash_serve.DFlashServing, shared
with vLLM serving). The drafter is built BEFORE the target.

env: POC_PROMPTS, POC_PROMPT_START, POC_NEW_TOKENS, POC_FORCE (fixed-text decode, see below), POC_MAX_SEQ, POC_LONG (a ~N-token code prompt from this repo's
sources), POC_WARMUP (warm-up generation tokens, default 64, 0 = off), DFLASH_K (drafts per pass, default 3), DFLASH_MEMONLY (print the footprint and stop),
plus dflash_serve's DFLASH_*. usage (in the image): python run_dflash_traced.py <out dir>
"""

import json
import os
import sys

sys.path.insert(0, "/poc")
import hybrid  # noqa: E402

HYB = hybrid.install()

import ttnn  # noqa: E402
from dflash_serve import DFlashServing, load_drafter  # noqa: E402
from hybrid.patch import PORT, _port  # noqa: E402

K = int(os.environ.get("DFLASH_K", "3"))  # drafts per pass (verify rows K + 1 = HYB_ROWS)
NEW = int(os.environ.get("POC_NEW_TOKENS", "128"))
# POC_FORCE: a result.json with recorded ids (a run with POC_NEW_TOKENS >= NEW + 8): the fixed-text benchmark decodes
# each prompt's recorded text (dflash_serve.generate(force=...))
_fz = os.environ.get("POC_FORCE")
forced = {r["prompt"]: r["ids"] for r in json.load(open(_fz))["runs"] if "ids" in r} if _fz else {}
HYB_KEYS = (
    "pf_act",
    "pf_pred",
    "pf_hit",
    "pf_mb",
    "t_pf",
    "t_lg2rows",
    "dec_picks",
    "dec_hits",
    "host_experts",
    "fetched",
    "t_read",
    "t_host",
    "t_up",
    "t_fetch",
    "t_route",
    "t_craft",
    "t_sync",
    "t_wait",
    "t_exec",
    "t_cb",
    "t_fetch_w",
    "t_up_cp",
    "t_up_w",
    "t_up_conv",
    "t_s1",
    "t_s2",
    "t_choose",
    "t_prep",
)


def main():
    out_dir = sys.argv[1]
    os.makedirs(out_dir, exist_ok=True)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    # CQ1: the async fetch writes / the prefill staging copies
    ncq = 2 if "1" in (os.environ.get("HYB_FETCH_ASYNC"), os.environ.get("HYB_STREAM_CQ1")) else 1
    mesh = ttnn.open_mesh_device(
        ttnn.MeshShape(1, 4),
        l1_small_size=24576,
        trace_region_size=96_000_000,
        num_command_queues=ncq,
    )
    res = {"runs": []}
    try:
        dr = load_drafter(mesh, K + 1)
        gen = _port("generator").build_generator(
            f"models/autoports/{PORT}",
            mesh,
            max_seq_len=int(os.environ.get("POC_MAX_SEQ", "4096")),
            warm_decode_trace=False,
            warm_prefill_trace=False,
        )
        srv = DFlashServing(gen, K, drafter=dr)
        if os.environ.get("POC_PROF_DRAIN") == "1":  # device profiler: drain its buffers after every pass
            step = srv.step

            def drained(*a, **kw):
                out = step(*a, **kw)
                ttnn.ReadDeviceProfiler(mesh)
                return out

            srv.step = drained
        srv.warm()
        if os.environ.get("DFLASH_MEMONLY") == "1":
            print("MEMONLY " + json.dumps(gen._memory_report()), flush=True)
            return
        import prompts_v2

        start = int(os.environ.get("POC_PROMPT_START", "0"))
        prompts = prompts_v2.HELDOUT[start : start + int(os.environ.get("POC_PROMPTS", "2"))]
        long_n = int(os.environ.get("POC_LONG", "0") or 0)
        if long_n:  # a realistic long input: source files of this repo (POC_LONG_FILES), cut to ~long_n tokens
            files = os.environ.get("POC_LONG_FILES", "/poc/hybrid/patch.py,/poc/dflash_serve.py,/poc/dflash_tt.py")
            body = "\n\n".join(f"# file: {os.path.basename(f)}\n" + open(f).read() for f in files.split(","))
            text = gen.tokenizer.decode(gen.tokenizer.encode(body)[:long_n])
            prompts.append("Here is some Python code:\n\n" + text + "\n\nExplain what this code does, file by file.")
        # warm-up: one throwaway generation first, so the measured prompts see filled LRU slots and touched pinned /
        # locked host pages (a serving process does this at startup: DFlashServing.capture)
        warm_n = int(os.environ.get("POC_WARMUP", "64"))
        if warm_n:
            r = srv.generate("Write a haiku about the sea.", warm_n, gen.page_table[:1])
            print(f"WARMUP done ({r['tok_s']} tok/s)", flush=True)
        # POC_FETCH_SWEEP="f:F,f:F": the prompts once per fetch setting (fraction f of the misses, at most F per layer and
        # step), in this process (direct fetch: both are plain attributes; the LRU keeps its contents across settings)
        # "f:F:a" = the same with the asynchronous CQ1 fetch (HYB_FETCH_ASYNC; the mesh is opened with 2 CQs then)
        # "dyn:B:F[:a]" = the dynamic policy (HYB_FETCH_POLICY=dyn) with a budget of B fetched experts per pass;
        # "auto:B:F[:a]" = the same with the slack controller moving the budget from B (HYB_FETCH_POLICY=auto)
        sweep = [c.split(":") for c in os.environ.get("POC_FETCH_SWEEP", "").split(",") if c]
        runs = [
            (
                (c[0], float(c[1])) if c[0] in ("dyn", "auto") else ("fixed", float(c[0])),
                int(c[2] if c[0] in ("dyn", "auto") else c[1]),
                c[-1] == "a",
            )
            for c in sweep
        ] or [None]
        if sweep and not (HYB.fetch.direct and all(F <= HYB.fetch.D for _, F, _ in runs)):
            raise ValueError(f"POC_FETCH_SWEEP {runs} needs direct fetch and F <= HYB_LRU ({HYB.fetch.D})")
        jobs = [(cfg, p) for cfg in runs for p in prompts]
        # POC_WINDOW_SWEEP="1024,512": the prompts once per drafter context window (DFlashServing.window, read when a
        # prompt's drafter context is filled and at every refill)
        # POC_ATTR_SWEEP="early_cpu:0,1": each prompt once per value of that HYB attribute, back to back (an in-process A/B
        # of a runtime switch: warm-up and host jitter fall on both settings alike)
        attr, _, vals = os.environ.get("POC_ATTR_SWEEP", "").partition(":")
        attr_vals = [int(v) for v in vals.split(",") if v] if attr else []
        if attr and not hasattr(HYB, attr):
            raise ValueError(f"POC_ATTR_SWEEP: HYB has no attribute {attr!r}")
        if attr_vals:
            jobs = [((cfg, av), p) for cfg, p in jobs for av in attr_vals]
        else:
            jobs = [((cfg, None), p) for cfg, p in jobs]
        windows = [int(w) for w in os.environ.get("POC_WINDOW_SWEEP", "").split(",") if w]
        if windows:
            jobs = [((cfg, w), p) for cfg, p in jobs for w in windows]
        else:
            jobs = [((cfg, None), p) for cfg, p in jobs]
        # jobs: (((fetch cfg, attr value), window), prompt)
        # POC_AB_ROWS_OPS=1: every job twice, rows-mode attention through the chain, then the prep / post ops (the traces
        # are captured per prompt, so the switch takes effect at the next prompt)
        ab = os.environ.get("POC_AB_ROWS_OPS") == "1"
        rows_ops = {li: layer.attention._rows_ops for li, layer in enumerate(gen.model.layers)}
        if ab:
            jobs = [(cfg, p, on) for cfg, p in jobs for on in (False, True)]
        else:
            jobs = [(cfg, p, None) for cfg, p in jobs]
        warmed = {True}  # the startup warm-up ran with the ops
        default_window = srv.window
        for ((cfg, av), window), p_text, on in jobs:
            srv.window = window or default_window
            if av is not None:
                setattr(HYB, attr, type(getattr(HYB, attr))(av))
            if on is not None:
                for li, layer in enumerate(gen.model.layers):
                    layer.attention._rows_ops = rows_ops[li] if on else None
                if on not in warmed:  # the chain's programs are not in the program cache yet (the capture needs them)
                    srv.ver.release()
                    srv.ver.stage([0] * srv.R, list(range(8, 8 + srv.R)), gen.page_table[0])
                    srv.ver.run()
                    warmed.add(on)
            if cfg is not None:
                HYB.fetch.commit(HYB.slot_of)  # writes of an async setting land before the mode changes
                (pol, v), HYB.fetch.F, HYB.fetch.async_on = cfg
                HYB.fetch.policy = pol
                HYB.fetch.auto = pol == "auto"
                if pol in ("dyn", "auto"):
                    HYB.fetch.budget = v
                else:
                    HYB.fetch.f = v
                if cfg[2] and ncq < 2:
                    raise ValueError("an async fetch setting needs HYB_FETCH_ASYNC=1 (two command queues)")
            r = srv.generate(p_text, NEW, gen.page_table[:1], force=forced.get(p_text[:80]))
            if HYB.fetch.auto:  # the slack controller's budget trajectory (pass, budget, window mean ms)
                r["fetch_auto"] = HYB.fetch.budget_log[-12:]
                print("FETCH_AUTO", json.dumps(HYB.fetch.budget_log[-12:]), flush=True)
            if cfg is not None:
                r["fetch"] = list(cfg)
            if window:
                r["window"] = window
            if av is not None:
                r[attr] = av
            if on is not None:
                r["rows_ops"] = on
            n = max(1, r["passes"])
            r["hyb_per_pass_ms_or_n"] = {
                k: round(HYB.st.get(k, 0.0) / n * (1e3 if k.startswith("t_") else 1), 2) for k in HYB_KEYS
            }
            ids = r.pop("ids")
            text = gen.tokenizer.decode(ids)
            r = {"prompt": p_text[:80], **r}
            print("DFLASHT " + json.dumps(r), flush=True)
            print("TEXT " + json.dumps(text[:300]), flush=True)
            if not res["runs"]:
                print("MEMONLY " + json.dumps(gen._memory_report()), flush=True)
            res["runs"].append({**r, "text": text, "ids": ids})
            json.dump(res, open(os.path.join(out_dir, "result.json"), "w"), indent=1)
    finally:
        HYB.fetch.release()  # pinned regions / views before the mesh closes
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()
