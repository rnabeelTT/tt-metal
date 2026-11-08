# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""DFlash speculative decoding on the hybrid MiniMax-M2.7, traced (the step itself: dflash_serve.DFlashServing, shared
with vLLM serving). The drafter is built BEFORE the target.

env: POC_PROMPTS, POC_PROMPT_START, POC_NEW_TOKENS, POC_MAX_SEQ, POC_LONG (a ~N-token code prompt from this repo's
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
HYB_KEYS = (
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
)


def main():
    out_dir = sys.argv[1]
    os.makedirs(out_dir, exist_ok=True)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    ncq = 2 if os.environ.get("HYB_FETCH_ASYNC") == "1" else 1  # CQ1: the async fetch writes
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
        sweep = [c.split(":") for c in os.environ.get("POC_FETCH_SWEEP", "").split(",") if c]
        runs = [(float(c[0]), int(c[1]), len(c) > 2 and c[2] == "a") for c in sweep] or [None]
        if sweep and not (HYB.fetch.direct and all(F <= HYB.fetch.D for _, F, _ in runs)):
            raise ValueError(f"POC_FETCH_SWEEP {runs} needs direct fetch and F <= HYB_LRU ({HYB.fetch.D})")
        jobs = [(cfg, p) for cfg in runs for p in prompts]
        # POC_AB_ROWS_OPS=1: every job twice, rows-mode attention through the chain, then the prep / post ops (the traces
        # are captured per prompt, so the switch takes effect at the next prompt)
        ab = os.environ.get("POC_AB_ROWS_OPS") == "1"
        rows_ops = {li: layer.attention._rows_ops for li, layer in enumerate(gen.model.layers)}
        if ab:
            jobs = [(cfg, p, on) for cfg, p in jobs for on in (False, True)]
        else:
            jobs = [(cfg, p, None) for cfg, p in jobs]
        warmed = {True}  # the startup warm-up ran with the ops
        for cfg, p_text, on in jobs:
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
                HYB.fetch.f, HYB.fetch.F, HYB.fetch.async_on = cfg
                if cfg[2] and ncq < 2:
                    raise ValueError("an async fetch setting needs HYB_FETCH_ASYNC=1 (two command queues)")
            r = srv.generate(p_text, NEW, gen.page_table[:1])
            if cfg is not None:
                r["fetch"] = list(cfg)
            if on is not None:
                r["rows_ops"] = on
            n = max(1, r["passes"])
            r["hyb_per_pass_ms_or_n"] = {
                k: round(HYB.st.get(k, 0.0) / n * (1e3 if k.startswith("t_") else 1), 2) for k in HYB_KEYS
            }
            text = gen.tokenizer.decode(r.pop("ids"))
            r = {"prompt": p_text[:80], **r}
            print("DFLASHT " + json.dumps(r), flush=True)
            print("TEXT " + json.dumps(text[:300]), flush=True)
            if not res["runs"]:
                print("MEMONLY " + json.dumps(gen._memory_report()), flush=True)
            res["runs"].append({**r, "text": text})
            json.dump(res, open(os.path.join(out_dir, "result.json"), "w"), indent=1)
    finally:
        HYB.fetch.release()  # pinned regions / views before the mesh closes
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()
