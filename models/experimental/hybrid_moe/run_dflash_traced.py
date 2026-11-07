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
)


def main():
    out_dir = sys.argv[1]
    os.makedirs(out_dir, exist_ok=True)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), l1_small_size=24576, trace_region_size=96_000_000)
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
        for p_text in prompts:
            r = srv.generate(p_text, NEW, gen.page_table[:1])
            n = max(1, r["passes"])
            r["hyb_per_pass_ms_or_n"] = {
                k: round(HYB.st[k] / n * (1e3 if k.startswith("t_") else 1), 2) for k in HYB_KEYS
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
