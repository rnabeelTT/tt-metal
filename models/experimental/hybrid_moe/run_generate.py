# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Hybrid-MoE prototype driver: build the Qwen3.5 autoport with the hybrid overlay and generate greedily (eager).

Env: POC_PROMPT_SET (train | heldout), POC_SKIP_SWEEP + POC_TF_REF (skip_sweep), HF_MODEL (122B or 397B snapshot id), HYB_SLOTS, HYB_PLAN, POC_PROMPTS (count), POC_PROMPT_START, POC_NEW_TOKENS, POC_LAYERS
(comma list -> reduced model), POC_MAX_SEQ (default 2048), HYB_RECORD=1 (save decode routing as routing.npz).
Writes <out_dir>/result.json.
"""

import traceback
import json
import os
import sys
import time

sys.path.insert(0, "/poc")
import hybrid  # noqa: E402

HYB = hybrid.install()

import numpy as np  # noqa: E402
import torch  # noqa: E402

import ttnn  # noqa: E402
from capture_routing import PROMPTS  # noqa: E402

if os.environ.get("POC_PROMPT_SET"):  # prompts_v2: "train", "heldout" or "all" (= train then heldout)
    import prompts_v2  # noqa: E402

    PROMPTS = {"train": prompts_v2.TRAIN, "heldout": prompts_v2.HELDOUT,
               "all": prompts_v2.TRAIN + prompts_v2.HELDOUT}[os.environ["POC_PROMPT_SET"]]
from models.autoports.qwen_qwen3_5_122b_a10b.tt.generator import build_generator  # noqa: E402

N_PROMPTS = int(os.environ.get("POC_PROMPTS", "2"))
PROMPT_START = int(os.environ.get("POC_PROMPT_START", "0"))  # held-out prompts: start past the plan's
NEW = int(os.environ.get("POC_NEW_TOKENS", "64"))


def main():
    out_dir = sys.argv[1]
    os.makedirs(out_dir, exist_ok=True)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), l1_small_size=24576, trace_region_size=64_000_000)
    res = {"env": {k: v for k, v in os.environ.items() if k.startswith(("HYB_", "POC_", "HF_MODEL", "QWEN35_"))}}
    try:
        t0 = time.time()
        kw = {}
        if os.environ.get("POC_LAYERS"):
            kw["layer_indices"] = [int(i) for i in os.environ["POC_LAYERS"].split(",")]
        gen = build_generator(
            "models/autoports/qwen_qwen3_5_122b_a10b",
            mesh,
            max_seq_len=int(os.environ.get("POC_MAX_SEQ", "2048")),
            warm_decode_trace=os.environ.get("POC_TRACE") == "1",
            warm_prefill_trace=False,
            **kw,
        )
        res["build_s"] = round(time.time() - t0, 1)
        res["E"], res["C"] = HYB.E, HYB.C
        print(f"BUILD {res['build_s']}s E={HYB.E} C={HYB.C}", flush=True)
        res["memory"] = gen._memory_report()
        print("MEMORY " + json.dumps(res["memory"]), flush=True)
        tok = gen.tokenizer
        if os.environ.get("POC_SKIP_SWEEP"):
            return skip_sweep(gen, tok, res, out_dir)
        runs = []
        rec_arrays = {}
        for i, p in enumerate(PROMPTS[PROMPT_START : PROMPT_START + N_PROMPTS], start=PROMPT_START):
            chat = tok.apply_chat_template(
                [{"role": "user", "content": p}], add_generation_prompt=True, enable_thinking=False, tokenize=False
            )
            ids = tok.encode(chat, add_special_tokens=False)
            HYB.reset_stats()
            HYB.record = [] if os.environ.get("HYB_RECORD") == "1" else None
            from hybrid.seg import SEG

            SEG.stats.update(replays=0, segments=0, t_first=None)
            t1 = time.time()
            out = gen.generate(list(ids), NEW, enable_trace=os.environ.get("POC_TRACE") == "1")
            dt = time.time() - t1
            st = dict(HYB.st)
            from hybrid.seg import SEG

            st["seg"] = dict(SEG.stats)
            text = tok.decode(out)
            r = {
                "prompt": p[:80],
                "prompt_tokens": len(ids),
                "wall_s": round(dt, 1),
                "ms_per_token_incl_prefill": round(1e3 * dt / NEW, 1),
                # traced decode only: prefill (TTFT) ends at the first trace replay
                "ttft_s": round(st["seg"]["t_first"] - t1, 2) if st["seg"].get("t_first") else None,
                "decode_ms_per_token": round(1e3 * (t1 + dt - st["seg"]["t_first"]) / max(1, st["seg"]["replays"]), 1)
                if st["seg"].get("t_first") else None,
                "hit_rate": round(st["hits"] / max(1, st["picks"]), 4),  # incl. prefill (padded bucket rows)
                "decode_hit_rate": round(st.get("dec_hits", 0) / max(1, st.get("dec_picks", 0)), 4),
                "decode_steps": st["steps"],
                "trace_replays": st["seg"]["replays"],
                "dyn_hits": st["dyn_hits"],
                "promoted": st["promoted"],
                "promote_s": round(st["t_promote"], 2),
                "host_s": round(st["t_host"], 2),
                "readback_s": round(st["t_read"], 2),
                "upload_s": round(st["t_up"], 2),
                "sync_s": round(st.get("t_sync", 0.0), 2),
                "upload_conv_s": round(st.get("t_up_conv", 0.0), 2),
                "lib_load_s": round(st.get("t_lib_load", 0.0), 2),
                "route_s": round(st.get("t_route", 0.0), 2),
                "craft_s": round(st.get("t_craft", 0.0), 2),
                "skipped": st.get("skipped", 0),
                "prefill_rows": st["prefill_rows"],
                "prefill_miss_pairs": st["prefill_miss_pairs"],
                "text": text,
                "tokens": [int(t) for t in out],
            }
            runs.append(r)
            print(f"P{i} {json.dumps({k: v for k, v in r.items() if k not in ('text', 'tokens')})}", flush=True)
            print(f"P{i} TEXT {text[:300]!r}", flush=True)
            if HYB.record:
                layers = sorted({rec[0] for rec in HYB.record})
                nl = len(layers)
                steps = len(HYB.record) // nl
                recs = HYB.record[: steps * nl]
                rec_arrays[f"p{i}"] = np.stack([rec[1].numpy() for rec in recs]).reshape(steps, nl, -1)
                rec_arrays[f"s{i}"] = np.stack([rec[2].numpy() for rec in recs]).reshape(steps, nl, -1).astype(np.float16)
                np.savez_compressed(os.path.join(out_dir, "routing.npz"), layers=np.array(layers), **rec_arrays)
            res["runs"] = runs
            json.dump(res, open(os.path.join(out_dir, "result.json"), "w"), indent=1)
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def skip_sweep(gen, tok, res, out_dir):
    """POC_SKIP_SWEEP=t0,t1,...: per threshold HYB_SKIP and prompt, (1) teacher-forced top-1 over the prompt + the
    reference continuation (POC_TF_REF: a result.json with "tokens"), against the reference tokens and against the
    first threshold's predictions, (2) a traced decode of POC_NEW_TOKENS for speed. One model build for all."""
    import torch
    from hybrid.seg import SEG

    taus = [float(t) for t in os.environ["POC_SKIP_SWEEP"].split(",")]
    ref = {r["prompt"]: r["tokens"] for r in json.load(open(os.environ["POC_TF_REF"]))["runs"]}
    base, rows = {}, []
    for tau in taus:
        HYB.skip = tau
        for i, p in enumerate(PROMPTS[PROMPT_START : PROMPT_START + N_PROMPTS], start=PROMPT_START):
            chat = tok.apply_chat_template(
                [{"role": "user", "content": p}], add_generation_prompt=True, enable_thinking=False, tokenize=False
            )
            ids = tok.encode(chat, add_special_tokens=False)
            cont = ref[p[:80]]
            seq = ids + cont
            HYB.reset_stats()
            gen.reset()
            t1 = time.time()
            logits = gen.prefill_forward(torch.tensor(seq).reshape(1, -1), page_table=gen.page_table[:1],
                                         prompt_lens=[len(seq)], return_all_logits=True)
            tf_s = time.time() - t1
            pred = torch.as_tensor(logits).float()[0, len(ids) - 1 : len(seq) - 1].argmax(-1)
            st_tf = dict(HYB.st)
            agree_ref = float((pred == torch.tensor(cont)).float().mean())
            base.setdefault(i, pred)
            agree_base = float((pred == base[i]).float().mean())
            HYB.reset_stats()
            gen.reset()
            SEG.stats.update(replays=0, segments=0, t_first=None)
            t1 = time.time()
            out = gen.generate(list(ids), NEW, enable_trace=os.environ.get("POC_TRACE") == "1")
            dt = time.time() - t1
            st = dict(HYB.st)
            tf = SEG.stats.get("t_first")
            r = {
                "tau": tau,
                "prompt": i,
                "tf_tokens": len(cont),
                "tf_top1_vs_ref": round(agree_ref, 4),
                "tf_top1_vs_tau0": round(agree_base, 4),
                "tf_prefill_s": round(tf_s, 2),
                "tf_skipped_pairs": st_tf["skipped"],
                "tf_skipped_mass_per_pick": round(st_tf["skipped_mass"] / max(1, st_tf["picks"]), 4),
                "ttft_s": round(tf - t1, 2) if tf else None,
                "decode_ms_per_token": round(1e3 * (t1 + dt - tf) / max(1, SEG.stats["replays"]), 1) if tf else None,
                "decode_hit_rate": round(st["dec_hits"] / max(1, st["dec_picks"]), 4),
                "decode_skipped_per_token": round(st["skipped"] / max(1, NEW), 2),
                "text": tok.decode(out)[:200],
            }
            rows.append(r)
            print(f"SWEEP {json.dumps(r)}", flush=True)
            res["sweep"] = rows
            json.dump(res, open(os.path.join(out_dir, "result.json"), "w"), indent=1)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush()
        os._exit(1)  # the hybrid prep thread pool and device threads otherwise keep a failed run alive
    sys.stdout.flush()
    os._exit(0)
