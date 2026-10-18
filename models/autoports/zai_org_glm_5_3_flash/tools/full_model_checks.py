# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage-6 free-running checks on one model load (the full model takes ~22 min to build): the readiness autoregressive
runner plus the shared qualitative suite, both through the public Glm53Generator (traced decode, on-device greedy
split sampling, token feedback on device).

    # device job (mesh opened like tools/readiness.py):
    python -m models.autoports.zai_org_glm_5_3_flash.tools.full_model_checks run \
        --hf-tokens doc/full_model/autoregressive/hf_tokens.json
    # host only, once the CPU HF controls exist (tools/hf_full_ref.py --mode qualitative):
    python -m models.autoports.zai_org_glm_5_3_flash.tools.full_model_checks compare

run (qualitative first, so the CPU HF autoregressive reference has time to finish):
  1. readiness_check.run_autoregressive (prompt file, 256 tokens) with the HF side precomputed by
     tools/hf_full_ref.py --mode autoregressive (tools/readiness.py --hf-tokens), the generator built once and shared;
     then the same prompt generated again to check greedy determinism across calls.
  2. qualitative suite: the prompts rendered by `tools/hf_full_ref.py --mode qualitative --render-only`
     (rendered_prompts.json, HF chat template, add_generation_prompt) re-checked against this tokenizer, each generated
     greedily via Glm53Generator.generate for up to --qual-max-new-tokens, stopping at the first generation_config eos
     id (the stop token is kept, as model.generate does). Writes tt_outputs.json and qualitative_outputs.json (the
     run_vllm_server format, for check_degenerate_output.py).
compare:
  token-level agreement (first divergence) and side-by-side texts of TT vs the HF controls -> comparison.json.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parents[1]
DOC = MODEL_DIR / "doc" / "full_model"
CKPT = os.environ.get("GLM53_CKPT", str(Path.home() / "glm53-flash-bringup/ckpt/glm53_flash_e192_fp8"))


class _Stop(Exception):
    pass


def eos_ids() -> list[int]:
    e = json.loads((Path(CKPT) / "generation_config.json").read_text())["eos_token_id"]
    return [int(e)] if isinstance(e, int) else [int(t) for t in e]


def generate_until_stop(gen, ids, max_new, stop_ids):
    """Glm53Generator.generate (traced, on-device greedy) observed through its next_input hook (which only returns
    None, i.e. never overrides the fed-back token); aborted after the first stop id."""
    seen = []

    def hook(i, pred):
        seen.append(int(pred))
        if int(pred) in stop_ids:
            raise _Stop

    t0 = time.time()
    try:
        preds = gen.generate(ids, max_new, next_input=hook, enable_trace=True)
        assert preds[: len(seen)] == seen
        seen = preds
    except _Stop:
        pass
    return seen, time.time() - t0


def run(a):
    import ttnn
    from models.autoports.zai_org_glm_5_3_flash.tools import readiness as R

    mesh = R._open_mesh(a.trace_region)
    try:
        from models.autoports.zai_org_glm_5_3_flash.tt.generator import build_generator

        t0 = time.time()
        gen = build_generator(MODEL_DIR, mesh)
        build_s = time.time() - t0
        print(f"generator built in {build_s:.0f} s", flush=True)
        run_with(gen, mesh, a, DOC, build_s)
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def run_with(gen, mesh, a, out_doc: Path, build_s: float = 0.0):
    """The qualitative suite + autoregressive check on an already-built generator; artifacts under out_doc
    (qualitative/, autoregressive/); prompts and the HF reference tokens come from doc/full_model."""
    from models.autoports.zai_org_glm_5_3_flash.tools import readiness as R

    out_doc = Path(out_doc)
    real_teardown = gen.teardown
    gen.teardown = lambda: None  # run_autoregressive tears its generator down; this one is shared
    tok = gen.tokenizer
    stop_ids = eos_ids()
    summary = {"build_s": round(build_s, 1), "env": {k: v for k, v in os.environ.items() if k.startswith("GLM53_")}}

    if not a.skip_qualitative:
        qd = out_doc / "qualitative"
        qd.mkdir(parents=True, exist_ok=True)
        rendered = json.loads((DOC / "qualitative" / "rendered_prompts.json").read_text())
        outs = []
        for r in rendered:
            mine = tok.apply_chat_template(
                [{"role": "user", "content": r["prompt"]}], tokenize=False, add_generation_prompt=True
            )
            mine_ids = tok.encode(mine, add_special_tokens=False)
            toks, secs = generate_until_stop(gen, r["token_ids"], a.qual_max_new_tokens, stop_ids)
            text = tok.decode(toks, skip_special_tokens=False)
            o = {
                "id": r["id"],
                "prompt": r["prompt"],
                "prompt_token_ids": r["token_ids"],
                "tt_tokenizer_render_matches": mine == r["rendered_text"] and mine_ids == r["token_ids"],
                "max_new_tokens": a.qual_max_new_tokens,
                "stopped_on_eos": bool(toks and toks[-1] in stop_ids),
                "num_tokens": len(toks),
                "token_ids": toks,
                "text": text,
                "first_256_text": tok.decode(toks[:256], skip_special_tokens=False),
                "seconds": round(secs, 1),
            }
            outs.append(o)
            print(f"QUAL {r['id']} {len(toks)} tok eos={o['stopped_on_eos']} {secs:.0f}s:\n{text}\n", flush=True)
            meta = {
                "generator": "Glm53Generator.generate (traced decode, on-device greedy split sampling), observed "
                "through next_input (never overrides), stopped after the first generation_config eos id",
                "checkpoint": CKPT,
                "stop_ids": stop_ids,
                "env": summary["env"],
            }
            (qd / "tt_outputs.json").write_text(
                json.dumps({"meta": meta, "outputs": outs}, indent=2, ensure_ascii=False) + "\n"
            )
            (qd / "qualitative_outputs.json").write_text(
                json.dumps(
                    [{"id": x["id"], "prompt": x["prompt"], "greedy_completion": x["text"]} for x in outs],
                    indent=2,
                    ensure_ascii=False,
                )
                + "\n"
            )
        summary["qualitative"] = [{k: o[k] for k in ("id", "num_tokens", "stopped_on_eos", "seconds")} for o in outs]
    if not a.skip_autoregressive:
        import readiness_check.run_autoregressive as ra

        deadline = time.time() + 60 * a.wait_hf_min  # the CPU HF reference may still be running
        while not Path(a.hf_tokens).exists() and time.time() < deadline:
            time.sleep(30)
        ref = R.patch_hf_reference(a.hf_tokens)
        ra._import_build_generator = lambda model_dir: (lambda **kw: gen)
        out_dir = out_doc / "autoregressive"
        t0 = time.time()
        ra.run_autoregressive(
            model_dir=MODEL_DIR,
            hf_model_id=CKPT,
            prompt_file=Path(a.prompt_file).resolve(),
            mesh_device=mesh,
            output_dir=out_dir,
            max_new_tokens=ref["max_new_tokens"],
        )
        R.annotate_hf_reference(out_dir, a.hf_tokens, ref)
        meta = json.loads((out_dir / "autoregressive_meta.json").read_text())
        first_s = time.time() - t0
        again, s2 = generate_until_stop(gen, meta["prompt_token_ids"], meta["max_new_tokens"], [])
        hf, tt = meta["hf"]["token_ids"], meta["tt"]["token_ids"]
        div = next((i for i, (x, y) in enumerate(zip(hf, tt)) if x != y), min(len(hf), len(tt)))
        meta["tt_repeat_identical"] = again == tt
        meta["hf_tt_first_divergence"] = div
        meta["hf_tt_matching_positions"] = sum(int(x == y) for x, y in zip(hf, tt))
        meta["tt_generator"] = {
            "path": "Glm53Generator.generate(next_input=None) -> traced decode + on-device greedy split sampling",
            "env": summary["env"],
            "first_run_s": round(first_s, 1),
            "repeat_run_s": round(s2, 1),
        }
        (out_dir / "autoregressive_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
        summary["autoregressive"] = {
            "repeat_identical": again == tt,
            "first_divergence": div,
            "hf_tokens": len(hf),
            "tt_tokens": len(tt),
        }
        print(f"AUTOREGRESSIVE {json.dumps(summary['autoregressive'])}", flush=True)

    print("CHECKS_DONE " + json.dumps(summary), flush=True)
    gen.teardown = real_teardown
    return summary


def compare(a):
    qd = DOC / "qualitative"
    hf = {o["id"]: o for o in json.loads((qd / "hf_outputs.json").read_text())["outputs"]}
    tt = {o["id"]: o for o in json.loads((qd / "tt_outputs.json").read_text())["outputs"]}
    rows = []
    for pid, t in tt.items():
        h = hf.get(pid)
        if h is None:
            rows.append({"id": pid, "hf": None})
            continue
        n = min(len(h["token_ids"]), len(t["token_ids"]))
        div = next((i for i in range(n) if h["token_ids"][i] != t["token_ids"][i]), n)
        rows.append(
            {
                "id": pid,
                "prompt": t["prompt"],
                "compared_tokens": n,
                "first_divergence": div,
                "identical_prefix": div == n,
                "hf_tokens": len(h["token_ids"]),
                "tt_tokens": len(t["token_ids"]),
                "tt_stopped_on_eos": t["stopped_on_eos"],
                "hf_text": h["text"],
                "tt_text_same_budget": None,
            }
        )
    # TT text over the HF budget (for side-by-side reading)
    try:
        from transformers import AutoTokenizer

        tk = AutoTokenizer.from_pretrained(CKPT)
        for r in rows:
            if r.get("hf_tokens"):
                r["tt_text_same_budget"] = tk.decode(
                    tt[r["id"]]["token_ids"][: r["hf_tokens"]], skip_special_tokens=False
                )
    except Exception as e:  # noqa: BLE001
        print(f"tokenizer unavailable for side-by-side text ({e})")
    (qd / "comparison.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n")
    hfq = [{"id": h["id"], "prompt": h["prompt"], "greedy_completion": h["text"]} for h in hf.values()]
    (qd / "hf_qualitative_outputs.json").write_text(json.dumps(hfq, indent=2, ensure_ascii=False) + "\n")
    for r in rows:
        print({k: r.get(k) for k in ("id", "compared_tokens", "first_divergence", "identical_prefix")})


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "compare"])
    rc = Path(os.environ.get("TT_MODEL_BRINGUP_ROOT", "")) / "runtime" / "readiness_check"
    ap.add_argument("--prompt-file", type=Path, default=rc / "autoregressive_prompt.txt")
    ap.add_argument("--hf-tokens", type=Path, default=DOC / "autoregressive" / "hf_tokens.json")
    ap.add_argument("--qual-max-new-tokens", type=int, default=2048)
    ap.add_argument("--trace-region", type=int, default=int(os.environ.get("GLM53_TRACE_REGION", 256 << 20)))
    ap.add_argument("--skip-autoregressive", action="store_true")
    ap.add_argument("--wait-hf-min", type=float, default=45.0, help="wait this long for --hf-tokens to appear")
    ap.add_argument("--skip-qualitative", action="store_true")
    a = ap.parse_args(argv)
    run(a) if a.cmd == "run" else compare(a)


if __name__ == "__main__":
    sys.exit(main())
