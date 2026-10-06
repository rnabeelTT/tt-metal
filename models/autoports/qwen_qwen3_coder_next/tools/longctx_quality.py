#!/usr/bin/env python3
"""Long-context text-quality gate for the served model.

Why this exists: the bundle's original validation ran 112-character prompts and ~200-token
answers, and gated on teacher-forcing top1 against a 100-token reference. Both are blind to
the failure that reached users - a wrong token sampled out of a quantization-noisy logit
tail during a long agentic turn, which the model then writes around, so the answer stays
on-topic while individual words come out wrong ("interpreterterpreter", "Typpos", "Hom").

This stage closes that gap: it drives the real server over prompts at the context lengths
the bundle advertises, with the sampling parameters the bundle actually serves, and scores
the prose with tools/text_quality.py.

Run against a live server:
    python -m models.autoports.qwen_qwen3_coder_next.tools.longctx_quality \
        --url http://127.0.0.1:8000 --tag p300x2 [--gate 25]

Exit code is non-zero when the mean garble score exceeds --gate, so it can sit in a release
chain. Compare two runs of the SAME prompt set; the absolute number is not meaningful across
different prompt sets.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from text_quality import score_text  # noqa: E402

MODEL_DIR = Path(__file__).resolve().parents[1]
TT = MODEL_DIR / "tt"

# Real source files, so the probe is deterministic and needs no external corpus. Chosen to
# straddle the gap between the 2048 tokens the datatype sweep validated and the 262144 the
# bundle advertises.
CONTEXT_SETS = {
    "ctx_small": ["decode_buckets.py", "precision.py"],  # ~6.5K tokens
    "ctx_mid": ["precision.py", "moe.py", "weights.py"],  # ~18K tokens
    "ctx_large": ["moe.py", "weights.py", "model.py", "generator_vllm.py"],  # ~38K tokens
}

ASK = (
    "Read the files above carefully.\n\n"
    "Write a detailed prose explanation, for an engineer new to this code, of:\n"
    "1. What each file is responsible for and how they fit together.\n"
    "2. How numerical precision is configured and where it is applied.\n"
    "3. How the mixture-of-experts routing works at decode time.\n"
    "4. Any design decisions that look deliberate and why they were probably made.\n\n"
    "Write at least 600 words of flowing prose. Do not reproduce the code; explain it."
)

PROSE_LONG = (
    "You are in a long technical conversation with an engineer. Explain, in at least 600 "
    "words of careful prose, why a quantized language model running on custom accelerator "
    "hardware might occasionally emit a wrong word, what the mechanisms are, and how an "
    "engineer would diagnose and fix it. Be precise and use correct terminology throughout."
)

# What the bundle serves by default must be the first entry: that is the configuration the
# gate is really protecting.
SETTINGS = {
    "served_default": None,  # filled from --served-sampling
    "checkpoint_t1.0": {"temperature": 1.0, "top_p": 0.95, "top_k": 40},
    "coder_rec_t0.7": {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "repetition_penalty": 1.05},
    "t0.6_k20": {"temperature": 0.6, "top_p": 0.95, "top_k": 20},
    "greedy": {"temperature": 0.0},
}


def build_prompts(names):
    out = {}
    for n in names:
        if n == "prose_long":
            out[n] = PROSE_LONG
            continue
        files = CONTEXT_SETS[n]
        parts = [f"===== FILE: tt/{f} =====\n{(TT / f).read_text()}" for f in files]
        out[n] = "\n\n".join(parts) + "\n\n" + ASK
    return out


def post(url, payload, timeout=1800):
    req = urllib.request.Request(
        url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-Coder-Next")
    ap.add_argument("--prompts", default="ctx_small,ctx_mid,prose_long")
    ap.add_argument("--settings", default="served_default")
    ap.add_argument(
        "--served-sampling",
        default="{}",
        help="JSON of the sampling params the bundle serves by default (empty = server default)",
    )
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=900)
    ap.add_argument(
        "--gate",
        type=float,
        default=None,
        help="fail (exit 1) if the mean garble_per_1k of served_default exceeds this",
    )
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument(
        "--extra-body",
        default="{}",
        help="JSON merged into every request body (e.g. chat_template_kwargs for a thinking model)",
    )
    a = ap.parse_args(argv)
    extra_body = json.loads(a.extra_body)

    SETTINGS["served_default"] = json.loads(a.served_sampling)
    prompts = build_prompts([p for p in a.prompts.split(",") if p])
    settings = [s for s in a.settings.split(",") if s]

    # No `seed` is sent. A seeded request forces the batch to the FULL decode width
    # (generator_vllm._select_decode_bucket: seeded rows run at max_batch_size so the numerics
    # stay composition-independent), which is both ~5x slower at one user and NOT the width a
    # single interactive user actually hits. This probe must measure the served path, so it
    # accepts run-to-run variation and uses repeats instead.
    rows = []
    for sname in settings:
        sp = SETTINGS[sname] or {}
        for pname, prompt in prompts.items():
            reps = 1 if sp.get("temperature", 1.0) == 0.0 else a.repeats
            for rep in range(reps):
                payload = {
                    "model": a.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": a.max_tokens,
                    **sp,
                    **extra_body,
                }
                t0 = time.time()
                try:
                    resp = post(a.url, payload)
                except Exception as e:  # noqa: BLE001
                    print(f"  !! {sname}/{pname}/{rep}: {e}", flush=True)
                    continue
                txt = resp["choices"][0]["message"]["content"] or ""
                sc = score_text(txt, prompt)
                u = resp.get("usage", {})
                rows.append(
                    {
                        "setting": sname,
                        "sampling": sp,
                        "prompt": pname,
                        "rep": rep,
                        "prompt_tokens": u.get("prompt_tokens"),
                        "completion_tokens": u.get("completion_tokens"),
                        "elapsed_s": round(time.time() - t0, 1),
                        **{
                            k: sc[k]
                            for k in (
                                "n_words",
                                "nonword",
                                "intraword_dup",
                                "dup_adjacent",
                                "dup_punct",
                                "delim_imbalance",
                                "language_skip",
                                "garble_per_1k",
                            )
                        },
                        "examples": sc["examples"],
                        "text": txt,
                    }
                )
                r = rows[-1]
                print(
                    f"  {sname:16s} {pname:10s} rep{rep}  ctx={r['prompt_tokens']:>6} out={r['completion_tokens']:>4}"
                    f"  garble/1k={r['garble_per_1k']:7.2f}  nonword={r['nonword']:3d}"
                    f"  {r['examples']['nonword'][:6]}",
                    flush=True,
                )

    summary = {}
    for sname in settings:
        v = [r["garble_per_1k"] for r in rows if r["setting"] == sname]
        if v:
            summary[sname] = {
                "runs": len(v),
                "garble_per_1k_mean": round(statistics.mean(v), 2),
                "garble_per_1k_max": round(max(v), 2),
                "nonword_total": sum(r["nonword"] for r in rows if r["setting"] == sname),
            }

    out = a.out or (MODEL_DIR / "doc" / "text_quality" / f"longctx_quality_{a.tag}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"tag": a.tag, "summary": summary, "rows": rows}, indent=1))
    print(f"\n=== {a.tag} ===")
    for k, v in summary.items():
        print(
            f"  {k:16s} mean={v['garble_per_1k_mean']:7.2f} max={v['garble_per_1k_max']:7.2f} "
            f"nonwords={v['nonword_total']:3d} (n={v['runs']})"
        )
    print("->", out)

    if a.gate is not None and "served_default" in summary:
        got = summary["served_default"]["garble_per_1k_mean"]
        if got > a.gate:
            print(f"GATE FAIL: served_default mean garble_per_1k {got} > {a.gate}")
            return 1
        print(f"GATE PASS: served_default mean garble_per_1k {got} <= {a.gate}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
