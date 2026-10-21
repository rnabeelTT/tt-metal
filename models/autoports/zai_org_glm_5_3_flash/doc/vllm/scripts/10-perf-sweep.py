"""GLM-5.3-Flash (E192) copy of the GLM-4.5-Air serving perf sweep (only the model constants changed) against a running OpenAI-compatible vLLM server (the tree server of 04-serve.sh on
:8131, or the tt-model bundle on :20000). Same methodology as the Ornith v2 / Coder-30B v2 cards: random-token prompts
of exact input length sent as token ids to plain /v1/completions (no chat template, so GLM's enable_thinking default
does not apply), greedy, ignore_eos (output length pinned), streamed with include_usage, closed-loop `users`
concurrent requests, N requests per row, one unrecorded warm-up per input length.
Per row: TTFT median (ms, time to the first streamed token chunk), TPOT ((E2EL - TTFT) / (out - 1), median over
requests), decode tok/s/user (1000 / TPOT), total output tok/s (all output tokens / row wall time), E2EL median (s),
prefill tok/s (input / TTFT, batch-1 rows). Each request's usage.prompt_tokens is checked against the intended input.

HOW TO RUN (host side; the server must already be up and nothing else should be sending traffic)
  source ~/glm45-air-bringup/env.sh            # python env + HF_HUB_OFFLINE=1 (tokenizer comes from the local cache)
  S=~/glm45-air-bringup/scripts; A=~/glm45-air-bringup/artifacts
  # 0) host-only check of the plan and prompt construction (no server, no device):
  python3 $S/perf-sweep.py /tmp/x.json --dry-run
  # 1) p300x2 (4 dies, default profile) - tree server held open by 04-serve.sh, then the sweep:
  MESH_DEVICE=P300x2 NOSTAGES=1 HOLD=1 bash $S/04-serve.sh
  python3 $S/perf-sweep.py $A/sweep_P300x2.json --url http://127.0.0.1:8131 --profile p300x2
  #    or in one go (04-serve.sh runs this script after boot, then stops the server):
  MESH_DEVICE=P300x2 NOSTAGES=1 SWEEP=1 bash $S/04-serve.sh          # -> $A/sweep_P300x2.json
  # 2) p300 (2 dies):
  MESH_DEVICE=P300 NOSTAGES=1 SWEEP=1 bash $S/04-serve.sh            # -> $A/sweep_P300.json
  # 3) the bundle (tt-model serve ... --port 20000 [--profile p300x2|p300]):
  python3 $S/perf-sweep.py $A/sweep_bundle_p300x2.json --url http://127.0.0.1:20000 --profile p300x2
  # p150x4 / p150x2 are the same meshes as p300x2 / p300 and are not run on hardware: no separate sweep.
  # 4) card tables / scorecard:
  python3 $S/card_tables.py "p300x2 (4 dies)=$A/sweep_P300x2.json" "p300 (2 dies)=$A/sweep_P300.json"
  python3 $S/scorecard.py $S/scorecard_cfg_glm.json $A/scorecard/glm45_air_scorecard.html
Rows (--rows scorecard, the default): batch 1 at 128/128, 1024/256, 4096/256, 16384/256, 32768/256, 65536/256,
130816/256 (130816 + 256 = 131072 = max_model_len), then users 2,4,8,16,32 at 128/128 and 1024/256.
The output JSON is appended row by row; rerunning with the same OUT.json resumes (finished rows are skipped).
Rows whose input + output exceed the server's max_model_len (read from /v1/models) are skipped and listed in
"skipped" instead of failing mid-sweep."""
import argparse
import concurrent.futures as cf
import json
import os
import random
import sys
import threading
import time

DEFAULT_MODEL = "zai-org/GLM-5.3-Flash"
HF_FUSED = os.environ.get(
    "GLM53_CKPT", os.path.expanduser("~/glm53-flash-bringup/ckpt/glm53_flash_e192_fp8")
)  # local tokenizer
GLM_EMBED_VOCAB = 154880  # GLM-5.3-Flash config.json text_config.vocab_size (embedding rows)
MAX_MODEL_LEN = 131072  # default only: 09-serve.sh SWEEP=1 passes --max-model-len from serving_profiles.json

ap = argparse.ArgumentParser(
    description=__doc__.split("\n\n")[0],
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="See the module docstring (head of this file) for per-profile invocations.",
)
ap.add_argument("out", help="output JSON (resumed if it exists)")
ap.add_argument("--url", default="http://127.0.0.1:8131", help="server base URL (04-serve.sh: 8131, bundle: 20000)")
ap.add_argument(
    "--model",
    default=os.environ.get("SWEEP_MODEL"),
    help="model name sent in requests (default: $SWEEP_MODEL, else the id the server lists in /v1/models)",
)
ap.add_argument(
    "--tokenizer",
    default=os.environ.get("SWEEP_TOKENIZER", DEFAULT_MODEL),
    help=f"tokenizer for the random prompts (default {DEFAULT_MODEL} from the HF cache, falls back to {HF_FUSED})",
)
ap.add_argument("--profile", default=None, help="label stored in the JSON (p300x2 | p150x4 | p300 | p150x2)")
ap.add_argument(
    "--rows",
    default="scorecard",
    choices=["scorecard", "grid", "default", "short"],
    help="scorecard: batch-1 ladder + concurrency at 128/1024 (default); grid/default: every input x every "
    "user count; short: grid over 128/1024/4096",
)
ap.add_argument("--sampled", action="store_true", help="also run the sampled (temperature 1.0) rows")
ap.add_argument("--inputs", default=None, help="comma list overriding the batch-1 input lengths (A/B runs)")
ap.add_argument("--max-users", type=int, default=32, help="max_num_seqs of the server")
ap.add_argument("--users", default="1,2,4,8,16,32", help="comma list of user counts to run")
ap.add_argument(
    "--max-model-len",
    type=int,
    default=None,
    help=f"override the server's max_model_len (default: /v1/models, else {MAX_MODEL_LEN})",
)
ap.add_argument(
    "--dry-run",
    action="store_true",
    help="no server: print the plan and verify the prompt construction (exact token counts, no specials)",
)
args = ap.parse_args()

from transformers import AutoTokenizer  # after argparse so --help is instant


def load_tokenizer(name):
    try:
        return AutoTokenizer.from_pretrained(name), name
    except Exception as e:  # offline cache miss -> the port's local copy of the tokenizer files
        print(f"tokenizer {name} unavailable ({type(e).__name__}); using {HF_FUSED}", file=sys.stderr)
        return AutoTokenizer.from_pretrained(HF_FUSED), HF_FUSED


tok, tok_src = load_tokenizer(args.tokenizer)
# Ordinary (non-added, non-special) ids only: tok.vocab_size is the base BPE vocab (151329 for GLM); the 36 added
# tokens (<|endoftext|>, [gMASK], <|user|>, <think>, ...) sit above it and are excluded explicitly as well.
EXCLUDED = set(tok.all_special_ids) | set(tok.get_added_vocab().values())
ORDINARY_HI = tok.vocab_size
assert ORDINARY_HI <= GLM_EMBED_VOCAB, (ORDINARY_HI, GLM_EMBED_VOCAB)
rng = random.Random(1234)


def prompt_ids(n):
    """n random ordinary token ids: sent as a token-id prompt to raw /v1/completions, so the server sees exactly n
    prompt tokens (no chat template, no BOS: GLM has none and vLLM adds no specials to token-id prompts)."""
    ids = []
    while len(ids) < n:
        t = rng.randrange(0, ORDINARY_HI)
        if t not in EXCLUDED:
            ids.append(t)
    return ids


INPUTS = [128, 1024, 4096, 16384, 32768, 65536, 130816]
out_len = lambda inp: 128 if inp == 128 else 256


def n_for(inp, users):
    return (4 if inp <= 4096 else 2 if inp <= 16384 else 1) * users


plan = []
users_list = [int(u) for u in args.users.split(",") if int(u) <= args.max_users]
batch1_inputs = [int(x) for x in args.inputs.split(",")] if args.inputs else INPUTS
if args.rows == "scorecard":
    # the published card layout: batch-1 latency at every input length, concurrency ladder at 128/128 and 1024/256
    for inp in batch1_inputs:
        plan.append((inp, out_len(inp), 1, n_for(inp, 1), 0.0))
    for inp in (128, 1024):
        for u in users_list:
            if u > 1:
                plan.append((inp, out_len(inp), u, n_for(inp, u), 0.0))
else:
    for inp in batch1_inputs if args.rows in ("grid", "default") else [128, 1024, 4096]:
        for u in users_list:
            plan.append((inp, out_len(inp), u, n_for(inp, u), 0.0))
if args.sampled:
    for inp in (128, 1024):
        for u in (1, 8, 32):
            if u <= args.max_users:
                plan.append((inp, out_len(inp), u, n_for(inp, u), 1.0))

if args.dry_run:
    mml = args.max_model_len or MAX_MODEL_LEN
    print(
        f"tokenizer {tok_src} ({type(tok).__name__}): base vocab {ORDINARY_HI}, {len(EXCLUDED)} excluded "
        f"added/special ids {min(EXCLUDED)}..{max(EXCLUDED)}, model embedding vocab {GLM_EMBED_VOCAB}"
    )
    print(f"bos_token={tok.bos_token!r}; tokenizer('') with specials -> {tok('')['input_ids']} (nothing prepended)")
    print(f"plan: {len(plan)} rows, max_model_len {mml}")
    ok = True
    for inp in sorted({p[0] for p in plan}):
        ids = prompt_ids(inp)
        o = max(p[1] for p in plan if p[0] == inp)
        bad = [t for t in ids if t in EXCLUDED or not 0 <= t < ORDINARY_HI]
        fits = inp + o <= mml and inp + 16 <= mml
        ok &= len(ids) == inp and not bad and fits
        print(
            f"  input {inp:>6}: {len(ids):>6} ids (exact={len(ids) == inp}), specials/out-of-range={len(bad)}, "
            f"min/max id {min(ids)}/{max(ids)}, +out {o} = {inp + o} <= {mml}: {fits}"
        )
    for inp, o, u, N, temp in plan:
        print(f"  ROW input {inp:>6} output {o} users {u:>2} N {N:>3} temperature {temp}")
    print("DRY RUN", "OK" if ok else "FAILED")
    sys.exit(0 if ok else 1)

import requests

served = {}
try:
    served = {m["id"]: m for m in requests.get(f"{args.url}/v1/models", timeout=30).json()["data"]}
except Exception as e:
    print(f"warning: /v1/models unreadable ({e})", file=sys.stderr)
MODEL = args.model or (DEFAULT_MODEL if DEFAULT_MODEL in served or not served else next(iter(served)))
if served and MODEL not in served:
    sys.exit(f"model {MODEL!r} not served at {args.url}; served: {list(served)}")
MML = args.max_model_len or (served.get(MODEL) or {}).get("max_model_len") or MAX_MODEL_LEN
print(
    f"=== sweep {MODEL} at {args.url} profile={args.profile} max_model_len={MML} tokenizer={tok_src} rows={len(plan)}",
    flush=True,
)


def one(ids, out_tokens, temperature):
    # stream_options.include_usage: the final chunk carries the exact completion count. Under MTP a stream chunk can
    # carry several tokens, so counting chunks undercounts (Coder-Next sweep: 47 of 128 tokens -> TPOT x4).
    # TTFT is the first chunk with a choice: vLLM only emits one once a token exists, and it can carry empty text
    # (a byte-level BPE token completing a UTF-8 sequence later), so waiting for non-empty text would add a step.
    body = {
        "model": MODEL,
        "prompt": ids,
        "max_tokens": out_tokens,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": temperature,
        **({"top_k": -1, "top_p": 1.0} if temperature > 0 else {}),
    }
    t0 = time.perf_counter()
    ttft = None
    n_out = 0
    n_prompt = None
    err = None
    try:
        with requests.post(f"{args.url}/v1/completions", json=body, stream=True, timeout=3600) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line or not line.startswith(b"data:"):
                    continue
                if line.strip() == b"data: [DONE]":
                    break
                d = json.loads(line[5:])
                u = d.get("usage")
                if u:
                    n_out = u.get("completion_tokens") or n_out
                    n_prompt = u.get("prompt_tokens", n_prompt)
                if not d.get("choices"):  # usage-only final chunk
                    continue
                if ttft is None:
                    ttft = time.perf_counter() - t0
                if not u:
                    n_out += 1  # placeholder until the usage chunk (one chunk per step, may hold several tokens)
    except Exception as e:  # timeout / HTTP error: counted in failed_requests, the sweep goes on
        err = f"{type(e).__name__}: {e}"[:300]
        ttft = None
    e2el = time.perf_counter() - t0
    return {"ttft": ttft, "e2el": e2el, "out": n_out, "prompt": n_prompt, "err": err}


def row(inp, out_tokens, users, N, temperature=0.0):
    prompts = [prompt_ids(inp) for _ in range(N)]
    results = [None] * N
    lock = threading.Lock()
    idx = {"i": 0}

    def worker():
        while True:
            with lock:
                i = idx["i"]
                idx["i"] += 1
            if i >= N:
                return
            results[i] = one(prompts[i], out_tokens, temperature)

    t0 = time.perf_counter()
    with cf.ThreadPoolExecutor(users) as ex:
        list(ex.map(lambda _: worker(), range(users)))
    wall = time.perf_counter() - t0
    errors = sorted({r["err"] for r in results if r["err"]})
    failed = sum(1 for r in results if r["ttft"] is None)  # a request that streamed no token (timeout / error)
    prompt_mismatch = sum(1 for r in results if r["prompt"] is not None and r["prompt"] != inp)
    results = [r for r in results if r["ttft"] is not None] or [{"ttft": float("nan"), "e2el": float("nan"), "out": 0}]
    N = len(results)
    ttfts = sorted(r["ttft"] for r in results)
    e2els = sorted(r["e2el"] for r in results)
    tpots = sorted((r["e2el"] - r["ttft"]) / max(1, r["out"] - 1) for r in results)
    med = lambda a: a[len(a) // 2] if len(a) % 2 else (a[len(a) // 2 - 1] + a[len(a) // 2]) / 2
    tot_out = sum(r["out"] for r in results)
    short = sum(1 for r in results if r["out"] < out_tokens)
    rec = {
        "input": inp,
        "output": out_tokens,
        "users": users,
        "N": N,
        "temperature": temperature,
        "ttft_ms": round(med(ttfts) * 1000, 1),
        "ttft_p99_ms": round(ttfts[min(N - 1, int(0.99 * N))] * 1000, 1),
        "tpot_ms": round(med(tpots) * 1000, 2),
        "decode_tps_user": round(1 / med(tpots), 1),
        "total_out_tps": round(tot_out / wall, 1),
        "e2el_s": round(med(e2els), 2),
        "wall_s": round(wall, 1),
        "prefill_tps": round(inp / med(ttfts), 0) if users == 1 else None,
        "out_tokens_total": tot_out,
        "short_outputs": short,
        "failed_requests": failed,
        "prompt_token_mismatch": prompt_mismatch,
    }
    if errors:
        rec["errors"] = errors[:3]
    if prompt_mismatch:
        print(f"WARNING input {inp}: {prompt_mismatch} requests reported prompt_tokens != {inp}", flush=True)
    return rec


out = {
    "model": MODEL,
    "url": args.url,
    "profile": args.profile,
    "max_model_len": MML,
    "tokenizer": tok_src,
    "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
    "rows": [],
    "skipped": [],
    "plan_rows": len(plan),
}
if os.path.exists(args.out):
    out = json.load(open(args.out))
    out.setdefault("skipped", [])
    if out.get("model") != MODEL:
        print(f"warning: resuming {args.out} recorded for {out.get('model')!r}, now {MODEL!r}", file=sys.stderr)
    done = {(r["input"], r["users"], r["temperature"]) for r in out["rows"]}
else:
    done = set()
warmed = set()
for inp, o, u, N, temp in plan:
    if (inp, u, temp) in done:
        continue
    if inp + o > MML:
        print(f"SKIP input {inp} + output {o} > max_model_len {MML}", flush=True)
        sk = {"input": inp, "output": o, "users": u, "reason": f"> max_model_len {MML}"}
        if sk not in out["skipped"]:  # resumed runs do not list it twice
            out["skipped"].append(sk)
        continue
    if inp not in warmed:
        one(prompt_ids(inp), 16, 0.0)
        warmed.add(inp)  # unrecorded warm-up per input length
    rec = row(inp, o, u, N, temp)
    out["rows"].append(rec)
    print("ROW " + json.dumps(rec), flush=True)
    json.dump(out, open(args.out, "w"), indent=1)
out["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
json.dump(out, open(args.out, "w"), indent=1)
print("=== sweep done", len(out["rows"]), "rows", f"({len(out['skipped'])} skipped)" if out["skipped"] else "")
