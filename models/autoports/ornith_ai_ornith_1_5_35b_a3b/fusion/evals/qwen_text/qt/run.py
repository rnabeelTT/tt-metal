"""qwen_text runner: send a bench's frozen items to an OpenAI-compatible server,
score `content` only, write results, print one RESULT line.

  python -m qt.run <bench> --out DIR --base-url URL --model ID --conc N
  python -m qt.run <bench> --out DIR --score-only      # rescore existing responses
"""
import argparse
import asyncio
import json
import os
import random
import sys
import time

from . import registry
from .common import DATA, read_jsonl, run_requests, load_responses, write_jsonl


def load_items(bench):
    p = os.path.join(DATA, bench, "items.jsonl")
    if not os.path.exists(p):
        sys.exit(f"missing {p}; run: python -m qt.prepare {bench}")
    return read_jsonl(p), json.load(open(os.path.join(DATA, bench, "meta.json")))


def smoke_subset(items, n):
    """LIMIT=n: a seeded pick that still touches as many groups as possible."""
    rng = random.Random(0)
    by = {}
    for it in items:
        by.setdefault(it["group"], []).append(it)
    groups = sorted(by)
    rng.shuffle(groups)
    out, k = [], 0
    while len(out) < n and any(by.values()):
        g = groups[k % len(groups)]
        if by[g]:
            out.append(by[g].pop(rng.randrange(len(by[g]))))
        k += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bench")
    ap.add_argument("--out", required=True)
    ap.add_argument("--base-url")
    ap.add_argument("--model")
    ap.add_argument("--conc", type=int, default=32)
    ap.add_argument("--max-tokens", type=int, default=int(os.environ.get("MAX_TOKENS", 32768)))
    ap.add_argument("--temperature", type=float, default=float(os.environ.get("TEMPERATURE", 0.6)))
    ap.add_argument("--top-p", type=float, default=float(os.environ.get("TOP_P", 0.95)))
    ap.add_argument("--top-k", type=int, default=int(os.environ.get("TOP_K", 20)))
    ap.add_argument("--presence-penalty", type=float, default=(float(os.environ["PRESENCE_PENALTY"]) if os.environ.get("PRESENCE_PENALTY") else None))
    ap.add_argument("--limit", type=int, default=int(os.environ.get("LIMIT") or 0))
    ap.add_argument("--timeout", type=float, default=float(os.environ.get("REQUEST_TIMEOUT", 14400)))
    ap.add_argument("--retries", type=int, default=int(os.environ.get("MAX_RETRIES", 5)))
    ap.add_argument("--score-only", action="store_true")
    a = ap.parse_args()

    mod = registry.module(a.bench)
    a.out = os.path.abspath(a.out)
    os.makedirs(a.out, exist_ok=True)
    logf = open(os.path.join(a.out, "run.log"), "a")

    def log(s):
        line = f"[{time.strftime('%H:%M:%S')}] {s}"
        print(line, file=sys.stderr, flush=True)
        logf.write(line + "\n")
        logf.flush()

    items, meta = load_items(a.bench)
    os.environ["QT_OUT"] = a.out
    cfgp = os.path.join(a.out, "config.json")
    if a.score_only and os.path.exists(cfgp):
        a.limit = json.load(open(cfgp)).get("limit") or 0
    if a.limit:
        items = smoke_subset(items, a.limit)
    # SAMPLE_N: run-size cap (1-2 h runs): fixed random subset (seed 1234) of the frozen items
    sample_n = int(os.environ["SAMPLE_N"]) if os.environ.get("SAMPLE_N") else 0
    if a.score_only and os.path.exists(cfgp):
        sample_n = json.load(open(cfgp)).get("sample_n") or 0
    if sample_n and len(items) > sample_n:
        keep = set(sorted(random.Random(1234).sample(range(len(items)), sample_n)))
        items = [it for k, it in enumerate(items) if k in keep]
    resp_path = os.path.join(a.out, "responses.jsonl")

    if not a.score_only:
        gen = {"max_tokens": a.max_tokens, "temperature": a.temperature, "top_p": a.top_p, "top_k": a.top_k}
        if a.presence_penalty is not None:
            gen["presence_penalty"] = a.presence_penalty
        cfg = {"bench": a.bench, "base_url": a.base_url, "model": a.model, "conc": a.conc, "gen": gen,
               "limit": a.limit, "sample_n": sample_n, "n_items": len(items), "items_file": os.path.join(DATA, a.bench, "items.jsonl"),
               "dataset_meta": meta, "started": time.strftime("%Y-%m-%d %H:%M:%S")}
        if os.path.exists(cfgp):
            old = json.load(open(cfgp))
            if (old["gen"], old["model"], old["limit"]) != (gen, a.model, a.limit):
                sys.exit(f"{a.out} holds a run with different settings ({old['gen']}, limit {old['limit']}); "
                         "use a new label or remove it")
        json.dump(cfg, open(cfgp, "w"), indent=1)
        log(f"{a.bench}: {len(items)} items -> {a.base_url} model={a.model} conc={a.conc} gen={gen}")
        asyncio.run(run_requests(items, resp_path, a.base_url, a.model, a.conc, gen, a.timeout, a.retries, log))

    resp = load_responses(resp_path)
    rows, missing = [], []
    fin = {}
    for it in items:
        r = resp.get(it["id"])
        if not r or not r.get("ok"):
            missing.append(it["id"])
            content = ""
        else:
            content = r["content"]
            fin[r.get("finish_reason")] = fin.get(r.get("finish_reason"), 0) + 1
        rows.append({"item": it, "content": content})
    if hasattr(mod, "score_batch"):
        for r, sc in zip(rows, mod.score_batch(a.bench, rows, a.out, log)):
            r["score"] = sc
    else:
        for r in rows:
            r["score"] = mod.score(a.bench, r["item"], r["content"])
    write_jsonl(os.path.join(a.out, "scored.jsonl"),
                [{"id": r["item"]["id"], "group": r["item"]["group"], **r["score"],
                  "gold": r["item"].get("gold")} for r in rows])
    agg = mod.aggregate(a.bench, meta, rows)
    pending = agg.pop("pending", None)
    toks = [((resp.get(i["id"]) or {}).get("usage") or {}).get("completion_tokens") or 0 for i in items]
    summ = {"bench": a.bench, **{k: v for k, v in agg.items()}, "n": len(items), "n_failed_requests": len(missing),
            "finish_reasons": fin, "mean_completion_tokens": sum(toks) / max(1, len(toks)),
            "full_size": meta.get("full_size"), "sample": meta.get("sample"), "limit": a.limit, "sample_n": sample_n}
    json.dump(summ, open(os.path.join(a.out, "summary.json"), "w"), indent=1, ensure_ascii=False)
    if missing:
        log(f"WARNING: {len(missing)} items have no successful response (scored as wrong); rerun the same command to retry them")
    val = agg["value"]
    vs = "NA" if val is None else f"{val:.2f}"
    line = (f"RESULT {a.bench} {agg['metric']}={vs} n={len(items)} failed={len(missing)} "
            f"truncated={fin.get('length', 0)} mean_out_tok={summ['mean_completion_tokens']:.0f}")
    for k in agg.get("result_extra", []):
        v = agg[k]
        line += f" {k}={v:.2f}" if isinstance(v, float) else f" {k}={'NA' if v is None else v}"
    log(line)
    print(line)
    if pending:          # e.g. "JUDGE_PENDING <path>" / "SCORE_PENDING <cmd>"
        log(pending)
        print(pending)


if __name__ == "__main__":
    main()
