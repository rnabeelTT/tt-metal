"""DeepPlanning (Qwen) through the official code: QwenLM/Qwen-Agent benchmark/deepplanning
(pinned 31a4d36d123688581a9e9744427272b33ce940e0; data HF Qwen/DeepPlanning @213876cc).

Two domains, both agentic with deterministic tool sandboxes (no simulated user):
  * Travel planning: 120 tasks x {zh, en}. Agent (official ToolsFnAgent + system prompt) queries flights/trains/
    hotels/restaurants/attractions/routes and writes a <plan>. Official stage 2 converts the free-text plan to JSON
    with an LLM (`qwen-plus`, FORMAT_CONVERT_PROMPT, max_tokens 10240); stage 3 scores it with rule-based
    commonsense + personalized constraints. We emit stage 2 as judge JSONL (or run it inline against
    DP_CONVERTER_BASE_URL/DP_CONVERTER_MODEL), then run stage 3 at ingest.
  * Shopping planning: levels 1/2/3 (50/50/20 cases). Agent edits a cart through tools; scoring is rule-based
    (cart vs ground-truth products/coupons). Scored immediately.
Metric (aggregate_results.py "avg_acc", the leaderboard "Avg Acc."): mean of shopping weighted_average_case_score
and travel case_acc (case_acc averaged over zh and en). Leaderboard rows satisfy Avg = (travel Case Acc +
shopping Case Acc)/2 (e.g. Qwen3-Max thinking (13.8+45.6)/2=29.7). Leaderboard averages 4 runs: TRIALS=4.

Env: TRIALS (1) MAX_LLM_CALLS (400, run_all.sh) DP_LANGS (zh,en) DP_LEVELS (1,2,3) DP_DOMAINS (travel,shopping)
     DP_CONVERTER_BASE_URL / DP_CONVERTER_MODEL / DP_CONVERTER_API_KEY (inline conversion instead of judge JSONL)
     LIMIT (first N tasks per language / per level).
"""
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from common.core import finish, read_jsonl
from common.judge import write_judge_inputs

BENCH_NAME = "deepplanning"
DP = Path.home() / "evals/deepplanning"
SRC = DP / "qwen-agent/benchmark/deepplanning"
PY = DP / ".venv/bin/python"
DATA = DP / "data"

JUDGE_INSTRUCTIONS = """DeepPlanning travel-plan CONVERSION (official stage 2, travelplanning/evaluation/convert_report.py; upstream runs
qwen-plus with max_tokens=10240). This is a parsing step, not a quality judgment: act as the conversion model.
Use `system_prompt` (FORMAT_CONVERT_PROMPT for the language) as the system message and `user_prompt` (the agent's
plan text) as the user message; `judge_prompt` is both concatenated. Losslessly convert the plan into the JSON
schema the prompt defines. Return {"id": ..., "raw_output": "<JSON>{...}</JSON>"} (parsed exactly like upstream:
text inside <JSON> tags, else the whole text, then json.loads) or {"id": ..., "plan": {...}}.
A row whose output does not parse counts as a failed conversion (no plan delivered), as upstream after 30 retries.
Rule-based scoring of the converted plans runs at ingest."""

VERDICT_SCHEMA = {
    "type": "object",
    "properties": {"id": {"type": "string"}, "raw_output": {"type": "string"}, "plan": {"type": "object"}},
    "required": ["id"],
}


def _sh(cmd, log: Path, env=None):
    with open(log, "a") as f:
        return subprocess.Popen([str(PY), "-u", str(DP / "dp_driver.py"), *cmd], stdout=f, stderr=subprocess.STDOUT,
                                env=env or os.environ.copy())


def _dp_json(log: Path):
    last = None
    for ln in log.read_text().splitlines():
        if ln.startswith("DP_JSON "):
            last = json.loads(ln[8:])
    return last


def _prepare_code(cfg, code: Path, limit):
    if not code.exists():
        shutil.copytree(SRC, code, ignore=shutil.ignore_patterns("results", "database*", "result_report",
                                                                  "aggregated_results", "__pycache__"))
    for lang in ("zh", "en"):
        (code / "travelplanning/database").mkdir(parents=True, exist_ok=True)
        link = code / f"travelplanning/database/database_{lang}"
        if not link.exists():
            link.symlink_to(DATA / f"travel/database_{lang}")  # read-only CSV databases
        q = json.loads((SRC / f"travelplanning/data/travelplanning_query_{lang}.json").read_text())
        (code / f"travelplanning/data/travelplanning_query_{lang}.json").write_text(
            json.dumps(q[:limit] if limit else q, ensure_ascii=False, indent=1))
    for lv in (1, 2, 3):
        q = json.loads((SRC / f"shoppingplanning/data/level_{lv}_query_meta.json").read_text())
        (code / f"shoppingplanning/data/level_{lv}_query_meta.json").write_text(
            json.dumps(q[:limit] if limit else q, ensure_ascii=False, indent=1))
    agent = {"model_name": cfg.model, "model_type": "openai", "base_url": cfg.base_url,
             "api_key_env": "DP_AGENT_API_KEY", "temperature": cfg.temperature,
             "extra_body": {"top_p": cfg.top_p, "max_tokens": cfg.max_tokens}}
    if cfg.top_k and cfg.top_k > 0:
        agent["extra_body"]["top_k"] = cfg.top_k
    if cfg.presence_penalty is not None:
        agent["extra_body"]["presence_penalty"] = cfg.presence_penalty
    conv = {"model_name": os.environ.get("DP_CONVERTER_MODEL", "qwen-plus"), "model_type": "openai",
            "base_url": os.environ.get("DP_CONVERTER_BASE_URL", "http://127.0.0.1:9/v1"),
            "api_key_env": "DP_CONVERTER_API_KEY"}
    (code / "models_config.json").write_text(json.dumps({"models": {"agent": agent, "qwen-plus": conv}}, indent=2))


def _env(cfg):
    return {**os.environ, "DP_AGENT_API_KEY": os.environ.get("OPENAI_API_KEY", "EMPTY"),
            "DP_CONVERTER_API_KEY": os.environ.get("DP_CONVERTER_API_KEY", "EMPTY"),
            "REQUEST_TIMEOUT": str(cfg.timeout), "PYTHONUNBUFFERED": "1"}


def _shop_chain(cfg, code, tdir: Path, levels, workers, limit, env):
    """levels sequentially (as run.sh): fresh db copy -> inference -> rule-based eval. Resumable per level."""
    script = []
    for lv in levels:
        ldir = tdir / f"shop/level{lv}"
        if (ldir / "report/summary_report.json").exists():
            continue
        ids = [str(x["id"]) for x in json.loads((code / f"shoppingplanning/data/level_{lv}_query_meta.json").read_text())]
        db = ldir / "db"
        if db.exists():
            shutil.rmtree(db)
        db.mkdir(parents=True)
        for i in ids:
            shutil.copytree(DATA / f"shopping/database_level{lv}/case_{i}", db / f"case_{i}")
        script.append((lv, db, ldir / "report"))
    return script


def _run_all(cfg, out: Path, code: Path, trials, env):
    langs = os.environ.get("DP_LANGS", "zh,en").split(",")
    levels = [int(x) for x in os.environ.get("DP_LEVELS", "1,2,3").split(",")]
    domains = os.environ.get("DP_DOMAINS", "travel,shopping").split(",")
    streams = (len(langs) if "travel" in domains else 0) + (1 if "shopping" in domains else 0)
    workers = max(1, cfg.concurrency // max(1, streams))
    mlc = os.environ.get("MAX_LLM_CALLS", "400")
    (out / "logs").mkdir(parents=True, exist_ok=True)
    for t in range(trials):
        tdir = out / f"t{t}"
        procs = []
        if "travel" in domains:
            for lang in langs:
                procs.append((f"travel-{lang}", _sh(["travel-infer", "--code", str(code), "--lang", lang, "--out",
                                                     str(tdir / f"travel_{lang}"), "--workers", str(workers),
                                                     "--max-llm-calls", mlc], out / f"logs/t{t}_travel_{lang}.log", env)))
        shop_proc = None
        if "shopping" in domains:
            chain = _shop_chain(cfg, code, tdir, levels, workers, cfg.limit, env)
            if chain:
                cmds = " && ".join(
                    f"{PY} -u {DP}/dp_driver.py shop-infer --code {code} --level {lv} --db {db} --workers {workers} "
                    f"--max-llm-calls {mlc} && {PY} -u {DP}/dp_driver.py shop-eval --code {code} --level {lv} "
                    f"--db {db} --report {rep} > {rep.parent}/eval.log 2>&1" for lv, db, rep in chain)
                with open(out / f"logs/t{t}_shopping.log", "a") as f:
                    shop_proc = subprocess.Popen(["bash", "-c", cmds], stdout=f, stderr=subprocess.STDOUT, env=env)
                procs.append(("shopping", shop_proc))
        print(f"[deepplanning] trial {t}: {', '.join(n for n, _ in procs) or 'nothing to run'} "
              f"({workers} workers each), logs in {out}/logs", flush=True)
        bad = [n for n, p in procs if p.wait() != 0]
        if bad:
            sys.exit(f"deepplanning stage failed: {bad}; see {out}/logs")
    return langs, levels, domains


def _format_prompt(code: Path, lang: str) -> str:
    spec = importlib.util.spec_from_file_location("dp_prompts", code / "travelplanning/agent/prompts.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.get_format_convert_prompt(lang)


def _extract_json(text: str):
    """upstream extract_json_from_response + json.loads"""
    if not text:
        return None
    m = re.search(r"<JSON>([\s\S]*?)</JSON>", text)
    payload = m.group(1).strip() if m else text
    try:
        return json.loads(payload)
    except Exception:  # noqa: BLE001
        return None


def _conversion_rows(out: Path, code: Path, trials, langs):
    rows = []
    for t in range(trials):
        for lang in langs:
            d = out / f"t{t}/travel_{lang}"
            sp = _format_prompt(code, lang)
            for rep in sorted((d / "reports").glob("id_*.txt")):
                n = rep.stem.split("_")[1]
                if (d / "converted_plans" / f"id_{n}_converted.json").exists():
                    continue
                txt = rep.read_text()
                rows.append({"id": f"t{t}|{lang}|{n}", "bench": "deepplanning", "question": f"travel plan id_{n} ({lang})",
                             "reference": None, "model_answer": txt, "judge_prompt": sp + "\n\n" + txt,
                             "system_prompt": sp, "user_prompt": txt, "verdict_schema": VERDICT_SCHEMA,
                             "meta": {"trial": t, "lang": lang, "sample_id": n}})
    return rows


def _score(cfg, out: Path, code: Path, conf: dict):
    trials, langs, levels, domains = conf["trials"], conf["langs"], conf["levels"], conf["domains"]
    env = _env(cfg)
    per_trial = []
    for t in range(trials):
        rec = {}
        if "travel" in domains:
            tl = {}
            for lang in langs:
                log = out / f"logs/t{t}_travel_eval_{lang}.log"
                p = _sh(["travel-eval", "--code", str(code), "--lang", lang, "--out", str(out / f"t{t}/travel_{lang}")], log, env)
                p.wait()
                js = _dp_json(log) or {}
                m = js.get("metrics") or {}
                tl[lang] = {k: m.get(k, 0.0) for k in ("case_acc", "composite_score", "commonsense_score",
                                                        "personalized_score", "delivery_rate")}
            rec["travel"] = tl
            rec["travel_case_acc"] = sum(v["case_acc"] for v in tl.values()) / len(tl)
            rec["travel_composite"] = sum(v["composite_score"] for v in tl.values()) / len(tl)
        if "shopping" in domains:
            lv = {}
            for L in levels:
                js = _dp_json(out / f"t{t}/shop/level{L}/eval.log") or {}
                lv[L] = js
            n = sum(v.get("total_cases", 0) for v in lv.values())
            rec["shopping_levels"] = lv
            rec["shopping_weighted_case_score"] = (sum(v.get("average_case_score", 0) * v.get("total_cases", 0)
                                                       for v in lv.values()) / n) if n else 0.0
            exp = sum(v.get("total_expected_products", 0) for v in lv.values())
            rec["shopping_match_rate"] = sum(v.get("total_matched_products", 0) for v in lv.values()) / exp if exp else 0.0
            rec["shopping_valid"] = all(v.get("valid", False) for v in lv.values())
        if "travel" in domains and "shopping" in domains:
            rec["avg_acc"] = (rec["shopping_weighted_case_score"] + rec["travel_case_acc"]) / 2
        per_trial.append(rec)
    key = "avg_acc" if ("travel" in domains and "shopping" in domains) else (
        "travel_case_acc" if "travel" in domains else "shopping_weighted_case_score")
    val = 100 * sum(r[key] for r in per_trial) / len(per_trial)
    agg = {k: 100 * sum(r.get(k, 0) for r in per_trial) / len(per_trial)
           for k in ("travel_case_acc", "travel_composite", "shopping_weighted_case_score", "shopping_match_rate")}
    n_items = (len(langs) * len(json.loads((code / "travelplanning/data/travelplanning_query_zh.json").read_text()))
               if "travel" in domains else 0) + (sum(len(json.loads((code / f"shoppingplanning/data/level_{L}_query_meta.json").read_text())) for L in levels)
                                                  if "shopping" in domains else 0)
    finish(cfg, key, val, n_items, {"trials": trials, "means_over_trials_pct": agg, "per_trial": per_trial})


def _write_gold(code: Path):
    """Self-test only (STUB_GOLD_MAP set): shopping = replay the ground-truth cart (add_product_to_cart for every
    ground-truth product, add_coupon_to_cart for every ground-truth coupon); travel has no reference plan, so the
    stub answers with a fixed <plan> (exercises report -> conversion -> rule-based scoring, not correctness)."""
    from common.core import write_gold_map
    ent = []
    for lv in (1, 2, 3):
        for q in json.loads((code / f"shoppingplanning/data/level_{lv}_query_meta.json").read_text()):
            vc = json.loads((DATA / f"shopping/database_level{lv}/case_{q['id']}/validation_cases.json").read_text())
            calls = [{"function": {"name": "add_product_to_cart",
                                   "arguments": json.dumps({"product_id": p["product_id"], "quantity": 1})}}
                     for p in vc.get("ground_truth_products", [])]
            calls += [{"function": {"name": "add_coupon_to_cart",
                                    "arguments": json.dumps({"coupon_name": c, "quantity": n})}}
                      for c, n in (vc.get("ground_truth_coupons") or {}).items()]
            ent.append(([{"role": "user", "content": q["query"]}], {"content": "", "tool_calls": calls}))
    ent.append(([{"role": "user", "content": (
        "Check whether the items in the shopping cart meet the requirements. "
        "If not, add the required items to the cart. If there are multiple possible solutions, "
        "choose the optimal one. The final result should be based on the items in the cart. "
        "If the task is already complete, then stop.")}], "Done."))
    for lang in ("zh", "en"):
        for q in json.loads((code / f"travelplanning/data/travelplanning_query_{lang}.json").read_text()):
            ent.append(([{"role": "user", "content": q["query"]}],
                        "<plan>**Budget Summary**:\n**Total Estimated Budget: 100 RMB**\n---\n**Day 1:**\n"
                        "Current City: A\nAccommodation: -\n08:00-09:00 | buffer | stub</plan>"))
    write_gold_map(ent)


def main(cfg):
    out = cfg.out
    out.mkdir(parents=True, exist_ok=True)
    trials = int(os.environ.get("TRIALS", 1))
    code = out / "code"
    _prepare_code(cfg, code, cfg.limit)
    if os.environ.get("STUB_GOLD_MAP"):
        _write_gold(code)
    env = _env(cfg)
    langs, levels, domains = _run_all(cfg, out, code, trials, env)
    conf = {"trials": trials, "langs": langs, "levels": levels, "domains": domains}
    cfg.dump(module="deepplanning", harness="QwenLM/Qwen-Agent@31a4d36d123688581a9e9744427272b33ce940e0 benchmark/deepplanning",
             data="Qwen/DeepPlanning@213876cce679f993a476d01042e13d111c0e3648", dp=conf,
             converter=os.environ.get("DP_CONVERTER_BASE_URL") or "judge JSONL")
    (out / "dp_conf.json").write_text(json.dumps(conf))
    if "travel" in domains:
        if os.environ.get("DP_CONVERTER_BASE_URL"):
            for t in range(trials):
                for lang in langs:
                    _sh(["travel-convert", "--code", str(code), "--lang", lang, "--out", str(out / f"t{t}/travel_{lang}"),
                         "--workers", str(cfg.concurrency)], out / f"logs/t{t}_travel_convert_{lang}.log", env).wait()
        rows = _conversion_rows(out, code, trials, langs)
        if rows:
            jp = write_judge_inputs(out, rows, [], JUDGE_INSTRUCTIONS)
            finish(cfg, "avg_acc", None, len(rows), {"pending_conversions": len(rows)}, judge_pending=jp)
            return
    _score(cfg, out, code, conf)


def ingest(cfg, verdicts, missing):
    out = cfg.out
    code = out / "code"
    conf = json.loads((out / "dp_conf.json").read_text())
    bad = 0
    for r in read_jsonl(out / "judge_inputs.jsonl"):
        v = verdicts.get(r["id"])
        plan = None
        if v is not None:
            plan = v.get("plan") if isinstance(v.get("plan"), (dict, list)) else _extract_json(v.get("raw_output") or "")
        t, lang, n = r["id"].split("|")
        if plan is None:
            bad += 1  # upstream: conversion failed after retries -> no converted plan -> scored as not delivered
            continue
        (out / f"{t}/travel_{lang}/converted_plans/id_{n}_converted.json").write_text(
            json.dumps(plan, ensure_ascii=False, indent=2))
    if bad:
        print(f"[deepplanning] {bad} conversions missing/unparsable -> counted as undelivered plans")
    _score(cfg, out, code, conf)
