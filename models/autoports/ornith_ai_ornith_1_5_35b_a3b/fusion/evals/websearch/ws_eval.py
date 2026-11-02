"""WideSearch evaluation replay with the OFFICIAL evaluation code (ByteDance-Seed/WideSearch src/evaluation @9825ba7)
and LLM calls answered from a judge cache (Claude agents act as the official eval model gpt-4.1-2025-04-14).

  ws_eval_venv/bin/python ws_eval.py <run_dir>

Reads run_dir/{sample_ids.json, responses.jsonl, judge_cache.jsonl}; for every instance runs
src.evaluation.evaluation.evaluate_single_query with src.utils.llm.llm_completion replaced by a cache lookup.
LLM calls happen in up to three dependent stages per instance:
  colmap  response-table columns -> required columns (only when the column sets differ)
  keymap  primary-key alignment for unique columns scored by exact_match / llm_judge
  eval    llm_judge_column cell grading (after the key alignment)
A cache miss is recorded in run_dir/ws_pending.jsonl and the instance is left unscored for this round (colmap /
keymap misses stop the instance before any dependent call; independent calls of the same stage are gathered).
Scored instances: run_dir/ws_eval/<id>.json (EvaluationResult) + .csv (official per-cell table).
Prints one JSON line: {"scored": n, "pending_instances": k, "pending_prompts": m}.
"""
import dataclasses
import hashlib
import json
import sys
import threading
import types
from pathlib import Path

REPO = Path.home() / "evals/src/WideSearch"
DATA = Path.home() / "evals/data/widesearch"
sys.path.insert(0, str(REPO))


class Pending(BaseException):
    """Raised past the official try/except to stop an instance whose next LLM call depends on a missing one."""


CTX = threading.local()
CACHE: dict[str, str] = {}
PENDING: dict[str, dict] = {}


def pkey(prompt: str) -> str:
    return "ws:" + hashlib.sha256(prompt.encode()).hexdigest()[:32]


def llm_completion(messages, tools=None, model_config_name="default_eval_config"):
    prompt = messages if isinstance(messages, str) else messages[-1]["content"]
    k = pkey(prompt)
    if k in CACHE:
        return types.SimpleNamespace(content=CACHE[k])
    stage = "eval" if prompt.startswith("You are an expert in grading answers") else CTX.stage
    if stage == "eval" and CTX.keymap_missing:
        raise Pending()
    PENDING[k] = {"id": k, "stage": stage, "instance_id": CTX.iid, "prompt": prompt}
    CTX.missing += 1
    if stage == "colmap":
        raise Pending()
    if stage == "keymap":
        CTX.keymap_missing = True
    return None


# stub modules the official code imports but does not need here (API SDKs, HF download)
_llm = types.ModuleType("src.utils.llm")
_llm.llm_completion = llm_completion
sys.modules["src.utils.llm"] = _llm
for name, attrs in {"datasets": ["load_dataset"], "huggingface_hub": ["snapshot_download", "try_to_load_from_cache"]}.items():
    m = types.ModuleType(name)
    for a in attrs:
        setattr(m, a, lambda *x, **y: None)
    sys.modules.setdefault(name, m)

import pandas as pd  # noqa: E402
from loguru import logger  # noqa: E402

logger.remove()
logger.add(sys.stderr, level="WARNING")

import src.evaluation.evaluation as ev  # noqa: E402
from src.evaluation.data_loader import WideSearchQuery, WideSearchResponse  # noqa: E402
from src.utils.utils import norm_column  # noqa: E402

_orig_pkp = ev.primary_key_preprocess


def _pkp(response, reference, model_config_name):
    CTX.stage = "colmap" if list(reference) == list(CTX.required) and CTX.stage == "start" else "keymap"
    return _orig_pkp(response, reference, model_config_name)


ev.primary_key_preprocess = _pkp


def load_query(item: dict):
    """= WideSearchDataLoaderHF.load_data/load_answer, reading the local HF snapshot."""
    item = dict(item)
    item["evaluation"] = json.loads(item["evaluation"])
    req = item["evaluation"]["required"]
    try:
        answer = pd.read_csv(DATA / "widesearch_gold" / f"{item['instance_id']}.csv")
    except Exception:  # noqa: BLE001
        return None
    answer.columns = [norm_column(c.strip()) for c in answer.columns]
    if any(c not in answer.columns for c in req):
        return None
    item["answer"] = answer[req]
    return WideSearchQuery(**item)


def main(run_dir: Path):
    ids = json.loads((run_dir / "sample_ids.json").read_text())["ids"]
    items = {}
    for line in open(DATA / "widesearch.jsonl"):
        d = json.loads(line)
        items[d["instance_id"]] = d
    resp = {}
    if (run_dir / "responses.jsonl").exists():
        for line in open(run_dir / "responses.jsonl"):
            r = json.loads(line)
            if r["id"] not in resp or resp[r["id"]].get("error"):
                resp[r["id"]] = r
    cp = run_dir / "judge_cache.jsonl"
    if cp.exists():
        for line in open(cp):
            if line.strip():
                r = json.loads(line)
                CACHE[r["id"]] = r["content"]
    out = run_dir / "ws_eval"
    out.mkdir(exist_ok=True)
    scored, pend_inst = 0, []
    for iid in ids:
        q = load_query(items[iid])
        if q is None:
            print(f"WARNING: no gold answer for {iid}", file=sys.stderr)
            continue
        r = resp.get(iid)
        response = WideSearchResponse(instance_id=iid, response=(r or {}).get("content") or "NULL") if r else None
        CTX.stage, CTX.keymap_missing, CTX.missing, CTX.iid, CTX.required = "start", False, 0, iid, q.evaluation["required"]
        try:
            res = ev.evaluate_single_query(q, response, str(out / f"{iid}.csv"), "default_eval_config")
        except Pending:
            res = None
        if res is None or CTX.missing:
            pend_inst.append(iid)
            (out / f"{iid}.json").unlink(missing_ok=True)
            continue
        (out / f"{iid}.json").write_text(json.dumps(dataclasses.asdict(res), ensure_ascii=False, indent=1))
        scored += 1
    with open(run_dir / "ws_pending.jsonl", "w") as f:
        for p in PENDING.values():
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(json.dumps({"scored": scored, "pending_instances": len(pend_inst), "pending_prompts": len(PENDING)}))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
