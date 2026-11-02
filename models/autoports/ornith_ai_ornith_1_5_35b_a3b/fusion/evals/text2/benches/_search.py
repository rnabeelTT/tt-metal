"""Shared pieces of the search-agent benches (browsecomp, browsecomp_zh, seal0, hle_tools, widesearch)."""
from __future__ import annotations

import json
import math
import os
import time

from common.core import count_errors, finish, select_items, write_gold_map, write_jsonl
from common.judge import write_judge_inputs
from common.search_agent import AgentCfg, agent_summary, run_agents

AGENT_DATE = os.environ.get("AGENT_DATE") or time.strftime("%Y-%m-%d")

# Tongyi DeepResearch inference/prompt.py SYSTEM_PROMPT, first paragraph, with the <answer>-tag requirement replaced
# by "answer in the format the task asks for" (the official benchmark answer formats are in the user turn); tool
# schemas are rendered by the chat template from `tools`, so the hand-written <tools> block is dropped.
AGENT_SYSTEM = (
    "You are a deep research assistant. Your core function is to conduct thorough, multi-source investigations into "
    "any topic. You must handle both broad, open-domain inquiries and queries within specialized academic fields. For "
    "every request, synthesize information from credible, diverse sources to deliver a comprehensive, accurate, and "
    "objective response. Use the provided tools to search the web and read pages; verify facts before relying on "
    "them. When you have gathered sufficient information and are ready to provide the definitive response, reply "
    "without calling any tool and give the final answer in exactly the format the task asks for.\n\n"
    f"Current date: {AGENT_DATE}"
)

HLE_STYLE_SCHEMA = {"type": "object", "properties": {
    "extracted_final_answer": {"type": "string"}, "reasoning": {"type": "string"},
    "correct": {"enum": ["yes", "no"]}, "confidence": {"type": "integer", "minimum": 0, "maximum": 100}},
    "required": ["extracted_final_answer", "reasoning", "correct", "confidence"]}


def sample_n() -> int | None:
    v = os.environ.get("SAMPLE_N")
    return int(v) if v not in (None, "", "0") else None


def select(all_items, cfg, id_key="id"):
    return select_items(all_items, cfg, id_key=id_key, force_sample=sample_n())


def stub_gold(items, user_msg_fn, answer_fn):
    """Self-tests: the agent stub looks the gold answer up by the first user message."""
    write_gold_map([([user_msg_fn(it)], answer_fn(it)) for it in items])


def judge_rows(items, resps, bench, prompt_fn, schema, meta_fn=None, question_fn=None, ref_fn=None):
    rows, auto = [], []
    for it in items:
        r = resps.get(it["id"], {})
        ans = r.get("content") or ""
        if not ans.strip():
            auto.append({"id": it["id"], "correct": "no", "grade": "C", "confidence": None, "auto": True,
                         "reasoning": f"auto: no final answer ({r.get('error') or (r.get('agent') or {}).get('end_reason') or r.get('finish_reason')})"})
            continue
        rows.append({"id": it["id"], "bench": bench, "question": question_fn(it) if question_fn else it["question"],
                     "reference": ref_fn(it) if ref_fn else it["answer"], "model_answer": ans,
                     "judge_prompt": prompt_fn(it, ans), "verdict_schema": schema,
                     "meta": meta_fn(it) if meta_fn else {}})
    return rows, auto


def run_and_write(cfg, acfg: AgentCfg, bench, items, messages_fn, question_fn, prompt_fn, schema, instructions,
                  meta_fn=None, ref_fn=None, extra_cfg=None):
    cfg.dump(module=bench, agent=acfg.__dict__, agent_date=AGENT_DATE, sample_n=sample_n(), **(extra_cfg or {}))
    resps = run_agents(cfg, acfg, bench, [(it["id"], messages_fn(it), question_fn(it)) for it in items])
    rows, auto = judge_rows(items, resps, bench, prompt_fn, schema, meta_fn, question_fn, ref_fn)
    p = write_judge_inputs(cfg.out, rows, auto, instructions)
    finish(cfg, "accuracy", None, len(items), {"judge_items": len(rows), "auto_incorrect": len(auto),
                                               "request_errors": count_errors(resps), "agent": agent_summary(resps)},
           judge_pending=p)


def accuracy_ingest(cfg, verdicts, missing, is_correct, calib: bool = True, extra=None):
    """accuracy over the evaluated ids (missing judgments / no answer = wrong) + Wald 95% half-width
    (+ HLE-style RMS calibration error over judged items when confidences exist)."""
    ids = json.loads((cfg.out / "sample_ids.json").read_text())["ids"]
    n = len(ids)
    rows, conf, corr = [], [], []
    for i in ids:
        v = verdicts.get(i)
        ok = bool(v) and is_correct(v)
        rows.append({"id": i, "correct": ok, "judged": bool(v) and not v.get("auto"), **({"verdict": v} if v else {})})
        if calib and v and not v.get("auto") and v.get("confidence") is not None:
            try:
                conf.append(float(v["confidence"]))
                corr.append(ok)
            except (TypeError, ValueError):
                pass
    write_jsonl(cfg.out / "scores.jsonl", rows)
    acc = round(100 * sum(r["correct"] for r in rows) / n, 2) if n else 0.0
    hw = round(1.96 * math.sqrt(acc * (100 - acc) / n), 2) if n else None
    summ = {"ci95_half_width": hw, "judged": sum(r["judged"] for r in rows), "missing_judgments": len(missing)}
    if calib and conf:
        import numpy as np

        from benches.hle import calib_err

        ce = float(calib_err(np.array(conf) / 100, np.array(corr)))
        summ["calibration_error"] = None if math.isnan(ce) else round(100 * ce, 2)  # needs >= 100 judged items
    prev = cfg.out / "summary.json"
    if prev.exists():  # keep the generation-time fields (agent / tool statistics)
        old = json.loads(prev.read_text())
        summ.update({k: old[k] for k in ("agent", "request_errors", "judge_items", "auto_incorrect") if k in old})
    if extra:
        summ.update(extra(rows))
    print(f"*** {cfg.bench}: accuracy {acc}% +/- {hw} | n = {n}" + (f" | calibration error {summ.get('calibration_error')}" if 'calibration_error' in summ else ""))
    finish(cfg, "accuracy", acc, n, summ)
