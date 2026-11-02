"""MultiChallenge (Scale AI, ekwinox117/multi-challenge @5ccefcc; 273 multi-turn conversations, 4 axes).

Generation: CONVERSATION verbatim (user/assistant turns, ends with a user turn), 1 attempt, thinking on, `content` only.
Judging (official src/evaluator.py): JUDGE_PROMPT.format(model_response, TARGET_QUESTION); judge returns
{reasoning, verdict: YES|NO} (official judge gpt-4o-2024-08-06, temp 0, structured output); passed = verdict == PASS_CRITERIA.
Metric (official src/result_parser.py): per-axis pass rate, overall = unweighted mean of the 4 axis scores.
Empty / failed responses are auto-judged NO (the official code would send the error text to the judge -> NO).
"""
from collections import defaultdict
from pathlib import Path

from common.core import count_errors, finish, generate_chat, read_jsonl, select_items, write_gold_map, write_jsonl
from common.judge import write_judge_inputs

REPO = Path.home() / "evals/src/multi-challenge"

JUDGE_PROMPT = '''You are tasked with evaluating a model response to see if it meets a specific criteria.
The criteria will always be YES/NO evaluation.

The model response is as follows:
<MODEL_RESPONSE>
{}
</MODEL_RESPONSE>

The criteria that the model response must meet is as follows. Be VERY STRICT!:
<CRITERIA>
{}
</CRITERIA>

Print your reasoning followed by your verdict, either "YES" or "NO".'''

SCHEMA = {"type": "object", "properties": {"reasoning": {"type": "string"}, "verdict": {"enum": ["YES", "NO"]}},
          "required": ["reasoning", "verdict"]}

INSTRUCTIONS = """MultiChallenge judging (official judge: gpt-4o-2024-08-06, temperature 0, structured output
JudgeResponse{reasoning, verdict: YES|NO}). For each line send `judge_prompt` as the only user message and return
{"id", "reasoning", "verdict": "YES"|"NO"}. The judge sees only the final response and the criterion question
(by design; do not consult the conversation)."""


def load_items():
    rows = read_jsonl(REPO / "data/benchmark_questions.jsonl")
    return [{"id": r["QUESTION_ID"], "axis": r["AXIS"], "conversation": r["CONVERSATION"],
             "target_question": r["TARGET_QUESTION"], "pass_criteria": r["PASS_CRITERIA"]} for r in rows]


def main(cfg):
    items = select_items(load_items(), cfg)
    cfg.dump(module="multichallenge", data=str(REPO / "data/benchmark_questions.jsonl"), attempts=1)
    reqs = [(it["id"], it["conversation"]) for it in items]
    write_gold_map([(it["conversation"], f"[stub answer for {it['id']}]") for it in items])
    resps = generate_chat(cfg, reqs)
    rows, auto = [], []
    for it in items:
        r = resps.get(it["id"], {})
        ans = r.get("content") or ""
        if not ans.strip():
            auto.append({"id": it["id"], "verdict": "NO", "reasoning": f"auto: no response ({r.get('error') or r.get('finish_reason')})"})
            continue
        rows.append({"id": it["id"], "bench": "multichallenge", "question": it["target_question"],
                     "reference": f"pass if verdict == {it['pass_criteria']}", "model_answer": ans,
                     "judge_prompt": JUDGE_PROMPT.format(ans, it["target_question"]), "verdict_schema": SCHEMA,
                     "meta": {"axis": it["axis"], "pass_criteria": it["pass_criteria"]}})
    p = write_judge_inputs(cfg.out, rows, auto, INSTRUCTIONS)
    finish(cfg, "overall", None, len(items), {"judge_items": len(rows), "auto_no": len(auto),
                                                "truncated_responses": sum(1 for r in resps.values() if r.get("finish_reason") == "length"),
                                                "request_errors": count_errors(resps)}, judge_pending=p)


def ingest(cfg, verdicts, missing):
    sel = set(read_jsonl_ids(cfg.out))
    items = [it for it in load_items() if it["id"] in sel]
    axis = defaultdict(lambda: [0, 0])
    rows = []
    for it in items:
        v = verdicts.get(it["id"], {}).get("verdict", "NO")
        ok = v == it["pass_criteria"]
        axis[it["axis"]][0] += ok
        axis[it["axis"]][1] += 1
        rows.append({"id": it["id"], "axis": it["axis"], "verdict": v, "passed": ok})
    write_jsonl(cfg.out / "scores.jsonl", rows)
    axis_scores = {a: 100 * p / t for a, (p, t) in axis.items()}
    overall = sum(axis_scores.values()) / len(axis_scores)
    for a, s in sorted(axis_scores.items()):
        print(f"  {a}: {s:.2f} ({axis[a][0]}/{axis[a][1]})")
    finish(cfg, "overall", overall, len(items), {"axis_scores": axis_scores, "missing_judgments": len(missing),
                                                 "micro_pass_rate": 100 * sum(r["passed"] for r in rows) / len(rows)})


def read_jsonl_ids(out):
    import json
    return json.loads((out / "sample_ids.json").read_text())["ids"]
