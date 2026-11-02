"""Deferred rubric judging for SWE Atlas Codebase QnA (stdlib only).

The official verifier (tests/evaluate_answer.py, run in the task container) asks an LLM judge
(Claude Opus 4.5 by default) one question per rubric item. Here the Harbor verifier is disabled
during the run; this script rebuilds exactly what that verifier would have sent (its own code is
imported from the task's tests/ with a capturing fake OpenAI client), writes it to files for Claude
judging agents, and ingests their verdicts with the verifier's own scoring code.

  sweatlas_judge.py prepare <run_dir>
      run_dir = ~/ornith-fusion/runs/<label>/sweatlas_qna (Harbor job in run_dir/job). Writes
      run_dir/judge/judge_inputs.jsonl (one line per trial that has an answer), auto_verdicts.jsonl
      (trials without an answer file / empty answer -> reward 0, like the verifier), JUDGE_README.txt,
      and prints RESULT + JUDGE_PENDING lines.
  sweatlas_judge.py ingest <run_dir> <judgments.jsonl> [--allow-missing]
      judgments.jsonl: {"id": <trial id>, "verdicts": {<rubric_id>: {"status": "YES"|"NO",
      "justification": "..."}}} per line. Applies the verifier's polarity flip for negative rubric
      items and its pass rule (reward 1 iff every scored must-have item scores 1; all items in this
      dataset are must-have), writes run_dir/sweatlas_qna_judged.json + trials tsv, prints RESULT.

Judge input line: {"id", "bench", "task", "trial", "question", "model_answer", "system_prompt",
"rubric_items": [{"rubric_id", "rubric_statement", "judge_prompt"}], "verdict_schema", "meta"}.
judge_prompt is the verifier's per-item user message verbatim (system_prompt is its system message);
the judge answers each item independently, YES iff the statement holds for the response, never
pre-flipping negative items (rubric type is kept in meta only, as the official judge never sees it).
"""
import collections
import glob
import importlib.util
import json
import os
import sys
import types

BENCH = "sweatlas_qna"
VERDICT_SCHEMA = {
    "type": "object",
    "required": ["id", "verdicts"],
    "properties": {
        "id": {"type": "string"},
        "verdicts": {"type": "object", "additionalProperties": {
            "type": "object", "required": ["status"],
            "properties": {"status": {"enum": ["YES", "NO"]}, "justification": {"type": "string"}}}},
    },
}
README = """SWE Atlas Codebase QnA -- deferred rubric judging (replaces the official LLM judge, Claude Opus 4.5).

Each line of judge_inputs.jsonl is one agent trial: the question, the agent's final answer (text between
the <<FINAL_ANSWER>> markers, exactly as the official verifier extracts it) and the rubric items.
For EVERY rubric item, act as the official judge: read `system_prompt` (the verifier's system message,
same for all items) and the item's `judge_prompt` (the verifier's user message for that item, verbatim),
and decide independently whether the behavior/condition in the rubric statement IS present in the
response (status YES) or not (NO). Do not flip anything for negatively phrased items -- the ingest
applies the verifier's polarity rule. Judge only from the response text (no repository access).

Write one JSON object per trial to a judgments file:
  {"id": "<id>", "verdicts": {"<rubric_id>": {"status": "YES"|"NO", "justification": "<one or two sentences>"}, ...}}
Every rubric_id of the trial must be present.
"""


def load_official(tests_dir):
    """Import the task's evaluate_answer.py with a stub `openai` module (it only needs OpenAI())."""
    if "openai" not in sys.modules:
        fake = types.ModuleType("openai")
        fake.OpenAI = object
        sys.modules["openai"] = fake
    spec = importlib.util.spec_from_file_location("sweatlas_evaluate_answer", os.path.join(tests_dir, "evaluate_answer.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _Capture:
    """Fake OpenAI client: records the messages evaluate_single_rubric would send, replies YES."""

    def __init__(self):
        self.calls = []
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self._create))

    def _create(self, model, messages, max_tokens):
        self.calls.append(messages)
        msg = types.SimpleNamespace(content='{"ratings": [{"rubric_statement": "x", "status": "YES", "score": "1", "justification": "capture"}]}')
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])


def trials(run_dir):
    out = []
    for d in sorted(glob.glob(os.path.join(run_dir, "job", "*__*"))):
        if not os.path.isdir(d):
            continue
        res = os.path.join(d, "result.json")
        r = json.load(open(res)) if os.path.exists(res) else {}
        task = (r.get("task_name") or os.path.basename(d).rsplit("__", 1)[0]).split("/")[-1]
        exc = (r.get("exception_info") or {}).get("exception_type") if r.get("exception_info") else None
        out.append({"trial": os.path.basename(d), "dir": d, "task": task, "finished": bool(r), "exception": exc})
    return out


def task_dir_for(tasks_root, task):
    for cand in (task, "task-" + task.split("task-")[-1]):
        p = os.path.join(tasks_root, cand)
        if os.path.isdir(p):
            return p
    raise SystemExit(f"task dir for {task} not found under {tasks_root}")


def extract_answer(trial_dir):
    """Same extraction as the verifier's main(): None = no file, '' = empty answer."""
    p = os.path.join(trial_dir, "agent", "answer.txt")
    if not os.path.exists(p):
        return None
    answer = open(p, errors="replace").read().strip()
    if "<<FINAL_ANSWER>>" in answer:
        parts = answer.split("<<FINAL_ANSWER>>")
        answer = parts[1].strip() if len(parts) >= 2 else answer
    return answer


def cmd_prepare(run_dir):
    tasks_root = os.environ.get("SWEATLAS_TASKS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "tasks-49e4af3"))
    jd = os.path.join(run_dir, "judge")
    os.makedirs(jd, exist_ok=True)
    rows, auto = [], []
    n_items = 0
    for t in trials(run_dir):
        if not t["finished"]:
            continue
        td = task_dir_for(tasks_root, t["task"])
        tests = os.path.join(td, "tests")
        mod = load_official(tests)
        answer = extract_answer(t["dir"])
        if not answer:
            auto.append({"id": t["trial"], "task": t["task"], "reward": 0,
                         "reason": "no answer file" if answer is None else "empty answer", "exception": t["exception"]})
            continue
        system_prompt = open(os.path.join(tests, "system_prompt.txt")).read()
        tmpl = open(os.path.join(tests, "user_prompt_template.txt")).read()
        rubrics = json.load(open(os.path.join(tests, "rubrics.json")))
        pp = os.path.join(tests, "prompt.txt")
        problem = open(pp).read().strip() if os.path.exists(pp) else ""
        items, meta_items = [], []
        for rub in rubrics:
            cap = _Capture()
            mod.evaluate_single_rubric(cap, "deferred-judge", system_prompt, tmpl, problem, answer, rub)
            msgs = cap.calls[0]
            assert msgs[0]["content"] == system_prompt
            items.append({"rubric_id": rub["id"], "rubric_statement": rub["title"], "judge_prompt": msgs[1]["content"]})
            ann = rub.get("annotations") or {}
            meta_items.append({"rubric_id": rub["id"], "type": ann.get("type", ""),
                               "importance": rub.get("importance", "must have")})
        n_items += len(items)
        rows.append({"id": t["trial"], "bench": BENCH, "task": t["task"], "trial": t["trial"], "question": problem,
                     "model_answer": answer, "system_prompt": system_prompt, "rubric_items": items,
                     "verdict_schema": VERDICT_SCHEMA, "meta": {"items": meta_items, "exception": t["exception"]}})
    with open(os.path.join(jd, "judge_inputs.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    with open(os.path.join(jd, "auto_verdicts.jsonl"), "w") as f:
        for r in auto:
            f.write(json.dumps(r) + "\n")
    with open(os.path.join(jd, "JUDGE_README.txt"), "w") as f:
        f.write(README + f"\n{len(rows)} trials / {n_items} rubric items to judge in {jd}/judge_inputs.jsonl; "
                f"{len(auto)} trials auto-scored 0 (auto_verdicts.jsonl).\nThen: python3 {os.path.abspath(__file__)} "
                f"ingest {run_dir} <judgments.jsonl>\n")
    n = len(rows) + len(auto)
    print(f"RESULT {BENCH} answers={len(rows)}/{n} no_answer={len(auto)} judge_items={n_items} (score pending judge)")
    print(f"JUDGE_PENDING {os.path.join(jd, 'judge_inputs.jsonl')}")


def cmd_ingest(run_dir, judgments, *flags):
    jd = os.path.join(run_dir, "judge")
    inputs = [json.loads(l) for l in open(os.path.join(jd, "judge_inputs.jsonl")) if l.strip()]
    auto = [json.loads(l) for l in open(os.path.join(jd, "auto_verdicts.jsonl")) if l.strip()]
    got = {}
    for l in open(judgments):
        if l.strip():
            r = json.loads(l)
            got[str(r["id"])] = r.get("verdicts") or {}
    tasks_root = os.environ.get("SWEATLAS_TASKS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "tasks-49e4af3"))
    missing, rows = [], []
    for row in inputs:
        v = got.get(row["id"])
        ids = [it["rubric_id"] for it in row["rubric_items"]]
        if v is None or any(i not in v for i in ids):
            missing.append(row["id"])
            if "--allow-missing" not in flags:
                continue
            v = v or {}
        mod = load_official(os.path.join(task_dir_for(tasks_root, row["task"]), "tests"))
        results = []
        for it, meta in zip(row["rubric_items"], row["meta"]["items"]):
            j = v.get(it["rubric_id"])
            parsed = None
            if j is not None:
                status = str(j.get("status", "")).strip().upper()
                parsed = {"rubric_statement": it["rubric_statement"], "status": status,
                          "score": {"YES": "1", "NO": "0"}.get(status), "justification": j.get("justification")}
            res = mod._canonicalize_judge_result(parsed, meta["type"]) if parsed else None
            results.append({"id": it["rubric_id"], "importance": meta["importance"], "score": res})
        # verifier's aggregation (main())
        must = [r for r in results if r["importance"] == "must have"]
        scored_must = [r for r in must if mod._is_scored(r["score"])]
        all_pass = len(scored_must) > 0 and all(str(r["score"]["score"]) == "1" for r in scored_must)
        scored = [r for r in results if mod._is_scored(r["score"])]
        agg = sum(int(r["score"]["score"]) for r in scored) / len(scored) if scored else 0.0
        rows.append({"trial": row["id"], "task": row["task"], "reward": 1 if all_pass else 0, "agg_score": round(agg, 4),
                     "n_items": len(results), "n_scored": len(scored), "n_pass": sum(1 for r in scored if r["score"]["score"] == "1")})
    if missing and "--allow-missing" not in flags:
        print(f"missing judgments for {len(missing)} trials (e.g. {missing[:3]}); rerun with --allow-missing to score "
              "them like the verifier does for unscored items")
        sys.exit(1)
    for a in auto:
        rows.append({"trial": a["id"], "task": a["task"], "reward": 0, "agg_score": 0.0, "n_items": 0, "n_scored": 0,
                     "n_pass": 0, "auto": a["reason"]})
    n = len(rows)
    mean = sum(r["reward"] for r in rows) / n if n else 0.0
    by_task = collections.defaultdict(list)
    for r in rows:
        by_task[r["task"]].append(r["reward"])
    task_mean = sum(sum(v) / len(v) for v in by_task.values()) / len(by_task) if by_task else 0.0
    summ = {"bench": BENCH, "n_trials": n, "n_tasks": len(by_task), "mean_reward": round(mean, 4),
            "task_mean_reward": round(task_mean, 4), "passed_trials": sum(r["reward"] for r in rows),
            "no_answer_trials": len(auto), "mean_agg_score": round(sum(r["agg_score"] for r in rows) / n, 4) if n else 0,
            "missing_judgments": len(missing)}
    json.dump({"summary": summ, "trials": rows}, open(os.path.join(run_dir, f"{BENCH}_judged.json"), "w"), indent=1)
    with open(os.path.join(run_dir, f"{BENCH}_judged_trials.tsv"), "w") as f:
        f.write("trial\ttask\treward\tagg_score\tn_pass\tn_items\n")
        for r in rows:
            f.write(f"{r['trial']}\t{r['task']}\t{r['reward']}\t{r['agg_score']}\t{r['n_pass']}\t{r['n_items']}\n")
    print(f"RESULT {BENCH} score={100 * mean:.1f} passed={summ['passed_trials']}/{n} tasks={len(by_task)} "
          f"task_mean={100 * task_mean:.1f} rubric_agg={summ['mean_agg_score']} no_answer={len(auto)} missing={len(missing)}")


if __name__ == "__main__":
    {"prepare": cmd_prepare, "ingest": cmd_ingest}[sys.argv[1]](*sys.argv[2:])
