"""Claw-Eval driver: official harness (claw-eval @5680b8b, sandbox mode) with the LLM judge deferred.

  claweval_driver.py run     --out OUT --variant all|general|multimodal|multi_turn --model M --base-url URL ...
  claweval_driver.py collect --out OUT            (judge_inputs.jsonl + auto_verdicts.jsonl + JUDGE_README.txt)
  claweval_driver.py ingest  --out OUT JUDGMENTS.jsonl [...]   (re-grade every trial with the verdicts -> final score)
  claweval_driver.py summary --out OUT            (provisional summary of what has run so far)

Per task this calls claw_eval.cli._run_single_task (the code `claw-eval batch --sandbox` runs per task: mock
services, sandbox container, agent loop, grader-file injection, env snapshot, grading), with these patches:
  * LLMJudge -> deferred_judge.DeferredJudge (records the exact judge requests, placeholder score 0.0);
  * the full env snapshot of each trial is saved next to its trace (<trace>.env_snapshot.json) so the
    official grader can be re-run at ingest time with the real verdicts (rule-based parts recomputed locally,
    final score = official compute_task_score of the official grader's DimensionScores);
  * sandbox port published on 127.0.0.1 only; per-request client timeout REQUEST_TIMEOUT (official 300 s);
  * optional TIMEOUT_SCALE on each task's timeout_seconds (1 = official);
  * USERSIM_EXCLUDE_WAIT=1: time spent waiting for the simulated user (Claude relay) is not charged to the
    task's wall-clock timeout.
Results: OUT/results/<task_id>.json (trials appended across resumed invocations), OUT/traces/*.jsonl.
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE / "repo"
sys.path.insert(0, str(HERE))

OFFICIAL_JUDGE = {"general": "google/gemini-3-flash-preview", "multimodal": "google/gemini-3-flash-preview",
                  "multi_turn": "anthropic/claude-opus-4.6"}
STRIDE = 50  # official cmd_batch port stride per worker slot


# ----------------------------------------------------------------------------------------------- helpers
def read_json(p, default=None):
    try:
        return json.loads(Path(p).read_text())
    except (OSError, json.JSONDecodeError):
        return default


def write_json(p, obj):
    p = Path(p)
    tmp = p.with_name(p.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1))
    os.replace(tmp, p)


def split_of(tags: list[str]) -> str:
    if "multimodal" in tags:
        return "multimodal"
    if "user_agent" in tags:
        return "multi_turn"
    return "general"


def list_tasks(tasks_dir: Path) -> list[dict]:
    import yaml
    out = []
    for d in sorted(p for p in tasks_dir.iterdir() if (p / "task.yaml").exists()):
        y = yaml.safe_load((d / "task.yaml").read_text())
        out.append({"dir": str(d), "task_id": y["task_id"], "split": split_of(y.get("tags") or []),
                    "timeout": (y.get("environment") or {}).get("timeout_seconds", 300),
                    "max_turns": (y.get("environment") or {}).get("max_turns", 20)})
    return out


def select_tasks(tasks, variant, only=None, exclude=None, limit=0):
    if variant != "all":
        tasks = [t for t in tasks if t["split"] == variant]
    if only:
        pats = [p.strip() for p in only.split(",") if p.strip()]
        tasks = [t for t in tasks if any(fnmatch.fnmatch(t["task_id"], p) or fnmatch.fnmatch(Path(t["dir"]).name, p)
                                         for p in pats)]
    if exclude:
        pats = [p.strip() for p in exclude.split(",") if p.strip()]
        tasks = [t for t in tasks if not any(fnmatch.fnmatch(t["task_id"], p) for p in pats)]
    if limit:
        tasks = tasks[:limit]
    return tasks


# ----------------------------------------------------------------------------------------------- patches
_USER_WAIT = [0.0]


class _ShiftedTime:
    """`time` module stand-in for runner.loop: monotonic() minus time spent waiting for the simulated user."""

    def __getattr__(self, name):
        return getattr(time, name)

    @staticmethod
    def monotonic():
        return time.monotonic() - _USER_WAIT[0]


_PATCHED = [False]


def apply_patches():
    if _PATCHED[0]:  # worker processes are reused across tasks: patch once
        return
    _PATCHED[0] = True
    import claw_eval.cli as cli
    import claw_eval.graders.llm_judge as llm_judge
    import claw_eval.models.task as task_mod
    import claw_eval.runner.loop as loop_mod
    import claw_eval.runner.providers.openai_compat as oc
    import claw_eval.runner.sandbox_runner as sr
    import claw_eval.runner.user_agent as ua
    from deferred_judge import DeferredJudge

    llm_judge.LLMJudge = DeferredJudge

    orig_save = cli._save_env_snapshot

    def save_env_snapshot(snapshot, trace_path, task_id):
        write_json(Path(trace_path).with_suffix(".env_snapshot.json"), snapshot or {})
        return orig_save(snapshot, trace_path, task_id)

    cli._save_env_snapshot = save_env_snapshot

    scale = float(os.environ.get("TIMEOUT_SCALE", "1") or 1)
    if scale != 1:
        orig_from_yaml = task_mod.TaskDefinition.from_yaml.__func__

        def from_yaml(cls, path):
            t = orig_from_yaml(cls, path)
            t.environment.timeout_seconds = int(t.environment.timeout_seconds * scale)
            return t

        task_mod.TaskDefinition.from_yaml = classmethod(from_yaml)

    req_timeout = float(os.environ.get("REQUEST_TIMEOUT", "300") or 300)
    orig_init = oc.OpenAICompatProvider.__init__

    def prov_init(self, *a, **kw):
        orig_init(self, *a, **kw)
        self.client = self.client.with_options(timeout=req_timeout, max_retries=0)

    oc.OpenAICompatProvider.__init__ = prov_init

    if os.environ.get("USERSIM_EXCLUDE_WAIT", "1") == "1":
        loop_mod.time = _ShiftedTime()
        orig_gen = ua.UserAgent.generate_response

        def gen(self, *a, **kw):
            t0 = time.monotonic()
            try:
                return orig_gen(self, *a, **kw)
            finally:
                _USER_WAIT[0] += time.monotonic() - t0

        ua.UserAgent.generate_response = gen

    def start_container(self, *, run_id):
        name = f"claw-agent-{run_id}"
        try:  # stale container from an interrupted run of this harness (same name) -> remove
            self._docker.containers.get(name).remove(force=True)
        except Exception:  # noqa: BLE001
            pass
        container = self._docker.containers.run(
            image=self._image, detach=True, name=name,
            mem_limit=self._config.memory_limit, nano_cpus=int(self._config.cpu_limit * 1e9),
            ports={f"{self._config.sandbox_port}/tcp": ("127.0.0.1", None)},
            labels={"app": "claw-eval", "role": "agent", "run_id": run_id,
                    "claweval.harness": os.environ.get("CLAWEVAL_LABEL", "")},
            environment=self._proxy_env())
        host_port = self._get_mapped_port(container)
        url = f"http://localhost:{host_port}"
        try:
            self._wait_healthy(f"{url}/health", timeout=60)
        except Exception:
            container.remove(force=True)
            raise
        print(f"[sandbox] Container {name} started at {url}")
        return sr.ContainerHandle(container=container, host_port=host_port, run_id=run_id, sandbox_url=url)

    sr.SandboxRunner.start_container = start_container


# ----------------------------------------------------------------------------------------------- run
def _trial_info(trial: dict, task: dict, scale: float) -> dict:
    """Add judge keys / env-snapshot path / timeout flags to a trial dict from _run_single_task."""
    tr = trial.get("trace")
    if not tr:
        return trial
    keys, end, ua_meta = [], None, {}
    try:
        for line in open(tr):
            ev = json.loads(line)
            if ev.get("type") == "grading_result":
                keys = [c.get("deferred_key") for c in ev.get("judge_calls") or [] if c.get("deferred_key")]
                ua_meta = ev.get("user_agent_meta") or {}
            elif ev.get("type") == "trace_end":
                end = ev
    except (OSError, json.JSONDecodeError):
        pass
    snap = Path(tr).with_suffix(".env_snapshot.json")
    trial["env_snapshot"] = str(snap) if snap.exists() else None
    trial["judge_keys"] = keys
    trial["turns"] = (end or {}).get("total_turns")
    trial["hit_max_turns"] = bool(end and end.get("total_turns", 0) >= task["max_turns"])
    trial["hit_timeout"] = bool(end and end.get("wall_time_s", 0) > task["timeout"] * scale)
    trial["loop_error"] = (end or {}).get("failure_modes") or []
    trial["user_agent_meta"] = ua_meta
    trial["provisional"] = bool(keys)
    return trial


def run_one(task: dict, opts: dict, slot: int) -> dict:
    os.environ["CLAWEVAL_JUDGE_DIR"] = opts["out"]
    os.environ["CLAWEVAL_JUDGE_MODE"] = "record"
    apply_patches()
    import claw_eval.cli as cli

    offset = opts["port_base_offset"] + slot * STRIDE
    base = opts["base_url"].rstrip("/")
    if opts["tag_requests"]:
        root = base[:-3] if base.endswith("/v1") else base
        base = f"{root}/t/{task['task_id']}/v1"
    t0 = time.time()
    res = cli._run_single_task(
        task_dir=task["dir"], config_path=opts["config"], model=opts["model"], api_key="EMPTY", base_url=base,
        trace_dir=str(Path(opts["out"]) / "traces"), port_offset=offset, no_judge=False,
        judge_model=OFFICIAL_JUDGE[task["split"]], trials=opts["trials"], sandbox=True,
        sandbox_image=opts["image"])
    scale = float(os.environ.get("TIMEOUT_SCALE", "1") or 1)
    res["trials"] = [_trial_info(t, task, scale) for t in res.get("trials", [])]
    res["split"] = task["split"]
    res["port_offset"] = offset
    res["elapsed_s"] = round(time.time() - t0, 1)
    return res


def cmd_run(a):
    out = Path(a.out)
    (out / "traces").mkdir(parents=True, exist_ok=True)
    (out / "results").mkdir(parents=True, exist_ok=True)
    tasks = select_tasks(list_tasks(Path(a.tasks_dir)), a.variant, a.tasks, a.exclude, a.limit)
    if a.no_multi_turn:
        tasks = [t for t in tasks if t["split"] != "multi_turn"]
    write_json(out / "selected_tasks.json", [t["task_id"] for t in tasks])
    todo = []
    for t in tasks:
        r = read_json(out / "results" / f"{t['task_id']}.json", {}) or {}
        trials = r.get("trials", [])
        if a.retry_errors:
            trials = [x for x in trials if not x.get("error")]
        need = a.n_runs - len(trials)
        if need > 0:
            todo.append((t, need))
    print(f"[claweval] {len(tasks)} tasks selected ({a.variant}); {len(todo)} need trials "
          f"({sum(n for _, n in todo)} trials); workers={a.conc}", flush=True)
    if not todo:
        return
    opts = {"out": str(out), "config": a.config, "model": a.model, "base_url": a.base_url,
            "port_base_offset": a.port_base_offset, "image": a.image, "tag_requests": a.tag_requests}
    t_start = time.time()
    done = 0
    with ProcessPoolExecutor(max_workers=a.conc) as pool:
        free = list(range(a.conc))
        pending = {}
        queue = list(todo)

        def submit():
            t, need = queue.pop(0)
            slot = free.pop(0)
            fut = pool.submit(run_one, t, dict(opts, trials=need), slot)
            pending[fut] = (t, slot)

        while queue and free:
            submit()
        while pending:
            fut = next(as_completed(pending))
            t, slot = pending.pop(fut)
            free.append(slot)
            try:
                res = fut.result()
            except Exception as e:  # noqa: BLE001
                res = {"task_id": t["task_id"], "split": t["split"], "error": f"{type(e).__name__}: {e}",
                       "trials": [{"error": f"worker crashed: {type(e).__name__}: {e}", "task_score": 0.0,
                                   "passed": False}]}
            rp = out / "results" / f"{t['task_id']}.json"
            prev = read_json(rp, {}) or {}
            old = prev.get("trials", [])
            if a.retry_errors:
                old = [x for x in old if not x.get("error")]
            if res.get("error") and not res.get("trials"):
                res["trials"] = [{"error": res["error"], "task_score": 0.0, "passed": False}]
            merged = {"task_id": t["task_id"], "task_dir": t["dir"], "split": t["split"],
                      "port_offset": res.get("port_offset"), "trials": old + res.get("trials", []),
                      "last_error": res.get("error")}
            write_json(rp, merged)
            done += 1
            sc = [x.get("task_score", 0.0) for x in res.get("trials", [])]
            errs = sum(1 for x in res.get("trials", []) if x.get("error"))
            el = time.time() - t_start
            print(f"  [{done}/{len(todo)}] {t['task_id']} ({t['split']}) provisional scores={sc} errors={errs} "
                  f"task_time={res.get('elapsed_s', 0)}s elapsed={el / 60:.1f}m", flush=True)
            if queue:
                submit()


# ----------------------------------------------------------------------------------------------- collect
JUDGE_README = """\
Claw-Eval deferred LLM judge
============================
Each line of judge_inputs.jsonl is ONE call the official Claw-Eval grader would have sent to its LLM judge
(google/gemini-3-flash-preview for general/multimodal tasks; anthropic/claude-opus-4.6 for multi_turn tasks;
temperature 0). judge_prompt is that call verbatim: the judge system prompt, then the user message with every
slot (task prompt, conversation, actions summary / audit-log artifacts, rubric) filled in by the grader.
system_prompt / user_prompt hold the two parts separately.

Visual items (meta.method == "evaluate_visual"): the user message interleaves text with images; each image is
shown as [IMAGE k: <path>] in judge_prompt and listed in `images` (reference images first, then candidate
images, exactly as the headers say). Open every image file (e.g. with the Read tool) before scoring.

How to judge (act exactly as the official judge model would):
  * Follow the rubric in the prompt literally; it defines the scale and the partial-credit rules. Be strict
    where it says so (the visual prompt demands harsh scoring). Judge only what the prompt shows.
  * Do not run, open or search anything else (no task files, no harness code, no web).
  * verdict_schema score-type items: {"score": <float 0..1>, "reasoning": "<brief explanation>"}.
  * meta.method == "chat" items are raw prompts (email triage graders) that ask for a JSON object in a given
    format: write that JSON object as a string in "response", e.g. {"response": "{\\"msg_001\\": \\"spam\\"}"}.

Write one JSON object per line to a judgments file:  {"id": "<id>", <verdict_schema fields>}
"""


def collect(out: Path, extra_keys: set[str] | None = None, path_name="judge_inputs.jsonl", quiet=False):
    """Build judge inputs for every judge request referenced by a valid trial (or only `extra_keys`)."""
    from deferred_judge import CHAT_SCHEMA, SCORE_SCHEMA
    rows, auto, refs = [], [], {}
    results, _ = load_results(out)
    for r in results:
        for i, t in enumerate(r.get("trials", [])):
            tid = f"{r['task_id']}#{i}"
            if t.get("error"):
                auto.append({"id": tid, "task_id": r["task_id"], "split": r.get("split"), "score": 0.0,
                             "reasoning": f"trial errored, scored 0 without judging: {str(t['error'])[:300]}"})
                continue
            for k in t.get("judge_keys") or []:
                refs.setdefault(k, {"task_id": r["task_id"], "split": r.get("split"), "trials": []})
                if tid not in refs[k]["trials"]:
                    refs[k]["trials"].append(tid)
    keys = sorted(refs) if extra_keys is None else sorted(k for k in extra_keys)
    for k in keys:
        req = read_json(out / "judge_requests" / f"{k}.json")
        if req is None:
            print(f"WARNING: judge request {k} missing on disk")
            continue
        ref = refs.get(k, {"task_id": None, "split": None, "trials": []})
        method = req["method"]
        system, user = req.get("system_prompt"), req.get("user_prompt")
        prompt = (system.rstrip() + "\n\n" if system else "") + user
        rows.append({"id": k, "bench": "claweval", "question": None, "reference": None, "model_answer": None,
                     "judge_prompt": prompt, "system_prompt": system, "user_prompt": user,
                     "images": req.get("images") or [],
                     "verdict_schema": CHAT_SCHEMA if method == "chat" else SCORE_SCHEMA,
                     "meta": {"method": method, "task_id": ref["task_id"], "split": ref["split"],
                              "trials": ref["trials"],
                              "judge_model_official": req.get("judge_model_official"), "temperature": 0.0}})
    p = out / path_name
    with open(p, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(out / "auto_verdicts.jsonl", "w") as f:
        for r in auto:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    ingest_cmd = f"~/evals/claweval_judge_ingest.sh {out} <judgments.jsonl> [more judgments.jsonl ...]"
    (out / "JUDGE_README.txt").write_text(
        JUDGE_README + f"\n{len(rows)} items to judge in {p}; {len(auto)} errored trials auto-scored 0 "
        f"(auto_verdicts.jsonl).\nThen: {ingest_cmd}\n"
        "If ingest reports new judge inputs (graders that branch on a verdict), judge those too and re-run ingest "
        "with all judgments files.\n")
    if not quiet:
        by = {}
        for r in rows:
            by[r["meta"]["method"]] = by.get(r["meta"]["method"], 0) + 1
        print(f"[claweval] judge inputs: {len(rows)} {by} -> {p}")
    return p, rows


def cmd_collect(a):
    collect(Path(a.out))


# ----------------------------------------------------------------------------------------------- scoring
def aggregate(results: list[dict], n_runs: int) -> dict:
    from claw_eval.models.scoring import is_pass
    out = {}
    for split in ("general", "multimodal", "multi_turn", "all"):
        rs = [r for r in results if split == "all" or r["split"] == split]
        if not rs:
            continue
        task_avgs, p1, phat, pat, n_trials, errs, tos, full = [], 0, 0, 0, 0, 0, 0, 0
        for r in rs:
            tr = r["trials"]
            sc = [t.get("task_score", 0.0) for t in tr]
            if not tr:
                continue
            task_avgs.append(sum(sc) / len(sc))
            p1 += sum(1 for s in sc if is_pass(s))
            n_trials += len(tr)
            errs += sum(1 for t in tr if t.get("error"))
            tos += sum(1 for t in tr if t.get("hit_timeout"))
            if len(tr) >= n_runs:
                full += 1
                phat += all(is_pass(s) for s in sc[:n_runs])
                pat += any(is_pass(s) for s in sc[:n_runs])
        n = len(task_avgs)
        out[split] = {"tasks": n, "trials": n_trials, "tasks_with_all_runs": full,
                      "avg_score": round(100 * sum(task_avgs) / n, 2) if n else None,
                      "pass_at_1": round(100 * p1 / n_trials, 2) if n_trials else None,
                      f"pass_hat_{n_runs}": round(100 * phat / full, 2) if full else None,
                      f"pass_at_{n_runs}": round(100 * pat / full, 2) if full else None,
                      "errored_trials": errs, "timeout_trials": tos}
    return out


def load_results(out: Path):
    sel = read_json(out / "selected_tasks.json", None)
    rs = []
    for rp in sorted((out / "results").glob("*.json")):
        r = read_json(rp)
        if r and (sel is None or r["task_id"] in sel):
            rs.append(r)
    return rs, sel


def cmd_summary(a):
    out = Path(a.out)
    rs, sel = load_results(out)
    agg = aggregate(rs, a.n_runs)
    pend = sum(1 for r in rs for t in r["trials"] if t.get("provisional") and not t.get("error"))
    write_json(out / "summary_provisional.json", {"n_runs": a.n_runs, "selected": len(sel or []),
                                                  "trials_with_pending_judge": pend, "splits": agg})
    print(json.dumps(agg, indent=1))


def regrade(out: Path, r: dict, t: dict, verdicts: dict):
    """Re-run the official grader for one trial with verdicts; returns (scores, task_score, passed, judge)."""
    import base64

    from claw_eval.cli import _grade_with_optional_params
    from claw_eval.graders.registry import get_grader
    from claw_eval.models.scoring import compute_task_score, is_pass
    from claw_eval.models.task import TaskDefinition
    from claw_eval.trace.reader import load_trace
    from deferred_judge import DeferredJudge

    task_dir = Path(r["task_dir"])
    task_yaml = task_dir / "task.yaml"
    task = TaskDefinition.from_yaml(task_yaml)
    if r.get("port_offset"):
        task.apply_port_offset(r["port_offset"])
    env_snapshot = read_json(t["env_snapshot"]) if t.get("env_snapshot") else None
    if task.local_grader_files:  # same as _run_single_task
        if env_snapshot is None:
            env_snapshot = {}
        for rel in task.local_grader_files:
            lp = task_dir / rel
            env_snapshot[f"local_file:{rel}"] = (
                {"encoding": "base64", "content": base64.b64encode(lp.read_bytes()).decode()}
                if lp.exists() else {"error": f"not found: {lp}"})
    judge = DeferredJudge(model_id=OFFICIAL_JUDGE[r["split"]], mode="replay", judge_dir=out, verdicts=verdicts)
    start, messages, dispatches, media_events, end, audit_data = load_trace(t["trace"])
    grader = get_grader(task.task_id, tasks_dir=task_dir.parent, task_dir=str(task_dir))
    scores, _calls = _grade_with_optional_params(grader, messages, dispatches, task, audit_data=audit_data,
                                                judge=judge, media_events=media_events, env_snapshot=env_snapshot)
    ts = compute_task_score(scores)
    return scores, ts, is_pass(ts), judge


def cmd_ingest(a):
    import contextlib
    import io

    out = Path(a.out)
    verdicts = {}
    for jp in a.judgments:
        for line in open(jp):
            if line.strip():
                v = json.loads(line)
                verdicts[str(v["id"])] = v
    rs, sel = load_results(out)
    known = {json.loads(l)["id"] for f in sorted(out.glob("judge_inputs*.jsonl")) for l in open(f) if l.strip()}
    missing_known, new_keys, used, n_regraded, fails = set(), set(), set(), 0, []
    final = []
    for r in rs:
        fr = {k: r[k] for k in ("task_id", "split") if k in r}
        fr["trials"] = []
        for i, t in enumerate(r["trials"]):
            ft = {k: t.get(k) for k in ("trace", "error", "turns", "hit_timeout", "hit_max_turns", "tokens",
                                        "wall_time_s", "model_time_s")}
            if t.get("error"):
                ft.update(task_score=0.0, passed=False)
            else:
                buf = io.StringIO()
                try:
                    with contextlib.redirect_stdout(buf):
                        scores, ts, passed, judge = regrade(out, r, t, verdicts)
                    n_regraded += 1
                    used |= set(judge.used)
                    for k in judge.missing:
                        (missing_known if k in known else new_keys).add(k)
                    ft.update(task_score=ts, passed=passed, completion=scores.completion,
                              robustness=scores.robustness, safety=scores.safety,
                              communication=scores.communication, judge_keys=judge.used,
                              judge_missing=judge.missing, provisional_score=t.get("task_score"))
                except Exception as e:  # noqa: BLE001
                    fails.append(f"{r['task_id']}#{i}: {type(e).__name__}: {e}")
                    ft.update(task_score=0.0, passed=False, regrade_error=f"{type(e).__name__}: {e}")
            fr["trials"].append(ft)
        final.append(fr)
    agg = aggregate(final, a.n_runs)
    unused = sorted(set(verdicts) - used)
    pending = len(new_keys) + len(missing_known)
    summary = {"n_runs": a.n_runs, "selected": len(sel or []), "regraded_trials": n_regraded,
               "judgments_loaded": len(verdicts), "judgments_unused": len(unused),
               "missing_judgments": sorted(missing_known), "new_judge_inputs": sorted(new_keys),
               "regrade_failures": fails, "status": "final" if pending == 0 else "incomplete", "splits": agg}
    write_json(out / "final_results.json", final)
    write_json(out / "summary.json", summary)
    with open(out / "scores.tsv", "w") as f:
        f.write("task_id\tsplit\ttrial\ttask_score\tpassed\tcompletion\trobustness\tsafety\terror\n")
        for r in final:
            for i, t in enumerate(r["trials"]):
                f.write(f"{r['task_id']}\t{r['split']}\t{i}\t{t.get('task_score')}\t{t.get('passed')}\t"
                        f"{t.get('completion', '')}\t{t.get('robustness', '')}\t{t.get('safety', '')}\t"
                        f"{(t.get('error') or t.get('regrade_error') or '')[:80]}\n")
    if fails:
        print(f"WARNING: {len(fails)} trials failed to re-grade (scored 0), e.g. {fails[:3]}")
    if missing_known:
        print(f"WARNING: {len(missing_known)} judge inputs have no judgment (placeholder score 0), "
              f"e.g. {sorted(missing_known)[:3]}")
    if unused:
        print(f"note: {len(unused)} judgments not used by any trial (ignored), e.g. {unused[:3]}")
    if new_keys:
        k = 2
        while (out / f"judge_inputs_round{k}.jsonl").exists():
            k += 1
        p, _ = collect(out, extra_keys=new_keys, path_name=f"judge_inputs_round{k}.jsonl", quiet=True)
        print(f"{len(new_keys)} NEW judge inputs (graders branched on a verdict): judge them, then re-run ingest "
              f"with all judgments files.")
        print(f"JUDGE_PENDING {p}")
    print_result(a.variant, agg, a.n_runs, pending=pending,
                 status="final" if pending == 0 and not fails else "incomplete")


def print_result(variant, agg, n_runs, pending, status):
    al = agg.get("all") or {}
    parts = [f"RESULT claweval_{variant}"]
    if pending:
        parts.append(f"pass^{n_runs}=NA avg_score=NA pass@1=NA")
    else:
        parts.append(f"pass^{n_runs}={al.get(f'pass_hat_{n_runs}')} avg_score={al.get('avg_score')} "
                     f"pass@1={al.get('pass_at_1')}")
    for s in ("general", "multimodal", "multi_turn"):
        if s in agg and not pending:
            parts.append(f"{s}:pass^{n_runs}={agg[s].get(f'pass_hat_{n_runs}')},avg={agg[s].get('avg_score')}")
    parts.append(f"tasks={al.get('tasks')} trials={al.get('trials')} full_tasks={al.get('tasks_with_all_runs')} "
                 f"errors={al.get('errored_trials')} timeouts={al.get('timeout_trials')} judge_pending={pending} "
                 f"status={status} card=72.5(35B)/79.4(122B)")
    print(" ".join(parts), flush=True)


def cmd_finish(a):
    """End of a run: collect judge inputs; print JUDGE_PENDING + provisional RESULT, or ingest directly."""
    out = Path(a.out)
    p, rows = collect(out)
    if rows:
        rs, _ = load_results(out)
        agg = aggregate(rs, a.n_runs)
        write_json(out / "summary_provisional.json", {"n_runs": a.n_runs, "judge_pending": len(rows), "splits": agg})
        print(f"JUDGE_PENDING {p}")
        print_result(a.variant, agg, a.n_runs, pending=len(rows), status="judge_pending")
    else:
        a.judgments = []
        cmd_ingest(a)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--out", required=True)
    r.add_argument("--variant", default="all", choices=["all", "general", "multimodal", "multi_turn"])
    r.add_argument("--tasks-dir", default=str(REPO / "tasks"))
    r.add_argument("--config", required=True)
    r.add_argument("--model", required=True)
    r.add_argument("--base-url", required=True)
    r.add_argument("--conc", type=int, default=4)
    r.add_argument("--n-runs", type=int, default=3)
    r.add_argument("--port-base-offset", type=int, default=20000)
    r.add_argument("--image", default="claw-eval-agent:5680b8b")
    r.add_argument("--tasks", default=None)
    r.add_argument("--exclude", default=None)
    r.add_argument("--limit", type=int, default=0)
    r.add_argument("--retry-errors", action="store_true")
    r.add_argument("--no-multi-turn", action="store_true")
    r.add_argument("--tag-requests", action="store_true")
    for name in ("collect", "summary", "finish", "ingest"):
        s = sub.add_parser(name)
        s.add_argument("--out", required=True)
        s.add_argument("--n-runs", type=int, default=3)
        s.add_argument("--variant", default="all")
        if name == "ingest":
            s.add_argument("judgments", nargs="+")
    a = ap.parse_args()
    a.out = str(Path(a.out).resolve())
    if a.cmd == "run":
        mx = 9129 + a.port_base_offset + (a.conc - 1) * STRIDE
        if mx >= 32768:
            sys.exit(f"port range would reach {mx} (>= 32768): lower --conc or --port-base-offset")
        cmd_run(a)
    else:
        {"collect": cmd_collect, "summary": cmd_summary, "finish": cmd_finish, "ingest": cmd_ingest}[a.cmd](a)


if __name__ == "__main__":
    main()
