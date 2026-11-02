"""VITA-Bench (VitaBench, meituan-longcat) through the official harness (pinned ~/evals/vitabench/repo @742e240).

Agent  = the served model (OpenAI tools, thinking on, sampling from env) via the official llm_agent.
User   = official user_simulator (prompts/user_system_prompt.yaml) talking to an OpenAI-compatible SHIM
         (USER_SIM_BASE_URL / USER_SIM_MODEL); spec in ~/evals/vitabench/USER_SIM_SHIM.md.
Judge  = the official rubric sliding-window evaluator (evaluation_type "trajectory"), emitted as judge JSONL
         rounds (one window per simulation per round; windows are sequential) -> judge_ingest.py.
Metric = mean over the 4 task sets (cross-scenario, delivery, in-store, OTA) of Avg@k (success rate; success =
         every rubric met). Qwen reports "the average score across 4 domains" (VitaBench README, 2026-01 news).

Env: USER_SIM_BASE_URL (required) USER_SIM_MODEL (default gpt-4.1) USER_SIM_TEMPERATURE (0.0, official models.yaml
     default) USER_SIM_API_KEY; VITA_LANGUAGE (chinese = official CLI default | english); TRIALS (1; the paper and
     leaderboard use 4 -> Avg@4); VITA_DOMAINS (cross_domain,delivery,instore,ota); EVAL_TYPE (trajectory |
     trajectory_full_traj_rubric); MAX_STEPS (300 = CLI default); LIMIT (first N tasks of each set);
     VITA_FORCE_FINAL=1 at ingest: finalize sims whose window verdict is missing (official 'keep states' path).
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

from common.core import finish
from common.judge import write_judge_inputs

BENCH_NAME = "vita"
VITA = Path.home() / "evals/vitabench"
PY = VITA / ".venv/bin/python"
DOMAIN_ARG = {"cross_domain": "delivery,instore,ota", "delivery": "delivery", "instore": "instore", "ota": "ota"}

JUDGE_INSTRUCTIONS = """VITA-Bench rubric evaluation (official sliding-window evaluator, vitabench prompts/sliding_window_eval_template.yaml).
Each row is ONE window of one simulation. Act as the evaluator: use `system_prompt` as the system message and
`user_prompt` as the user message (`judge_prompt` is the same text concatenated). Return the updated rubric list
(every rubric_idx in <current_rubrics>, with justification and meetExpectation) as {"id": ..., "rubrics": [...]}
or the evaluator's raw text answer as {"id": ..., "raw_output": "..."}.
The official evaluator is an LLM (paper: claude-3.7-sonnet). Windows are sequential: after ingest, the next round's
judge_inputs.jsonl (next window, carrying the updated rubric states) is written and JUDGE_PENDING is printed again,
until every simulation's last window is judged."""


def _models_yaml(cfg, path: Path):
    ub = os.environ.get("USER_SIM_BASE_URL")
    if not ub:
        sys.exit("VITA-Bench needs a simulated-user endpoint: set USER_SIM_BASE_URL (OpenAI-compatible, see "
                 f"{VITA}/USER_SIM_SHIM.md) and USER_SIM_MODEL")
    agent = {"name": "agent", "model": cfg.model, "base_url": cfg.base_url + "/chat/completions",
             "temperature": cfg.temperature, "top_p": cfg.top_p, "max_tokens": cfg.max_tokens}
    if cfg.top_k and cfg.top_k > 0:
        agent["top_k"] = cfg.top_k
    if cfg.presence_penalty is not None:
        agent["presence_penalty"] = cfg.presence_penalty
    user = {"name": "user_sim", "model": os.environ.get("USER_SIM_MODEL", "gpt-4.1"),
            "base_url": ub.rstrip("/") + "/chat/completions",
            "temperature": float(os.environ.get("USER_SIM_TEMPERATURE", 0.0)),
            "headers": {"Authorization": f"Bearer {os.environ.get('USER_SIM_API_KEY', 'EMPTY')}",
                        "Content-Type": "application/json"}}
    d = {"default": {"base_url": "unused", "temperature": 0.0, "max_input_tokens": 262144,
                     "headers": {"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', 'EMPTY')}",
                                 "Content-Type": "application/json"}},
         "models": [agent, user]}
    path.write_text(yaml.safe_dump(d, allow_unicode=True, sort_keys=False))


def _vita_eval(out: Path, *args) -> dict:
    r = subprocess.run([str(PY), str(VITA / "vita_eval.py"), *args], capture_output=True, text=True,
                       env={**os.environ, "VITA_MODEL_CONFIG_PATH": str(out / "models.yaml")})
    if r.returncode != 0:
        sys.exit(f"vita_eval failed:\n{r.stderr[-4000:]}")
    return json.loads(r.stdout.strip().splitlines()[-1])


def _report(cfg, res: dict):
    out = cfg.out
    counts = {}
    if (out / "request_counts.json").exists():
        counts = json.loads((out / "request_counts.json").read_text())
    n_sims = res["final"] + res["pending"]
    if res["pending"]:
        p = out / "judge_inputs.jsonl"
        if p.exists():
            p.rename(out / f"judge_inputs.round{res['round'] - 1}.jsonl")
        jp = write_judge_inputs(out, res["rows"], [], JUDGE_INSTRUCTIONS + f"\n\nRound {res['round']}.")
        finish(cfg, "avg_success", None, n_sims,
               {"judge_round": res["round"], "pending_windows": res["pending"], "final_sims": res["final"],
                "request_counts": counts}, judge_pending=jp)
    else:
        finish(cfg, "avg_success", res["overall"], n_sims,
               {"per_domain": res["per_domain"], "judge_rounds": res["round"] - 1, "request_counts": counts,
                "note": "mean over task sets of Avg@k (success = all rubrics met)"})


def main(cfg):
    out = cfg.out
    (out / "sims").mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(exist_ok=True)
    lang = os.environ.get("VITA_LANGUAGE", "chinese")
    trials = int(os.environ.get("TRIALS", 1))
    doms = os.environ.get("VITA_DOMAINS", "cross_domain,delivery,instore,ota").split(",")
    eval_type = os.environ.get("EVAL_TYPE", "trajectory")
    _models_yaml(cfg, out / "models.yaml")
    cfg.dump(module="vita", language=lang, trials=trials, domains=doms, eval_type=eval_type,
             user_sim_model=os.environ.get("USER_SIM_MODEL", "gpt-4.1"),
             user_sim_base_url=os.environ.get("USER_SIM_BASE_URL"),
             harness="meituan-longcat/vitabench@742e240855bf8686a0842360749d5ea970ea3987")
    per = max(1, cfg.concurrency // len(doms))
    env = {**os.environ, "VITA_MODEL_CONFIG_PATH": str(out / "models.yaml"), "REQUEST_TIMEOUT": str(cfg.timeout)}
    procs = []
    for d in doms:
        cmd = [str(PY), "-u", str(VITA / "vita_driver.py"), "--domain", DOMAIN_ARG[d], "--language", lang,
               "--trials", str(trials), "--concurrency", str(per), "--save-to", str(out / "sims" / f"{d}.json"),
               "--max-steps", os.environ.get("MAX_STEPS", "300"), "--counts", str(out / f"logs/{d}.counts.json")]
        if cfg.limit:
            cmd += ["--num-tasks", str(cfg.limit)]
        log = open(out / "logs" / f"{d}.log", "a")
        print(f"[vita] {d}: {per} concurrent simulations, log {out}/logs/{d}.log", flush=True)
        procs.append((d, subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)))
    bad = []
    for d, p in procs:
        if p.wait() != 0:
            bad.append(d)
    if bad:
        sys.exit(f"vita driver failed for {bad}; see {out}/logs/")
    counts = {}
    for d in doms:
        f = out / f"logs/{d}.counts.json"
        if f.exists():
            for k, v in json.loads(f.read_text()).items():
                counts[k] = counts.get(k, 0) + v
    (out / "request_counts.json").write_text(json.dumps(counts))
    res = _vita_eval(out, "prepare", str(out), json.dumps({"eval_type": eval_type, "language": lang}))
    _report(cfg, res)


def ingest(cfg, verdicts, missing):
    out = cfg.out
    st = json.loads((out / "eval_state.json").read_text())
    vf = out / f"verdicts.round{st['round']}.json"
    vf.write_text(json.dumps(verdicts, ensure_ascii=False))
    args = ["ingest", str(out), str(vf)] + (["force"] if os.environ.get("VITA_FORCE_FINAL") == "1" else [])
    res = _vita_eval(out, *args)
    _report(cfg, res)
