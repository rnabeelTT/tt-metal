"""Helpers for ~/evals/tbench_run.sh (stdlib only).

  tb_helper.py config <out.json>      write the Harbor JobConfig from env (see tbench_run.sh)
  tb_helper.py cleanup <job_dir> <tasks_dir> <preexisting.txt> [final]
        loop: remove the docker image of every finished trial unless it was on disk before
        the run or another unfinished trial of the same task still needs it; exits when
        <job_dir>/result.json reports no running/pending trials.
  tb_helper.py summarize <job_dir> <bench> <label> <proxy_log>
        per-task table + RESULT line (mean reward over all trials; errored trials count 0).
"""
import collections
import glob
import json
import os
import re
import subprocess
import sys
import time

E = os.environ.get


def cmd_config(out):
    s = {}
    if E("TEMPERATURE"):
        s["temperature"] = float(E("TEMPERATURE"))
    llm_kwargs = {}
    if E("TOP_P"):
        llm_kwargs["top_p"] = float(E("TOP_P"))
    if E("TOP_K"):
        llm_kwargs["extra_body"] = {"top_k": int(E("TOP_K"))}
    if E("MAX_TOKENS"):
        llm_kwargs["max_tokens"] = int(E("MAX_TOKENS"))
    ctx = int(E("CTX", "131072"))
    kwargs = {"parser_name": E("PARSER", "json"), "api_base": E("AGENT_API_BASE"),
              "interleaved_thinking": E("INTERLEAVED_THINKING", "1") == "1",
              "model_info": {"max_input_tokens": ctx, "max_tokens": ctx,
                             "max_output_tokens": int(E("MAX_TOKENS") or ctx),
                             "input_cost_per_token": 0.0, "output_cost_per_token": 0.0},
              **s}
    if llm_kwargs:
        kwargs["llm_kwargs"] = llm_kwargs
    agent = {"name": "terminus-2", "model_name": "openai/" + E("MODEL_ID"), "kwargs": kwargs,
             "env": {"OPENAI_API_KEY": "dummy"}}
    if E("AGENT_TIMEOUT"):
        agent["override_timeout_sec"] = float(E("AGENT_TIMEOUT"))
    if E("AGENT") == "oracle":
        agent = {"name": "oracle"}
    elif E("AGENT") == "claude-code":
        # Claude Code (Anthropic Messages API) in the task container -> proxy on the docker
        # bridge gateway -> vLLM /v1/messages. Card: Claude Code 2.1.126, max_new_tokens 131072.
        agent = {"name": "claude-code", "model_name": E("MODEL_ID"), "base_url": E("AGENT_API_BASE_ROOT"),
                 "model_api_format": "anthropic_messages",
                 "kwargs": {"version": E("CC_VERSION", "2.1.126")},
                 "env": {"ANTHROPIC_API_KEY": "dummy", "ANTHROPIC_MODEL": E("MODEL_ID"),
                         "ANTHROPIC_SMALL_FAST_MODEL": E("MODEL_ID"),
                         "CLAUDE_CODE_MAX_OUTPUT_TOKENS": E("MAX_TOKENS") or "131072",
                         "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1", "DISABLE_TELEMETRY": "1"}}
        if E("CC_CONTEXT_WINDOW"):
            # Claude Code 2.1.x auto-compacts at min(model window, CLAUDE_CODE_AUTO_COMPACT_WINDOW); the
            # model window is 200K for unknown ids unless the id ends in "[1m]" (llm_proxy strips it)
            agent["env"]["CLAUDE_CODE_AUTO_COMPACT_WINDOW"] = E("CC_CONTEXT_WINDOW")
            if int(E("CC_CONTEXT_WINDOW")) > 200000:
                agent["model_name"] = E("MODEL_ID") + "[1m]"
                for k in ("ANTHROPIC_MODEL", "ANTHROPIC_SMALL_FAST_MODEL"):
                    agent["env"][k] = agent["model_name"]
        if E("CC_DISALLOWED_TOOLS"):
            agent["kwargs"]["disallowed_tools"] = E("CC_DISALLOWED_TOOLS")
        if E("AGENT_TIMEOUT"):
            agent["override_timeout_sec"] = float(E("AGENT_TIMEOUT"))
    elif E("AGENT") == "mini-swe-agent":
        # mini-swe-agent (LiteLLM, tool calling) installed in the task container; it calls the
        # proxy on the docker bridge gateway. Sampling is forced by the proxy.
        kw = {"version": E("MSWEA_VERSION", "2.4.6")}
        if E("MSWEA_CONFIG"):
            kw["config_file"] = E("MSWEA_CONFIG")
        if E("MAX_TOKENS"):
            kw["max_tokens"] = int(E("MAX_TOKENS"))
        agent = {"name": "mini-swe-agent", "model_name": "openai/" + E("MODEL_ID"), "base_url": E("AGENT_API_BASE"),
                 "model_api_format": "openai_chat_completions", "kwargs": kw,
                 "env": {"OPENAI_API_KEY": "dummy", "MSWEA_COST_TRACKING": "ignore_errors"}}
        if E("AGENT_TIMEOUT"):
            agent["override_timeout_sec"] = float(E("AGENT_TIMEOUT"))
    ds = {"path": E("TASKS_DIR")}
    inc = [x for x in re.split(r"[,\s]+", E("TASKS", "")) if x]
    if inc:
        ds["task_names"] = inc
    exc = [x for x in re.split(r"[,\s]+", E("EXCLUDE_TASKS", "")) if x]
    if exc:
        ds["exclude_task_names"] = exc
    if E("LIMIT"):
        ds["n_tasks"] = int(E("LIMIT"))
    env = {"type": "docker"}
    extra = [x for x in E("EXTRA_COMPOSE", "").split(",") if x]
    if extra:
        env["extra_docker_compose"] = extra
    if E("CPUS"):
        env["override_cpus"] = float(E("CPUS"))
    if E("MEM_MB"):
        env["override_memory_mb"] = int(E("MEM_MB"))
    cfg = {"job_name": E("JOB_NAME"), "jobs_dir": E("JOBS_DIR"), "n_attempts": int(E("N_ATTEMPTS", "1")),
           "n_concurrent_trials": int(E("CONC")), "quiet": True,
           "retry": {"max_retries": int(E("MAX_RETRIES", "1"))},
           "environment": env, "agents": [agent], "datasets": [ds]}
    if E("VERIFIER_TIMEOUT_MULT"):
        cfg["verifier_timeout_multiplier"] = float(E("VERIFIER_TIMEOUT_MULT"))
    if E("VERIFIER_DISABLE") == "1":
        cfg["verifier"] = {"disable": True}
    json.dump(cfg, open(out, "w"), indent=1)
    print(json.dumps(cfg, indent=1))


def task_images(tasks_dir):
    m = {}
    for f in glob.glob(os.path.join(tasks_dir, "*", "task.toml")):
        r = re.search(r'docker_image = "([^"]+)"', open(f).read())
        if r:
            m[os.path.basename(os.path.dirname(f))] = r.group(1)
    return m


def dockerfile_bases(tasks_dir, tasks):
    """FROM images of the Dockerfiles of tasks built locally (no docker_image), incl. tests/."""
    out = set()
    for t in tasks:
        for f in glob.glob(os.path.join(tasks_dir, t, "environment", "Dockerfile")) + \
                glob.glob(os.path.join(tasks_dir, t, "tests", "Dockerfile")):
            stages = set()
            for line in open(f, errors="replace"):
                m = re.match(r"\s*FROM\s+(?:--platform=\S+\s+)?(\S+)(?:\s+AS\s+(\S+))?", line, re.I)
                if m:
                    img = m.group(1)
                    if img.lower() not in stages and "$" not in img:
                        out.add(img if ":" in img.split("/")[-1] or "@" in img else img + ":latest")
                    if m.group(2):
                        stages.add(m.group(2).lower())
    return out


def selected_tasks(job_json, tasks_dir):
    import fnmatch
    ts = sorted(os.path.basename(os.path.dirname(p)) for p in glob.glob(os.path.join(tasks_dir, "*", "task.toml")))
    try:
        d = json.load(open(job_json))["datasets"][0]
    except Exception:  # noqa: BLE001
        return ts
    if d.get("task_names"):
        ts = [t for t in ts if any(fnmatch.fnmatch(t, p) for p in d["task_names"])]
    if d.get("exclude_task_names"):
        ts = [t for t in ts if not any(fnmatch.fnmatch(t, p) for p in d["exclude_task_names"])]
    if d.get("n_tasks"):
        ts = ts[:d["n_tasks"]]
    return ts


def cmd_cleanup(job_dir, tasks_dir, pre_file, mode="loop"):
    """Remove a task image once every selected task that uses it has n_attempts finished trials
    and none running (images are shared across tasks in some benchmarks); at the end of the job
    remove the rest, plus base images pulled to build tasks locally. Never removes images that
    were on disk before the run (pre_file) or that a container still uses (docker refuses)."""
    job_json = os.path.join(os.path.dirname(job_dir.rstrip("/")), "job.json")
    try:
        n_att = int(json.load(open(job_json)).get("n_attempts", 1))
    except Exception:  # noqa: BLE001
        n_att = 1
    sel = selected_tasks(job_json, tasks_dir)
    imgs = {t: i for t, i in task_images(tasks_dir).items() if t in sel}
    pre = set(open(pre_file).read().split()) if os.path.exists(pre_file) else set()
    removed = set()

    def rmi(img):
        if img in pre or img in removed:
            return
        if subprocess.run(["docker", "rmi", img], capture_output=True).returncode == 0:
            removed.add(img)

    while True:
        if mode != "final":
            time.sleep(30)
        done, running = collections.Counter(), collections.Counter()
        for d in glob.glob(os.path.join(job_dir, "*__*")):
            if os.path.isdir(d):
                t = os.path.basename(d).rsplit("__", 1)[0]
                (done if os.path.exists(os.path.join(d, "result.json")) else running)[t] += 1
        final = mode == "final"  # one synchronous pass at the end of the run script
        res = os.path.join(job_dir, "result.json")
        if os.path.exists(res):
            try:
                r = json.load(open(res))
                st = r["stats"]
                final = bool(r.get("finished_at")) and st.get("n_running_trials", 1) == 0 and st.get("n_pending_trials", 1) == 0
            except Exception:  # noqa: BLE001
                pass
        # per-trial images some harnesses leave behind (Pier compose: <trial>-main, <trial>-pier-egress-proxy)
        # (and Harbor's separate verifier env: <trial>__verifier__trial-main)
        fin = [os.path.basename(d).lower() for d in glob.glob(os.path.join(job_dir, "*__*"))
               if os.path.exists(os.path.join(d, "result.json"))]
        if fin:
            have = subprocess.run(["docker", "images", "--format", "{{.Repository}}:{{.Tag}}"],
                                  capture_output=True, text=True).stdout.split()
            for img in have:
                if any(img.startswith(f + "-") or img.startswith(f + "__") for f in fin):
                    rmi(img)
        for img in set(imgs.values()):
            users = [t for t, i in imgs.items() if i == img]
            if final or all(done[t] >= n_att and running[t] == 0 for t in users):
                rmi(img)
        if final:
            for img in dockerfile_bases(tasks_dir, [t for t in sel if t not in imgs]):
                rmi(img)
            break


def cmd_summarize(job_dir, bench, label, proxy_log):
    rows = []
    for f in sorted(glob.glob(os.path.join(job_dir, "*__*", "result.json"))):
        d = json.load(open(f))
        vr = (d.get("verifier_result") or {}).get("rewards") or {}
        exc = (d.get("exception_info") or {}).get("exception_type") if d.get("exception_info") else None
        ae = d.get("agent_execution") or {}
        t0, t1 = ae.get("started_at"), ae.get("finished_at")
        rows.append({"task": d.get("task_name", "").split("/")[-1], "trial": d.get("trial_name"),
                     "reward": vr.get("reward"), "exception": exc, "agent_started": t0, "agent_finished": t1})
    tasks = sorted({r["task"] for r in rows})
    n_trials = len(rows)
    rewards = [float(r["reward"] or 0.0) for r in rows]
    solved = sum(1 for x in rewards if x >= 1.0)
    errors = sum(1 for r in rows if r["exception"])
    ex = collections.Counter(r["exception"] for r in rows if r["exception"])
    calls = turns = 0
    maxp = []
    if os.path.exists(proxy_log):
        per = collections.defaultdict(int)
        for line in open(proxy_log):
            try:
                d = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            pth = d.get("path", "").split("?")[0]
            if pth.endswith("chat/completions") or pth.endswith("/messages"):
                calls += 1
                maxp.append(d.get("prompt_tokens") or 0)
    with open(os.path.join(job_dir, "..", f"{bench}_trials.tsv"), "w") as f:
        f.write("task\ttrial\treward\texception\n")
        for r in rows:
            f.write(f"{r['task']}\t{r['trial']}\t{r['reward']}\t{r['exception'] or ''}\n")
    mean = sum(rewards) / n_trials if n_trials else 0.0
    summ = {"bench": bench, "label": label, "n_tasks": len(tasks), "n_trials": n_trials, "solved_trials": solved,
            "mean_reward": round(mean, 4), "errored_trials": errors, "exceptions": dict(ex),
            "llm_calls": calls, "avg_llm_calls_per_trial": round(calls / n_trials, 1) if n_trials else None,
            "max_prompt_tokens_seen": max(maxp) if maxp else None}
    json.dump(summ, open(os.path.join(job_dir, "..", f"{bench}_summary.json"), "w"), indent=1)
    exp = int(E("EXPECTED_TRIALS", "0") or 0)
    flag = "  INCOMPLETE" if exp and n_trials < exp else ""
    if E("VERIFIER_DISABLE") == "1":  # rewards come later (deferred judge)
        print(f"AGENT_RUN {bench} trials={n_trials} label={label} tasks={len(tasks)} errored_trials={errors} "
              f"exceptions={dict(ex)} avg_llm_calls={summ['avg_llm_calls_per_trial']} "
              f"max_prompt_tok={summ['max_prompt_tokens_seen']}{flag}")
        return
    print(f"RESULT {bench} resolved={solved}/{n_trials} mean_reward={summ['mean_reward']} label={label} "
          f"tasks={len(tasks)} errored_trials={errors} exceptions={dict(ex)} "
          f"avg_llm_calls={summ['avg_llm_calls_per_trial']} max_prompt_tok={summ['max_prompt_tokens_seen']}{flag}")


if __name__ == "__main__":
    c = sys.argv[1]
    {"config": cmd_config, "cleanup": cmd_cleanup, "summarize": cmd_summarize}[c](*sys.argv[2:])
