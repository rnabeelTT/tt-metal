"""Per-instance SWE-bench driver: pull image -> agent (OpenHands or SWE-agent) -> official
evaluation -> cleanup, with N instances in flight. Stdlib only; each harness runs in its
own venv as a subprocess. Invoked by ~/evals/swebench_run.sh (see there for env knobs).

Layout of OUT (= ~/ornith-fusion/runs/<label>/swebench_<variant>):
  config.json                      every setting of this run
  proxy.log, proxy_requests.jsonl  llm_proxy (one line per LLM call, tagged with instance)
  instances/<iid>/                 select.txt, llm.json, agent/ (harness output), agent.log,
                                   patch.diff, eval/ (official harness logs), eval.log,
                                   result.json
  summary.json, results.tsv, run.log
"""
import argparse
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor

HOME = os.path.expanduser("~")
EV = os.path.join(HOME, "evals")
SUP = os.path.join(EV, "swebench_support")
OH = os.path.join(EV, "oh-benchmarks")
SA = os.path.join(EV, "SWE-agent")
SA_PRO = os.path.join(EV, "SWE-bench_Pro-os", "SWE-agent")
SB = os.path.join(EV, "SWE-bench")
PRO = os.path.join(EV, "SWE-bench_Pro-os")
DATA = {"verified": "verified.jsonl", "multilingual": "multilingual.jsonl", "pro": "pro_v1.jsonl"}
SB_ALIAS = {"verified": "verified", "multilingual": "multilingual"}
OH_MOD = {"verified": "swebench", "multilingual": "swebenchmultilingual", "pro": "swebenchpro"}
OH_SDK_SHA = None

E = os.environ.get
log_lock = threading.Lock()
stop = threading.Event()


def log(msg):
    line = time.strftime("%H:%M:%S ") + msg
    with log_lock:
        print(line, flush=True)


def sh(cmd, timeout=None, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, **kw)


def free_gb():
    st = os.statvfs("/")
    return st.f_bavail * st.f_frsize / 1e9


def image_exists(img):
    return sh(["docker", "image", "inspect", img]).returncode == 0


def image_size(img):
    r = sh(["docker", "image", "inspect", "-f", "{{.Size}}", img])
    return int(r.stdout.strip()) if r.returncode == 0 and r.stdout.strip().isdigit() else None


def docker_pull(img, tries=6):
    for i in range(tries):
        r = sh(["docker", "pull", "-q", img], timeout=3600)
        if r.returncode == 0:
            return True
        err = r.stderr.lower()
        wait = 600 if ("toomanyrequests" in err or "rate limit" in err) else 30 * (i + 1)
        log(f"pull failed ({img}): {r.stderr.strip()[-200:]}; retry in {wait}s")
        time.sleep(wait)
    return False


def official_image(variant, row):
    if variant == "pro":
        tag = row["dockerhub_tag"]
        return f"jefzda/sweap-images:{tag}"
    iid = row["instance_id"].replace("__", "_1776_").lower()
    return f"swebench/sweb.eval.x86_64.{iid}:latest"


def free_port(lo=28200, hi=28299, host="0.0.0.0"):
    for p in range(lo, hi):
        with socket.socket() as s:
            try:
                s.bind((host, p))
                return p
            except OSError:
                continue
    raise RuntimeError("no free port in 28200-28299")


# ----------------------------------------------------------------------------- agents
def sampling():
    d = {}
    if E("TEMPERATURE"):
        d["temperature"] = float(E("TEMPERATURE"))
    if E("TOP_P"):
        d["top_p"] = float(E("TOP_P"))
    if E("TOP_K"):
        d["top_k"] = int(E("TOP_K"))
    return d


def run_openhands(cfg, row, idir, base_url):
    iid = row["instance_id"]
    open(os.path.join(idir, "select.txt"), "w").write(iid + "\n")
    s = sampling()
    llm = {"model": "openai/" + cfg["model"], "base_url": base_url, "api_key": "dummy",
           "timeout": int(E("LLM_TIMEOUT", "14400")), "num_retries": int(E("LLM_RETRIES", "6")),
           "native_tool_calling": True, "max_input_tokens": int(E("MAX_INPUT_TOKENS", "262144")),
           "drop_params": True}
    llm.update({k: v for k, v in s.items()})
    if E("MAX_TOKENS"):
        llm["max_output_tokens"] = int(E("MAX_TOKENS"))
    json.dump(llm, open(os.path.join(idir, "llm.json"), "w"), indent=1)
    out = os.path.join(idir, "agent")
    cmd = [os.path.join(OH, ".venv/bin/python"), os.path.join(SUP, "oh_entry.py"), OH_MOD[cfg["variant"]],
           os.path.join(idir, "llm.json"), "--dataset", os.path.join(SUP, "data", DATA[cfg["variant"]]),
           "--split", "train", "--select", os.path.join(idir, "select.txt"), "--workspace", "docker",
           "--num-workers", "1", "--max-iterations", E("MAX_ITERS", "500"), "--output-dir", out,
           "--n-critic-runs", E("OH_CRITIC_RUNS", "3"), "--max-retries", E("OH_MAX_RETRIES", "3")]
    if E("OH_CONDENSER", "1") == "0":
        cmd.append("--disable-condenser")
    if E("NOHACK_PROMPT", "0") == "1":
        cmd += ["--prompt-path", nohack_prompt(cfg)]
    env = dict(os.environ, BUILDX_BUILDER=E("BUILDX_BUILDER_NAME", "swe-oh-builder"),
               IMAGE_TAG_PREFIX=OH_SDK_SHA, SWE_RUN_LABEL=cfg["label"],
               SWE_DOCKER_NETWORK=E("AGENT_NETWORK", "swe-nonet") if E("NO_NETWORK", "1") == "1" else "")
    t0 = time.time()
    with open(os.path.join(idir, "agent.log"), "w") as lf:
        p = subprocess.Popen(cmd, cwd=OH, env=env, stdout=lf, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            p.wait(timeout=int(E("AGENT_TIMEOUT", "28800")))
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGTERM)
            time.sleep(20)
            os.killpg(p.pid, signal.SIGKILL)
            p.wait()
            return None, {"agent_error": "agent wall-clock timeout", "agent_s": time.time() - t0}
    meta = {"agent_rc": p.returncode, "agent_s": round(time.time() - t0, 1)}
    outs = sorted(
        (os.path.join(dp, f) for dp, _, fs in os.walk(out) for f in fs if f == "output.jsonl"),
        key=os.path.getmtime)
    patch = None
    for f in reversed(outs):
        for line in open(f):
            d = json.loads(line)
            if d.get("instance_id") == iid:
                patch = (d.get("test_result") or {}).get("git_patch")
                meta["agent_error"] = d.get("error")
                meta["n_events"] = len(d.get("history") or [])
                meta["attempt"] = d.get("attempt")
        if patch is not None:
            break
    if patch is None and "agent_error" not in meta:
        meta["agent_error"] = "no output.jsonl row"
    # The run's own agent-server images (built for this instance) are not needed anymore.
    tag = OH_SDK_SHA
    r = sh(["docker", "images", "--no-trunc", "--format", "{{.ID}} {{.Repository}}:{{.Tag}}",
            "ghcr.io/openhands/eval-agent-server"])
    pairs = [ln.split() for ln in r.stdout.splitlines() if len(ln.split()) == 2]
    key = official_image(cfg["variant"], row).split("/")[-1].split(":")[0 if cfg["variant"] != "pro" else 1]
    key = re.sub(r"[^a-z0-9_.-]+", "-", key.lower())[:60]
    ids = {i for i, t in pairs if key in t and tag in t}
    mine = [t for i, t in pairs if i in ids]  # every tag of those images (one tag form is truncated)
    meta["agent_image_bytes"] = image_size(mine[0]) if mine else None
    meta["agent_image_tags"] = mine
    if mine and E("KEEP_IMAGES", "0") != "1":
        sh(["docker", "rmi", *mine])
    return patch, meta


NOHACK_TEXT = """

IMPORTANT (evaluation rules): solve the issue with a genuine change to the source code that the project's maintainers would accept. Do not try to obtain the upstream fix: do not look for it in git history, other branches, tags, stashes or reflogs, in package registries or on the internet. Do not modify, delete or skip tests to make them pass, and do not special-case test inputs or hard-code expected outputs. Any solution that does any of this is marked as incorrect.
"""


def nohack_prompt(cfg):
    """OpenHands' own default.j2 for the variant + an explicit anti-reward-hacking notice
    (Ornith-1.5-122B card: "models are explicitly informed of potential reward-hacking
    behaviors and instructed to avoid them"; the card's exact wording is not published)."""
    src = os.path.join(OH, "benchmarks", OH_MOD[cfg["variant"]], "prompts", "default.j2")
    dst = os.path.join(cfg["out"], "prompt_nohack.j2")
    if not os.path.exists(dst):
        open(dst, "w").write(open(src).read().rstrip("\n") + NOHACK_TEXT)
    return dst


def run_sweagent(cfg, row, idir, base_url):
    iid = row["instance_id"]
    out = os.path.join(idir, "agent")
    s = sampling()
    pro = cfg["variant"] == "pro"
    repo = SA_PRO if pro else SA
    conf = E("SWEAGENT_CONFIG") or (
        os.path.join(repo, "config", "tool_use.yaml") if pro else  # Scale's Pro config
        os.path.join(SA, "config", "benchmarks", "anthropic_filemap_multilingual.yaml")
        if cfg["variant"] == "multilingual" else os.path.join(SA, "config", "default.yaml"))
    ckw = {}
    if "top_k" in s:
        ckw["top_k"] = s["top_k"]
    cmd = [os.path.join(repo, ".venv/bin/sweagent"), "run-batch", "--config", conf, "--output_dir", out,
           "--num_workers", "1",
           "--agent.model.name", "openai/" + cfg["model"], "--agent.model.api_base", base_url,
           "--agent.model.api_key", "dummy", "--agent.model.per_instance_cost_limit", "0",
           "--agent.model.total_cost_limit", "0",
           "--agent.model.per_instance_call_limit", E("MAX_ITERS", "300"),
           "--agent.model.max_input_tokens", E("MAX_INPUT_TOKENS", "262144"),
           "--instances.deployment.startup_timeout", "1800"]
    if "temperature" in s:
        cmd += ["--agent.model.temperature", str(s["temperature"])]
    if "top_p" in s:
        cmd += ["--agent.model.top_p", str(s["top_p"])]
    if ckw:
        cmd += ["--agent.model.completion_kwargs", json.dumps(ckw)]
    if E("MAX_TOKENS"):
        cmd += ["--agent.model.max_output_tokens", E("MAX_TOKENS")]
    if pro:
        inst = os.path.join(idir, "instance.json")
        json.dump([sweagent_pro_instance(row)], open(inst, "w"))
        cmd += ["--instances.type", "file", "--instances.path", inst]
    else:
        cmd += ["--instances.type", "swe_bench", "--instances.subset", cfg["variant"], "--instances.split", "test",
                "--instances.filter", "^" + re.escape(iid) + "$"]
    san = E("SANITIZE_GIT", "off")
    dargs = ["--label", f"swe-run={cfg['label']}"]
    if E("SWE_DOCKER_CPUS"):
        dargs += ["--cpus", E("SWE_DOCKER_CPUS")]
    if E("SWE_DOCKER_MEM"):
        dargs += ["--memory", E("SWE_DOCKER_MEM")]
    cmd += ["--instances.deployment.docker_args", json.dumps(dargs)]
    env = dict(os.environ, SWE_AGENT_NONET=("1" if E("NO_NETWORK", "1") == "1" else "0"),
               SWE_AGENT_NETWORK=E("AGENT_NETWORK", "swe-nonet"),
               SWE_AGENT_ENTRYPOINT_RESET="1" if pro else "0",
               SWE_AGENT_POST_STARTUP=json.dumps([SAN_SCRIPT[san] + "; echo SANITIZED $(git log --oneline | wc -l) "
                                                  "$(git rev-list --all | wc -l)"]) if san in ("refs", "full") else "")
    env["PYTHONPATH"] = SUP + os.pathsep + env.get("PYTHONPATH", "")  # sitecustomize: no-net patch
    t0 = time.time()
    with open(os.path.join(idir, "agent.log"), "w") as lf:
        p = subprocess.Popen(cmd, cwd=repo, env=env, stdout=lf, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            p.wait(timeout=int(E("AGENT_TIMEOUT", "28800")))
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGTERM)
            time.sleep(20)
            os.killpg(p.pid, signal.SIGKILL)
            p.wait()
            return None, {"agent_error": "agent wall-clock timeout", "agent_s": time.time() - t0}
    meta = {"agent_rc": p.returncode, "agent_s": round(time.time() - t0, 1)}
    patch = None
    for dp, _, fs in os.walk(out):
        for f in fs:
            if f.endswith(".pred"):
                d = json.load(open(os.path.join(dp, f)))
                patch = d.get("model_patch") or ""
            if f.endswith(".traj"):
                try:
                    t = json.load(open(os.path.join(dp, f)))
                    meta["n_steps"] = len(t.get("trajectory") or [])
                    meta["exit_status"] = (t.get("info") or {}).get("exit_status")
                except Exception:  # noqa: BLE001
                    pass
    if patch is None:
        meta["agent_error"] = "no .pred file"
    return patch, meta


def sweagent_pro_instance(row):
    """SWE-agent `file` instance for a SWE-bench Pro v1 row, exactly as Scale's
    helper_code/generate_sweagent_instances.py builds it (problem statement template,
    image, repo_name 'app')."""
    ps = f"""{row['problem_statement']}

Requirements:
{row['requirements']}

New interfaces introduced:
{row['interface']}"""
    return {"image_name": "jefzda/sweap-images:" + row["dockerhub_tag"], "problem_statement": ps,
            "instance_id": row["instance_id"], "base_commit": row["base_commit"], "repo_name": "app"}


# Anti-hacking git sanitization run in the repo before the agent's first step
# (OpenHands: oh_entry.py does the same). refs: no remotes/tags/other branches/stash/
# reflog, unreachable objects pruned; full: history squashed to one root commit.
_GIT_REFS = ("git remote | xargs -r -n1 git remote remove; "
             "git for-each-ref --format='%(refname)' refs/tags refs/remotes refs/stash refs/notes "
             "| xargs -r -n1 git update-ref -d; cur=$(git symbolic-ref -q HEAD || true); "
             "git for-each-ref --format='%(refname)' refs/heads | grep -vxF \"$cur\" "
             "| xargs -r -n1 git update-ref -d; git reflog expire --expire=now --all; git gc -q --prune=now")
_GIT_FULL = ("git -c user.email=a@b -c user.name=base checkout -q --orphan __base && "
             "git -c user.email=a@b -c user.name=base commit -q --no-verify --allow-empty -m base && "
             "git for-each-ref --format='%(refname)' refs/heads | grep -vx refs/heads/__base "
             "| xargs -r -n1 git update-ref -d; git branch -q -m __base main; " + _GIT_REFS)
SAN_SCRIPT = {k: f'cd "${{ROOT:-$(pwd)}}" && ( {v} ) > /dev/null 2>&1; true' for k, v in
              (("refs", _GIT_REFS), ("full", _GIT_FULL))}


# ------------------------------------------------------------------------------ eval
def remove_root_files(patch, files):
    blocks = re.split(r"(?m)^(?=diff --git )", patch)
    out = []
    for b in blocks:
        m = re.match(r"diff --git a/(.+) b/(.+)", b.split("\n", 1)[0])
        if m and (m.group(1) in files or m.group(2) in files):
            continue
        out.append(b)
    return "".join(out)


def strip_mode_only(patch):
    """Drop diff blocks that only change file modes (the rebuilt SWE-bench images carry a
    top commit that chmods every file, which OpenHands' base_commit..HEAD diff picks up)."""
    blocks = re.split(r"(?m)^(?=diff --git )", patch)
    keep = []
    for b in blocks:
        if b.startswith("diff --git ") and "\n@@" not in b and "Binary files" not in b and \
                not re.search(r"(?m)^(new file|deleted file|rename from|copy from|similarity index)", b):
            continue
        keep.append(b)
    return "".join(keep)


def eval_swebench(cfg, row, idir, patch):
    iid = row["instance_id"]
    ed = os.path.join(idir, "eval")
    os.makedirs(ed, exist_ok=True)
    model = re.sub(r"[^A-Za-z0-9_.-]", "_", cfg["label"])
    with open(os.path.join(ed, "preds.jsonl"), "w") as f:
        f.write(json.dumps({"instance_id": iid, "model_name_or_path": model, "model_patch": patch}) + "\n")
    run_id = "eval"
    cmd = [os.path.join(SB, ".venv/bin/swebench"), "eval", SB_ALIAS[cfg["variant"]], "-p", "preds.jsonl",
           "-i", iid, "--run-id", run_id, "-j", "1", "-t", E("EVAL_TIMEOUT", "3600")]
    r = sh(cmd, cwd=ed, timeout=int(E("EVAL_TIMEOUT", "3600")) + 1800)
    open(os.path.join(idir, "eval.log"), "w").write(r.stdout[-20000:] + "\n" + r.stderr[-20000:])
    rep = os.path.join(ed, "logs", "evaluation", run_id, model, iid, "report.json")
    if not os.path.exists(rep):
        return None, "no report.json"
    d = json.load(open(rep)).get(iid, {})
    return bool(d.get("resolved")), None


def eval_pro(cfg, row, idir, patch):
    iid = row["instance_id"]
    ed = os.path.join(idir, "eval")
    os.makedirs(ed, exist_ok=True)
    pj = os.path.join(ed, "patches.json")
    json.dump([{"instance_id": iid, "patch": patch, "prefix": "run"}], open(pj, "w"))
    cmd = [os.path.join(PRO, ".venv/bin/python"), "swe_bench_pro_eval.py",
           "--raw_sample_path", os.path.join(SUP, "data", "pro_v1.jsonl"), "--patch_path", pj,
           "--output_dir", ed, "--scripts_dir", "run_scripts", "--num_workers", "1",
           "--dockerhub_username", "jefzda", "--use_local_docker"]
    if E("PRO_EVAL_BLOCK_NETWORK", "0") == "1":
        cmd.append("--block_network")
    r = sh(cmd, cwd=PRO, timeout=int(E("EVAL_TIMEOUT", "3600")) + 1800)
    open(os.path.join(idir, "eval.log"), "w").write(r.stdout[-20000:] + "\n" + r.stderr[-20000:])
    res = os.path.join(ed, "eval_results.json")
    if not os.path.exists(res):
        return None, "no eval_results.json"
    d = json.load(open(res))
    out_json = [os.path.join(dp, f) for dp, _, fs in os.walk(ed) for f in fs if f.endswith("_output.json")]
    if iid not in d:
        return None, "instance missing from eval_results.json"
    if not out_json:
        return None, "evaluator produced no test output (infra error)"
    return bool(d[iid]), None


# ------------------------------------------------------------------------------ main
def run_instance(cfg, row, sems, proxy_base):
    iid = row["instance_id"]
    idir = os.path.join(cfg["out"], "instances", iid)
    rpath = os.path.join(idir, "result.json")
    if os.path.exists(rpath):
        old = json.load(open(rpath))
        if old.get("status") in ("resolved", "unresolved", "empty_patch") or E("RETRY_ERRORS", "0") != "1":
            return old
    if stop.is_set():
        return None
    if os.path.isdir(idir):
        shutil.rmtree(idir)
    os.makedirs(idir)
    res = {"instance_id": iid, "variant": cfg["variant"], "harness": cfg["harness"], "label": cfg["label"]}
    t0 = time.time()
    img = official_image(cfg["variant"], row)
    res["image"] = img
    while free_gb() < float(E("MIN_FREE_GB", "40")) and not stop.is_set():
        log(f"[{iid}] waiting: only {free_gb():.0f} GB free (< MIN_FREE_GB)")
        time.sleep(120)
    pre = image_exists(img)
    with sems["prep"]:
        if not pre and not docker_pull(img):
            res.update(status="error", error="image pull failed")
            json.dump(res, open(rpath, "w"), indent=1)
            return res
    res["image_bytes"] = image_size(img)
    res["image_preexisting"] = pre
    base = proxy_base + "/t/" + urllib.parse.quote(iid, safe="") + "/v1" if proxy_base else cfg["base_url"]
    log(f"[{iid}] agent start ({cfg['harness']})")
    try:
        patch, meta = (run_openhands if cfg["harness"] == "openhands" else run_sweagent)(cfg, row, idir, base)
    except Exception as e:  # noqa: BLE001
        patch, meta = None, {"agent_error": f"driver exception: {e!r}"}
    res.update(meta)
    res["free_gb_after_agent"] = round(free_gb(), 1)
    if patch is None:
        res.update(status="error", error=meta.get("agent_error"))
    else:
        open(os.path.join(idir, "patch.raw.diff"), "w").write(patch)
        if E("STRIP_MODE_ONLY", "1") == "1":
            patch = strip_mode_only(patch)
        if cfg["harness"] == "openhands" and cfg["variant"] != "pro":
            # same post-processing as OpenHands' swebench(-multilingual) eval_infer
            patch = remove_root_files(patch, ("pyproject.toml", "tox.ini", "setup.py"))
        open(os.path.join(idir, "patch.diff"), "w").write(patch)
        res["patch_chars"] = len(patch)
        if not patch.strip():
            res.update(status="empty_patch", resolved=False)
        else:
            t1 = time.time()
            with sems["eval"]:
                ok, err = (eval_pro if cfg["variant"] == "pro" else eval_swebench)(cfg, row, idir, patch)
            res["eval_s"] = round(time.time() - t1, 1)
            if ok is None:
                res.update(status="error", error=f"eval: {err}")
            else:
                res.update(status="resolved" if ok else "unresolved", resolved=ok)
    if not pre and E("KEEP_IMAGES", "0") != "1":
        sh(["docker", "rmi", img])
    if cfg["harness"] == "openhands" and E("KEEP_IMAGES", "0") != "1":
        sh(["docker", "buildx", "prune", "--builder", E("BUILDX_BUILDER_NAME", "swe-oh-builder"), "-f",
            "--max-used-space", E("BUILDER_KEEP", "25gb")])
    res["total_s"] = round(time.time() - t0, 1)
    json.dump(res, open(rpath, "w"), indent=1)
    log(f"[{iid}] {res['status']} agent={res.get('agent_s')}s eval={res.get('eval_s')}s "
        f"patch={res.get('patch_chars')} {res.get('error') or ''}")
    return res


def proxy_stats(path):
    by = {}
    if not os.path.exists(path):
        return by
    for line in open(path):
        try:
            d = json.loads(line)
        except Exception:  # noqa: BLE001
            continue
        if not d.get("path", "").endswith("chat/completions"):
            continue
        s = by.setdefault(d.get("tag") or "-", {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
                                               "max_prompt_tokens": 0, "errors": 0, "llm_s": 0.0})
        s["calls"] += 1
        s["prompt_tokens"] += d.get("prompt_tokens") or 0
        s["completion_tokens"] += d.get("completion_tokens") or 0
        s["max_prompt_tokens"] = max(s["max_prompt_tokens"], d.get("prompt_tokens") or 0)
        s["errors"] += d.get("status") != 200
        s["llm_s"] += d.get("latency_s") or 0
    return by


def main():
    global OH_SDK_SHA
    ap = argparse.ArgumentParser()
    for a in ("variant", "harness", "label", "out", "base_url", "model"):
        ap.add_argument("--" + a, required=True)
    ap.add_argument("--concurrency", type=int, required=True)
    a = ap.parse_args()
    cfg = vars(a)
    os.makedirs(os.path.join(a.out, "instances"), exist_ok=True)
    rows = [json.loads(line) for line in open(os.path.join(SUP, "data", DATA[a.variant]))]
    if E("INSTANCES"):
        src = E("INSTANCES")
        want = [x.strip() for x in (open(src).read().split() if os.path.exists(src) else re.split(r"[,\s]+", src)) if x.strip()]
        rows = [r for r in rows if r["instance_id"] in set(want)]
    if E("INSTANCE_REGEX"):
        rows = [r for r in rows if re.search(E("INSTANCE_REGEX"), r["instance_id"])]
    if E("LIMIT"):
        rows = rows[: int(E("LIMIT"))]
    if a.harness == "openhands":
        OH_SDK_SHA = sh(["git", "-C", os.path.join(OH, "vendor/software-agent-sdk"), "rev-parse", "--short=7", "HEAD"]).stdout.strip()
    cfg.update(n=len(rows), sampling=sampling(), max_tokens=E("MAX_TOKENS"), max_iters=E("MAX_ITERS"),
               no_network=E("NO_NETWORK", "1"), proxy=E("PROXY", "1"), reasoning=E("PROXY_REASONING", "keep"),
               oh_critic_runs=E("OH_CRITIC_RUNS", "3") if a.harness == "openhands" else None,
               oh_sdk=OH_SDK_SHA, started=time.strftime("%Y-%m-%d %H:%M:%S"),
               env={k: v for k, v in os.environ.items() if k in (
                   "TEMPERATURE", "TOP_P", "TOP_K", "MAX_TOKENS", "MAX_ITERS", "LIMIT", "INSTANCES", "NO_NETWORK",
                   "AGENT_TIMEOUT", "EVAL_TIMEOUT", "MIN_FREE_GB", "SWE_DOCKER_CPUS", "SWE_DOCKER_MEM",
                   "PREP_CONCURRENCY", "EVAL_CONCURRENCY", "OH_CRITIC_RUNS", "OH_CONDENSER", "MAX_INPUT_TOKENS",
                   "PROXY_REASONING", "SWEAGENT_CONFIG", "SANITIZE_GIT", "NOHACK_PROMPT", "HARNESS")})
    json.dump(cfg, open(os.path.join(a.out, "config.json"), "w"), indent=1)

    proxy = None
    proxy_base = None
    # Where the agent's LLM client runs: OpenHands V1 runs the agent loop inside the
    # agent-server container, so the endpoint must be reachable from the container's
    # network (its bridge gateway); SWE-agent calls the LLM from the host.
    bind = "127.0.0.1"
    if a.harness == "openhands":
        net = E("AGENT_NETWORK", "swe-nonet") if E("NO_NETWORK", "1") == "1" else "bridge"
        bind = sh(["docker", "network", "inspect", net, "-f",
                   "{{range .IPAM.Config}}{{.Gateway}}{{end}}"]).stdout.strip() or "172.17.0.1"
        if E("PROXY", "1") != "1":
            a.base_url = re.sub(r"//(localhost|127\.0\.0\.1)", "//" + bind, a.base_url)
    cfg["llm_bind"] = bind
    if E("PROXY", "1") == "1":
        port = free_port()
        penv = dict(os.environ, PROXY_LOG=os.path.join(a.out, "proxy_requests.jsonl"),
                    PROXY_SAMPLING=json.dumps(sampling()), PROXY_MAX_TOKENS=E("MAX_TOKENS", "0"))
        up = a.base_url[:-3] if a.base_url.rstrip("/").endswith("/v1") else a.base_url
        proxy = subprocess.Popen([sys.executable, os.path.join(EV, "agent_proxy", "llm_proxy.py"), bind,
                                  str(port), up.rstrip("/")], env=penv,
                                 stdout=open(os.path.join(a.out, "proxy.log"), "a"), stderr=subprocess.STDOUT)
        time.sleep(1.5)
        proxy_base = f"http://{bind}:{port}"
    sems = {"prep": threading.Semaphore(int(E("PREP_CONCURRENCY", "2"))),
            "eval": threading.Semaphore(int(E("EVAL_CONCURRENCY", "3")))}
    minfree = [free_gb()]

    def watch():
        while not stop.is_set():
            minfree[0] = min(minfree[0], free_gb())
            time.sleep(20)
    threading.Thread(target=watch, daemon=True).start()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    log(f"swebench {a.variant} harness={a.harness} n={len(rows)} conc={a.concurrency} proxy={proxy_base} "
        f"sampling={sampling()} out={a.out}")
    t0 = time.time()
    results = []
    try:
        with ThreadPoolExecutor(a.concurrency) as ex:
            for r in ex.map(lambda row: run_instance(cfg, row, sems, proxy_base), rows):
                if r:
                    results.append(r)
    finally:
        if proxy:
            proxy.terminate()
    ps = proxy_stats(os.path.join(a.out, "proxy_requests.jsonl"))
    # also count instances finished by earlier (resumed) invocations
    results = [json.load(open(os.path.join(a.out, "instances", r["instance_id"], "result.json")))
               for r in rows if os.path.exists(os.path.join(a.out, "instances", r["instance_id"], "result.json"))]
    n = len(rows)
    resolved = sum(1 for r in results if r.get("status") == "resolved")
    errors = sum(1 for r in results if r.get("status") == "error")
    empty = sum(1 for r in results if r.get("status") == "empty_patch")
    with open(os.path.join(a.out, "results.tsv"), "w") as f:
        f.write("instance_id\tstatus\tagent_s\teval_s\tpatch_chars\tllm_calls\tmax_prompt_tokens\tcompletion_tokens\timage_bytes\terror\n")
        for r in results:
            p = ps.get(r["instance_id"], {})
            f.write("\t".join(str(x) for x in (r["instance_id"], r.get("status"), r.get("agent_s"), r.get("eval_s"),
                                                 r.get("patch_chars"), p.get("calls"), p.get("max_prompt_tokens"),
                                                 p.get("completion_tokens"), r.get("image_bytes"),
                                                 (r.get("error") or "")[:120])) + "\n")
    calls = [ps[r["instance_id"]]["calls"] for r in results if r["instance_id"] in ps]
    maxp = [ps[r["instance_id"]]["max_prompt_tokens"] for r in results if r["instance_id"] in ps]
    comp = [ps[r["instance_id"]]["completion_tokens"] for r in results if r["instance_id"] in ps]
    avg = (lambda xs: round(sum(xs) / len(xs), 1) if xs else None)
    summ = {"bench": f"swebench_{a.variant}", "label": a.label, "harness": a.harness, "model": a.model,
            "n": n, "finished": len(results), "resolved": resolved, "errors": errors, "empty_patch": empty,
            "resolved_rate": round(resolved / n, 4) if n else None,
            "avg_llm_calls": avg(calls), "avg_max_prompt_tokens": avg(maxp), "avg_completion_tokens": avg(comp),
            "min_free_gb": round(minfree[0], 1), "wall_h": round((time.time() - t0) / 3600, 2)}
    json.dump(summ, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    flag = "" if len(results) == n else "  INCOMPLETE"
    print(f"RESULT swebench_{a.variant} resolved={resolved}/{n} rate={summ['resolved_rate']} harness={a.harness} "
          f"label={a.label} finished={len(results)} errors={errors} empty_patch={empty} "
          f"avg_llm_calls={summ['avg_llm_calls']} avg_final_prompt_tok={summ['avg_max_prompt_tokens']} "
          f"avg_completion_tok={summ['avg_completion_tokens']} min_free_gb={summ['min_free_gb']} "
          f"wall_h={summ['wall_h']}{flag}", flush=True)


if __name__ == "__main__":
    main()
