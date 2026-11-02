#!/usr/bin/env python3
"""Quota-aware, resumable driver for Toolathlon-Verified on the authors' evaluation
service (eval_client.py --mode private). Stdlib only; the client itself runs in
~/evals/toolathlon/.venv. Called by ~/evals/toolathlon_run.sh (see its header).

Service facts this is built on (client/eval_server.py, client/EVAL_SERVICE_README.md):
  * One job at a time on the whole service (503 "Server is busy" otherwise).
  * Quota per client IP over a ROLLING 24 h window (records pruned by submitted_at):
    a submission is admitted while the summed duration of the IP's *completed* jobs
    in the window is < 180 min; after that only 3 submissions / 24 h. Duration =
    completion (or cancel/timeout) time - submission time, so it includes the
    server's container redeploy step. A running job is not cut at 180 min (the
    server only kills jobs at 240 min), so staying under the quota is on us.
  * Private mode: the agent runs on the service; its LLM calls come back over a
    WebSocket that OUR client opens (outbound only) and are replayed against the
    local base_url. Per LLM call the service waits at most 600 s.

What this driver does per invocation:
  1. Single-runner lock (quota is per IP, so per box, across all labels).
  2. Clean up a job left over by a killed run (cancel on the service, record it).
  3. Harvest finished tasks into state.json; pending = not finished yet, or
     errored (preprocess/run fail) with attempts < MAX_ERROR_ATTEMPTS.
  4. Quota: used = our ledger over the rolling window (a running job counts up to
     now), reconciled upward with what the service reports at each submission.
     Submit only if BUDGET (QUOTA_MIN - SAFETY_MIN) - used >= MIN_SESSION_MIN;
     otherwise print when the window frees up and exit 3.
  5. Wait while the service is busy with someone else's job (BUSY_WAIT_MIN; exit 4),
     exit 5 if the service is unreachable.
  6. Submit ONE job with the pending task list (fresh random job id, own output
     dir), follow the client's background worker, and cancel the job when the
     session budget is used up. Interrupted tasks go back to pending.
Exit codes: 0 all tasks finished | 3 quota exhausted (re-run after NEXT_ELIGIBLE)
            | 4 service busy | 5 service unreachable / submit error | 6 lock held
            | 2 usage | 130 interrupted
"""
import argparse
import datetime as dt
import fcntl
import hashlib
import io
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

E = os.environ


def fenv(name, default):
    v = E.get(name, "")
    return float(v) if v.strip() else float(default)


QUOTA_MIN = fenv("QUOTA_MIN", 180)          # service: cumulative minutes per IP per window
WINDOW_H = fenv("WINDOW_H", 24)             # service: rolling window
SAFETY_MIN = fenv("SAFETY_MIN", 10)         # kept free: cancel latency, clock skew, rounding
MIN_SESSION_MIN = fenv("MIN_SESSION_MIN", 30)  # don't start a job with less budget than this
BUSY_WAIT_MIN = fenv("BUSY_WAIT_MIN", 120)  # wait this long for someone else's job to end
POLL_S = fenv("POLL_S", 15)
BUSY_POLL_S = fenv("BUSY_POLL_S", 60)
MAX_ERROR_ATTEMPTS = int(fenv("MAX_ERROR_ATTEMPTS", 2))
MAX_SUBMITS = int(fenv("MAX_SUBMITS", 3))   # submissions per invocation (quota permitting)
DRY_RUN = E.get("DRY_RUN", "0") == "1"
WINDOW_S = WINDOW_H * 3600.0
BUDGET_S = (QUOTA_MIN - SAFETY_MIN) * 60.0

STOP = {"sig": None}


def now():
    return time.time()


def iso(t):
    return dt.datetime.fromtimestamp(t, dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


LOGFILE = {"f": None}


def log(msg):
    line = f"[{iso(now())}] {msg}"
    if LOGFILE["f"]:
        LOGFILE["f"].write(line + "\n")
        LOGFILE["f"].flush()
    try:
        print(line, flush=True)
    except OSError:  # stdout pipe (tee) gone after a group signal: keep going, driver.log has it
        pass


def emit(line):
    log(line)


def jload(path, default):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def jsave(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, sort_keys=True)
    os.replace(tmp, path)


# ----------------------------------------------------------------------------- ledger
class Ledger:
    """Box-wide record of service usage (the quota is per client IP, not per label)."""

    def __init__(self, path, service):
        self.path, self.service = path, service
        self.d = jload(path, {})
        self.d.setdefault(service, [])

    @property
    def entries(self):
        return self.d[self.service]

    def save(self):
        jsave(self.path, self.d)

    def used_s(self, t=None):
        t = now() if t is None else t
        tot = 0.0
        for e in self.entries:
            if e.get("rejected") or e["submitted_at"] <= t - WINDOW_S:
                continue
            tot += e["duration_s"] if e.get("ended_at") else max(0.0, t - e["submitted_at"])
        return tot

    def next_eligible(self, need_s):
        """Earliest time at which budget - used >= need_s (window entries age out)."""
        t = now()
        if BUDGET_S - self.used_s(t) >= need_s:
            return t
        for e in sorted(self.entries, key=lambda e: e["submitted_at"]):
            if e.get("rejected"):
                continue
            t_exp = e["submitted_at"] + WINDOW_S + 1
            if t_exp > t and BUDGET_S - self.used_s(t_exp) >= need_s:
                return t_exp
        return t + WINDOW_S

    def per_utc_day(self):
        days = {}
        for e in self.entries:
            if e.get("rejected"):
                continue
            d = iso(e["submitted_at"])[:10]
            dur = e["duration_s"] if e.get("ended_at") else now() - e["submitted_at"]
            days[d] = days.get(d, 0.0) + dur / 60
        return {k: round(v, 1) for k, v in sorted(days.items())}


# ----------------------------------------------------------------------------- service
class Service:
    def __init__(self, host, port):
        self.base = f"http://{host}:{port}"

    def get(self, path, params=None, timeout=30, raw=False):
        url = self.base + path + ("?" + urllib.parse.urlencode(params) if params else "")
        with urllib.request.urlopen(url, timeout=timeout) as r:
            data = r.read()
            return (r.status, dict(r.headers), data) if raw else json.loads(data)

    def check(self):
        """-> ('idle'|'busy'|'down', info)"""
        try:
            d = self.get("/check_server_status", timeout=20)
        except Exception as e:  # noqa: BLE001
            return "down", repr(e)
        return ("busy" if d.get("busy") else "idle"), d


# ----------------------------------------------------------------------------- tasks
def task_outcome(task_dir):
    """Classify a downloaded task dir like eval_server.is_task_finished does.
    -> None (not finished) | dict(kind=done|error, passed=bool, ...)"""
    st = jload(os.path.join(task_dir, "status.json"), None)
    if not isinstance(st, dict):
        return None
    pre, run, ev = st.get("preprocess"), st.get("running"), st.get("evaluation")
    out = {"preprocess": pre, "running": run, "evaluation": ev}
    if pre == "fail":
        return dict(out, kind="error", passed=False, why="preprocess_fail")
    if pre == "done" and run == "fail":
        return dict(out, kind="error", passed=False, why="run_fail")
    if pre == "done" and run in ("done", "timeout", "max_turn_exceeded") and ev is not None:
        res = jload(os.path.join(task_dir, "eval_res.json"), {})
        passed = (res.get("pass") is True) if isinstance(res, dict) and "pass" in res else (ev is True or ev == "pass")
        return dict(out, kind="done", passed=bool(passed))
    return None


class Driver:
    def __init__(self, a):
        self.a = a
        self.out = a.out
        self.state_path = os.path.join(self.out, "state.json")
        self.st = jload(self.state_path, {"tasks": {}, "jobs": []})
        self.svc = Service(a.server_host, a.server_port)
        self.ledger = Ledger(a.ledger, f"{a.server_host}:{a.server_port}")
        self.job = None  # the job currently running in this process

    def save(self):
        jsave(self.state_path, self.st)
        self.ledger.save()

    # -- task bookkeeping
    def sync_task_list(self):
        names = [l.strip() for l in open(self.a.tasks_file) if l.strip() and not l.startswith("#")]
        for n in names:
            self.st["tasks"].setdefault(n, {"status": "pending", "attempts": 0, "interrupted": 0})
        self.st["task_order"] = names

    def pending(self):
        out = []
        for n in self.st["task_order"]:
            t = self.st["tasks"][n]
            if t["status"] == "pending" or (t["status"] == "error" and t["attempts"] < MAX_ERROR_ATTEMPTS):
                out.append(n)
        return out

    def harvest(self, job, final):
        """Record finished tasks of `job`; on final harvest, fetch stragglers directly
        (GET /get_task_archive, no quota) and return unfinished ones to pending."""
        fp = os.path.join(job["dir"], "finalpool")
        for n in job["tasks"]:
            t = self.st["tasks"][n]
            if t["status"] in ("done", "error"):  # all of a job's tasks are set to running at submit
                continue
            td = os.path.join(fp, n)
            res = task_outcome(td) if os.path.isdir(td) else None
            if res is None and final:
                res = self.fetch_archive(job, n)
            if res is not None:
                t.update(status=res["kind"], passed=res["passed"], job_id=job["job_id"], dir=td,
                         finished_at=now(), attempts=t.get("attempts", 0) + 1,
                         preprocess=res["preprocess"], running=res["running"], evaluation=res["evaluation"],
                         why=res.get("why"))
            elif final:
                t["status"] = "pending" if t["status"] in ("pending", "running") else t["status"]
                if t["status"] == "pending":
                    t["interrupted"] = t.get("interrupted", 0) + 1
            else:
                t["status"] = "running"
        self.save()

    def fetch_archive(self, job, name):
        if DRY_RUN:
            return None
        try:
            code, hdr, data = self.svc.get("/get_task_archive", {"job_id": job["job_id"], "task_name": name},
                                           timeout=120, raw=True)
        except Exception:  # noqa: BLE001 - 404 = never ran / cleaned up
            return None
        md5 = {k.lower(): v for k, v in hdr.items()}.get("x-content-md5")
        if md5 and hashlib.md5(data).hexdigest() != md5:
            return None
        tmp = os.path.join(job["dir"], ".fetch_" + name)
        shutil.rmtree(tmp, ignore_errors=True)
        os.makedirs(tmp)
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
            safe = [m for m in tar.getmembers()
                    if not os.path.isabs(m.name) and ".." not in m.name.split("/") and (m.isfile() or m.isdir())]
            tar.extractall(tmp, members=safe)
        res = task_outcome(os.path.join(tmp, name))
        if res is not None:
            dst = os.path.join(job["dir"], "finalpool", name)
            shutil.rmtree(dst, ignore_errors=True)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.move(os.path.join(tmp, name), dst)
            log(f"fetched straggler archive {name}")
        shutil.rmtree(tmp, ignore_errors=True)
        return res

    # -- job lifecycle
    def worker_alive(self, pid):
        if not pid:
            return False
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                cmd = f.read()
            with open(f"/proc/{pid}/stat") as f:
                if f.read().split(") ", 1)[1].startswith("Z"):
                    return False
            return b"private_worker" in cmd or b"eval_client" in cmd
        except OSError:
            return False

    def client(self, *args, timeout=120):
        return subprocess.run([self.a.python, "eval_client.py", *args], cwd=self.a.client_dir,
                              capture_output=True, text=True, timeout=timeout)

    def cancel(self, job, reason):
        log(f"cancelling {job['job_id']} on the service ({reason})")
        if not DRY_RUN:
            try:
                r = self.client("cancel", "--job-id", job["job_id"], "--server-host", self.a.server_host,
                                "--server-port", str(self.a.server_port), timeout=90)
                log("cancel: " + " ".join((r.stdout + r.stderr).split())[:300])
            except Exception as e:  # noqa: BLE001
                log(f"cancel failed: {e!r}")
        pid = job.get("worker_pid")
        if self.worker_alive(pid):
            try:
                os.kill(pid, signal.SIGTERM)  # its handler also posts /cancel_job
            except OSError:
                pass
            for _ in range(60):
                if not self.worker_alive(pid):
                    break
                time.sleep(1)
            if self.worker_alive(pid):
                os.kill(pid, signal.SIGKILL)

    def end_job(self, job, status, t=None):
        t = now() if t is None else t
        job.update(status=status, ended_at=t)
        for e in self.ledger.entries:
            if e["job_id"] == job["job_id"] and not e.get("ended_at"):
                e.update(ended_at=t, duration_s=round(t - e["submitted_at"], 1), status=status)
        self.harvest(job, final=True)
        log(f"job {job['job_id']} ended: {status}, {round((t - job['submitted_at']) / 60, 1)} min; "
            f"window usage now {self.ledger.used_s() / 60:.1f}/{QUOTA_MIN:g} min")
        self.save()

    def cleanup_stale(self):
        for job in self.st["jobs"]:
            if job.get("ended_at"):
                continue
            log(f"found unfinished job {job['job_id']} from an earlier run (worker pid {job.get('worker_pid')})")
            svc_status = None
            if not self.worker_alive(job.get("worker_pid")) and not DRY_RUN:
                try:
                    svc_status = self.svc.get("/poll_job_status", {"job_id": job["job_id"]}).get("status")
                except Exception:  # noqa: BLE001
                    pass
            if svc_status in ("completed", "failed", "timeout", "cancelled"):
                # ended on the service while we were gone: last client.log write ~ end time
                mt = [os.path.getmtime(os.path.join(r, f)) for r, _, fs in os.walk(job["dir"]) for f in fs]
                self.end_job(job, "stale_" + svc_status, t=max(mt) if mt else None)
            else:
                self.cancel(job, "runner restarted; no model proxy behind the old worker")
                self.end_job(job, "stale_cancelled")

    def submit(self, tasks, remaining_s):
        n = len(self.st["jobs"]) + 1
        job_id = "job_" + uuid.uuid4().hex[:12]
        jdir = os.path.join(self.out, "jobs", f"{n:02d}_{job_id}")
        os.makedirs(jdir)
        tl = os.path.join(self.out, "jobs", f"{n:02d}_{job_id}.tasks.txt")
        with open(tl, "w") as f:
            f.write("\n".join(tasks) + "\n")
        job = {"n": n, "job_id": job_id, "dir": jdir, "tasks": tasks, "submitted_at": now(),
               "budget_s": remaining_s, "deadline": now() + remaining_s, "status": "submitting"}
        args = ["run", "--mode", "private", "--base-url", self.a.llm_base_url, "--model-name", self.a.model,
                "--output-dir", jdir, "--server-host", self.a.server_host, "--server-port", str(self.a.server_port),
                "--ws-proxy-port", str(self.a.ws_port), "--workers", str(self.a.workers), "--job-id", job_id,
                "--task-list-file", tl, "--api-key", "dummy"]
        if self.a.model_params:
            args += ["--model-params-file", self.a.model_params]
        log(f"submitting {job_id}: {len(tasks)} tasks, workers={self.a.workers}, session budget "
            f"{remaining_s / 60:.1f} min (deadline {iso(job['deadline'])})")
        if DRY_RUN:
            log("DRY_RUN=1: would run eval_client.py " + " ".join(args))
            os.rmdir(jdir)
            os.remove(tl)
            return None
        # ledger first: a crash between submit and bookkeeping must not under-count
        entry = {"job_id": job_id, "label": self.a.label, "submitted_at": job["submitted_at"]}
        self.ledger.entries.append(entry)
        self.ledger.save()
        try:
            r = self.client(*args, timeout=180)
        except subprocess.TimeoutExpired:
            r = subprocess.CompletedProcess(args, 1, "", "client timed out")
        txt = r.stdout + r.stderr
        with open(os.path.join(self.out, "submit.log"), "a") as f:
            f.write(f"=== {iso(now())} {job_id}\n{txt}\n")
        m = re.search(r"Background worker started \(PID: (\d+)\)", txt)
        if r.returncode != 0 or not m:
            entry.update(rejected=True, ended_at=now(), duration_s=0.0)
            shutil.rmtree(jdir, ignore_errors=True)
            busy = "Server is busy" in txt or "busy" in txt.lower()
            ratel = "Rate limit exceeded" in txt
            ra = re.search(r"Retry after: (\S+)", txt)
            if ratel:
                entry["service_rate_limited"] = True
                entry["retry_after_service_clock"] = ra.group(1) if ra else None
            self.ledger.save()
            log("submission refused: " + " ".join(txt.split())[-500:])
            return "busy" if busy and not ratel else ("ratelimited" if ratel else "error")
        job["worker_pid"] = int(m.group(1))
        q = re.search(r"Duration: ([\d.]+) / ([\d.]+) minutes used \(([-\d.]+) remaining\)", txt)
        if q:
            svc_used = float(q.group(1)) * 60
            job["service_used_min_at_submit"] = float(q.group(1))
            # The service only counts *completed* jobs; ours just started (counts ~0 here).
            ours = sum(e["duration_s"] for e in self.ledger.entries
                       if e.get("ended_at") and not e.get("rejected") and e["job_id"] != job_id
                       and e["submitted_at"] > job["submitted_at"] - WINDOW_S)
            if svc_used > ours + max(6.0, 0.005 * QUOTA_MIN * 60):  # someone else on our IP, or another box
                extra = svc_used - ours
                log(f"service reports {svc_used / 60:.1f} min used vs {ours / 60:.1f} in our ledger: "
                    f"recording {extra / 60:.1f} min of external usage on this IP")
                self.ledger.entries.append({"job_id": "external_" + job_id, "label": "(other users of this IP)",
                                            "submitted_at": job["submitted_at"] - 1, "ended_at": job["submitted_at"],
                                            "duration_s": extra, "external": True})
                job["deadline"] = min(job["deadline"], job["submitted_at"] + BUDGET_S - (ours + extra))
        job["status"] = "running"
        self.st["jobs"].append(job)
        for t in tasks:
            self.st["tasks"][t]["status"] = "running"
        self.save()
        log(f"accepted {job_id}; worker pid {job['worker_pid']}; service usage line: "
            f"{q.group(0) if q else 'n/a'}")
        return job

    def follow(self, job):
        self.job = job
        last = None
        while True:
            if STOP["sig"]:
                self.cancel(job, f"signal {STOP['sig']}")
                self.end_job(job, "interrupted")
                self.job = None
                return "interrupted"
            alive = self.worker_alive(job["worker_pid"])
            self.harvest(job, final=False)
            done = sum(1 for n in job["tasks"] if self.st["tasks"][n]["status"] in ("done", "error")
                       and self.st["tasks"][n].get("job_id") == job["job_id"])
            msg = f"{job['job_id']}: {done}/{len(job['tasks'])} finished, {(now() - job['submitted_at']) / 60:.1f}/" \
                  f"{job['budget_s'] / 60:.1f} min"
            if msg.split(",")[0] != last:
                log(msg)
                last = msg.split(",")[0]
            if not alive:
                cl = open(os.path.join(job["dir"], "client.log"), errors="replace").read() \
                    if os.path.exists(os.path.join(job["dir"], "client.log")) else ""
                status = "completed" if "Task completed successfully" in cl else \
                    ("service_timeout" if "timeout" in cl.lower() else "failed")
                self.end_job(job, status)
                self.job = None
                return status
            if now() >= job["deadline"]:
                self.cancel(job, "session quota budget reached")
                self.end_job(job, "budget_cancelled")
                self.job = None
                return "budget_cancelled"
            time.sleep(POLL_S)

    # -- main
    def summary(self):
        ts = [self.st["tasks"][n] for n in self.st["task_order"]]
        npass = sum(1 for t in ts if t["status"] in ("done", "error") and t.get("passed"))
        fin = sum(1 for t in ts if t["status"] == "done") + \
            sum(1 for t in ts if t["status"] == "error" and t["attempts"] >= MAX_ERROR_ATTEMPTS)
        err = sum(1 for t in ts if t["status"] == "error")
        rows = ["task\tstatus\tpassed\tattempts\tinterrupted\tpreprocess\trunning\tevaluation\tjob_id"]
        for n in self.st["task_order"]:
            t = self.st["tasks"][n]
            rows.append("\t".join(str(x) for x in (n, t["status"], t.get("passed"), t.get("attempts", 0),
                                                   t.get("interrupted", 0), t.get("preprocess"), t.get("running"),
                                                   t.get("evaluation"), t.get("job_id"))))
        with open(os.path.join(self.out, "results.tsv"), "w") as f:
            f.write("\n".join(rows) + "\n")
        mine = [e for e in self.ledger.entries if e.get("label") == self.a.label and not e.get("rejected")]
        s = {"label": self.a.label, "model": self.a.model, "n_tasks": len(ts), "passed": npass,
             "finished": fin, "errors": err, "pending": len(self.pending()),
             "pass_rate": round(npass / len(ts), 4) if ts else None,
             "jobs": len(self.st["jobs"]), "service_minutes_this_label": round(sum(
                 e.get("duration_s", 0) for e in mine) / 60, 1),
             "window_used_min": round(self.ledger.used_s() / 60, 1), "per_utc_day_min": self.ledger.per_utc_day()}
        jsave(os.path.join(self.out, "summary.json"), s)
        return s

    def run(self):
        self.sync_task_list()
        self.save()
        self.cleanup_stale()
        submits = 0
        while True:
            pend = self.pending()
            if not pend:
                return 0
            if STOP["sig"]:
                return 130
            used = self.ledger.used_s()
            remaining = BUDGET_S - used
            log(f"pending {len(pend)} tasks; window usage {used / 60:.1f} min of {QUOTA_MIN:g} "
                f"(budget {BUDGET_S / 60:g}, remaining {remaining / 60:.1f}); per UTC day: {self.ledger.per_utc_day()}")
            if remaining < MIN_SESSION_MIN * 60 or submits >= MAX_SUBMITS:
                t = self.ledger.next_eligible(MIN_SESSION_MIN * 60)
                if submits >= MAX_SUBMITS:
                    t = max(t, now() + 600)
                self.st["next_eligible"] = t
                self.save()
                log(f"QUOTA: not enough budget for another session; NEXT_ELIGIBLE {iso(t)} ({int(t)})")
                emit(f"NEXT_ELIGIBLE {int(t)}")
                return 3
            state, info = self.svc.check() if not DRY_RUN else ("idle", {})
            t0 = now()
            while state == "busy" and now() - t0 < BUSY_WAIT_MIN * 60 and not STOP["sig"]:
                log(f"service busy with another job (started {info.get('started_at')}, model {info.get('model')}); waiting")
                time.sleep(BUSY_POLL_S)
                state, info = self.svc.check()
            if state == "down":
                log(f"service unreachable at {self.svc.base}: {info}")
                emit(f"NEXT_ELIGIBLE {int(now() + 3600)}")
                return 5
            if state == "busy":
                emit(f"NEXT_ELIGIBLE {int(now() + 1800)}")
                return 4
            # the clock for the session budget starts at submission
            remaining = BUDGET_S - self.ledger.used_s()
            job = self.submit(pend, remaining)
            submits += 1
            if DRY_RUN:
                return 0
            if job == "busy":
                continue
            if job == "ratelimited":
                t = now() + 3600
                emit(f"NEXT_ELIGIBLE {int(t)}")
                return 3
            if job == "error" or job is None:
                return 5
            res = self.follow(job)
            if res == "interrupted":
                return 130


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--workers", type=int, default=10)
    p.add_argument("--llm-base-url", required=True)
    p.add_argument("--server-host", required=True)
    p.add_argument("--server-port", type=int, default=8080)
    p.add_argument("--ws-port", type=int, default=8081)
    p.add_argument("--tasks-file", required=True)
    p.add_argument("--model-params", default="")
    p.add_argument("--client-dir", required=True)
    p.add_argument("--python", required=True)
    p.add_argument("--ledger", required=True)
    p.add_argument("--lock", required=True)
    p.add_argument("--status", action="store_true", help="print state/quota summary and exit")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    LOGFILE["f"] = open(os.path.join(a.out, "driver.log"), "a")
    lockf = open(a.lock, "a+")
    try:
        fcntl.flock(lockf, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        log(f"another Toolathlon service run holds {a.lock} (one job at a time per IP); exiting")
        emit(f"NEXT_ELIGIBLE {int(now() + 1800)}")
        return 6
    d = Driver(a)
    if a.status:
        d.sync_task_list()
        print(json.dumps(d.summary(), indent=1))
        return 0

    def on_sig(signum, _frame):
        STOP["sig"] = signum
    signal.signal(signal.SIGTERM, on_sig)
    signal.signal(signal.SIGINT, on_sig)
    signal.signal(signal.SIGHUP, on_sig)
    try:
        rc = d.run()
    finally:
        if d.job is not None:  # exception while following a job: never leave it burning quota
            d.cancel(d.job, "driver error")
            d.end_job(d.job, "driver_error")
        s = d.summary()
        log("summary: " + json.dumps(s))
    return rc


if __name__ == "__main__":
    sys.exit(main())
