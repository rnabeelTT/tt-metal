"""OJBench (232 NOI/ICPC competition problems), official prompts + official judge (ojbench.judge_jsonl over DMOJ).

Data:   ~/evals/ojbench/OJBench_testdata (HF He-Ren/OJBench_testdata; prompts/full.jsonl = 232 problems x {cpp, python})
Judge:  docker image text2-ojbench (python 3.11 + g++ 12 + pypy3, DMOJ judge-server @ f098cd3 as pinned by the OJBench
        README, OJBench @ 5e94480), run as a throw-away container per judging pass (needs SYS_PTRACE for DMOJ's sandbox).
        Official judge settings: time limit 10 s, memory 1 GB, all test cases run, verdict = first non-AC case.
        3 problems are skipped by the official judger (_skip_id: nwerc2023_H, swerc2023_L, loj-3897) and count as fail.
Prompt: the dataset's `prompt` verbatim as the single user turn; the judge extracts the last ```cpp / ```python block of
        `content` (proc_code).
Metric: pass@1 (%) = mean AC over problems x N_SAMPLES samples (official leaderboard: pass@1 averaged over 8 samples
        and pass@8; primary column C++).

Env: OJ_LANG=cpp|python|both (default cpp)  N_SAMPLES (default 1; leaderboard uses 8)  OJ_JUDGE_WORKERS (default 8, max 8)
     OJ_IDS=comma list of problem ids (restrict; e.g. self-tests)  OJ_DATA (testdata dir)
"""
import hashlib
import json
import os
import subprocess
from collections import Counter, defaultdict
from math import comb
from pathlib import Path

from common.core import count_errors, finish, generate_chat, read_jsonl, select_items, write_gold_map, write_jsonl

BENCH_NAME = "ojbench"
SUPPORT = Path.home() / "evals/ojbench"
DATA = Path(os.environ.get("OJ_DATA", SUPPORT / "OJBench_testdata"))
IMAGE = "text2-ojbench:latest"
SKIP_IDS = {"nwerc2023_H", "swerc2023_L", "loj-3897"}


def pass_at_k(n, c, k):
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def judge(cfg, entries: list[dict]) -> dict[str, dict]:
    """entries: {rid, id, language, content}. Returns rid -> {verdict, is_passed, ...}; cached in judged.jsonl."""
    cache_p = cfg.out / "judged.jsonl"
    cache = {}
    for r in read_jsonl(cache_p):
        cache[(r["rid"], r["content_sha"])] = r
    todo = []
    for e in entries:
        e["content_sha"] = hashlib.sha256(e["content"].encode()).hexdigest()
        if (e["rid"], e["content_sha"]) not in cache:
            todo.append(e)
    print(f"[ojbench] judging {len(todo)} submissions ({len(entries) - len(todo)} cached)", flush=True)
    if todo:
        work = cfg.out / "judge_work"
        work.mkdir(parents=True, exist_ok=True)
        write_jsonl(work / "in.jsonl", todo)
        (work / "judge_driver.py").write_text((SUPPORT / "judge_driver.py").read_text())
        workers = max(1, min(8, int(os.environ.get("OJ_JUDGE_WORKERS", 8))))
        name = f"text2-ojbench-{cfg.label}-{os.getpid()}".replace("/", "_")[:120]
        cmd = ["docker", "run", "--rm", "--name", name, "--user", f"{os.getuid()}:{os.getgid()}", "--cap-add", "SYS_PTRACE", "--security-opt", "seccomp=unconfined",
               f"--cpus={workers}", "--memory=48g", "--network=none",
               "-v", f"{DATA}:/data:ro", "-v", f"{work}:/work", IMAGE,
               "python", "/work/judge_driver.py", "/work/in.jsonl", "/work/out.jsonl", str(workers)]
        with open(work / "judge_stdout.log", "w") as lf:
            rc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT).returncode
        if rc != 0:
            raise SystemExit(f"judge container failed (rc={rc}); see {work}/judge_stdout.log")
        with open(cache_p, "a") as f:
            for r in read_jsonl(work / "out.jsonl"):
                det = r.get("detailed_results") or {}
                cases = det.get("results", []) if isinstance(det, dict) else []
                slim = {
                    "rid": r["rid"], "content_sha": r["content_sha"], "id": r["id"], "language": r["language"],
                    "verdict": r.get("verdict"), "is_passed": bool(r.get("is_passed")), "is_skip": bool(r.get("is_skip")),
                    "case_codes": [c.get("readable_main_code") for c in cases],
                    "first_feedback": next((c.get("feedback") for c in cases if c.get("feedback")), None),
                }
                f.write(json.dumps(slim, ensure_ascii=False) + "\n")
                cache[(slim["rid"], slim["content_sha"])] = slim
        (work / "out.jsonl").unlink(missing_ok=True)
    return {e["rid"]: cache[(e["rid"], e["content_sha"])] for e in entries}


def main(cfg):
    lang_opt = os.environ.get("OJ_LANG", "cpp")
    langs = ["cpp", "python"] if lang_opt == "both" else [lang_opt]
    n_samples = int(os.environ.get("N_SAMPLES", 1))
    rows = read_jsonl(DATA / "prompts/full.jsonl")
    ids_filter = {s.strip() for s in os.environ.get("OJ_IDS", "").split(",") if s.strip()}
    items = []
    for r in rows:
        if r["language"] not in langs:
            continue
        if ids_filter and str(r["id"]) not in ids_filter:
            continue
        items.append({**r, "pid": str(r["id"]), "key": f"{r['id']}|{r['language']}"})
    items = select_items(items, cfg, id_key="key")
    cfg.dump(module="ojbench", languages=langs, n_samples=n_samples, data=str(DATA), judge_image=IMAGE,
             judge_time_limit_s=10, judge_memory_mb=1024)

    # self-test gold answers (hand-written known-correct solutions for a few easy problems; the dataset ships none)
    known = json.loads((SUPPORT / "known_solutions/solutions.json").read_text())
    write_gold_map([([{"role": "user", "content": it["prompt"]}], known[it["key"]]) for it in items if it["key"] in known])

    reqs = [(f"{it['key']}#{s}", [{"role": "user", "content": it["prompt"]}]) for it in items for s in range(n_samples)]
    resps = generate_chat(cfg, reqs)

    entries = []
    for it in items:
        for s in range(n_samples):
            rid = f"{it['key']}#{s}"
            r = resps.get(rid, {})
            entries.append({"rid": rid, "id": it["id"], "language": it["language"], "content": r.get("content") or ""})
    judged = judge(cfg, entries)

    scores, by = [], defaultdict(list)
    for it in items:
        for s in range(n_samples):
            rid = f"{it['key']}#{s}"
            j = judged[rid]
            r = resps.get(rid, {})
            scores.append({"id": rid, "language": it["language"], "difficulty": it["difficulty"], "dataset": it["dataset"],
                           "verdict": j["verdict"], "passed": j["is_passed"], "skipped": j["is_skip"],
                           "finish_reason": r.get("finish_reason"), "error": r.get("error")})
            by[(it["language"], it["key"])].append(j["is_passed"])
    write_jsonl(cfg.out / "scores.jsonl", scores)

    summary = {"n_samples_per_problem": n_samples, "request_errors": count_errors(resps),
               "truncated_responses": sum(1 for s in scores if s["finish_reason"] == "length"), "per_language": {}}
    for lang in langs:
        ss = [s for s in scores if s["language"] == lang]
        d = {"pass@1": 100 * sum(s["passed"] for s in ss) / max(1, len(ss)), "problems": len(ss) // max(1, n_samples),
             "skipped_by_official_judge": sum(s["skipped"] for s in ss),
             "verdicts": dict(Counter(s["verdict"] for s in ss).most_common())}
        for key in ("difficulty", "dataset"):
            g = defaultdict(list)
            for s in ss:
                g[s[key]].append(s["passed"])
            d[f"pass@1_by_{key}"] = {k: 100 * sum(v) / len(v) for k, v in sorted(g.items())}
        if n_samples > 1:
            d[f"pass@{n_samples}"] = 100 * sum(pass_at_k(n_samples, sum(v), n_samples)
                                                for (lg, _), v in by.items() if lg == lang) / max(1, d["problems"])
        summary["per_language"][lang] = d
    head = langs[0]
    finish(cfg, f"pass@1_{head}", summary["per_language"][head]["pass@1"], len(items) if len(langs) == 1 else
           len(items) // len(langs), summary)
