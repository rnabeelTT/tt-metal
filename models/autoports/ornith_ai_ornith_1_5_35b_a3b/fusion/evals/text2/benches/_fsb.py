"""FullStack Bench (ByteDance), en / zh: 1,687 problems each, 16 languages, executed in SandboxFusion.

Protocol = the official runner (FullStackBench/src/main.py @ a3c99a9):
  prompt     = sample["content"] as the single user turn (zero-shot, no template)
  evaluation = SandboxFusion POST /submit {dataset: "FullStackBench", id, completion,
               config: {compile_timeout: 50, run_timeout: 50, dataset_type: "AutoEvalDataset", provided_data: sample}}
  metric     = pass rate = mean(accepted)
Data: FullStackBench/data/fsb_{en,zh}_20241204.jsonl (repo copy, same version as the pinned sandbox image).
Sandbox: docker image volcengine/sandbox-fusion:server-20241204 (the tag pinned in the FullStackBench README), container
  text2-sandboxfusion on 127.0.0.1:28250 (started on demand; never touches other containers). Deviation from main.py:
  thinking-mode sampling from RunCfg instead of temperature 0 / max_tokens 2048; only `content` is submitted.

Env: FSB_SANDBOX_URL (default http://127.0.0.1:28250)  FSB_EXEC_CONCURRENCY (default 8, max 8)
"""
import asyncio
import hashlib
import json
import os
import subprocess
import time
from collections import defaultdict
from pathlib import Path

import httpx

from common.core import count_errors, finish, generate_chat, read_jsonl, select_items, write_gold_map, write_jsonl

DATA_DIR = Path.home() / "evals/src/FullStackBench/data"
IMAGE = "volcengine/sandbox-fusion:server-20241204"
CONTAINER = "text2-sandboxfusion"
PORT = 28250
FENCE = {"D": "d", "R": "r"}


def ensure_sandbox(url: str) -> None:
    def ok():
        try:
            return httpx.get(url + "/v1/ping", timeout=5).status_code == 200
        except Exception:  # noqa: BLE001
            return False

    if ok():
        return
    if url.rstrip("/") != f"http://127.0.0.1:{PORT}":
        raise SystemExit(f"sandbox not reachable at {url}")
    st = subprocess.run(["docker", "inspect", "-f", "{{.State.Running}}", CONTAINER], capture_output=True, text=True)
    if st.returncode == 0 and st.stdout.strip() != "true":
        subprocess.run(["docker", "start", CONTAINER], check=True, capture_output=True)
    elif st.returncode != 0:
        subprocess.run(["docker", "run", "-d", "--name", CONTAINER, "-p", f"127.0.0.1:{PORT}:8080", "--cpus", "8",
                        "--memory", "64g", IMAGE], check=True, capture_output=True)
    for _ in range(120):
        if ok():
            return
        time.sleep(1)
    raise SystemExit("SandboxFusion did not come up")


async def _submit_all(url: str, subs: list[dict], conc: int) -> list[dict]:
    sem = asyncio.Semaphore(conc)
    async with httpx.AsyncClient(timeout=httpx.Timeout(600, connect=30)) as http:
        async def one(s):
            body = {"dataset": "FullStackBench", "id": s["sample"]["id"], "completion": s["completion"],
                    "config": {"compile_timeout": 50, "run_timeout": 50, "dataset_type": "AutoEvalDataset",
                               "provided_data": s["sample"]}}
            last = None
            for attempt in range(4):
                try:
                    async with sem:
                        r = await http.post(url + "/submit", json=body)
                    r.raise_for_status()
                    j = r.json()
                    ex = (j.get("tests") or [{}])[0].get("exec_info") or {}
                    return {"rid": s["rid"], "accepted": bool(j.get("accepted")), "status": ex.get("status"),
                            "compile_status": (ex.get("compile_result") or {}).get("status"),
                            "run_status": (ex.get("run_result") or {}).get("status"),
                            "stderr_tail": ((ex.get("run_result") or {}).get("stderr") or "")[-500:],
                            "extracted_code_chars": len(j.get("extracted_code") or "")}
                except Exception as e:  # noqa: BLE001
                    last = f"{type(e).__name__}: {e}"
                    await asyncio.sleep(2 ** attempt)
            return {"rid": s["rid"], "accepted": False, "status": "SubmitError", "error": last}

        return await asyncio.gather(*(one(s) for s in subs))


def run(cfg, locale: str):
    url = os.environ.get("FSB_SANDBOX_URL", f"http://127.0.0.1:{PORT}").rstrip("/")
    conc = max(1, min(8, int(os.environ.get("FSB_EXEC_CONCURRENCY", 8))))
    path = DATA_DIR / f"fsb_{locale}_20241204.jsonl"
    rows = read_jsonl(path)
    items = select_items(rows, cfg, id_key="id")
    cfg.dump(module=cfg.bench, data=str(path), sandbox_image=IMAGE, sandbox_url=url, exec_concurrency=conc)
    ensure_sandbox(url)

    def msgs(it):
        return [{"role": "user", "content": it["content"]}]

    lang_of = lambda it: it["labels"]["programming_language"]  # noqa: E731
    write_gold_map([(msgs(it), f"```{FENCE.get(lang_of(it), lang_of(it))}\n{it['canonical_solution']}\n```")
                    for it in items])
    resps = generate_chat(cfg, [(str(it["id"]), msgs(it)) for it in items])

    # judge (cached per id + content)
    cache_p = cfg.out / "judged.jsonl"
    cache = {(r["rid"], r["completion_sha"]): r for r in read_jsonl(cache_p)}
    subs = []
    for it in items:
        c = resps.get(str(it["id"]), {}).get("content") or ""
        sha = hashlib.sha256(c.encode()).hexdigest()
        if (str(it["id"]), sha) not in cache:
            subs.append({"rid": str(it["id"]), "completion": c, "sample": it, "sha": sha})
    print(f"[{cfg.bench}] executing {len(subs)} submissions in SandboxFusion ({len(items) - len(subs)} cached)", flush=True)
    if subs:
        t0 = time.time()
        res = asyncio.run(_submit_all(url, subs, conc))
        with open(cache_p, "a") as f:
            for s, r in zip(subs, res):
                r["completion_sha"] = s["sha"]
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                cache[(r["rid"], s["sha"])] = r
        print(f"[{cfg.bench}] execution took {(time.time() - t0) / 60:.1f} min", flush=True)

    scores, by_lang, by_cat, by_diff = [], defaultdict(list), defaultdict(list), defaultdict(list)
    for it in items:
        r = resps.get(str(it["id"]), {})
        c = r.get("content") or ""
        j = cache[(str(it["id"]), hashlib.sha256(c.encode()).hexdigest())]
        lab = it["labels"]
        scores.append({"id": it["id"], "language": lab["programming_language"], "category": lab["category"],
                       "difficulty": lab["difficulty"], "accepted": j["accepted"], "status": j.get("status"),
                       "finish_reason": r.get("finish_reason"), "error": r.get("error"),
                       "submit_error": j.get("error")})
        by_lang[lab["programming_language"]].append(j["accepted"])
        by_cat[lab["category"]].append(j["accepted"])
        by_diff[lab["difficulty"]].append(j["accepted"])
    write_jsonl(cfg.out / "scores.jsonl", scores)
    pct = lambda v: 100 * sum(v) / max(1, len(v))  # noqa: E731
    finish(cfg, "pass_rate", pct([s["accepted"] for s in scores]), len(items), {
        "by_language": {k: {"n": len(v), "pass": pct(v)} for k, v in sorted(by_lang.items())},
        "by_category": {k: {"n": len(v), "pass": pct(v)} for k, v in sorted(by_cat.items())},
        "by_difficulty": {k: {"n": len(v), "pass": pct(v)} for k, v in sorted(by_diff.items())},
        "sandbox_submit_errors": sum(1 for s in scores if s["submit_error"]),
        "truncated_responses": sum(1 for s in scores if s["finish_reason"] == "length"),
        "request_errors": count_errors(resps),
    })
