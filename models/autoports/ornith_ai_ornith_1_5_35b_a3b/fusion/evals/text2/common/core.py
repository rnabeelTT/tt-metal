"""Shared plumbing for text2 benchmarks: sampling config, async OpenAI-compatible chat calls with
retries + resume, fixed subsampling, result/judge file conventions.

Conventions (every bench module follows them):
  OUT/config.json        run config (bench, model, sampling, item count, sample ids file)
  OUT/sample_ids.json    ids actually evaluated (written whenever a subsample / LIMIT applies)
  OUT/responses.jsonl    one line per request: id, content, reasoning_chars, tool_calls, finish_reason, usage, error
  OUT/reasoning.jsonl    id -> full reasoning text (SAVE_REASONING=0 disables)
  OUT/scores.jsonl       per-item scoring details
  OUT/summary.json       metrics
  OUT/judge_inputs.jsonl when the official metric needs an LLM judge (see common/judge.py)
  stdout last line:      RESULT <bench> <metric>=<value> n=<items>   (+ JUDGE_PENDING <path> before it)
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterable

import httpx

SAMPLE_SEED = 1234
SAMPLE_SIZE = 1000
SAMPLE_THRESHOLD = 3000


# ----------------------------------------------------------------------------------------------
# configuration
# ----------------------------------------------------------------------------------------------
def default_sampling(model_id: str) -> dict:
    """Card-protocol defaults per model (overridable by env)."""
    m = model_id.lower()
    if "35b" in m and "ornith" in m:
        return {"temperature": 1.0, "top_p": 0.95, "top_k": 20}
    return {"temperature": 0.6, "top_p": 0.95, "top_k": 20}


@dataclass
class RunCfg:
    bench: str
    label: str
    base_url: str
    model: str
    concurrency: int
    out: Path
    temperature: float
    top_p: float
    top_k: int
    max_tokens: int
    presence_penalty: float | None
    limit: int | None
    timeout: float
    retries: int
    save_reasoning: bool
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_env(cls, bench: str, label: str, base_url: str, model: str, concurrency: int) -> "RunCfg":
        d = default_sampling(model)
        env = os.environ
        runs_root = Path(env.get("RUNS_ROOT", str(Path.home() / "ornith-fusion" / "runs")))
        pp = env.get("PRESENCE_PENALTY")
        lim = env.get("LIMIT")
        return cls(
            bench=bench,
            label=label,
            base_url=base_url.rstrip("/"),
            model=model,
            concurrency=int(concurrency),
            out=runs_root / label / bench,
            temperature=float(env.get("TEMPERATURE", d["temperature"])),
            top_p=float(env.get("TOP_P", d["top_p"])),
            top_k=int(env.get("TOP_K", d["top_k"])),
            max_tokens=int(env.get("MAX_TOKENS", 32768)),
            presence_penalty=float(pp) if pp not in (None, "") else None,
            limit=int(lim) if lim not in (None, "", "0") else None,
            timeout=float(env.get("REQUEST_TIMEOUT", 14400)),
            retries=int(env.get("MAX_RETRIES", 6)),
            save_reasoning=env.get("SAVE_REASONING", "1") != "0",
        )

    def sampling_body(self) -> dict:
        b = {"temperature": self.temperature, "top_p": self.top_p, "max_tokens": self.max_tokens}
        if self.top_k and self.top_k > 0:
            b["top_k"] = self.top_k  # vLLM accepts top_k as a top-level extra param
        if self.presence_penalty is not None:
            b["presence_penalty"] = self.presence_penalty
        return b

    def dump(self, **more) -> None:
        self.out.mkdir(parents=True, exist_ok=True)
        d = {k: (str(v) if isinstance(v, Path) else v) for k, v in self.__dict__.items()}
        d.update(more)
        d["time"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        (self.out / "config.json").write_text(json.dumps(d, indent=2, ensure_ascii=False))


# ----------------------------------------------------------------------------------------------
# sampling of large sets / LIMIT
# ----------------------------------------------------------------------------------------------
def select_items(items: list[dict], cfg: RunCfg, id_key: str = "id", force_sample: int | None = None) -> list[dict]:
    """>3000 items -> fixed random 1000 (seed 1234). LIMIT -> first N of the (sampled) list.
    The chosen ids are saved to OUT/sample_ids.json."""
    sel = items
    note = "full set"
    env_n = int(os.environ["SAMPLE_N"]) if os.environ.get("SAMPLE_N") else None  # run-size cap (1-2 h runs)
    n = force_sample or env_n or (SAMPLE_SIZE if len(items) > SAMPLE_THRESHOLD else None)
    if n and n < len(items):
        rng = random.Random(SAMPLE_SEED)
        idx = sorted(rng.sample(range(len(items)), n))
        sel = [items[i] for i in idx]
        note = f"random {n} of {len(items)} (seed {SAMPLE_SEED})"
    if cfg.limit:
        sel = sel[: cfg.limit]
        note += f", LIMIT {cfg.limit}"
    cfg.out.mkdir(parents=True, exist_ok=True)
    (cfg.out / "sample_ids.json").write_text(
        json.dumps({"note": note, "total": len(items), "ids": [str(x[id_key]) for x in sel]}, ensure_ascii=False)
    )
    print(f"[{cfg.bench}] items: {len(sel)} ({note})", flush=True)
    return sel


# ----------------------------------------------------------------------------------------------
# HTTP
# ----------------------------------------------------------------------------------------------
def check_server(cfg: RunCfg) -> None:
    try:
        r = httpx.get(cfg.base_url + "/models", timeout=20)
        r.raise_for_status()
        ids = [m["id"] for m in r.json()["data"]]
    except Exception as e:  # noqa: BLE001
        sys.exit(f"server not reachable: {cfg.base_url}/models ({e})")
    if cfg.model not in ids:
        sys.exit(f"model {cfg.model} not served; have {ids}")


def parse_message(msg: dict) -> dict:
    reasoning = msg.get("reasoning") or msg.get("reasoning_content") or ""
    return {
        "content": msg.get("content") or "",
        "reasoning": reasoning,
        "tool_calls": msg.get("tool_calls") or None,
    }


class Client:
    """Minimal async chat-completions client (raw httpx, so every vLLM extra param passes through)."""

    def __init__(self, cfg: RunCfg):
        self.cfg = cfg
        self.http = httpx.AsyncClient(
            timeout=httpx.Timeout(cfg.timeout, connect=30),
            limits=httpx.Limits(max_connections=cfg.concurrency * 2 + 4, max_keepalive_connections=cfg.concurrency * 2 + 4),
            headers={"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', 'EMPTY')}"},
        )
        self.sem = asyncio.Semaphore(cfg.concurrency)

    async def chat(self, messages: list[dict], tools: list | None = None, **override) -> dict:
        body = {"model": self.cfg.model, "messages": messages, **self.cfg.sampling_body()}
        if tools:
            body["tools"] = tools
            body.setdefault("tool_choice", "auto")
        body.update({k: v for k, v in override.items() if v is not None})
        last_err = None
        for attempt in range(self.cfg.retries + 1):
            t0 = time.time()
            try:
                async with self.sem:
                    r = await self.http.post(self.cfg.base_url + "/chat/completions", json=body)
                if r.status_code == 400:
                    # deterministic client error (e.g. context overflow) -> no retry
                    return {"error": f"HTTP 400: {r.text[:2000]}", "content": "", "reasoning": "", "tool_calls": None,
                            "finish_reason": None, "usage": None, "latency": time.time() - t0}
                r.raise_for_status()
                j = r.json()
                ch = j["choices"][0]
                out = parse_message(ch["message"])
                out.update(finish_reason=ch.get("finish_reason"), usage=j.get("usage"), latency=time.time() - t0, error=None)
                return out
            except Exception as e:  # noqa: BLE001
                last_err = f"{type(e).__name__}: {e}"
                await asyncio.sleep(min(60, 2 ** attempt))
        return {"error": last_err, "content": "", "reasoning": "", "tool_calls": None, "finish_reason": None,
                "usage": None, "latency": None}

    async def aclose(self):
        await self.http.aclose()


# ----------------------------------------------------------------------------------------------
# generation loop with resume
# ----------------------------------------------------------------------------------------------
def read_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    out = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def write_jsonl(p: Path, rows: Iterable[dict]) -> None:
    with open(p, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


async def _generate(cfg: RunCfg, jobs: list[tuple[str, Callable[[Client], Awaitable[dict]]]]) -> dict[str, dict]:
    """jobs: (request_id, coroutine-factory(client) -> result dict). Results with error=None are resumable."""
    resp_p = cfg.out / "responses.jsonl"
    reas_p = cfg.out / "reasoning.jsonl"
    done = {r["id"]: r for r in read_jsonl(resp_p) if not r.get("error")}
    todo = [(rid, fn) for rid, fn in jobs if rid not in done]
    print(f"[{cfg.bench}] requests: {len(jobs)} total, {len(done)} cached, {len(todo)} to run", flush=True)
    if not todo:
        return done
    client = Client(cfg)
    lock = asyncio.Lock()
    n_done = 0
    t_start = time.time()
    fresp = open(resp_p, "a")
    freas = open(reas_p, "a") if cfg.save_reasoning else None

    async def one(rid, fn):
        nonlocal n_done
        res = await fn(client)
        rec = {"id": rid, **{k: v for k, v in res.items() if k != "reasoning"}}
        rec["reasoning_chars"] = len(res.get("reasoning") or "")
        async with lock:
            fresp.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fresp.flush()
            if freas is not None and res.get("reasoning"):
                freas.write(json.dumps({"id": rid, "reasoning": res["reasoning"]}, ensure_ascii=False) + "\n")
                freas.flush()
            done[rid] = rec
            n_done += 1
            if n_done % max(1, len(todo) // 20) == 0 or n_done == len(todo):
                el = time.time() - t_start
                errs = sum(1 for r in done.values() if r.get("error"))
                print(f"[{cfg.bench}] {n_done}/{len(todo)} done, {el/60:.1f} min, errors so far {errs}", flush=True)

    try:
        await asyncio.gather(*(one(rid, fn) for rid, fn in todo))
    finally:
        fresp.close()
        if freas:
            freas.close()
        await client.aclose()
    # keep the newest record per id
    latest = {}
    for r in read_jsonl(resp_p):
        if r["id"] not in latest or latest[r["id"]].get("error"):
            latest[r["id"]] = r
    return latest


def generate_chat(cfg: RunCfg, requests: list[tuple[str, list[dict]]], **override) -> dict[str, dict]:
    """Single-shot chat for each (id, messages)."""
    jobs = [(rid, (lambda msgs: (lambda c: c.chat(msgs, **override)))(m)) for rid, m in requests]
    return asyncio.run(_generate(cfg, jobs))


def generate_custom(cfg: RunCfg, jobs) -> dict[str, dict]:
    return asyncio.run(_generate(cfg, jobs))


# ----------------------------------------------------------------------------------------------
# stub-server gold map (self-tests only)
# ----------------------------------------------------------------------------------------------
def stub_key(messages: list[dict]) -> str:
    """Key used by stub_server.py: sha256 of the last message's content (JSON-dumped if not str)."""
    c = messages[-1].get("content")
    if not isinstance(c, str):
        c = json.dumps(c, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(c.encode()).hexdigest()


def write_gold_map(entries: list[tuple[list[dict], Any]]) -> None:
    """When STUB_GOLD_MAP is set, append {key: response} lines the stub server will answer with.
    response: str (content) or {"content":..., "tool_calls":[...]}"""
    p = os.environ.get("STUB_GOLD_MAP")
    if not p:
        return
    with open(p, "a") as f:
        for msgs, resp in entries:
            f.write(json.dumps({"key": stub_key(msgs), "response": resp}, ensure_ascii=False) + "\n")


# ----------------------------------------------------------------------------------------------
# result reporting
# ----------------------------------------------------------------------------------------------
def finish(cfg: RunCfg, metric: str, value: float | None, n: int, summary: dict, judge_pending: Path | None = None) -> None:
    summary = {"bench": cfg.bench, "label": cfg.label, "model": cfg.model, "metric": metric, "value": value, "n": n, **summary}
    (cfg.out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    errs = summary.get("request_errors")
    if errs:
        print(f"[{cfg.bench}] WARNING: {errs} requests failed after retries (scored as wrong); rerun the same command to retry them")
    if judge_pending:
        print(f"JUDGE_PENDING {judge_pending}")
    v = "pending" if value is None else (f"{value:.4f}" if isinstance(value, float) else str(value))
    line = f"RESULT {cfg.bench} {metric}={v} n={n}"
    (cfg.out / "result.txt").write_text(line + (f"\nJUDGE_PENDING {judge_pending}" if judge_pending else "") + "\n")
    print(line, flush=True)


def count_errors(resps: dict[str, dict]) -> int:
    return sum(1 for r in resps.values() if r.get("error"))
