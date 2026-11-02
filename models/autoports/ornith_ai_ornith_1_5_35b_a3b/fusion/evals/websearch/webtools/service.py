"""webtools: search + fetch service for the search-agent benchmarks (one process, shared by all runs).

  python -m webtools.service [--port 28011]      (run_webtools.sh starts it as a user systemd unit)

POST /search {query, bench, tag, question?, n?, lang?, blocklist?}
    -> {results: [{title, url, snippet, engines}], n_raw, n_filtered, cached, engines_used, unresponsive}
     | {blocked: rule} | {error}
POST /fetch  {url, bench, tag, max_chars?, raw?, blocklist?}
    -> {url, final_url, title, content, truncated, total_chars, cached, content_type} | {blocked: rule} | {error}
GET  /stats, /health

* SearXNG backend (SEARXNG_URL, default http://127.0.0.1:28010), global throttle SEARCH_QPS (default 1.0 query/s
  across all callers, cache hits are free). Primary engines by query script (CJK -> zh engines), fallback engines
  when the primary set returns < 3 results. Circuit breaker: when >= 50% of the last 20 uncached searches of a
  language group came back empty (engines captcha / rate-limit this host), a search with no results pauses and
  retries (60 s doubling to 600 s, up to SEARCH_EMPTY_WAIT_MAX = 3600 s) instead of returning "no results".
* Cache: sqlite (cache/webcache.sqlite, zlib-compressed). Search results cached when non-empty; fetches cached on
  success and on permanent errors (4xx, robots, unsupported type). The blocklist is re-applied to cached data.
* Blocklist profiles: full (CAIS HLE blocklist + extra_blocklist.json, default), bench (extra only), none.
* Logs: logs/calls.jsonl (every call), logs/blocked.jsonl (every blocked query / result / page, with the rule).
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import sqlite3
import time
import zlib
from collections import Counter, deque
from pathlib import Path

import httpx
import uvicorn
from fastapi import FastAPI, Request

from .blocklist import HERE as BL_DIR
from .blocklist import Blocklist
from .fetch import USER_AGENT, FetchError, Robots, fetch_document, html_to_markdown

ROOT = Path(__file__).resolve().parent.parent
SEARXNG_URL = os.environ.get("SEARXNG_URL", "http://127.0.0.1:28010")
SEARCH_QPS = float(os.environ.get("SEARCH_QPS", "1.0"))
FETCH_TIMEOUT = float(os.environ.get("FETCH_TIMEOUT", "30"))
FETCH_PARALLEL = int(os.environ.get("FETCH_PARALLEL", "12"))
EMPTY_WAIT_MAX = float(os.environ.get("SEARCH_EMPTY_WAIT_MAX", "3600"))
ENGINES = {
    # SearXNG's bing engine returns unrelated/spam results from this host (2026-10-28 check); yahoo is Bing-backed
    # and returns relevant results. google: "access denied", brave: "too many requests" from this host -- kept in the
    # set (SearXNG suspends them, so they cost nothing) in case they recover.
    "en": os.environ.get("ENGINES_EN", "duckduckgo,yahoo,brave,google"),
    "en_fallback": os.environ.get("ENGINES_EN_FALLBACK", "mojeek"),
    "zh": os.environ.get("ENGINES_ZH", "baidu,sogou,360search"),
    "zh_fallback": os.environ.get("ENGINES_ZH_FALLBACK", "duckduckgo,yahoo"),
}
CJK = re.compile(r"[㐀-䶿一-鿿]")

app = FastAPI()
STATE: dict = {}


# ------------------------------------------------------------------------------------------------ cache
class Cache:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("CREATE TABLE IF NOT EXISTS search (key TEXT PRIMARY KEY, query TEXT, lang TEXT, data BLOB, ts REAL)")
        self.db.execute("CREATE TABLE IF NOT EXISTS fetch (key TEXT PRIMARY KEY, url TEXT, data BLOB, ts REAL)")
        self.lock = asyncio.Lock()

    def get(self, table: str, key: str):
        row = self.db.execute(f"SELECT data FROM {table} WHERE key=?", (key,)).fetchone()
        return json.loads(zlib.decompress(row[0])) if row else None

    def put(self, table: str, key: str, label: str, data: dict, lang: str = ""):
        blob = zlib.compress(json.dumps(data, ensure_ascii=False).encode(), 6)
        if table == "search":
            self.db.execute("INSERT OR REPLACE INTO search VALUES (?,?,?,?,?)", (key, label, lang, blob, time.time()))
        else:
            self.db.execute("INSERT OR REPLACE INTO fetch VALUES (?,?,?,?)", (key, label, blob, time.time()))


def _key(*parts) -> str:
    return hashlib.sha256("\x1f".join(str(p) for p in parts).encode()).hexdigest()


# ------------------------------------------------------------------------------------------------ logging
def _log(name: str, rec: dict):
    rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), **rec}
    with open(ROOT / "logs" / name, "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _blocklist(profile: str | None) -> Blocklist | None:
    p = profile or os.environ.get("BLOCKLIST", "full")
    return STATE["blocklists"].get(p, STATE["blocklists"]["full"])


# ------------------------------------------------------------------------------------------------ search
async def _throttle():
    async with STATE["throttle_lock"]:
        wait = STATE["next_search"] - time.monotonic()
        if wait > 0:
            await asyncio.sleep(wait)
        STATE["next_search"] = time.monotonic() + 1.0 / SEARCH_QPS


async def _searx(query: str, engines: str, lang: str) -> tuple[list[dict], list]:
    await _throttle()
    STATE["stats"]["searx_requests"] += 1
    params = {"q": query, "format": "json", "engines": engines, "language": lang, "safesearch": 0}
    for attempt in range(3):
        try:
            r = await STATE["searx"].get(SEARXNG_URL + "/search", params=params, timeout=30)
            r.raise_for_status()
            d = r.json()
            return d.get("results", []), d.get("unresponsive_engines", [])
        except Exception as e:  # noqa: BLE001
            err = f"{type(e).__name__}: {e}"
            await asyncio.sleep(2 * (attempt + 1))
    raise RuntimeError(f"search backend error: {err}")


@app.post("/search")
async def search(req: Request):
    b = await req.json()
    q = (b.get("query") or "").strip()
    bench, tag = b.get("bench", "?"), b.get("tag", "")
    n = max(1, min(int(b.get("n") or 10), 20))
    bl = _blocklist(b.get("blocklist"))
    st = STATE["stats"]
    st["search_calls"] += 1
    t0 = time.time()
    if not q:
        return {"error": "empty query"}
    rule = bl.check_query(q, b.get("question")) if bl else None
    if rule:
        st["search_blocked"] += 1
        _log("blocked.jsonl", {"kind": "query", "bench": bench, "tag": tag, "target": q, "rule": rule})
        _log("calls.jsonl", {"kind": "search", "bench": bench, "tag": tag, "query": q, "blocked": rule})
        return {"blocked": rule}
    lang = b.get("lang") or ("zh-CN" if CJK.search(q) else "en-US")
    group = "zh" if lang.lower().startswith("zh") else "en"
    key = _key("v1", q, lang, ENGINES[group], ENGINES[group + "_fallback"])
    data = STATE["cache"].get("search", key)
    cached = data is not None
    if not cached:
        waited = 0.0
        while True:
            try:
                res, unresp = await _searx(q, ENGINES[group], lang)
                used = ENGINES[group]
                if len(res) < 3:
                    res2, unresp2 = await _searx(q, ENGINES[group + "_fallback"], lang)
                    seen = {r.get("url") for r in res}
                    res += [r for r in res2 if r.get("url") not in seen]
                    unresp += unresp2
                    used += "+" + ENGINES[group + "_fallback"]
            except Exception as e:  # noqa: BLE001
                st["search_errors"] += 1
                _log("calls.jsonl", {"kind": "search", "bench": bench, "tag": tag, "query": q, "error": str(e)})
                return {"error": str(e)}
            rec = STATE["recent_empty"][group]
            rec.append(not res)
            # circuit breaker: most recent uncached searches came back empty -> the engines are probably
            # rate-limiting / captcha-blocking this host. Pause (this call and, in effect, the agents) and retry
            # instead of feeding "no results" to the models.
            if res or len(rec) < 10 or sum(rec) / len(rec) < 0.5 or waited >= EMPTY_WAIT_MAX:
                break
            wait = min(600.0, 60.0 * 2 ** min(5, int(waited // 60).bit_length()))
            st["degraded_waits"] += 1
            print(f"{time.strftime('%H:%M:%S')} search backend degraded ({sum(rec)}/{len(rec)} recent {group} searches "
                  f"empty; unresponsive {unresp}); pausing {wait:.0f}s", flush=True)
            await asyncio.sleep(wait)
            waited += wait
        for e in unresp:
            st["unresponsive"][f"{e[0]}:{e[1]}"] += 1
        data = {"results": [{"title": r.get("title") or "", "url": r.get("url") or "", "snippet": r.get("content") or "",
                             "engines": r.get("engines") or [], "date": r.get("publishedDate")} for r in res[:20]],
                "engines_used": used, "unresponsive": unresp}
        if data["results"]:
            STATE["cache"].put("search", key, q, data, lang)
        else:
            st["search_empty"] += 1
    else:
        st["search_cache_hits"] += 1
    out, n_filt = [], 0
    for r in data["results"]:
        rule = bl.check_result(r["url"], r["title"], r["snippet"]) if bl else None
        if rule:
            n_filt += 1
            _log("blocked.jsonl", {"kind": "result", "bench": bench, "tag": tag, "target": r["url"], "title": r["title"][:200], "rule": rule, "query": q})
            continue
        if len(out) < n:
            out.append(r)
    st["results_filtered"] += n_filt
    _log("calls.jsonl", {"kind": "search", "bench": bench, "tag": tag, "query": q, "lang": lang, "cached": cached,
                         "n_raw": len(data["results"]), "n_filtered": n_filt, "n": len(out), "ms": int(1000 * (time.time() - t0))})
    return {"results": out, "n_raw": len(data["results"]), "n_filtered": n_filt, "cached": cached,
            "engines_used": data.get("engines_used"), "unresponsive": data.get("unresponsive")}


# ------------------------------------------------------------------------------------------------ fetch
@app.post("/fetch")
async def fetch(req: Request):
    b = await req.json()
    url = (b.get("url") or "").strip()
    bench, tag = b.get("bench", "?"), b.get("tag", "")
    raw = bool(b.get("raw"))
    max_chars = int(b.get("max_chars") or 60000)
    bl = _blocklist(b.get("blocklist"))
    st = STATE["stats"]
    st["fetch_calls"] += 1
    t0 = time.time()

    def blocked(rule, where):
        st["fetch_blocked"] += 1
        _log("blocked.jsonl", {"kind": where, "bench": bench, "tag": tag, "target": url, "rule": rule})
        _log("calls.jsonl", {"kind": "fetch", "bench": bench, "tag": tag, "url": url, "blocked": rule})
        return {"blocked": rule}

    rule = bl.check_url(url) if bl else None
    if rule:
        return blocked(rule, "fetch_url")
    key = _key("v1", url, "raw" if raw else "md")
    data = STATE["cache"].get("fetch", key)
    cached = data is not None
    if not cached:
        async with STATE["fetch_sem"]:
            try:
                doc = await fetch_document(STATE["http"], STATE["robots"], url,
                                           (lambda u: bl.check_url(u)) if bl else (lambda u: None), FETCH_TIMEOUT, raw=raw)
                if raw and doc["content_type"] in ("text/html", "application/xhtml+xml"):
                    t, md = await asyncio.to_thread(html_to_markdown, doc["text"].encode("utf-8", "replace"), url)
                    doc["title"], doc["check_text"] = t, md
                data = {"ok": True, **doc}
                STATE["cache"].put("fetch", key, url, data)
            except FetchError as e:
                msg = str(e)
                if msg.startswith("BLOCKED:"):
                    return blocked(msg[8:], "fetch_redirect")
                st["fetch_errors"] += 1
                data = {"ok": False, "error": msg}
                if e.cacheable:
                    STATE["cache"].put("fetch", key, url, data)
            except Exception as e:  # noqa: BLE001
                st["fetch_errors"] += 1
                data = {"ok": False, "error": f"{type(e).__name__}: {e}"}
    else:
        st["fetch_cache_hits"] += 1
    if not data.get("ok"):
        _log("calls.jsonl", {"kind": "fetch", "bench": bench, "tag": tag, "url": url, "cached": cached, "error": data["error"][:300]})
        return {"error": data["error"]}
    if bl:
        rule = bl.check_url(data["final_url"]) or bl.check_page(data.get("title"), data.get("check_text") or data["text"])
        if rule:
            return blocked(rule, "fetch_page")
    text = data["text"]
    total = len(text)
    trunc = total > max_chars
    if trunc:
        text = text[:max_chars] + f"\n\n[... truncated: the page has {total} characters; only the first {max_chars} are shown ...]"
    _log("calls.jsonl", {"kind": "fetch", "bench": bench, "tag": tag, "url": url, "cached": cached, "chars": total,
                         "ctype": data.get("content_type"), "ms": int(1000 * (time.time() - t0))})
    return {"url": url, "final_url": data["final_url"], "title": data.get("title") or "", "content": text,
            "truncated": trunc, "total_chars": total, "cached": cached, "content_type": data.get("content_type")}


@app.get("/health")
async def health():
    return {"ok": True}


@app.get("/stats")
async def stats():
    s = dict(STATE["stats"])
    s["unresponsive"] = dict(STATE["stats"]["unresponsive"].most_common(20))
    s["uptime_s"] = int(time.time() - STATE["t0"])
    s["config"] = {"searxng": SEARXNG_URL, "qps": SEARCH_QPS, "engines": ENGINES, "user_agent": USER_AGENT,
                   "blocklist_default": os.environ.get("BLOCKLIST", "full"), "blocklist_files": STATE["blocklists"]["full"].sources}
    return s


@app.on_event("startup")
async def startup():
    (ROOT / "logs").mkdir(exist_ok=True)
    STATE.update(
        t0=time.time(),
        cache=Cache(ROOT / "cache" / "webcache.sqlite"),
        searx=httpx.AsyncClient(),
        http=httpx.AsyncClient(timeout=httpx.Timeout(FETCH_TIMEOUT, connect=10), headers={"User-Agent": USER_AGENT},
                               limits=httpx.Limits(max_connections=64), verify=True),
        throttle_lock=asyncio.Lock(), next_search=0.0, recent_empty={"en": deque(maxlen=20), "zh": deque(maxlen=20)}, fetch_sem=asyncio.Semaphore(FETCH_PARALLEL),
        blocklists={"full": Blocklist(), "bench": Blocklist([BL_DIR / "extra_blocklist.json"]), "none": None},
    )
    STATE["stats"] = Counter()
    STATE["stats"]["unresponsive"] = Counter()
    STATE["robots"] = Robots(STATE["http"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=int(os.environ.get("WEBTOOLS_PORT", 28011)))
    a = ap.parse_args()
    uvicorn.run(app, host="127.0.0.1", port=a.port, log_level="warning")
