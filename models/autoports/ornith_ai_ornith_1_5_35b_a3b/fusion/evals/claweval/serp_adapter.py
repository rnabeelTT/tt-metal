"""Novada-SERP-compatible search endpoint backed by the local webtools service (SearXNG), stdlib only.

Claw-Eval's real-web tasks (mock_services/web_real*/search_serp.py) call
  GET $SERP_API_URL?engine=google&api_key=..&q=..&num=..&hl=..&gl=..&start=..
and read data.organic_results[].{title,url,description,date}. The official runs use a paid Google SERP API
(novada.com, SERP_DEV_KEY). Point SERP_API_URL at this adapter instead; it forwards the query to
~/evals/websearch webtools (POST /search -> SearXNG at 127.0.0.1:28010, caching, rate limiting, blocklist)
and answers in the Novada shape. No API key is needed or sent anywhere.

usage: serp_adapter.py <port> [webtools_url=http://127.0.0.1:28011]
env:   SERP_BLOCKLIST (webtools blocklist profile; unset = webtools default), SERP_LOG (jsonl call log)
"""
import json
import os
import sys
import threading
import time
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = int(sys.argv[1])
WEBTOOLS = (sys.argv[2] if len(sys.argv) > 2 else "http://127.0.0.1:28011").rstrip("/")
BLOCKLIST = os.environ.get("SERP_BLOCKLIST") or None
LOG = open(os.environ.get("SERP_LOG", os.devnull), "a", buffering=1)
LOCK = threading.Lock()


def search(q, num, start, hl):
    n = max(1, min(20, (start - 1) + num))
    body = {"query": q, "bench": "claweval", "tag": "serp_adapter", "n": n,
            "lang": "zh-CN" if hl == "zh" else "en-US"}
    if BLOCKLIST:
        body["blocklist"] = BLOCKLIST
    req = urllib.request.Request(WEBTOOLS + "/search", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        d = json.loads(r.read())
    res = d.get("results") or []
    res = res[start - 1:start - 1 + num]
    return d, [{"title": x.get("title", ""), "url": x.get("url", ""), "description": x.get("snippet", ""),
                "date": x.get("date") or ""} for x in res]


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _send(self, code, obj):
        b = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        if u.path.rstrip("/") == "/health":
            return self._send(200, {"ok": True})
        qs = urllib.parse.parse_qs(u.query)
        q = (qs.get("q") or [""])[0].strip()
        num = int((qs.get("num") or ["10"])[0] or 10)
        start = max(1, int((qs.get("start") or ["1"])[0] or 1))
        hl = (qs.get("hl") or ["en"])[0]
        t0 = time.time()
        rec = {"ts": t0, "q": q, "num": num, "start": start, "hl": hl}
        if not q:
            return self._send(400, {"error": "empty q"})
        try:
            d, items = search(q, num, start, hl)
            rec.update(n=len(items), blocked=d.get("blocked"), error=d.get("error"), cached=d.get("cached"))
            code = 200 if not d.get("error") else 502
            self._send(code, {"code": code, "data": {"organic_results": items}})
        except Exception as e:  # noqa: BLE001
            rec["error"] = repr(e)
            self._send(502, {"code": 502, "error": repr(e), "data": {"organic_results": []}})
        rec["ms"] = int(1000 * (time.time() - t0))
        with LOCK:
            LOG.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), H)
    srv.daemon_threads = True
    print(f"serp_adapter on 127.0.0.1:{PORT} -> {WEBTOOLS}/search blocklist={BLOCKLIST or 'webtools default'}",
          flush=True)
    srv.serve_forever()
