#!/usr/bin/env -S /usr/bin/python3
"""OpenAI-compatible user-simulator endpoint whose answers come from Claude worker agents via a file spool.

  /usr/bin/python3 ~/evals/usersim/shim.py [--host 127.0.0.1] [--port 28400] [--timeout 1800] [--model claude-usersim]

POST /v1/chat/completions (non-streaming): writes spool/pending/<key>.json (key = sha256 of the whole simulator input,
so a retried request re-joins the same job and an answered one is served from done/ at once), blocks until
spool/done/<key>.json exists, returns it as a chat.completion (content, or tool_calls with JSON-string arguments).
On timeout: HTTP 500 (both litellm/tau2 and vita's requests loop retry 500; the job stays pending for the retry).
GET /v1/models, GET /stats, GET /health.  Background reaper: expired leases -> pending; jobs nobody has waited for
during --grace seconds -> abandoned/ (revived if the harness retries). Stdlib only. Workers: worker.py, WORKER.md.
"""
import argparse
import json
import os
import statistics
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import usersim_common as C  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--host", default="127.0.0.1")
ap.add_argument("--port", type=int, default=int(os.environ.get("USERSIM_PORT", 28400)))
ap.add_argument("--timeout", type=float, default=float(os.environ.get("USERSIM_TIMEOUT", 1800)))
ap.add_argument("--grace", type=float, default=600, help="abandon a job after this long without any waiter")
ap.add_argument("--poll", type=float, default=0.02, help="done/ poll interval, s")
ap.add_argument("--model", default=os.environ.get("USER_SIM_MODEL", "claude-usersim"))
args = ap.parse_args()

C.ensure_dirs()
LOCK = threading.Lock()
WAITERS = {}        # key -> number of HTTP requests currently blocked on it
LAST_WAIT = {}      # key -> time the last waiter left (or shim start, for jobs found on disk)
STATS = {"requests": 0, "cache_hits": 0, "timeouts": 0, "bad_requests": 0, "invalid_done": 0, "lat": [], "ovh": []}
START = time.time()
for st in ("pending", "claimed"):
    for p in C.d(st).glob("*.json"):
        LAST_WAIT[p.stem] = START


def completion(model, done):
    msg = {"role": "assistant", "content": done.get("content")}
    fr = "stop"
    if done.get("tool_calls"):
        msg["content"] = None
        msg["tool_calls"] = [{"id": tc["id"], "type": "function",
                              "function": {"name": tc["name"], "arguments": tc["arguments"]}} for tc in done["tool_calls"]]
        fr = "tool_calls"
    out_tok = len(json.dumps(msg, ensure_ascii=False)) // 4
    return {"id": "chatcmpl-" + uuid.uuid4().hex, "object": "chat.completion", "created": int(time.time()),
            "model": model, "choices": [{"index": 0, "message": msg, "finish_reason": fr}],
            "usage": {"prompt_tokens": 0, "completion_tokens": out_tok, "total_tokens": out_tok}}


def done_ok(done):
    if done.get("tool_calls"):
        try:
            return all(isinstance(json.loads(tc["arguments"]), dict) and tc.get("name") for tc in done["tool_calls"])
        except Exception:  # noqa: BLE001
            return False
    return isinstance(done.get("content"), str) and bool(done["content"].strip())


def enqueue(key, body, t_recv):
    """Make sure a job for key exists (pending/claimed) unless it is already done. Caller holds LOCK."""
    if (C.d("done") / f"{key}.json").exists():
        return "done"
    st, p, _ = C.find_request(key)
    if st in ("pending", "claimed"):
        return st
    if st in ("abandoned", "answered"):     # answered without done = done removed by hand: ask again
        os.utime(p)
        os.replace(p, C.d("pending") / f"{key}.json")
        return "revived"
    msgs = body.get("messages") or []
    rec = {"key": key, "conv": C.conv_id(body), "n_msgs": len(msgs), "bench": C.detect_bench(body),
           "created": t_recv, "model": body.get("model"), "seed": body.get("seed"),
           "temperature": body.get("temperature"), "messages": msgs, "tools": body.get("tools") or None}
    C.atomic_write(C.d("pending") / f"{key}.json", rec)
    return "new"


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _send(self, code, obj):
        b = json.dumps(obj, ensure_ascii=False).encode()
        try:
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_GET(self):
        path = self.path.rstrip("/")
        if path.endswith("/models"):
            return self._send(200, {"object": "list", "data": [{"id": args.model, "object": "model", "owned_by": "usersim"}]})
        if path.endswith("/stats") or path.endswith("/health"):
            return self._send(200, stats())
        self._send(404, {"error": "not found"})

    def do_POST(self):
        t_recv = time.time()
        if not self.path.rstrip("/").endswith("/chat/completions"):
            return self._send(404, {"error": "not found"})
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            assert isinstance(body.get("messages"), list) and body["messages"]
        except Exception as e:  # noqa: BLE001
            STATS["bad_requests"] += 1
            return self._send(400, {"error": {"message": f"bad request: {e}", "type": "invalid_request_error"}})
        if body.get("stream"):
            return self._send(400, {"error": {"message": "streaming not supported", "type": "invalid_request_error"}})
        key = C.request_key(body)
        with LOCK:
            STATS["requests"] += 1
            WAITERS[key] = WAITERS.get(key, 0) + 1
            how = enqueue(key, body, t_recv)
        t_enq = time.time()
        donep = C.d("done") / f"{key}.json"
        deadline = t_recv + args.timeout
        try:
            while True:
                if donep.exists():
                    try:
                        done = C.read_json(donep)
                    except (OSError, json.JSONDecodeError):
                        time.sleep(args.poll)
                        continue
                    if not done_ok(done):
                        with LOCK:
                            STATS["invalid_done"] += 1
                            os.replace(donep, C.d("log") / f"invalid_{key}_{int(time.time())}.json")
                            enqueue(key, body, t_recv)
                        continue
                    break
                if time.time() > deadline:
                    STATS["timeouts"] += 1
                    return self._send(500, {"error": {"message": f"usersim: no worker answer within {args.timeout:.0f}s "
                                                                 f"(job {key[:12]} stays queued; retry re-joins it)",
                                                      "type": "server_error"}})
                time.sleep(args.poll)
            t_done = os.stat(donep).st_mtime
            resp = completion(body.get("model") or args.model, done)
            self._send(200, resp)
            t_ret = time.time()
            if how == "done":
                STATS["cache_hits"] += 1
            else:
                STATS["lat"].append(t_ret - t_recv)
                STATS["ovh"].append((t_enq - t_recv) + max(0.0, t_ret - t_done))
            C.log_line("served.jsonl", {"key": key, "conv": C.conv_id(body), "n_msgs": len(body["messages"]),
                                        "how": how, "t_recv": t_recv, "enqueue_s": round(t_enq - t_recv, 4),
                                        "wait_s": round(t_done - t_enq, 3), "return_s": round(t_ret - t_done, 4),
                                        "total_s": round(t_ret - t_recv, 3),
                                        "tool_call": bool(done.get("tool_calls")), "worker": done.get("worker")})
        finally:
            with LOCK:
                WAITERS[key] -= 1
                if WAITERS[key] <= 0:
                    WAITERS.pop(key, None)
                    LAST_WAIT[key] = time.time()


def stats():
    cnt = {st: sum(1 for _ in C.d(st).glob("*.json")) for st in ("pending", "claimed", "done", "abandoned")}
    lat, ovh = STATS["lat"][-2000:], STATS["ovh"][-2000:]
    oldest = min((p.stat().st_mtime for p in C.d("pending").glob("*.json")), default=None)
    return {"model": args.model, "uptime_s": round(time.time() - START), "spool": str(C.SPOOL), **cnt,
            "in_flight_http": sum(WAITERS.values()), "oldest_pending_s": round(time.time() - oldest, 1) if oldest else 0,
            "requests": STATS["requests"], "answered_http": len(STATS["lat"]), "cache_hits": STATS["cache_hits"],
            "timeouts": STATS["timeouts"], "bad_requests": STATS["bad_requests"], "invalid_done": STATS["invalid_done"],
            "latency_mean_s": round(statistics.mean(lat), 2) if lat else None,
            "latency_p50_s": round(statistics.median(lat), 2) if lat else None,
            "relay_overhead_mean_ms": round(1000 * statistics.mean(ovh), 2) if ovh else None}


def reaper():
    while True:
        time.sleep(5)
        now = time.time()
        try:
            with C.state_lock():
                C.requeue_expired_leases()
            with LOCK:
                for st in ("pending", "claimed"):
                    for p in C.d(st).glob("*.json"):
                        k = p.stem
                        if k in WAITERS:
                            continue
                        if now - LAST_WAIT.setdefault(k, now) > args.grace:
                            try:
                                os.replace(p, C.d("abandoned") / p.name)
                            except FileNotFoundError:
                                pass
                            (C.d("claimed") / f"{k}.lease").unlink(missing_ok=True)
        except Exception as e:  # noqa: BLE001
            print(f"reaper: {e!r}", file=sys.stderr, flush=True)


class Server(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 256


if __name__ == "__main__":
    threading.Thread(target=reaper, daemon=True).start()
    srv = Server((args.host, args.port), H)
    print(f"usersim shim on http://{args.host}:{args.port}/v1 model={args.model} spool={C.SPOOL} "
          f"timeout={args.timeout}s", flush=True)
    srv.serve_forever()
