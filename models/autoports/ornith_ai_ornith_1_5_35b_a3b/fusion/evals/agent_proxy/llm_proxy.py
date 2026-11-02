"""OpenAI-compatible pass-through proxy that sits between an agent harness and the
model server (vLLM at :20000). Stdlib only, so it runs with any python3.

What it does (all optional, all logged):
  * Sampling: PROXY_SAMPLING='{"temperature":1.0,"top_p":0.95,"top_k":40}' is applied to
    every /chat/completions request. PROXY_SAMPLING_MODE=force (default) overwrites what
    the harness sent; =default only fills keys the harness left out. Harnesses differ in
    which sampling knobs they can pass at all (top_k especially), so forcing them here is
    the only way to run every harness at the card's settings.
  * PROXY_MAX_TOKENS caps max_tokens / max_completion_tokens (and sets it if absent).
  * Reasoning round-trip: vLLM 0.26 returns chain-of-thought in message.reasoning. The
    proxy mirrors it into message.reasoning_content (the key LiteLLM-based clients keep),
    remembers it by tool-call id / content hash, and re-inserts it as `reasoning` into
    later requests whose assistant messages arrive without it (OpenHands, SWE-agent and
    Terminus-2 drop it for these model ids). The Ornith / Qwen3.5 chat templates render
    past reasoning for every assistant turn after the last user message, so dropping it
    changes the prompt the model sees and defeats prefix caching. PROXY_REASONING=keep
    (default) | drop (strip it everywhere) | passthrough (do nothing).
  * Path prefix /t/<tag>/... tags requests (e.g. with an instance id) for the log.
  * One JSON line per request in PROXY_LOG: tag, message count, prompt/completion/cached
    tokens, latency, finish reason, reasoning injected/kept counts, status.

usage: llm_proxy.py <listen_host> <listen_port> <upstream_base e.g. http://127.0.0.1:20000>
"""
import hashlib
import http.client
import json
import os
import sys
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HOST, PORT, UPSTREAM = sys.argv[1], int(sys.argv[2]), sys.argv[3].rstrip("/")
UP = urllib.parse.urlparse(UPSTREAM)
UP_PREFIX = UP.path.rstrip("/")  # e.g. "" or "/v1" stripped below
if UP_PREFIX.endswith("/v1"):
    UP_PREFIX = UP_PREFIX[:-3]
SAMPLING = json.loads(os.environ.get("PROXY_SAMPLING", "{}") or "{}")
SAMPLING_MODE = os.environ.get("PROXY_SAMPLING_MODE", "force")
MAX_TOKENS = int(os.environ.get("PROXY_MAX_TOKENS", "0") or 0)
REASONING = os.environ.get("PROXY_REASONING", "keep")
LOG_PATH = os.environ.get("PROXY_LOG", "proxy_requests.jsonl")
TIMEOUT = float(os.environ.get("PROXY_UPSTREAM_TIMEOUT", "14400"))
# PROXY_MAX_CONTEXT=N emulates a server with max_model_len=N on top of a longer-context one: a
# non-streaming chat response whose usage shows prompt_tokens >= N is replaced by the 400 error
# vLLM returns for an over-long prompt (one wasted prefill per overflow, then the harness sees the
# same context-window error it would on an N-token server). 0 = off.
MAX_CONTEXT = int(os.environ.get("PROXY_MAX_CONTEXT", "0") or 0)
CACHE_MAX = 200000

_lock = threading.Lock()
_log = open(LOG_PATH, "a", buffering=1)
_cache = {}  # signature -> reasoning text
_inflight = 0
_peak = 0


def _sig_tool(tc_id):
    return "tc:" + tc_id


def _sig_content(content):
    if isinstance(content, list):
        content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
    if not content:
        return None
    return "c:" + hashlib.sha1(content.strip().encode("utf-8", "replace")).hexdigest()


def _remember(msg):
    r = msg.get("reasoning") or msg.get("reasoning_content")
    if not r:
        return
    keys = [_sig_tool(tc["id"]) for tc in (msg.get("tool_calls") or []) if tc.get("id")]
    c = _sig_content(msg.get("content"))
    if c:
        keys.append(c)
    with _lock:
        if len(_cache) > CACHE_MAX:
            _cache.clear()
        for k in keys:
            _cache[k] = r


def _lookup(msg):
    with _lock:
        for tc in msg.get("tool_calls") or []:
            r = _cache.get(_sig_tool(tc.get("id", "")))
            if r:
                return r
        c = _sig_content(msg.get("content"))
        return _cache.get(c) if c else None


def _rewrite_request(body, stats):
    if SAMPLING:
        for k, v in SAMPLING.items():
            if SAMPLING_MODE == "force" or k not in body:
                stats.setdefault("client_sampling", {})[k] = body.get(k)
                body[k] = v
    if MAX_TOKENS:
        for k in ("max_tokens", "max_completion_tokens"):
            if k in body and body[k] and body[k] > MAX_TOKENS:
                body[k] = MAX_TOKENS
        if not body.get("max_tokens") and not body.get("max_completion_tokens"):
            body["max_tokens"] = MAX_TOKENS
    kept = injected = 0
    for m in body.get("messages", []):
        if m.get("role") != "assistant":
            continue
        has = m.get("reasoning") or m.get("reasoning_content")
        if REASONING == "drop":
            m.pop("reasoning", None)
            m.pop("reasoning_content", None)
        elif REASONING == "keep":
            if has:
                kept += 1
                if not m.get("reasoning"):
                    m["reasoning"] = m.pop("reasoning_content")
                else:
                    m.pop("reasoning_content", None)
            else:
                r = _lookup(m)
                if r:
                    m["reasoning"] = r
                    injected += 1
    stats["n_msgs"] = len(body.get("messages", []))
    stats["n_assistant"] = sum(1 for m in body.get("messages", []) if m.get("role") == "assistant")
    stats["reasoning_kept"], stats["reasoning_injected"] = kept, injected
    stats["n_tools"] = len(body.get("tools") or [])
    stats["stream"] = bool(body.get("stream"))
    stats["req_bytes"] = 0
    return body


def _fix_response_message(msg):
    if REASONING == "drop":
        msg.pop("reasoning", None)
        msg.pop("reasoning_content", None)
        return
    r = msg.get("reasoning") or msg.get("reasoning_content")
    if r and REASONING == "keep":
        msg["reasoning"] = r
        msg["reasoning_content"] = r
        _remember(msg)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _split_tag(self):
        path, tag = self.path, None
        if path.startswith("/t/"):
            parts = path.split("/", 3)
            tag = urllib.parse.unquote(parts[2])
            path = "/" + (parts[3] if len(parts) > 3 else "")
        return path, tag

    def _upstream(self):
        cls = http.client.HTTPSConnection if UP.scheme == "https" else http.client.HTTPConnection
        return cls(UP.hostname, UP.port or (443 if UP.scheme == "https" else 80), timeout=TIMEOUT)

    def _headers_out(self, n):
        h = {k: v for k, v in self.headers.items()
             if k.lower() not in ("host", "content-length", "connection", "accept-encoding", "transfer-encoding")}
        h["Content-Length"] = str(n)
        return h

    def do_GET(self):
        path, _ = self._split_tag()
        if path == "/proxy/peak":
            return self._send_json(200, {"peak_inflight": _peak, "inflight": _inflight})
        conn = self._upstream()
        try:
            conn.request("GET", UP_PREFIX + path, headers=self._headers_out(0))
            r = conn.getresponse()
            data = r.read()
            self._send_raw(r.status, r.getheader("Content-Type", "application/json"), data)
        except Exception as e:  # noqa: BLE001
            self._send_json(502, {"error": f"proxy upstream error: {e}"})
        finally:
            conn.close()

    def _send_json(self, code, obj):
        self._send_raw(code, "application/json", json.dumps(obj).encode())

    def _send_raw(self, code, ctype, data):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        global _inflight, _peak
        path, tag = self._split_tag()
        raw = self.rfile.read(int(self.headers.get("Content-Length", 0) or 0))
        stats = {"ts": time.time(), "tag": tag, "path": path}
        bare = path.split("?", 1)[0].rstrip("/")  # Claude Code posts /v1/messages?beta=true
        is_chat = bare.endswith("/chat/completions")
        is_msgs = bare.endswith("/messages")
        body = None
        if "/messages" in bare and b'[1m]"' in raw:
            # Claude Code only uses a context window > 200K (auto-compact) for model ids ending in
            # "[1m]"; the harness may append it, and it is stripped here before the server sees it
            try:
                b = json.loads(raw)
                if isinstance(b.get("model"), str) and b["model"].endswith("[1m]"):
                    b["model"] = b["model"][:-4]
                    stats["model_suffix_stripped"] = True
                    raw = json.dumps(b).encode()
            except Exception as e:  # noqa: BLE001
                stats["rewrite_error"] = repr(e)
        if is_msgs and (SAMPLING or MAX_TOKENS):
            # Anthropic Messages API (Claude Code): same sampling / max_tokens policy
            try:
                b = json.loads(raw)
                for k, v in SAMPLING.items():
                    if SAMPLING_MODE == "force" or k not in b:
                        stats.setdefault("client_sampling", {})[k] = b.get(k)
                        b[k] = v
                if MAX_TOKENS and b.get("max_tokens", 0) > MAX_TOKENS:
                    b["max_tokens"] = MAX_TOKENS
                stats["n_msgs"] = len(b.get("messages", []))
                raw = json.dumps(b).encode()
            except Exception as e:  # noqa: BLE001
                stats["rewrite_error"] = repr(e)
        if is_chat:
            try:
                body = _rewrite_request(json.loads(raw), stats)
                raw = json.dumps(body).encode()
            except Exception as e:  # noqa: BLE001
                stats["rewrite_error"] = repr(e)
        stats["req_bytes"] = len(raw)
        with _lock:
            _inflight += 1
            _peak = max(_peak, _inflight)
        t0 = time.time()
        conn = self._upstream()
        try:
            conn.request("POST", UP_PREFIX + path, body=raw, headers=self._headers_out(len(raw)))
            r = conn.getresponse()
            stats["status"] = r.status
            ctype = r.getheader("Content-Type", "application/json")
            if "text/event-stream" in ctype:
                self._relay_stream(r, ctype, stats, is_chat)
            else:
                data = r.read()
                if is_chat and r.status == 200:
                    try:
                        obj = json.loads(data)
                        for ch in obj.get("choices", []):
                            _fix_response_message(ch.get("message") or {})
                            stats["finish_reason"] = ch.get("finish_reason")
                            msg = ch.get("message") or {}
                            stats["n_tool_calls"] = len(msg.get("tool_calls") or [])
                            stats["reasoning_chars"] = len(msg.get("reasoning") or "")
                            stats["content_chars"] = len(msg.get("content") or "")
                        u = obj.get("usage") or {}
                        stats["prompt_tokens"] = u.get("prompt_tokens")
                        stats["completion_tokens"] = u.get("completion_tokens")
                        stats["cached_tokens"] = (u.get("prompt_tokens_details") or {}).get("cached_tokens")
                        data = json.dumps(obj).encode()
                        if MAX_CONTEXT and (u.get("prompt_tokens") or 0) >= MAX_CONTEXT:
                            stats["context_overflow"] = True
                            msg = (f"This model's maximum context length is {MAX_CONTEXT} tokens. However, your "
                                   f"request has {u.get('prompt_tokens')} input tokens. Please reduce the length "
                                   "of the input messages. (llm_proxy PROXY_MAX_CONTEXT)")
                            stats["status"] = 400
                            return self._send_json(400, {"object": "error", "message": msg, "type": "BadRequestError",
                                                         "param": None, "code": 400,
                                                         "error": {"message": msg, "type": "BadRequestError", "code": 400}})
                    except Exception as e:  # noqa: BLE001
                        stats["parse_error"] = repr(e)
                elif not is_chat and r.status == 200:
                    try:
                        u = json.loads(data).get("usage") or {}
                        stats["prompt_tokens"] = u.get("input_tokens", u.get("prompt_tokens"))
                        stats["completion_tokens"] = u.get("output_tokens", u.get("completion_tokens"))
                    except Exception:  # noqa: BLE001
                        pass
                else:
                    stats["error_body"] = data[:500].decode("utf-8", "replace")
                self._send_raw(r.status, ctype, data)
        except Exception as e:  # noqa: BLE001
            stats["status"] = 502
            stats["error_body"] = repr(e)
            try:
                self._send_json(502, {"error": {"message": f"proxy upstream error: {e}"}})
            except Exception:  # noqa: BLE001
                pass
        finally:
            conn.close()
            with _lock:
                _inflight -= 1
            stats["latency_s"] = round(time.time() - t0, 3)
            _log.write(json.dumps(stats) + "\n")

    def _relay_stream(self, r, ctype, stats, is_chat):
        """Pass SSE through line by line; for chat streams, reassemble reasoning and
        tool-call ids so later requests can get their reasoning re-inserted, and add a
        reasoning_content mirror to each delta that carries reasoning."""
        self.send_response(r.status)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        reasoning, content, tc_ids = [], [], []
        usage = {}

        def emit(b):
            self.wfile.write(b"%x\r\n%s\r\n" % (len(b), b))
            self.wfile.flush()

        while True:
            line = r.readline()
            if not line:
                break
            if is_chat and line.startswith(b"data: ") and not line.startswith(b"data: [DONE]"):
                try:
                    obj = json.loads(line[6:])
                    for ch in obj.get("choices", []):
                        d = ch.get("delta") or {}
                        rr = d.get("reasoning") or d.get("reasoning_content")
                        if rr and REASONING == "keep":
                            d["reasoning"] = rr
                            d["reasoning_content"] = rr
                            reasoning.append(rr)
                        elif REASONING == "drop":
                            d.pop("reasoning", None)
                            d.pop("reasoning_content", None)
                        if d.get("content"):
                            content.append(d["content"])
                        for tc in d.get("tool_calls") or []:
                            if tc.get("id"):
                                tc_ids.append(tc["id"])
                        if ch.get("finish_reason"):
                            stats["finish_reason"] = ch["finish_reason"]
                    if obj.get("usage"):
                        usage = obj["usage"]
                    line = b"data: " + json.dumps(obj).encode() + b"\n"
                except Exception:  # noqa: BLE001
                    pass
            elif not is_chat and line.startswith(b"data: "):
                try:
                    obj = json.loads(line[6:])
                    u = obj.get("usage") or (obj.get("message") or {}).get("usage")
                    if u:
                        usage.update(u)
                except Exception:  # noqa: BLE001
                    pass
            emit(line)
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()
        if reasoning:
            _remember({"reasoning": "".join(reasoning), "content": "".join(content),
                       "tool_calls": [{"id": i} for i in tc_ids]})
        stats["prompt_tokens"] = usage.get("prompt_tokens", usage.get("input_tokens"))
        stats["completion_tokens"] = usage.get("completion_tokens", usage.get("output_tokens"))
        stats["reasoning_chars"] = sum(len(x) for x in reasoning)
        stats["n_tool_calls"] = len(tc_ids)


if __name__ == "__main__":
    srv = ThreadingHTTPServer((HOST, PORT), Handler)
    srv.daemon_threads = True
    print(f"llm_proxy listening on {HOST}:{PORT} -> {UPSTREAM} sampling={SAMPLING} mode={SAMPLING_MODE} "
          f"max_tokens={MAX_TOKENS or 'client'} reasoning={REASONING} max_context={MAX_CONTEXT or 'server'} log={LOG_PATH}", flush=True)
    srv.serve_forever()
