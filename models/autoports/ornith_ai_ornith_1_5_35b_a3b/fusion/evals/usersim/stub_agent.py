#!/usr/bin/env -S /usr/bin/python3
"""Scripted agent-under-test stub for relay plumbing tests (never for scores). Stdlib only.

  /usr/bin/python3 ~/evals/usersim/stub_agent.py --port 28403

Plays the AGENT in VITA (or any) conversations, by agent-turn index (n = number of the agent's earlier turns):
  0: asks the user for details;  1: one call to the first offered tool whose name contains "search" (required args
  filled from the user's last message), else a text reply;  after a tool result: claims the request was handled and
  asks if anything else is needed;  later: "Done. Is there anything else?";  from its 5th turn on it ends the
  dialogue with ###STOP### (as the official VITA agent prompt instructs the agent to).
"""
import argparse
import json
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, required=True)
ap.add_argument("--models", default="Qwen/Qwen3.5-122B-A10B,ornith-ai/Ornith-1.5-122B,ornith-ai/Ornith-1.5-35B-A3B")
args = ap.parse_args()
STATS = {"requests": 0, "tool_calls": 0}


def fill(schema, text):
    t = schema.get("type")
    if t == "array":
        return [text[:20]]
    if t in ("integer", "number"):
        return 1
    if t == "boolean":
        return False
    if "enum" in schema:
        return schema["enum"][0]
    return text[:20]


def turn(body):
    msgs = body["messages"]
    n_agent = sum(1 for m in msgs if m["role"] == "assistant")
    last = msgs[-1]
    zh = any("一" <= ch <= "鿿" for ch in json.dumps(msgs[:1], ensure_ascii=False)[:3000])
    if last["role"] == "tool":
        return ("好的，我已经根据您的要求处理好了。还有其他需要吗？" if zh else
                "I have handled that according to your request. Is there anything else?"), None
    if n_agent == 0:
        return ("好的，请问您具体需要什么？有什么要求吗？" if zh else "Sure. What exactly do you need, and any constraints?"), None
    if n_agent == 1:
        user_txt = next((m.get("content") or "" for m in reversed(msgs) if m["role"] == "user"), "")
        for t in body.get("tools") or []:
            f = t["function"]
            if "search" in f["name"]:
                p = f.get("parameters") or {}
                a = {k: fill(p["properties"][k], user_txt) for k in p.get("required") or []}
                return None, (f["name"], a)
    if n_agent >= 4:   # the official VITA agent prompt ends the dialogue with ###STOP### (the user prompt has no stop token)
        return ("好的，祝您用餐愉快，再见！###STOP###" if zh else "Great, goodbye! ###STOP###"), None
    return ("好的，已完成。还有其他需要吗？" if zh else "Done. Is there anything else?"), None


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _send(self, obj):
        b = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path.rstrip("/").endswith("/models"):
            return self._send({"object": "list", "data": [{"id": m, "object": "model", "max_model_len": 262144}
                                                          for m in args.models.split(",")]})
        self._send(STATS)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        STATS["requests"] += 1
        content, tc = turn(body)
        msg = {"role": "assistant", "content": content, "reasoning": "stub reasoning"}
        fr = "stop"
        if tc:
            STATS["tool_calls"] += 1
            msg["tool_calls"] = [{"id": "call_" + uuid.uuid4().hex[:12], "type": "function",
                                  "function": {"name": tc[0], "arguments": json.dumps(tc[1], ensure_ascii=False)}}]
            fr = "tool_calls"
        self._send({"id": "chatcmpl-" + uuid.uuid4().hex, "object": "chat.completion", "created": int(time.time()),
                    "model": body.get("model"), "choices": [{"index": 0, "message": msg, "finish_reason": fr}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20}})


class S(ThreadingHTTPServer):
    daemon_threads = True


if __name__ == "__main__":
    S(("127.0.0.1", args.port), H).serve_forever()
