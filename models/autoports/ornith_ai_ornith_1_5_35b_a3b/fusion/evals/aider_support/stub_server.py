"""Tiny OpenAI-compatible stub used only for offline plumbing checks of the
aider harness. It records each request body to $STUB_LOG and answers with
$STUB_REPLY_FILE (or a fixed string). It is NOT a model server."""
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

LOG = os.environ.get("STUB_LOG", "/tmp/stub_requests.jsonl")
REPLY_FILE = os.environ.get("STUB_REPLY_FILE")
MODEL = os.environ.get("STUB_MODEL", "Qwen/Qwen3-Coder-Next")


class H(BaseHTTPRequestHandler):
    def _send(self, obj):
        b = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        self._send({"object": "list", "data": [{"id": MODEL, "object": "model", "max_model_len": 65536}]})

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n) or b"{}")
        with open(LOG, "a") as f:
            f.write(json.dumps({k: v for k, v in body.items() if k != "messages"} | {"n_messages": len(body.get("messages", []))}) + "\n")
        text = open(REPLY_FILE).read() if REPLY_FILE else "No changes."
        self._send({
            "id": "stub", "object": "chat.completion", "created": int(time.time()), "model": body.get("model"),
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": text}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
        })

    def log_message(self, *a):
        pass


HTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
