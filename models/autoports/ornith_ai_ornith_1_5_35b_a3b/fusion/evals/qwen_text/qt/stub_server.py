"""OpenAI-compatible stub for offline checks (NOT a model server).
  python -m qt.stub_server <port> <bench> <mode>
mode: gold   -> replies with the gold answer in the expected format (expect ~100%)
      random -> a well-formed but random answer (expect ~chance)
      junk   -> unparseable text (expect ~0)
Each reply carries a fake `reasoning` field; only `content` must be scored."""
import json
import random
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import registry
from .common import msg_key
from .run import load_items

port, bench, mode = int(sys.argv[1]), sys.argv[2], sys.argv[3]
mod = registry.module(bench)
items, _ = load_items(bench)
if hasattr(mod, "init_stub"):
    mod.init_stub(items)
by_key = {msg_key(it["messages"]): it for it in items}
rng = random.Random(7)
MODEL = "Qwen/Qwen3.5-122B-A10B"


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _send(self, obj, code=200):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        self._send({"object": "list", "data": [{"id": MODEL, "object": "model", "max_model_len": 262144}]})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        it = by_key.get(msg_key(body.get("messages", [])))
        if it is None:
            return self._send({"error": "unknown prompt"}, 400)
        if mode == "gold":
            text = mod.gold_reply(bench, it)
        elif mode == "random":
            text = mod.random_reply(bench, it, rng)
        else:
            text = "I am not sure. Lorem ipsum dolor sit amet."
        # decoy in reasoning: a wrong-looking answer that must NOT be scored
        self._send({"id": "stub", "object": "chat.completion", "created": int(time.time()), "model": body.get("model"),
                    "choices": [{"index": 0, "finish_reason": "stop",
                                 "message": {"role": "assistant", "content": text,
                                             "reasoning": 'thinking... "answer": "A" The best answer is: A \\boxed{A}'}}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20}})

    def log_message(self, *a):
        pass


ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
