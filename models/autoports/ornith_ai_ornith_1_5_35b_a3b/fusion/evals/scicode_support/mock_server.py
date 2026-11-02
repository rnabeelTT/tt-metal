"""Stub OpenAI-compatible server for offline harness self-tests (no model).

Answers each SciCode step with the dev-set ground-truth code in `content`, puts a
decoy failing code block in the separate `reasoning` field (so a harness that
parses reasoning would fail every step), records every request body, and tracks
peak in-flight requests. Problem 1 gets an empty, length-truncated answer.
usage: mock_server.py <port> <requests.jsonl> <split>
"""
import json, sys, threading, time
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from datasets import load_dataset

port, log_path, split = int(sys.argv[1]), sys.argv[2], sys.argv[3]
steps = []
for p in load_dataset("SciCode1/SciCode", split=split):
    for s in p["sub_steps"]:
        steps.append((p["problem_id"], s["function_header"] + "\n\n" + s["return_line"], s["ground_truth_code"]))
lock = threading.Lock(); inflight = 0; peak = 0
logf = open(log_path, "a")

class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _send(self, obj):
        b = json.dumps(obj).encode(); self.send_response(200)
        self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(b)))
        self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        if self.path.endswith("/models"):
            self._send({"object": "list", "data": [{"id": "mock/model", "object": "model", "max_model_len": 65536}]})
        elif self.path.endswith("/peak"):
            self._send({"peak": peak})
        else:
            self.send_error(404)
    def do_POST(self):
        global inflight, peak
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        with lock:
            inflight += 1; peak = max(peak, inflight)
        try:
            prompt = body["messages"][-1]["content"]
            if isinstance(prompt, list):
                prompt = "".join(c.get("text", "") for c in prompt)
            hits = [(pid, gt) for pid, key, gt in steps if key in prompt]
            pid, gt = hits[-1] if hits else ("?", None)
            with lock:
                logf.write(json.dumps({k: v for k, v in body.items() if k != "messages"} | {"n_messages": len(body["messages"]), "roles": [m["role"] for m in body["messages"]], "pid": pid}) + "\n"); logf.flush()
            time.sleep(float(__import__("os").environ.get("MOCK_DELAY", "0.5")))
            if pid == "1":
                content, finish = "", "length"
            else:
                content, finish = f"Here is the code.\n```python\n{gt}\n```\n", "stop"
            msg = {"role": "assistant", "content": content,
                   "reasoning": "Thinking...\n```python\nraise SystemExit('reasoning was parsed')\n```\n"}
            self._send({"id": "x", "object": "chat.completion", "created": int(time.time()), "model": body["model"],
                        "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
                        "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}})
        finally:
            with lock:
                inflight -= 1

ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
