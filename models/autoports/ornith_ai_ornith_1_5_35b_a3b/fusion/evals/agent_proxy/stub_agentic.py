"""Scripted stub "model" (no inference) for offline plumbing checks of the Harbor/Pier agent
harnesses. Speaks both OpenAI chat completions and the Anthropic Messages API (streaming and
not), plays a "reference-solution agent" and logs what each request carried.

  Anthropic /v1/messages (Claude Code): finds the task whose instruction.md appears in the first
    user message (task dirs listed in $STUB_TASK_DIRS, ':'-separated; each holds <task>/instruction.md)
    and, on the first agent turn, calls the `Bash` tool with
      - <task>/solution/solution.patch present (DeepSWE): git apply + commit on a new branch in /app
      - else <task>/solution/solve.sh (Terminal-Bench): run it
    then answers with text and stop_reason=end_turn. Every assistant turn carries a thinking block
    (with signature), so the log shows whether the client sends thinking back in later requests.
    /v1/messages/count_tokens is answered.
  OpenAI chat (mini-swe-agent, SWE Atlas QnA): matches the task by its <question> text; turn 0
    writes <task>/solution/answer.txt to /logs/agent/answer.txt with the `bash` tool, turn 1 runs
    `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`. Terminus-2 (json parser) prompts get solve.sh as
    keystrokes, then task_complete (same as stub_llm.py).
  STUB_MODE=empty: never solve (submit/stop immediately).

Log ($STUB_LOG, one JSON line per request): api, path, model, temperature/top_p/top_k/max_tokens,
stream, n_msgs, n_tools, matched task, step, assistant turns with thinking/reasoning sent back.
usage: stub_agentic.py <host> <port>
"""
import base64
import glob
import json
import os
import re
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HOST, PORT = sys.argv[1], int(sys.argv[2])
LOG = open(os.environ.get("STUB_LOG", "stub_agentic.jsonl"), "a", buffering=1)
MODE = os.environ.get("STUB_MODE", "gold")
MODEL = os.environ.get("STUB_MODEL", "stub/model")
DELAY = float(os.environ.get("STUB_DELAY", "0.2"))
PRECMD = os.environ.get("STUB_PRECMD", "")  # shell prefix for the first agent command (e.g. a network probe)
lock = threading.Lock()

TASKS = []  # (key text, task dir)
for root in [d for d in os.environ.get("STUB_TASK_DIRS", "").split(":") if d]:
    for ins in glob.glob(os.path.join(root, "*", "instruction.md")):
        txt = open(ins).read()
        m = re.search(r"<question>\s*(.*?)\s*</question>", txt, re.S)
        key = re.sub(r"<!--.*?-->", "", (m.group(1) if m else txt), flags=re.S).strip()  # canary comments
        TASKS.append((re.sub(r"\s+", " ", key)[:160], os.path.dirname(ins)))


def text_of(c):
    if isinstance(c, list):
        out = []
        for p in c:
            if isinstance(p, dict):
                if p.get("type") == "tool_result":
                    out.append(text_of(p.get("content")))
                else:
                    out.append(p.get("text", "") or "")
        return "\n".join(out)
    return c or ""


def find_task(prompt):
    flat = re.sub(r"\s+", " ", prompt)
    best = None
    for key, d in TASKS:
        k = key[:120]
        if k and k in flat and (best is None or len(k) > len(best[0])):
            best = (k, d)
    return best[1] if best else None


def solve_command(task_dir):
    return PRECMD + _solve_command(task_dir)


def _solve_command(task_dir):
    patch = os.path.join(task_dir, "solution", "solution.patch")
    if os.path.exists(patch):
        b64 = base64.b64encode(open(patch, "rb").read()).decode()
        return ("cd /app && echo " + b64 + " | base64 -d > /tmp/stub_solution.patch && "
                "git apply --whitespace=nowarn /tmp/stub_solution.patch && git checkout -b stub/solution && "
                "git add -A && git -c user.name=stub -c user.email=stub@local commit -q --no-verify -m 'stub solution' "
                "&& git log --oneline -2")
    sol = os.path.join(task_dir, "solution", "solve.sh")
    b64 = base64.b64encode(open(sol, "rb").read()).decode()
    return f"echo {b64} | base64 -d > /tmp/stub_solve.sh && bash /tmp/stub_solve.sh; echo STUB_DONE"


def log(rec):
    with lock:
        LOG.write(json.dumps(rec) + "\n")


# ---------------------------------------------------------------- Anthropic Messages
def anthropic_turn(body):
    msgs = body.get("messages", [])
    tools = [t.get("name") for t in body.get("tools") or []]
    first_user = next((text_of(m.get("content")) for m in msgs if m.get("role") == "user"), "")
    sysp = text_of(body.get("system")) if isinstance(body.get("system"), list) else (body.get("system") or "")
    task = find_task(first_user + "\n" + "\n".join(text_of(m.get("content")) for m in msgs if m.get("role") == "user"))
    n_tool_turns = sum(1 for m in msgs if m.get("role") == "assistant" and isinstance(m.get("content"), list)
                       and any(b.get("type") == "tool_use" for b in m["content"]))
    blocks = [{"type": "thinking", "thinking": f"stub thinking, step {n_tool_turns}", "signature": uuid.uuid4().hex}]
    stop = "end_turn"
    if "Bash" in tools and task and n_tool_turns == 0 and MODE == "gold":
        blocks.append({"type": "text", "text": f"Solving {os.path.basename(task)} with the reference solution."})
        blocks.append({"type": "tool_use", "id": "toolu_" + uuid.uuid4().hex[:20], "name": "Bash",
                       "input": {"command": solve_command(task), "description": "apply reference solution",
                                 "timeout": 600000}})
        stop = "tool_use"
    elif "Bash" in tools:
        blocks.append({"type": "text", "text": "Done. The work is committed."})
    else:  # side requests (titles, summaries, quota checks)
        blocks.append({"type": "text", "text": "stub"})
    rec = {"api": "anthropic", "model": body.get("model"), "temperature": body.get("temperature"),
           "top_p": body.get("top_p"), "top_k": body.get("top_k"), "max_tokens": body.get("max_tokens"),
           "stream": body.get("stream"), "n_msgs": len(msgs), "n_tools": len(tools),
           "has_bash": "Bash" in tools, "system_chars": len(sysp), "thinking_param": body.get("thinking"),
           "matched": os.path.basename(task) if task else None, "step": n_tool_turns, "stop": stop,
           "assistant_turns": sum(1 for m in msgs if m.get("role") == "assistant"),
           "assistant_with_thinking": sum(1 for m in msgs if m.get("role") == "assistant" and isinstance(m.get("content"), list)
                                          and any(b.get("type") == "thinking" for b in m["content"])),
           "last_tool_result": next((text_of([b])[:300] for m in reversed(msgs) if m.get("role") == "user"
                                     and isinstance(m.get("content"), list)
                                     for b in m["content"] if b.get("type") == "tool_result"), None),
           "user_turns_with_text_and_tool_result": sum(
               1 for m in msgs if m.get("role") == "user" and isinstance(m.get("content"), list)
               and any(b.get("type") == "tool_result" for b in m["content"])
               and any(b.get("type") == "text" for b in m["content"]))}
    return blocks, stop, rec


def sse(wfile, event, obj):
    b = f"event: {event}\ndata: {json.dumps(obj)}\n\n".encode()
    wfile.write(b"%x\r\n%s\r\n" % (len(b), b))


# ---------------------------------------------------------------- OpenAI chat
def openai_turn(body):
    msgs = body.get("messages", [])
    prompt = "\n".join(text_of(m.get("content")) for m in msgs if m.get("role") in ("user", "system"))
    assists = [m for m in msgs if m.get("role") == "assistant"]
    tools = [t.get("function", {}).get("name") for t in body.get("tools") or []]
    task = find_task(prompt)
    calls, content = [], ""
    if "task_complete" in prompt and "keystrokes" in prompt:  # Terminus-2 json
        if task and not assists and MODE == "gold":
            content = json.dumps({"analysis": "stub", "plan": "run reference solution",
                                  "commands": [{"keystrokes": solve_command(task) + "\n", "duration": 60}],
                                  "task_complete": False})
        else:
            content = json.dumps({"analysis": "done", "plan": "", "commands": [], "task_complete": True})
    elif "bash" in tools:  # mini-swe-agent (tool calling)
        ans = os.path.join(task, "solution", "answer.txt") if task else None
        if not assists and ans and os.path.exists(ans) and MODE == "gold":
            a = open(ans).read()
            if "<<FINAL_ANSWER>>" not in a:
                a = "<<FINAL_ANSWER>>\n" + a.strip() + "\n<<FINAL_ANSWER>>\n"
            b64 = base64.b64encode(a.encode()).decode()
            cmd = f"mkdir -p /logs/agent && echo {b64} | base64 -d > /logs/agent/answer.txt && wc -c /logs/agent/answer.txt"
            content = "Writing the answer."
        else:
            cmd = "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
            content = "Submitting."
        calls = [{"id": "call_" + uuid.uuid4().hex[:24], "type": "function",
                  "function": {"name": "bash", "arguments": json.dumps({"command": cmd})}}]
    else:
        content = "stub"
    rec = {"api": "openai", "model": body.get("model"), "temperature": body.get("temperature"),
           "top_p": body.get("top_p"), "top_k": body.get("top_k"),
           "max_tokens": body.get("max_tokens") or body.get("max_completion_tokens"), "stream": body.get("stream"),
           "n_msgs": len(msgs), "n_tools": len(tools), "tools": tools[:8],
           "matched": os.path.basename(task) if task else None, "step": len(assists),
           "assistant_with_reasoning": sum(1 for m in assists if m.get("reasoning") or m.get("reasoning_content")),
           "prompt_chars": len(prompt)}
    return content, calls, rec


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _send(self, obj, code=200):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        self._send({"object": "list", "data": [{"id": MODEL, "object": "model", "max_model_len": 262144}]})

    def do_HEAD(self):
        self.send_response(200)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_POST(self):
        path = self.path.split("?")[0].rstrip("/")
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0) or 0)) or b"{}")
        if path.endswith("/messages/count_tokens"):
            log({"api": "anthropic", "path": path, "count_tokens": True})
            return self._send({"input_tokens": len(json.dumps(body)) // 4})
        if path.endswith("/messages"):
            return self._anthropic(body, path)
        if path.endswith("/chat/completions"):
            return self._openai(body, path)
        self._send({"error": {"message": f"stub: unknown path {path}"}}, 404)

    def _anthropic(self, body, path):
        blocks, stop, rec = anthropic_turn(body)
        rec["path"] = path
        log(rec)
        time.sleep(DELAY)
        usage = {"input_tokens": len(json.dumps(body.get("messages", []))) // 4, "output_tokens": 40}
        mid = "msg_" + uuid.uuid4().hex[:20]
        if not body.get("stream"):
            return self._send({"id": mid, "type": "message", "role": "assistant", "model": body.get("model"),
                               "content": blocks, "stop_reason": stop, "stop_sequence": None, "usage": usage})
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        w = self.wfile
        sse(w, "message_start", {"type": "message_start", "message": {
            "id": mid, "type": "message", "role": "assistant", "model": body.get("model"), "content": [],
            "stop_reason": None, "stop_sequence": None, "usage": {"input_tokens": usage["input_tokens"], "output_tokens": 1}}})
        for i, b in enumerate(blocks):
            if b["type"] == "thinking":
                sse(w, "content_block_start", {"type": "content_block_start", "index": i,
                                               "content_block": {"type": "thinking", "thinking": ""}})
                sse(w, "content_block_delta", {"type": "content_block_delta", "index": i,
                                               "delta": {"type": "thinking_delta", "thinking": b["thinking"]}})
                sse(w, "content_block_delta", {"type": "content_block_delta", "index": i,
                                               "delta": {"type": "signature_delta", "signature": b["signature"]}})
            elif b["type"] == "text":
                sse(w, "content_block_start", {"type": "content_block_start", "index": i,
                                               "content_block": {"type": "text", "text": ""}})
                sse(w, "content_block_delta", {"type": "content_block_delta", "index": i,
                                               "delta": {"type": "text_delta", "text": b["text"]}})
            else:
                sse(w, "content_block_start", {"type": "content_block_start", "index": i, "content_block": {
                    "type": "tool_use", "id": b["id"], "name": b["name"], "input": {}}})
                sse(w, "content_block_delta", {"type": "content_block_delta", "index": i,
                                               "delta": {"type": "input_json_delta", "partial_json": json.dumps(b["input"])}})
            sse(w, "content_block_stop", {"type": "content_block_stop", "index": i})
        sse(w, "message_delta", {"type": "message_delta", "delta": {"stop_reason": stop, "stop_sequence": None},
                                 "usage": {"output_tokens": usage["output_tokens"]}})
        sse(w, "message_stop", {"type": "message_stop"})
        w.write(b"0\r\n\r\n")
        w.flush()

    def _openai(self, body, path):
        content, calls, rec = openai_turn(body)
        rec["path"] = path
        log(rec)
        time.sleep(DELAY)
        msg = {"role": "assistant", "content": content, "reasoning": f"stub reasoning, step {rec['step']}"}
        if calls:
            msg["tool_calls"] = calls
        usage = {"prompt_tokens": len(json.dumps(body.get("messages", []))) // 4, "completion_tokens": 40,
                 "total_tokens": 0}
        fr = "tool_calls" if calls else "stop"
        if body.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            chunks = [{"role": "assistant", "reasoning": msg["reasoning"]}, {"content": content}]
            if calls:
                chunks.append({"tool_calls": [dict(c, index=i) for i, c in enumerate(calls)]})
            evs = [{"id": "chatcmpl-stub", "object": "chat.completion.chunk", "model": body.get("model"),
                    "choices": [{"index": 0, "delta": d, "finish_reason": None}]} for d in chunks]
            evs.append({"id": "chatcmpl-stub", "object": "chat.completion.chunk", "model": body.get("model"),
                        "choices": [{"index": 0, "delta": {}, "finish_reason": fr}], "usage": usage})
            for ev in evs:
                b = b"data: " + json.dumps(ev).encode() + b"\n\n"
                self.wfile.write(b"%x\r\n%s\r\n" % (len(b), b))
            b = b"data: [DONE]\n\n"
            self.wfile.write(b"%x\r\n%s\r\n0\r\n\r\n" % (len(b), b))
            return
        self._send({"id": "chatcmpl-stub", "object": "chat.completion", "created": int(time.time()),
                    "model": body.get("model"), "choices": [{"index": 0, "message": msg, "finish_reason": fr}],
                    "usage": usage})


if __name__ == "__main__":
    print(f"stub_agentic on {HOST}:{PORT} tasks={len(TASKS)} mode={MODE}", flush=True)
    srv = ThreadingHTTPServer((HOST, PORT), H)
    srv.daemon_threads = True
    srv.serve_forever()
