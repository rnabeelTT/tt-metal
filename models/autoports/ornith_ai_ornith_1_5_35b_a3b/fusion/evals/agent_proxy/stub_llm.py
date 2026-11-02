"""Scripted OpenAI-compatible stub "model" for offline plumbing checks of the agentic
harnesses (NOT a model server; no inference). It plays a gold-patch agent:

  SWE-bench (OpenHands / SWE-agent / mini-swe-agent): finds the instance whose problem
  statement appears in the first user message (datasets in $STUB_DATA/*.jsonl, field
  `patch`), then emits a native tool call to the harness's bash-like tool that writes the
  gold patch (base64) and `git apply`s it in the repo directory named in the prompt, and
  on the next turn calls the finish/submit tool. STUB_MODE=empty submits without editing.
  Terminal-Bench / Terminus-2 (json parser): if the instruction matches a task dir in
  $STUB_TB_TASKS, sends that task's solution/solve.sh as keystrokes, then task_complete.

Every response carries a `reasoning` field (like vLLM 0.26 with a reasoning parser); every
request is logged to $STUB_LOG with whether earlier assistant turns came back with
reasoning, so the reasoning round-trip of each harness (and of llm_proxy.py) is visible.
usage: stub_llm.py <port>
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

PORT = int(sys.argv[1])
DATA = os.environ.get("STUB_DATA", os.path.expanduser("~/evals/swebench_support/data"))
LOG = open(os.environ.get("STUB_LOG", "stub_requests.jsonl"), "a", buffering=1)
MODE = os.environ.get("STUB_MODE", "gold")
MODEL = os.environ.get("STUB_MODEL", "stub/model")
TB_TASKS = os.environ.get("STUB_TB_TASKS", "")
DELAY = float(os.environ.get("STUB_DELAY", "0.2"))

ROWS = []
for f in glob.glob(os.path.join(DATA, "*.jsonl")):
    for line in open(f):
        r = json.loads(line)
        ROWS.append((r["problem_statement"].strip()[:300], r["instance_id"], r["patch"]))
TB = []
if TB_TASKS:
    for d in glob.glob(os.path.join(TB_TASKS, "*")):
        ins, sol = os.path.join(d, "instruction.md"), os.path.join(d, "solution", "solve.sh")
        if os.path.exists(ins) and os.path.exists(sol):
            TB.append((open(ins).read().strip()[:200], os.path.basename(d), open(sol).read()))
lock = threading.Lock()


def text_of(c):
    if isinstance(c, list):
        return "".join(p.get("text", "") for p in c if isinstance(p, dict))
    return c or ""


def find_instance(prompt):
    best = None
    for key, iid, patch in ROWS:
        k = key[:150]
        if k and k in prompt and (best is None or len(k) > len(best[0])):
            best = (k, iid, patch)
    return best


def repo_dir(prompt):
    for pat in (r"(/workspace/[\w.\-]+)/?", r"(/testbed)\b", r"(/app)\b"):
        m = re.search(pat, prompt)
        if m:
            return m.group(1)
    return "/testbed"


def pick_tool(tools, kinds):
    names = [t.get("function", {}).get("name", "") for t in tools or []]
    for k in kinds:
        for n in names:
            if k in n.lower():
                return n, next(t for t in tools if t["function"]["name"] == n)
    return None, None


def tool_call(name, args):
    return {"id": "call_" + uuid.uuid4().hex[:24], "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)}}


def swe_turn(body, prompt, n_assist):
    tools = body.get("tools") or []
    hit = find_instance(prompt)
    bash, bash_t = pick_tool(tools, ["terminal", "execute_bash", "bash"])
    fin, _ = pick_tool(tools, ["finish", "submit"])
    rd = repo_dir(prompt)
    if n_assist == 0 and hit and MODE == "gold" and bash:
        b64 = base64.b64encode(hit[2].encode()).decode()
        cmd = (f"cd {rd} && echo {b64} | base64 -d > /tmp/stub_gold.diff && "
               f"git apply -v /tmp/stub_gold.diff && git status --short")
        params = (bash_t["function"].get("parameters") or {}).get("properties", {})
        args = {"command": cmd}
        if "is_input" in params:
            args["is_input"] = False
        if "security_risk" in params:
            args["security_risk"] = "LOW"
        if "summary" in params:
            args["summary"] = "apply patch"
        return f"Applying the fix for {hit[1]}.", [tool_call(bash, args)], hit[1]
    if fin:
        props = (next(t for t in tools if t["function"]["name"] == fin)["function"].get("parameters") or {}).get("properties", {})
        args = {"message": "done"} if "message" in props else {}
        return "Done.", [tool_call(fin, args)], hit[1] if hit else None
    # no finish tool (e.g. mini-swe-agent text mode): print the submit sentinel
    return ("```bash\necho COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n```", [], hit[1] if hit else None)


def terminus_turn(prompt, n_assist):
    task = None
    for key, name, sol in TB:
        if key[:120] in prompt:
            task = (name, sol)
    if n_assist == 0 and task:
        b64 = base64.b64encode(task[1].encode()).decode()
        ks = f"echo {b64} | base64 -d > /tmp/stub_solve.sh && bash /tmp/stub_solve.sh; echo STUB_DONE\n"
        return json.dumps({"analysis": f"stub solving {task[0]}", "plan": "run reference solution",
                           "commands": [{"keystrokes": ks, "duration": 60}], "task_complete": False}), task[0]
    return json.dumps({"analysis": "done", "plan": "", "commands": [], "task_complete": True}), task[0] if task else None


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

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        msgs = body.get("messages", [])
        users = [text_of(m.get("content")) for m in msgs if m.get("role") in ("user", "system")]
        prompt = "\n".join(users)
        assists = [m for m in msgs if m.get("role") == "assistant"]
        is_terminus = "task_complete" in prompt and "keystrokes" in prompt
        if is_terminus:
            content, tag = terminus_turn(prompt, len(assists))
            calls = []
        else:
            content, calls, tag = swe_turn(body, prompt, len(assists))
        time.sleep(DELAY)
        rec = {k: body.get(k) for k in ("model", "temperature", "top_p", "top_k", "max_tokens", "max_completion_tokens", "stream")}
        rec.update({"n_msgs": len(msgs), "n_assistant": len(assists), "n_tools": len(body.get("tools") or []),
                    "tool_names": [t.get("function", {}).get("name") for t in body.get("tools") or []],
                    "assistant_with_reasoning": sum(1 for m in assists if m.get("reasoning") or m.get("reasoning_content")),
                    "assistant_reasoning_keys": sorted({k for m in assists for k in m if k.startswith("reason")}),
                    "matched": tag, "terminus": is_terminus, "prompt_chars": len(prompt)})
        with lock:
            LOG.write(json.dumps(rec) + "\n")
        msg = {"role": "assistant", "content": content, "reasoning": f"stub reasoning for turn {len(assists)}"}
        if calls:
            msg["tool_calls"] = calls
        usage = {"prompt_tokens": len(json.dumps(msgs)) // 4, "completion_tokens": 50, "total_tokens": 0}
        if body.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            cid = "chatcmpl-stub"
            chunks = [{"role": "assistant", "reasoning": msg["reasoning"]}, {"content": content}]
            if calls:
                chunks.append({"tool_calls": [dict(c, index=i) for i, c in enumerate(calls)]})
            for d in chunks:
                ev = {"id": cid, "object": "chat.completion.chunk", "model": body.get("model"),
                      "choices": [{"index": 0, "delta": d, "finish_reason": None}]}
                b = b"data: " + json.dumps(ev).encode() + b"\n\n"
                self.wfile.write(b"%x\r\n%s\r\n" % (len(b), b))
            ev = {"id": cid, "object": "chat.completion.chunk", "model": body.get("model"),
                  "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls" if calls else "stop"}], "usage": usage}
            for b in (b"data: " + json.dumps(ev).encode() + b"\n\n", b"data: [DONE]\n\n"):
                self.wfile.write(b"%x\r\n%s\r\n" % (len(b), b))
            self.wfile.write(b"0\r\n\r\n")
            return
        self._send({"id": "chatcmpl-stub", "object": "chat.completion", "created": int(time.time()),
                    "model": body.get("model"),
                    "choices": [{"index": 0, "message": msg, "finish_reason": "tool_calls" if calls else "stop"}],
                    "usage": usage})


if __name__ == "__main__":
    print(f"stub_llm on :{PORT} rows={len(ROWS)} tb_tasks={len(TB)} mode={MODE}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", PORT), H).serve_forever()
