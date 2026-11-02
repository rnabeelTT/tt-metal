"""OpenAI-compatible stub server for plumbing self-tests (never used for real scores).

  python stub_server.py --port 28xxx --mode gold|garbage [--gold-map FILE] [--models a,b,c]

gold:    answers from FILE (jsonl {"key": sha256(last message content), "response": str | {content, tool_calls}});
         the file is re-read when it changes; a miss falls back to garbage.
garbage: a fixed wrong-ish answer ("I am not sure. The answer is 0.") and no tool calls.
agent / agent-garbage: scripted search agent for the search benches (common/search_agent.py). Per question (keyed by
         sha256 of the FIRST user message, which the benches put in the gold map): step 0 search (normal query, a
         long verbatim span of the question, a site: query on a blocked domain); step 1 visit (first result URL, a
         blocked URL) + a malformed tool call; step 2 python (if offered: computation + network probe); then
         --agent-visits more visits of search-result URLs (grows the context for folding tests); then the final
         answer (gold from the map in `agent`, garbage in `agent-garbage`). tool_choice="none" -> final answer.
Every reply carries a short `reasoning` field, like vLLM with a reasoning parser.
"""
import argparse
import hashlib
import json
import os
import re
import time
import uuid

import uvicorn
from fastapi import FastAPI, Request

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, required=True)
ap.add_argument("--mode", default="gold")
ap.add_argument("--gold-map", default=None)
ap.add_argument("--agent-visits", type=int, default=3)
ap.add_argument("--models", default="Qwen/Qwen3.5-122B-A10B,ornith-ai/Ornith-1.5-122B,ornith-ai/Ornith-1.5-35B-A3B")
args = ap.parse_args()

app = FastAPI()
_gold = {"mtime": None, "map": {}}
stats = {"requests": 0, "hits": 0, "agent_final": 0, "forced_final": 0, "max_prompt_chars": 0,
         "requests_with_folded_history": 0, "requests_with_reasoning_in_history": 0, "assistant_msgs_seen": 0,
         "requests_with_tool_msgs": 0, "max_tool_msgs": 0}
STEP: dict = {}
VISITED: dict = {}
FOLD = "[This earlier tool response was removed from the context to save space.]"


def gold_map():
    p = args.gold_map
    if not p or not os.path.exists(p):
        return {}
    m = os.path.getmtime(p)
    if m != _gold["mtime"]:
        d = {}
        with open(p) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    d[r["key"]] = r["response"]
        _gold.update(mtime=m, map=d)
    return _gold["map"]


def key(messages):
    c = messages[-1].get("content")
    if not isinstance(c, str):
        c = json.dumps(c, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(c.encode()).hexdigest()


@app.get("/v1/models")
async def models():
    return {"object": "list", "data": [{"id": m, "object": "model", "max_model_len": 262144} for m in args.models.split(",")]}


@app.get("/stats")
async def get_stats():
    return stats


def _text(c):
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return " ".join(x.get("text", "") for x in c if isinstance(x, dict))
    return ""


def agent_step(body):
    msgs = body["messages"]
    user = next(m for m in msgs if m["role"] == "user")
    qk = key([user])
    q = _text(user.get("content"))
    tools = [t["function"]["name"] for t in body.get("tools") or []]
    stats["max_prompt_chars"] = max(stats["max_prompt_chars"], sum(len(json.dumps(m, ensure_ascii=False)) for m in msgs))
    if any(m["role"] == "tool" and m.get("content") == FOLD for m in msgs):
        stats["requests_with_folded_history"] += 1
    asst = [m for m in msgs if m["role"] == "assistant"]
    stats["assistant_msgs_seen"] += len(asst)
    if any(m.get("reasoning") or m.get("reasoning_content") for m in asst):
        stats["requests_with_reasoning_in_history"] += 1
    ntool = sum(1 for m in msgs if m["role"] == "tool")
    if ntool:
        stats["requests_with_tool_msgs"] += 1
    stats["max_tool_msgs"] = max(stats["max_tool_msgs"], ntool)

    def final():
        stats["agent_final"] += 1
        g = gold_map().get(qk) if args.mode == "agent" else None
        if g is not None:
            stats["hits"] += 1
        return g if g is not None else "Explanation: not sure.\nExact Answer: 0\nConfidence: 5%"

    if body.get("tool_choice") == "none":
        stats["forced_final"] += 1
        return final()
    step = STEP.get(qk, 0)
    STEP[qk] = step + 1
    words = q.split()
    tc = lambda name, a: {"name": name, "arguments": json.dumps(a, ensure_ascii=False)}  # noqa: E731
    urls = []
    for m in msgs:
        if m["role"] == "tool":
            urls += re.findall(r"\]\((https?://[^)\s]+)\)", m.get("content") or "")
    seen = VISITED.setdefault(qk, set())
    fresh = [u for u in urls if u not in seen]
    if step == 0:
        cjk = re.findall(r"[\u4e00-\u9fff]", q)
        qs = ["".join(cjk[4:16]) if len(cjk) > 20 else (" ".join(words[:8]) or "test query"),
              " ".join(words[2:16]) if len(words) >= 16 else (q[:40] if len(q) >= 35 else q),
              "site:huggingface.co dataset answers"]
        return {"content": "", "tool_calls": [tc("search", {"query": qs})]}
    if step == 1:
        u = fresh[:1]
        seen.update(u)
        return {"content": "", "tool_calls": [tc("visit", {"url": u + ["https://huggingface.co/datasets/cais/hle"], "goal": "find the answer"}),
                                              {"name": "search", "arguments": "{bad json"}]}
    if step == 2 and "python" in tools:
        code = ("import math, socket\nprint('factorial', math.factorial(20))\n"
                "try:\n    socket.create_connection(('1.1.1.1', 443), timeout=3)\n    print('NETWORK OPEN')\n"
                "except Exception as e:\n    print('network blocked:', type(e).__name__)\n"
                "open('state.txt', 'w').write('kept')\n")
        return {"content": "", "tool_calls": [tc("python", {"code": code})]}
    if step == 3 and "python" in tools:
        return {"content": "", "tool_calls": [tc("python", {"code": "print(open('state.txt').read())"})]}
    if step < (4 if "python" in tools else 2) + args.agent_visits and fresh:
        u = fresh[:2]
        seen.update(u)
        return {"content": "", "tool_calls": [tc("visit", {"url": u, "goal": "more context"})]}
    return final()


@app.post("/v1/chat/completions")
async def chat(req: Request):
    body = await req.json()
    stats["requests"] += 1
    resp = None
    if args.mode in ("agent", "agent-garbage"):
        resp = agent_step(body)
    elif args.mode == "gold":
        resp = gold_map().get(key(body["messages"]))
        if resp is not None:
            stats["hits"] += 1
    if resp is None:
        resp = "I am not sure. The answer is 0."
    if isinstance(resp, str):
        resp = {"content": resp}
    msg = {"role": "assistant", "content": resp.get("content"), "reasoning": f"stub reasoning {stats['requests']}"}
    finish = "stop"
    if resp.get("tool_calls"):
        msg["tool_calls"] = [
            {"id": tc.get("id") or f"call_{uuid.uuid4().hex[:12]}", "type": "function",
             "function": {"name": tc["function"]["name"] if "function" in tc else tc["name"],
                          "arguments": tc["function"]["arguments"] if "function" in tc else
                          (tc["arguments"] if isinstance(tc["arguments"], str) else json.dumps(tc["arguments"]))}}
            for tc in resp["tool_calls"]]
        finish = "tool_calls"
    n_in = sum(len(json.dumps(m.get("content"))) for m in body["messages"]) // 4
    n_out = len(resp.get("content") or "") // 4 + 1
    return {
        "id": "chatcmpl-" + uuid.uuid4().hex, "object": "chat.completion", "created": int(time.time()),
        "model": body.get("model"),
        "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
        "usage": {"prompt_tokens": n_in, "completion_tokens": n_out, "total_tokens": n_in + n_out},
    }


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")
