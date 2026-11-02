"""OpenAI-compatible stub for tau2 plumbing tests (both roles on one port; never used for real scores).

  .venv/bin/python stub_tau2.py --port 284xx --mode gold|garbage [--user-model gpt-4.1-2025-04-14]

Role is chosen by the request's `model`: == --user-model -> user simulator, anything else -> agent.
gold:
  user  - identifies its task from <stub_task_id> (added to its system prompt by run_tau2.py when
          TAU2_STUB_TASK_TAG=1; telecom scenarios are shared by many tasks) or else the <scenario>, opens with "<<task:<domain>/<id>>> ...",
          executes the task's user-side gold actions (telecom device tools) as OpenAI tool_calls when the agent
          says "[step]", reports progress as "[done k]", and ends with ###STOP### once the agent says "[final]".
  agent - reads the task tag, replays the task's assistant-side gold actions in order as OpenAI tool_calls
          (one per turn, progress = its own prior tool calls + the user's "[done k]"), asks the user to act for
          user-side actions, and finally states every communicate_info string followed by "[final]".
  => reward 1.0 on every task if the harness plumbing (HTTP, tool-call round trip, user tools, evaluator) works.
garbage: agent always answers "I'm sorry, I cannot help with that."; user opens, then ###STOP###.
GET /stats -> request counts per role and the sampling keys seen in agent requests.
"""
import argparse
import json
import re
import time
import uuid

import uvicorn
from fastapi import FastAPI, Request

from tau2.run import load_tasks

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, required=True)
ap.add_argument("--mode", default="gold")
ap.add_argument("--user-model", default="gpt-4.1-2025-04-14")
ap.add_argument("--domains", default="airline,retail,telecom")
args = ap.parse_args()

SCEN = {}   # str(user_scenario) -> (domain, task)
TASKS = {}  # "domain/id" -> task
for dom in args.domains.split(","):
    for t in load_tasks(dom, "base"):
        SCEN.setdefault(str(t.user_scenario), (dom, t))
        TASKS[f"{dom}/{t.id}"] = t

stats = {"agent_requests": 0, "user_requests": 0, "agent_body_keys": {}, "user_body_keys": {}, "unknown_task": 0}
app = FastAPI()


def gold_actions(task):
    ec = task.evaluation_criteria
    return list(ec.actions or []) if ec else []


def comm_info(task):
    ec = task.evaluation_criteria
    return list(ec.communicate_info or []) if ec and ec.communicate_info else []


def reply(body, content=None, tool_call=None):
    msg = {"role": "assistant", "content": content, "reasoning": "stub reasoning"}
    fr = "stop"
    if tool_call:
        name, arguments = tool_call
        msg["tool_calls"] = [{"id": "call_" + uuid.uuid4().hex[:16], "type": "function",
                              "function": {"name": name, "arguments": json.dumps(arguments)}}]
        fr = "tool_calls"
    return {"id": "chatcmpl-" + uuid.uuid4().hex, "object": "chat.completion", "created": int(time.time()),
            "model": body.get("model"), "choices": [{"index": 0, "message": msg, "finish_reason": fr}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20}}


def text(m):
    c = m.get("content")
    if isinstance(c, list):
        return " ".join(p.get("text", "") for p in c if isinstance(p, dict))
    return c or ""


def agent_turn(body):
    msgs = body["messages"]
    if args.mode != "gold":
        return reply(body, "I'm sorry, I cannot help with that.")
    tag = None
    for m in msgs:
        if m["role"] == "user":
            mt = re.search(r"<<task:(.+?)>>", text(m))
            if mt:
                tag = mt.group(1)
                break
    task = TASKS.get(tag)
    if task is None:
        stats["unknown_task"] += 1
        return reply(body, "Could you describe your issue? [no task tag]")
    a_done = sum(1 for m in msgs if m["role"] == "assistant" and m.get("tool_calls"))
    u_done = 0
    for m in msgs:
        if m["role"] == "user":
            for k in re.findall(r"\[done (\d+)\]", text(m)):
                u_done = max(u_done, int(k))
    ai = ui = 0
    nxt = None
    for act in gold_actions(task):
        if act.requestor == "assistant":
            if ai < a_done:
                ai += 1
                continue
        else:
            if ui < u_done:
                ui += 1
                continue
        nxt = act
        break
    if nxt is None:
        return reply(body, "Everything is complete. " + " ".join(comm_info(task)) + " [final]")
    if nxt.requestor == "assistant":
        return reply(body, None, (nxt.name, nxt.arguments))
    return reply(body, f"Please perform the next step on your device: {nxt.name}. [step]")


def user_turn(body):
    msgs = body["messages"]
    sysm = text(msgs[0]) if msgs and msgs[0]["role"] == "system" else ""
    dom_task = None
    mt = re.search(r"<stub_task_id>(.+?)</stub_task_id>", sysm)
    if mt and mt.group(1) in TASKS:
        dom_task = (mt.group(1).split("/", 1)[0], TASKS[mt.group(1)])
    for scen, dt in ([] if dom_task else SCEN.items()):
        if scen in sysm:
            dom_task = dt
            break
    own = [m for m in msgs if m["role"] == "assistant"]
    if dom_task is None:
        stats["unknown_task"] += 1
        return reply(body, "###STOP###")
    dom, task = dom_task
    if not own:
        return reply(body, f"<<task:{dom}/{task.id}>> Hi, I need some help please.")
    if args.mode != "gold":
        return reply(body, "###STOP###")
    u_done = sum(1 for m in own if m.get("tool_calls"))
    last = msgs[-1]
    if last["role"] == "tool":
        return reply(body, f"Done. [done {u_done}]")
    lt = text(last)
    if "[final]" in lt:
        return reply(body, "Thanks, that's all. ###STOP###")
    if "[step]" in lt:
        ua = [a for a in gold_actions(task) if a.requestor == "user"]
        if u_done < len(ua):
            return reply(body, None, (ua[u_done].name, ua[u_done].arguments))
    return reply(body, f"Okay. [done {u_done}]")


@app.get("/v1/models")
async def models():
    ids = ["Qwen/Qwen3.5-122B-A10B", "ornith-ai/Ornith-1.5-122B", "ornith-ai/Ornith-1.5-35B-A3B", args.user_model]
    return {"object": "list", "data": [{"id": i, "object": "model"} for i in ids]}


@app.get("/stats")
async def get_stats():
    return stats


@app.post("/v1/chat/completions")
async def chat(req: Request):
    body = await req.json()
    role = "user" if body.get("model") == args.user_model else "agent"
    stats[f"{role}_requests"] += 1
    keys = stats[f"{role}_body_keys"]
    for k, v in body.items():
        if k not in ("messages", "tools"):
            keys[k] = v if k != "extra_body" else v
    keys["n_tools"] = len(body.get("tools") or [])
    return agent_turn(body) if role == "agent" else user_turn(body)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")
