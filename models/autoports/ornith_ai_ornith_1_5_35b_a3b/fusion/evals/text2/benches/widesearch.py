"""WideSearch (ByteDance Seed; Qwen3.5-122B card 60.5, Ornith-1.5-35B 67.8, Ornith-1.5-122B 75.2) as a search agent.

Data: 200 tasks (100 en + 100 zh), HF ByteDance-Seed/WideSearch (~/evals/data/widesearch: widesearch.jsonl +
widesearch_gold/<id>.csv). The answer is one markdown table with the requested columns.
Agent: official single-agent system prompt (src/agent/prompt.py default_system_prompt_{en,zh}) + the task query as
the user turn; tools search + visit (ours; the official repo leaves search tools to the user); CTX_MODE none (Qwen3.5
card: "WideSearch: we use a 256k context window without any context management"); AGENT_MAX_TURNS 50 (official
DEFAULT_MAX_STEPS). The final answer is the last assistant content (as the official runner).
Evaluation: OFFICIAL src/evaluation code (@9825ba7) run by ~/evals/websearch/ws_eval.py in its own venv (pinned
pandas 2.3.1 etc.), LLM calls (column mapping, primary-key alignment, llm_judge cells; official eval model
gpt-4.1-2025-04-14, temperature 0) answered by Claude agents through judge rounds: every round writes
judge_inputs.jsonl with the prompts the official code asked for; ingest adds the answers to judge_cache.jsonl and
replays the evaluation; repeat until no JUDGE_PENDING (usually 1-2 rounds).
Metric: Item-F1 averaged over tasks (f1_by_item avg@1; the card's WideSearch number) + row F1, precision/recall,
success rate.
"""
import json
import os
import subprocess
from pathlib import Path

from benches._search import select
from common.core import count_errors, finish, read_jsonl, write_gold_map, write_jsonl
from common.judge import write_judge_inputs
from common.search_agent import AgentCfg, agent_summary, run_agents

BENCH_NAME = "widesearch"
DATA = Path.home() / "evals/data/widesearch"
WS = Path.home() / "evals/websearch"
WS_PY = WS / "ws_eval_venv/bin/python"

SYSTEM_EN = """# Role
You are an expert in online search. You task is gathering relevant information using advanced online search tools based on the user's query, and providing accurate answers according to the search results.

# Task Description
Upon receiving the user's query, you must thoroughly analyze and understand the user's requirements. In order to effectively address the user's query, you should make the best use of the provided tools to acquire comprehensive and reliable information and data. Below are the principles you should adhere to while performing this task:

- Fully understand the user's needs: Analyze the user's query, if necessary, break it down into smaller components to ensure a clear understanding of the user's primary intent.
- Flexibly use tools: After fully comprehending the user's needs, employ the provided tools to retrieve the necessary information.If the information retrieved previously is deemed incomplete or inaccurate and insufficient to answer the user's query, reassess what additional information is required and invoke the tool again until all necessary data is obtained."""

SYSTEM_ZH = """# 角色设定
你是一位联网信息搜索专家，你需要根据用户的问题，通过联网搜索来搜集相关信息，然后根据这些信息来回答用户的问题。

# 任务描述
当你接收到用户的问题后，你需要充分理解用户的需求，利用我提供给你的工具，获取相对应的信息、资料，以解答用户的问题。
以下是你在执行任务过程中需要遵循的原则：
- 充分理解用户需求：你需要全面分析和理解用户的问题，必要时对用户的问题进行拆解，以确保领会到用户问题的主要意图。
- 灵活使用工具：当你充分理解用户需求后，请你使用我提供的工具获取信息；当你认为上次工具获取到的信息不全或者有误，以至于不足以回答用户问题时，请思考还需要搜索什么信息，再次调用工具获取信息，直至信息完备。"""

INSTRUCTIONS = """WideSearch judging = acting as the official evaluation model (gpt-4.1-2025-04-14, temperature 0) for the prompts
the official evaluation code issued (meta.stage: colmap / keymap = vocabulary alignment, eval = column grading).
Answer judge_prompt exactly as that model would: the reply text goes in `content` and MUST end with the
```json {...}``` block the prompt asks for (alignment: {"origin": "transform", ...}; grading: {"idx_0": 0|1, ...}).
Return {"id", "content"}. After ingesting, the evaluation is replayed; a later round may ask for more prompts."""

SCHEMA = {"type": "object", "properties": {"content": {"type": "string"}}, "required": ["content"]}


def load_items():
    return [json.loads(line) | {"id": json.loads(line)["instance_id"]} for line in open(DATA / "widesearch.jsonl")]


def messages(it):
    return [{"role": "system", "content": SYSTEM_ZH if it["language"] == "zh" else SYSTEM_EN},
            {"role": "user", "content": it["query"]}]


def _gold_table(it) -> str:
    import pandas as pd

    df = pd.read_csv(DATA / "widesearch_gold" / f"{it['instance_id']}.csv").fillna("")
    req = json.loads(it["evaluation"])["required"]
    df = df[[c for c in df.columns if c.strip().lower().replace(" ", "") in req]]
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(str(x).replace("|", "/").replace("\n", " ") for x in row) + " |" for row in df.values]
    return "```markdown\n" + "\n".join(lines) + "\n```"


def replay(cfg, round_no: int):
    r = subprocess.run([str(WS_PY), str(WS / "ws_eval.py"), str(cfg.out)], capture_output=True, text=True)
    (cfg.out / f"ws_eval_round{round_no}.log").write_text(r.stdout + "\n" + r.stderr[-20000:])
    if r.returncode != 0:
        raise SystemExit(f"ws_eval failed ({r.returncode}); see {cfg.out}/ws_eval_round{round_no}.log")
    st = json.loads(r.stdout.strip().splitlines()[-1])
    pend = read_jsonl(cfg.out / "ws_pending.jsonl")
    if pend:
        rows = [{"id": p["id"], "bench": BENCH_NAME, "question": "", "reference": "", "model_answer": "",
                 "judge_prompt": p["prompt"], "verdict_schema": SCHEMA,
                 "meta": {"stage": p["stage"], "instance_id": p["instance_id"], "round": round_no}} for p in pend]
        p = write_judge_inputs(cfg.out, rows, [], INSTRUCTIONS)
        write_jsonl(cfg.out / f"judge_inputs.round{round_no}.jsonl", rows)
        print(f"[widesearch] judge round {round_no}: {len(rows)} prompts for {st['pending_instances']} tasks "
              f"({st['scored']} tasks fully scored)")
        finish(cfg, "item_f1", None, len(json.loads((cfg.out / 'sample_ids.json').read_text())["ids"]),
               {"judge_round": round_no, "pending_prompts": len(rows), **st,
                **json.loads((cfg.out / "agent_summary.json").read_text() or "{}")}, judge_pending=p)
        return False
    return True


def metrics(cfg):
    ids = json.loads((cfg.out / "sample_ids.json").read_text())["ids"]
    keys = ["score", "precision_by_row", "recall_by_row", "f1_by_row", "precision_by_item", "recall_by_item", "f1_by_item"]
    rows, agg = [], {k: 0.0 for k in keys}
    for i in ids:
        f = cfg.out / "ws_eval" / f"{i}.json"
        d = json.loads(f.read_text()) if f.exists() else {k: 0.0 for k in keys} | {"msg": "not evaluated"}
        rows.append({"id": i, **{k: d.get(k, 0.0) for k in keys}, "msg": str(d.get("msg", ""))[:300]})
        for k in keys:
            agg[k] += float(d.get(k, 0.0) or 0.0)
    n = len(ids)
    agg = {k: round(100 * v / n, 2) for k, v in agg.items()}
    by_lang = {}
    for lang in ("en", "zh"):
        rs = [r for r in rows if f"_{lang}_" in r["id"]]
        if rs:
            by_lang[lang] = {"n": len(rs), "f1_by_item": round(100 * sum(r["f1_by_item"] for r in rs) / len(rs), 2)}
    write_jsonl(cfg.out / "scores.jsonl", rows)
    print(f"*** widesearch: item F1 {agg['f1_by_item']} | row F1 {agg['f1_by_row']} | success {agg['score']} | n = {n}")
    summ = json.loads((cfg.out / "agent_summary.json").read_text() or "{}")
    finish(cfg, "item_f1", agg["f1_by_item"], n, {"avg@1": agg, "by_language": by_lang, **summ})


def main(cfg):
    items = select(load_items(), cfg)
    acfg = AgentCfg.from_env(max_turns=50, ctx_mode="none")
    if os.environ.get("STUB_GOLD_MAP"):  # self-tests: the stub answers with the gold table
        write_gold_map([([messages(it)[1]], _gold_table(it)) for it in items])
    cfg.dump(module=BENCH_NAME, agent=acfg.__dict__)
    resps = run_agents(cfg, acfg, BENCH_NAME, [(it["id"], messages(it), it["query"]) for it in items])
    (cfg.out / "agent_summary.json").write_text(json.dumps({"request_errors": count_errors(resps), "agent": agent_summary(resps)}))
    for f in ("judge_cache.jsonl",):
        (cfg.out / f).touch()
    if replay(cfg, 1):
        metrics(cfg)


def ingest(cfg, verdicts, missing):
    cache = {r["id"]: r for r in read_jsonl(cfg.out / "judge_cache.jsonl")}
    added = 0
    for k, v in verdicts.items():
        if k.startswith("ws:") and isinstance(v.get("content"), str) and k not in cache:
            cache[k] = {"id": k, "content": v["content"]}
            added += 1
    write_jsonl(cfg.out / "judge_cache.jsonl", cache.values())
    rounds = sorted(int(p.stem.split("round")[1]) for p in cfg.out.glob("judge_inputs.round*.jsonl"))
    nxt = (rounds[-1] + 1) if rounds else 2
    print(f"[widesearch] added {added} judgments to the cache ({len(cache)} total); replaying evaluation (round {nxt})")
    if replay(cfg, nxt):
        metrics(cfg)
