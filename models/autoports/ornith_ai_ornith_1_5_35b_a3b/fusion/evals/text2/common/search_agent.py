"""Search-agent loop shared by the search benches (browsecomp, browsecomp_zh, widesearch, seal0, hle_tools).

Tongyi-DeepResearch-style tool set, exposed as OpenAI function tools (the server parses tool calls, qwen3_xml):
  search(query: [str])            web search through the webtools service (SearXNG backend, blocklist, cache)
  visit(url: [str], goal: str)    page / PDF main content as markdown, truncated (no summarizer by default)
  python(code: str)               HLE w/ tools only: docker sandbox, no network (webtools/sandbox.py)
Loop: chat(messages, tools) -> execute every tool call (in parallel) -> append assistant + tool messages -> repeat,
until the model answers without a tool call, or a budget ends (then one last call with tool_choice="none" asks for
the final answer, as Tongyi's react_agent does on context overflow).

Context management (CTX_MODE):
  fold         (default; Qwen3.5 card: "once the cumulative Tool Response length reaches a preset threshold, earlier
               Tool Responses are pruned from the history"). Here: when the estimated next prompt exceeds CTX_TRIGGER
               tokens (240k), the oldest tool responses (never those of the latest step) are replaced by a short
               placeholder until the estimate is <= FOLD_TARGET (160k); if that is not enough, reasoning of older
               steps is dropped; if still over the trigger -> discard_all.
  discard_all  (DeepSeek-V3.2 / Kimi K2.5 style; the Qwen3.5-397B card's 78.6 BrowseComp setting): over the trigger,
               the history is reset to [system, user] and the agent continues with the remaining turn budget.
  none         (Qwen3.5 card for WideSearch: "256k context window without any context management"): over the
               trigger, the final answer is requested.
Tokens are estimated with the model's tokenizer (~/evals/data/tokenizers/<model>/tokenizer.json), calibrated each
turn against the server's usage.prompt_tokens. Per-turn max_tokens is clamped to the room left in CTX_LIMIT.

Env (defaults in parentheses; per-bench defaults may differ): AGENT_MAX_TURNS (bench) AGENT_MAX_WALL (14400 s)
  CTX_MODE (bench) CTX_LIMIT (262144) CTX_TRIGGER (240000) FOLD_TARGET (160000)
  KEEP_REASONING (1: prior steps' reasoning is sent back as the assistant message's `reasoning` field, which vLLM
  maps to reasoning_content and the Qwen3.5 chat template renders as <think> for steps after the last user turn)
  REASONING_FIELD (reasoning) SEARCH_TOPK (10) VISIT_MAX_CHARS (60000 per call, split over the URLs of the call)
  VISIT_SUMMARY (0; 1 = Tongyi EXTRACTOR_PROMPT summary by the same served model, not an external API)
  MAX_QUERIES_PER_CALL (5) MAX_URLS_PER_CALL (5) BLOCKLIST (service default: full) WEBTOOLS_URL
  EPISODES (= concurrency: agent episodes in flight; each has at most one LLM request outstanding)
"""
from __future__ import annotations

import asyncio
import gzip
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path.home() / "evals/websearch"))
from webtools.client import AsyncWebTools, check_service  # noqa: E402
from webtools.sandbox import PythonSandbox  # noqa: E402

from .core import RunCfg, generate_custom  # noqa: E402

TOKENIZER_ROOT = Path.home() / "evals/data/tokenizers"
FOLD_PLACEHOLDER = "[This earlier tool response was removed from the context to save space.]"
FINAL_PROMPT = ("You have reached the limit of steps / context you can use for this task. Stop making tool calls and, "
                "based on all the information above, think again and give the most likely final answer now, in the "
                "format required by the task.")

TOOL_SEARCH = {"type": "function", "function": {
    "name": "search",
    "description": "Web search. Supply an array `query` of one or more queries; the tool returns the top results "
                   "(title, URL, snippet) for each query. Include several complementary queries in one call.",
    "parameters": {"type": "object", "properties": {
        "query": {"type": "array", "items": {"type": "string"}, "description": "Array of query strings (at most 5)."}},
        "required": ["query"]}}}
TOOL_VISIT = {"type": "function", "function": {
    "name": "visit",
    "description": "Visit webpage(s) or PDF(s) and return their main content as markdown text. Long pages are "
                   "truncated (the output says so). Image, audio and video content is not returned.",
    "parameters": {"type": "object", "properties": {
        "url": {"type": "array", "items": {"type": "string"}, "description": "The URL(s) to visit (at most 5)."},
        "goal": {"type": "string", "description": "The specific information goal for visiting the page(s)."}},
        "required": ["url", "goal"]}}}
TOOL_PYTHON = {"type": "function", "function": {
    "name": "python",
    "description": "Run Python 3.11 code in a sandbox without internet access (numpy, scipy, sympy, pandas, networkx, "
                   "mpmath, statsmodels, scikit-learn, matplotlib, pillow are installed). Each call runs in a fresh "
                   "process: variables do not persist between calls, files written to the working directory do. "
                   "Print everything you want to see; stdout and stderr are returned (long output is truncated).",
    "parameters": {"type": "object", "properties": {"code": {"type": "string", "description": "Python source code."}},
                   "required": ["code"]}}}

EXTRACTOR_PROMPT = """Please process the following webpage content and user goal to extract relevant information:

## **Webpage Content**
{webpage_content}

## **User Goal**
{goal}

## **Task Guidelines**
1. **Content Scanning for Rationale**: Locate the **specific sections/data** directly related to the user's goal within the webpage content
2. **Key Extraction for Evidence**: Identify and extract the **most relevant information** from the content, you never miss any important information, output the **full original context** of the content as far as possible, it can be more than three paragraphs.
3. **Summary Output for Summary**: Organize into a concise paragraph with logical flow, prioritizing clarity and judge the contribution of the information to the goal.

**Final Output Format using JSON format has "rational", "evidence", "summary" feilds**
"""  # Tongyi DeepResearch inference/prompt.py (verbatim, incl. typos)


def _env(name, default, typ=str):
    v = os.environ.get(name)
    return typ(v) if v not in (None, "") else default


@dataclass
class AgentCfg:
    max_turns: int = 100
    max_wall: float = 14400
    ctx_mode: str = "fold"
    ctx_limit: int = 262144
    ctx_trigger: int = 240000
    fold_target: int = 160000
    keep_reasoning: bool = True
    reasoning_field: str = "reasoning"
    search_topk: int = 10
    visit_max_chars: int = 60000
    visit_summary: bool = False
    max_queries: int = 5
    max_urls: int = 5
    use_python: bool = False

    @classmethod
    def from_env(cls, **bench_defaults) -> "AgentCfg":
        d = cls(**bench_defaults)
        return cls(
            max_turns=_env("AGENT_MAX_TURNS", d.max_turns, int), max_wall=_env("AGENT_MAX_WALL", d.max_wall, float),
            ctx_mode=_env("CTX_MODE", d.ctx_mode), ctx_limit=_env("CTX_LIMIT", d.ctx_limit, int),
            ctx_trigger=_env("CTX_TRIGGER", d.ctx_trigger, int), fold_target=_env("FOLD_TARGET", d.fold_target, int),
            keep_reasoning=_env("KEEP_REASONING", "1" if d.keep_reasoning else "0") != "0",
            reasoning_field=_env("REASONING_FIELD", d.reasoning_field), search_topk=_env("SEARCH_TOPK", d.search_topk, int),
            visit_max_chars=_env("VISIT_MAX_CHARS", d.visit_max_chars, int),
            visit_summary=_env("VISIT_SUMMARY", "1" if d.visit_summary else "0") != "0",
            max_queries=_env("MAX_QUERIES_PER_CALL", d.max_queries, int), max_urls=_env("MAX_URLS_PER_CALL", d.max_urls, int),
            use_python=d.use_python,
        )

    def tools(self) -> list[dict]:
        return [TOOL_SEARCH, TOOL_VISIT] + ([TOOL_PYTHON] if self.use_python else [])


class TokenCounter:
    def __init__(self, model: str):
        self.tok = None
        p = TOKENIZER_ROOT / model / "tokenizer.json"
        if p.exists():
            from tokenizers import Tokenizer

            self.tok = Tokenizer.from_file(str(p))

    def count(self, text: str) -> int:
        if not text:
            return 0
        if self.tok is None:
            return int(len(text) / 2.5) + 1
        return len(self.tok.encode(text, add_special_tokens=False).ids)


_TC: dict[str, TokenCounter] = {}


def _msg_text(m: dict, field: str) -> str:
    parts = []
    c = m.get("content")
    if isinstance(c, str):
        parts.append(c)
    elif isinstance(c, list):
        parts += [x.get("text", "") for x in c if isinstance(x, dict) and x.get("type") == "text"]
    if m.get(field):
        parts.append(m[field])
    for tc in m.get("tool_calls") or []:
        parts.append(tc["function"]["name"] + tc["function"].get("arguments", ""))
    return "\n".join(parts)


@dataclass
class Episode:
    cfg: RunCfg
    acfg: AgentCfg
    item_id: str
    messages0: list[dict]
    question: str
    web: AsyncWebTools
    tag: str
    events: list = field(default_factory=list)
    stats: dict = field(default_factory=lambda: {"turns": 0, "search_calls": 0, "queries": 0, "visit_urls": 0,
                                                  "python_calls": 0, "blocked": 0, "tool_errors": 0,
                                                  "prompt_tokens_sum": 0, "completion_tokens_sum": 0,
                                                  "max_prompt_tokens": 0})

    def __post_init__(self):
        if self.cfg.model not in _TC:
            _TC[self.cfg.model] = TokenCounter(self.cfg.model)
        self.tc = _TC[self.cfg.model]
        self.ntok: dict[int, int] = {}  # id(msg) -> tokens
        self.offset = 0
        self.sandbox = PythonSandbox(self.tag) if self.acfg.use_python else None

    # -------------------------------------------------------------------------------- token estimate
    def _tok(self, m: dict) -> int:
        k = id(m)
        if k not in self.ntok:
            self.ntok[k] = self.tc.count(_msg_text(m, self.acfg.reasoning_field)) + 8
        return self.ntok[k]

    def est(self, msgs: list[dict]) -> int:
        return sum(self._tok(m) for m in msgs) + self.offset

    def _retok(self, m: dict):
        self.ntok.pop(id(m), None)

    # -------------------------------------------------------------------------------- context management
    def manage(self, msgs: list[dict]) -> list[dict]:
        a = self.acfg
        e0 = self.est(msgs)
        if e0 <= a.ctx_trigger or a.ctx_mode == "none":
            return msgs
        if a.ctx_mode == "fold":
            last_asst = max((i for i, m in enumerate(msgs) if m["role"] == "assistant"), default=-1)
            pruned = 0
            for i, m in enumerate(msgs):
                if self.est(msgs) <= a.fold_target:
                    break
                if m["role"] == "tool" and i < last_asst and m["content"] != FOLD_PLACEHOLDER:
                    m["content"] = FOLD_PLACEHOLDER
                    self._retok(m)
                    pruned += 1
            dropped_r = 0
            if self.est(msgs) > a.fold_target:
                for i, m in enumerate(msgs):
                    if self.est(msgs) <= a.fold_target:
                        break
                    if m["role"] == "assistant" and i < last_asst and m.get(a.reasoning_field):
                        m.pop(a.reasoning_field)
                        self._retok(m)
                        dropped_r += 1
            self.events.append({"turn": self.stats["turns"], "event": "fold", "est_before": e0, "est_after": self.est(msgs),
                                "tool_responses_pruned": pruned, "reasoning_dropped": dropped_r})
            if self.est(msgs) <= a.ctx_trigger:
                return msgs
        # discard_all (or fold fallback)
        new = [dict(m) for m in self.messages0]
        self.ntok = {}
        self.events.append({"turn": self.stats["turns"], "event": "discard_all", "est_before": e0, "est_after": self.est(new)})
        return new

    # -------------------------------------------------------------------------------- tools
    async def tool_search(self, args: dict) -> str:
        q = args.get("query")
        if isinstance(q, str):
            try:
                q2 = json.loads(q)
                q = q2 if isinstance(q2, list) else [q]
            except Exception:  # noqa: BLE001
                q = [q]
        if not isinstance(q, list) or not q:
            return "[search] Invalid request: `query` must be a non-empty array of strings."
        q = [str(x) for x in q][: self.acfg.max_queries]
        self.stats["search_calls"] += 1
        self.stats["queries"] += len(q)
        outs = await asyncio.gather(*(self.web.search(x, self.tag, self.question, self.acfg.search_topk) for x in q))
        parts = []
        for x, d in zip(q, outs):
            if d.get("blocked"):
                self.stats["blocked"] += 1
                why = ("it quotes a long span of the task text" if "verbatim" in d["blocked"]
                       else "it targets sources blocked for this evaluation")
                parts.append(f"The search for '{x}' was refused because {why}. Rephrase the query.")
            elif d.get("error"):
                self.stats["tool_errors"] += 1
                parts.append(f"The search for '{x}' failed: {d['error'][:300]}. Try again or use a different query.")
            elif not d.get("results"):
                parts.append(f"No results found for '{x}'. Try a more general query.")
            else:
                lines = []
                for i, r in enumerate(d["results"], 1):
                    date = f"\nDate published: {r['date']}" if r.get("date") else ""
                    lines.append(f"{i}. [{r['title']}]({r['url']}){date}\n{r['snippet']}")
                parts.append(f"A web search for '{x}' found {len(lines)} results:\n\n## Web Results\n" + "\n\n".join(lines))
        return "\n=======\n".join(parts)

    async def _summarize(self, client, url: str, goal: str, content: str) -> str:
        msgs = [{"role": "user", "content": EXTRACTOR_PROMPT.format(webpage_content=content, goal=goal)}]
        r = await client.chat(msgs, max_tokens=8192)
        raw = (r.get("content") or "").replace("```json", "").replace("```", "").strip()
        try:
            j = json.loads(raw)
            ev, su = j.get("evidence", ""), j.get("summary", "")
        except Exception:  # noqa: BLE001
            ev, su = "The provided webpage content could not be processed.", "No information is available."
        return (f"The useful information in {url} for user goal {goal} as follows: \n\n"
                f"Evidence in page: \n{ev}\n\nSummary: \n{su}\n\n")

    async def tool_visit(self, args: dict, client) -> str:
        urls = args.get("url")
        goal = str(args.get("goal") or "")
        if isinstance(urls, str):
            try:
                u2 = json.loads(urls)
                urls = u2 if isinstance(u2, list) else [urls]
            except Exception:  # noqa: BLE001
                urls = [urls]
        if not isinstance(urls, list) or not urls:
            return "[visit] Invalid request: `url` must be a non-empty array of URLs."
        urls = [str(u).strip() for u in urls][: self.acfg.max_urls]
        self.stats["visit_urls"] += len(urls)
        cap = max(5000, self.acfg.visit_max_chars // len(urls))
        outs = await asyncio.gather(*(self.web.fetch(u, self.tag, max_chars=cap) for u in urls))
        parts = []
        for u, d in zip(urls, outs):
            if d.get("blocked"):
                self.stats["blocked"] += 1
                parts.append(f"[visit] {u}\nThis page is blocked for this evaluation.")
            elif d.get("error"):
                self.stats["tool_errors"] += 1
                parts.append(f"[visit] {u}\nThe page could not be retrieved: {d['error'][:300]}")
            elif self.acfg.visit_summary:
                parts.append(await self._summarize(client, u, goal, d["content"]))
            else:
                head = f"[visit] {u}"
                if d.get("final_url") and d["final_url"] != u:
                    head += f"\n(redirected to {d['final_url']})"
                if d.get("title"):
                    head += f"\nTitle: {d['title']}"
                head += f"\nGoal: {goal}" if goal else ""
                head += f"\nContent ({d['total_chars']} characters{', truncated' if d['truncated'] else ''}):\n"
                parts.append(head + d["content"])
        return "\n=======\n".join(parts)

    async def tool_python(self, args: dict) -> str:
        code = args.get("code")
        if not isinstance(code, str) or not code.strip():
            return "[python] Invalid request: `code` must be a non-empty string."
        self.stats["python_calls"] += 1
        try:
            return await self.sandbox.run(code)
        except Exception as e:  # noqa: BLE001
            self.stats["tool_errors"] += 1
            return f"[python] sandbox error: {e}"

    async def call_tool(self, tc: dict, client) -> str:
        name = tc["function"]["name"]
        raw = tc["function"].get("arguments") or "{}"
        try:
            args = json.loads(raw) if isinstance(raw, str) else raw
            if not isinstance(args, dict):
                raise ValueError("arguments must be a JSON object")
        except Exception as e:  # noqa: BLE001
            self.stats["tool_errors"] += 1
            return f"Invalid tool arguments ({e}). Arguments must be a JSON object matching the tool schema."
        if name == "search":
            return await self.tool_search(args)
        if name == "visit":
            return await self.tool_visit(args, client)
        if name == "python" and self.sandbox is not None:
            return await self.tool_python(args)
        self.stats["tool_errors"] += 1
        return f"Unknown tool '{name}'. Available tools: {', '.join(t['function']['name'] for t in self.acfg.tools())}."

    # -------------------------------------------------------------------------------- loop
    async def run(self, client) -> dict:
        a, cfg = self.acfg, self.cfg
        tools = a.tools()
        msgs = [dict(m) for m in self.messages0]
        trace = [dict(m) for m in self.messages0]
        t0 = time.time()
        final, end_reason, last = None, None, {}
        forced = None
        try:
            while True:
                if forced is None:
                    if self.stats["turns"] >= a.max_turns:
                        forced = "max_turns"
                    elif time.time() - t0 > a.max_wall:
                        forced = "max_wall"
                msgs = self.manage(msgs)
                if forced is None and a.ctx_mode == "none" and self.est(msgs) > a.ctx_trigger:
                    forced = "context"
                if forced and not (msgs and msgs[-1].get("content") == FINAL_PROMPT):
                    fm = {"role": "user", "content": FINAL_PROMPT}
                    msgs.append(fm)
                    trace.append(dict(fm))
                    self.events.append({"turn": self.stats["turns"], "event": "force_final", "reason": forced})
                room = a.ctx_limit - self.est(msgs) - 256
                max_tok = min(cfg.max_tokens, room)
                if max_tok < 512:
                    end_reason = "context_full"
                    break
                sent = list(msgs)
                r = await client.chat(sent, tools=tools, max_tokens=max_tok,
                                      **({"tool_choice": "none"} if forced else {}))
                if forced and str(r.get("error") or "").startswith("HTTP 400") and "tool_choice" in str(r.get("error")):
                    r = await client.chat(sent, tools=tools, max_tokens=max_tok)  # server without tool_choice=none
                last = r
                self.stats["turns"] += 1
                if r.get("error"):
                    end_reason = "llm_error"
                    return self._result(None, r, end_reason, forced, t0, trace, error=r["error"])
                u = r.get("usage") or {}
                if u.get("prompt_tokens"):
                    self.offset += u["prompt_tokens"] - self.est(sent)
                    self.stats["prompt_tokens_sum"] += u["prompt_tokens"]
                    self.stats["max_prompt_tokens"] = max(self.stats["max_prompt_tokens"], u["prompt_tokens"])
                self.stats["completion_tokens_sum"] += u.get("completion_tokens") or 0
                am = {"role": "assistant", "content": r.get("content") or ""}
                if r.get("tool_calls") and not forced:
                    am["tool_calls"] = [{"id": tc.get("id") or f"call_{self.stats['turns']}_{j}", "type": "function",
                                         "function": {"name": tc["function"]["name"],
                                                      "arguments": tc["function"].get("arguments") or "{}"}}
                                        for j, tc in enumerate(r["tool_calls"])]
                if a.keep_reasoning and r.get("reasoning"):
                    am[a.reasoning_field] = r["reasoning"]
                msgs.append(am)
                tr = dict(am)
                tr["reasoning"] = r.get("reasoning") or ""
                trace.append(tr)
                if "tool_calls" not in am:
                    final = am["content"]
                    end_reason = "forced_" + forced if forced else "answer"
                    break
                results = await asyncio.gather(*(self.call_tool(tc, client) for tc in am["tool_calls"]))
                for tc, res in zip(am["tool_calls"], results):
                    tm = {"role": "tool", "tool_call_id": tc["id"], "content": res}
                    msgs.append(tm)
                    trace.append(dict(tm))
        finally:
            if self.sandbox is not None:
                await self.sandbox.close()
        return self._result(final, last, end_reason, forced, t0, trace)

    def _result(self, final, last, end_reason, forced, t0, trace, error=None) -> dict:
        out_dir = self.cfg.out / "trajectories"
        out_dir.mkdir(parents=True, exist_ok=True)
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", str(self.item_id))[:120]
        with gzip.open(out_dir / f"{safe}.json.gz", "wt") as f:
            json.dump({"id": self.item_id, "end_reason": end_reason, "events": self.events, "stats": self.stats,
                       "messages": trace}, f, ensure_ascii=False)
        return {
            "content": final or "", "reasoning": (last or {}).get("reasoning") or "", "tool_calls": None,
            "finish_reason": (last or {}).get("finish_reason"), "usage": {
                "prompt_tokens": self.stats["prompt_tokens_sum"], "completion_tokens": self.stats["completion_tokens_sum"],
                "max_prompt_tokens": self.stats["max_prompt_tokens"]},
            "latency": time.time() - t0, "error": error,
            "agent": {**self.stats, "end_reason": end_reason, "forced": forced, "ctx_events": self.events[-20:],
                      "n_ctx_events": len(self.events)},
        }


def run_agents(cfg: RunCfg, acfg: AgentCfg, bench: str, jobs: list[tuple[str, list[dict], str]]) -> dict[str, dict]:
    """jobs: (item_id, initial messages, question text for the verbatim-span check). Resumable like generate_chat."""
    err = check_service()
    if err:
        sys.exit(err)
    web_holder = {}
    n_eps = int(os.environ.get("EPISODES") or cfg.concurrency)

    def factory(item_id, msgs0, question):
        async def fn(client):
            if "web" not in web_holder:
                web_holder["web"] = AsyncWebTools(bench)
                web_holder["sem"] = asyncio.Semaphore(n_eps)  # whole episodes in flight (keeps prefix-cache locality)
            ep = Episode(cfg, acfg, item_id, msgs0, question, web_holder["web"], f"{cfg.label}/{bench}/{item_id}")
            try:
                async with web_holder["sem"]:
                    return await ep.run(client)
            except Exception as e:  # noqa: BLE001
                import traceback

                return {"content": "", "reasoning": "", "tool_calls": None, "finish_reason": None, "usage": None,
                        "latency": None, "error": f"agent exception: {type(e).__name__}: {e}",
                        "agent": {"traceback": traceback.format_exc()[-2000:]}}
        return fn

    print(f"[{bench}] agent: {json.dumps(acfg.__dict__)}", flush=True)
    return generate_custom(cfg, [(i, factory(i, m, q)) for i, m, q in jobs])


def agent_summary(resps: dict[str, dict]) -> dict:
    """Aggregate per-episode tool / token statistics for summary.json."""
    rs = [r for r in resps.values() if r.get("agent") and "turns" in r["agent"]]
    if not rs:
        return {}
    def mean(k):
        return round(sum(r["agent"].get(k, 0) for r in rs) / len(rs), 2)
    ends = {}
    for r in rs:
        ends[r["agent"].get("end_reason")] = ends.get(r["agent"].get("end_reason"), 0) + 1
    return {"episodes": len(rs), "mean_turns": mean("turns"), "mean_queries": mean("queries"),
            "mean_search_calls": mean("search_calls"), "mean_visit_urls": mean("visit_urls"),
            "mean_python_calls": mean("python_calls"), "mean_blocked": mean("blocked"), "mean_tool_errors": mean("tool_errors"),
            "mean_completion_tokens": mean("completion_tokens_sum"), "mean_prompt_tokens": mean("prompt_tokens_sum"),
            "max_prompt_tokens": max(r["agent"].get("max_prompt_tokens", 0) for r in rs),
            "episodes_with_ctx_events": sum(1 for r in rs if r["agent"].get("n_ctx_events")),
            "end_reasons": ends}
