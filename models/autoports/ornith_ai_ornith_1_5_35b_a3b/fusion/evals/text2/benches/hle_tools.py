"""Humanity's Last Exam WITH tools ("HLE w/ tool" 47.5 on the Qwen3.5-122B card; "HLE (with tools)" 33.4 / 48.0 on
the Ornith-1.5-35B / -122B cards).

Setting follows CAIS hle docs/evaluation-with-tools.md (@22ed307): web search (<= 20 results per call before
filtering) + web fetch (main content as markdown, 60,000-char cap with a truncation indicator, no image/audio/video
payloads), docs/blocklist.json enforced on queries, results and fetched pages (webtools service), and a Python 3.11.16
sandbox with the pinned requirements-tools.txt and no internet (evals-pysandbox image, webtools/sandbox.py).
Deviation: CAIS runs each model in its provider's agent harness (Claude Code / Codex CLI / ...); here the common
search-agent loop (common/search_agent.py) with tools search / visit / python, CTX_MODE fold.
Data / prompts / judge / metric: as benches/hle.py (official SYSTEM_PROMPT answer format, image questions include
the image, official JUDGE_PROMPT, accuracy + calibration error). HLE_TEXT_ONLY=1 drops the image questions.
"""
import os

from benches import hle
from benches._search import AGENT_SYSTEM, run_and_write, select, stub_gold
from common.search_agent import AgentCfg

BENCH_NAME = "hle_tools"

SYSTEM = AGENT_SYSTEM + "\n\nYou also have a Python sandbox (no internet access) for computation.\n\n" + hle.SYSTEM_PROMPT


def user_msg(it):
    content = [{"type": "text", "text": it["question"]}]
    if it["image"]:
        content.append({"type": "image_url", "image_url": {"url": it["image"]}})
    return {"role": "user", "content": content}


def main(cfg):
    all_items = hle.load_items()
    items = select(all_items, cfg)
    acfg = AgentCfg.from_env(max_turns=100, ctx_mode="fold", search_topk=20, use_python=True)
    stub_gold(items, user_msg, lambda it: f"Explanation: stub.\nAnswer: {it['answer']}\nConfidence: 90%")
    run_and_write(cfg, acfg, BENCH_NAME, items,
                  messages_fn=lambda it: [{"role": "system", "content": SYSTEM}, user_msg(it)],
                  question_fn=lambda it: it["question"],
                  prompt_fn=lambda it, ans: hle.JUDGE_PROMPT.format(question=it["question"], correct_answer=it["answer"], response=ans),
                  schema=hle.SCHEMA, instructions=hle.INSTRUCTIONS,
                  meta_fn=lambda it: {"answer_type": it["answer_type"], "category": it["category"], "has_image": bool(it["image"])},
                  extra_cfg={"text_only": os.environ.get("HLE_TEXT_ONLY", "0") == "1", "n_total": len(all_items)})


def ingest(cfg, verdicts, missing):
    hle.ingest(cfg, verdicts, missing)
