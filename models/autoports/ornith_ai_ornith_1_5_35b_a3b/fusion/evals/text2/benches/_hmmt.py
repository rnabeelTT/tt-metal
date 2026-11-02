"""HMMT Feb 2025 / Nov 2025 (30 final-answer problems each), MathArena data + MathArena's answer parser.

Prompt (Qwen card "Best Practices / Standardize Output Format"): problem + "\n\nPlease reason step by step, and put
your final answer within \\boxed{}."   PROMPT_STYLE=matharena uses MathArena's own template instead
("Put your final answer within \\boxed{}.\n\n{problem}").
Scoring: MathArena grader.extract_and_grade (non-lean, non-exact-match branch: extract_answer(strict_parsing=False)
-> parse_answer(gold) -> check_answers), applied to `content` only.
Metric: mean accuracy over N_SAMPLES (default 4, as MathArena) samples per problem.
"""
import os
import sys
from pathlib import Path

from datasets import load_dataset

from common.core import count_errors, finish, generate_chat, select_items, write_gold_map, write_jsonl

MATHARENA_SRC = Path.home() / "evals/src/matharena/src"
sys.path.insert(0, str(MATHARENA_SRC))
from loguru import logger  # noqa: E402

logger.remove()
from matharena.parser import check_answers, extract_answer, parse_answer  # noqa: E402

QWEN_SUFFIX = "\n\nPlease reason step by step, and put your final answer within \\boxed{}."
MATHARENA_TEMPLATE = "Put your final answer within \\boxed{{}}.\n\n{problem}"


def grade(content: str, gold: str):
    is_list = "," in gold
    try:
        ans, _ = extract_answer(content, strict_parsing=False, parse=True, list_answer=is_list, typed_delimiters=True)
        g, _ = parse_answer(gold, list_answer=is_list, typed_delimiters=True)
        return str(ans), bool(check_answers(ans, g))
    except Exception as e:  # noqa: BLE001
        return f"<parse error {e}>", False


def run(cfg, dataset_path: str):
    n_samples = int(os.environ.get("N_SAMPLES", 4))
    style = os.environ.get("PROMPT_STYLE", "qwen")
    ds = load_dataset(dataset_path, split="train")
    items = [{"id": str(r["problem_idx"]), "problem": r["problem"], "answer": str(r["answer"])} for r in ds]
    items = select_items(items, cfg)
    cfg.dump(module=cfg.bench, dataset=dataset_path, n_samples=n_samples, prompt_style=style)

    def prompt(it):
        if style == "matharena":
            return MATHARENA_TEMPLATE.format(problem=it["problem"])
        return it["problem"] + QWEN_SUFFIX

    reqs = []
    for it in items:
        msgs = [{"role": "user", "content": prompt(it)}]
        for s in range(n_samples):
            reqs.append((f"{it['id']}#{s}", msgs))
    write_gold_map([([{"role": "user", "content": prompt(it)}], f"Thus the answer is $\\boxed{{{it['answer']}}}$.")
                    for it in items])
    resps = generate_chat(cfg, reqs)

    rows, per_problem = [], {}
    for it in items:
        for s in range(n_samples):
            r = resps.get(f"{it['id']}#{s}", {})
            ans, ok = grade(r.get("content") or "", it["answer"])
            rows.append({"id": f"{it['id']}#{s}", "gold": it["answer"], "parsed": ans, "correct": ok,
                         "finish_reason": r.get("finish_reason"), "error": r.get("error")})
            per_problem.setdefault(it["id"], []).append(ok)
    write_jsonl(cfg.out / "scores.jsonl", rows)
    acc = sum(r["correct"] for r in rows) / max(1, len(rows))
    finish(cfg, "accuracy", acc * 100, len(items), {
        "n_samples_per_problem": n_samples,
        "pass_at_1_mean": acc * 100,
        "per_problem_acc": {k: sum(v) / len(v) for k, v in per_problem.items()},
        "truncated_responses": sum(1 for r in rows if r["finish_reason"] == "length"),
        "request_errors": count_errors(resps),
    })
