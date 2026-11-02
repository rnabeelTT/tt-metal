"""IFBench (allenai/IFBench, single-turn OOD test set, 300 prompts / 58 constraint types).

Generation: the dataset `prompt` verbatim as one user turn (no system prompt), thinking on, `content` only.
Scoring: the official repo's run_eval.py (evaluation_lib strict + loose) run in its own venv ~/evals/ifbench/.venv
from the pinned clone ~/evals/src/IFBench.
Headline metric: prompt-level LOOSE accuracy (README: "In the paper we generally report the prompt-level loose
accuracy of IFBench"). prompt-level strict and instruction-level strict/loose are reported alongside.
"""
import json
import os
import subprocess
from pathlib import Path

from common.core import count_errors, finish, generate_chat, read_jsonl, select_items, write_gold_map, write_jsonl

REPO = Path.home() / "evals/src/IFBench"
VENV_PY = Path.home() / "evals/ifbench/.venv/bin/python"
NLTK = Path.home() / "evals/ifbench/nltk_data"


def official_eval(out: Path, items: list[dict], prompt_to_resp: dict[str, str]) -> dict:
    ev = out / "official_eval"
    ev.mkdir(exist_ok=True)
    write_jsonl(ev / "input_data.jsonl", items)
    write_jsonl(ev / "model-responses.jsonl", [{"prompt": it["prompt"], "response": prompt_to_resp.get(it["prompt"], "")}
                                                for it in items])
    env = dict(os.environ, NLTK_DATA=str(NLTK), PYTHONPATH=str(REPO))
    p = subprocess.run([str(VENV_PY), "-m", "run_eval", f"--input_data={ev/'input_data.jsonl'}",
                        f"--input_response_data={ev/'model-responses.jsonl'}", f"--output_dir={ev}"],
                       cwd=REPO, env=env, capture_output=True, text=True)
    (ev / "run_eval.log").write_text(p.stdout + "\n" + p.stderr)
    if p.returncode != 0:
        raise SystemExit(f"official IFBench eval failed, see {ev/'run_eval.log'}")
    res = {}
    for mode in ("strict", "loose"):
        rows = read_jsonl(ev / f"model-eval_results_{mode}.jsonl")
        res[f"prompt_level_{mode}"] = 100 * sum(r["follow_all_instructions"] for r in rows) / len(rows)
        inst = [x for r in rows for x in r["follow_instruction_list"]]
        res[f"inst_level_{mode}"] = 100 * sum(inst) / len(inst)
        res[f"_rows_{mode}"] = rows
    return res


def main(cfg):
    items = read_jsonl(REPO / "data/IFBench_test.jsonl")
    for it in items:
        it["id"] = it["key"]
    items = select_items(items, cfg)
    cfg.dump(module="ifbench", data=str(REPO / "data/IFBench_test.jsonl"), metric_note="prompt-level loose (paper)")
    reqs = [(it["id"], [{"role": "user", "content": it["prompt"]}]) for it in items]
    # self-test only: STUB_GOLD_RESPONSES=<jsonl with prompt/response> replays known responses through the stub
    gold_src = os.environ.get("STUB_GOLD_RESPONSES")
    if gold_src:
        known = {r["prompt"]: r["response"] for r in read_jsonl(Path(gold_src))}
        write_gold_map([(m, known[m[0]["content"]]) for _, m in reqs if m[0]["content"] in known])
    resps = generate_chat(cfg, reqs)
    p2r = {it["prompt"]: resps.get(it["id"], {}).get("content") or "" for it in items}
    res = official_eval(cfg.out, [{k: v for k, v in it.items() if k != "id"} for it in items], p2r)
    rows = []
    for it, rs, rl in zip(items, res.pop("_rows_strict"), res.pop("_rows_loose")):
        r = resps.get(it["id"], {})
        rows.append({"id": it["id"], "instruction_id_list": it["instruction_id_list"],
                     "strict": rs["follow_all_instructions"], "loose": rl["follow_all_instructions"],
                     "finish_reason": r.get("finish_reason"), "error": r.get("error")})
    write_jsonl(cfg.out / "scores.jsonl", rows)
    finish(cfg, "prompt_loose", res["prompt_level_loose"], len(items), {
        **res, "truncated_responses": sum(1 for r in rows if r["finish_reason"] == "length"),
        "request_errors": count_errors(resps)})
