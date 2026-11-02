"""LongBench v2 (THUDM/LongBench-v2, 503 four-choice questions, 8k-2M-word contexts).

Follows the official THUDM/LongBench pred.py (@2e00731) "0shot" setting for a reasoning model:
  prompt = prompts/0shot.txt with $DOC$/$Q$/$C_*$ replaced (.strip()), one user turn;
  middle truncation of the prompt in model tokens: ids[:L//2] + ids[-L//2:], decode(skip_special_tokens=True);
  answer = official extract_answer() ("The correct answer is (X)" after removing '*') on `content`;
  metric = accuracy over all 503 (official result.py; no 0.25 compensation), with easy/hard/short/medium/long.
The model thinks natively, so the separate CoT pass (prompts/0shot_cot*.txt) is not used (LB2_COT=1 enables the
official two-pass CoT protocol instead: CoT pass with max_tokens=MAX_TOKENS, then the answer-extraction pass).
L (LB2_MAX_LEN) defaults to max_model_len (262144) - MAX_TOKENS - 1024 (chat template / safety margin), so the
prompt + full generation budget fits the served context. The official pred.py uses model2maxlen.json (128k for
most models); Qwen's value is not published (inferred).
"""
import os
import re
from pathlib import Path

from datasets import load_dataset
from tokenizers import Tokenizer

from common.core import count_errors, finish, generate_chat, generate_custom, select_items, write_gold_map, write_jsonl

REPO = Path.home() / "evals/src/LongBench"
TOK = Path.home() / "evals/data/tokenizers/Qwen/Qwen3.5-122B-A10B/tokenizer.json"  # identical for all 3 models (md5)
T0 = (REPO / "prompts/0shot.txt").read_text(encoding="utf-8")
T_COT = (REPO / "prompts/0shot_cot.txt").read_text(encoding="utf-8")
T_COT_ANS = (REPO / "prompts/0shot_cot_ans.txt").read_text(encoding="utf-8")


def extract_answer(response):  # verbatim from pred.py
    response = response.replace('*', '')
    match = re.search(r'The correct answer is \(([A-D])\)', response)
    if match:
        return match.group(1)
    else:
        match = re.search(r'The correct answer is ([A-D])', response)
        if match:
            return match.group(1)
        else:
            return None


def fill(t, it, context):
    return (t.replace('$DOC$', context.strip()).replace('$Q$', it['question'].strip())
            .replace('$C_A$', it['choice_A'].strip()).replace('$C_B$', it['choice_B'].strip())
            .replace('$C_C$', it['choice_C'].strip()).replace('$C_D$', it['choice_D'].strip()))


def truncate(tok, prompt, max_len):
    ids = tok.encode(prompt, add_special_tokens=False).ids
    if len(ids) > max_len:
        ids = ids[:max_len // 2] + ids[-max_len // 2:]
        return tok.decode(ids, skip_special_tokens=True), True, len(ids)
    return prompt, False, len(ids)


def main(cfg):
    cot = os.environ.get("LB2_COT", "0") == "1"
    max_len = int(os.environ.get("LB2_MAX_LEN", 262144 - cfg.max_tokens - 1024))
    ds = load_dataset("THUDM/LongBench-v2", split="train")
    items = [{"id": r["_id"], **{k: r[k] for k in ("domain", "sub_domain", "difficulty", "length", "question", "choice_A",
                                                    "choice_B", "choice_C", "choice_D", "answer", "context")}} for r in ds]
    items = select_items(items, cfg)
    cfg.dump(module="longbench_v2", max_len=max_len, cot=cot, prompt="0shot" + ("_cot" if cot else ""))
    tok = Tokenizer.from_file(str(TOK))
    prompts, trunc = {}, {}
    for it in items:
        p, was, n = truncate(tok, fill(T_COT if cot else T0, it, it["context"]), max_len)
        prompts[it["id"]] = p
        trunc[it["id"]] = (was, n)
    print(f"[longbench_v2] truncated prompts: {sum(w for w, _ in trunc.values())}/{len(items)} (max_len {max_len})", flush=True)
    write_gold_map([([{"role": "user", "content": prompts[it["id"]]}], f"The correct answer is ({it['answer']})")
                    for it in items])
    if not cot:
        resps = generate_chat(cfg, [(it["id"], [{"role": "user", "content": prompts[it["id"]]}]) for it in items])
    else:
        by_id = {it["id"]: it for it in items}

        def job(rid):
            async def run(c):
                r1 = await c.chat([{"role": "user", "content": prompts[rid]}])
                if r1.get("error") or not r1.get("content"):
                    return r1
                p2 = fill(T_COT_ANS, by_id[rid], "").replace("$COT$", r1["content"].strip())
                r2 = await c.chat([{"role": "user", "content": p2}])
                r2["cot"] = r1["content"]
                return r2
            return run
        resps = generate_custom(cfg, [(it["id"], job(it["id"])) for it in items])

    rows = []
    for it in items:
        r = resps.get(it["id"], {})
        pred = extract_answer((r.get("content") or "").strip())
        rows.append({"id": it["id"], "difficulty": it["difficulty"], "length": it["length"], "domain": it["domain"],
                     "answer": it["answer"], "pred": pred, "judge": pred == it["answer"], "truncated_input": trunc[it["id"]][0],
                     "prompt_tokens": trunc[it["id"]][1], "finish_reason": r.get("finish_reason"), "error": r.get("error")})
    write_jsonl(cfg.out / "scores.jsonl", rows)

    def acc(sel):
        s = [r for r in rows if sel(r)]
        return round(100 * sum(r["judge"] for r in s) / len(s), 1) if s else None
    overall = 100 * sum(r["judge"] for r in rows) / len(rows)
    summ = {"easy": acc(lambda r: r["difficulty"] == "easy"), "hard": acc(lambda r: r["difficulty"] != "easy"),
            "short": acc(lambda r: r["length"] == "short"), "medium": acc(lambda r: r["length"] == "medium"),
            "long": acc(lambda r: r["length"] == "long"),
            "no_answer_extracted": sum(1 for r in rows if r["pred"] is None),
            "truncated_inputs": sum(1 for r in rows if r["truncated_input"]),
            "truncated_responses": sum(1 for r in rows if r["finish_reason"] == "length"),
            "request_errors": count_errors(resps), "max_len": max_len}
    print("  " + "  ".join(f"{k}={summ[k]}" for k in ("easy", "hard", "short", "medium", "long")))
    finish(cfg, "accuracy", overall, len(rows), summ)
