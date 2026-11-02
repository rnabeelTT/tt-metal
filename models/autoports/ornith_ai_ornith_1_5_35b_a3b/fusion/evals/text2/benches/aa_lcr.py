"""AA-LCR (Artificial Analysis Long Context Reasoning; 100 questions over ~100k-token document sets).

Data: HF ArtificialAnalysis/AA-LCR. Default AA_LCR_VERSION=1.0.0 (revision bdae010; the version current when the
Qwen3.5 card was published, Feb 2026). AA_LCR_VERSION=1.1 uses the Sept-2026 answer-key corrections and the v1.1
judge system prompt (scores of the two versions are not comparable, per the dataset card).
Prompt: the dataset card's template verbatim (documents in data_source_filenames order, "BEGIN DOCUMENT i:" blocks).
Judging: LLM equality checker with the card's prompt (v1.0.0: no system prompt, "Reply only with CORRECT or
INCORRECT."; v1.1: system prompt + "Reply as JSON"). Metric: accuracy = CORRECT / 100.
"""
import csv
import os
import unicodedata
from pathlib import Path

from common.core import count_errors, finish, generate_chat, select_items, write_gold_map, write_jsonl
from common.judge import write_judge_inputs

DATA = Path.home() / "evals/data/aa_lcr"
TEXT = DATA / "text/lcr"

PROMPT = """BEGIN INPUT DOCUMENTS

{documents_text}

END INPUT DOCUMENTS

Answer the following question using the input documents provided above.

START QUESTION

{question}

END QUESTION
"""

JUDGE_USER = """Assess whether the following CANDIDATE ANSWER is CORRECT or INCORRECT.
For the CANDIDATE ANSWER to be correct, it must be consistent with the OFFICIAL ANSWER.

The question, for reference only: START QUESTION {question}

END QUESTION

The OFFICIAL ANSWER: {official_answer}

END OFFICIAL ANSWER

BEGIN CANDIDATE ANSWER TO ASSESS

{candidate_answer}

END CANDIDATE ANSWER TO ASSESS

"""
JUDGE_TAIL = {"1.0.0": "Reply only with CORRECT or INCORRECT.",
              "1.1": "Reply as JSON, with a verdict of CORRECT or INCORRECT."}
JUDGE_SYSTEM_11 = """Decide whether the CANDIDATE ANSWER is correct or incorrect against the OFFICIAL ANSWER.

Note the following points when assessing correctness:

- Numbers should still match when they are the same value written differently, e.g., a
  percentage, a count of percentage points, and the equivalent decimal fraction are the same
  value: 0.675, "67.5%" and "67.5 percentage points" all match. So do different scales
  (thousand, million, bn) and different notations (thousands separators, currency symbols,
  LaTeX markup, and numbers written as words).
- Where the question asks for a particular format (e.g., a percentage, a number of decimal
  places, a unit, a rounding, or an ordering) the CANDIDATE ANSWER must meet it. If the
  question asks for no particular format, accept any equivalent form.
- In cases where the question asks for an ordered list, a title, honorific or article added
  to an entry in the CANDIDATE ANSWER can change where that entry sorts. Accept the ordering
  if it is correct either with those additions or without them.
- Grade the value the CANDIDATE ANSWER finally commits to, and it must commit to one. Values
  reached while working, and alternatives it considers and sets aside, do not count. If it
  offers several values without selecting one, it is incorrect even if one of them is right.
  Hedging is fine as long as one clearly definitive answer is given."""

SCHEMA = {"type": "object", "properties": {"verdict": {"enum": ["CORRECT", "INCORRECT"]}}, "required": ["verdict"]}


def version():
    return os.environ.get("AA_LCR_VERSION", "1.0.0")


def load_items():
    csv_path = DATA / ("v1.0.0/AA-LCR_Dataset.csv" if version() == "1.0.0" else "AA-LCR_Dataset.csv")
    items = []
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            files = row["data_source_filenames"].split(";")
            docs = []
            d = TEXT / row["document_category"] / row["document_set_id"]
            # zip filenames are partly NFD-normalized; match on NFC
            names = {unicodedata.normalize("NFC", x.name): x for x in d.iterdir()}
            for fn in files:
                docs.append(names[unicodedata.normalize("NFC", fn)].read_text(encoding="utf-8"))
            items.append({"id": row["question_id"], "question": row["question"], "answer": row["answer"],
                          "docs": docs, "category": row["document_category"], "input_tokens_cl100k": int(row["input_tokens"])})
    return items


def build_prompt(it):
    documents_text = "\n\n".join(f"BEGIN DOCUMENT {i + 1}:\n{doc}\nEND DOCUMENT {i + 1}" for i, doc in enumerate(it["docs"]))
    return PROMPT.format(documents_text=documents_text, question=it["question"])


def judge_prompt(question, answer, cand):
    v = version()
    user = JUDGE_USER.format(question=question, official_answer=answer, candidate_answer=cand) + JUDGE_TAIL[v]
    if v == "1.0.0":
        return user
    return f"[SYSTEM]\n{JUDGE_SYSTEM_11}\n\n[USER]\n{user}"


def main(cfg):
    items = select_items(load_items(), cfg)
    cfg.dump(module="aa_lcr", aa_lcr_version=version())
    reqs = [(it["id"], [{"role": "user", "content": build_prompt(it)}]) for it in items]
    write_gold_map([(m, it["answer"]) for it, (_, m) in zip(items, reqs)])
    resps = generate_chat(cfg, reqs)
    rows, auto = [], []
    for it in items:
        r = resps.get(it["id"], {})
        ans = r.get("content") or ""
        if not ans.strip():
            auto.append({"id": it["id"], "verdict": "INCORRECT", "reason": f"auto: no response ({r.get('error') or r.get('finish_reason')})"})
            continue
        rows.append({"id": it["id"], "bench": "aa_lcr", "question": it["question"], "reference": it["answer"],
                     "model_answer": ans, "judge_prompt": judge_prompt(it["question"], it["answer"], ans),
                     "verdict_schema": SCHEMA, "meta": {"category": it["category"], "aa_lcr_version": version()}})
    instr = (f"AA-LCR v{version()} equality checker (AA used an OpenAI model as checker; Qwen's checker is not stated). "
             "Send judge_prompt as the user message" + (" with no system prompt" if version() == "1.0.0" else
             " ([SYSTEM] part as system prompt)") + "; return {\"id\", \"verdict\": \"CORRECT\"|\"INCORRECT\"}.")
    p = write_judge_inputs(cfg.out, rows, auto, instr)
    finish(cfg, "accuracy", None, len(items), {
        "judge_items": len(rows), "auto_incorrect": len(auto), "request_errors": count_errors(resps),
        "truncated_responses": sum(1 for r in resps.values() if r.get("finish_reason") == "length"),
        "prompt_tokens": {k: (v.get("usage") or {}).get("prompt_tokens") for k, v in resps.items()}}, judge_pending=p)


def ingest(cfg, verdicts, missing):
    import json
    ids = json.loads((cfg.out / "sample_ids.json").read_text())["ids"]
    rows = [{"id": i, "verdict": verdicts.get(i, {}).get("verdict", "INCORRECT")} for i in ids]
    for r in rows:
        r["correct"] = str(r["verdict"]).strip().upper() == "CORRECT"
    write_jsonl(cfg.out / "scores.jsonl", rows)
    acc = 100 * sum(r["correct"] for r in rows) / len(rows)
    finish(cfg, "accuracy", acc, len(rows), {"missing_judgments": len(missing)})
