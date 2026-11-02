"""Humanity's Last Exam, no tools ("HLE w/ CoT" on the Qwen card; "HLE (no tools)" on the Ornith cards).

Data: gated HF dataset cais/hle (test, 2,500 questions, ~14% with an image), read from the local parquet
~/evals/data/hle/data/test-*.parquet (`hf download cais/hle --repo-type dataset --local-dir ~/evals/data/hle`
after accepting the terms on the dataset page) or HLE_PARQUET=<file>. HLE_TEXT_ONLY=1 drops image questions
(not the card setting; all three models are multimodal so the default is the full set).
Generation: official hle_eval/run_model_predictions.py (@22ed307): system prompt SYSTEM_PROMPT, user content
[text, image_url(data URI)] ; thinking on, `content` only.
Judging: official run_judge_results.py JUDGE_PROMPT, structured verdict {extracted_final_answer, reasoning,
correct: yes|no, confidence, strict}; official judge o3-mini-2025-01-31; Ornith cards: Claude 4.6 Opus.
Metric: accuracy = #correct / #questions (missing predictions count wrong, as official), + RMS calibration error
(beta=100) and the Wald 95% half-width.
"""
import glob
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

from common.core import count_errors, finish, generate_chat, select_items, write_gold_map, write_jsonl
from common.judge import write_judge_inputs

DATA = Path.home() / "evals/data/hle/data"

SYSTEM_PROMPT = "Your response should be in the following format:\nExplanation: {your explanation for your answer choice}\nAnswer: {your chosen answer}\nConfidence: {your confidence score between 0% and 100% for your answer}"

JUDGE_PROMPT = """Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.


confidence: The extracted confidence score between 0|\\%| and 100|\\%| from [response]. Put 100 if there is no confidence score available."""

SCHEMA = {"type": "object", "properties": {
    "extracted_final_answer": {"type": "string"}, "reasoning": {"type": "string"},
    "correct": {"enum": ["yes", "no"]}, "confidence": {"type": "integer", "minimum": 0, "maximum": 100},
    "strict": {"const": True}}, "required": ["extracted_final_answer", "reasoning", "correct", "confidence"]}

INSTRUCTIONS = """HLE judging (official: o3-mini-2025-01-31 with structured output ExtractedAnswer; Ornith cards: Claude 4.6
Opus). Send judge_prompt as the only user message; return {"id", "extracted_final_answer", "reasoning",
"correct": "yes"|"no", "confidence": 0-100, "strict": true}. The judge sees the question text only (no image),
as in the official script."""


def load_items():
    src = os.environ.get("HLE_PARQUET")
    files = [src] if src else sorted(glob.glob(str(DATA / "test-*.parquet")))
    if not files:
        raise SystemExit("HLE data missing: accept the terms at https://huggingface.co/datasets/cais/hle, then "
                         "`hf download cais/hle --repo-type dataset --local-dir ~/evals/data/hle`")
    df = pd.concat([pd.read_parquet(f, columns=["id", "question", "image", "answer", "answer_type", "category"]) for f in files])
    items = df.to_dict("records")
    if os.environ.get("HLE_TEXT_ONLY", "0") == "1":
        items = [it for it in items if not it["image"]]
    return items


def messages(it):
    content = [{"type": "text", "text": it["question"]}]
    if it["image"]:
        content.append({"type": "image_url", "image_url": {"url": it["image"]}})
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": content}]


def main(cfg):
    all_items = load_items()
    items = select_items(all_items, cfg, force_sample=int(os.environ["SAMPLE_N"]) if os.environ.get("SAMPLE_N") else None)
    cfg.dump(module="hle", text_only=os.environ.get("HLE_TEXT_ONLY", "0") == "1", n_total=len(all_items))
    write_gold_map([(messages(it), f"Explanation: stub.\nAnswer: {it['answer']}\nConfidence: 90%") for it in items])
    resps = generate_chat(cfg, [(it["id"], messages(it)) for it in items])
    rows, auto = [], []
    for it in items:
        r = resps.get(it["id"], {})
        ans = r.get("content") or ""
        meta = {"answer_type": it["answer_type"], "category": it["category"], "has_image": bool(it["image"])}
        if not ans.strip():
            # official: no prediction -> not judged -> counted wrong in accuracy, excluded from calibration
            auto.append({"id": it["id"], "correct": "no", "confidence": None, "auto": True,
                         "reasoning": f"auto: no response ({r.get('error') or r.get('finish_reason')})"})
            continue
        rows.append({"id": it["id"], "bench": "hle", "question": it["question"], "reference": it["answer"],
                     "model_answer": ans, "judge_prompt": JUDGE_PROMPT.format(question=it["question"], correct_answer=it["answer"], response=ans),
                     "verdict_schema": SCHEMA, "meta": meta})
    p = write_judge_inputs(cfg.out, rows, auto, INSTRUCTIONS)
    finish(cfg, "accuracy", None, len(items), {"judge_items": len(rows), "auto_incorrect": len(auto),
                                                "request_errors": count_errors(resps),
                                                "truncated_responses": sum(1 for r in resps.values() if r.get("finish_reason") == "length")},
           judge_pending=p)


def calib_err(confidence, correct, p='2', beta=100):  # verbatim logic from run_judge_results.py
    idxs = np.argsort(confidence)
    confidence = confidence[idxs]
    correct = correct[idxs]
    bins = [[i * beta, (i + 1) * beta] for i in range(len(confidence) // beta)]
    if not bins:
        return float("nan")
    bins[-1] = [bins[-1][0], len(confidence)]
    cerr = 0
    total_examples = len(confidence)
    for i in range(len(bins) - 1):
        bin_confidence = confidence[bins[i][0]:bins[i][1]]
        bin_correct = correct[bins[i][0]:bins[i][1]]
        num_examples_in_bin = len(bin_confidence)
        if num_examples_in_bin > 0:
            difference = np.abs(np.nanmean(bin_confidence) - np.nanmean(bin_correct))
            cerr += num_examples_in_bin / total_examples * np.square(difference)
    return np.sqrt(cerr)


def ingest(cfg, verdicts, missing):
    ids = json.loads((cfg.out / "sample_ids.json").read_text())["ids"]
    n = len(ids)
    rows, correct, conf = [], [], []
    for i in ids:
        v = verdicts.get(i)
        ok = bool(v) and "yes" in str(v.get("correct", "")).lower()
        rows.append({"id": i, "correct": ok, "confidence": v.get("confidence") if v else None, "judged": bool(v) and not v.get("auto")})
        if v and not v.get("auto"):
            correct.append(ok)
            conf.append(float(v.get("confidence", 100)))
    write_jsonl(cfg.out / "scores.jsonl", rows)
    acc = round(100 * sum(r["correct"] for r in rows) / n, 2)
    hw = round(1.96 * math.sqrt(acc * (100 - acc) / n), 2)
    ce = 100 * round(float(calib_err(np.array(conf) / 100, np.array(correct))), 2) if conf else None
    print(f"*** Metrics ***\nAccuracy: {acc}% +/- {hw}% | n = {n}\nCalibration Error: {ce}")
    finish(cfg, "accuracy", acc, n, {"ci95_half_width": hw, "calibration_error": ce, "judged": len(correct),
                                     "missing_judgments": len(missing)})
