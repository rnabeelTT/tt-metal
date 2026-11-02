"""Judge-input / judgment-ingest conventions.

judge_inputs.jsonl (one line per item that needs an LLM verdict):
  {"id", "bench", "question", "reference", "model_answer", "judge_prompt", "verdict_schema", "meta"}
  judge_prompt  = the benchmark's official judge prompt with every slot filled in (send it as the single user turn)
  verdict_schema = JSON schema of the object the judge must return (one JSON object per id)

judgments.jsonl (written by the judging agents):
  {"id": <same id>, <fields of verdict_schema>}

Items with no usable model answer (request error / empty content) are auto-verdicted as incorrect and are NOT
put in judge_inputs.jsonl; they are listed in auto_verdicts.jsonl and counted in the denominator.

Ingest:  ~/evals/text2/.venv/bin/python ~/evals/text2/judge_ingest.py <run_dir> <judgments.jsonl>
"""
from __future__ import annotations

import json
from pathlib import Path

from .core import read_jsonl, write_jsonl


def write_judge_inputs(out: Path, rows: list[dict], auto: list[dict], instructions: str) -> Path:
    p = out / "judge_inputs.jsonl"
    write_jsonl(p, rows)
    write_jsonl(out / "auto_verdicts.jsonl", auto)
    (out / "JUDGE_README.txt").write_text(
        instructions.strip()
        + f"\n\n{len(rows)} items to judge in {p}; {len(auto)} auto-verdicted (no answer) in auto_verdicts.jsonl.\n"
        "Write one JSON object per line to judgments.jsonl: {\"id\": ..., <verdict_schema fields>}.\n"
        "Then: ~/evals/text2/.venv/bin/python ~/evals/text2/judge_ingest.py " + str(out) + " <judgments.jsonl>\n"
    )
    return p


def load_verdicts(out: Path, judgments: Path) -> dict[str, dict]:
    v = {str(r["id"]): r for r in read_jsonl(out / "auto_verdicts.jsonl")}
    pending = {str(r["id"]) for r in read_jsonl(out / "judge_inputs.jsonl")}
    got = {str(r["id"]): r for r in read_jsonl(judgments)}
    missing = sorted(pending - set(got))
    extra = sorted(set(got) - pending)
    if extra:
        print(f"note: {len(extra)} judgments for ids not in judge_inputs (ignored), e.g. {extra[:3]}")
    v.update({k: got[k] for k in pending if k in got})
    return v, missing
