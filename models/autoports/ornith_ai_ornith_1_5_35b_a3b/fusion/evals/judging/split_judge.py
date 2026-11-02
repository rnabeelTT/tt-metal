"""split_judge.py <judge_inputs.jsonl> <workdir> [--chunk 100]: split judge inputs into chunk_NNN.jsonl files
(only id, judge_prompt, verdict_schema kept: the judge works from the official prompt alone)."""
import json, sys, argparse
from pathlib import Path
ap = argparse.ArgumentParser(); ap.add_argument("inp"); ap.add_argument("work"); ap.add_argument("--chunk", type=int, default=100)
a = ap.parse_args()
w = Path(a.work); w.mkdir(parents=True, exist_ok=True)
rows = [json.loads(l) for l in open(a.inp)]
for k in range(0, len(rows), a.chunk):
    with open(w / f"chunk_{k // a.chunk:03d}.jsonl", "w") as f:
        for r in rows[k:k + a.chunk]:
            f.write(json.dumps({"id": r["id"], "judge_prompt": r["judge_prompt"], "verdict_schema": r["verdict_schema"]}) + "\n")
print(f"{len(rows)} items -> {(len(rows) + a.chunk - 1) // a.chunk} chunks in {w}")
