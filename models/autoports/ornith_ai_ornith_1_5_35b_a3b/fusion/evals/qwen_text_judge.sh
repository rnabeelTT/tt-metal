#!/usr/bin/env bash
# Ingest offline LLM-judge verdicts and print the final RESULT line.
#   qwen_text_judge.sh <bench> <run_dir> <judgments.jsonl>
# judgments.jsonl: one {"id": <id from judge_inputs.jsonl>, "verdict": <judge output>} per line.
# For maxife the verdict is the judge's raw reply; per the official parser it must be
# 0, 0.7 or 1 (anything else counts 0). Unknown ids are rejected; missing ids keep the
# result pending.
set -euo pipefail
[ $# -eq 3 ] || { sed -n '2,7p' "$0"; exit 2; }
BENCH=$1 RUN=$(cd "$2" && pwd) J=$3
H=$(cd "$(dirname "$0")" && pwd)/qwen_text
[ -f "$RUN/judge_inputs.jsonl" ] || { echo "no $RUN/judge_inputs.jsonl"; exit 1; }
python3 - "$RUN/judge_inputs.jsonl" "$J" "$RUN/judgments.jsonl" <<'PY'
import json, sys
want = {json.loads(l)["id"] for l in open(sys.argv[1]) if l.strip()}
got = {}
for l in open(sys.argv[2]):
    if l.strip():
        d = json.loads(l)
        if d["id"] not in want:
            sys.exit(f"unknown judgment id {d['id']}")
        got[d["id"]] = d["verdict"]
with open(sys.argv[3], "w") as f:
    for k, v in got.items():
        f.write(json.dumps({"id": k, "verdict": v}) + "\n")
print(f"{len(got)}/{len(want)} judgments ingested", file=sys.stderr)
PY
cd "$H" && exec .venv/bin/python -m qt.run "$BENCH" --out "$RUN" --score-only
