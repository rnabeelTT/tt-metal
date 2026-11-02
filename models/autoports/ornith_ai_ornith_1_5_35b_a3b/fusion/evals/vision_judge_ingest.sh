#!/usr/bin/env bash
# Ingest LLM-judge verdicts and re-score with the official metric.
#   vision_judge_ingest.sh <bench> <label> [judgments.jsonl]
# judgments.jsonl: one {"id": "<id from judge_inputs.jsonl>", "verdict": "<raw judge reply>"} per line.
# Verdicts are merged into <out>/judge/judgments.jsonl (later lines win), then the official
# VLMEvalKit scorer runs again; prompts still unanswered (or new multi-stage prompts) are
# written to judge_inputs.jsonl and reported as JUDGE_PENDING.
set -euo pipefail
[ $# -ge 2 ] || { sed -n '2,8p' "$0"; exit 2; }
BENCH=$1 LABEL=$2 SRC=${3:-}
OUT=${RUNS_ROOT:-$HOME/ornith-fusion/runs}/$LABEL/$BENCH
[ -d "$OUT" ] || { echo "no run at $OUT"; exit 1; }
mkdir -p "$OUT/judge"
if [ -n "$SRC" ] && [ "$(realpath "$SRC")" != "$(realpath -m "$OUT/judge/judgments.jsonl")" ]; then
  $HOME/evals/vision/.venv/bin/python - "$SRC" "$OUT/judge/judgments.jsonl" <<'PY'
import json, sys, os
src, dst = sys.argv[1], sys.argv[2]
have = {}
if os.path.exists(dst):
    for l in open(dst):
        if l.strip():
            r = json.loads(l); have[r['id']] = r
n = 0
for l in open(src):
    if l.strip():
        r = json.loads(l)
        assert 'id' in r and 'verdict' in r, f'bad line: {l[:200]}'
        have[r['id']] = {'id': r['id'], 'verdict': str(r['verdict'])}; n += 1
with open(dst, 'w') as f:
    for r in have.values():
        f.write(json.dumps(r, ensure_ascii=False) + '\n')
print(f'merged {n} verdicts -> {dst} ({len(have)} total)')
PY
fi
PHASE=score exec $HOME/evals/vision_run.sh "$BENCH" "$LABEL" http://unused.invalid/v1 unused 1
