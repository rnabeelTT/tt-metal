#!/usr/bin/env bash
# selftest.sh <gold|garbage> <cat1,cat2,...> [PORT 28300-28399] [LIMIT]   BFCL plumbing check against tt_bfcl.stub
set -euo pipefail
cd "$(dirname "$0")"
MODE=$1 CATS=$2 PORT=${3:-28301} LIM=${4:-}
TAG=$MODE-$(echo $CATS | tr ',' '+' | cut -c1-80)
RUNS=$PWD/selftest_runs; rm -rf "$RUNS/$TAG"; mkdir -p "$RUNS"
PYTHONPATH=$PWD BFCL_PROJECT_ROOT=$RUNS/stubproj .venv/bin/python -m tt_bfcl.stub --port $PORT --mode $MODE --categories $CATS > "$RUNS/stub_$PORT.log" 2>&1 &
SP=$!; trap 'kill $SP 2>/dev/null' EXIT
for i in $(seq 300); do curl -sf localhost:$PORT/v1/models >/dev/null && break; sleep 0.5; done
RUNS_ROOT=$RUNS BFCL_CATEGORIES=$CATS BFCL_PER_CAT=0 LIMIT=$LIM ~/evals/text2_run.sh bfcl_v4 $TAG http://127.0.0.1:$PORT/v1 ornith-ai/Ornith-1.5-122B 16 2>&1 | grep -E "^bfcl_v4|^RESULT|Error|error|Traceback|WARNING" | grep -v "^bfcl_v4 official" | head -60
echo "stub stats: $(curl -s localhost:$PORT/stats)"
