#!/usr/bin/env bash
# selftest.sh <bench> <gold|garbage> [LIMIT] [PORT] [MODEL]
# Starts common/stub_server.py on PORT (default 28100+), runs the bench against it into
# ~/evals/text2/selftest_runs/<mode>/<bench>/ and prints the RESULT line. In gold mode the bench writes the
# reference answers to $STUB_GOLD_MAP before sending requests, so a correct pipeline scores ~100%.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
B=$1 MODE=$2 LIM=${3:-20} PORT=${4:-$((28100 + RANDOM % 800))} MODEL=${5:-ornith-ai/Ornith-1.5-122B}
RUNS=$HERE/selftest_runs/$MODE
rm -rf "$RUNS/stub/$B"; mkdir -p "$RUNS"
GM=$RUNS/gold_map_$B.jsonl; rm -f "$GM"; touch "$GM"
"$HERE/.venv/bin/python" "$HERE/common/stub_server.py" --port $PORT --mode $MODE --gold-map "$GM" > "$RUNS/stub_$B.log" 2>&1 &
SP=$!; trap 'kill $SP 2>/dev/null' EXIT
for i in $(seq 50); do curl -sf localhost:$PORT/v1/models >/dev/null && break; sleep 0.2; done
STUB_GOLD_MAP=$([ $MODE = gold ] && echo "$GM" || echo "") RUNS_ROOT=$RUNS LIMIT=$LIM \
  "$HERE/text2_run.sh" "$B" stub "http://127.0.0.1:$PORT/v1" "$MODEL" 8
echo "stub stats: $(curl -s localhost:$PORT/stats)"
