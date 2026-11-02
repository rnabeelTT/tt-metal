#!/usr/bin/env bash
# Offline harness test with a local stub server (no model):
#   selftest.sh <bench> [limit]   -> gold run (expect ~100%) and garbage run (expect ~chance)
# Writes to ~/evals/vision/selftest/<bench>/{gold,garbage}; port 28xxx.
set -uo pipefail
B=$1; LIM=${2:-40}
V=$(cd "$(dirname "$0")" && pwd)
PY=$V/.venv/bin/python
T=$V/selftest/$B; rm -rf "$T"; mkdir -p "$T"
PORT=$(( 28000 + RANDOM % 900 ))
while ss -ltn 2>/dev/null | grep -q ":$PORT "; do PORT=$(( 28000 + RANDOM % 900 )); done
cd "$V"
LIMIT=$LIM $PY -m vtt.run "$B" --out "$T/keys" --write-stub-key "$T/keys.jsonl" 2>&1 | grep -v -E "load_env|^\s*$" | tail -3
for MODE in gold garbage; do
  $PY -I vtt/stub_server.py --port $PORT --keys "$T/keys.jsonl" --mode $MODE > "$T/stub_$MODE.log" 2>&1 &
  SP=$!; sleep 1
  LIMIT=$LIM MAX_TOKENS=64 $PY -m vtt.run "$B" --out "$T/$MODE" --base-url http://127.0.0.1:$PORT/v1 --model Qwen/Qwen3.5-122B-A10B --concurrency 8 > "$T/$MODE.log" 2>&1
  echo "[$MODE] stub $(curl -s http://127.0.0.1:$PORT/v1/stats)"; grep -E "^(RESULT|JUDGE_PENDING)|Traceback|Error" "$T/$MODE.log" | tail -4
  kill $SP; wait $SP 2>/dev/null
  if grep -q '^JUDGE_PENDING' "$T/$MODE.log"; then   # simulate a perfect judge, ingest, re-score
    $PY -m vtt.simjudge "$B" "$T/$MODE" $MODE 2>&1 | grep -v load_env | tail -1
    $PY -m vtt.run "$B" --out "$T/$MODE" --phase score > "$T/${MODE}_rescore.log" 2>&1
    echo "[$MODE after simulated judge]"; grep -E "^(RESULT|JUDGE_PENDING)|Traceback|Error" "$T/${MODE}_rescore.log" | tail -3
  fi
done
