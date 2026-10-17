#!/usr/bin/env bash
# Stage 1 Tracy windows: warmed prefill (128 / 2048 rows) and traced decode (batch 1 / 32) per layer kind, one die.
# Output: $MODEL_DIR/doc/functional_decoder/tracy/<case>/{ops.csv, perf_report.csv, perf_report.txt, console.log}
source "$HOME/glm53-flash-bringup/env.sh"; cd "$TT_METAL_HOME" || exit 1
export TT_METAL_LOGGER_LEVEL=Error
F=$MODEL_DIR/tests/test_perf_decoder.py; OUT=${TRACY_OUT:-$MODEL_DIR/doc/functional_decoder/tracy}
exec 9>"$HOME/qwen3-coder-next-bringup/.device.lock"; flock 9
reset() { for i in 1 2 3; do timeout 300 env -u PYTHONPATH -u VIRTUAL_ENV /home/raahem/.tenstorrent-venv/bin/tt-smi -r >/dev/null 2>&1 && { sleep 5; return 0; }; sleep 20; done; echo "reset FAILED"; return 1; }
for case in ${CASES:-$(cd $TT_METAL_HOME && pytest --collect-only -q $F 2>/dev/null | grep "::" | sed 's/.*:://')}; do
  name=$(echo "$case" | tr '[]' '__' | tr -d ' '); out=$OUT/$name; mkdir -p $out; reset
  echo "=== tracy $case $(date -Is)"
  rm -rf generated/profiler/reports 2>/dev/null
  timeout 2400 python -m tracy -r -p -v -m pytest "$F::$case" -s -p no:cacheprovider > $out/console.log 2>&1; echo "tracy $case exit=$?"
  csv=$(ls -t generated/profiler/reports/*/ops_perf_results_*.csv 2>/dev/null | head -1)
  [ -z "$csv" ] && { echo "no ops csv $case"; continue; }
  cp "$csv" $out/ops.csv
  case $case in *prefill*) sp=PERF_PREFILL;; *) sp=PERF_DECODE;; esac
  tt-perf-report $out/ops.csv --start-signpost $sp --end-signpost ${sp}_END --csv $out/perf_report.csv --no-advice > $out/perf_report.console.log 2>&1
  tt-perf-report $out/ops.csv --start-signpost $sp --end-signpost ${sp}_END --no-summary --no-advice > $out/perf_report.txt 2>&1
  grep -hE "^PERF " $out/console.log | tail -1
done
reset; echo "=== done $(date -Is)"
