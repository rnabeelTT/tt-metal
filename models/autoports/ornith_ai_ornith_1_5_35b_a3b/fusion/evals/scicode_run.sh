#!/usr/bin/env bash
# SciCode (official harness, inspect_ai port) against an OpenAI-compatible server.
#
# Usage:
#   scicode_run.sh <label> <base_url> <model_id> <concurrency> <max_tokens>
#   e.g. scicode_run.sh ornith122b-ref http://localhost:20000/v1 ornith-ai/Ornith-1.5-122B 32 32768
#
# Default variant = test split (65 main problems, 288 generated sub-steps), WITH
# scientist-annotated background (the Artificial Analysis setting; AA reports
# sub-problem pass@1, which is what model cards quote as "SciCode").
# Final line starts with SCICODE: subproblem acc (headline), main-problem acc, plus
# the same results under the inspect_ai port's denominator (291) for reference.
#
# Results: ~/ornith-fusion/runs/<label>/scicode/
#   eval_results/<model>/generated_code|prompt|evaluation_logs/<bg>/  per-step files
#   inspect_logs/*.eval   full transcripts (reasoning + content; `inspect view`)
#   run.log, summary.json
#
# Env overrides:
#   BACKGROUND=1        0 = without background (the paper's "standard" setting)
#   SPLIT=test          validation = 15 problems / 50 steps (dev set)
#   TEMPERATURE=0.6 TOP_P=0.95 TOP_K=20   (TOP_K sent as extra_body.top_k)
#   LIMIT=              N problems only, smoke test (samples run longest chain first)
#   MODE=normal         dummy|gold: no model calls (gold only with SPLIT=validation)
#   SCORE_CONCURRENCY=8 problems whose tests run at once (CPU side)
#   REQUEST_TIMEOUT=14400  per-request HTTP read timeout, s (non-streaming)
#   MAX_RETRIES=10      inspect retries (openai SDK adds 2 of its own per attempt)
#   PREFILLED_VERBATIM=0  1 = inject curated code for pre-filled steps 13.6/62.1/76.3
#                         verbatim (fixes upstream #49/#59; official harness truncates
#                         the 13.6/62.1 classes to a bare __init__). Off = official.
#   RUNS_ROOT=~/ornith-fusion/runs   (self-tests write elsewhere)
set -euo pipefail

if [ $# -ne 5 ]; then
  sed -n '2,33p' "$0"; exit 2
fi
LABEL=$1 BASE_URL=${2%/} MODEL_ID=$3 CONC=$4 MAX_TOKENS=$5

EVALS=$(cd "$(dirname "$0")" && pwd)
REPO=$EVALS/SciCode
SUPPORT=$EVALS/scicode_support
OUT=${RUNS_ROOT:-$HOME/ornith-fusion/runs}/$LABEL/scicode
BACKGROUND=${BACKGROUND:-1} SPLIT=${SPLIT:-test} MODE=${MODE:-normal}
TEMPERATURE=${TEMPERATURE:-0.6} TOP_P=${TOP_P:-0.95} TOP_K=${TOP_K:-20}
H5=$REPO/eval/data/test_data.h5

[ -f "$H5" ] || { echo "missing $H5"; exit 1; }
[ -x "$REPO/.venv/bin/inspect" ] || { echo "missing harness venv $REPO/.venv"; exit 1; }
[ -x "$REPO/.venv-exec/bin/python" ] || { echo "missing exec venv $REPO/.venv-exec"; exit 1; }
if [ -e "$OUT/eval_results" ]; then
  # The scorer reuses existing per-step pass/fail logs, so never mix runs.
  echo "prior run exists in $OUT; use a new label (or remove $OUT)"; exit 1
fi

if [ "$MODE" = normal ]; then
  MODELS_JSON=$(curl -sf --max-time 20 "$BASE_URL/models") || { echo "server not reachable: $BASE_URL/models"; exit 1; }
  python3 -c 'import json,sys; d=json.loads(sys.argv[1])["data"]; m=[x for x in d if x["id"]==sys.argv[2]]; sys.exit("model %s not served; have %s" % (sys.argv[2], [x["id"] for x in d])) if not m else print("served max_model_len:", m[0].get("max_model_len"))' "$MODELS_JSON" "$MODEL_ID"
fi

mkdir -p "$OUT/work" "$OUT/inspect_logs"
# The official module reads ../data relative to cwd (templates + pre-filled steps).
ln -sfn "$REPO/eval/data" "$OUT/data"
cp "$SUPPORT/scicode_tt.py" "$OUT/work/scicode_tt.py"   # inspect wants a relative task path

# Test scripts run as `python <step>.py` via PATH: put the pinned exec venv first
# (numpy 1.26 / scipy 1.13; scipy>=1.14 removed integrate.simps, which problems 2
# and 28 import). One BLAS thread per test process; tests run in parallel.
export PATH="$REPO/.venv-exec/bin:$PATH"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export SCICODE_PREFILLED_VERBATIM=${PREFILLED_VERBATIM:-0}
export SCICODE_REPO=$REPO SCICODE_SCORE_CONCURRENCY=${SCORE_CONCURRENCY:-8}
export TT_BASE_URL=$BASE_URL TT_API_KEY=${OPENAI_API_KEY:-EMPTY}
export INSPECT_HTTP_REQUEST_TIMEOUT=${REQUEST_TIMEOUT:-14400}
export INSPECT_LOG_DIR=$OUT/inspect_logs

BG_ARG=$([ "$BACKGROUND" = 1 ] && echo True || echo False)
ARGS=(
  scicode_tt.py
  --model "openai-api/tt/$MODEL_ID"
  -T split="$SPLIT" -T with_background="$BG_ARG" -T mode="$MODE"
  -T output_dir="$OUT/eval_results" -T h5py_file="$H5"
  --max-connections "$CONC" --max-samples "$CONC"
  --max-tokens "$MAX_TOKENS" --temperature "$TEMPERATURE" --top-p "$TOP_P"
  --extra-body "{\"top_k\": $TOP_K}"
  --max-retries "${MAX_RETRIES:-10}"
  --no-fail-on-error --log-dir "$OUT/inspect_logs" --display plain
)
[ -n "${LIMIT:-}" ] && ARGS+=(--limit "$LIMIT")

echo "scicode: label=$LABEL model=$MODEL_ID base=$BASE_URL split=$SPLIT background=$BACKGROUND prefilled_verbatim=${PREFILLED_VERBATIM:-0} mode=$MODE conc=$CONC max_tokens=$MAX_TOKENS temp=$TEMPERATURE top_p=$TOP_P top_k=$TOP_K out=$OUT" | tee "$OUT/run.log"
START=$(date +%s)
( cd "$OUT/work" && "$REPO/.venv/bin/inspect" eval "${ARGS[@]}" ) 2>&1 | tee -a "$OUT/run.log"
echo "wall_s=$(( $(date +%s) - START ))" | tee -a "$OUT/run.log"

"$REPO/.venv/bin/python" "$SUPPORT/summarize.py" "$OUT" "$SPLIT" "$BACKGROUND" "$OUT/inspect_logs" "${LIMIT:-}" | tee -a "$OUT/run.log" | tail -1
