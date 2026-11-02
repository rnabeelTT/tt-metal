#!/usr/bin/env bash
# Image benchmarks from the Qwen3.5-122B-A10B card, scored "the way the card does" as far as
# the published protocol allows. Harness: ~/evals/vision (README.md there).
#
# Usage: vision_run.sh <bench> <label> <base_url> <model_id> <concurrency>
#   e.g. vision_run.sh mmmu ornith122b http://localhost:20000/v1 Qwen/Qwen3.5-122B-A10B 32
#        vision_run.sh list                    # benchmarks, card figures, request counts
# Output: ~/ornith-fusion/runs/<label>/<bench>/
#   responses.jsonl (resumable cache: rerun the same command to retry failures / continue),
#   sample_ids.json, run_config.json, score/ (VLMEvalKit scorer artefacts), result.json, run.log
# Final line: RESULT <bench> <metric>=<value> n=<items> errors= truncated= empty= judge_pending= status= card=
# If an LLM judge is needed (official protocol), the line before it is
#   JUDGE_PENDING <out>/judge/judge_inputs.jsonl
# then write verdicts to <out>/judge/judgments.jsonl ({"id":..., "verdict":...} per line) and run
#   ~/evals/vision_judge_ingest.sh <bench> <label> [judgments.jsonl]
# which re-scores with the official metric (status=final once nothing is pending).
#
# Env: MAX_TOKENS=32768 TEMPERATURE=0.6 TOP_P=0.95 TOP_K=20 (sent as top_k) PRESENCE_PENALTY=(unset)
#      LIMIT= (first N selected items per sub-dataset; smoke test)  REQUEST_TIMEOUT=14400
#      RUNS_ROOT=~/ornith-fusion/runs  PHASE=all|infer|score  KEEP_REASONING=0 (1 = store reasoning text)
set -euo pipefail
V=$HOME/evals/vision
PY=$V/.venv/bin/python
if [ "${1:-}" = list ]; then
  cd "$V" && exec $PY -m vtt.list_benches
fi
if [ $# -ne 5 ]; then sed -n '2,25p' "$0"; exit 2; fi
BENCH=$1 LABEL=$2 BASE_URL=${3%/} MODEL_ID=$4 CONC=$5
OUT=${RUNS_ROOT:-$HOME/ornith-fusion/runs}/$LABEL/$BENCH
mkdir -p "$OUT"
if [ "${PHASE:-all}" != score ]; then
  MODELS_JSON=$(curl -sf --max-time 20 "$BASE_URL/models") || { echo "server not reachable: $BASE_URL/models"; exit 1; }
  $PY -c 'import json,sys; d=json.loads(sys.argv[1])["data"]; m=[x for x in d if x["id"]==sys.argv[2]]; sys.exit("model %s not served; have %s" % (sys.argv[2], [x["id"] for x in d])) if not m else print("served max_model_len:", m[0].get("max_model_len"))' "$MODELS_JSON" "$MODEL_ID"
fi
echo "vision_run: bench=$BENCH label=$LABEL model=$MODEL_ID base=$BASE_URL conc=$CONC max_tokens=${MAX_TOKENS:-32768} temp=${TEMPERATURE:-0.6} top_p=${TOP_P:-0.95} top_k=${TOP_K:-20} presence_penalty=${PRESENCE_PENALTY:-unset} limit=${LIMIT:-none} out=$OUT" | tee -a "$OUT/run.log"
cd "$V"
set +e
$PY -m vtt.run "$BENCH" --out "$OUT" --base-url "$BASE_URL" --model "$MODEL_ID" --concurrency "$CONC" --phase "${PHASE:-all}" 2>&1 \
  | grep --line-buffered -v -E "load_env - 20: Did not detect the .env|GPT API failed to answer|^\s*$" | tee -a "$OUT/run.log"
rc=${PIPESTATUS[0]}
exit $rc
