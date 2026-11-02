#!/usr/bin/env bash
# Qwen3.5-122B-A10B model-card language benchmarks against an OpenAI-compatible server.
#
# Usage:
#   qwen_text_run.sh <bench> <label> <base_url> <model_id> <concurrency>
#   e.g. qwen_text_run.sh mmlu_pro ref_q122 http://localhost:20000/v1 Qwen/Qwen3.5-122B-A10B 32
#
# bench: mmlu_pro mmlu_redux ceval supergpqa mmmlu mmlu_prox include global_piqa
#        polymath wmt24pp maxife nova63            (card value in brackets in README)
#
# Results: ${RUNS_ROOT:-~/ornith-fusion/runs}/<label>/<bench>/
#   config.json, responses.jsonl (content + reasoning per item; appended as they finish),
#   scored.jsonl, summary.json, run.log
# Last stdout line: RESULT <bench> <metric>=<value> n=<items> ...
#   (+ JUDGE_PENDING <path> for maxife: LLM-judge inputs to be judged offline, then
#    `qwen_text_judge.sh maxife <run dir> <judgments.jsonl>`)
#   (+ SCORE_PENDING <cmd> for wmt24pp when XCOMET scoring is deferred, WMT_SCORE=0)
# Rerunning the same command resumes: only items without a successful response are sent.
# Items are frozen in ~/evals/qwen_text/data/<bench>/ (items.jsonl, sample_ids.txt; seed 1234).
#
# Env:
#   MAX_TOKENS=32768 (polymath: 81920, Qwen's setting for PolyMATH)
#   TEMPERATURE=0.6 TOP_P=0.95 TOP_K=20   checkpoint defaults; TOP_K sent as top-level `top_k`
#   PRESENCE_PENALTY=                     unset = not sent (card's "thinking general" preset uses 1.5 with T=1.0)
#   LIMIT=                                N items only (smoke test; touches as many languages as possible)
#   REQUEST_TIMEOUT=14400 MAX_RETRIES=5
#   WMT_SCORE=1                           wmt24pp: run XCOMET on CPU after generation (0 = defer)
#   COMET_MODEL=Unbabel/XCOMET-XXL COMET_THREADS=8 COMET_PRECISION=bf16
#   SKIP_SERVER_CHECK=0                   1 = do not query <base_url>/models first
#   RUNS_ROOT=~/ornith-fusion/runs
set -euo pipefail
if [ $# -ne 5 ]; then sed -n '2,32p' "$0"; exit 2; fi
BENCH=$1 LABEL=$2 BASE_URL=${3%/} MODEL_ID=$4 CONC=$5
EVALS=$(cd "$(dirname "$0")" && pwd)
H=$EVALS/qwen_text
OUT=${RUNS_ROOT:-$HOME/ornith-fusion/runs}/$LABEL/$BENCH

if [ "$BENCH" = nova63 ]; then
  echo "nova63: NOVA-63 (EMNLP 2025, Qwen team co-authors) is not publicly released (HF zjy1298/NOVA-63 holds only a README: 'currently under review'); nothing to run." >&2
  echo "RESULT nova63 acc=NA n=0 reason=dataset_not_public"
  exit 0
fi
[ -x "$H/.venv/bin/python" ] || { echo "missing venv $H/.venv"; exit 1; }
[ -f "$H/data/$BENCH/items.jsonl" ] || { echo "no frozen items for $BENCH; run: cd $H && .venv/bin/python -m qt.prepare $BENCH"; exit 1; }

if [ "${SKIP_SERVER_CHECK:-0}" != 1 ]; then
  M=$(curl -sf --max-time 20 "$BASE_URL/models") || { echo "server not reachable: $BASE_URL/models"; exit 1; }
  python3 -c 'import json,sys; d=json.loads(sys.argv[1])["data"]; m=[x for x in d if x["id"]==sys.argv[2]]; sys.exit("model %s not served; have %s" % (sys.argv[2], [x["id"] for x in d])) if not m else print("served max_model_len:", m[0].get("max_model_len"), file=sys.stderr)' "$M" "$MODEL_ID"
fi

if [ "$BENCH" = polymath ]; then export MAX_TOKENS=${MAX_TOKENS:-81920}; fi
mkdir -p "$OUT"
cd "$H"
exec .venv/bin/python -m qt.run "$BENCH" --out "$OUT" --base-url "$BASE_URL" --model "$MODEL_ID" --conc "$CONC"
