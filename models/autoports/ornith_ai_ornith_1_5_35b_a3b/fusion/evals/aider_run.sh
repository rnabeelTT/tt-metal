#!/usr/bin/env bash
# Aider polyglot benchmark (225 Exercism exercises: cpp/go/java/js/python/rust)
# against an OpenAI-compatible server, run inside the aider-benchmark container.
#
# Usage:
#   aider_run.sh <label> <base_url> <model_id> <threads> <edit_format>
#   e.g. aider_run.sh coder-next-ref http://localhost:20000/v1 Qwen/Qwen3-Coder-Next 32 diff
#
# Results: ~/ornith-fusion/runs/<label>/aider/<date>--<label>/ (per-exercise
# .aider.results.json + .aider.chat.history.md), stats.yaml, run.log,
# model_settings.yml, model_metadata.json. Final line starts with "AIDER".
#
# Env overrides (defaults = Qwen3-Coder-Next checkpoint sampling):
#   TEMPERATURE=1.0 TOP_P=0.95 TOP_K=40   sampling sent on every request
#   MAX_TOKENS=                           per-request max_tokens (unset = server default)
#   TRIES=2                               standard aider protocol (pass_rate_1 / pass_rate_2)
#   MAX_INPUT_TOKENS=                     context window told to aider (default: server
#                                         max_model_len from /v1/models, else 262144)
#   NUM_TESTS=-1 KEYWORDS= LANGUAGES=     subset runs (smoke tests only; full run = 225)
#   CONT=1                                resume an existing <date>--<label> dir
#   DOCKER_MEMORY=48g AIDER_IMAGE=aider-benchmark:jest29 (upstream image + jest pinned to 29)
set -euo pipefail

if [ $# -ne 5 ]; then
  sed -n '2,24p' "$0"; exit 2
fi
LABEL=$1 BASE_URL=${2%/} MODEL_ID=$3 THREADS=$4 EDIT_FORMAT=$5

EVALS=$(cd "$(dirname "$0")" && pwd)
AIDER_DIR=$EVALS/aider
SUPPORT=$EVALS/aider_support
EXDIR=$AIDER_DIR/tmp.benchmarks/polyglot-benchmark
OUT=$HOME/ornith-fusion/runs/$LABEL/aider
IMAGE=${AIDER_IMAGE:-aider-benchmark:jest29}
TEMPERATURE=${TEMPERATURE:-1.0} TOP_P=${TOP_P:-0.95} TOP_K=${TOP_K:-40}
TRIES=${TRIES:-2} NUM_TESTS=${NUM_TESTS:--1}
AIDER_MODEL="openai/$MODEL_ID"

[ -d "$EXDIR/python" ] || { echo "missing $EXDIR"; exit 1; }
docker image inspect "$IMAGE" >/dev/null || { echo "missing image $IMAGE (run benchmark/docker_build.sh)"; exit 1; }
mkdir -p "$OUT" "$SUPPORT/cache/gradle"

# Preflight: server reachable, model id served, context length.
MODELS_JSON=$(curl -sf --max-time 20 "$BASE_URL/models") || { echo "server not reachable: $BASE_URL/models"; exit 1; }
python3 -c 'import json,sys; ids=[m["id"] for m in json.loads(sys.argv[1])["data"]]; sys.exit(0 if sys.argv[2] in ids else "model %s not served; have %s" % (sys.argv[2], ids))' "$MODELS_JSON" "$MODEL_ID"
if [ -z "${MAX_INPUT_TOKENS:-}" ]; then
  MAX_INPUT_TOKENS=$(python3 -c 'import json,sys; d=[m for m in json.loads(sys.argv[1])["data"] if m["id"]==sys.argv[2]][0]; print(d.get("max_model_len") or 262144)' "$MODELS_JSON" "$MODEL_ID")
fi

# Aider model settings (exact-name match) + litellm metadata for the unknown model.
python3 - "$OUT" "$AIDER_MODEL" "$EDIT_FORMAT" "$TEMPERATURE" "$TOP_P" "$TOP_K" "${MAX_TOKENS:-}" "$MAX_INPUT_TOKENS" <<'PY'
import json, sys
out, name, fmt, temp, top_p, top_k, max_tokens, ctx = sys.argv[1:]
extra = {"temperature": float(temp), "top_p": float(top_p), "extra_body": {"top_k": int(top_k)}}
if max_tokens:
    extra["max_tokens"] = int(max_tokens)
# Written as JSON (valid YAML) so no yaml dependency is needed on the host.
settings = [{"name": name, "edit_format": fmt, "use_repo_map": fmt in ("diff", "udiff", "diff-fenced"),
             "use_temperature": float(temp), "streaming": False, "extra_params": extra}]
open(f"{out}/model_settings.yml", "w").write(json.dumps(settings, indent=2) + "\n")
ctx = int(ctx)
meta = {name: {"max_input_tokens": ctx, "max_tokens": ctx, "max_output_tokens": ctx,
               "input_cost_per_token": 0, "output_cost_per_token": 0,
               "litellm_provider": "openai", "mode": "chat"}}
open(f"{out}/model_metadata.json", "w").write(json.dumps(meta, indent=2) + "\n")
PY

RUN_DIR_ARG=$LABEL
EXTRA_ARGS=()
if ls -d "$OUT"/*--"$LABEL" >/dev/null 2>&1; then
  if [ "${CONT:-0}" = 1 ]; then EXTRA_ARGS+=(--cont)
  else echo "prior run exists in $OUT; set CONT=1 to resume or use a new label"; exit 1; fi
fi
[ -n "${KEYWORDS:-}" ] && EXTRA_ARGS+=(--keywords "$KEYWORDS")
[ -n "${LANGUAGES:-}" ] && EXTRA_ARGS+=(--languages "$LANGUAGES")

echo "aider polyglot: label=$LABEL model=$AIDER_MODEL base=$BASE_URL threads=$THREADS edit_format=$EDIT_FORMAT tries=$TRIES temp=$TEMPERATURE top_p=$TOP_P top_k=$TOP_K max_tokens=${MAX_TOKENS:-server} ctx=$MAX_INPUT_TOKENS out=$OUT"

docker run --rm --network host \
  --memory="${DOCKER_MEMORY:-48g}" --memory-swap="${DOCKER_MEMORY:-48g}" \
  -v "$AIDER_DIR":/aider \
  -v "$EXDIR":/polyglot:ro \
  -v "$OUT":/benchmarks \
  -v "$SUPPORT":/support:ro \
  -v "$SUPPORT/cache/gradle":/root/.gradle \
  -e AIDER_DOCKER=1 -e AIDER_BENCHMARK_DIR=/benchmarks \
  -e OPENAI_API_BASE="$BASE_URL" -e OPENAI_API_KEY=dummy \
  -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)" \
  --name "aider-bench-$LABEL" \
  "$IMAGE" bash -c '
    trap "chown -R $HOST_UID:$HOST_GID /benchmarks /root/.gradle 2>/dev/null || true" EXIT
    cd /aider
    python3 /support/bench_wrapper.py /benchmarks/model_metadata.json \
      "$0" --model "$1" --edit-format "$2" --threads "$3" --tries "$4" --num-tests "$5" \
      --read-model-settings /benchmarks/model_settings.yml --exercises-dir /polyglot "${@:6}"
  ' "$RUN_DIR_ARG" "$AIDER_MODEL" "$EDIT_FORMAT" "$THREADS" "$TRIES" "$NUM_TESTS" "${EXTRA_ARGS[@]}" \
  2>&1 | tee -a "$OUT/run.log"

RUN_DIR=$(ls -d "$OUT"/*--"$LABEL" | tail -1)
# Stats (no code execution; same as benchmark.py --stats).
docker run --rm -v "$AIDER_DIR":/aider -v "$RUN_DIR":/benchmarks/"$(basename "$RUN_DIR")":ro \
  -e AIDER_BENCHMARK_DIR=/benchmarks "$IMAGE" \
  python3 /aider/benchmark/benchmark.py --stats "/benchmarks/$(basename "$RUN_DIR")" 2>/dev/null \
  | sed 's/\x1b\[[0-9;]*m//g' > "$OUT/stats.yaml" || true
cat "$OUT/stats.yaml"

python3 - "$OUT/stats.yaml" "$LABEL" "$EDIT_FORMAT" <<'PY'
import re, sys
txt = open(sys.argv[1]).read()
g = lambda k: (re.search(rf"^\s*{k}: (\S+)", txt, re.M) or [None, "NA"])[1]
done, total = g("test_cases"), g("total_tests")
flag = "" if done == total == "225" else "  INCOMPLETE"
pr1, pr2 = g("pass_rate_1"), g("pass_rate_2")
if pr2 == "NA":  # harness only emits pass_rate_2 if some case actually used a 2nd try
    pr2 = pr1
print(f"AIDER label={sys.argv[2]} edit_format={sys.argv[3]} pass_rate_1={pr1} "
      f"pass_rate_2={pr2} well_formed={g('percent_cases_well_formed')} "
      f"cases={done}/{total} malformed={g('num_malformed_responses')} "
      f"error_outputs={g('error_outputs')} timeouts={g('test_timeouts')} "
      f"exhausted_ctx={g('exhausted_context_windows')}{flag}")
PY
