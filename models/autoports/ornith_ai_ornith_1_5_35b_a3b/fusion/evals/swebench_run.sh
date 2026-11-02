#!/usr/bin/env bash
# SWE-bench Verified (500) / Multilingual (300) / Pro public v1 (731) against an
# OpenAI-compatible server, one instance at a time per worker:
#   pull official image -> agent (OpenHands or SWE-agent) -> official evaluation
#   (swebench v5 harness; SWE-bench Pro: scaleapi swe_bench_pro_eval.py --use_local_docker)
#   -> remove that instance's images + prune the isolated builder cache.
#
# Usage:
#   swebench_run.sh <variant> <label> <base_url> <model_id> <concurrency>
#     variant: verified | multilingual | pro
#   e.g. swebench_run.sh verified ornith35b-ref http://localhost:20000/v1 ornith-ai/Ornith-1.5-35B-A3B 8
#
# Results: ~/ornith-fusion/runs/<label>/swebench_<variant>/  (instances/<iid>/{agent.log,
#   patch.diff,eval/,result.json}, proxy_requests.jsonl, results.tsv, summary.json, run.log).
# Final line: RESULT swebench_<variant> resolved=<x>/<n> ...   Re-running the same command
# resumes (finished instances are skipped; RETRY_ERRORS=1 redoes errored ones).
#
# Harness (HARNESS=openhands|sweagent). Default per model, matching the cards:
#   Qwen/Qwen3-Coder-Next -> sweagent (card chart: "w/ SWE-Agent"; 300 turns)
#   everything else       -> openhands (Ornith cards: OpenHands; Qwen3.5: internal bash+edit
#                            scaffold, OpenHands is the closest public equivalent)
#
# Env (defaults = the card for the model id; see model_defaults below):
#   TEMPERATURE TOP_P TOP_K   sampling, forced on every request by the proxy
#   MAX_TOKENS=               per-request output cap (unset = server default)
#   MAX_ITERS                 agent step cap (openhands 500 = OH default; sweagent 300 = Qwen report)
#   MAX_INPUT_TOKENS          context told to the harness (262144; Qwen3.5: 204800 = Qwen's
#                             "200K context window" for its SWE-bench runs)
#   LIMIT=N INSTANCES=a,b|file INSTANCE_REGEX=re   subsets (smoke tests)
#   AGENT_TIMEOUT=28800       wall-clock seconds per instance agent phase (killed -> error)
#   EVAL_TIMEOUT=3600         per-instance test run timeout
#   NO_NETWORK=1              agent container on the --internal docker network swe-nonet (no egress)
#   SANITIZE_GIT              full (Ornith: history squashed to one commit) | refs (Coder-Next/
#                             Qwen3.5/default: no remotes/branches/tags/reflog, future objects
#                             pruned) | off. Runs in the agent's repo before its first step.
#   NOHACK_PROMPT             1 = append an explicit "don't reward-hack" notice to OpenHands'
#                             prompt (default 1 for Ornith-1.5-122B, as its card describes)
#   OH_CRITIC_RUNS=3          OpenHands iterative mode: re-run instances that end without
#                             finish+non-empty patch (OH default); 1 = single attempt
#   OH_CONDENSER=1            OpenHands LLM summarizing condenser (OH default on, 240 events)
#   PROXY=1                   route through ~/evals/agent_proxy/llm_proxy.py (sampling, reasoning
#                             round-trip, per-call log); PROXY_REASONING=keep|drop|passthrough
#   PREP_CONCURRENCY=2 EVAL_CONCURRENCY=3   pulls/builds and test runs in flight
#   MIN_FREE_GB=40            don't start a new instance below this much free disk
#   SWE_DOCKER_CPUS= SWE_DOCKER_MEM=        per agent container limits (e.g. 4, 16g)
#   KEEP_IMAGES=0             1 = keep pulled/built images (debug)
#   RUNS_ROOT=~/ornith-fusion/runs
set -euo pipefail
if [ $# -ne 5 ]; then sed -n '2,50p' "$0"; exit 2; fi
VARIANT=$1 LABEL=$2 BASE_URL=${3%/} MODEL_ID=$4 CONC=$5
case "$VARIANT" in verified|multilingual|pro) ;; *) echo "variant must be verified|multilingual|pro"; exit 2;; esac
EVALS=$(cd "$(dirname "$0")" && pwd)
OUT=${RUNS_ROOT:-$HOME/ornith-fusion/runs}/$LABEL/swebench_$VARIANT
mkdir -p "$OUT"

# Card defaults per model (only applied when the variable is unset).
case "$MODEL_ID" in
  *Coder-Next*)    : "${HARNESS:=sweagent}" "${TEMPERATURE:=1.0}" "${TOP_P:=0.95}" "${TOP_K:=40}" "${SANITIZE_GIT:=refs}";;
  *Ornith*)        : "${HARNESS:=openhands}" "${TEMPERATURE:=1.0}" "${TOP_P:=0.95}" "${SANITIZE_GIT:=full}";;
  *Qwen3.5*)       : "${HARNESS:=openhands}" "${TEMPERATURE:=1.0}" "${TOP_P:=0.95}" "${SANITIZE_GIT:=refs}" "${MAX_INPUT_TOKENS:=204800}";;
  *)               : "${HARNESS:=openhands}" "${SANITIZE_GIT:=refs}";;
esac
case "$MODEL_ID" in *Ornith-1.5-122B*) : "${NOHACK_PROMPT:=1}";; esac
export SANITIZE_GIT NOHACK_PROMPT=${NOHACK_PROMPT:-0} MAX_INPUT_TOKENS=${MAX_INPUT_TOKENS:-262144}
if [ "$HARNESS" = sweagent ]; then : "${MAX_ITERS:=300}"; else : "${MAX_ITERS:=500}"; fi
export TEMPERATURE TOP_P TOP_K MAX_ITERS
export TEMPERATURE=${TEMPERATURE:-} TOP_P=${TOP_P:-} TOP_K=${TOP_K:-}

# Preflight: server + model id (skipped for PREFLIGHT=0).
if [ "${PREFLIGHT:-1}" = 1 ]; then
  MODELS_JSON=$(curl -sf --max-time 20 "$BASE_URL/models") || { echo "server not reachable: $BASE_URL/models"; exit 1; }
  python3 -c 'import json,sys; ids=[m["id"] for m in json.loads(sys.argv[1])["data"]]; sys.exit(0 if sys.argv[2] in ids else "model %s not served; have %s" % (sys.argv[2], ids))' "$MODELS_JSON" "$MODEL_ID"
fi
docker network inspect swe-nonet >/dev/null 2>&1 || docker network create --internal swe-nonet >/dev/null
if [ "$HARNESS" = openhands ]; then
  docker buildx inspect swe-oh-builder >/dev/null 2>&1 || \
    docker buildx create --name swe-oh-builder --driver docker-container --driver-opt network=host >/dev/null
fi

python3 -u "$EVALS/swebench_support/swe_driver.py" --variant "$VARIANT" --harness "$HARNESS" \
  --label "$LABEL" --out "$OUT" --base_url "$BASE_URL" --model "$MODEL_ID" --concurrency "$CONC" \
  2>&1 | tee -a "$OUT/run.log"
