#!/usr/bin/env bash
# Harbor benchmarks against an OpenAI-compatible server, through ~/evals/agent_proxy/llm_proxy.py
# (sampling forced on every request, reasoning round-tripped into the history, one log line per LLM call):
#   2.0 | 2.1      Terminal-Bench 2.0 / 2.1 (89 tasks), Harbor + Terminus-2 (parser=json) by default
#   frontier       Frontier-Bench v0.1 = terminal-bench@3.0.0 (74 tasks; tasks needing a GPU are excluded
#                  by default -> 70), Terminus-2 by default (the cards give no harness for it)
#   sweatlas-qna   SWE Atlas Codebase QnA (124 tasks, scaleapi/SWE-Atlas@49e4af3 = the data current at the
#                  Ornith-35B card), mini-swe-agent 2.4.6 + SWE Atlas' qa config; the rubric LLM judge is
#                  deferred to files (JUDGE_PENDING line; ingest with ~/evals/sweatlas/sweatlas_judge.py)
#
# Usage:
#   tbench_run.sh <variant> <label> <base_url> <model_id> <concurrency>
#     (pinned trees: ~/evals/tbench/tb-2.0 = terminal-bench-2@f5b891c, tb-2.1 = @2fd12b8, i.e. after
#      "Various task fixes for TB2.1"; ~/evals/frontierbench/tasks-3.0.0; ~/evals/sweatlas/tasks-49e4af3)
#   e.g. tbench_run.sh 2.1 ornith35b-ref http://localhost:20000/v1 ornith-ai/Ornith-1.5-35B-A3B 4
#   Wrappers with the <bench>_run.sh interface: frontierbench_run.sh v0.1 ..., sweatlas_run.sh qna ...
#
# Results: ~/ornith-fusion/runs/<label>/<bench>/ with <bench> = tbench_<variant> (Terminus-2),
#   tbench_<variant>_<agent> (other agents), frontierbench[_<agent>], sweatlas_qna. Contents: job/ =
#   Harbor job dir with per-trial agent logs + trajectories, job.json, proxy_requests.jsonl, *_trials.tsv,
#   *_summary.json, run.log. Final line: RESULT <bench> resolved=<x>/<trials> ... (SWE Atlas: RESULT
#   with answers written + JUDGE_PENDING <judge_inputs.jsonl>). Re-running with the same label resumes
#   the Harbor job (finished trials are kept).
#
# Env (defaults = the card for the model id, applied only when unset):
#   TEMPERATURE TOP_P TOP_K MAX_TOKENS   sampling (forced by the proxy on every request)
#   CTX                     context told to Terminus-2 (drives its summarization): Ornith
#                           131072 (card: 128K), Qwen3.5 / Coder-Next 262144
#   AGENT_TIMEOUT           absolute agent timeout per trial, s (Ornith card 4 h = 14400;
#                           Qwen3.5 card 3 h = 10800; Coder-Next: not stated -> 10800).
#                           AGENT_TIMEOUT="" = each task's own timeout (default for frontier, sweatlas-qna)
#   CPUS=8 MEM_MB=49152     per-trial container limits (card: 32 CPU / 48 GB; box has 16
#                           cores, so CPU is capped at 8 and trials share cores). frontier: task's
#                           own (CPUS= MEM_MB=); sweatlas-qna: CPUS=8 (tasks ask 16), task's 16 GB
#   N_ATTEMPTS=1            trials per task (cards average 5 runs)
#   TASKS=a,b  EXCLUDE_TASKS=  LIMIT=N   subsets (glob patterns allowed); frontier adds the GPU tasks
#                           to EXCLUDE_TASKS unless GPU_TASKS=1
#   MAX_RETRIES=1           Harbor retries for infrastructure exceptions
#   INTERLEAVED_THINKING=1  Terminus-2 keeps reasoning_content in its history (Ornith card:
#                           "modify Harbor to align with vLLM's reasoning_content key")
#   PROXY=1 PROXY_REASONING=keep  PROXY_MAX_CONTEXT=N (proxy turns responses with prompt >= N tokens
#                           into vLLM's context-length 400; sweatlas-qna default 131072 = card's 128K)
#   KEEP_IMAGES=0           1 = keep task images after each task finishes (frontier: builds go to the
#                           dedicated buildx builder harbor-tb3-builder, pruned to BUILDER_MAX_GB=60, emptied at the end)
#   AGENT=terminus-2        oracle = reference solutions (no model; checks tasks/verifiers);
#                           claude-code = Claude Code CC_VERSION=2.1.126 in the task container,
#                           Anthropic Messages API via the proxy -> vLLM /v1/messages (preflight checks
#                           /v1/messages/count_tokens); CC_CONTEXT_WINDOW (auto-compact window),
#                           CC_DISALLOWED_TOOLS (e.g. WebSearch,WebFetch);
#                           mini-swe-agent = MSWEA_VERSION=2.4.6 with MSWEA_CONFIG (yaml) in the container
#   RUNS_ROOT=~/ornith-fusion/runs
set -euo pipefail
if [ $# -ne 5 ]; then sed -n '2,51p' "$0"; exit 2; fi
VARIANT=$1 LABEL=$2 BASE_URL=${3%/} MODEL_ID=$4 CONC=$5
EVALS=$(cd "$(dirname "$0")" && pwd)
: "${AGENT:=terminus-2}"
SUFFIX=; [ "$AGENT" != terminus-2 ] && SUFFIX=_$AGENT
case "$VARIANT" in
  2.0|2.1) TASKS_DIR=$EVALS/tbench/tb-$VARIANT; BENCH=tbench_$VARIANT$SUFFIX;;
  frontier|3.0.0)
    TASKS_DIR=$EVALS/frontierbench/tasks-3.0.0/terminal-bench; BENCH=frontierbench$SUFFIX
    : "${AGENT_TIMEOUT=}" "${CPUS=}" "${MEM_MB=}"
    if [ "${GPU_TASKS:-0}" != 1 ]; then  # tasks declaring gpus >= 1 (H100) cannot run here
      GPU=$(grep -l -E '^gpus *= *[1-9]' "$TASKS_DIR"/*/task.toml | xargs -n1 dirname | xargs -n1 basename | paste -sd, -)
      EXCLUDE_TASKS=${EXCLUDE_TASKS:+$EXCLUDE_TASKS,}$GPU
    fi;;
  sweatlas-qna)
    TASKS_DIR=$EVALS/sweatlas/tasks-49e4af3; AGENT=mini-swe-agent; BENCH=sweatlas_qna
    : "${MSWEA_CONFIG:=$EVALS/sweatlas/mswea_qa_config.yaml}" "${AGENT_TIMEOUT=}" "${CPUS=8}" "${MEM_MB=}"
    : "${PROXY_MAX_CONTEXT=131072}"
    export EXTRA_COMPOSE=$EVALS/sweatlas/compose-keepalive.yaml  # v1.0 images: ENTRYPOINT bash
    case "$MODEL_ID" in *Ornith*) : "${TEMPERATURE:=1.0}" "${TOP_P:=0.95}";; esac  # card: T 1.0, top_p 0.95, 128K
    # the official verifier's judge credentials are required by Harbor's env check even with the
    # verifier disabled; they are never used (judging is deferred to files)
    export VERIFIER_DISABLE=1 MSWEA_CONFIG OPENAI_API_KEY=deferred-judge OPENAI_API_BASE=http://127.0.0.1:9/v1;;
  *) echo "variant must be 2.0 | 2.1 | frontier | sweatlas-qna"; exit 2;;
esac
[ -d "$TASKS_DIR" ] || { echo "missing task dir $TASKS_DIR"; exit 2; }
OUT=${RUNS_ROOT:-$HOME/ornith-fusion/runs}/$LABEL/$BENCH
mkdir -p "$OUT"
HARBOR=$EVALS/harbor/.venv/bin/harbor

case "$MODEL_ID" in
  *Ornith*)     : "${TEMPERATURE:=1.0}" "${TOP_P:=1.0}" "${CTX:=131072}" "${AGENT_TIMEOUT=14400}";;
  *Qwen3.5*)    : "${TEMPERATURE:=1.0}" "${TOP_P:=0.95}" "${TOP_K:=20}" "${MAX_TOKENS:=81920}" "${CTX:=262144}" "${AGENT_TIMEOUT=10800}";;
  *Coder-Next*) : "${TEMPERATURE:=1.0}" "${TOP_P:=0.95}" "${TOP_K:=40}" "${CTX:=262144}" "${AGENT_TIMEOUT=10800}";;
  *)            : "${CTX:=131072}" "${AGENT_TIMEOUT=}";;
esac
export TEMPERATURE=${TEMPERATURE:-} TOP_P=${TOP_P:-} TOP_K=${TOP_K:-} MAX_TOKENS=${MAX_TOKENS:-} CTX AGENT_TIMEOUT
export CPUS=${CPUS-8} MEM_MB=${MEM_MB-49152} MODEL_ID TASKS_DIR CONC AGENT EXCLUDE_TASKS=${EXCLUDE_TASKS:-}
export JOB_NAME=job JOBS_DIR=$OUT

if [ "$AGENT" != oracle ] && [ "${PREFLIGHT:-1}" = 1 ]; then
  MODELS_JSON=$(curl -sf --max-time 20 "$BASE_URL/models") || { echo "server not reachable: $BASE_URL/models"; exit 1; }
  python3 -c 'import json,sys; ids=[m["id"] for m in json.loads(sys.argv[1])["data"]]; sys.exit(0 if sys.argv[2] in ids else "model %s not served; have %s" % (sys.argv[2], ids))' "$MODELS_JSON" "$MODEL_ID"
  if [ "$AGENT" = claude-code ]; then  # Claude Code needs the Anthropic Messages API on the server
    code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 60 -H 'content-type: application/json' \
      -d "{\"model\":\"$MODEL_ID\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}" "$BASE_URL/messages/count_tokens")
    [ "$code" = 200 ] || { echo "server has no Anthropic /v1/messages API ($BASE_URL/messages/count_tokens -> HTTP $code); put a LiteLLM translation proxy in front"; exit 1; }
  fi
fi

PROXY_PID= CLEAN_PID= PRUNE_PID=
cleanup() { for p in $PROXY_PID $CLEAN_PID $PRUNE_PID; do kill "$p" 2>/dev/null || true; done; }
trap cleanup EXIT
if [ "${PROXY:-1}" = 1 ] && [ "$AGENT" != oracle ]; then
  PORT=$(python3 -c 'import socket
for p in range(28300,28399):
    s=socket.socket()
    try: s.bind(("",p)); print(p); break
    except OSError: pass
    finally: s.close()')
  SAMPLING=$(python3 -c 'import json,os; d={}
e=os.environ
if e.get("TEMPERATURE"): d["temperature"]=float(e["TEMPERATURE"])
if e.get("TOP_P"): d["top_p"]=float(e["TOP_P"])
if e.get("TOP_K"): d["top_k"]=int(e["TOP_K"])
print(json.dumps(d))')
  UP=${BASE_URL%/v1}
  BIND=127.0.0.1   # Terminus-2 calls the LLM from the host
  # Claude Code / mini-swe-agent run inside the task container: listen on the docker bridge gateway
  case "$AGENT" in claude-code|mini-swe-agent) BIND=$(docker network inspect bridge -f '{{range .IPAM.Config}}{{.Gateway}}{{end}}');; esac
  PROXY_LOG=$OUT/proxy_requests.jsonl PROXY_SAMPLING=$SAMPLING PROXY_MAX_TOKENS=${MAX_TOKENS:-0} PROXY_MAX_CONTEXT=${PROXY_MAX_CONTEXT:-0} \
    python3 "$EVALS/agent_proxy/llm_proxy.py" "$BIND" "$PORT" "$UP" >> "$OUT/proxy.log" 2>&1 &
  PROXY_PID=$!
  sleep 1.5
  export AGENT_API_BASE=http://$BIND:$PORT/v1 AGENT_API_BASE_ROOT=http://$BIND:$PORT
else
  export AGENT_API_BASE=$BASE_URL
fi

if [ "$BENCH" = "frontierbench$SUFFIX" ] && [ "${KEEP_IMAGES:-0}" != 1 ]; then
  # Frontier-Bench tasks are built locally (Dockerfile + separate verifier image): build in a dedicated buildx
  # builder whose cache (pulled base images, layers) is kept under BUILDER_MAX_GB and emptied at the end,
  # without touching the default builder's cache.
  export BUILDX_BUILDER=${BUILDX_BUILDER:-harbor-tb3-builder}
  docker buildx inspect "$BUILDX_BUILDER" >/dev/null 2>&1 || docker buildx create --name "$BUILDX_BUILDER" --driver docker-container >/dev/null
  docker buildx inspect --bootstrap "$BUILDX_BUILDER" >/dev/null
  ( while sleep 600; do docker buildx prune --builder "$BUILDX_BUILDER" -f --max-used-space "${BUILDER_MAX_GB:-60}gb" >/dev/null 2>&1; done ) &
  PRUNE_PID=$!
fi
python3 "$EVALS/tbench/tb_helper.py" config "$OUT/job.json" > /dev/null
[ -f "$OUT/preexisting_images.txt" ] || docker images --format '{{.Repository}}:{{.Tag}}' > "$OUT/preexisting_images.txt"
if [ "${KEEP_IMAGES:-0}" != 1 ]; then
  python3 "$EVALS/tbench/tb_helper.py" cleanup "$OUT/job" "$TASKS_DIR" "$OUT/preexisting_images.txt" &
  CLEAN_PID=$!
fi
echo "$BENCH variant=$VARIANT label=$LABEL model=$MODEL_ID agent=$AGENT conc=$CONC cpus=$CPUS mem_mb=$MEM_MB ctx=$CTX timeout=${AGENT_TIMEOUT:-task} sampling: T=$TEMPERATURE top_p=$TOP_P top_k=$TOP_K max_tokens=${MAX_TOKENS:-server} max_context=${PROXY_MAX_CONTEXT:-server} exclude=${EXCLUDE_TASKS:-none} api=$AGENT_API_BASE" | tee -a "$OUT/run.log"
if [ -d "$OUT/job" ]; then
  "$HARBOR" job resume -p "$OUT/job" 2>&1 | tee -a "$OUT/run.log" || true
else
  "$HARBOR" run -c "$OUT/job.json" -y 2>&1 | tee -a "$OUT/run.log" || true
fi
NT=$(python3 -c 'import json,sys; c=json.load(open(sys.argv[1])); print(c["n_attempts"])' "$OUT/job.json")
EXPECTED_TRIALS=$(( $(python3 -c 'import json,sys,glob,os,fnmatch
c=json.load(open(sys.argv[1])); d=c["datasets"][0]
ts=sorted(os.path.basename(os.path.dirname(p)) for p in glob.glob(os.path.join(d["path"],"*","task.toml")))
if d.get("task_names"): ts=[t for t in ts if any(fnmatch.fnmatch(t,p) for p in d["task_names"])]
if d.get("exclude_task_names"): ts=[t for t in ts if not any(fnmatch.fnmatch(t,p) for p in d["exclude_task_names"])]
if d.get("n_tasks"): ts=ts[:d["n_tasks"]]
print(len(ts))' "$OUT/job.json") * NT )) \
  python3 "$EVALS/tbench/tb_helper.py" summarize "$OUT/job" "$BENCH" "$LABEL" "$OUT/proxy_requests.jsonl" | tee -a "$OUT/run.log"
if [ "${KEEP_IMAGES:-0}" != 1 ]; then  # final pass (the background cleaner may not have seen the end)
  kill "$CLEAN_PID" 2>/dev/null || true; CLEAN_PID=
  python3 "$EVALS/tbench/tb_helper.py" cleanup "$OUT/job" "$TASKS_DIR" "$OUT/preexisting_images.txt" final || true
  [ "${BUILDX_BUILDER:-}" = harbor-tb3-builder ] && docker buildx prune --builder harbor-tb3-builder -af >/dev/null 2>&1 || true
fi
if [ "$BENCH" = sweatlas_qna ]; then  # rubric judge deferred to files (official prompts, verdicts by Claude agents)
  python3 "$EVALS/sweatlas/sweatlas_judge.py" prepare "$OUT" | tee -a "$OUT/run.log"
fi
