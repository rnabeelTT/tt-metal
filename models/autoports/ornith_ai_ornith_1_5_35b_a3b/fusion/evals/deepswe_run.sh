#!/usr/bin/env bash
# DeepSWE (Datacurve, 113 tasks, Apache-2.0; datacurve-ai/deep-swe@435ee89e = HEAD at the Ornith-35B card
# release, v1.1 task format) with Pier 0.3.1 (Datacurve's Harbor fork, the official DeepSWE runner) and the
# Claude Code agent, against an OpenAI/Anthropic-compatible server through ~/evals/agent_proxy/llm_proxy.py.
# Card (Ornith-1.5-35B: 22): "Claude Code harness with temperature=1.0, top_p=0.95, and a 256K context window".
#
# Task protocol (as published): the agent container has no network except, through Pier's squid sidecar,
# the inference host (here: the proxy on the docker bridge gateway); the agent must commit its work; a
# [[verifier.collect]] hook extracts `git diff <base> HEAD`, which is applied and graded with hidden tests
# in a separate, network-less verifier container. Reward = binary (all fail->pass and pass->pass tests).
#
# Usage:
#   deepswe_run.sh <variant> <label> <base_url> <model_id> <concurrency>
#     variant: v1.1 (all 113 tasks)
#   e.g. deepswe_run.sh v1.1 ornith35b-ref http://localhost:20000/v1 ornith-ai/Ornith-1.5-35B-A3B 4
#
# Results: ~/ornith-fusion/runs/<label>/deepswe/ (job/ = Pier job dir with per-trial agent logs, Claude Code
#   sessions, verifier reports; job.json, proxy_requests.jsonl, deepswe_trials.tsv, deepswe_summary.json,
#   run.log). Final line: RESULT deepswe resolved=<x>/<trials> ... Re-running with the same label resumes.
#
# Env (defaults = the card for the model id, applied only when unset):
#   TEMPERATURE TOP_P TOP_K MAX_TOKENS   sampling, forced by the proxy on every /v1/messages request
#                           (Ornith: T 1.0, top_p 0.95; Claude Code sends none of them itself)
#   CC_VERSION=2.1.126      Claude Code version (the cards' Terminal-Bench CC version; DeepSWE's is unstated)
#   CC_CONTEXT_WINDOW=262144  auto-compact window (card 256K); >200K uses the "[1m]" model-id suffix that
#                           Claude Code needs to accept it (stripped again by the proxy)
#   MAX_TOKENS=             per-request output cap (CLAUDE_CODE_MAX_OUTPUT_TOKENS; unset = 64000)
#   AGENT_TIMEOUT=          agent timeout per trial, s (unset = task's own: 5400 at 435ee89e; upstream raised
#                           it to 10800 on 2026-08-26)
#   N_ATTEMPTS=1  TASKS=a,b  EXCLUDE_TASKS=  LIMIT=N  MAX_RETRIES=1  CPUS= MEM_MB= (task's own: 2 / 8 GB)
#   KEEP_IMAGES=0           1 = keep task images and the build cache
#   BUILDX_BUILDER=deepswe-builder  BUILDER_MAX_GB=60   dedicated builder (docker-container driver), pruned
#   RUNS_ROOT=~/ornith-fusion/runs
set -euo pipefail
if [ $# -ne 5 ]; then sed -n '2,36p' "$0"; exit 2; fi
VARIANT=$1 LABEL=$2 BASE_URL=${3%/} MODEL_ID=$4 CONC=$5
case "$VARIANT" in v1.1|1.1|all) ;; *) echo "variant must be v1.1"; exit 2;; esac
EVALS=$(cd "$(dirname "$0")" && pwd)
D=$EVALS/deepswe
TASKS_DIR=$D/tasks-435ee89e
OUT=${RUNS_ROOT:-$HOME/ornith-fusion/runs}/$LABEL/deepswe
mkdir -p "$OUT"
case "$MODEL_ID" in
  *Ornith*|*Qwen3.5*) : "${TEMPERATURE:=1.0}" "${TOP_P:=0.95}";;
esac
: "${CC_VERSION:=2.1.126}" "${CC_CONTEXT_WINDOW:=262144}"
export TEMPERATURE=${TEMPERATURE:-} TOP_P=${TOP_P:-} TOP_K=${TOP_K:-} MAX_TOKENS=${MAX_TOKENS:-}

if [ "${PREFLIGHT:-1}" = 1 ]; then
  MODELS_JSON=$(curl -sf --max-time 20 "$BASE_URL/models") || { echo "server not reachable: $BASE_URL/models"; exit 1; }
  python3 -c 'import json,sys; ids=[m["id"] for m in json.loads(sys.argv[1])["data"]]; sys.exit(0 if sys.argv[2] in ids else "model %s not served; have %s" % (sys.argv[2], ids))' "$MODELS_JSON" "$MODEL_ID"
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 60 -H 'content-type: application/json' \
    -d "{\"model\":\"$MODEL_ID\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}" "$BASE_URL/messages/count_tokens")
  [ "$code" = 200 ] || { echo "server has no Anthropic /v1/messages API ($BASE_URL/messages/count_tokens -> HTTP $code)"; exit 1; }
fi

PROXY_PID= CLEAN_PID=
PRUNE_PID=
cleanup() { for p in $PROXY_PID $CLEAN_PID $PRUNE_PID; do kill "$p" 2>/dev/null || true; done; }
trap cleanup EXIT
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
# Claude Code runs in the task container and reaches the proxy through Pier's squid sidecar
BIND=$(docker network inspect bridge -f '{{range .IPAM.Config}}{{.Gateway}}{{end}}')
PROXY_LOG=$OUT/proxy_requests.jsonl PROXY_SAMPLING=$SAMPLING PROXY_MAX_TOKENS=${MAX_TOKENS:-0} \
  python3 "$EVALS/agent_proxy/llm_proxy.py" "$BIND" "$PORT" "${BASE_URL%/v1}" >> "$OUT/proxy.log" 2>&1 &
PROXY_PID=$!
sleep 1.5
export PIER_EXTRA_SAFE_PORTS=$PORT

# Image builds (agent install layer, verifier image, squid sidecar) go to a dedicated buildx builder so
# their cache (incl. the pulled task base images, ~1-2 GB per task) can be pruned without touching
# anyone else's: kept under BUILDER_MAX_GB while running, emptied at the end (KEEP_IMAGES=1 keeps it).
export BUILDX_BUILDER=${BUILDX_BUILDER:-deepswe-builder}
docker buildx inspect "$BUILDX_BUILDER" >/dev/null 2>&1 || docker buildx create --name "$BUILDX_BUILDER" --driver docker-container >/dev/null
docker buildx inspect --bootstrap "$BUILDX_BUILDER" >/dev/null
( while sleep 600; do docker buildx prune --builder "$BUILDX_BUILDER" -f --max-used-space "${BUILDER_MAX_GB:-60}gb" >/dev/null 2>&1; done ) &
PRUNE_PID=$!

python3 - "$OUT/job.json" "$TASKS_DIR" "$MODEL_ID" "http://$BIND:$PORT" "$CONC" <<'EOF'
import json, os, re, sys
out, tasks, model, base, conc = sys.argv[1:]
E = os.environ.get
win = E("CC_CONTEXT_WINDOW")
name = model + ("[1m]" if win and int(win) > 200000 else "")
env = {"ANTHROPIC_BASE_URL": base, "ANTHROPIC_API_KEY": "dummy", "DISABLE_TELEMETRY": "1",
       "DISABLE_ERROR_REPORTING": "1", "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1"}
if win:
    env["CLAUDE_CODE_AUTO_COMPACT_WINDOW"] = win
if E("MAX_TOKENS"):
    env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] = E("MAX_TOKENS")
agent = {"name": "claude-code", "model_name": name, "kwargs": {"version": E("CC_VERSION")}, "env": env}
if E("AGENT_TIMEOUT"):
    agent["override_timeout_sec"] = float(E("AGENT_TIMEOUT"))
ds = {"path": tasks}
inc = [x for x in re.split(r"[,\s]+", E("TASKS", "")) if x]
exc = [x for x in re.split(r"[,\s]+", E("EXCLUDE_TASKS", "")) if x]
if inc: ds["task_names"] = inc
if exc: ds["exclude_task_names"] = exc
if E("LIMIT"): ds["n_tasks"] = int(E("LIMIT"))
envc = {"type": "docker"}
if E("CPUS"): envc["override_cpus"] = int(float(E("CPUS")))
if E("MEM_MB"): envc["override_memory_mb"] = int(E("MEM_MB"))
cfg = {"job_name": "job", "jobs_dir": os.path.dirname(out), "n_attempts": int(E("N_ATTEMPTS", "1")),
       "n_concurrent_trials": int(conc), "quiet": True, "retry": {"max_retries": int(E("MAX_RETRIES", "1"))},
       "environment": envc, "agents": [agent], "datasets": [ds]}
json.dump(cfg, open(out, "w"), indent=1)
EOF

[ -f "$OUT/preexisting_images.txt" ] || docker images --format '{{.Repository}}:{{.Tag}}' > "$OUT/preexisting_images.txt"
if [ "${KEEP_IMAGES:-0}" != 1 ]; then
  python3 "$EVALS/tbench/tb_helper.py" cleanup "$OUT/job" "$TASKS_DIR" "$OUT/preexisting_images.txt" &
  CLEAN_PID=$!
fi
echo "deepswe $VARIANT label=$LABEL model=$MODEL_ID agent=claude-code@$CC_VERSION conc=$CONC ctx=$CC_CONTEXT_WINDOW timeout=${AGENT_TIMEOUT:-task} sampling: T=$TEMPERATURE top_p=$TOP_P top_k=$TOP_K max_tokens=${MAX_TOKENS:-cc-default} api=http://$BIND:$PORT" | tee -a "$OUT/run.log"
PIER="$D/.venv/bin/python $D/pier_launch.py"
if [ -d "$OUT/job" ]; then
  $PIER job resume -p "$OUT/job" 2>&1 | tee -a "$OUT/run.log" || true
else
  $PIER run -c "$OUT/job.json" -y 2>&1 | tee -a "$OUT/run.log" || true
fi
NT=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["n_attempts"])' "$OUT/job.json")
NTASK=$(python3 -c 'import sys; sys.path.insert(0, sys.argv[1]); import tb_helper as t; print(len(t.selected_tasks(sys.argv[2], sys.argv[3])))' "$EVALS/tbench" "$OUT/job.json" "$TASKS_DIR")
EXPECTED_TRIALS=$((NTASK * NT)) python3 "$EVALS/tbench/tb_helper.py" summarize "$OUT/job" deepswe "$LABEL" "$OUT/proxy_requests.jsonl" | tee -a "$OUT/run.log"
if [ "${KEEP_IMAGES:-0}" != 1 ]; then
  kill "$CLEAN_PID" 2>/dev/null || true; CLEAN_PID=
  python3 "$EVALS/tbench/tb_helper.py" cleanup "$OUT/job" "$TASKS_DIR" "$OUT/preexisting_images.txt" final || true
  [ "$BUILDX_BUILDER" = deepswe-builder ] && docker buildx prune --builder deepswe-builder -af >/dev/null 2>&1 || true
fi
