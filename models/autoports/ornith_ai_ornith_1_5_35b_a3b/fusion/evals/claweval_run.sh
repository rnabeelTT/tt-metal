#!/usr/bin/env bash
# Claw-Eval (claw-eval/claw-eval @5680b8b = v1.1.0 + fixes, 300 tasks: general 161 / multimodal 101 /
# multi_turn 38) against an OpenAI-compatible server, official harness in sandbox mode (`claw-eval batch
# --sandbox` per-task code path), through ~/evals/agent_proxy/llm_proxy.py (sampling forced on every request,
# reasoning round-tripped, one log line per LLM call). The LLM judge (gemini-3-flash; opus-4.6 for multi_turn)
# is DEFERRED: the run records every judge call verbatim; Claude agents judge offline; the ingest re-runs the
# official graders with the verdicts (rule-based parts recomputed locally) and prints the final score.
#
# Usage:
#   claweval_run.sh <variant> <label> <base_url> <model_id> <concurrency>
#     variant: all | general | multimodal | multi_turn
#   e.g. claweval_run.sh all ornith35b-ref http://localhost:20000/v1 ornith-ai/Ornith-1.5-35B-A3B 8
#
# Results: ~/ornith-fusion/runs/<label>/claweval/ (traces/<task>_<id>.jsonl + .env_snapshot.json,
#   results/<task>.json, judge_requests/, judge_assets/ (images for visual judging), proxy_requests.jsonl,
#   serp_requests.jsonl, run.log). Re-running the same command resumes (tasks with N_RUNS trials are skipped;
#   RETRY_ERRORS=1 redoes errored trials).
# Final lines: JUDGE_PENDING <out>/judge_inputs.jsonl   (judge per JUDGE_README.txt, then
#              ~/evals/claweval_judge_ingest.sh <out> <judgments.jsonl>  -> final RESULT line)
#              RESULT claweval_<variant> pass^N=.. avg_score=.. pass@1=.. <split>:.. tasks= trials= errors= ...
#
# Env (defaults = the card for the model id, applied only when unset):
#   TEMPERATURE TOP_P TOP_K   sampling forced by the proxy (Ornith card: temp=0.6; top_p 0.95 / top_k 20 =
#                             the model's generation_config, what vLLM uses when a request omits them)
#   MAX_TOKENS=               per-request output cap (unset = server default; claw-eval sends none)
#   CTX=262144                context_window told to claw-eval (only drives its compaction, which no task
#                             enables; card: 256K)
#   N_RUNS=3                  trials per task (official Pass^3 protocol; Ornith card averages 5 runs)
#   TIMEOUT_SCALE=1           multiplies each task's timeout_seconds (120-1800 s, wall clock incl. model time,
#                             checked between turns). 1 = official; raise it if the server is much slower
#                             than the API models of the leaderboard (see timeouts= in RESULT)
#   REQUEST_TIMEOUT=1800      per-model-request client timeout, s (official 300; no extra client retries)
#   INPUT_MODALITIES=text     claw-eval model.input_modalities (official default: text). Images returned by
#                             the sandbox tools (ReadMedia/BrowserScreenshot) are sent regardless, as officially
#   USER_SIM=relay            multi_turn simulated user (official: claude-opus-4.6 via OpenRouter):
#                             relay = Claude user-simulator relay ~/evals/usersim (USER_SIM_BASE_URL
#                             http://127.0.0.1:28400/v1, USER_SIM_MODEL claude-usersim; Claude workers answer,
#                             see ~/evals/claweval/USERSIM_CLAWEVAL.txt); self = the model under test via the
#                             proxy; off = skip multi_turn tasks
#   USERSIM_EXCLUDE_WAIT=1    time waiting for the simulated user is not charged to the task timeout
#   WEB_SEARCH=searxng        web_real tasks (62): official = paid Google SERP API (SERP_DEV_KEY). Here a local
#                             adapter (~/evals/claweval/serp_adapter.py) answers in the same format from the
#                             webtools/SearXNG service (~/evals/websearch, started if down); SERP_BLOCKLIST=
#                             webtools blocklist profile (unset = its default). Page fetches go direct (official)
#   SANDBOX_CPUS=2 SANDBOX_MEM=4g   per-trial sandbox container limits (official defaults)
#   IMAGE=claw-eval-agent:5680b8b   sandbox image (built from ~/evals/claweval/docker/Dockerfile.agent if missing)
#   TASKS=a,b EXCLUDE_TASKS= LIMIT=N   subsets (glob patterns on task ids)
#   RETRY_ERRORS=0  PORT_BASE_OFFSET=20000 (mock-service ports 9100-9129 + offset + 50*slot)
#   PROXY=1 PROXY_REASONING=keep  PREFLIGHT=1  RUNS_ROOT=~/ornith-fusion/runs
# Self-test (stub model, no real server): ~/evals/claweval/selftest/selftest.sh
set -euo pipefail
if [ $# -ne 5 ]; then sed -n '2,50p' "$0"; exit 2; fi
VARIANT=$1 LABEL=$2 BASE_URL=${3%/} MODEL_ID=$4 CONC=$5
case "$VARIANT" in all|general|multimodal|multi_turn) ;; *) echo "variant must be all|general|multimodal|multi_turn"; exit 2;; esac
EVALS=$(cd "$(dirname "$0")" && pwd)
CE=$EVALS/claweval
PY=$CE/.venv/bin/python
OUT=${RUNS_ROOT:-$HOME/ornith-fusion/runs}/$LABEL/claweval
mkdir -p "$OUT"

case "$MODEL_ID" in
  *Ornith*)     : "${TEMPERATURE:=0.6}" "${TOP_P:=0.95}" "${TOP_K:=20}" "${CTX:=262144}";;
  *Qwen3.5*)    : "${TEMPERATURE:=1.0}" "${TOP_P:=0.95}" "${TOP_K:=20}" "${CTX:=262144}";;
  *Coder-Next*) : "${TEMPERATURE:=1.0}" "${TOP_P:=0.95}" "${TOP_K:=40}" "${CTX:=262144}";;
  *)            : "${CTX:=262144}";;
esac
export TEMPERATURE=${TEMPERATURE:-} TOP_P=${TOP_P:-} TOP_K=${TOP_K:-} MAX_TOKENS=${MAX_TOKENS:-} CTX
export N_RUNS=${N_RUNS:-3} TIMEOUT_SCALE=${TIMEOUT_SCALE:-1} REQUEST_TIMEOUT=${REQUEST_TIMEOUT:-1800}
export USERSIM_EXCLUDE_WAIT=${USERSIM_EXCLUDE_WAIT:-1} CLAWEVAL_LABEL=$LABEL
USER_SIM=${USER_SIM:-relay} WEB_SEARCH=${WEB_SEARCH:-searxng} IMAGE=${IMAGE:-claw-eval-agent:5680b8b}
PORT_BASE_OFFSET=${PORT_BASE_OFFSET:-20000}

# one run per port block at a time
mkdir -p "$CE/locks"
exec 9>"$CE/locks/ports_$PORT_BASE_OFFSET.lock"
flock -n 9 || { echo "another claweval run holds PORT_BASE_OFFSET=$PORT_BASE_OFFSET (lock $CE/locks/ports_$PORT_BASE_OFFSET.lock)"; exit 1; }

if [ "${PREFLIGHT:-1}" = 1 ]; then
  MODELS_JSON=$(curl -sf --max-time 20 "$BASE_URL/models") || { echo "server not reachable: $BASE_URL/models"; exit 1; }
  python3 -c 'import json,sys; ids=[m["id"] for m in json.loads(sys.argv[1])["data"]]; sys.exit(0 if sys.argv[2] in ids else "model %s not served; have %s" % (sys.argv[2], ids))' "$MODELS_JSON" "$MODEL_ID"
fi
docker image inspect "$IMAGE" >/dev/null 2>&1 || {
  echo "building sandbox image $IMAGE"
  docker build --label claweval.created-by=claweval-harness -t "$IMAGE" -f "$CE/docker/Dockerfile.agent" "$CE/repo" >> "$OUT/docker_build.log" 2>&1; }

# task selection (needed for the service checks below)
SEL=$("$PY" - "$VARIANT" "${TASKS:-}" "${EXCLUDE_TASKS:-}" "${LIMIT:-0}" "$USER_SIM" <<'EOF'
import sys
sys.path.insert(0, __import__("os").path.expanduser("~/evals/claweval"))
from pathlib import Path
import claweval_driver as d
v, only, exc, lim, us = sys.argv[1:6]
ts = d.select_tasks(d.list_tasks(d.REPO / "tasks"), v, only or None, exc or None, int(lim))
if us == "off":
    ts = [t for t in ts if t["split"] != "multi_turn"]
import yaml
web = sum(1 for t in ts if any(s.get("name", "").startswith("web_real") for s in (yaml.safe_load(open(Path(t["dir"]) / "task.yaml")).get("services") or [])))
print(len(ts), sum(t["split"] == "multi_turn" for t in ts), web)
EOF
)
read -r N_SEL N_MT N_WEB <<< "$SEL"
[ "$N_SEL" -gt 0 ] || { echo "no tasks selected"; exit 2; }

# mock-service port block must be free (stale mock services of this harness are killed)
LO=$((9100 + PORT_BASE_OFFSET)); HI=$((9129 + PORT_BASE_OFFSET + (CONC - 1) * 50))
BUSY=$(ss -ltnpH 2>/dev/null | awk -v lo=$LO -v hi=$HI '{n=split($4,a,":"); p=a[n]+0; if (p>=lo && p<=hi) print p" "$0}' || true)
if [ -n "$BUSY" ]; then
  for pid in $(echo "$BUSY" | grep -o 'pid=[0-9]*' | cut -d= -f2 | sort -u); do
    if [ "$(readlink -f /proc/$pid/cwd 2>/dev/null)" = "$(readlink -f "$CE/repo")" ] && tr '\0' ' ' < /proc/$pid/cmdline | grep -q mock_services/; then
      echo "killing stale claw-eval mock service pid $pid"; kill "$pid" 2>/dev/null || true
    fi
  done
  sleep 1
  BUSY=$(ss -ltnH 2>/dev/null | awk -v lo=$LO -v hi=$HI '{n=split($4,a,":"); p=a[n]+0; if (p>=lo && p<=hi) print $4}' || true)
  [ -z "$BUSY" ] || { echo "ports $LO-$HI in use ($BUSY): set PORT_BASE_OFFSET"; exit 1; }
fi

free_port() { python3 -c 'import socket
for p in range(28300,28399):
    s=socket.socket()
    try: s.bind(("127.0.0.1",p)); print(p); break
    except OSError: pass
    finally: s.close()'; }
PIDS=()
cleanup() { for p in "${PIDS[@]}"; do kill "$p" 2>/dev/null || true; done; }
trap cleanup EXIT

if [ "${PROXY:-1}" = 1 ]; then
  PORT=$(free_port)
  SAMPLING=$(python3 -c 'import json,os; d={}
e=os.environ
if e.get("TEMPERATURE"): d["temperature"]=float(e["TEMPERATURE"])
if e.get("TOP_P"): d["top_p"]=float(e["TOP_P"])
if e.get("TOP_K"): d["top_k"]=int(e["TOP_K"])
print(json.dumps(d))')
  PROXY_LOG=$OUT/proxy_requests.jsonl PROXY_SAMPLING=$SAMPLING PROXY_MAX_TOKENS=${MAX_TOKENS:-0} \
    python3 "$EVALS/agent_proxy/llm_proxy.py" 127.0.0.1 "$PORT" "${BASE_URL%/v1}" >> "$OUT/proxy.log" 2>&1 &
  PIDS+=($!)
  for _ in $(seq 30); do curl -sf --max-time 2 "http://127.0.0.1:$PORT/proxy/peak" >/dev/null && break; sleep 0.2; done
  AGENT_BASE=http://127.0.0.1:$PORT/v1; TAG=--tag-requests
else
  AGENT_BASE=$BASE_URL; TAG=
fi

# web search for the web_real tasks
if [ "$N_WEB" -gt 0 ] && [ "$WEB_SEARCH" = searxng ]; then
  curl -sf --max-time 3 http://127.0.0.1:28011/health >/dev/null || "$EVALS/websearch/run_webtools.sh" start
  SPORT=$(free_port)
  SERP_LOG=$OUT/serp_requests.jsonl python3 "$CE/serp_adapter.py" "$SPORT" http://127.0.0.1:28011 >> "$OUT/serp_adapter.log" 2>&1 &
  PIDS+=($!)
  for _ in $(seq 30); do curl -sf --max-time 2 "http://127.0.0.1:$SPORT/health" >/dev/null && break; sleep 0.2; done
  export SERP_API_URL=http://127.0.0.1:$SPORT/search SERP_DEV_KEY=local-searxng-adapter
fi
export WEB_REAL_CACHE_DIR=$OUT/web_real_cache

# simulated user for multi_turn
UA_BASE= UA_MODEL=
if [ "$N_MT" -gt 0 ]; then
  case "$USER_SIM" in
    relay) UA_BASE=${USER_SIM_BASE_URL:-http://127.0.0.1:28400/v1}; UA_MODEL=${USER_SIM_MODEL:-claude-usersim}
           curl -sf --max-time 3 "${UA_BASE%/v1}/health" >/dev/null || { echo "user-sim relay not up at $UA_BASE (start ~/evals/usersim/start_shim.sh; workers: ~/evals/claweval/USERSIM_CLAWEVAL.txt) or set USER_SIM=self|off"; exit 1; } ;;
    self)  UA_BASE=${AGENT_BASE%/v1}; [ -n "$TAG" ] && UA_BASE=$UA_BASE/t/usersim; UA_BASE=$UA_BASE/v1; UA_MODEL=$MODEL_ID ;;
  esac
fi

# claw-eval config (judge = deferred stand-in; no external API anywhere)
CFG=$OUT/claweval_config.yaml
"$PY" - "$CFG" "$MODEL_ID" "$AGENT_BASE" "$UA_BASE" "$UA_MODEL" "$IMAGE" <<'EOF'
import json, os, sys, yaml
cfg, model, base, ua_base, ua_model, image = sys.argv[1:7]
e = os.environ
extra = {}
if e.get("TOP_P"): extra["top_p"] = float(e["TOP_P"])
if e.get("TOP_K"): extra["top_k"] = int(e["TOP_K"])
if e.get("MAX_TOKENS"): extra["max_tokens"] = int(e["MAX_TOKENS"])
mods = [m.strip() for m in e.get("INPUT_MODALITIES", "text").split(",") if m.strip()]
c = {"model": {"api_key": "EMPTY", "base_url": base, "model_id": model, "context_window": int(e["CTX"]),
               "temperature": float(e["TEMPERATURE"]) if e.get("TEMPERATURE") else 0.0,
               "input_modalities": mods, "extra_body": extra or None},
     "judge": {"api_key": "deferred-no-api", "base_url": "http://deferred.invalid/v1",
               "model_id": "google/gemini-3-flash-preview", "enabled": True},
     "sandbox": {"enabled": True, "image": image, "cpu_limit": float(e.get("SANDBOX_CPUS", "2")),
                 "memory_limit": e.get("SANDBOX_MEM", "4g")},
     "user_agent_model": {"api_key": "EMPTY", "base_url": ua_base or "http://user-sim-disabled.invalid/v1",
                          "model_id": ua_model or "none"},
     "defaults": {"trace_dir": "traces", "tasks_dir": "tasks"}}
yaml.safe_dump(c, open(cfg, "w"), sort_keys=False, allow_unicode=True)
EOF

echo "claweval $VARIANT label=$LABEL model=$MODEL_ID conc=$CONC tasks=$N_SEL (multi_turn $N_MT, web_real $N_WEB) n_runs=$N_RUNS sampling: T=$TEMPERATURE top_p=$TOP_P top_k=$TOP_K max_tokens=${MAX_TOKENS:-server} ctx=$CTX timeout_scale=$TIMEOUT_SCALE user_sim=$USER_SIM${UA_BASE:+($UA_BASE)} web_search=$WEB_SEARCH api=$AGENT_BASE image=$IMAGE" | tee -a "$OUT/run.log"
cd "$CE/repo"
ARGS=(--out "$OUT" --variant "$VARIANT" --config "$CFG" --model "$MODEL_ID" --base-url "$AGENT_BASE" --conc "$CONC"
      --n-runs "$N_RUNS" --port-base-offset "$PORT_BASE_OFFSET" --image "$IMAGE")
[ -n "${TASKS:-}" ] && ARGS+=(--tasks "$TASKS")
[ -n "${EXCLUDE_TASKS:-}" ] && ARGS+=(--exclude "$EXCLUDE_TASKS")
[ "${LIMIT:-0}" != 0 ] && ARGS+=(--limit "$LIMIT")
[ "${RETRY_ERRORS:-0}" = 1 ] && ARGS+=(--retry-errors)
[ "$USER_SIM" = off ] && ARGS+=(--no-multi-turn)
[ -n "$TAG" ] && ARGS+=("$TAG")
# full agent/grader logs -> run.log; progress lines -> stdout
{ "$PY" -u "$CE/claweval_driver.py" run "${ARGS[@]}" 2>&1 || echo "[claweval] driver exited with $?"; } | tee -a "$OUT/run.log" \
  | grep --line-buffered -E '^  \[[0-9]+/[0-9]+\] |^\[claweval\]' || true
"$PY" "$CE/claweval_driver.py" finish --out "$OUT" --n-runs "$N_RUNS" --variant "$VARIANT" 2>&1 | tee -a "$OUT/run.log"
