#!/usr/bin/env bash
# Toolathlon-Verified (108 tasks) through the authors' OFFICIAL evaluation service
# (hkust-nlp/Toolathlon eval_client.py, private mode), quota-aware and resumable across days.
#
# How the service reaches the model (client/EVAL_SERVICE_README.md "Private Mode",
# client/eval_client.py private_worker, client/simple_client_ws.py): the agent scaffold,
# MCP servers, task containers and graders all run on the service. Our side only opens
# OUTBOUND connections: HTTP to <host>:8080 (submit / poll / download results) and a
# WebSocket to <host>:8081 over which the service pushes chat/completions bodies; the local
# simple_client_ws.py POSTs each to <proxy>/v1/chat/completions and sends the response
# back. No inbound port, no public endpoint, no API key, no registration. Plain HTTP/WS
# (no TLS): prompts, tool outputs and our model's outputs (incl. reasoning) cross in clear.
#
# Usage:
#   toolathlon_run.sh <variant> <label> <base_url> <model_id> <concurrency>
#     variant: verified     (client checkout ~/evals/toolathlon/client = Toolathlon main,
#                            the Toolathlon-Verified release; the service runs ITS copy)
#     concurrency: service workers = parallel tasks = concurrent LLM calls (service max 10)
#   e.g. toolathlon_run.sh verified ornith122b-ref http://localhost:20000/v1 ornith-ai/Ornith-1.5-122B 10
#
# Results: ~/ornith-fusion/runs/<label>/toolathlon/ (state.json per-task status, jobs/NN_<job>/
#   = one service job each: client.log, server.log, ws_client.log, finalpool/<task>/ (status.json,
#   eval_res.json, traj_log.json, ...), proxy_requests.jsonl, results.tsv, summary.json, run.log).
# Final line: RESULT toolathlon pass=<x>/<108> ... when every task has finished; otherwise
#   PENDING toolathlon ... next_eligible=<UTC> and a non-zero exit (3 quota, 4 busy, 5 down,
#   6 another run holds the box-wide lock). Re-run the same command to resume; or loop:
#   ~/evals/toolathlon/toolathlon_loop.sh <same 5 args>  (sleeps until NEXT_ELIGIBLE; not installed)
#
# Quota (service: 180 min cumulative job time per client IP per ROLLING 24 h, one job at a
# time service-wide; duration = submit -> completion/cancel incl. container redeploy). Box-wide
# ledger ~/evals/toolathlon/quota_ledger.json (shared by all labels: same IP).
#   QUOTA_MIN=180 WINDOW_H=24   service limits (only change if the service changes)
#   SAFETY_MIN=10               kept unused; a job is cancelled at QUOTA_MIN-SAFETY_MIN-used
#   MIN_SESSION_MIN=30          don't submit with less budget than this
#   BUSY_WAIT_MIN=120           wait for someone else's job before giving up (exit 4)
#   MAX_ERROR_ATTEMPTS=2        tasks ending in preprocess/run failure are re-run until this
#   DRY_RUN=1                   show what would be submitted; no service calls except /models
#   SERVER_HOST=47.253.6.47 SERVER_PORT=8080 WS_PORT=8081
#
# Model / sampling (official defaults: 64K max output per turn, temperature/top_p NOT set
# = server defaults; 5400 s per task, 100 steps; the service fixes those, we only send
# model_params). Defaults per model id, applied only when unset:
#   MAX_TOKENS        Ornith: 131072 (card: "maximum token limit set to 128K"); else 65536
#   TEMPERATURE TOP_P TOP_K   Ornith: T=1.0 (card: "to reproduce the reported benchmarks"),
#                     top_p/top_k unset (= vLLM generation_config defaults, as the official
#                     setup leaves them to the provider); Qwen3.5: 1.0/0.95/20; Coder-Next: 1.0/0.95/40
#   Sent as model_params (so the service records them) AND forced by the proxy.
#   NB: the service waits at most 600 s per LLM call (simple_server_ws.py) and the agent then
#   retries the same call up to 10x (model_provider.py retry_times=10): a turn that needs
#   more than 600 s of decode at our speed fails. proxy_requests.jsonl latency shows this.
#   PROXY=1 PROXY_REASONING=keep  ~/evals/agent_proxy/llm_proxy.py between the WS client and
#                     the server (forces sampling, round-trips reasoning, logs every call)
#   TASKS=a,b  LIMIT=N  subsets of the 108 tasks (by name / first N)
#   RUNS_ROOT=~/ornith-fusion/runs
set -euo pipefail
if [ $# -ne 5 ]; then sed -n '2,58p' "$0"; exit 2; fi
VARIANT=$1 LABEL=$2 BASE_URL=${3%/} MODEL_ID=$4 CONC=$5
[ "$VARIANT" = verified ] || { echo "variant must be: verified"; exit 2; }
EVALS=$(cd "$(dirname "$0")" && pwd)
TDIR=$EVALS/toolathlon
CLIENT=$TDIR/client
PY=$TDIR/.venv/bin/python
[ -x "$PY" ] && [ -f "$CLIENT/eval_client.py" ] || { echo "client missing: $CLIENT / $PY"; exit 2; }
OUT=${RUNS_ROOT:-$HOME/ornith-fusion/runs}/$LABEL/toolathlon
mkdir -p "$OUT"
[ "$CONC" -ge 1 ] && [ "$CONC" -le 10 ] || { echo "concurrency must be 1..10 (service limit)"; exit 2; }

case "$MODEL_ID" in
  *Ornith*)     : "${MAX_TOKENS:=131072}" "${TEMPERATURE:=1.0}";;
  *Qwen3.5*)    : "${MAX_TOKENS:=65536}" "${TEMPERATURE:=1.0}" "${TOP_P:=0.95}" "${TOP_K:=20}";;
  *Coder-Next*) : "${MAX_TOKENS:=65536}" "${TEMPERATURE:=1.0}" "${TOP_P:=0.95}" "${TOP_K:=40}";;
  *)            : "${MAX_TOKENS:=65536}";;
esac
export TEMPERATURE=${TEMPERATURE:-} TOP_P=${TOP_P:-} TOP_K=${TOP_K:-} MAX_TOKENS
export SERVER_HOST=${SERVER_HOST:-47.253.6.47} SERVER_PORT=${SERVER_PORT:-8080} WS_PORT=${WS_PORT:-8081}

if [ "${PREFLIGHT:-1}" = 1 ]; then
  MODELS_JSON=$(curl -sf --max-time 20 "$BASE_URL/models") || { echo "server not reachable: $BASE_URL/models"; exit 1; }
  python3 -c 'import json,sys; ids=[m["id"] for m in json.loads(sys.argv[1])["data"]]; sys.exit(0 if sys.argv[2] in ids else "model %s not served; have %s" % (sys.argv[2], ids))' "$MODELS_JSON" "$MODEL_ID"
fi

# Task list: all task dirs of the Verified release (108), optionally filtered.
ls "$CLIENT/tasks/finalpool" | while read -r t; do [ -f "$CLIENT/tasks/finalpool/$t/task_config.json" ] && echo "$t"; done > "$OUT/all_tasks.txt"
python3 - "$OUT/all_tasks.txt" "$OUT/tasks.txt" <<'EOF'
import fnmatch, os, sys
ts = [l.strip() for l in open(sys.argv[1]) if l.strip()]
if os.environ.get("TASKS"):
    pats = os.environ["TASKS"].split(",")
    ts = [t for t in ts if any(fnmatch.fnmatch(t, p) for p in pats)]
if os.environ.get("LIMIT"):
    ts = ts[: int(os.environ["LIMIT"])]
prev = sys.argv[2]
if os.path.exists(prev):  # never shrink a label's task set silently
    old = [l.strip() for l in open(prev) if l.strip()]
    if old != ts:
        print(f"note: task list changed for this label ({len(old)} -> {len(ts)} tasks)")
open(prev, "w").write("\n".join(ts) + "\n")
EOF

# model_params sent to the service (replaces the scaffold's defaults wholesale, so keep
# parallel_tool_calls=true, which the default mode sends for non-OpenAI/Gemini models).
python3 - "$OUT/model_params.json" <<'EOF'
import json, os, sys
e = os.environ
d = {"max_tokens": int(e["MAX_TOKENS"]), "parallel_tool_calls": True}
if e.get("TEMPERATURE"): d["temperature"] = float(e["TEMPERATURE"])
if e.get("TOP_P"): d["top_p"] = float(e["TOP_P"])
if e.get("TOP_K"): d["top_k"] = int(e["TOP_K"])
json.dump(d, open(sys.argv[1], "w"), indent=1)
EOF

PROXY_PID=
cleanup() { [ -n "$PROXY_PID" ] && kill "$PROXY_PID" 2>/dev/null || true; }
trap cleanup EXIT
if [ "${PROXY:-1}" = 1 ]; then
  PORT=$(python3 -c 'import socket
for p in range(int("'"${PROXY_PORT_LO:-28650}"'"), int("'"${PROXY_PORT_HI:-28699}"'")+1):
    s=socket.socket()
    try: s.bind(("127.0.0.1",p)); print(p); break
    except OSError: pass
    finally: s.close()')
  SAMPLING=$(python3 -c 'import json,os; d={}
e=os.environ
if e.get("TEMPERATURE"): d["temperature"]=float(e["TEMPERATURE"])
if e.get("TOP_P"): d["top_p"]=float(e["TOP_P"])
if e.get("TOP_K"): d["top_k"]=int(e["TOP_K"])
print(json.dumps(d))')
  PROXY_LOG=$OUT/proxy_requests.jsonl PROXY_SAMPLING=$SAMPLING PROXY_MAX_TOKENS=$MAX_TOKENS \
    python3 "$EVALS/agent_proxy/llm_proxy.py" 127.0.0.1 "$PORT" "${BASE_URL%/v1}" >> "$OUT/proxy.log" 2>&1 &
  PROXY_PID=$!
  sleep 1.5
  kill -0 "$PROXY_PID" 2>/dev/null || { echo "proxy failed to start (see $OUT/proxy.log)"; exit 1; }
  LLM_BASE=http://127.0.0.1:$PORT/v1
else
  LLM_BASE=$BASE_URL
fi

echo "toolathlon $VARIANT label=$LABEL model=$MODEL_ID workers=$CONC service=$SERVER_HOST:$SERVER_PORT/ws:$WS_PORT max_tokens=$MAX_TOKENS sampling: T=$TEMPERATURE top_p=$TOP_P top_k=$TOP_K llm=$LLM_BASE quota=${QUOTA_MIN:-180}min/${WINDOW_H:-24}h safety=${SAFETY_MIN:-10}" | tee -a "$OUT/run.log"
set +e
"$PY" -u "$TDIR/toolathlon_driver.py" --out "$OUT" --label "$LABEL" --model "$MODEL_ID" --workers "$CONC" \
  --llm-base-url "$LLM_BASE" --server-host "$SERVER_HOST" --server-port "$SERVER_PORT" --ws-port "$WS_PORT" \
  --tasks-file "$OUT/tasks.txt" --model-params "$OUT/model_params.json" --client-dir "$CLIENT" --python "$PY" \
  --ledger "${TOOLATHLON_LEDGER:-$TDIR/quota_ledger.json}" --lock "${TOOLATHLON_LOCK:-$TDIR/.service.lock}" \
  2>&1 | tee -a "$OUT/run.log"
RC=${PIPESTATUS[0]}
set -e
python3 - "$OUT" "$RC" <<'EOF' | tee -a "$OUT/run.log"
import json, os, sys, datetime as dt
out, rc = sys.argv[1], int(sys.argv[2])
try:
    s = json.load(open(os.path.join(out, "summary.json")))
    st = json.load(open(os.path.join(out, "state.json")))
except (OSError, ValueError):
    print(f"PENDING toolathlon exit={rc} (no state yet; see run.log)")
    sys.exit(0)
lat = []
p = os.path.join(out, "proxy_requests.jsonl")
if os.path.exists(p):
    for l in open(p):
        try: r = json.loads(l)
        except ValueError: continue
        lat.append((r.get("latency_s") or 0, r.get("status"), r.get("completion_tokens") or 0, r.get("prompt_tokens") or 0))
calls = len(lat); over = sum(1 for x in lat if x[0] >= 590); bad = sum(1 for x in lat if x[1] != 200)
ctok = sum(x[2] for x in lat); ptok = max([x[3] for x in lat] or [0])
extra = (f"jobs={s['jobs']} service_min={s['service_minutes_this_label']} errors={s['errors']} "
         f"llm_calls={calls} non200={bad} calls_ge590s={over} completion_tokens={ctok} max_prompt_tokens={ptok}")
if s["pending"] == 0 and s["n_tasks"]:
    print(f"RESULT toolathlon pass={s['passed']}/{s['n_tasks']} ({100*s['pass_rate']:.1f}%) {extra}")
else:
    ne = st.get("next_eligible")
    ne = dt.datetime.fromtimestamp(ne, dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ") if ne else "-"
    print(f"PENDING toolathlon pass={s['passed']}/{s['n_tasks']} finished={s['finished']} pending={s['pending']} "
          f"exit={rc} next_eligible={ne} {extra}")
EOF
exit "$RC"
