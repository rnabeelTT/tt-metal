#!/usr/bin/env bash
# tau2.sh tau2 <label> <base_url> <model_id> <concurrency>      (called by text2_run.sh)
# TAU2-Bench, official harness (sierra-research/tau2-bench @ c5b2d22, main as of Qwen3.5's release, airline/retail/
# telecom base tasks identical to v0.1.3 which Qwen used for its own Qwen3-Max submissions) + PR #94 (Anthropic's
# Claude Opus 4.5 airline policy/task fixes) applied as a local commit (~/evals/tau2/pr94_opus45_airline_fixes.patch). Shim spec: ~/evals/tau2/USER_SIM_SHIM.md
#
# Agent = our model (litellm openai/<model_id> at <base_url>), thinking on (reasoning not fed back into history).
# User simulator = an OpenAI-compatible endpoint we do NOT host here (Claude-agent shim, ~/evals/tau2/USER_SIM_SHIM.md):
#   USER_SIM_BASE_URL (required)  USER_SIM_MODEL (gpt-4.1-2025-04-14: the official/Qwen user model it stands in for)
#   USER_SIM_TEMPERATURE (0.0, official)
# Env: DOMAINS ("airline retail telecom") TRIALS (4, official leaderboard / Qwen's own submissions) MAX_STEPS (200)
#      MAX_ERRORS (10) SEED (300) MAX_TOKENS (32768) TEMPERATURE TOP_P TOP_K (model defaults as text2) PRESENCE_PENALTY
#      LIMIT (first N tasks per domain, smoke) TASK_IDS ("id1 id2", one domain) REQUEST_TIMEOUT (14400)
#      SIM_RETRIES (2) RETRY_FAILED (0; 1 = drop harness-exception sims from the save files and re-run them)
# Output: ${RUNS_ROOT:-~/ornith-fusion/runs}/<label>/tau2/{airline,retail,telecom}.json (tau2 Results, resumable),
#         <domain>.log, summary.json, request_counts.json; last line RESULT tau2 pass^1=<x> n=<tasks>
set -euo pipefail
[ $# -eq 5 ] || { sed -n 2,19p "$0"; exit 2; }
BENCH=$1 LABEL=$2 BASE_URL=${3%/} MODEL_ID=$4 CONC=$5
T2=$HOME/evals/tau2
PY=$T2/.venv/bin/python
OUT=${RUNS_ROOT:-$HOME/ornith-fusion/runs}/$LABEL/tau2
DOMAINS=${DOMAINS:-airline retail telecom}
TRIALS=${TRIALS:-4}
case "$MODEL_ID" in *[Oo]rnith*35[Bb]*) DT=1.0;; *) DT=0.6;; esac
TEMPERATURE=${TEMPERATURE:-$DT} TOP_P=${TOP_P:-0.95} TOP_K=${TOP_K:-20} MAX_TOKENS=${MAX_TOKENS:-32768}
: "${USER_SIM_BASE_URL:?set USER_SIM_BASE_URL to the OpenAI-compatible user-simulator shim (see ~/evals/tau2/USER_SIM_SHIM.md)}"
USER_SIM_MODEL=${USER_SIM_MODEL:-gpt-4.1-2025-04-14}
USER_SIM_TEMPERATURE=${USER_SIM_TEMPERATURE:-0.0}

export LITELLM_LOCAL_MODEL_COST_MAP=True   # no network fetch of litellm's price table
export TAU2_DATA_DIR=$T2/tau2-bench/data
mkdir -p "$OUT"

if [ "${SKIP_SERVER_CHECK:-0}" != 1 ]; then
  M=$(curl -sf --max-time 20 "$BASE_URL/models") || { echo "server not reachable: $BASE_URL/models"; exit 1; }
  echo "$M" | "$PY" -c 'import json,sys; ids=[m["id"] for m in json.load(sys.stdin)["data"]]; sys.exit(0 if sys.argv[1] in ids else "model %s not served; have %s" % (sys.argv[1], ids))' "$MODEL_ID"
fi

AGENT_ARGS=$("$PY" - "$BASE_URL" "$TEMPERATURE" "$TOP_P" "$TOP_K" "$MAX_TOKENS" "${PRESENCE_PENALTY:-}" "${REQUEST_TIMEOUT:-14400}" <<'EOF'
import json, sys
base, t, p, k, mt, pp, to = sys.argv[1:8]
a = {"api_base": base, "api_key": "EMPTY", "temperature": float(t), "top_p": float(p), "max_tokens": int(mt),
     "timeout": float(to)}
if int(k) > 0:
    a["extra_body"] = {"top_k": int(k)}
if pp:
    a["presence_penalty"] = float(pp)
print(json.dumps(a))
EOF
)
USER_ARGS=$("$PY" -c 'import json,sys; print(json.dumps({"api_base": sys.argv[1], "api_key": "EMPTY", "temperature": float(sys.argv[2]), "timeout": float(sys.argv[3])}))' "${USER_SIM_BASE_URL%/}" "$USER_SIM_TEMPERATURE" "${USER_SIM_TIMEOUT:-3600}")

cat > "$OUT/config.json" <<EOF
{"bench": "tau2", "label": "$LABEL", "model": "$MODEL_ID", "base_url": "$BASE_URL", "concurrency": $CONC,
 "domains": "$DOMAINS", "trials": $TRIALS, "agent_llm_args": $AGENT_ARGS,
 "user_sim_model": "$USER_SIM_MODEL", "user_sim_base_url": "$USER_SIM_BASE_URL", "user_sim_temperature": $USER_SIM_TEMPERATURE,
 "max_steps": ${MAX_STEPS:-200}, "max_errors": ${MAX_ERRORS:-10}, "seed": ${SEED:-300}, "limit": "${LIMIT:-}",
 "tau2_commit": "$(git -C $T2/tau2-bench rev-parse HEAD)", "time": "$(date -Is)"}
EOF

for D in $DOMAINS; do
  SAVE=$OUT/$D.json   # tau2 appends .json to --save-to
  if [ "${RETRY_FAILED:-0}" = 1 ] && [ -f "$SAVE" ]; then
    "$PY" - "$SAVE" <<'EOF'
import json, sys
p = sys.argv[1]; r = json.load(open(p))
keep = [s for s in r["simulations"] if "harness_exception" not in str(((s.get("reward_info") or {}).get("info") or {}).get("note", ""))]
print(f"RETRY_FAILED: dropping {len(r['simulations']) - len(keep)} harness-exception simulations from {p}")
r["simulations"] = keep; json.dump(r, open(p, "w"), indent=2)
EOF
  fi
  EXTRA=()
  [ -n "${LIMIT:-}" ] && EXTRA+=(--num-tasks "$LIMIT")
  [ -n "${TASK_IDS:-}" ] && EXTRA+=(--task-ids $TASK_IDS)
  echo "== tau2 $D: agent openai/$MODEL_ID @ $BASE_URL, user openai/$USER_SIM_MODEL @ $USER_SIM_BASE_URL, trials $TRIALS, conc $CONC $(date +%T)"
  # `yes` answers tau2's interactive "resume?" prompt when the save file exists (resume = official behaviour).
  { yes y 2>/dev/null || true; } | "$PY" "$T2/run_tau2.py" --domain "$D" --task-split-name base \
    --agent llm_agent --agent-llm "openai/$MODEL_ID" --agent-llm-args "$AGENT_ARGS" \
    --user user_simulator --user-llm "openai/$USER_SIM_MODEL" --user-llm-args "$USER_ARGS" \
    --num-trials "$TRIALS" --max-steps "${MAX_STEPS:-200}" --max-errors "${MAX_ERRORS:-10}" \
    --seed "${SEED:-300}" --max-concurrency "$CONC" --save-to "$OUT/$D" --log-level "${LOG_LEVEL:-ERROR}" \
    "${EXTRA[@]}" > "$OUT/$D.log" 2>&1 || { echo "tau2 run for $D failed; tail of $OUT/$D.log:"; tail -20 "$OUT/$D.log"; exit 1; }
  grep -E "harness\]|Error running" "$OUT/$D.log" | tail -3 || true
done
"$PY" "$T2/summarize.py" "$OUT" $DOMAINS
