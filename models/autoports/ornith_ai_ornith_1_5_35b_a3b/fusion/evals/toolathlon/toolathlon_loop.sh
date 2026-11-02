#!/usr/bin/env bash
# Re-runs ~/evals/toolathlon_run.sh until every task has finished, sleeping until the
# NEXT_ELIGIBLE time the driver prints (quota window frees up / service busy / service down).
# Not installed anywhere (no cron/systemd); start it yourself, e.g. in tmux:
#   ~/evals/toolathlon/toolathlon_loop.sh verified ornith122b-ref http://localhost:20000/v1 ornith-ai/Ornith-1.5-122B 10
# Same args/env as toolathlon_run.sh. LOOP_MAX_ROUNDS=30 bounds it; exit codes:
#   0 done | 1/2 preflight or usage error (not retried) | other = gave up after LOOP_MAX_ROUNDS
# Cron-style alternative (also not installed): a daily entry that runs toolathlon_run.sh once
# a bit after the previous session's submit time; the driver exits 3 harmlessly if early.
set -uo pipefail
[ $# -eq 5 ] || { sed -n '2,10p' "$0"; exit 2; }
RUN=$(cd "$(dirname "$0")/.." && pwd)/toolathlon_run.sh
OUT=${RUNS_ROOT:-$HOME/ornith-fusion/runs}/$2/toolathlon
for round in $(seq 1 "${LOOP_MAX_ROUNDS:-30}"); do
  echo "[loop] round $round $(date -u +%FT%TZ)"
  "$RUN" "$@"
  rc=$?
  case $rc in
    0) echo "[loop] done"; exit 0;;
    1|2) echo "[loop] preflight/usage error (rc=$rc); stopping"; exit "$rc";;
  esac
  ne=$(grep -ao 'NEXT_ELIGIBLE [0-9]*' "$OUT/run.log" | tail -1 | awk '{print $2}')
  now=$(date +%s)
  wait=$(( ${ne:-$((now + 3600))} - now + ${LOOP_SLACK_S:-120} ))
  [ "$wait" -lt 60 ] && wait=60
  echo "[loop] rc=$rc; sleeping ${wait}s until $(date -u -d @$((now + wait)) +%FT%TZ)"
  sleep "$wait"
done
echo "[loop] giving up after ${LOOP_MAX_ROUNDS:-30} rounds"; exit 3
