#!/usr/bin/env bash
# run_webtools.sh start|stop|status|restart|fg
# Starts SearXNG (docker, 127.0.0.1:28010) if needed and the webtools search/fetch service (127.0.0.1:28011) as the
# user systemd unit `evals-webtools` (survives the Claude session). Env passed through: SEARCH_QPS (1.0),
# BLOCKLIST (full|bench|none, default full), FETCH_TIMEOUT (30), FETCH_PARALLEL (12), ENGINES_EN, ENGINES_ZH, ...
set -euo pipefail
HERE=$(cd "$(dirname "$(readlink -f "$0")")" && pwd)
PY=$HOME/evals/text2/.venv/bin/python
PORT=${WEBTOOLS_PORT:-28011}
UNIT=evals-webtools
cmd=${1:-start}
up() { curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; }
case $cmd in
  fg) exec env PYTHONPATH="$HERE" "$PY" -m webtools.service --port "$PORT" ;;
  start)
    curl -sf http://127.0.0.1:${SEARXNG_PORT:-28010}/healthz >/dev/null 2>&1 || "$HERE/searxng/run_searxng.sh"
    if up; then echo "webtools already up on 127.0.0.1:$PORT"; exit 0; fi
    systemctl --user reset-failed $UNIT >/dev/null 2>&1 || true
    envs=(); for v in RESPECT_ROBOTS SEARCH_QPS SEARCH_EMPTY_WAIT_MAX BLOCKLIST FETCH_TIMEOUT FETCH_PARALLEL ENGINES_EN ENGINES_EN_FALLBACK ENGINES_ZH ENGINES_ZH_FALLBACK SEARXNG_URL; do
      [ -n "${!v:-}" ] && envs+=(--setenv=$v="${!v}"); done
    systemd-run --user --unit=$UNIT --setenv=HOME="$HOME" --setenv=PYTHONPATH="$HERE" "${envs[@]}" \
      -p WorkingDirectory="$HERE" -p CPUQuota=200% -p MemoryMax=8G \
      bash -c "exec $PY -m webtools.service --port $PORT >> $HERE/logs/service.log 2>&1" >/dev/null
    for i in $(seq 40); do up && { echo "webtools up on 127.0.0.1:$PORT (unit $UNIT)"; exit 0; }; sleep 0.5; done
    echo "webtools failed to start; see $HERE/logs/service.log"; tail -20 "$HERE/logs/service.log"; exit 1 ;;
  stop) systemctl --user stop $UNIT 2>/dev/null || true; echo stopped ;;
  restart) "$0" stop; sleep 1; "$0" start ;;
  status) systemctl --user --no-pager status $UNIT 2>/dev/null | head -5 || true; curl -s "http://127.0.0.1:$PORT/stats"; echo ;;
  *) sed -n 2,5p "$0"; exit 2 ;;
esac
