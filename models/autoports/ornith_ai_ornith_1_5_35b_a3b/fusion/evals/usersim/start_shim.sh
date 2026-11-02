#!/usr/bin/env bash
# start_shim.sh [extra shim.py args]   -> starts the usersim shim in the background (idempotent), log + pid in spool/
# stop: kill $(cat ~/evals/usersim/spool/shim.pid)
set -euo pipefail
HERE=$(cd "$(dirname "$(readlink -f "$0")")" && pwd)
SPOOL=${USERSIM_SPOOL:-$HERE/spool}; PORT=${USERSIM_PORT:-28400}
mkdir -p "$SPOOL"
if curl -sf --max-time 3 "http://127.0.0.1:$PORT/health" >/dev/null; then echo "shim already up on :$PORT"; exit 0; fi
USERSIM_SPOOL=$SPOOL setsid nohup /usr/bin/python3 "$HERE/shim.py" --port "$PORT" "$@" >> "$SPOOL/shim.log" 2>&1 < /dev/null &
echo $! > "$SPOOL/shim.pid"
for _ in $(seq 50); do curl -sf --max-time 1 "http://127.0.0.1:$PORT/health" >/dev/null && { echo "shim up on :$PORT pid $(cat "$SPOOL/shim.pid")"; exit 0; }; sleep 0.2; done
echo "shim failed to start; see $SPOOL/shim.log"; exit 1
