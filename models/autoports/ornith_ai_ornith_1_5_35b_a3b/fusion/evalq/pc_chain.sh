#!/bin/bash
set -o pipefail
R=$HOME/ornith-fusion/runs
wait_up() { until curl -sf localhost:20000/v1/models >/dev/null; do docker ps --format '{{.Names}}' | grep -qx $1 || { echo "$1 died"; return 1; }; sleep 10; done; }
echo "== pc_on $(date +%T)"; $R/pc_tests.sh pc-on pc_on
docker rm -f pc-on >/dev/null
VARGS=--enable-prefix-caching $R/pc_serve.sh pc-off --env QWEN35_PREFIX_STATE_SLOTS=0 >/dev/null
wait_up pc-off || exit 1
echo "== pc_off $(date +%T)"; $R/pc_tests.sh pc-off pc_off
docker rm -f pc-off >/dev/null
echo "== done $(date +%T)"
