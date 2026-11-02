#!/bin/bash
set -o pipefail
R=$HOME/ornith-fusion/runs
wait_up() { until curl -sf localhost:20000/v1/models >/dev/null; do docker ps --format '{{.Names}}' | grep -qx $1 || { echo "$1 died"; return 1; }; sleep 10; done; }
VARGS=--enable-prefix-caching $R/pc_serve.sh pc-on >/dev/null
wait_up pc-on || exit 1
echo "== pc_on2 $(date +%T)"; $R/pc_tests.sh pc-on pc_on2
docker rm -f pc-on >/dev/null
echo "== done $(date +%T)"
