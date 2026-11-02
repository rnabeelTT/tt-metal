#!/bin/bash
# p300 (2 dies) prefix-cache check: Ornith-35B and Coder-Next, caching on
set -o pipefail
R=$HOME/ornith-fusion/runs
T=$HOME/tt-metal-bringups/models/autoports/ornith_ai_ornith_1_5_35b_a3b/fusion/tools
wait_up() { until curl -sf localhost:20000/v1/models >/dev/null; do docker ps --format '{{.Names}}' | grep -qx $1 || { echo "$1 died"; docker logs $1 2>&1 | grep -v DEBUG | grep -E "Error|assert|Traceback" -A3 | tail -n 60 > $R/$1.boot_fail.log; return 1; }; sleep 10; done; }
for spec in "o35 raahemnabeel/ornith-1.5-35b-a3b $HOME/tt-metal-bringups" "cn raahemnabeel/qwen3-coder-next $HOME/tt-metal-codernext"; do
  set -- $spec
  PROFILE=p300 VARGS=--enable-prefix-caching python3 $R/pc_serve2.py $2 pc-p300-$1 $3 2>/dev/null
  if wait_up pc-p300-$1; then
    mkdir -p $R/pc_p300_$1
    docker logs pc-p300-$1 2>&1 | grep -E "prefix cache:|Recurrent-state|KV token pool" | grep -v DEBUG > $R/pc_p300_$1/boot.txt
    echo "== p300 $1 $(date +%T)"
    (cd $T && TAG=p300_$1 python3 prefix_bench.py repeat 8192 32768 | tee $R/pc_p300_$1/repeat.log; TAG=p300_$1 python3 prefix_bench.py chat 8192 4 | tee $R/pc_p300_$1/chat.log)
  fi
  docker rm -f pc-p300-$1 >/dev/null
done
echo "== done $(date +%T)"
