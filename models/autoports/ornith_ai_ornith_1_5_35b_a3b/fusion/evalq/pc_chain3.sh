#!/bin/bash
# on / off prefix-cache tests for Coder-Next, Qwen3.5-122B, Ornith-1.5-122B, then Ornith-35B chat with full replies
set -o pipefail
R=$HOME/ornith-fusion/runs
wait_up() { until curl -sf localhost:20000/v1/models >/dev/null; do docker ps --format '{{.Names}}' | grep -qx $1 || { echo "$1 died"; docker logs $1 2>&1 | grep -v DEBUG | tail -n 60 > $R/$1.boot_fail.log; return 1; }; sleep 10; done; }
run() {  # run <label> <bundle> <tree> <model id> on|off
  L=$1; B=$2; TREE=$3; export MODEL_ID=$4; MODE=$5
  if [ $MODE = on ]; then VARGS=--enable-prefix-caching python3 $R/pc_serve2.py $B pc-$L $TREE 2>/dev/null
  else VARGS=--enable-prefix-caching python3 $R/pc_serve2.py $B pc-$L $TREE --env QWEN35_PREFIX_STATE_SLOTS=0 2>/dev/null; fi
  wait_up pc-$L || return 1
  docker logs pc-$L 2>&1 | grep -E "prefix cache:|Recurrent-state|KV token pool|Automatic prefix" | grep -v DEBUG | tail -n 5 > $R/pc_$L.boot.txt
  echo "== pc_$L $(date +%T)"; $R/pc_tests.sh pc-$L pc_$L
  docker rm -f pc-$L >/dev/null
}
run cn_on raahemnabeel/qwen3-coder-next $HOME/tt-metal-codernext Qwen/Qwen3-Coder-Next on
run cn_off raahemnabeel/qwen3-coder-next $HOME/tt-metal-codernext Qwen/Qwen3-Coder-Next off
run q122_on raahemnabeel/qwen3-5-122b-a10b $HOME/tt-metal-q122 Qwen/Qwen3.5-122B-A10B on
run q122_off raahemnabeel/qwen3-5-122b-a10b $HOME/tt-metal-q122 Qwen/Qwen3.5-122B-A10B off
run o122_on raahemnabeel/ornith-1.5-122b $HOME/tt-metal-q122 ornith-ai/Ornith-1.5-122B on
run o122_off raahemnabeel/ornith-1.5-122b $HOME/tt-metal-q122 ornith-ai/Ornith-1.5-122B off
echo "== done $(date +%T)"
