#!/bin/bash
set -o pipefail
R=$HOME/ornith-fusion/runs
while systemctl --user is-active --quiet pc-chain3; do sleep 30; done
wait_up() { until curl -sf localhost:20000/v1/models >/dev/null; do docker ps --format '{{.Names}}' | grep -qx $1 || { echo "$1 died"; docker logs $1 2>&1 | grep -v DEBUG | grep -E "Error|assert|Traceback" -A3 | tail -n 60 > $R/$1.boot_fail.log; return 1; }; sleep 10; done; }
docker rm -f pc-cn_on >/dev/null 2>&1
export MODEL_ID=Qwen/Qwen3-Coder-Next
VARGS=--enable-prefix-caching python3 $R/pc_serve2.py raahemnabeel/qwen3-coder-next pc-cn_on $HOME/tt-metal-codernext 2>/dev/null
if wait_up pc-cn_on; then
  docker logs pc-cn_on 2>&1 | grep -E "prefix cache:|Recurrent-state|KV token pool|Automatic prefix" | grep -v DEBUG | tail -n 5 > $R/pc_cn_on.boot.txt
  echo "== pc_cn_on $(date +%T)"; $R/pc_tests.sh pc-cn_on pc_cn_on
fi
docker rm -f pc-cn_on >/dev/null 2>&1
# Ornith-35B: the 8k chat with full replies, on then off
export MODEL_ID=ornith-ai/Ornith-1.5-35B-A3B
T=$HOME/tt-metal-bringups/models/autoports/ornith_ai_ornith_1_5_35b_a3b/fusion/tools
for m in on off; do
  if [ $m = on ]; then VARGS=--enable-prefix-caching $R/pc_serve.sh pc-o35 >/dev/null; else VARGS=--enable-prefix-caching $R/pc_serve.sh pc-o35 --env QWEN35_PREFIX_STATE_SLOTS=0 >/dev/null; fi
  wait_up pc-o35 || continue
  echo "== o35 chat $m $(date +%T)"; (cd $T && TAG=o35_$m python3 prefix_bench.py chat 8192 8 | tee $R/pc_o35_chat_$m.log)
  docker rm -f pc-o35 >/dev/null
done
echo "== done $(date +%T)"
