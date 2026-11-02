#!/bin/bash
# Qwen3-Coder-Next fused vs published, p300x2: greedy, bench rows, evals (one server at a time)
export PATH=$HOME/ornith-fusion/bin:$PATH MODEL_ID=Qwen/Qwen3-Coder-Next CACHE_NAME=qwen3-coder-next-blackhole
T=$HOME/tt-metal-codernext/models/autoports/ornith_ai_ornith_1_5_35b_a3b/fusion/tools
IMG=tt-model/qwen3-coder-next-blackhole:bd0ca927a46b
ROWS="128:128:1:8 2048:128:1:4 128:128:8:32 128:128:16:64 128:128:32:128 1024:256:32:128"
while [ -n "$(docker ps -q)" ]; do sleep 30; done
for cfg in "cn_new:1" "cn_pub:0"; do
  L=${cfg%%:*}; F=${cfg#*:}
  if [ $F = 1 ]; then E="--env QWEN35_MOE_FUSED_DECODE=1 --env QWEN35_DECODE_REPLICATED=1"; M=1; else E=""; M=0; fi
  MODELS=$M PORT=20000 GARBLE=0 KEEP=1 ROWS="$ROWS" $T/milestone.sh $L cn-srv $IMG p300x2 $E || continue
  URL=http://localhost:20000 MODEL_ID=$MODEL_ID python3 $T/realtext.py 384 | tee $HOME/ornith-fusion/runs/$L/realtext.txt
  PORT=20000 $T/eval_coder.sh $L 2>&1 | grep -E "^==|\|.*(pass|exact_match|strict|loose)"
  docker rm -f cn-srv >/dev/null
done
echo "cn chain done $(date +%T)"
