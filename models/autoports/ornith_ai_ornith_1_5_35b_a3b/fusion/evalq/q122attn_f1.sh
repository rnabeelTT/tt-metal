#!/bin/bash
# 122B served attention ops on (rerun after the per-call output fix; f0 measured in ports_attn)
export PATH=$HOME/ornith-fusion/bin:$PATH
R=$HOME/ornith-fusion/runs
while pgrep -f "addnorm_layer.sh" >/dev/null || chips_busy; do sleep 60; done
export MODEL_ID=Qwen/Qwen3.5-122B-A10B CACHE_NAME=qwen3-5-122b-a10b
T=$HOME/tt-metal-q122/models/autoports/ornith_ai_ornith_1_5_35b_a3b/fusion/tools
L=q122attn_f1; rm -rf $R/$L
MODELS=1 KEEP=1 ROWS="128:128:1:8 128:128:8:32 128:128:32:128 2048:128:1:4 8192:64:1:2" $T/milestone.sh $L q122-attn tt-model/ornith-1.5-35b-a3b-blackhole-fast:e7d45a33dd80 p300x2 \
  --env QWEN35_MOE_FUSED_DECODE=1 --env QWEN35_DECODE_REPLICATED=1 --env VLLM_ENGINE_READY_TIMEOUT_S=14400 --env QWEN35_MTP=0 --env QWEN35_GDN_DECODE_OPS=1 --env QWEN35_ATTN_DECODE_OPS=1 && {
  URL=http://localhost:20000 python3 $T/realtext.py 384 | tee $R/$L/realtext.txt | tail -n 1
  docker rm -f q122-attn >/dev/null; }
echo "== q122 greedy off vs on"; python3 $T/greedy.py compare q122attn_f0 q122attn_f1 2>&1 | tail -n 3
echo "q122attn_f1 done $(date +%T)"
