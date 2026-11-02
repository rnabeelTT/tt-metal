#!/bin/bash
# Ornith one die (p150) with the general-m fused MoE (m = 4 up|gate columns per bank), after the GLM chain: the fast
# image + ornith_fusion's models/ mounted, MTP off (adaptive MTP made p150 single-user rows unrepeatable), fused MoE on
# (p150_m4) vs off (p150_ref = what the published fast bundle runs on one die). Same code otherwise.
export PATH=$HOME/ornith-fusion/bin:$PATH
T=$HOME/tt-metal-bringups/models/autoports/ornith_ai_ornith_1_5_35b_a3b/fusion/tools
IMG=tt-model/ornith-1.5-35b-a3b-blackhole-fast:e7d45a33dd80
while pgrep -fx "bash $HOME/ornith-fusion/runs/glm_chain.sh" >/dev/null || pgrep -fx "bash $HOME/ornith-fusion/runs/q122_chain.sh" >/dev/null || $HOME/ornith-fusion/bin/chips_busy; do sleep 60; done
echo "ornith p150 chain start $(date +%T)"
ROWS="128:128:1:16 128:128:8:32 128:128:32:128 2048:128:1:8 8192:64:1:4"
for cfg in "p150_m4:1" "p150_ref:0"; do
  L=${cfg%%:*}; F=${cfg#*:}
  MODELS=1 PORT=20000 ROWS="$ROWS" $T/milestone.sh $L orn-p150 $IMG p150 \
    --env QWEN35_MTP=0 --env QWEN35_MOE_FUSED_DECODE=$F --env QWEN35_DECODE_REPLICATED=1 || continue
done
python3 $T/greedy.py compare p150_ref p150_m4 2>&1 | tail -n 3
echo "ornith p150 chain done $(date +%T)"
