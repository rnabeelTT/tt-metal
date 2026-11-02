#!/bin/bash
# Step 5 served milestone (attention decode ops, QWEN35_ATTN_DECODE_OPS): Ornith p300x2, the published fast image with
# gdn_wip's models/ mounted, the fast env + GDN ops on; MTP off (the faster single-user setting since step 4); attention
# ops off vs on, everything else identical: 1 / 8 / 32 users, TTFT rows 2K / 8K / 16K, real text, greedy + garble gate.
export PATH=$HOME/ornith-fusion/bin:$PATH
T=$HOME/tt-metal-gdnwip/models/autoports/ornith_ai_ornith_1_5_35b_a3b/fusion/tools
IMG=tt-model/ornith-1.5-35b-a3b-blackhole-fast:e7d45a33dd80
FAST="--env QWEN35_MOE_FUSED_DECODE=1 --env QWEN35_DECODE_REPLICATED=1 --env QWEN35_GDN_DECODE_OPS=1 --env QWEN35_MTP=0"
while chips_busy; do sleep 60; done
echo "attn milestone start $(date +%T)"
for F in 0 1; do
  L=attnms_f$F
  MODELS=1 KEEP=1 ROWS="128:128:1:16 128:128:8:32 128:128:32:128 2048:128:1:8 8192:64:1:4 16384:128:1:2" $T/milestone.sh $L orn-attn $IMG p300x2 \
    $FAST --env QWEN35_ATTN_DECODE_OPS=$F || continue
  URL=http://localhost:20000 python3 $T/realtext.py 384 | tee $HOME/ornith-fusion/runs/$L/realtext.txt | tail -n 1
  docker rm -f orn-attn >/dev/null
done
echo "== greedy off vs on"; python3 $T/greedy.py compare attnms_f0 attnms_f1 2>&1 | tail -n 3
echo "attn milestone done $(date +%T)"
