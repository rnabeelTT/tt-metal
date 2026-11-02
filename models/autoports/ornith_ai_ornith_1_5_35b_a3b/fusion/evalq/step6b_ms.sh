#!/bin/bash
# Step 6 served milestone (add + RMSNorm boundary op QWEN35_ADDNORM_DECODE + GDN head reader fix): Ornith p300x2, the same
# setup as attn_ms.sh (published fast image + gdn_wip models/, fast env + GDN + attention ops, MTP off); compared with
# attnms_f1 (step 5 state).
export PATH=$HOME/ornith-fusion/bin:$PATH
T=$HOME/tt-metal-gdnwip/models/autoports/ornith_ai_ornith_1_5_35b_a3b/fusion/tools
IMG=tt-model/ornith-1.5-35b-a3b-blackhole-fast:e7d45a33dd80
FAST="--env QWEN35_MOE_FUSED_DECODE=1 --env QWEN35_DECODE_REPLICATED=1 --env QWEN35_GDN_DECODE_OPS=1 --env QWEN35_MTP=0 --env QWEN35_ATTN_DECODE_OPS=1"
while chips_busy; do sleep 60; done
echo "step6 milestone start $(date +%T)"
L=step6b
MODELS=1 KEEP=1 ROWS="128:128:1:16 128:128:8:32 128:128:32:128 2048:128:1:8 8192:64:1:4 16384:128:1:2" $T/milestone.sh $L orn-s6b $IMG p300x2 \
  $FAST --env QWEN35_ADDNORM_DECODE=0 && {
  URL=http://localhost:20000 python3 $T/realtext.py 384 | tee $HOME/ornith-fusion/runs/$L/realtext.txt | tail -n 1
  docker rm -f orn-s6b >/dev/null; }
echo "== greedy step5 vs step6b"; python3 $T/greedy.py compare attnms_f1 step6b 2>&1 | tail -n 3
echo "step6 milestone done $(date +%T)"
