#!/bin/bash
# Step 4 served milestone (GDN decode mixer ops, QWEN35_GDN_DECODE_OPS): Ornith p300x2, the published fast image with
# ornith_fusion's models/ mounted, the fast bundle's env; ops off vs on, everything else identical. MTP on (the shipped
# default: adaptive k = 3) rows: 1 / 8 / 32 users + TTFT rows + real text; then MTP off (the -longctx setting) single
# user + real text. Greedy + garble gate on every server; greedy compared off vs on.
export PATH=$HOME/ornith-fusion/bin:$PATH
T=$HOME/tt-metal-bringups/models/autoports/ornith_ai_ornith_1_5_35b_a3b/fusion/tools
IMG=tt-model/ornith-1.5-35b-a3b-blackhole-fast:e7d45a33dd80
FAST="--env QWEN35_MOE_FUSED_DECODE=1 --env QWEN35_DECODE_REPLICATED=1"
while chips_busy; do sleep 60; done
echo "gdn milestone start $(date +%T)"
for F in 0 1; do
  L=gdnms_mtp_f$F
  MODELS=1 KEEP=1 ROWS="128:128:1:16 128:128:8:32 128:128:32:128 2048:128:1:8 8192:64:1:4" $T/milestone.sh $L orn-gdn $IMG p300x2 \
    $FAST --env QWEN35_GDN_DECODE_OPS=$F || continue
  URL=http://localhost:20000 python3 $T/realtext.py 384 | tee $HOME/ornith-fusion/runs/$L/realtext.txt | tail -n 1
  docker rm -f orn-gdn >/dev/null
  L=gdnms_nomtp_f$F
  MODELS=1 KEEP=1 ROWS="128:128:1:16 16384:128:1:2" $T/milestone.sh $L orn-gdn $IMG p300x2 \
    $FAST --env QWEN35_MTP=0 --env QWEN35_GDN_DECODE_OPS=$F || continue
  URL=http://localhost:20000 python3 $T/realtext.py 384 | tee $HOME/ornith-fusion/runs/$L/realtext.txt | tail -n 1
  docker rm -f orn-gdn >/dev/null
done
for m in mtp nomtp; do echo "== greedy $m off vs on"; python3 $T/greedy.py compare gdnms_${m}_f0 gdnms_${m}_f1 2>&1 | tail -n 3; done
echo "gdn milestone done $(date +%T)"
