#!/bin/bash
# Step 7 on the ports (qwen36 tree level with ornith_fusion + the model-level hooks, uncommitted in tt-metal-codernext /
# tt-metal-q122): served p300x2 milestones with the step-7 env on top of the v2 env; vs card_cnv2_p300x2 / card_q122v2_p300x2.
export PATH=$HOME/ornith-fusion/bin:$PATH
R=$HOME/ornith-fusion/runs
until grep -q "evals done" $R/ov3_evals.log; do sleep 60; done
docker rm -f orn-ev >/dev/null 2>&1
while chips_busy; do sleep 60; done
echo "ports step7 start $(date +%T)"
OPS="--env QWEN35_MOE_FUSED_DECODE=1 --env QWEN35_DECODE_REPLICATED=1 --env QWEN35_GDN_DECODE_OPS=1 --env QWEN35_ATTN_DECODE_OPS=1 --env QWEN35_MTP=0"
S7="--env QWEN35_ROWS_TOPK=1 --env QWEN35_FABRIC_BOUNDARY=1 --env QWEN35_FB_MAX_ROWS=4 --env QWEN35_FB_CONSUMER=1 --env QWEN35_FB_ROUTER=1 --env QWEN35_MOE_SPLIT_PAIR=1"
ROWS="128:128:1:8 128:128:2:8 128:128:4:16 128:128:8:32 128:128:32:128 2048:128:1:4 8192:64:1:2"
for M in cn q122; do
  if [ $M = cn ]; then
    export MODEL_ID=Qwen/Qwen3-Coder-Next CACHE_NAME=qwen3-coder-next-blackhole; TREE=tt-metal-codernext
    IMG=tt-model/qwen3-coder-next-blackhole-fast-v2:12d907552029; BASE=card_cnv2_p300x2; G=0; X=""
  else
    export MODEL_ID=Qwen/Qwen3.5-122B-A10B CACHE_NAME=qwen3-5-122b-a10b; TREE=tt-metal-q122
    IMG=tt-model/qwen3-5-122b-a10b-blackhole-fast-v2:3a9fa14cc3dd; BASE=card_q122v2_p300x2; G=1; X="--env VLLM_ENGINE_READY_TIMEOUT_S=14400"
  fi
  T=$HOME/$TREE/models/autoports/ornith_ai_ornith_1_5_35b_a3b/fusion/tools
  L=${M}s7; rm -rf $R/$L
  MODELS=1 GARBLE=$G KEEP=1 ROWS="$ROWS" $T/milestone.sh $L $M-s7 $IMG p300x2 $OPS $S7 $X && {
    docker logs $M-s7 2>&1 | grep -E "rows top-k|fabric boundary|split" | sort -u | head -5
    URL=http://localhost:20000 MODEL_ID=$MODEL_ID python3 $T/realtext.py 384 | tee $R/$L/realtext.txt | tail -n 1
    docker rm -f $M-s7 >/dev/null; }
  echo "== $M greedy $BASE vs $L"; python3 $T/greedy.py compare $BASE $L 2>&1 | tail -n 3
done
echo "ports step7 done $(date +%T)"
