#!/bin/bash
# Where an MTP pass goes on the fused build (Ornith p300x2, step-7 env, MTP k = 3 adaptive): MTP_STAGES medians
# (draft = one head step incl. its id read, pass = the k+1-row verify, commit, rows = head K/V rows), real text 1 user.
export PATH=$HOME/ornith-fusion/bin:$HOME/.local/bin:$PATH MODEL_ID=ornith-ai/Ornith-1.5-35B-A3B
R=$HOME/ornith-fusion/runs; T=$HOME/tt-metal-bringups/models/autoports/ornith_ai_ornith_1_5_35b_a3b/fusion/tools
IMG=tt-model/ornith-1.5-35b-a3b-blackhole-fast:0d330be83ac1
while systemctl --user is-active -q ornith-publish2 || systemctl --user is-active -q cn-publish2 || chips_busy; do sleep 60; done
echo "mtp stages start $(date +%T)"
E="--env QWEN35_MOE_FUSED_DECODE=1 --env QWEN35_DECODE_REPLICATED=1 --env QWEN35_GDN_DECODE_OPS=1 --env QWEN35_ATTN_DECODE_OPS=1 --env QWEN35_ROWS_TOPK=1"
FB="--env QWEN35_FABRIC_BOUNDARY=1 --env QWEN35_FB_MAX_ROWS=4 --env QWEN35_FB_CONSUMER=1 --env QWEN35_FB_ROUTER=1 --env QWEN35_MOE_SPLIT_PAIR=1"
L=mtpst_k3; rm -rf $R/$L
MODELS=1 KEEP=1 ROWS="128:128:1:4" $T/milestone.sh $L orn-mtp $IMG p300x2 $E $FB --env QWEN35_MTP=3 && {
  URL=http://localhost:20000 python3 $T/realtext.py 384 | tee $R/$L/realtext.txt | tail -n 1
  docker logs orn-mtp 2>&1 | grep -E "MTP_STATS|MTP_STAGES" | tail -n 6 | cut -c 1-400 | tee $R/$L/stages.txt
  docker rm -f orn-mtp >/dev/null; }
echo "mtp stages done $(date +%T)"
