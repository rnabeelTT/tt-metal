#!/bin/bash
# Quality + 2-chip checkpoint of the day's dev configuration (gdn_wip models/, v2 env + fabric boundary, rows top-k,
# consumer / router fusion, MoE split-pair, 2-4 row boundary, head value split): (1) p300 (2 chips) boot + bench rows,
# (2) p300x2 scorecard subsets one request at a time (CONC=1: the 1-row decode path) + garble.
export PATH=$HOME/ornith-fusion/bin:$PATH MODEL_ID=ornith-ai/Ornith-1.5-35B-A3B
T=$HOME/tt-metal-gdnwip/models/autoports/ornith_ai_ornith_1_5_35b_a3b/fusion/tools
R=$HOME/ornith-fusion/runs
IMG=tt-model/ornith-1.5-35b-a3b-blackhole-fast:e7d45a33dd80
DEV="--env QWEN35_MOE_FUSED_DECODE=1 --env QWEN35_DECODE_REPLICATED=1 --env QWEN35_GDN_DECODE_OPS=1 --env QWEN35_MTP=0 --env QWEN35_ATTN_DECODE_OPS=1 --env QWEN35_FABRIC_BOUNDARY=1 --env QWEN35_ROWS_TOPK=1 --env QWEN35_FB_CONSUMER=1 --env QWEN35_FB_ROUTER=1 --env QWEN35_MOE_SPLIT_PAIR=1 --env QWEN35_FB_MAX_ROWS=4"
echo "devq start $(date +%T)"
MODELS=1 ROWS="128:128:1:8 128:128:2:8 128:128:4:16 128:128:8:32 128:128:32:128" $T/milestone.sh devq_p300 orn-dq $IMG p300 $DEV
MODELS=1 GARBLE=1 KEEP=1 ROWS="128:128:1:4" $T/milestone.sh devq_p300x2 orn-dq $IMG p300x2 $DEV && {
  CONC=1 PORT=20000 RUNS=$R $T/eval.sh devq_p300x2 2>&1 | grep -E "^==|\|.*(acc|exact|strict|loose|pass)"
  docker rm -f orn-dq >/dev/null; }
echo "devq done $(date +%T)"
