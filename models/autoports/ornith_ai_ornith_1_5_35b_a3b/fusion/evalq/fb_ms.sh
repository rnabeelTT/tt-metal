#!/bin/bash
# Fabric boundary (QWEN35_FABRIC_BOUNDARY=1: one-row decode boundaries as one op each) served milestone: Ornith p300x2,
# the step-6 setup (published fast image + gdn_wip models/, fast env + GDN + attention ops, MTP off); vs step6b.
export PATH=$HOME/ornith-fusion/bin:$PATH
T=$HOME/tt-metal-gdnwip/models/autoports/ornith_ai_ornith_1_5_35b_a3b/fusion/tools
IMG=tt-model/ornith-1.5-35b-a3b-blackhole-fast:e7d45a33dd80
FAST="--env QWEN35_MOE_FUSED_DECODE=1 --env QWEN35_DECODE_REPLICATED=1 --env QWEN35_GDN_DECODE_OPS=1 --env QWEN35_MTP=0 --env QWEN35_ATTN_DECODE_OPS=1"
echo "fb milestone start $(date +%T)"
L=${L:-fbms}
MODELS=1 KEEP=1 ROWS="${ROWS_MS:-128:128:1:16 128:128:8:32 128:128:32:128 2048:128:1:8 8192:64:1:4 16384:128:1:2}" $T/milestone.sh $L orn-fb $IMG p300x2 \
  $FAST --env QWEN35_FABRIC_BOUNDARY=1 ${EXTRA:-} && {
  docker logs orn-fb 2>&1 | grep -E "rows top-k|fabric boundary" | sort -u | head -5
  URL=http://localhost:20000 python3 $T/realtext.py 384 | tee $HOME/ornith-fusion/runs/$L/realtext.txt | tail -n 1
  docker rm -f orn-fb >/dev/null; }
echo "== greedy ${BASE:-step6b} vs $L"; python3 $T/greedy.py compare ${BASE:-step6b} $L 2>&1 | tail -n 3
python3 -c "
import json
for L in ('${BASE:-step6b}','$L'):
    try: print(L, json.load(open(f'/home/raahem/ornith-fusion/runs/{L}/garble.json'))['summary'])
    except Exception as e: print(L, e)"
echo "fb milestone done $(date +%T)"
