#!/bin/bash
# Rows top-k with UINT32 indices for any width (gdn_wip 32687bad143): op test at the 2-die width (124160) and the 4-die
# width (62080, the merge changed for it too), then Ornith served on p300 (rows top-k now installed on 2 dies) and a
# p300x2 check; vs card_fast2_p300 / card_ov3_p300x2.
export PATH=$HOME/ornith-fusion/bin:$HOME/.local/bin:$PATH MODEL_ID=ornith-ai/Ornith-1.5-35B-A3B
R=$HOME/ornith-fusion/runs
T=$HOME/tt-metal-gdnwip/models/autoports/ornith_ai_ornith_1_5_35b_a3b/fusion/tools
IMG=tt-model/ornith-1.5-35b-a3b-blackhole-fast:801c00a07a4c
while systemctl --user is-active -q hub-smoke2 || systemctl --user is-active -q q122-fast || chips_busy; do sleep 60; done
echo "topk32 start $(date +%T)"
IMAGE=$IMG MESH=P300 DEVS=0,1 TNAME=rtk2 TENV="--env RTK_ND=2 --env RTK_W=124160" timeout 1800 $T/trun.sh \
  "python models/autoports/ornith_ai_ornith_1_5_35b_a3b/tests/test_rows_topk.py 2>&1 | grep -E 'ok on|traced|Error|assert|Traceback' | tail -8"
docker rm -f rtk2 >/dev/null 2>&1
IMAGE=$IMG MESH=P300x2 TNAME=rtk4 timeout 1800 $T/trun.sh \
  "python models/autoports/ornith_ai_ornith_1_5_35b_a3b/tests/test_rows_topk.py 2>&1 | grep -E 'ok on|traced|Error|assert|Traceback' | tail -8"
docker rm -f rtk4 >/dev/null 2>&1
E="--env QWEN35_MOE_FUSED_DECODE=1 --env QWEN35_DECODE_REPLICATED=1 --env QWEN35_GDN_DECODE_OPS=1 --env QWEN35_ATTN_DECODE_OPS=1 --env QWEN35_MTP=0 --env QWEN35_ROWS_TOPK=1"
FB="--env QWEN35_FABRIC_BOUNDARY=1 --env QWEN35_FB_MAX_ROWS=4 --env QWEN35_FB_CONSUMER=1 --env QWEN35_FB_ROUTER=1 --env QWEN35_MOE_SPLIT_PAIR=1"
L=tk32_p300; rm -rf $R/$L
MODELS=1 GARBLE=1 KEEP=1 ROWS="128:128:1:8 128:128:2:8 128:128:4:16 128:128:8:32 128:128:16:64 128:128:32:128 2048:128:1:4" $T/milestone.sh $L orn-tk $IMG p300 $E && {
  docker logs orn-tk 2>&1 | grep -E "rows top-k" | sort -u | head -3
  URL=http://localhost:20000 python3 $T/realtext.py 384 | tee $R/$L/realtext.txt | tail -n 1
  docker rm -f orn-tk >/dev/null; }
python3 -c "
import json; d=json.load(open('$R/$L/garble.json')); print('garble', d['summary']); print([(r['n_words'], r['text'][:40]) for r in d['rows']][:3])"
python3 $T/greedy.py compare card_fast2_p300 $L 2>&1 | tail -n 2
L=tk32_p300x2; rm -rf $R/$L
MODELS=1 GARBLE=1 ROWS="128:128:1:8 128:128:8:32" $T/milestone.sh $L orn-tk4 $IMG p300x2 $E $FB
python3 -c "
import json; d=json.load(open('$R/$L/garble.json')); print('garble', d['summary'])"
python3 $T/greedy.py compare card_ov3_p300x2 $L 2>&1 | tail -n 2
echo "topk32 done $(date +%T)"
