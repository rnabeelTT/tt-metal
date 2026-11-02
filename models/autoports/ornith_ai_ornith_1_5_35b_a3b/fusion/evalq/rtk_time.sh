#!/bin/bash
# Rows top-k op time by row count on 4 dies (62080 columns): the 16-bit kernels (ornith_fusion c1f25770879~1, tree
# ~/ornith-fusion/rtk_old) vs the 32-bit ones (current); served p300x2 8 users was 9.44 -> 9.71 ms.
export PATH=$HOME/ornith-fusion/bin:$HOME/.local/bin:$PATH
T=$HOME/tt-metal-bringups/models/autoports/ornith_ai_ornith_1_5_35b_a3b/fusion/tools
IMG=tt-model/ornith-1.5-35b-a3b-blackhole-fast:801c00a07a4c
while systemctl --user is-active -q tk32-ports || chips_busy; do sleep 60; done
echo "rtk time start $(date +%T)"
for side in old new; do
  M=$HOME/tt-metal-bringups; [ $side = old ] && M=$HOME/ornith-fusion/rtk_old
  echo "== $side"
  for rep in 1 2; do
    METAL=$M IMAGE=$IMG MESH=P300x2 TNAME=rtkt TENV="--env RTK_TB=1,2,4,8" timeout 1800 $T/trun.sh \
      "python models/autoports/ornith_ai_ornith_1_5_35b_a3b/tests/test_rows_topk.py 2>&1 | grep -E 'ok on|traced|Error|assert|Traceback' | tail -9"
    docker rm -f rtkt >/dev/null 2>&1
  done
done
echo "rtk time done $(date +%T)"
