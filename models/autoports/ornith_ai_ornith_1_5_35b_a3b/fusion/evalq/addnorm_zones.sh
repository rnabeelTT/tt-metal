#!/bin/bash
export PATH=$HOME/ornith-fusion/bin:$PATH
while pgrep -f "q122attn_f1.sh" >/dev/null || chips_busy; do sleep 30; done
T=$HOME/tt-metal-gdnwip/models/autoports/ornith_ai_ornith_1_5_35b_a3b/fusion/tools
L=${1:-addnormzones}; rm -rf $HOME/ornith-fusion/runs/prof/$L
cd $T && MESH=P300 DEVS=0,1 PROF_ENV="--env BS=1 ${ZENV:-}" ./prof_cmd.sh $L models/autoports/ornith_ai_ornith_1_5_35b_a3b/tests/test_addnorm_ops.py prof
docker rm -f ornith-prof-$L >/dev/null 2>&1
grep -E "trial|Error" $HOME/ornith-fusion/runs/prof/$L.log | tail -4
f=$(find $HOME/ornith-fusion/runs/prof/$L -name profile_log_device.csv | head -1); python3 $T/zones.py "$f" | tail -60
