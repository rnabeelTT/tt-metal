#!/bin/bash
# pc_tests.sh <container> <label>: prefix-cache tests on a running server at :20000 -> runs/<label>/
set -o pipefail
C=$1; L=$2
T=$HOME/tt-metal-bringups/models/autoports/ornith_ai_ornith_1_5_35b_a3b/fusion/tools
R=$HOME/ornith-fusion/runs/$L; mkdir -p $R
cd $T
TAG=$L python3 prefix_bench.py repeat 2048 8192 32768 | tee $R/repeat.log
TAG=$L python3 prefix_bench.py chat 8192 6 | tee $R/chat.log
TAG=$L THINK=1 python3 prefix_bench.py chat 4096 4 | tee $R/chat_think.log
# no-hit traffic: random prompts, distinct seeds per row, no warm-up
for r in 128:128:1:8:11 2048:128:1:8:12 8192:64:1:4:13 128:128:32:128:14; do
  IFS=: read I O U N S <<< "$r"
  WARM=0 SEED=$S RUNS=$HOME/ornith-fusion/runs ./bench.sh $C $L $I:$O:$U:$N 2>&1 | tee -a $R/bench.log
done
docker logs $C 2>&1 | grep -E "prefix state:|prefix cache:|Recurrent-state" | tail -n 40 > $R/prefix_log.txt
