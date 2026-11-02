#!/bin/bash
cd /home/raahem/ornith-fusion/runs
PORT=20000 RUNS=/home/raahem/ornith-fusion/runs /home/raahem/tt-metal-bringups/models/autoports/ornith_ai_ornith_1_5_35b_a3b/fusion/tools/eval.sh ov3_p300 livecodebench_v6
GPQA_LIMIT=198 PORT=20000 RUNS=/home/raahem/ornith-fusion/runs /home/raahem/tt-metal-bringups/models/autoports/ornith_ai_ornith_1_5_35b_a3b/fusion/tools/eval.sh ov3_p300_full r1_gpqa_diamond
echo 'evals done'
