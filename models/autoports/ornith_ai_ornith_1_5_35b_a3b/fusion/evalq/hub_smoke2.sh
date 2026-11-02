#!/bin/bash
# Hub smoke of the 2026-10-26 revisions (pull + serve p300x2 as a consumer would, greedy vs the card run of the same
# build: expect identical): Ornith, Coder-Next.
export PATH=$HOME/ornith-fusion/bin:$HOME/.local/bin:$PATH
R=$HOME/ornith-fusion/runs
while systemctl --user is-active -q q122-fast || chips_busy; do sleep 60; done
echo "hub smoke2 start $(date +%T)"
T=$HOME/tt-metal-bringups/models/autoports/ornith_ai_ornith_1_5_35b_a3b/fusion/tools
rm -rf $R/orn_hub_smoke2; MODEL_ID=ornith-ai/Ornith-1.5-35B-A3B GREEDY=1 $T/card_bench.sh orn_hub_smoke2 raahemnabeel/ornith-1.5-35b-a3b-blackhole-fast p300x2 128:128:1:4 128:128:32:128
grep -h "image\|pull" $R/orn_hub_smoke2/serve.log | head -3
MODEL_ID=ornith-ai/Ornith-1.5-35B-A3B python3 $T/greedy.py compare card_ov3_p300x2 orn_hub_smoke2 2>&1 | tail -n 3
T=$HOME/tt-metal-codernext/models/autoports/ornith_ai_ornith_1_5_35b_a3b/fusion/tools
rm -rf $R/cn_hub_smoke2; MODEL_ID=Qwen/Qwen3-Coder-Next GREEDY=1 $T/card_bench.sh cn_hub_smoke2 raahemnabeel/qwen3-coder-next-blackhole-fast p300x2 128:128:1:4 128:128:32:128
grep -h "image\|pull" $R/cn_hub_smoke2/serve.log | head -3
MODEL_ID=Qwen/Qwen3-Coder-Next python3 $T/greedy.py compare card_cnfast_p300x2 cn_hub_smoke2 2>&1 | tail -n 3
echo "hub smoke2 done $(date +%T)"
