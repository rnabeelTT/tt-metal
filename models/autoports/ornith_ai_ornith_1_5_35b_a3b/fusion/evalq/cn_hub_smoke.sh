#!/bin/bash
# Hub smoke test of raahemnabeel/qwen3-coder-next-blackhole-fast (after the Ornith p150 chain): pull + serve p300x2 as a
# consumer would, greedy, compare with the card run of the locally built bundle (same image and code: expect identical).
export PATH=$HOME/ornith-fusion/bin:$PATH MODEL_ID=Qwen/Qwen3-Coder-Next
T=$HOME/tt-metal-codernext/models/autoports/ornith_ai_ornith_1_5_35b_a3b/fusion/tools
R=$HOME/ornith-fusion/runs
while pgrep -fx "bash $R/ornith_p150_chain.sh" >/dev/null || pgrep -fx "bash $R/glm_chain.sh" >/dev/null || pgrep -fx "bash $R/q122_chain.sh" >/dev/null || chips_busy; do sleep 60; done
echo "cn hub smoke start $(date +%T)"
GREEDY=1 $T/card_bench.sh cn_hub_smoke raahemnabeel/qwen3-coder-next-blackhole-fast p300x2 128:128:1:4 128:128:32:128
python3 $T/greedy.py compare card_cnnew_p300x2 cn_hub_smoke 2>&1 | tail -n 3
echo "cn hub smoke done $(date +%T)"
# appended 2026-10-24: Coder-30B blackhole-fast hub smoke (run_queue.sh is already running; it reads its list once)
bash $R/c30_hub_smoke.sh > $R/c30_hub_smoke.log 2>&1
